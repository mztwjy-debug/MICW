#!/usr/bin/env python3
"""Extract 60-second state tokens and preceding onset windows.

Inputs must already be aligned/processed; this does not reconstruct unverified
upstream filtering or overwrite source signals.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from feature_extraction import (
    compute_signal_features,
    compute_vital_features,
    prefix_features,
)

SIGNALS = ["EDA", "EGG", "EMG"]
VITALS = ["SBP", "DBP", "HR", "PR", "SpO2", "PTT"]


def seconds(series):
    def parse(text):
        parts = str(text).split(":")
        if len(parts) != 3:
            raise ValueError("Use H:MM:SS timestamps without crossing midnight")
        h, m, s = map(int, parts)
        if not (0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60):
            raise ValueError("Invalid time")
        return h * 3600 + m * 60 + s

    return series.map(parse)


def features(signal, vital, start, end):
    result = {}
    s = signal[(signal.seconds >= start) & (signal.seconds < end)]
    v = vital[(vital.seconds >= start) & (vital.seconds < end)]
    for channel in SIGNALS:
        result.update(
            prefix_features(
                "signal_" + channel,
                compute_signal_features(
                    s[channel].dropna().to_numpy(float), end - start
                ),
            )
        )
    for channel in VITALS:
        result.update(
            prefix_features(
                "vital_" + channel,
                compute_vital_features(v[channel].to_numpy(float), end - start),
            )
        )
    return result, len(s), len(v)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="record_id,group_id,signal_csv,vital_csv; paths relative to manifest",
    )
    parser.add_argument(
        "--labels", type=Path, required=True, help="record_id,minute_index,misc_score"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest, dtype=str)
    labels = pd.read_csv(args.labels)
    required_manifest = ["record_id", "group_id", "signal_csv", "vital_csv"]
    required_labels = ["record_id", "minute_index", "misc_score"]
    if not set(required_manifest) <= set(manifest) or not set(required_labels) <= set(
        labels
    ):
        raise ValueError("Missing required manifest or label columns")
    if (
        manifest[required_manifest].isna().any().any()
        or labels[required_labels].isna().any().any()
    ):
        raise ValueError("Manifest and labels cannot contain missing metadata")
    for column in ["record_id", "group_id"]:
        if not manifest[column].str.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*").all():
            raise ValueError("Use anonymous ASCII identifiers")
    for column in ["minute_index", "misc_score"]:
        values = pd.to_numeric(labels[column], errors="raise").to_numpy(float)
        if (
            not np.isfinite(values).all()
            or not np.equal(values, np.floor(values)).all()
        ):
            raise ValueError("Minutes and MISC scores must be finite integers")
        labels[column] = values.astype(int)
    if (
        not labels.minute_index.ge(1).all()
        or not labels.misc_score.between(0, 10).all()
    ):
        raise ValueError("Invalid minute index or MISC score")

    if (
        manifest.record_id.duplicated().any()
        or labels.duplicated(["record_id", "minute_index"]).any()
    ):
        raise ValueError("Duplicate record mapping or minute label")
    if set(labels.record_id) != set(manifest.record_id):
        raise ValueError("Manifest and labels record sets differ")
    state = []
    onset = {w: [] for w in [60, 120, 180, 300]}
    for row in manifest.sort_values("record_id").itertuples():
        signal = pd.read_csv(args.manifest.parent / row.signal_csv)
        vital = pd.read_csv(args.manifest.parent / row.vital_csv)
        signal["seconds"] = seconds(signal.time)
        vital["seconds"] = seconds(vital.time)
        if (
            not signal.seconds.is_monotonic_increasing
            or not vital.seconds.is_monotonic_increasing
        ):
            raise ValueError(
                "Time must be ordered; repeated signal seconds must retain sample order"
            )
        if vital.seconds.duplicated().any():
            raise ValueError("Vital export must contain one row per second")
        for channel in SIGNALS:
            signal[channel] = pd.to_numeric(signal[channel], errors="coerce")
        for channel in VITALS:
            vital[channel] = pd.to_numeric(vital[channel], errors="coerce")
        start = int(signal.seconds.min())
        end = int(min(signal.seconds.max(), vital.seconds.max())) + 1
        if start != int(vital.seconds.min()) or end - start < 300:
            raise ValueError(
                "Inputs must share start and contain complete 300-second reference"
            )
        baseline, _, _ = features(signal, vital, start, start + 300)
        lab = labels[labels.record_id.eq(row.record_id)].sort_values("minute_index")
        positive = lab[lab.misc_score >= 2]
        first = int(positive.minute_index.min()) if len(positive) else None
        for label in lab.itertuples():
            minute = int(label.minute_index)
            score = float(label.misc_score)
            level = "low" if score <= 1 else "medium" if score <= 4 else "high"
            # Require complete coverage of the labeled minute.
            if start + minute * 60 > end:
                continue
            for task, window in [("state", 60)] + [("onset", w) for w in onset]:
                window_end = start + (minute if task == "state" else minute - 1) * 60
                window_start = window_end - window
                if window_start < start:
                    continue
                if task == "onset" and first is not None and minute > first:
                    continue
                current, ns, nv = features(signal, vital, window_start, window_end)
                item = {
                    "record_id": row.record_id,
                    "group_id": row.group_id,
                    "minute_index": minute,
                    "misc_score": score,
                    "severity_label": level,
                    "is_baseline": minute <= 5,
                    "window_seconds": window,
                    "window_start_seconds": window_start,
                    "window_end_seconds": window_end,
                    "signal_row_count": ns,
                    "vital_row_count": nv,
                    "task_label": int(minute == first) if task == "onset" else level,
                }
                # Preserve each channel's absolute/delta column order.
                for modality, channels in [("signal", SIGNALS), ("vital", VITALS)]:
                    for channel in channels:
                        prefix = "feature__" + modality + "_" + channel + "_"
                        for key, value in current.items():
                            if key.startswith(prefix):
                                item[key] = value
                        for key, value in current.items():
                            if key.startswith(prefix):
                                item[
                                    "feature__delta_" + key.removeprefix("feature__")
                                ] = value - baseline[key]
                (state if task == "state" else onset[window]).append(item)
    target = args.out / "state/tokens.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(state).to_csv(target, index=False)
    for window, rows in onset.items():
        target = args.out / f"onset/window_{window}s.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(target, index=False)
    print(
        "Extracted state rows:",
        len(state),
        "onset rows:",
        {w: len(rows) for w, rows in onset.items()},
    )


if __name__ == "__main__":
    main()
