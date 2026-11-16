#!/usr/bin/env python3
"""Build matched signal-version and vital-missingness sensitivity inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from build_tokens import SIGNALS, VITALS, seconds


def read_export(path, channels):
    frame = pd.read_csv(path, dtype={"time": str})
    if not {"time", *channels} <= set(frame):
        raise ValueError("Export is missing required channels")
    frame = frame[["time"] + channels].copy()
    for channel in channels:
        frame[channel] = pd.to_numeric(frame[channel], errors="coerce")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    required = [
        "record_id",
        "group_id",
        "signal_csv",
        "vital_csv",
        "earlier_signal_csv",
        "earlier_vital_csv",
    ]
    manifest = pd.read_csv(args.manifest, dtype=str)
    if not set(required) <= set(manifest) or manifest[required].isna().any().any():
        raise ValueError("Manifest requires " + ",".join(required))
    if manifest.record_id.duplicated().any():
        raise ValueError("Duplicate recording in manifest")
    for column in ["record_id", "group_id"]:
        if not manifest[column].str.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*").all():
            raise ValueError("Use anonymous ASCII identifiers")
    manifests = {"earlier_signal": [], "missingness": []}
    for row in manifest.sort_values("record_id").itertuples():
        current_signal = read_export(args.manifest.parent / row.signal_csv, SIGNALS)
        current_vital = read_export(args.manifest.parent / row.vital_csv, VITALS)
        source_signal = read_export(
            args.manifest.parent / row.earlier_signal_csv, SIGNALS
        )
        source_vital = read_export(args.manifest.parent / row.earlier_vital_csv, VITALS)
        current_seconds = seconds(current_signal.time)
        old_seconds = seconds(source_signal.time)
        selected = source_signal.loc[
            old_seconds.between(current_seconds.min(), current_seconds.max())
        ].copy()
        if not np.array_equal(
            seconds(selected.time).unique(), current_seconds.unique()
        ):
            raise ValueError("Earlier signal does not cover the same seconds")
        selected["EMG"] = selected.EMG * 1000.0
        old_vital_seconds = seconds(source_vital.time)
        new_vital_seconds = seconds(current_vital.time)
        if new_vital_seconds.duplicated().any():
            raise ValueError("Current vitals require one row per second")
        masked = current_vital.copy()
        for channel in VITALS:
            missing_by_second = (
                source_vital[channel].isna().groupby(old_vital_seconds).any()
            )
            missing = (
                new_vital_seconds.map(missing_by_second).fillna(True).to_numpy(bool)
            )
            masked.loc[missing, channel] = np.nan
        for variant, signal_frame, vital_frame in [
            ("earlier_signal", selected, current_vital),
            ("missingness", current_signal, masked),
        ]:
            folder = args.out / variant / "inputs" / row.record_id
            folder.mkdir(parents=True, exist_ok=True)
            signal_frame.to_csv(folder / "signal.csv", index=False)
            vital_frame.to_csv(folder / "vital.csv", index=False)
            manifests[variant].append(
                dict(
                    record_id=row.record_id,
                    group_id=row.group_id,
                    signal_csv=f"{row.record_id}/signal.csv",
                    vital_csv=f"{row.record_id}/vital.csv",
                )
            )
    for variant, rows in manifests.items():
        folder = args.out / variant
        manifest_path = folder / "inputs/recordings.csv"
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("build_tokens.py")),
                "--manifest",
                str(manifest_path),
                "--labels",
                str(args.labels),
                "--out",
                str(folder / "prepared"),
            ],
            check=True,
        )
    print("Matched sensitivity inputs written.")


if __name__ == "__main__":
    main()
