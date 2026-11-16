#!/usr/bin/env python3
"""Generate artificial feature tokens and explicit participant groups."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SIGNAL_FEATURES = [
    "mean",
    "median",
    "std",
    "min",
    "max",
    "range",
    "rms",
    "slope",
    "diff_mean",
    "diff_std",
    "dominant_freq",
    "total_power",
    "spectral_centroid",
    "spectral_entropy",
]
VITAL_FEATURES = ["mean", "std", "min", "max", "range", "slope", "last_value"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    groups = pd.DataFrame(
        {
            "record_id": ["R901", "R902", "R903", "R904", "R905"],
            "group_id": ["G901", "G901", "G902", "G903", "G904"],
        }
    )
    groups.to_csv(args.out / "groups.csv", index=False)
    rng = np.random.default_rng(12345)
    columns = []
    for modality, channels, names in [
        ("signal", ["EDA", "EGG", "EMG"], SIGNAL_FEATURES),
        ("vital", ["SBP", "DBP", "HR", "PR", "SpO2", "PTT"], VITAL_FEATURES),
    ]:
        for channel in channels:
            columns.append([f"{modality}_{channel}_{name}" for name in names])
    rows = []
    for record in groups.record_id:
        reference = {
            name: float(value)
            for channel in columns
            for name, value in zip(channel, rng.normal(2, 0.3, len(channel)))
        }
        for minute in range(1, 13):
            row = dict(
                record_id=record,
                minute_index=minute,
                misc_score=[0, 2, 5][(minute - 1) % 3],
            )
            for channel in columns:
                values = {name: reference[name] + rng.normal() for name in channel}
                for name, value in values.items():
                    row["feature__" + name] = value
                for name, value in values.items():
                    row["feature__delta_" + name] = value - reference[name]
            rows.append(row)
    pd.DataFrame(rows).to_csv(args.out / "tokens.csv", index=False)
    print("Created artificial tokens for five recordings in four participant groups.")


if __name__ == "__main__":
    main()
