#!/usr/bin/env python3
"""Summarize aligned seed-42 conditions with paired participant bootstrap."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from summarize_grouped import group_matrices, read_groups, scores, validate_predictions

METRICS = ["macro_f1", "balanced_accuracy", "high_recall", "severe_error_rate"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--conditions-dir", type=Path, required=True)
    parser.add_argument("--groups-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    mapping = read_groups(args.groups_csv)
    parts = []
    for name, path in [("reference", args.reference)] + [
        (p.parent.parent.name, p)
        for p in sorted(
            args.conditions_dir.glob("*/multiscale_transformer_seed42/predictions.csv")
        )
    ]:
        frame = pd.read_csv(path)
        frame["model"] = name
        frame["seed"] = 42
        parts.append(frame)
    if len(parts) < 2:
        parser.error("No condition predictions found")
    pred = validate_predictions(
        pd.concat(parts, ignore_index=True), mapping, "reference", 42
    )
    groups = sorted(set(mapping.values()))
    weights = np.random.default_rng(20261114).multinomial(
        len(groups), np.full(len(groups), 1 / len(groups)), size=10000
    )
    estimates, differences = [], []
    for subset in ["all", "post_baseline"]:
        selected = pred if subset == "all" else pred[pred.minute_index > 5]
        reference = group_matrices(selected[selected.model == "reference"], groups)
        reference_point = scores(reference.sum(0))
        reference_boot = scores(np.einsum("bg,gij->bij", weights, reference))
        for name, frame in selected.groupby("model", sort=True):
            if name == "reference":
                continue
            matrices = group_matrices(frame, groups)
            point = scores(matrices.sum(0))
            boot = scores(np.einsum("bg,gij->bij", weights, matrices))
            estimates.append(
                dict(
                    condition=name,
                    subset=subset,
                    rows=len(frame),
                    **{key: float(value) for key, value in point.items()},
                )
            )
            for metric in METRICS:
                low, high = np.quantile(
                    boot[metric] - reference_boot[metric], [0.025, 0.975]
                )
                differences.append(
                    dict(
                        condition=name,
                        subset=subset,
                        metric=metric,
                        difference=float(point[metric] - reference_point[metric]),
                        lower=float(low),
                        upper=float(high),
                    )
                )
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(estimates).to_csv(args.out / "sensitivity_metrics.csv", index=False)
    pd.DataFrame(differences).to_csv(
        args.out / "sensitivity_paired_differences.csv", index=False
    )
    print("Sensitivity summaries and paired intervals written.")


if __name__ == "__main__":
    main()
