#!/usr/bin/env python3
"""Summarize anonymous OOF predictions with paired, whole-group resampling."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "low_f1",
    "medium_f1",
    "high_f1",
    "high_recall",
    "severe_error_rate",
]
KEYS = ["record_id", "minute_index"]


def scores(cm):
    """The same pooled three-class metric definitions used in the study."""
    cm = np.asarray(cm, dtype=float)
    diag = np.diagonal(cm, axis1=-2, axis2=-1)
    support = cm.sum(axis=-1)
    predicted = cm.sum(axis=-2)
    n = cm.sum(axis=(-1, -2))
    recall = np.divide(diag, support, out=np.zeros_like(diag), where=support > 0)
    f1 = np.divide(
        2 * diag,
        support + predicted,
        out=np.zeros_like(diag),
        where=(support + predicted) > 0,
    )
    return dict(
        accuracy=diag.sum(-1) / n,
        balanced_accuracy=recall.sum(-1) / np.maximum((support > 0).sum(-1), 1),
        macro_f1=f1.mean(-1),
        weighted_f1=(f1 * support).sum(-1) / n,
        low_f1=f1[..., 0],
        medium_f1=f1[..., 1],
        high_f1=f1[..., 2],
        high_recall=recall[..., 2],
        severe_error_rate=(cm[..., 0, 2] + cm[..., 2, 0]) / n,
    )


def anonymous_id(value):
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", str(value)))


def read_groups(path):
    df = pd.read_csv(path, dtype=str)
    if not {"record_id", "group_id"} <= set(df):
        raise ValueError("groups-csv requires record_id and group_id")
    df = df[["record_id", "group_id"]]
    if df.isna().any().any() or df.record_id.duplicated().any():
        raise ValueError("Each record must have exactly one nonempty group")
    if not all(df[c].map(anonymous_id).all() for c in df):
        raise ValueError("Use anonymous ASCII record/group identifiers")
    if df.group_id.nunique() < 2:
        raise ValueError("At least two explicit groups are required")
    return dict(zip(df.record_id, df.group_id))


def integer_column(series, name):
    values = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or not (values == np.floor(values)).all():
        raise ValueError(f"{name} must contain finite integers")
    return values.astype(np.int64)


def read_predictions(args):
    """Accept a tidy model/seed CSV or the grouped runner's output directories."""
    if args.predictions_csv:
        return pd.read_csv(args.predictions_csv), [args.predictions_csv]
    parts, paths = [], []
    for path in sorted(args.runs_dir.glob("*_seed*/predictions.csv")):
        match = re.fullmatch(r"(.+)_seed(-?\d+)", path.parent.name)
        if match is None:
            continue
        df = pd.read_csv(path)
        model, seed = match.group(1), int(match.group(2))
        for column, value in [("model", model), ("seed", seed)]:
            if column in df and not df[column].eq(value).all():
                raise ValueError(f"Conflicting {column} in {path.name}")
            df[column] = value
        parts.append(df)
        paths.append(path)
    if not parts:
        raise ValueError("No <model>_seed<integer>/predictions.csv files found")
    return pd.concat(parts, ignore_index=True), paths


def validate_predictions(raw, mapping, reference_model, ci_seed):
    raw = raw.copy()
    if "record_id" not in raw and "subject" in raw:
        raw = raw.rename(columns={"subject": "record_id"})
    required = [
        "model",
        "seed",
        "record_id",
        "minute_index",
        "true_label",
        "pred_label",
    ]
    if not set(required) <= set(raw):
        raise ValueError("Predictions require " + ", ".join(required))
    if raw[required].isna().any().any():
        raise ValueError("Missing prediction metadata/labels")
    for column in ["seed", "minute_index", "true_label", "pred_label"]:
        raw[column] = integer_column(raw[column], column)
    if (
        not raw.record_id.map(anonymous_id).all()
        or not raw.model.map(anonymous_id).all()
    ):
        raise ValueError("Use anonymous ASCII recording IDs and simple model names")
    if (
        not raw.minute_index.ge(1).all()
        or not raw[["true_label", "pred_label"]].isin([0, 1, 2]).all().all()
    ):
        raise ValueError("Minutes must be positive and labels must be 0/1/2")
    if raw.duplicated(["model", "seed"] + KEYS).any():
        raise ValueError("Duplicate model/seed/record/minute key")
    if set(raw.record_id) != set(mapping):
        raise ValueError(
            "Predictions and grouping manifest must cover identical recording IDs"
        )
    mapped = raw.record_id.map(mapping)
    for column in ["fold_group", "group_id"]:
        if column in raw and not raw[column].eq(mapped).all():
            raise ValueError(
                "Reported held-out group disagrees with explicit group manifest"
            )
    if reference_model not in set(raw.model):
        raise ValueError("Reference model is absent")
    seed_sets = raw.groupby("model").seed.apply(lambda x: tuple(sorted(x.unique())))
    if len(set(seed_sets)) != 1 or any(ci_seed not in seeds for seeds in seed_sets):
        raise ValueError("All models require the same seed set, including the CI seed")
    raw = raw[required].copy()  # Do not propagate optional paths or other metadata.
    raw["group_id"] = raw.record_id.map(mapping)
    raw["is_baseline"] = raw.minute_index.le(5)
    canonical = None
    for _, df in raw.groupby(["model", "seed"], sort=True):
        aligned = df.sort_values(KEYS)[KEYS + ["true_label"]].reset_index(drop=True)
        if canonical is None:
            canonical = aligned
        elif not aligned.equals(canonical):
            raise ValueError(
                "All model/seed predictions must have identical keys and true labels"
            )
    return raw


def subset_rows(df, subset):
    if subset == "all":
        return df
    return df[df.is_baseline if subset == "baseline" else ~df.is_baseline]


def group_matrices(df, groups):
    return np.stack(
        [
            confusion_matrix(
                df.loc[df.group_id.eq(g), "true_label"],
                df.loc[df.group_id.eq(g), "pred_label"],
                labels=[0, 1, 2],
            )
            for g in groups
        ]
    )


def summarize(
    pred, mapping, reference_model, ci_seed=42, draws=10000, bootstrap_seed=20261114
):
    groups = sorted(set(mapping.values()))
    weights = np.random.default_rng(bootstrap_seed).multinomial(
        len(groups), np.full(len(groups), 1 / len(groups)), size=draws
    )
    detailed, intervals, paired, matrices = [], [], [], {}
    models = sorted(pred.model.unique())
    seeds = sorted(pred.seed.unique())
    for (model, seed), data in pred.groupby(["model", "seed"], sort=True):
        for subset in ["all", "post_baseline", "baseline"]:
            selected = subset_rows(data, subset)
            if selected.empty:
                continue
            cm = group_matrices(selected, groups)
            matrices[model, seed, subset] = cm
            point = scores(cm.sum(0))
            detailed.append(
                dict(
                    model=model,
                    seed=int(seed),
                    subset=subset,
                    rows=len(selected),
                    groups=selected.group_id.nunique(),
                    **{k: float(v) for k, v in point.items()},
                )
            )
            if seed == ci_seed:
                boot = scores(np.einsum("bg,gij->bij", weights, cm))
                for metric in METRICS:
                    if not np.isfinite(boot[metric]).all():
                        raise ValueError(
                            "An empty resampled subset prevents finite intervals"
                        )
                    low, high = np.quantile(boot[metric], [0.025, 0.975])
                    intervals.append(
                        dict(
                            model=model,
                            subset=subset,
                            metric=metric,
                            value=float(point[metric]),
                            lower=float(low),
                            upper=float(high),
                        )
                    )
    detail = pd.DataFrame(detailed)
    aggregate = detail.groupby(["model", "subset"])[METRICS].agg(["mean", "std"])
    aggregate.columns = ["_".join(x) for x in aggregate.columns]
    for comparator in models:
        if comparator == reference_model:
            continue
        for subset in ["all", "post_baseline"]:
            if (reference_model, seeds[0], subset) not in matrices:
                continue
            differences, boot_differences = [], []
            for seed in seeds:
                a = matrices[reference_model, seed, subset]
                b = matrices[comparator, seed, subset]
                differences.append(
                    scores(a.sum(0))["macro_f1"] - scores(b.sum(0))["macro_f1"]
                )
                boot_differences.append(
                    scores(np.einsum("bg,gij->bij", weights, a))["macro_f1"]
                    - scores(np.einsum("bg,gij->bij", weights, b))["macro_f1"]
                )
            low, high = np.quantile(np.mean(boot_differences, axis=0), [0.025, 0.975])
            paired.append(
                dict(
                    reference=reference_model,
                    comparator=comparator,
                    subset=subset,
                    mean_macro_f1_difference=float(np.mean(differences)),
                    lower=float(low),
                    upper=float(high),
                    bootstrap_unit="explicit group",
                    bootstrap_draws=draws,
                )
            )
    return (
        detail,
        aggregate.reset_index(),
        pd.DataFrame(intervals),
        pd.DataFrame(paired),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--predictions-csv", type=Path)
    inputs.add_argument("--runs-dir", type=Path)
    parser.add_argument("--groups-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--time-only-predictions", type=Path)
    parser.add_argument("--reference-model", default="multiscale_transformer")
    parser.add_argument("--ci-seed", type=int, default=42)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261114)
    args = parser.parse_args()
    if args.bootstrap_draws < 1:
        parser.error("bootstrap-draws must be positive")
    mapping = read_groups(args.groups_csv)
    raw, sources = read_predictions(args)
    pred = validate_predictions(raw, mapping, args.reference_model, args.ci_seed)
    detail, aggregate, ci, paired = summarize(
        pred,
        mapping,
        args.reference_model,
        args.ci_seed,
        args.bootstrap_draws,
        args.bootstrap_seed,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    outputs = {
        "main_metrics_by_seed": detail,
        "main_metrics_seed_summary": aggregate,
        "reference_seed_group_bootstrap": ci,
        "paired_architecture_differences": paired,
    }
    reference = pred[
        (pred.model == args.reference_model) & (pred.seed == args.ci_seed)
    ].copy()
    constant = reference.copy()
    constant["model"] = "constant_low"
    constant["pred_label"] = 0
    baseline_parts = [reference, constant]
    if args.time_only_predictions:
        time_only = pd.read_csv(args.time_only_predictions)
        if "record_id" not in time_only:
            time_only = time_only.rename(columns={"subject": "record_id"})
        time_only = time_only.rename(
            columns={"y_true": "true_label", "y_pred": "pred_label"}
        )
        time_only = time_only[
            ["record_id", "minute_index", "true_label", "pred_label"]
        ].copy()
        time_only["model"] = "time_only_random_forest"
        time_only["seed"] = args.ci_seed
        time_only["group_id"] = time_only.record_id.map(mapping)
        baseline_parts.append(time_only)
        sources.append(args.time_only_predictions)
    baselines = validate_predictions(
        pd.concat(baseline_parts, ignore_index=True),
        mapping,
        args.reference_model,
        args.ci_seed,
    )
    baseline_detail, _, baseline_ci, _ = summarize(
        baselines,
        mapping,
        args.reference_model,
        args.ci_seed,
        args.bootstrap_draws,
        args.bootstrap_seed,
    )
    outputs["constant_low_metrics"] = baseline_detail[
        baseline_detail.model == "constant_low"
    ]
    outputs["reference_baseline_metrics"] = baseline_detail
    outputs["reference_baseline_confidence_intervals"] = baseline_ci
    for name, df in outputs.items():
        df.to_csv(args.out / (name + ".csv"), index=False)
    # DataFrame JSON represents a one-seed undefined sample SD as null, not NaN.
    result = {
        name: json.loads(df.to_json(orient="records", double_precision=15))
        for name, df in outputs.items()
    }
    protocol = dict(
        records=len(mapping),
        groups=len(set(mapping.values())),
        rows_per_model_seed=int(len(pred) // pred.groupby(["model", "seed"]).ngroups),
        models=sorted(pred.model.unique()),
        seeds=sorted(map(int, pred.seed.unique())),
        reference_model=args.reference_model,
        ci_seed=args.ci_seed,
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_unit="explicit group; all member recordings/minutes retained",
        paired_method="same group draws across models/seeds; mean per-seed macro-F1 difference",
        standard_deviation="sample SD across training seeds (ddof=1)",
        primary_subset="minute_index > 5; fitting scope must be documented by the prediction producer",
        identity_policy="explicit manifest; no group inference from recording names",
        groups_sha256=hashlib.sha256(args.groups_csv.read_bytes()).hexdigest(),
        prediction_sha256=[
            hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
        ],
    )
    result["protocol"] = protocol
    (args.out / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    (args.out / "protocol.json").write_text(
        json.dumps(protocol, indent=2, allow_nan=False) + "\n"
    )
    print("Grouped summaries and paired intervals written.")


if __name__ == "__main__":
    main()
