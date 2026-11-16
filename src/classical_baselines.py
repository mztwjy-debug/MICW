#!/usr/bin/env python3
"""Fixed SVM and time-only baselines with participant-group exclusion."""

from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
)
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

WINDOWS = (60, 120, 180, 300)
GROUP_MAP = {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_groups(df):
    df = df.copy()
    df["subject"] = df.record_id.astype(str)
    df["minute_index"] = df.minute_index.astype(int)
    df["group"] = df.subject.map(GROUP_MAP)
    if df.group.isna().any():
        raise ValueError("Missing explicit group for a record")
    if "group_id" in df:
        assert df.group.eq(df.group_id).all(), "Manifest and token groups disagree"
    assert not df.duplicated(["subject", "minute_index"]).any()
    return df


def metrics(pred, binary=False):
    y = pred.y_true.to_numpy(dtype=int)
    p = pred.y_pred.to_numpy(dtype=int)
    out = {
        "rows": len(y),
        "records": int(pred.subject.nunique()),
        "groups": int(pred.group.nunique()),
        "accuracy": float(accuracy_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, p, average="weighted", zero_division=0)),
    }
    if binary:
        out.update(
            {
                "positive_rows": int((y == 1).sum()),
                "negative_rows": int((y == 0).sum()),
                "precision": float(precision_score(y, p, zero_division=0)),
                "recall": float(recall_score(y, p, zero_division=0)),
                "f1": float(f1_score(y, p, zero_division=0)),
                "roc_auc": float(roc_auc_score(y, pred.prob_1))
                if len(set(y)) == 2
                else None,
                "pr_auc": float(average_precision_score(y, pred.prob_1))
                if (y == 1).any()
                else None,
                "confusion_matrix": confusion_matrix(y, p, labels=[0, 1]).tolist(),
            }
        )
    else:
        out.update(
            {
                "low_rows": int((y == 0).sum()),
                "medium_rows": int((y == 1).sum()),
                "high_rows": int((y == 2).sum()),
                "high_recall": float(recall_score(y == 2, p == 2, zero_division=0)),
                "severe_error": float(np.mean(np.abs(y - p) == 2)),
                "adjacent_accuracy": float(np.mean(np.abs(y - p) <= 1)),
                "confusion_matrix": confusion_matrix(y, p, labels=[0, 1, 2]).tolist(),
            }
        )
    return out


def fit_grouped(df, columns, model, target, context):
    pieces, audits = [], []
    metadata = [
        c
        for c in [
            "subject",
            "record_id",
            "group_id",
            "group",
            "minute_index",
            "misc_score",
            "is_baseline",
            "window_seconds",
            "window_start_seconds",
            "window_end_seconds",
        ]
        if c in df
    ]
    for group in sorted(df.group.unique()):
        train = df[df.group != group]
        test = df[df.group == group]
        train_groups, test_groups = set(train.group), set(test.group)
        train_records, test_records = set(train.subject), set(test.subject)
        assert not train_groups & test_groups
        assert not train_records & test_records
        assert len(set(train[target])) >= 2, (
            f"Single-class train fold: {context}, {group}"
        )
        estimator = clone(model)
        estimator.fit(train[columns], train[target])
        pred = test[metadata].copy()
        pred["y_true"] = test[target].to_numpy(dtype=int)
        pred["y_pred"] = estimator.predict(test[columns]).astype(int)
        proba = estimator.predict_proba(test[columns])
        for label in range(2 if context.startswith("onset") else 3):
            loc = np.flatnonzero(estimator.classes_ == label)
            pred[f"prob_{label}"] = proba[:, loc[0]] if len(loc) else 0.0
        pred["held_out_group"] = group
        pieces.append(pred)
        audits.append(
            {
                "context": context,
                "test_group": group,
                "test_records": sorted(test_records),
                "train_groups": sorted(train_groups),
                "train_records": sorted(train_records),
                "train_rows": len(train),
                "test_rows": len(test),
                "group_overlap_count": 0,
                "record_overlap_count": 0,
            }
        )
    result = (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["subject", "minute_index"])
        .reset_index(drop=True)
    )
    assert len(result) == len(df)
    assert not result.duplicated(["subject", "minute_index"]).any()
    return result, audits


def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--groups-csv", type=Path, required=True)
    parser.add_argument(
        "--skip-onset",
        action="store_true",
        help="Reuse saved onset predictions after exact input SHA verification; rerun time-only models.",
    )
    args = parser.parse_args()
    global GROUP_MAP
    groups = pd.read_csv(args.groups_csv, dtype=str)
    assert not groups.record_id.duplicated().any()
    GROUP_MAP = dict(zip(groups.record_id, groups.group_id))
    base = args.base_dir.resolve()
    output = (args.output_dir or base / "models" / "classical").resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = {str(args.groups_csv): digest(args.groups_csv)}
    onset = {}
    for window in WINDOWS:
        path = base / "onset" / f"window_{window}s.csv"
        sources[str(path)] = digest(path)
        onset[window] = add_groups(pd.read_csv(path))
        onset[window]["task_label"] = onset[window].task_label.astype(int)
        assert onset[window].group.nunique() <= len(set(GROUP_MAP.values()))
        assert (
            (onset[window].window_end_seconds - onset[window].window_start_seconds)
            .eq(window)
            .all()
        )
        # Each record may contribute at most its first observed onset event.
        assert onset[window].groupby("subject").task_label.sum().le(1).all()
    keys = lambda df: set(zip(df.subject, df.minute_index))
    common = set.intersection(*(keys(onset[w]) for w in WINDOWS))
    assert common == keys(onset[300]), "Common set differs from longest-window set"
    post_keys = set.intersection(
        *(keys(onset[w][onset[w].minute_index > 5]) for w in WINDOWS)
    )
    assert post_keys == common
    audits, summary = [], []
    if args.skip_onset:
        prior = json.loads((output / "protocol.json").read_text())
        for path, checksum in sources.items():
            assert prior["input_sha256"][path] == checksum, (
                "Changed onset input cannot reuse prior fit"
            )
        summary = json.loads((output / "results.json").read_text())["onset"]
        audits = [
            r
            for r in json.loads((output / "fold_group_assertions.json").read_text())
            if r["context"].startswith("onset")
        ]
    # Match fixed SVM settings, including libsvm probability calibration
    # within each outer training set; no held-out group enters preprocessing/fit.
    svm = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42,
        ),
    )
    with threadpool_limits(limits=1):
        for window in () if args.skip_onset else WINDOWS:
            df = onset[window]
            columns = [c for c in df if c.startswith("feature__")]
            assert len(columns) == 168
            for cohort in ["postbaseline_common_300s"]:
                current = (
                    df[df.minute_index > 5].copy()
                    if cohort.startswith("postbaseline")
                    else df
                )
                if cohort.startswith("postbaseline"):
                    assert keys(current) == common
                pred, folds = fit_grouped(
                    current, columns, svm, "task_label", f"onset_{window}_{cohort}"
                )
                audits.extend(folds)
                pred["cohort"] = cohort
                pred["model_name"] = "svm_rbf"
                save_csv(
                    pred,
                    output / "onset" / f"window_{window}s_{cohort}_predictions.csv",
                )
                row = {
                    "cohort": cohort,
                    "window_seconds": window,
                    "model_name": "svm_rbf",
                    **metrics(pred, True),
                }
                summary.append(row)
                print(
                    json.dumps(
                        {k: v for k, v in row.items() if k != "confusion_matrix"},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        token_path = base / "state" / "tokens.csv"
        sources[str(token_path)] = digest(token_path)
        tokens = add_groups(pd.read_csv(token_path))
        tokens["target"] = tokens.severity_label.map(
            {"low": 0, "medium": 1, "high": 2}
        ).astype(int)
        expected = tokens.misc_score.map(
            lambda x: 0 if x <= 1 else (1 if x <= 4 else 2)
        ).astype(int)
        assert tokens.target.eq(expected).all(), (
            "Severity labels disagree with the MISC cutpoints"
        )
        assert set(tokens.subject) == set(GROUP_MAP)
        # Frozen single-feature classifier; capacity is not tuned on the labels.
        rf = RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=1
        )
        time_summary = []
        for train_scope in ["postbaseline", "all_tokens"]:
            current = (
                tokens[tokens.minute_index > 5].copy()
                if train_scope == "postbaseline"
                else tokens
            )
            pred, folds = fit_grouped(
                current, ["minute_index"], rf, "target", f"time_only_{train_scope}"
            )
            audits.extend(folds)
            pred["model_name"] = "time_only_random_forest"
            pred["train_scope"] = train_scope
            save_csv(pred, output / "time_only" / f"{train_scope}_predictions.csv")
            masks = {"postbaseline": pred.minute_index > 5}
            if train_scope == "all_tokens":
                masks["all_tokens"] = np.ones(len(pred), dtype=bool)
            for eval_scope, mask in masks.items():
                row = {
                    "model_name": "time_only_random_forest",
                    "train_scope": train_scope,
                    "evaluation_scope": eval_scope,
                    **metrics(pred.loc[mask]),
                }
                time_summary.append(row)
                print(
                    json.dumps(
                        {k: v for k, v in row.items() if k != "confusion_matrix"}
                    ),
                    flush=True,
                )
    pd.DataFrame(summary).drop(columns="confusion_matrix").to_csv(
        output / "onset_summary.csv", index=False
    )
    pd.DataFrame(time_summary).drop(columns="confusion_matrix").to_csv(
        output / "time_only_summary.csv", index=False
    )
    protocol = {
        "grouping": "Explicit record_id-to-group_id manifest; no identifier parsing.",
        "onset_event": "First labeled minute with MISC>=2; include earlier minutes and event minute, exclude later minutes.",
        "onset_input_interval": "[s_t-W,s_t), where s_t=(minute_index-1)*60 relative to original recording start.",
        "onset_primary": "Train and test on minute_index>5, after the entire 300-second reference interval is available. The same (record,minute) risk set is used for W=60,120,180,300.",
        "onset_not_forecast": "Window length is amount of preceding physiological input, not a validated warning lead time. Offline upstream filtering also uses future samples.",
        "onset_features": 168,
        "onset_model": {
            "class": "SVC",
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "class_weight": "balanced",
            "probability": True,
            "random_state": 42,
            "preprocessing": "Median imputation and StandardScaler fitted solely on outer training groups.",
        },
        "onset_probabilities": "libsvm internal probability calibration uses training rows only and is not group-aware; held-out groups are never used.",
        "time_only_model": {
            "class": "RandomForestClassifier",
            "feature": ["minute_index"],
            "n_estimators": 300,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": 1,
            "other_parameters": "scikit-learn defaults, no tuning",
        },
        "time_only_evaluation": "All-token fitting with postbaseline evaluation is primary, matching neural fitting scope; independent postbaseline-only fitting and all-token evaluation are auxiliary.",
        "common_risk_set_rows": len(common),
        "input_sha256": sources,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n"
    )
    (output / "fold_group_assertions.json").write_text(
        json.dumps(audits, ensure_ascii=False, indent=2) + "\n"
    )
    (output / "results.json").write_text(
        json.dumps(
            {"onset": summary, "time_only": time_summary}, ensure_ascii=False, indent=2
        )
        + "\n"
    )
    lines = [
        "# Classical baselines with explicit anonymous grouping",
        "",
        "Primary onset analysis uses postbaseline rows only. All four windows use the same record/minute risk set. Window duration is preceding input length, not an established lead time.",
        "",
        "| W (s) | Rows | Positive | Groups | Precision | Recall | F1 | ROC AUC | PR AUC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        if r["cohort"].startswith("postbaseline"):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(r[k])
                        if k in ["window_seconds", "rows", "positive_rows", "groups"]
                        else f"{r[k]:.4f}"
                        for k in [
                            "window_seconds",
                            "rows",
                            "positive_rows",
                            "groups",
                            "precision",
                            "recall",
                            "f1",
                            "roc_auc",
                            "pr_auc",
                        ]
                    ]
                )
                + " |"
            )
    lines += [
        "",
        "Time-only uses minute_index and a fixed 300-tree class-balanced random forest. No physiological feature or record identifier enters the predictor.",
        "",
        "| Training scope | Evaluation scope | Rows | Accuracy | Balanced accuracy | Macro-F1 | High recall | Severe error |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in time_summary:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r[k])
                    if k in ["train_scope", "evaluation_scope", "rows"]
                    else f"{r[k]:.4f}"
                    for k in [
                        "train_scope",
                        "evaluation_scope",
                        "rows",
                        "accuracy",
                        "balanced_accuracy",
                        "macro_f1",
                        "high_recall",
                        "severe_error",
                    ]
                ]
            )
            + " |"
        )
    lines += [
        "",
        "Every outer fold excludes an entire explicit participant group. See fold_group_assertions.json and row-level predictions.",
        "",
        "Do not assume the reference interval is symptom-free. Group assignments must be supported by participant metadata.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("Completed:", output, flush=True)


if __name__ == "__main__":
    main()
