#!/usr/bin/env python3
"""Evaluate four fixed decision rules on complete recording histories."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from train_core import PROBA_COLUMNS, apply_monitoring_decision_layer
from summarize_grouped import scores

RULES = [
    ("No rejection", 1, 0.0, 0.0, 1),
    ("Margin >= 0.20", 1, 0.0, 0.20, 1),
    ("Confidence >= 0.60", 1, 0.60, 0.0, 1),
    ("3-min average; confidence >= 0.65; 3 High", 3, 0.65, 0.0, 3),
]


def evaluate(predictions):
    data = predictions.copy()
    if "subject" not in data and "record_id" in data:
        data = data.rename(columns={"record_id": "subject"})
    required = ["subject", "minute_index", "true_label"] + PROBA_COLUMNS
    if not set(required) <= set(data) or data[required].isna().any().any():
        raise ValueError(
            "Complete recording/minute keys, labels and probabilities are required"
        )
    data = data[required].copy()
    if not data.subject.astype(str).str.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*").all():
        raise ValueError("Use anonymous recording identifiers")
    for name in ["minute_index", "true_label"]:
        values = pd.to_numeric(data[name], errors="raise").to_numpy(float)
        if (
            not np.isfinite(values).all()
            or not np.equal(values, np.floor(values)).all()
        ):
            raise ValueError(f"{name} must contain finite integers")
        data[name] = values.astype(int)
    if (
        data.duplicated(["subject", "minute_index"]).any()
        or not data.true_label.isin([0, 1, 2]).all()
    ):
        raise ValueError("Invalid prediction keys or class labels")
    for _, one in data.groupby("subject"):
        minutes = sorted(one.minute_index.tolist())
        if minutes != list(range(1, max(minutes) + 1)):
            raise ValueError(
                "Decisions require complete histories starting at minute one"
            )
    probabilities = data[PROBA_COLUMNS].to_numpy(float)
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(1), 1, atol=1e-5)
    ):
        raise ValueError("Invalid class probabilities")
    rows = []
    for name, window, confidence, margin, persistence in RULES:
        decided = apply_monitoring_decision_layer(
            data,
            smoothing_window=window,
            confidence_threshold=confidence,
            margin_threshold=margin,
            high_persistence_minutes=persistence,
        )
        for subset in ["all", "post_baseline"]:
            selected = decided if subset == "all" else decided[decided.minute_index > 5]
            if selected.empty:
                continue
            actual = selected.true_label.to_numpy(int)
            predicted = selected.decision_label.to_numpy(int)
            covered = predicted >= 0
            accepted = int(covered.sum())
            cm = confusion_matrix(actual[covered], predicted[covered], labels=[0, 1, 2])
            metric = scores(cm) if accepted else None
            high = int((actual == 2).sum())
            true_positive = int(((actual == 2) & (predicted == 2)).sum())
            predicted_high = int((predicted == 2).sum())
            severe = int((covered & (np.abs(actual - predicted) == 2)).sum())
            rows.append(
                dict(
                    rule=name,
                    subset=subset,
                    total=len(selected),
                    accepted=accepted,
                    rejected=int((~covered).sum()),
                    coverage=accepted / len(selected),
                    covered_accuracy=float(metric["accuracy"]) if accepted else None,
                    covered_macro_f1=float(metric["macro_f1"]) if accepted else None,
                    high_count=high,
                    high_true_positive=true_positive,
                    high_precision=true_positive / max(predicted_high, 1),
                    high_recall=true_positive / max(high, 1),
                    severe_count=severe,
                    severe_per_all=severe / len(selected),
                    severe_per_accepted=severe / accepted if accepted else None,
                )
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(pd.read_csv(args.predictions_csv))
    args.out.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out / "fixed_decision_rules.csv", index=False)
    (args.out / "fixed_decision_rules.json").write_text(
        result.to_json(orient="records", indent=2) + "\n"
    )
    print("Fixed decision metrics written.")


if __name__ == "__main__":
    main()
