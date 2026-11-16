#!/usr/bin/env python3
"""Train recording-level histories using an explicit, anonymous grouping manifest."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch
import train_core as core


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_inputs(tokens_path, groups_path):
    tokens = pd.read_csv(tokens_path)
    groups = pd.read_csv(groups_path, dtype=str)
    if not {"record_id", "group_id"} <= set(groups):
        raise ValueError("groups-csv requires record_id and group_id")
    if (
        groups.record_id.duplicated().any()
        or groups[["record_id", "group_id"]].isna().any().any()
    ):
        raise ValueError("Each record must map to exactly one nonempty group")
    for column in ["record_id", "group_id"]:
        if (
            not groups[column]
            .map(lambda s: bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", s)))
            .all()
        ):
            raise ValueError("Use anonymous ASCII record/group identifiers")
    if not {"record_id", "minute_index", "misc_score"} <= set(tokens):
        raise ValueError("tokens require record_id, minute_index, and misc_score")
    tokens["record_id"] = tokens.record_id.astype(str)
    if set(tokens.record_id) != set(groups.record_id):
        raise ValueError(
            "Token and grouping manifests must contain the same record IDs"
        )
    features = [c for c in tokens if c.startswith("feature__")]
    if not features:
        raise ValueError("No feature__ columns found")
    # Do not propagate any optional name/path/metadata columns into outputs.
    tokens = tokens[["record_id", "minute_index", "misc_score"] + features].copy()
    tokens["subject"] = tokens.record_id
    minutes = pd.to_numeric(tokens.minute_index, errors="raise").to_numpy(float)
    if not np.isfinite(minutes).all() or not np.equal(minutes, np.floor(minutes)).all():
        raise ValueError("Minute indices must be finite integers")
    tokens["minute_index"] = minutes.astype(int)
    tokens["misc_score"] = pd.to_numeric(tokens.misc_score, errors="raise")
    if tokens.duplicated(["record_id", "minute_index"]).any():
        raise ValueError("Duplicate record/minute")
    if (
        not tokens.misc_score.between(0, 10).all()
        or not np.equal(tokens.misc_score, np.floor(tokens.misc_score)).all()
        or not tokens.minute_index.ge(1).all()
    ):
        raise ValueError("Invalid MISC score or minute index")
    numeric = tokens[features].apply(pd.to_numeric, errors="raise")
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("Infinite features are invalid; use NaN for missing values")
    tokens[features] = numeric
    tokens = tokens.copy()
    tokens["is_baseline"] = tokens.minute_index.le(5)
    tokens["window_seconds"] = 60
    tokens["current_severity_level"] = tokens.misc_score.map(
        core.severity_level
    ).astype(int)
    tokens = tokens.sort_values(["record_id", "minute_index"]).reset_index(drop=True)
    return tokens, dict(zip(groups.record_id, groups.group_id))


def split_plan(records, mapping, validation_seed=42):
    core.GROUP_MAP = mapping
    core.VALIDATION_SEED = validation_seed
    plan = []
    for group in sorted(set(mapping.values())):
        test = [r for r in records if mapping[r] == group]
        candidates = [r for r in records if mapping[r] != group]
        train, validation = core.split_train_val_subjects(candidates)
        partitions = [{mapping[r] for r in ids} for ids in (train, validation, test)]
        if any(partitions[a] & partitions[b] for a, b in [(0, 1), (0, 2), (1, 2)]):
            raise ValueError("Overlapping train/validation/test groups")
        plan.append(
            {
                "group_id": group,
                "train_records": train,
                "validation_records": validation,
                "test_records": test,
            }
        )
    return plan


def check_predictions(pred, expected):
    key = ["subject", "minute_index"]
    if pred.duplicated(key).any():
        raise ValueError("Duplicate prediction key")
    joined = expected[key + ["current_severity_level"]].merge(
        pred, on=key, how="outer", validate="one_to_one", indicator=True
    )
    if (
        not joined["_merge"].eq("both").all()
        or not joined.true_label.eq(joined.current_severity_level).all()
    ):
        raise ValueError("Prediction rows or labels do not match this fold")
    probabilities = pred[core.PROBA_COLUMNS].to_numpy()
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
    ):
        raise ValueError("Invalid prediction probabilities")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--groups-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-manifest", type=Path)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["multiscale_transformer", "transformer"],
        choices=["multiscale_transformer", "transformer", "patchtst", "itransformer"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    parser.add_argument(
        "--variant",
        default="absolute_plus_relative_delta",
        choices=[
            "absolute_only",
            "relative_delta",
            "absolute_plus_delta",
            "absolute_plus_relative_delta",
            "absolute_plus_baseline_zscore",
        ],
    )
    parser.add_argument(
        "--mask",
        choices=["none", "electro", "vitals", "bp", "missing20"],
        default="none",
    )
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.sequence_length < 1:
        parser.error("epochs and sequence-length must be positive")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    df, mapping = read_inputs(args.tokens, args.groups_csv)
    records = sorted(df.subject.unique())
    plan = split_plan(records, mapping)
    feature_cols = [c for c in df if c.startswith("feature__")]
    masked = []
    if args.mask in ["electro", "vitals", "bp"]:
        for column in feature_cols:
            low = column.lower()
            if (
                (args.mask == "electro" and "signal_" in low)
                or (args.mask == "vitals" and "vital_" in low)
                or (
                    args.mask == "bp"
                    and any(
                        x in low
                        for x in ["vital_sbp_", "vital_dbp_", "systolic", "diastolic"]
                    )
                )
            ):
                masked.append(column)
        if not masked:
            raise ValueError("No features matched the requested ablation")
        df[masked] = np.nan
    df = core.add_relative_delta_features(df)
    df = core.add_subject_baseline_features(df, baseline_minutes=5)
    columns = core.feature_columns_for_variant(df, args.variant)
    missing = np.zeros(len(df), dtype=bool)
    if args.mask == "missing20":
        missing = np.random.default_rng(20261114).random(len(df)) < 0.20
        df.loc[missing, columns] = np.nan
    seeds = pd.read_csv(args.seed_manifest) if args.seed_manifest else None
    source_root = Path(__file__).resolve().parent.parent
    hashes = {
        str(p.relative_to(source_root)): sha(p)
        for p in sorted(source_root.rglob("*.py"))
    }
    config = {
        "tokens_sha256": sha(args.tokens),
        "groups_sha256": sha(args.groups_csv),
        "seed_manifest_sha256": sha(args.seed_manifest) if args.seed_manifest else None,
        "models": args.models,
        "seeds": args.seeds,
        "variant": args.variant,
        "mask": args.mask,
        "features": columns,
        "masked_columns": masked,
        "masked_tokens": int(missing.sum()),
        "epochs": args.epochs,
        "sequence_length": args.sequence_length,
        "validation_seed": 42,
        "batch_size": 64,
        "learning_rate": 0.001,
        "label_smoothing": 0.05,
        "source_sha256": hashes,
        "records": len(records),
        "groups": len(plan),
        "rows": len(df),
        "torch": torch.__version__,
        "device_request": args.device,
        "seed_policy": "frozen initialization table"
        if seeds is not None
        else "hash of anonymous configuration and test record IDs",
        "scope": "offline retrospective; missing20 masks tokens after reference features are computed",
    }
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = out / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != config:
        raise ValueError("Existing output has a different protocol")
    protocol_path.write_text(json.dumps(config, indent=2) + "\n")
    (out / "split_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if args.dry_run:
        print("Input, explicit group assignments, and split plan verified.")
        return
    if any(model in ["patchtst", "itransformer"] for model in args.models):
        import modern_models

        default_builder = core.build_model
        core.build_model = lambda name, **kw: (
            modern_models.builder(name, **kw)
            if name in ["patchtst", "itransformer"]
            else default_builder(name, **kw)
        )
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    default_stable_seed = core.stable_seed
    summaries = []
    for seed in args.seeds:
        core.SEED = seed
        for model in args.models:
            folder = out / f"{model}_seed{seed}"
            folder.mkdir(exist_ok=True)
            predictions = []
            for fold, item in enumerate(plan):
                test = item["test_records"]
                group = item["group_id"]
                expected = df[df.subject.isin(test)]
                predpath = folder / f"fold_{fold:02d}.csv"
                donepath = folder / f"fold_{fold:02d}.complete.json"
                if seeds is not None:
                    match = seeds[
                        (seeds.seed == seed)
                        & (seeds.model_name == model)
                        & (seeds.group_id == group)
                        & (seeds.feature_variant == args.variant)
                        & (seeds.objective == "ce")
                        & (seeds.sequence_length == args.sequence_length)
                        & (seeds.label_smoothing == 0.05)
                    ]
                    if len(match) != 1:
                        raise ValueError(
                            "Expected exactly one initialization seed for this configuration/fold"
                        )
                    init_seed = int(match.initialization_seed.iloc[0])
                    core.stable_seed = lambda *parts, value=init_seed: value
                else:
                    core.stable_seed = default_stable_seed
                    init_seed = core.stable_seed(
                        args.variant, "ce", 0.05, args.sequence_length, model, test
                    )
                if predpath.exists() and donepath.exists():
                    done = json.loads(donepath.read_text())
                    if (
                        done["prediction_sha256"] != sha(predpath)
                        or done["initialization_seed"] != init_seed
                    ):
                        raise ValueError("Resumed fold integrity mismatch")
                    pred = pd.read_csv(predpath)
                else:
                    core.FOLD_ARTIFACT_PATH = folder / f"fold_{fold:02d}.json"
                    pred = core.train_one_fold(
                        model_name=model,
                        token_df=df,
                        feature_columns=columns,
                        train_subjects=sorted(
                            item["train_records"] + item["validation_records"]
                        ),
                        test_subject=test,
                        feature_variant=args.variant,
                        objective="ce",
                        label_smoothing=0.05,
                        sequence_length=args.sequence_length,
                        epochs=args.epochs,
                        batch_size=64,
                        learning_rate=0.001,
                        device=device,
                    )
                    pred["fold_group"] = group
                    check_predictions(pred, expected)
                    temporary = predpath.with_suffix(".csv.tmp")
                    pred.to_csv(temporary, index=False)
                    temporary.replace(predpath)
                    donepath.write_text(
                        json.dumps(
                            {
                                "prediction_sha256": sha(predpath),
                                "initialization_seed": init_seed,
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                check_predictions(pred, expected)
                predictions.append(pred)
                print(f"{model} seed={seed} fold={fold + 1}/{len(plan)}", flush=True)
            prediction = pd.concat(predictions, ignore_index=True)
            prediction = prediction.merge(
                df[["subject", "minute_index", "is_baseline"]],
                on=["subject", "minute_index"],
                validate="one_to_one",
            )
            check_predictions(prediction, df)
            prediction.to_csv(folder / "predictions.csv", index=False)
            for subset, p in [
                ("all", prediction),
                ("post_baseline", prediction[prediction.minute_index > 5]),
            ]:
                if len(p):
                    row = core.compute_multiclass_metrics(p)
                    row.update(seed=seed, subset=subset, groups=p.fold_group.nunique())
                    summaries.append(row)
            pd.DataFrame(summaries).to_csv(out / "metrics.csv", index=False)
    (out / "COMPLETE.json").write_text(
        json.dumps(
            {
                "model_seed_runs": len(args.models) * len(args.seeds),
                "records": len(records),
                "groups": len(plan),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
