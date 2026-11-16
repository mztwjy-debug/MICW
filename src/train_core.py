"""Participant-grouped CT and MCT classifiers with weighted cross-entropy."""

from __future__ import annotations


import hashlib


import random


from dataclasses import dataclass


from pathlib import Path


import numpy as np


import pandas as pd


import torch


import torch.nn as nn


import warnings


from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


from torch.utils.data import DataLoader, Dataset


SEED = 42


VALIDATION_SEED = 42


GROUP_MAP = {}


FOLD_ARTIFACT_PATH = None


SEVERITY_ORDER = [0, 1, 2]


SEVERITY_NAME = {0: "low", 1: "medium", 2: "high"}


ACTION_NAME = {-1: "uncertain", 0: "safe", 1: "watch", 2: "intervene"}


PROBA_COLUMNS = [f"pred_proba_{SEVERITY_NAME[label]}" for label in SEVERITY_ORDER]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_seed(*parts: object) -> int:
    key = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return SEED + int(digest[:8], 16) % 1_000_000


def severity_level(score: float) -> int:
    if score <= 1:
        return 0
    if score <= 4:
        return 1
    return 2


@dataclass
class SubjectSequence:
    values: np.ndarray
    feature_masks: np.ndarray
    minutes: np.ndarray
    labels: np.ndarray


@dataclass
class Sample:
    subject: str
    end_idx: int
    minute_index: int
    label: int
    misc_score: float


class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        sequences: dict[str, SubjectSequence],
        samples: list[Sample],
        sequence_length: int,
    ) -> None:
        self.sequences = sequences
        self.samples = samples
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample = self.samples[idx]
        seq = self.sequences[sample.subject]
        start_idx = max(0, sample.end_idx - self.sequence_length + 1)
        window_values = seq.values[start_idx : sample.end_idx + 1]
        window_masks = seq.feature_masks[start_idx : sample.end_idx + 1]
        valid_len = int(window_values.shape[0])
        feature_dim = int(seq.values.shape[1])

        values = np.zeros((self.sequence_length, feature_dim), dtype=np.float32)
        feature_masks = np.zeros((self.sequence_length, feature_dim), dtype=np.float32)
        token_padding_mask = np.ones(self.sequence_length, dtype=bool)
        values[:valid_len] = window_values
        feature_masks[:valid_len] = window_masks
        token_padding_mask[:valid_len] = False

        return {
            "x": values,
            "feature_mask": feature_masks,
            "token_padding_mask": token_padding_mask,
            "valid_token_count": valid_len,
            "label": np.int64(sample.label),
            "misc_score": np.float32(sample.misc_score),
            "subject": sample.subject,
            "minute_index": sample.minute_index,
        }


def collate_batch(items: list[dict[str, object]]) -> dict[str, object]:
    labels = torch.tensor([int(item["label"]) for item in items], dtype=torch.long)
    x = torch.stack([torch.from_numpy(item["x"]).float() for item in items])
    feature_mask = torch.stack(
        [torch.from_numpy(item["feature_mask"]).float() for item in items]
    )
    token_padding_mask = torch.stack(
        [torch.from_numpy(item["token_padding_mask"]).bool() for item in items]
    )
    valid_token_counts = torch.tensor(
        [int(item["valid_token_count"]) for item in items], dtype=torch.long
    )
    misc_scores = torch.tensor(
        [float(item["misc_score"]) for item in items], dtype=torch.float32
    )
    return {
        "x": x,
        "feature_mask": feature_mask,
        "token_padding_mask": token_padding_mask,
        "valid_token_counts": valid_token_counts,
        "labels": labels,
        "misc_scores": misc_scores,
        "subjects": [str(item["subject"]) for item in items],
        "minute_indices": [int(item["minute_index"]) for item in items],
    }


class BaseTemporalModel(nn.Module):
    def gather_current(
        self, encoded: torch.Tensor, valid_token_counts: torch.Tensor
    ) -> torch.Tensor:
        current_idx = valid_token_counts - 1
        batch_idx = torch.arange(encoded.size(0), device=encoded.device)
        return encoded[batch_idx, current_idx]


class TransformerModel(BaseTemporalModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        max_sequence_length: int,
        d_model: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim * 2, d_model)
        self.positional = nn.Embedding(max_sequence_length, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, output_dim)

    def forward(
        self,
        values: torch.Tensor,
        feature_masks: torch.Tensor,
        token_padding_mask: torch.Tensor,
        valid_token_counts: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, _ = values.shape
        positions = (
            torch.arange(seq_len, device=values.device)
            .unsqueeze(0)
            .expand(batch, seq_len)
        )
        x = torch.cat([values, feature_masks], dim=-1)
        x = self.input_proj(x) + self.positional(positions)
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=values.device),
            diagonal=1,
        )
        encoded = self.encoder(
            x, mask=causal_mask, src_key_padding_mask=token_padding_mask
        )
        pooled = self.dropout(self.gather_current(encoded, valid_token_counts))
        return self.head(pooled)


class MultiScaleTransformerModel(BaseTemporalModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        max_sequence_length: int,
        d_model: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        pooling_windows: tuple[int, ...] = (5, 10, 20),
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim * 2, d_model)
        self.positional = nn.Embedding(max_sequence_length, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pooling_windows = pooling_windows
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model * (1 + len(pooling_windows)), output_dim)

    def forward(
        self,
        values: torch.Tensor,
        feature_masks: torch.Tensor,
        token_padding_mask: torch.Tensor,
        valid_token_counts: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, _ = values.shape
        positions = (
            torch.arange(seq_len, device=values.device)
            .unsqueeze(0)
            .expand(batch, seq_len)
        )
        x = torch.cat([values, feature_masks], dim=-1)
        x = self.input_proj(x) + self.positional(positions)
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=values.device),
            diagonal=1,
        )
        encoded = self.encoder(
            x, mask=causal_mask, src_key_padding_mask=token_padding_mask
        )

        current = self.gather_current(encoded, valid_token_counts)
        pooled_parts = [current]
        for window in self.pooling_windows:
            pooled = []
            for row_idx, valid_len in enumerate(valid_token_counts.tolist()):
                start = max(0, int(valid_len) - window)
                pooled.append(encoded[row_idx, start : int(valid_len)].mean(dim=0))
            pooled_parts.append(torch.stack(pooled, dim=0))
        return self.head(self.dropout(torch.cat(pooled_parts, dim=1)))


def build_model(
    model_name: str, input_dim: int, sequence_length: int, objective: str
) -> nn.Module:
    output_dim = 3
    if model_name == "transformer":
        return TransformerModel(
            input_dim=input_dim,
            output_dim=output_dim,
            max_sequence_length=sequence_length,
        )
    if model_name == "multiscale_transformer":
        return MultiScaleTransformerModel(
            input_dim=input_dim,
            output_dim=output_dim,
            max_sequence_length=sequence_length,
        )
    raise ValueError(f"Unknown model: {model_name}")


def add_relative_delta_features(token_df: pd.DataFrame) -> pd.DataFrame:
    token_df = token_df.copy()
    delta_cols = [col for col in token_df.columns if col.startswith("feature__delta_")]
    for delta_col in delta_cols:
        base_col = "feature__" + delta_col.removeprefix("feature__delta_")
        if base_col not in token_df.columns:
            continue
        rel_col = "feature__rel_delta_" + delta_col.removeprefix("feature__delta_")
        baseline_value = token_df[base_col] - token_df[delta_col]
        token_df[rel_col] = token_df[delta_col] / (baseline_value.abs() + 1e-6)
    return token_df


def absolute_feature_columns(token_df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in token_df.columns
        if col.startswith("feature__")
        and not col.startswith("feature__delta_")
        and not col.startswith("feature__rel_delta_")
        and not col.startswith("feature__baseline_centered_")
        and not col.startswith("feature__baseline_z_")
    ]


def add_subject_baseline_features(
    token_df: pd.DataFrame, baseline_minutes: int
) -> pd.DataFrame:
    token_df = token_df.copy()
    abs_cols = absolute_feature_columns(token_df)
    if not abs_cols:
        return token_df

    centered_values = np.full((len(token_df), len(abs_cols)), np.nan, dtype=np.float32)
    z_values = np.full((len(token_df), len(abs_cols)), np.nan, dtype=np.float32)

    for _, subject_df in token_df.groupby("subject", sort=False):
        subject_df = subject_df.sort_values("minute_index")
        if "is_baseline" in subject_df.columns:
            baseline_df = subject_df[subject_df["is_baseline"].astype(bool)]
        else:
            baseline_df = subject_df.head(baseline_minutes)
        if baseline_df.empty:
            baseline_df = subject_df.head(baseline_minutes)

        baseline_raw = baseline_df[abs_cols].to_numpy(dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            centers = np.nanmedian(baseline_raw, axis=0)
            scales = np.nanstd(baseline_raw, axis=0)
        centers = np.nan_to_num(centers, nan=0.0)
        scales = np.nan_to_num(scales, nan=1.0)
        scales[scales < 1e-6] = 1.0

        positions = subject_df.index.to_numpy()
        raw = subject_df[abs_cols].to_numpy(dtype=np.float32)
        centered = raw - centers
        centered_values[positions] = centered
        z_values[positions] = centered / scales

    clean_names = [col.removeprefix("feature__") for col in abs_cols]
    centered_df = pd.DataFrame(
        centered_values,
        columns=[f"feature__baseline_centered_{name}" for name in clean_names],
        index=token_df.index,
    )
    z_df = pd.DataFrame(
        z_values,
        columns=[f"feature__baseline_z_{name}" for name in clean_names],
        index=token_df.index,
    )
    return pd.concat([token_df, centered_df, z_df], axis=1)


def feature_columns_for_variant(token_df: pd.DataFrame, variant: str) -> list[str]:
    absolute = absolute_feature_columns(token_df)
    relative = [c for c in token_df if c.startswith("feature__rel_delta_")]
    variants = {
        "absolute_only": absolute,
        "relative_delta": relative,
        "absolute_plus_delta": [
            c
            for c in token_df
            if c.startswith("feature__")
            and not c.startswith(
                (
                    "feature__rel_delta_",
                    "feature__baseline_centered_",
                    "feature__baseline_z_",
                )
            )
        ],
        "absolute_plus_relative_delta": absolute + relative,
        "absolute_plus_baseline_zscore": absolute
        + [c for c in token_df if c.startswith("feature__baseline_z_")],
    }
    if variant not in variants:
        raise ValueError(f"Unknown feature variant: {variant}")
    if not variants[variant]:
        raise ValueError("No features match the requested variant")
    return variants[variant]


def standardize_sequences(
    token_df: pd.DataFrame,
    feature_columns: list[str],
    train_subjects: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    train_values = token_df[token_df["subject"].isin(train_subjects)][
        feature_columns
    ].to_numpy(dtype=np.float32)
    means = np.nanmean(train_values, axis=0)
    stds = np.nanstd(train_values, axis=0)
    means = np.nan_to_num(means, nan=0.0)
    stds = np.nan_to_num(stds, nan=1.0)
    stds[stds < 1e-6] = 1.0
    return means.astype(np.float32), stds.astype(np.float32)


def build_subject_sequences(
    token_df: pd.DataFrame,
    feature_columns: list[str],
    means: np.ndarray,
    stds: np.ndarray,
) -> dict[str, SubjectSequence]:
    sequences: dict[str, SubjectSequence] = {}
    for subject, subject_df in token_df.groupby("subject", sort=True):
        subject_df = subject_df.sort_values("minute_index").reset_index(drop=True)
        raw_values = subject_df[feature_columns].to_numpy(dtype=np.float32)
        finite_mask = np.isfinite(raw_values).astype(np.float32)
        values = (np.nan_to_num(raw_values, nan=means) - means) / stds
        values = np.nan_to_num(values, nan=0.0).astype(np.float32)
        labels = subject_df["current_severity_level"].to_numpy(dtype=int)
        sequences[subject] = SubjectSequence(
            values=values,
            feature_masks=finite_mask,
            minutes=subject_df["minute_index"].to_numpy(dtype=int),
            labels=labels,
        )
    return sequences


def build_samples(token_df: pd.DataFrame) -> list[Sample]:
    samples: list[Sample] = []
    for subject, subject_df in token_df.groupby("subject", sort=True):
        subject_df = subject_df.sort_values("minute_index").reset_index(drop=True)
        for end_idx, row in subject_df.iterrows():
            if pd.isna(row["misc_score"]) or pd.isna(row["current_severity_level"]):
                continue
            samples.append(
                Sample(
                    subject=subject,
                    end_idx=int(end_idx),
                    minute_index=int(row["minute_index"]),
                    label=int(row["current_severity_level"]),
                    misc_score=float(row["misc_score"]),
                )
            )
    return samples


def split_train_val_subjects(subjects: list[str]) -> tuple[list[str], list[str]]:
    # Split groups without joining their separate recording timelines.
    groups = sorted({GROUP_MAP.get(s, s) for s in subjects})
    if len(groups) < 2:
        raise ValueError("At least two training groups are required")
    rng = np.random.default_rng(VALIDATION_SEED)
    val_count = max(1, round(len(groups) * 0.2))
    val_groups = set(rng.permutation(groups)[:val_count])
    val_subjects = sorted(s for s in subjects if GROUP_MAP.get(s, s) in val_groups)
    train_subjects = sorted(
        s for s in subjects if GROUP_MAP.get(s, s) not in val_groups
    )
    return train_subjects, val_subjects


def evaluate_loader(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list: list[np.ndarray] = []
    label_list: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["x"].to(device),
                batch["feature_mask"].to(device),
                batch["token_padding_mask"].to(device),
                batch["valid_token_counts"].to(device),
            )
            logits_list.append(logits.cpu().numpy())
            label_list.append(batch["labels"].cpu().numpy())
    if not logits_list:
        return np.empty((0, 3), dtype=np.float32), np.array([], dtype=int)
    return np.concatenate(logits_list), np.concatenate(label_list)


def logits_to_predictions(
    logits: np.ndarray, objective: str
) -> tuple[np.ndarray, np.ndarray]:
    class_probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    return (class_probs.argmax(axis=1).astype(int), class_probs)


def train_one_fold(
    *,
    model_name: str,
    token_df: pd.DataFrame,
    feature_columns: list[str],
    train_subjects: list[str],
    test_subject: str | list[str],
    feature_variant: str,
    objective: str,
    label_smoothing: float,
    sequence_length: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> pd.DataFrame:
    if objective != "ce":
        raise ValueError("Only weighted cross-entropy is supported")
    set_seed(
        stable_seed(
            feature_variant,
            objective,
            label_smoothing,
            sequence_length,
            model_name,
            test_subject,
        )
    )
    train_subjects, val_subjects = split_train_val_subjects(sorted(train_subjects))
    means, stds = standardize_sequences(token_df, feature_columns, train_subjects)
    sequences = build_subject_sequences(token_df, feature_columns, means, stds)
    all_samples = build_samples(token_df)
    train_samples = [
        sample for sample in all_samples if sample.subject in train_subjects
    ]
    val_samples = [sample for sample in all_samples if sample.subject in val_subjects]
    test_subjects = (
        [test_subject] if isinstance(test_subject, str) else list(test_subject)
    )
    test_samples = [sample for sample in all_samples if sample.subject in test_subjects]
    split_groups = [
        {GROUP_MAP.get(s, s) for s in records}
        for records in (train_subjects, val_subjects, test_subjects)
    ]
    assert not (
        split_groups[0] & split_groups[1]
        or split_groups[0] & split_groups[2]
        or split_groups[1] & split_groups[2]
    ), "Group leakage"
    if not train_samples or not val_samples or (not test_samples):
        raise ValueError("Empty fold subset")
    train_loader = DataLoader(
        SlidingWindowDataset(sequences, train_samples, sequence_length),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        SlidingWindowDataset(sequences, val_samples, sequence_length),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        SlidingWindowDataset(sequences, test_samples, sequence_length),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    model = build_model(
        model_name,
        input_dim=len(feature_columns),
        sequence_length=sequence_length,
        objective=objective,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_labels = np.array([sample.label for sample in train_samples], dtype=int)
    counts = np.bincount(train_labels, minlength=3).astype(np.float32)
    class_weights = len(train_labels) / np.maximum(counts * 3.0, 1.0)
    class_weight_tensor = torch.tensor(
        class_weights, dtype=torch.float32, device=device
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weight_tensor, label_smoothing=label_smoothing
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_val_metric = -float("inf")
    patience = 5
    stale_epochs = 0
    best_epoch = 0
    completed_epochs = 0
    for epoch in range(epochs):
        completed_epochs = epoch + 1
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(
                batch["x"].to(device),
                batch["feature_mask"].to(device),
                batch["token_padding_mask"].to(device),
                batch["valid_token_counts"].to(device),
            )
            labels = batch["labels"].to(device)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        val_logits, val_labels = evaluate_loader(model, val_loader, device)
        if val_labels.size:
            val_pred, _ = logits_to_predictions(val_logits, objective)
            val_metric = f1_score(
                val_labels,
                val_pred,
                average="macro",
                labels=SEVERITY_ORDER,
                zero_division=0,
            )
        else:
            val_metric = -float("inf")
        if val_metric > best_val_metric:
            best_val_metric = float(val_metric)
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    if FOLD_ARTIFACT_PATH is not None:
        import json

        Path(FOLD_ARTIFACT_PATH).write_text(
            json.dumps(
                {
                    "train_records": train_subjects,
                    "validation_records": val_subjects,
                    "test_records": test_subjects,
                    "best_epoch": best_epoch,
                    "completed_epochs": completed_epochs,
                    "best_validation_macro_f1": best_val_metric,
                    "feature_count": len(feature_columns),
                    "parameter_count": sum((p.numel() for p in model.parameters())),
                    "train_mean": means.tolist(),
                    "train_std": stds.tolist(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    model.eval()
    prediction_rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(
                batch["x"].to(device),
                batch["feature_mask"].to(device),
                batch["token_padding_mask"].to(device),
                batch["valid_token_counts"].to(device),
            )
            preds, probs = logits_to_predictions(logits.cpu().numpy(), objective)
            labels = batch["labels"].cpu().numpy().astype(int)
            for idx, subject in enumerate(batch["subjects"]):
                row = {
                    "feature_variant": feature_variant,
                    "objective": objective,
                    "label_smoothing": float(label_smoothing),
                    "model_name": model_name,
                    "subject": subject,
                    "minute_index": int(batch["minute_indices"][idx]),
                    "misc_score": float(batch["misc_scores"][idx]),
                    "true_label": int(labels[idx]),
                    "true_label_name": SEVERITY_NAME[int(labels[idx])],
                    "pred_label": int(preds[idx]),
                    "pred_label_name": SEVERITY_NAME[int(preds[idx])],
                    "window_seconds": int(token_df["window_seconds"].iloc[0]),
                    "sequence_length": int(sequence_length),
                    "valid_token_count": int(batch["valid_token_counts"][idx]),
                    "validation_macro_f1": float(best_val_metric),
                    "validation_subjects": ";".join(val_subjects),
                }
                for class_idx in SEVERITY_ORDER:
                    row[f"pred_proba_{SEVERITY_NAME[class_idx]}"] = float(
                        probs[idx, class_idx]
                    )
                prediction_rows.append(row)
    return pd.DataFrame(prediction_rows)


def compute_multiclass_metrics(pred_df: pd.DataFrame) -> dict[str, object]:
    y_true = pred_df["true_label"].to_numpy(dtype=int)
    y_pred = pred_df["pred_label"].to_numpy(dtype=int)
    counts = pd.Series(y_true).value_counts().reindex(SEVERITY_ORDER, fill_value=0)
    per_class_f1 = f1_score(
        y_true, y_pred, average=None, labels=SEVERITY_ORDER, zero_division=0
    )
    adjacent_correct = np.abs(y_true - y_pred) <= 1
    severe_error = np.abs(y_true - y_pred) == 2
    return {
        "feature_variant": pred_df["feature_variant"].iloc[0],
        "objective": pred_df["objective"].iloc[0],
        "label_smoothing": float(pred_df["label_smoothing"].iloc[0]),
        "sequence_length": int(pred_df["sequence_length"].iloc[0]),
        "model_name": pred_df["model_name"].iloc[0],
        "rows": len(pred_df),
        "subjects": pred_df["subject"].nunique(),
        "low_rows": int(counts.loc[0]),
        "medium_rows": int(counts.loc[1]),
        "high_rows": int(counts.loc[2]),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true, y_pred, average="macro", labels=SEVERITY_ORDER, zero_division=0
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                labels=SEVERITY_ORDER,
                zero_division=0,
            )
        ),
        "low_f1": float(per_class_f1[0]),
        "medium_f1": float(per_class_f1[1]),
        "high_f1": float(per_class_f1[2]),
        "adjacent_accuracy": float(adjacent_correct.mean()),
        "severe_error_rate": float(severe_error.mean()),
    }


def apply_monitoring_decision_layer(
    pred_df: pd.DataFrame,
    *,
    smoothing_window: int,
    confidence_threshold: float,
    margin_threshold: float,
    high_persistence_minutes: int,
) -> pd.DataFrame:
    decision_df = (
        pred_df.sort_values(["subject", "minute_index"]).reset_index(drop=True).copy()
    )
    smoothed_columns: list[str] = []
    for col in PROBA_COLUMNS:
        smooth_col = f"smoothed_{col}"
        smoothed_columns.append(smooth_col)
        if smoothing_window <= 1:
            decision_df[smooth_col] = decision_df[col].astype(float)
        else:
            decision_df[smooth_col] = (
                decision_df.groupby("subject", sort=False)[col]
                .transform(
                    lambda values: values.rolling(
                        window=smoothing_window, min_periods=1
                    ).mean()
                )
                .astype(float)
            )

    probabilities = decision_df[smoothed_columns].to_numpy(dtype=float)
    smoothed_pred = probabilities.argmax(axis=1).astype(int)
    sorted_probs = np.sort(probabilities, axis=1)
    max_probs = sorted_probs[:, -1]
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    confident = (max_probs >= confidence_threshold) & (margins >= margin_threshold)
    decision_labels = np.where(confident, smoothed_pred, -1).astype(int)

    if high_persistence_minutes > 1:
        base_decisions = decision_labels.copy()
        for positions in decision_df.groupby("subject", sort=False).indices.values():
            high_streak = 0
            for pos in positions:
                if base_decisions[pos] == 2:
                    high_streak += 1
                    if high_streak < high_persistence_minutes:
                        decision_labels[pos] = 1
                else:
                    high_streak = 0

    decision_df["smoothing_window"] = int(smoothing_window)
    decision_df["confidence_threshold"] = float(confidence_threshold)
    decision_df["margin_threshold"] = float(margin_threshold)
    decision_df["high_persistence_minutes"] = int(high_persistence_minutes)
    decision_df["smoothed_pred_label"] = smoothed_pred
    decision_df["smoothed_pred_label_name"] = [
        SEVERITY_NAME[int(label)] for label in smoothed_pred
    ]
    decision_df["decision_label"] = decision_labels
    decision_df["decision_label_name"] = [
        ACTION_NAME[int(label)] for label in decision_labels
    ]
    decision_df["decision_max_proba"] = max_probs
    decision_df["decision_margin"] = margins
    decision_df["is_confident_decision"] = decision_labels >= 0
    return decision_df
