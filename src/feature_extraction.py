"""Time- and frequency-domain feature functions for physiological windows."""

import numpy as np


def compute_slope(values: np.ndarray, sample_rate: float) -> float:
    if values.size < 2 or sample_rate <= 0:
        return np.nan
    x = np.arange(values.size, dtype=float) / sample_rate
    x_mean = x.mean()
    y_mean = values.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return 0.0
    return float(np.sum((x - x_mean) * (values - y_mean)) / denom)


def compute_signal_features(
    values: np.ndarray, duration_seconds: int
) -> dict[str, float]:
    feature_names = [
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
    if values.size == 0:
        return {name: np.nan for name in feature_names}

    sample_rate = values.size / max(duration_seconds, 1)
    diffs = np.diff(values)

    centered = values - np.mean(values)
    spectrum = np.fft.rfft(centered)
    power = (np.abs(spectrum) ** 2) / max(values.size, 1)
    freqs = (
        np.fft.rfftfreq(values.size, d=1.0 / sample_rate)
        if sample_rate > 0
        else np.zeros_like(power)
    )
    if power.size:
        power_no_dc = power.copy()
        power_no_dc[0] = 0.0
    else:
        power_no_dc = power

    total_power = float(np.sum(power_no_dc)) if power_no_dc.size else 0.0
    if power_no_dc.size > 1 and total_power > 0:
        dominant_freq = float(freqs[np.argmax(power_no_dc)])
        spectral_centroid = float(np.sum(freqs * power_no_dc) / total_power)
        prob = power_no_dc[1:] / np.sum(power_no_dc[1:])
        prob = prob[prob > 0]
        spectral_entropy = (
            float(-np.sum(prob * np.log(prob)) / np.log(len(prob)))
            if len(prob) > 1
            else 0.0
        )
    else:
        dominant_freq = 0.0
        spectral_centroid = 0.0
        spectral_entropy = 0.0

    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "range": float(np.max(values) - np.min(values)),
        "rms": float(np.sqrt(np.mean(values**2))),
        "slope": compute_slope(values, sample_rate),
        "diff_mean": float(np.mean(diffs)) if diffs.size else np.nan,
        "diff_std": float(np.std(diffs)) if diffs.size else np.nan,
        "dominant_freq": dominant_freq,
        "total_power": total_power,
        "spectral_centroid": spectral_centroid,
        "spectral_entropy": spectral_entropy,
    }


def compute_vital_features(
    values: np.ndarray, duration_seconds: int
) -> dict[str, float]:
    feature_names = [
        "mean",
        "std",
        "min",
        "max",
        "range",
        "slope",
        "last_value",
    ]
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return {name: np.nan for name in feature_names}

    sample_rate = finite_values.size / max(duration_seconds, 1)
    return {
        "mean": float(np.mean(finite_values)),
        "std": float(np.std(finite_values)),
        "min": float(np.min(finite_values)),
        "max": float(np.max(finite_values)),
        "range": float(np.max(finite_values) - np.min(finite_values)),
        "slope": compute_slope(finite_values, sample_rate),
        "last_value": float(finite_values[-1]),
    }


def prefix_features(prefix: str, features: dict[str, float]) -> dict[str, float]:
    return {f"feature__{prefix}_{name}": value for name, value in features.items()}
