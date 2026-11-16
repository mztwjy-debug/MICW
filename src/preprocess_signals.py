#!/usr/bin/env python3
"""Offline EMG processing and final EDA clipping for aligned signal exports."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import PchipInterpolator


def clean_points(x, k=6.0, floor=150.0, pad=3, iters=2):
    """Expand absolute-amplitude outliers and interpolate with PCHIP."""
    y = x.copy()
    total = 0
    for _ in range(iters):
        robust_sd = 1.4826 * np.median(np.abs(y - np.median(y)))
        mask = np.abs(y) > max(k * robust_sd, floor)
        if not mask.any():
            break
        extended = mask.copy()
        for distance in range(1, pad + 1):
            extended[distance:] |= mask[:-distance]
            extended[:-distance] |= mask[distance:]
        index = np.arange(y.size)
        good = ~extended
        if good.sum() < 10:
            break
        y[extended] = PchipInterpolator(index[good], y[good])(index[extended])
        total += int(extended.sum())
    return y, total


def process_emg(raw, rows, source_hz, lag_samples):
    """Convert the source export scale, align, clean and filter at 500 Hz."""
    if source_hz not in (500, 1000) or lag_samples < 0 or rows < 32:
        raise ValueError(
            "Require 500/1000 Hz, nonnegative lag and at least 32 aligned samples"
        )
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Raw EMG must be a finite one-dimensional array")
    values = values * 1000.0
    if source_hz == 1000:
        values = signal.resample_poly(values, 1, 2)
    if len(values) < rows:
        raise ValueError("Raw EMG is shorter than the aligned export")
    lag = (
        lag_samples if lag_samples + rows <= len(values) else max(0, len(values) - rows)
    )
    segment = values[lag : lag + rows]
    cleaned, count = clean_points(segment)
    if count / len(segment) > 0.02:
        cleaned, _ = clean_points(segment, floor=300.0)
    bandpass = signal.butter(4, (20.0, 240.0), btype="band", fs=500.0, output="sos")
    return signal.sosfiltfilt(bandpass, signal.detrend(cleaned)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aligned", type=Path, required=True, help="Aligned time,EDA,EGG CSV"
    )
    parser.add_argument(
        "--raw-emg", type=Path, required=True, help="Source CSV with EMG column"
    )
    parser.add_argument("--source-hz", type=int, choices=[500, 1000], required=True)
    parser.add_argument(
        "--lag-samples",
        type=int,
        required=True,
        help="Offset after resampling to 500 Hz",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.resolve() in {args.aligned.resolve(), args.raw_emg.resolve()}:
        parser.error("Output must differ from both inputs")
    aligned = pd.read_csv(args.aligned, dtype={"time": str, "EDA": str, "EGG": str})
    raw = pd.to_numeric(pd.read_csv(args.raw_emg)["EMG"], errors="raise").to_numpy(
        float
    )
    eda = pd.to_numeric(aligned.EDA, errors="raise").to_numpy(float)
    result = pd.DataFrame(
        dict(
            time=aligned.time,
            EDA=np.where(eda < 0, "0.0", aligned.EDA.to_numpy()),
            EGG=aligned.EGG.to_numpy(),
            EMG=process_emg(raw, len(aligned), args.source_hz, args.lag_samples).astype(
                np.float64
            ),
        )
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print("Processed signal export written.")


if __name__ == "__main__":
    main()
