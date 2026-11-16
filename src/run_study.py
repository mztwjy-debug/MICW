#!/usr/bin/env python3
"""Run fixed architecture, sensitivity, classical and decision analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(script, *arguments):
    subprocess.run(
        [sys.executable, str(ROOT / "src" / script), *map(str, arguments)], check=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--groups-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["all", "main", "sensitivity", "classical", "decisions"],
        default=["main"],
    )
    parser.add_argument("--earlier-prepared", type=Path)
    parser.add_argument("--missingness-prepared", type=Path)
    parser.add_argument(
        "--seed-manifest",
        type=Path,
        default=ROOT / "manifests/initialization_seeds.csv",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    tasks = (
        {"main", "sensitivity", "classical", "decisions"}
        if "all" in args.tasks
        else set(args.tasks)
    )
    if "sensitivity" in tasks and not (
        args.earlier_prepared and args.missingness_prepared
    ):
        parser.error(
            "All ten sensitivities require --earlier-prepared and --missingness-prepared"
        )
    if tasks & {"sensitivity", "decisions"} and "main" not in tasks:
        reference = args.out / "main/multiscale_transformer_seed42/predictions.csv"
        if not reference.is_file():
            parser.error(
                "Run the main analysis before sensitivity or decision evaluation"
            )
    tokens = args.prepared / "state/tokens.csv"
    config = json.loads((ROOT / "configs/study.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    common = [
        "--groups-csv",
        args.groups_csv,
        "--seed-manifest",
        args.seed_manifest,
        "--device",
        args.device,
        "--epochs",
        config["training"]["max_epochs"],
    ]
    if "main" in tasks:
        run(
            "run_grouped.py",
            "--tokens",
            tokens,
            *common,
            "--models",
            *config["models"],
            "--seeds",
            *config["training"]["seeds"],
            "--out",
            args.out / "main",
        )
    if "classical" in tasks:
        run(
            "classical_baselines.py",
            "--base-dir",
            args.prepared,
            "--groups-csv",
            args.groups_csv,
            "--output-dir",
            args.out / "classical",
        )
    if (
        tasks & {"main", "classical"}
        and (args.out / "main/multiscale_transformer_seed42/predictions.csv").is_file()
    ):
        options = []
        time_only = args.out / "classical/time_only/all_tokens_predictions.csv"
        if time_only.is_file():
            options = ["--time-only-predictions", time_only]
        run(
            "summarize_grouped.py",
            "--runs-dir",
            args.out / "main",
            "--groups-csv",
            args.groups_csv,
            "--out",
            args.out / "statistics",
            *options,
        )
    if "sensitivity" in tasks:
        datasets = {
            "main": tokens,
            "earlier_signal": args.earlier_prepared / "state/tokens.csv",
            "missingness": args.missingness_prepared / "state/tokens.csv",
        }
        for condition in config["sensitivities"]:
            options = ["--variant", condition["variant"], "--mask", condition["mask"]]
            run(
                "run_grouped.py",
                "--tokens",
                datasets[condition["dataset"]],
                *common,
                "--models",
                "multiscale_transformer",
                "--seeds",
                "42",
                *options,
                "--out",
                args.out / "sensitivities" / condition["name"],
            )
        run(
            "summarize_sensitivities.py",
            "--reference",
            args.out / "main/multiscale_transformer_seed42/predictions.csv",
            "--conditions-dir",
            args.out / "sensitivities",
            "--groups-csv",
            args.groups_csv,
            "--out",
            args.out / "sensitivity_statistics",
        )
    if "decisions" in tasks:
        predictions = args.out / "main/multiscale_transformer_seed42/predictions.csv"
        if not predictions.is_file():
            parser.error(
                "Decision evaluation requires the main seed-42 MCT predictions"
            )
        run(
            "evaluate_decisions.py",
            "--predictions-csv",
            predictions,
            "--out",
            args.out / "decisions",
        )


if __name__ == "__main__":
    main()
