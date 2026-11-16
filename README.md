# MICW

Code for **Retrospective Minute-Level Motion Sickness Monitoring in Battery
Electric Vehicle Occupants: The MICW Framework**.

MICW classifies minute-level MISC categories from physiological features using
participant-specific references and historical context. It includes CT, MCT,
PatchTST and iTransformer, fixed classical baselines, sensitivity analyses,
decision rules and participant-level bootstrap statistics.

## Installation

Use Python 3.12 in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`. Select a PyTorch build
compatible with your CPU or CUDA environment if needed. The study used
Python 3.12.7 and the package versions pinned in `requirements.txt`.

## Quick start

Run a small artificial-data example on CPU:

```bash
python examples/make_synthetic_example.py --out synthetic
python src/run_grouped.py --tokens synthetic/tokens.csv --groups-csv synthetic/groups.csv --models transformer --seeds 42 --epochs 1 --device cpu --out outputs/example
```

The synthetic scores have no scientific interpretation.

## Study analyses

Prepare anonymous, aligned and processed signal/vital exports as described in
[Data format](docs/DATA_FORMAT.md):

```bash
python src/build_tokens.py --manifest inputs/recordings.csv --labels inputs/labels.csv --out prepared
python src/run_study.py --prepared prepared --groups-csv manifests/group_manifest.csv --out outputs/study --tasks main classical decisions
```

For all ten sensitivities, supply the matched earlier exports through the
extended manifest described in the data-format documentation:

```bash
python src/build_variants.py --manifest inputs/versions.csv --labels inputs/labels.csv --out variants
python src/run_study.py --prepared prepared --groups-csv manifests/group_manifest.csv --earlier-prepared variants/earlier_signal/prepared --missingness-prepared variants/missingness/prepared --out outputs/study --tasks sensitivity
```

Use `--tasks all` with both variant arguments to run all analyses together.
The complete study involves 47 outer folds per neural model and seed;
`--device cpu` is supported and `--device auto` selects CUDA when available.

`configs/study.json` records the fixed protocol. The scripts expose the matching
training settings and named analyses; this file is not a hyperparameter-search
interface. See [Methods](docs/METHODS.md) for analysis definitions and outputs.

## Data and reproducibility

The study comprises 47 independent participants, with one recording per person.
The manifests contain only anonymous participant groups and initialization
seeds for the included experiments. Train/validation/test splits are generated
deterministically by the runner using validation seed 42. Pass
`--seed-manifest manifests/initialization_seeds.csv` when invoking
`src/run_grouped.py` directly to preserve study initialization.

Participant recordings, MISC labels, feature tables, identity mappings,
participant-level predictions and trained weights are not distributed.
Access to study data is subject to ethics and privacy restrictions through
the corresponding authors. Reproducing the reported numerical results requires
authorized access to those inputs; the synthetic example demonstrates execution.
GPU arithmetic may differ across hardware and library builds.

The analyses are offline and retrospective. A full five-minute reference and
bidirectional signal processing prevent an end-to-end real-time interpretation.
Preceding input-window length is not a validated warning lead time.

## Repository

- `src/`: feature extraction, models, training and evaluation.
- `configs/`: fixed study protocol.
- `manifests/`: anonymous grouping and initialization seeds.
- `examples/`: a synthetic input generator.
- `vendor/`: pinned third-party model implementations and license.
- `docs/`: input specifications, methods and output definitions.

## License

MICW code and documentation are distributed under the [MIT License](LICENSE).
The model implementations in `vendor/tslib_modern_baselines/` are from
[Time-Series-Library](https://github.com/thuml/Time-Series-Library), commit
`4e938a1767106324dd753b2a44832bf870a0252e`, Copyright (c) 2021 THUML @ Tsinghua
University. Their [MIT license](vendor/tslib_modern_baselines/LICENSE) is retained.
Dependencies installed through `requirements.txt` retain their respective licenses.
