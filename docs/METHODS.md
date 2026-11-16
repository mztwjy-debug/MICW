# Methods and outputs

## Features and classification

Three electrophysiological channels contribute 14 features each; six vital-sign
channels contribute seven each, giving 84 absolute features. The first 300 seconds
provide a recording-specific reference `b`. Relative features are
`(a - b) / (abs(b) + 1e-6)`. The reference interval is not assumed symptom-free.
Each state sample describes a complete 60-second window; Low, Medium and High
correspond to MISC 0–1, 2–4 and 5–10. Historical sequences contain at most 20
minute tokens from one recording and are right-padded.

CT and MCT use 64-dimensional embeddings, two encoder layers, four heads,
128-dimensional feed-forward blocks and dropout 0.2. MCT concatenates the
current hidden state and means over the last 5, 10 and 20 valid tokens.
CT/MCT mask future and padded tokens. The pinned PatchTST and iTransformer
classification implementations receive identical values and validity channels
but retain their own attention and padding behavior; model sizes differ.

Leave-one-participant-out evaluation separates fitting, validation and test
participants. Validation uses 20% of the remaining groups and seed 42.
Standardization, missing-feature replacement and class weights use only fitting
participants. Training includes reference tokens; primary evaluation uses
`minute_index > 5`. Adam uses learning rate 0.001, batch size 64, weighted
cross-entropy with label smoothing 0.05, at most 18 epochs and early-stopping
patience five. The matched comparison uses seeds 42, 7 and 123.

## Sensitivities and baselines

All sensitivities use MCT and seed 42. Conditions cover absolute-only,
relative-only, absolute plus unnormalized reference differences, absolute plus
reference Z-scores, omission of blood pressure/electrophysiology/vitals,
20% Bernoulli token masking, matched earlier signal exports and restored earlier
vital missingness. Feature selection preserves source column order. Z-scores
use the median and population SD of the first five absolute-feature tokens;
scales below 1e-6 are set to one. This differs from the full-interval main reference.

Modality masks apply during both fitting and evaluation. Random masking uses
seed 20261114 after reference features are computed. It tests post-calibration
missing input vectors, not raw-sensor dropout or motion artifacts.

The time-only baseline is a 300-tree class-balanced random forest on minute index.
Its primary analysis fits all training tokens and evaluates post-reference
minutes. Post-reference-only fitting is a separate auxiliary analysis.
The constant-Low baseline uses the same evaluation keys.

## First-threshold classification

The first labeled MISC ≥2 minute defines the event. Pre-event minutes and the
event minute are eligible; post-event minutes are excluded. Participants crossing
during the reference period contribute no primary at-risk samples. For minute
`t`, the input interval is `[60(t-1)-W, 60(t-1))` for W = 60, 120, 180 or 300 seconds.
Primary fits and evaluation use the same post-reference keys for all four windows.
The model is an RBF SVM with C=1, gamma=scale and balanced class weights;
median imputation and scaling use outer-training participants only. Probability
calibration uses training rows and is not group-aware. These endpoints do not
establish physical onset timing or a prospective warning horizon.

## Decisions and statistics

`evaluate_decisions.py` applies probability averaging, confidence/margin rejection
and High persistence to complete recording histories before selecting evaluation
minutes. The four fixed rules are no rejection; margin ≥0.20; confidence ≥0.60;
and a three-minute average with confidence ≥0.65 and three consecutive High
predictions. High predictions awaiting persistence are assigned Watch.

Covered accuracy and macro F1 use accepted samples. High recall includes all
true High targets. Extreme Low/High errors are reported over both all targets
and accepted targets. These are classification outcomes, not observed safety
or intervention effects.

Statistics pool target minutes, report mean and sample SD across training seeds,
and resample complete participant groups 10,000 times with seed 20261114.
Paired comparisons share resampling weights and average within-seed differences.
Three classes always enter macro F1; balanced accuracy averages classes present
in a resample. No minute-wise independence is assumed.

## Signal preprocessing

`preprocess_signals.py` accepts an aligned EDA/EGG export and source EMG with
explicit source sampling rate and post-resampling lag. It scales EMG by 1,000,
selectively resamples 1,000 Hz to 500 Hz, expands absolute-amplitude outliers by
three samples and performs up to two PCHIP interpolation passes. The threshold
is max(6 × 1.4826 × MAD, 150 µV); if the interpolated count exceeds 2% of samples,
the cleaning is repeated with a 300-µV floor. Linear detrending and a fourth-order
20–240-Hz Butterworth design are applied with forward–backward filtering.
Negative EDA values are clipped to zero; EGG is retained. No PSD correction is
written to the time-domain export. Do not process already filtered study exports
a second time. Source rates, offsets and instrument calibration require their
own acquisition evidence; this script does not infer them.

Upstream EDA/EGG passbands and some vital-sign modifications are incompletely
documented. A shared blood-pressure sequence remains a measurement-provenance
limitation. Sensitivity experiments do not establish measurement authenticity.
The available data represent one vehicle/route and an internal cohort; prospective
alarm benefit and external-population performance remain unestablished.

## Output mapping

| Analysis | Output under the study output directory |
|---|---|
| Matched architecture comparison | `statistics/main_metrics_seed_summary.csv` |
| Seed-42 confidence intervals | `statistics/reference_seed_group_bootstrap.csv` |
| Paired architecture differences | `statistics/paired_architecture_differences.csv` |
| Constant-Low baseline | `statistics/constant_low_metrics.csv` |
| Reference baseline confidence intervals | `statistics/reference_baseline_confidence_intervals.csv` |
| Time-only baseline | `classical/time_only_summary.csv` |
| Representation and other sensitivities | `sensitivity_statistics/sensitivity_metrics.csv` |
| Paired sensitivity differences | `sensitivity_statistics/sensitivity_paired_differences.csv` |
| Fixed decision rules | `decisions/fixed_decision_rules.csv` |
| First-threshold window comparison | `classical/onset_summary.csv` |

Training creates per-record predictions and fold fitting statistics locally.
Keep generated outputs and all research inputs outside the public repository;
they are excluded by `.gitignore`.
