# Data format

All paths in input manifests are relative to the manifest file. Use anonymous
ASCII recording and group identifiers, such as `R001` and `G001`. Every
recording from one participant must share one group; never derive groups by
removing digits from recording identifiers. Preserve recording order and
within-second acquisition order when preparing exports.

## Recording manifest

`recordings.csv` requires these columns:

| Column | Meaning |
|---|---|
| `record_id` | Unique anonymous recording ID |
| `group_id` | Anonymous participant ID |
| `signal_csv` | Aligned, processed `time,EDA,EGG,EMG` CSV |
| `vital_csv` | Aligned `time,SBP,DBP,HR,PR,SpO2,PTT` CSV |

Signal exports retain repeated second labels representing nominal 500-Hz
samples. Vital exports have one row per second. Timestamps use `H:MM:SS`, share
the same recording origin and do not cross midnight. The first 300 seconds
must be available. The processing convention is EDA in µS, EGG in mV and EMG
in µV; source instrument calibration is not independently established.
Use UTF-8 and decimal points. Unavailable values may be empty or `NaN`.
No vital-value interpolation or carry-forward is applied by feature extraction.

## Labels

`labels.csv` requires `record_id,minute_index,misc_score`.
`minute_index` is a positive integer starting at one; `misc_score` is an integer
from 0 to 10. Every record/minute pair is unique. Labels must refer to the same
original recording origin as the exports. Incomplete terminal minutes are
excluded; missing symptom labels are not invented or extended.

## Version-sensitivity inputs

`versions.csv` contains all recording-manifest columns plus:

| Column | Meaning |
|---|---|
| `earlier_signal_csv` | Corresponding earlier aligned signal export |
| `earlier_vital_csv` | Corresponding earlier vital export; repeated seconds allowed |

The earlier signal export uses the source EMG numerical convention to which
`build_variants.py` applies a factor of 1,000. Do not supply already converted
EMG in that column. Other channel names match the recording manifest.

The correspondence must be independently established before supplying this
manifest. Its rows are anonymous record-version links, not person-name maps.
The signal-version condition retains current vitals and labels. The missingness
condition retains current signals and values except cells that were missing
in any earlier same-second/channel candidate, or whose earlier second is absent.
It does not reconstruct true measurements or reverse other value edits.

## Prepared output

`build_tokens.py` writes:

- `state/tokens.csv`: 60-second state samples.
- `onset/window_60s.csv`, `window_120s.csv`, `window_180s.csv`,
  `window_300s.csv`: preceding-window first-threshold samples.

State tables contain anonymous `record_id,group_id`, minute index, MISC score,
class label, window boundaries and `feature__...` columns. Each channel's absolute
features are followed by its reference differences. The runner derives normalized
relative differences and the requested representation. Do not reorder feature
columns when comparing exact initialization and training behavior.

## Anonymous study configuration

`manifests/group_manifest.csv` maps 47 recordings to 47 participants.
The runner generates 47 outer folds with validation seed 42, each with 37
fitting, nine validation and one test participant. `initialization_seeds.csv` contains 752
configuration/group seeds: four models × three seeds × 47 participants for the
main representation, and four additional MCT representations × seed 42 ×
47 participants. Modality and data-version sensitivities share the main
representation's initialization policy.

For a different cohort, supply your own group manifest and omit the study seed
table when invoking `run_grouped.py`. The resulting anonymous-ID seed policy
is deterministic but does not reproduce the study's initialization.
