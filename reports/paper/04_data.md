# §4 Data

## Source

We used the NOAA Storm Events Database for calendar years 2023 and 2024,
downloaded from the National Centers for Environmental Information FTP
archive (`StormEvents_details-ftp_v1.0_d2023` and the 2024 release of the
same table). Each row in the raw archive represents one reported severe-
weather event and carries 51 columns covering identifiers, temporal fields,
geographic fields, magnitude and damage estimates, and the categorical
`EVENT_TYPE` label that we treat as the classification target. The 51
raw-column count is what we use to satisfy the project's §2.2 high-volume
tabular data requirement at the source level; the working feature set
described below is a smaller, engineered subset.

## Filtering and retained classes

The raw `EVENT_TYPE` vocabulary in the NOAA archive contains dozens of
categories with a heavy long tail (single-digit annual counts for some
labels). We restricted the analysis to the ten most frequent event types
in the combined 2023 + 2024 archive, which together account for 117,717
records. The retained classes and their support in the combined dataset
are shown in Figure&nbsp;1.

| class | combined N | share |
| --- | ---: | ---: |
| Thunderstorm Wind | 41,177 | 35.0% |
| Hail | 20,702 | 17.6% |
| Drought | 9,288 | 7.9% |
| Flash Flood | 8,286 | 7.0% |
| Excessive Heat | 7,684 | 6.5% |
| High Wind | 7,485 | 6.4% |
| Heat | 6,494 | 5.5% |
| Winter Weather | 6,448 | 5.5% |
| Flood | 5,318 | 4.5% |
| Marine Thunderstorm Wind | 4,835 | 4.1% |
| **total** | **117,717** | 100% |

(Source: `tool/weather.ipynb` data-loading cell; class supports also
recoverable as `random_split.*.support` × 5 in `temporal_per_class.json`.)
The distribution is moderately imbalanced — Thunderstorm Wind alone
covers a third of the dataset and the bottom six classes each fall below
8%. We report macro-F1 throughout as the headline metric so the long-tail
classes are not drowned out by Thunderstorm Wind, and we do not apply
class-reweighting to either model so the comparison is a like-for-like
test of representation rather than of loss-weighting tricks.

## Feature engineering

We engineered eight features from the raw schema
(`reports/feature_cols.json`): `MONTH` (1–12 from `MONTH_NAME`), `HOUR`
(extracted from the four-digit `BEGIN_TIME`), `DURATION_MIN` (the
difference between `END_DATE_TIME` and `BEGIN_DATE_TIME` in minutes),
`BEGIN_LAT` and `BEGIN_LON` (event-onset coordinates), `CZ_TYPE_ENC` (a
binary indicator for County vs Zone reporting), and label-encoded
`STATE_ENC` and `WFO_ENC` (state code and NWS Forecast Office). All eight
features are shared by both models so that the XGBoost-vs-TabNet
comparison is not confounded by feature availability.

This is a deliberately small working set. The motivation is twofold.
First, the raw NOAA columns are dominated by sparse free-text and post-
event damage estimates that leak label-correlated information at training
time — restricting to pre-event temporal and geographic features keeps
the task honest. Second, the small feature count makes the regime
interesting for our central question: TabNet's design advantage is
attention-based feature *filtering*, and it is not obvious it can pay
that advantage off when only eight features are available and all of
them carry signal. We return to this point in §11.

## Splits

We evaluated each model under two splits.

**Random split.** A single 80/20 stratified split over the combined
2023 + 2024 dataset, with `random_state = 42`. The train fold has 94,173
rows and the test fold has 23,544 rows (confirmed against
`error_overlap.n_test` in `week4_results.json`). All headline numbers in
§6, §7, and §8 are computed on this test fold unless otherwise stated.

**Temporal split.** A pre-registered out-of-time evaluation in which the
2023 archive is the train fold (62,656 rows) and the 2024 archive is the
test fold (55,061 rows; confirmed against the `macro avg.support` entries
under `temporal_split.{xgb,tabnet}` in `temporal_per_class.json`). The
class supports differ markedly between random and temporal test folds
because 2024 saw a substantially higher Thunderstorm Wind and Hail count
than the dataset average; this is an artifact of inter-annual variability
in severe-weather frequency, not of our split construction. The temporal
split exists to test generalization under year-to-year distribution
shift, and we report it alongside the random-split numbers in §6 and §9.

## Missing-value treatment

`BEGIN_LAT` and `BEGIN_LON` are missing for roughly 39.7% of records in
the combined dataset. We chose to impute rather than drop, using a
three-level hierarchical median fill fit on the *training* rows only:
first by `(STATE, CZ_FIPS)`, then by `(STATE, WFO)`, finally by `STATE`.
We checked that this choice does not change the headline conclusions by
training both models on a drop-NaN variant and evaluating on a common
native-coord test subset; the XGBoost macro-F1 changed by less than
6e-5 between impute and drop, while TabNet preferred imputation by
0.022 macro-F1 (`week4_results.json: nan_coord_policy`). We therefore
report impute throughout. No other feature contains missing values.

---

**Figure&nbsp;1.** `reports/figures/class_distribution.png` — class-
frequency bar chart for the combined 2023 + 2024 retained-class dataset
(N = 117,717). Source: `tool/_make_class_distribution_fig.py`.
