# Severe-Weather Event Classification: XGBoost vs. TabNet

Multi-class classification of U.S. severe-weather events from the **NOAA Storm Events Database (2023–2024)**, comparing a tuned **gradient-boosted tree model (XGBoost)** against a deep-learning tabular model (**TabNet**) on the same engineered features.

## The result up front

On the primary random-split evaluation, **tuned XGBoost outperforms TabNet**: macro-F1 **0.900 vs. 0.851**. The gap is statistically reliable — a paired bootstrap puts the difference at **+0.049** (95% CI [+0.044, +0.054], p < 10⁻³), and a McNemar test on per-row correctness gives χ² = 591.6 (p ≪ 0.001). The project set out to test whether TabNet's attention-based feature selection would beat a strong tree baseline on this kind of tabular data; the honest answer here is that it did not.

## Problem and motivation

The task is to predict **which type of severe-weather event** occurred (out of 10 classes) from contextual tabular features — location, timing, duration, and reporting metadata — *without* using any of the event's measured outcomes. That last constraint matters: the raw database contains fields (deaths, injuries, damage, narratives, magnitude) that effectively reveal the answer. Those are deliberately excluded to avoid **target leakage**, which leaves a deliberately hard problem driven by geography and seasonality alone.

The class distribution is also heavily imbalanced — Thunderstorm Wind and Hail dominate, while several classes have far fewer examples — so **macro-F1** (which weights every class equally) is used as the headline metric rather than raw accuracy.

The ten classes:

`Drought`, `Excessive Heat`, `Flash Flood`, `Flood`, `Hail`, `Heat`, `High Wind`, `Marine Thunderstorm Wind`, `Thunderstorm Wind`, `Winter Weather`

## Data

- **Source:** [NOAA Storm Events Database](https://www.ncdc.noaa.gov/stormevents/) (NCEI), storm-details files for 2023 and 2024.
- **Files:** included in `data/` as `StormEvents_details-ftp_v1.0_d2023_*.csv[.gz]` and `..._d2024_*.csv[.gz]`. The `.csv.gz` archives are the source of truth; the notebook can read either.
- **Size after filtering:** the two years are combined and filtered to the **top 10 event types**, yielding **117,717 records**. These are split 80/20 (stratified) into **94,173 training** and **23,544 test** rows.
- **Raw width:** 51 columns per record, reduced to 8 engineered features (below).

## Approach

**Feature engineering (51 raw columns → 8 features).** All transforms are vectorized:

| Feature | Derived from |
|---|---|
| `MONTH` | `MONTH_NAME` mapped to 1–12 |
| `HOUR` | hour extracted from `BEGIN_TIME` |
| `DURATION_MIN` | `END_DATE_TIME` − `BEGIN_DATE_TIME` |
| `BEGIN_LAT`, `BEGIN_LON` | event coordinates (with imputation, below) |
| `STATE_ENC`, `WFO_ENC` | label-encoded state and forecast office |
| `CZ_TYPE_ENC` | county/zone reporting flag (binary) |

**Leakage control.** The train/test split is performed **before** imputation, and the lat/lon imputation medians are fit on the *training rows only*. A large share of records (the notebook reports ~40%) are missing coordinates; these are filled with a hierarchical median (state+county → state+forecast-office → state). An explicit experiment compares imputing vs. dropping those rows and finds the choice barely moves XGBoost (Δ ≈ 0.00005 macro-F1).

**Models compared.**
- **XGBoost** — evaluated both at default settings and after hyperparameter tuning.
- **TabNet** — a baseline plus a tuned configuration found via a coarse sweep over width / decision steps / learning rate, then refined over sparsity (`gamma`) and batch size with k-fold cross-validation.

**Validation.** The headline numbers come from the stratified random split. As a robustness check, a separate **temporal split** trains on 2023 and tests on 2024 (with imputation medians re-fit on 2023 alone) to confirm the result isn't an artifact of mixing years.

## Results

### Random split (primary)

| Model | Macro-F1 | Macro ROC-AUC |
|---|---|---|
| XGBoost (default) | 0.889 | 0.988 |
| **XGBoost (tuned)** | **0.900** | **0.991** |
| TabNet (baseline) | 0.783 | 0.976 |
| TabNet (tuned/refined) | 0.851 | 0.979 |

Tuned XGBoost wins, and the margin survives statistical testing (paired bootstrap and McNemar, above).

### Temporal split (train 2023 → test 2024)

| Model | Macro-F1 |
|---|---|
| XGBoost (tuned) | 0.756 |
| TabNet (tuned) | 0.749 |

Both models drop under year-over-year shift and the gap narrows, but XGBoost still leads on the aggregate metric.

### Error analysis

- **Hardest classes:** `Hail` and `Thunderstorm Wind` are the most confused — unsurprisingly, since they co-occur and share context. XGBoost beats TabNet on both under the random split (Hail by ≈0.11 F1, Thunderstorm Wind by ≈0.06 F1).
- **Easiest classes:** `Drought`, `Marine Thunderstorm Wind`, and `High Wind` are near-perfectly separated by both models.
- **Where the models disagree:** errors are only moderately correlated (Cohen's κ ≈ 0.47); on ~900 of the 23,544 test rows TabNet is correct where XGBoost is wrong, hinting an ensemble *could* help — though that wasn't pursued here.
- **One nuance, reported honestly:** under the 2023→2024 temporal shift, TabNet actually generalizes *better* than XGBoost on the two hardest classes (Hail and Thunderstorm Wind). This is a genuine but narrow exception to the headline; one annual fold is too thin to draw a broad robustness conclusion from.

### A calibration caveat (and why it doesn't change the conclusion)

TabNet posts a much lower Expected Calibration Error (0.004 vs. XGBoost's 0.025), which initially looked like a point in its favor. A short audit shows this doesn't hold up: proper scoring rules that reward *both* calibration and sharpness favor XGBoost (Brier 0.179 vs. 0.248; NLL 0.312 vs. 0.408), and a single-parameter **temperature-scaling** fix (T ≈ 1.26) drops XGBoost's ECE to ~0.007 — below TabNet's — without changing its predictions. TabNet's apparent edge came from being underconfident, not better-calibrated.

Result artifacts backing every number above live in `reports/*.json`; figures (confusion matrices, reliability diagrams, ROC curves, feature importance, attention masks) are in `reports/figures/`.

## How to run

The full pipeline lives in one notebook: `tool/weather.ipynb`. An already-executed copy with all outputs is checked in as `tool/weather.executed.ipynb`.

```bash
# from the repo root
python3 -m venv .venv && source .venv/bin/activate
pip install pandas numpy scikit-learn xgboost pytorch-tabnet torch matplotlib jupyter

# open and run top-to-bottom
jupyter notebook tool/weather.ipynb
```

Notes:
- The notebook reads the data via relative paths (`../data/...`), so run it from the `tool/` directory (Jupyter does this automatically when you open it there).
- The data files ship with the repo, so no download is needed; to refresh, pull the 2023/2024 storm-details files from the NOAA link above.
- Running all cells regenerates the metrics in `reports/*.json`. TabNet training is the slow part and benefits from a GPU but runs on CPU.

## Repository structure

```
data/      NOAA Storm Events CSVs for 2023 and 2024 (raw input)
tool/      Pipeline and analysis
  weather.ipynb            main notebook (load → features → models → eval)
  weather.executed.ipynb   same notebook with outputs saved
  _calibration_audit.py    calibration / proper-scoring audit
  _make_class_distribution_fig.py, _run_*.py   figure + step helper scripts
reports/   Results and write-up
  *.json                   metrics, per-class scores, statistical tests, calibration
  *.npy                    saved predictions, probabilities, TabNet attention masks
  figures/                 generated plots
  framing.md, paper_writing_plan.md   analysis framing and write-up planning
  week*-updates.pdf        weekly progress reports
```

## Limitations and honest notes

- **Scope.** This is academic coursework (CISC 484, Machine Learning) and a **team project** — work and commits are shared across the team. It is an analysis study, not a deployable system.
- **Negative result, by design.** The interesting finding is that, in this low-feature / high-volume tabular regime, TabNet's attention offers no demonstrable advantage over a well-tuned tree model. With only 8 informative features, gradient boosting already captures the structure that TabNet's feature-selection mechanism is meant to find.
- **What's next.** The moderate error disagreement suggests an XGBoost+TabNet ensemble is worth trying; richer feature engineering (more geographic/temporal context) and a multi-year temporal split would test the distribution-shift hypothesis more seriously than a single 2023→2024 fold.
