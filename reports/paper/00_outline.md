# Outline — Final Paper

Target: 10 pages, `\documentclass[11pt]{article}` (spec range 8–12). At ~500
words/page that is ~5,000 words total. Section word budgets below sum to
~4,900 to leave headroom for tables, equations, and figure floats.

Framing reference: `reports/framing.md` (Option A′, audit-strengthened
negative result with temporal-split qualification). Every section's claim is
checked against framing.md in the "Consistency" line.

Conventions: "we" + past tense, American spelling. Authors: Nathaniel Black,
Dhruv Patel, Marcos Diaz Vazquez.

---

## §1 Abstract — `01_abstract.md`

- **Scope.** Single paragraph, ≤200 words. State the pre-registered
  hypothesis, the random-split falsification, the calibration audit, and the
  temporal per-class qualification — in that order.
- **Numbers cited.** XGB macro-F1 0.9001, TabNet macro-F1 0.8513
  (`week4_results.json: random_split` and `tabnet_refined_tuning`);
  bootstrap diff +0.049 (`statistical_tests.json: bootstrap.diff_mean`);
  apparent ECE gap and the temp-scaling reversal
  (`calibration_audit.json: temperature_scaling`).
- **Figures.** None.
- **Budget.** ~180 words.
- **Consistency.** Headline = random-split negative result + audit. The
  temporal qualification gets at most one clause; it is not promoted to the
  headline.

## §2 Introduction — `02_introduction.md`

- **Scope.** Motivate the question (do attention-based tabular DL methods
  beat gradient-boosted baselines on a real, high-volume, low-feature
  classification task?). Name the pre-registered hypothesis from the
  proposal. List three contributions: (i) honest test of TabNet vs XGBoost
  on NOAA Storm Events with statistical testing, (ii) calibration audit
  showing the apparent ECE advantage does not survive proper scoring rules
  and temperature scaling, (iii) a temporal-split qualification on the
  proposal's named boundary classes.
- **Numbers cited.** Sparing — headline macro-F1 numbers and the bootstrap
  CI. Defer detail to §6/§7.
- **Figures.** None (or one teaser figure pulled from §6 if space allows).
- **Budget.** ~500 words.
- **Consistency.** Frame the negative result as the central finding;
  temporal qualification mentioned as a third contribution but not as a
  vindication of the proposal.

## §3 Related Work — `03_related_work.md`

- **Scope.** Five threads in one paragraph each (or compact bullets):
  TabNet and attention-based tabular DL; gradient-boosted baselines
  (XGBoost); the tabular DL vs tree-based debate (Grinsztajn 2022,
  Shwartz-Ziv & Armon 2022); calibration of neural classifiers
  (Guo 2017); severe-weather classification context (1–2 NOAA / met
  references if found, else omit).
- **Citations.** TabNet (Arik & Pfister 2021), XGBoost (Chen & Guestrin
  2016), Grinsztajn 2022, Shwartz-Ziv & Armon 2022, Guo 2017. Bibkeys in
  `references.md`.
- **Figures.** None.
- **Budget.** ~250 words.
- **Consistency.** Sets up the audit and the low-feature-regime argument
  without pre-empting them.

## §4 Data — `04_data.md`

- **Scope.** NOAA Storm Events 2023 (and 2024 for temporal split). Raw
  source has 51 columns; we retained 10 severe-weather classes after
  filtering (sample sizes: 117,720 train / 23,544 random-split test;
  55,061 rows in the 2024 temporal hold-out — confirm against notebook).
  Engineered to 8 informative features (named: `CZ_TYPE_ENC`,
  `DURATION_MIN`, `HOUR`, `BEGIN_LAT`, `BEGIN_LON`, `MONTH`, `STATE_ENC`,
  `WFO_ENC`). Class distribution (Thunderstorm Wind dominant ~35%, Hail
  next, then long tail) — show in a small table.
- **Numbers cited.** Row counts (confirm via `feature_cols.json`,
  notebook cells), class supports from `temporal_per_class.json`
  (`random_split.xgb.<class>.support` and
  `temporal_split.xgb.<class>.support`).
- **Figures.** Class distribution bar chart (does any figure exist? — if
  not, defer or build during §6 stage; do not invent).
- **Budget.** ~600 words.
- **Consistency.** §2.2 footnote: the raw 51-column count meets the
  spec's "high-volume tabular" criterion at the source level; the
  engineered 8-feature working set is what the models actually consume.

## §5 Methods — `05_methods.md`

- **Scope.** Three subsections.
  1. **Baseline.** XGBoost configuration: tuned hyperparameters from the
     grid (confirm against the notebook cell that ran the grid search).
  2. **TabNet.** Two-stage tuning — a coarse sweep over `(n_da, n_steps,
     lr)` then a refined sweep adding `gamma` and `batch_size`, validated
     with k-fold CV. Final config: `n_da=16, n_steps=7, lr=0.02, gamma=1.0,
     batch_size=1024` (`week4_results.json: tabnet_refined_tuning.best_config`).
  3. **Protocol.** Train/test split, random seed (42), preprocessing
     (encoding scheme for categoricals — confirm from notebook), missing-
     coordinate policy (per `nan_coord_policy` results), evaluation
     metrics (macro-F1, macro AUC, ECE, Brier, NLL), and the temporal
     hold-out construction (2023 → 2024).
- **Numbers cited.** All hyperparameters cross-checked against
  `tool/weather.ipynb`. CV mean/std from
  `week4_results.json: tabnet_refined_tuning.cv_mean_macro_f1` (0.8298 ±
  0.0076).
- **Figures.** None.
- **Budget.** ~700 words.
- **Consistency.** Methods section is descriptive only — no claims.

## §6 Experiments — `06_experiments.md`

- **Scope.** The locked headline table:
  | model | random macro-F1 | temporal macro-F1 | macro AUC |
  | tuned XGBoost | 0.9001 | 0.7556 | 0.9907 |
  | refined TabNet | 0.8513 | 0.7487 | 0.9792 |
  Brief narrative: XGBoost wins on both splits in macro-F1 terms; AUC tracks.
- **Numbers cited.** `week4_results.json: random_split`,
  `temporal_split_2023_to_2024`, `tabnet_refined_tuning.test_macro_f1`,
  `macro_roc_auc`. Note that the headline TabNet number is the *refined*
  result (0.8513), not the coarse-sweep result (0.8231).
- **Figures.** `figures/macro_f1_with_ci.png`,
  `figures/model_comparison_macro_f1.png`,
  `figures/model_comparison_macro_auc.png`,
  `figures/random_vs_temporal_split.png`.
- **Budget.** ~900 words.
- **Consistency.** Headline reads as a clean win for XGBoost on aggregate;
  the per-class temporal nuance is deferred to §9.

## §7 Statistical Analysis — `07_statistical_analysis.md`

- **Scope.** Paired bootstrap on the macro-F1 difference; McNemar on
  per-row correctness. State that both reject the null of equivalence
  decisively.
- **Numbers cited.** `statistical_tests.json: bootstrap.diff_mean =
  +0.0489`, `bootstrap.diff_ci = [0.0444, 0.0536]`, `bootstrap.p_two_sided
  ≈ 0`; `mcnemar.chi2 = 591.6`, `mcnemar.p ≈ 1.1e-130`. Mention 1,000
  paired bootstrap resamples and the continuity-corrected χ².
- **Figures.** `figures/macro_f1_with_ci.png` (re-cite from §6 if
  it appears there; otherwise place here).
- **Budget.** ~350 words.
- **Consistency.** Pure confirmation of the negative result.

## §8 Calibration and Uncertainty (the audit) — `08_uncertainty.md`

- **Scope.** This is the paper's distinctive methodological contribution
  under Framing A′. Structure:
  1. **The apparent finding.** TabNet ECE 0.0043 vs XGBoost 0.0250 on
     the full random-split test set, bootstrap diff CI excludes zero.
  2. **Audit 1 — proper scoring rules.** Brier (XGB 0.179, TabNet 0.248)
     and NLL (XGB 0.312, TabNet 0.408) both favor XGBoost. Argue why this
     matters: Brier and NLL reward sharpness *and* calibration; TabNet's
     low ECE is paid for by under-confidence.
  3. **Audit 2 — temperature scaling.** A single learned T = 1.26 on
     XGBoost reduces ECE from 0.0269 to 0.0074 on the held-out 50% eval
     half — *below* TabNet's 0.0097 on that same half. TabNet's optimal
     T = 1.01 (essentially identity), so it has no temperature-scaling
     headroom. The gap reverses sign: gap_pre +0.021, gap_post −0.002.
  4. **Audit 3 — argmax preserved.** Temperature scaling does not change
     predictions, so the macro-F1 gap survives.
  5. **Synthesis.** Report the apparent ECE win as audited and overturned;
     state explicitly that this is a caution against ECE-only calibration
     claims in the tabular DL literature.
- **Numbers cited.** Full-test ECE values from
  `week4_results.json: uncertainty` and `uncertainty.json`
  (XGB 0.0250, TabNet 0.0043). Bootstrap CI on the diff from
  `calibration_audit.json: bootstrap_ece`. Proper scores from
  `calibration_audit.json: proper_scores`. Temperature scaling block from
  `calibration_audit.json: temperature_scaling`. **Critical labeling:**
  the temperature-scaling numbers are on a 50% eval half, not the full
  test set — label this in prose and in the table caption so the
  TabNet pre_ece = 0.0084 (half) does not appear to contradict the
  full-test 0.0043. Entropy diagnostics
  (`uncertainty.json` mean entropies on correct vs wrong) feed the
  under-confidence interpretation.
- **Figures.** `figures/reliability_diagrams.png`,
  `figures/entropy_correct_vs_incorrect.png`.
- **Budget.** ~900 words. Spend more here than the original Step 8 outline
  suggested; this section carries the contribution claim.
- **Consistency.** Section delivers Framing A′'s second pillar (the
  audit). Do **not** revert to Framing B's "calibration tradeoff" thesis.

## §9 Error Analysis — `09_error_analysis.md`

- **Scope.** Three pieces.
  1. **Confusion matrices** for both models on the random-split test set.
  2. **Cross-model error overlap.** Cohen's κ = 0.47 (moderate). 908 of
     23,544 rows are TabNet-only-correct; 2,283 are XGB-only-correct;
     1,934 both wrong. Interpret: errors are not redundant — there is
     room for an ensemble (deferred to §11 as future work).
  3. **Per-class F1: random vs temporal, side-by-side.** Call out
     explicitly the temporal-split qualification: TabNet beats XGB on
     Hail (+0.037) and Thunderstorm Wind (+0.017) under 2023→2024 shift,
     while losing both on the random split (Hail −0.113, TS Wind −0.058).
     Also report the asymmetric F1 drops: XGB loses −0.207 on Hail and
     −0.120 on TS Wind under the shift; TabNet loses −0.057 and −0.045.
     Tone: substantive qualification, not vindication. One annual fold.
- **Numbers cited.** `week4_results.json: error_overlap`,
  `temporal_per_class.json: random_split.{xgb,tabnet}` and
  `temporal_split.{xgb,tabnet}`.
- **Figures.** `figures/confusion_matrix_tuned_xgb.png`,
  `figures/confusion_matrix_tabnet.png`, `figures/error_overlap.png`,
  `figures/error_overlap_per_class.png`,
  `figures/per_class_f1_random_vs_temporal.png`.
- **Budget.** ~700 words.
- **Consistency.** This is the only section that promotes the temporal
  qualification to first-class status. Tone follows framing.md
  "Temporal-split qualification" verbatim.

## §10 Interpretability — `10_interpretability.md`

- **Scope.** Compare XGBoost gain-based feature importance with TabNet
  step-wise attention masks. Both rank the same small set of features
  (CZ_TYPE_ENC, DURATION_MIN, HOUR, geo, MONTH). Note that the top
  attended features (`tabnet_diagnostics.top_attended_features`) match
  the XGBoost ranking — consistent with the §11 mechanistic claim that
  there is no irrelevant signal for TabNet's attention to filter out.
- **Numbers cited.** `tabnet_diagnostics.top_attended_features` and the
  XGB importance ranks (`xgb_feature_importance.npy`,
  `xgb_feature_cols.json`).
- **Figures.** `figures/feature_importance_tuned_xgb.png`,
  `figures/tabnet_attention_masks.png`,
  `figures/feature_attribution_comparison.png`.
- **Budget.** ~400 words.
- **Consistency.** Sets up the §11 mechanistic argument without making
  the claim there.

## §11 Discussion — `11_discussion.md`

- **Scope.** Three threads, in order:
  1. **Mechanistic explanation of the random-split negative result.**
     With only 8 informative features, gain-based splits already
     enumerate near-optimal partitions; TabNet's value proposition —
     attention-based feature filtering — has nothing to filter. The §10
     evidence (both models concentrate on the same small feature set)
     supports this.
  2. **Why the temporal qualification is consistent with the mechanism.**
     TabNet's softer, more regularized representations degrade more
     gracefully on the boundary classes under shift; XGBoost's tight
     2023 fit hurts it more on 2024. State as hypothesis-of-mechanism,
     not proven claim — one annual fold.
  3. **Cross-model error independence and the ensemble follow-up.**
     κ = 0.47 suggests an ensemble could help; flagged as future work
     only.
- **Numbers cited.** Top-attended-features overlap, κ.
- **Figures.** None (re-cite §10/§9 figures by reference).
- **Budget.** ~500 words.
- **Consistency.** Discussion uses exactly the three-thread order
  framing.md prescribes.

## §12 Limitations — `12_limitations.md`

- **Scope.** Four bullets:
  1. **Feature count.** 8 engineered features is a low-feature regime;
     results may not generalize to wider tabular tasks.
  2. **Single-year temporal fold.** 2023 → 2024 is one annual shift;
     the per-class temporal-shift result on Hail / TS Wind is suggestive,
     not robust.
  3. **No ensemble baseline.** We argue κ = 0.47 motivates ensembling but
     did not evaluate one.
  4. **ECE binning sensitivity.** ECE depends on bin count and width;
     we used 10 equal-width bins. Brier and NLL (used in the audit)
     mitigate this.
- **Numbers cited.** None new.
- **Figures.** None.
- **Budget.** ~250 words.
- **Consistency.** Limitations match the "Out of scope" section of
  framing.md.

## §13 Conclusion — `13_conclusion.md`

- **Scope.** 3–5 sentences. Recap the falsified pre-registered hypothesis,
  the audit, the temporal qualification, and the methodological caution
  about ECE-only calibration claims.
- **Numbers cited.** None.
- **Figures.** None.
- **Budget.** ~120 words.
- **Consistency.** Mirrors the abstract; written before it so the
  abstract can paraphrase the conclusion (a common workflow trick).

## References — `references.md`

- See §3. Five core works minimum; bibkeys carried into LaTeX `references.bib`.

## Figure captions — `figure_captions.md`

- One caption per figure used in the paper, with path, source data, and
  one-sentence takeaway. Written after each section that cites the figure
  is locked.

---

## Section-to-figure map (cross-check)

| Section | Figures |
|---|---|
| §6  | macro_f1_with_ci, model_comparison_macro_f1, model_comparison_macro_auc, random_vs_temporal_split |
| §7  | macro_f1_with_ci (re-cite) |
| §8  | reliability_diagrams, entropy_correct_vs_incorrect |
| §9  | confusion_matrix_tuned_xgb, confusion_matrix_tabnet, error_overlap, error_overlap_per_class, per_class_f1_random_vs_temporal |
| §10 | feature_importance_tuned_xgb, tabnet_attention_masks, feature_attribution_comparison |

Figures present on disk but **not** referenced above (decide later if any
belong in supplementary / appendix material):
`cz_type_ablation_and_priors.png`, `nan_coord_policy.png`,
`roc_curves_tuned_xgb.png`, `roc_curves_tabnet.png`. The ROC plots are
likely worth a half-column in §6 or §8 — flag for review.

---

## Budget total

§1 abstract 180 + §2 intro 500 + §3 RW 250 + §4 data 600 + §5 methods 700 +
§6 exp 900 + §7 stats 350 + §8 audit 900 + §9 errors 700 + §10 interp 400 +
§11 disc 500 + §12 limits 250 + §13 conc 120 ≈ 6,350 words.

That is above a strict 500-words-per-page × 10 pages = 5,000-word target,
which is fine: 11pt article with figures and tables typically runs closer
to 600–650 words per page net of floats. If the rendered PDF goes past
12 pages, the first cuts to make are §4 (data) and §5 (methods), which
can compress to bullet-and-table form, and §10 (interpretability), which
can drop one of its three figures.

---

## Open questions to flag during drafting

1. **Row counts for §4.** Exact train/test sizes — read from the notebook
   cell, not from this outline. The 23,544 random-split test figure is
   confirmed via `error_overlap.n_test`; the 55,061 temporal-test figure
   is confirmed via `temporal_per_class.json: temporal_split.xgb.macro
   avg.support`. Train size and total filtered N still need notebook
   confirmation.
2. **Class-distribution figure.** No `class_distribution.png` exists in
   `reports/figures/`. Either build one as part of §4 or describe in a
   small inline table.
3. **ROC plots.** `roc_curves_tuned_xgb.png` and `roc_curves_tabnet.png`
   exist but are not currently slotted into a section. Either §6 (next to
   the AUC table) or §8 (as a sharpness illustration alongside reliability
   diagrams).
