# Figure Captions

One caption per figure used in the paper, with the path to the
source image, the data artifact it is built from, and a one-sentence
takeaway. Figure numbering below is the global paper-wide sequence;
the markdown section drafts use section-local numbering that the
LaTeX pass will replace with these globals.

---

## Figure 1 — Class distribution

**Path:** `reports/figures/class_distribution.png`
**Source data:** per-class row counts on the random-split test fold
(`temporal_per_class.json: random_split.xgb.<class>.support`,
reproducible from `tool/_make_class_distribution_fig.py`).
**Cited in:** §4.
**Caption.** Class distribution on the 23,544-row random-split test
fold. Thunderstorm Wind is the modal class at ~35% of test rows;
Hail is second at ~18%. The remaining eight classes share the long
tail, with Marine Thunderstorm Wind, Heat, Excessive Heat, High Wind,
Flash Flood, Drought, Winter Weather, and Flood each between 4% and
8%.

---

## Figure 2 — Macro-F1 with 95% bootstrap CI

**Path:** `reports/figures/macro_f1_with_ci.png`
**Source data:** `statistical_tests.json: bootstrap` over 1,000
paired resamples of the random-split test fold.
**Cited in:** §6.4, §7.
**Caption.** Macro-F1 on the random-split test fold for the tuned
XGBoost and the refined TabNet, with 95% paired-bootstrap confidence
intervals. The intervals are disjoint at every resample; the bootstrap
difference is +0.0489 in XGBoost's favor with 95% CI [+0.0444,
+0.0536].

---

## Figure 3 — Macro-F1, four-model comparison

**Path:** `reports/figures/model_comparison_macro_f1.png`
**Source data:** `week4_results.json: random_split` and
`tabnet_refined_tuning.test_macro_f1`.
**Cited in:** §6.4.
**Caption.** Random-split macro-F1 for the four configurations: the
default and tuned XGBoost (0.8887, 0.9001) and the baseline and
refined TabNet (0.7828, 0.8513). Tuning helped TabNet more in
absolute terms (+0.069) than XGBoost (+0.011), but the refined
TabNet still trails the tuned XGBoost by 0.049 macro-F1.

---

## Figure 4 — Macro one-vs-rest AUC, four-model comparison

**Path:** `reports/figures/model_comparison_macro_auc.png`
**Source data:** `week4_results.json: macro_roc_auc` and
`tabnet_diagnostics.roc_auc_ovr_macro`.
**Cited in:** §6.4.
**Caption.** Random-split macro one-vs-rest AUC for the four
configurations. All four models are highly discriminating in
absolute terms (AUC ≥ 0.97); the tuned-XGBoost vs refined-TabNet
gap is 0.0076, narrower than the 0.0488 macro-F1 gap.

---

## Figure 5 — Random vs 2023→2024 split, macro-F1

**Path:** `reports/figures/random_vs_temporal_split.png`
**Source data:** `week4_results.json: random_split` and
`temporal_split_2023_to_2024`.
**Cited in:** §6.4, §9.4.
**Caption.** Macro-F1 under the random split and the 2023→2024
temporal hold-out for the tuned XGBoost (0.9001 → 0.7556; Δ
−0.1445) and the matched-config coarse TabNet (0.8231 → 0.7487; Δ
−0.0744). The temporal gap between the two models (0.0069)
narrows substantially relative to the random-split gap (0.0488 to
the refined TabNet, 0.0770 to the matched-config TabNet) because
XGBoost drops further under shift.

---

## Figure 6 — Per-class ROC, tuned XGBoost

**Path:** `reports/figures/roc_curves_tuned_xgb.png`
**Source data:** tuned-XGBoost predicted probabilities on the
random-split test fold (`reports/xgb_test_proba.npy`).
**Cited in:** §6.4.
**Caption.** Per-class one-vs-rest ROC curves for the tuned XGBoost
on the random-split test fold. Hail has the lowest one-vs-rest AUC
of any class; Marine Thunderstorm Wind sits at AUC ≈ 1.

---

## Figure 7 — Per-class ROC, refined TabNet

**Path:** `reports/figures/roc_curves_tabnet.png`
**Source data:** refined-TabNet predicted probabilities on the
random-split test fold (`reports/tabnet_test_proba.npy`).
**Cited in:** §6.4.
**Caption.** Per-class one-vs-rest ROC curves for the refined
TabNet on the random-split test fold. Hail's AUC is 0.927
(`tabnet_diagnostics.roc_auc_ovr_per_class`), the lowest of the
ten classes.

---

## Figure 8 — Reliability diagrams

**Path:** `reports/figures/reliability_diagrams.png`
**Source data:** `reports/uncertainty.json` and
`calibration_audit.json` (ten equal-width bins per the Guo et al.
2017 convention).
**Cited in:** §8.6.
**Caption.** Reliability diagrams on the full random-split test fold,
before temperature scaling. XGBoost's curve is noisier, with its
largest deviation a downward dip in the 0.45-confidence bin
(empirical accuracy ≈ 0.37); TabNet's curve tracks the diagonal
closely from its lowest populated bin onward. The diagram does not
directly visualize the under-confidence story — that interpretation
depends on the proper-score-rule numbers of §8.2 and the entropy
diagnostics of §8.5.

---

## Figure 9 — Entropy of predicted distribution, correct vs incorrect

**Path:** `reports/figures/entropy_correct_vs_incorrect.png`
**Source data:** per-row entropy of predicted distributions
(`reports/uncertainty.json`).
**Cited in:** §8.6.
**Caption.** Predicted-distribution entropy histograms for both
models, stratified by whether the top-1 prediction matched the
label. TabNet's correct-prediction histogram is shifted right
(mean H = 0.352 nats vs XGBoost's 0.192), confirming that TabNet
produces softer distributions even when it gets the answer right.

---

## Figure 10 — Confusion matrix, tuned XGBoost

**Path:** `reports/figures/confusion_matrix_tuned_xgb.png`
**Source data:** tuned-XGBoost predictions on the random-split test
fold (`reports/xgb_test_preds.npy`).
**Cited in:** §9.4.
**Caption.** Row-normalized confusion matrix for the tuned XGBoost
on the random-split test fold. The dominant off-diagonal mass is
Hail → Thunderstorm Wind. Marine Thunderstorm Wind is classified
perfectly.

---

## Figure 11 — Confusion matrix, refined TabNet

**Path:** `reports/figures/confusion_matrix_tabnet.png`
**Source data:** refined-TabNet predictions on the random-split test
fold (`reports/tabnet_test_preds.npy`).
**Cited in:** §9.4.
**Caption.** Row-normalized confusion matrix for the refined TabNet
on the random-split test fold. Hail → Thunderstorm Wind confusion
is larger than under XGBoost (Hail recall 0.630 vs 0.737); Heat ↔
Excessive Heat confusion is also more pronounced, with TabNet
trading Excessive Heat recall (0.675) for Heat recall (0.948).

---

## Figure 12 — Cross-model error overlap

**Path:** `reports/figures/error_overlap.png`
**Source data:** `week4_results.json: error_overlap.table`.
**Cited in:** §9.4.
**Caption.** Per-row correctness agreement table for the two
finalized models on the 23,544-row random-split test fold:
18,419 both-correct, 2,283 XGB-only-correct, 908 TabNet-only-correct,
1,934 both-wrong. Cohen's κ = 0.47.

---

## Figure 13 — Cross-model error overlap, per class

**Path:** `reports/figures/error_overlap_per_class.png`
**Source data:** `week4_results.json: error_overlap.per_class`.
**Cited in:** §9.4.
**Caption.** Per-class breakdown of the four agreement cells. Hail
has the largest both-wrong cell (790 rows) and the largest XGB-only-
correct cell (743). Heat is the only class where TabNet-only-correct
(140) substantially exceeds XGB-only-correct (29). Marine Thunderstorm
Wind has zero off-diagonal mass under either model.

---

## Figure 14 — Per-class F1, random vs 2023→2024

**Path:** `reports/figures/per_class_f1_random_vs_temporal.png`
**Source data:** `temporal_per_class.json`.
**Cited in:** §9.4.
**Caption.** Per-class F1 under the random split and the 2023→2024
temporal hold-out for both finalized models. TabNet loses on every
class on the random split. On the temporal hold-out, TabNet flips
sign on Hail (+0.037), Thunderstorm Wind (+0.017), and Heat
(+0.065), with random→temporal degradation 3–4× smaller than
XGBoost's on the two proposal-named boundary classes.

---

## Figure 15 — XGBoost gain-based feature importance

**Path:** `reports/figures/feature_importance_tuned_xgb.png`
**Source data:** `reports/xgb_feature_importance.npy` indexed by
`reports/xgb_feature_cols.json`.
**Cited in:** §10.4.
**Caption.** Tuned-XGBoost split-gain importance over the 8
engineered features. CZ_TYPE_ENC alone accounts for 90.4% of total
gain; DURATION_MIN takes 3.6%; the remaining six features share the
final 6%.

---

## Figure 16 — TabNet per-step attention masks

**Path:** `reports/figures/tabnet_attention_masks.png`
**Source data:** `reports/tabnet_masks.npy` (7 steps × 23,544 rows ×
8 features), averaged over rows.
**Cited in:** §10.4.
**Caption.** Heatmap of mean attention mask weight per (decision
step, feature) for the refined TabNet on the random-split test fold.
Each step specializes on a different feature subset — step 5 places
62% of its mask weight on DURATION_MIN; step 2 on BEGIN_LAT/LON; step
4 on HOUR. CZ_TYPE_ENC is in the top-4 of every step but never
dominates as exclusively as it does for XGBoost.

---

## Figure 17 — Feature attribution, XGBoost gain vs TabNet explain

**Path:** `reports/figures/feature_attribution_comparison.png`
**Source data:** `reports/xgb_feature_importance.npy` and
`reports/tabnet_explain_matrix.npy`.
**Cited in:** §10.4.
**Caption.** Side-by-side comparison of XGBoost gain importance and
TabNet local-attribution (mean |explain|) on the common 8-feature
axis. Both rank CZ_TYPE_ENC and DURATION_MIN at the top and WFO_ENC
at the bottom; the middle ordering disagrees, with XGBoost routing
location through STATE_ENC and TabNet routing it through raw
BEGIN_LAT / BEGIN_LON.
