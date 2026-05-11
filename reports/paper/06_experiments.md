# §6 Experiments

This section reports the locked headline numbers under both splits and
both models. Per-row statistical testing is deferred to §7; per-class and
error-overlap analyses are deferred to §9; calibration and probability
quality are deferred to §8.

## 6.1 Random-split macro-F1

We report four configurations on the random-split test fold (23,544
rows): a default-hyperparameter XGBoost, the tuned XGBoost selected in
§5.1, the baseline TabNet, and the refined TabNet selected in §5.2.
Numbers are from `week4_results.json: random_split` and
`tabnet_refined_tuning.test_macro_f1`.

| model | test macro-F1 |
| --- | ---: |
| default XGBoost | 0.8887 |
| **tuned XGBoost** | **0.9001** |
| baseline TabNet | 0.7828 |
| TabNet (coarse-sweep best) | 0.8231 |
| **refined TabNet** | **0.8513** |

The two bolded rows are the configurations we treat as the final models;
all subsequent analyses use them. Two observations.

First, tuning helped both architectures, but it helped TabNet more in
absolute terms (+0.0685 macro-F1 from baseline to refined) than it helped
XGBoost (+0.0114 from default to tuned). That is consistent with TabNet's
greater hyperparameter surface: the refined sweep alone moved it from
0.8231 to 0.8513 by varying only `gamma` and `batch_size` at fixed
`(n_d, n_steps, lr)`. The baseline XGBoost is already close to its
tuned ceiling.

Second, the tuned XGBoost outperforms the refined TabNet by 0.0488
macro-F1 (`week4_results.json: tabnet_refined_tuning.delta_vs_xgboost`).
This is the pre-registered comparison: the proposal predicted that
TabNet's attention-based feature selection would outperform a gradient-
boosted baseline on this task. On the random split, by the macro-F1
metric, the prediction is falsified — and §7 will show the falsification
is statistically robust.

## 6.2 Discrimination by ROC-AUC

Macro one-versus-rest ROC-AUC tracks the macro-F1 ordering but with a
narrower spread (`week4_results.json: macro_roc_auc` and
`tabnet_diagnostics.roc_auc_ovr_macro`).

| model | macro OvR AUC |
| --- | ---: |
| tuned XGBoost | 0.9907 |
| refined TabNet | 0.9831 |
| baseline TabNet | 0.9760 |
| default XGBoost | 0.9882 |

Both models are highly discriminating in absolute terms — AUC near 0.99
for the tuned XGBoost — and the AUC gap (0.0076) is smaller than the
macro-F1 gap (0.0488). This is unsurprising. AUC scores the model on its
*ranking* of probabilities rather than its hard-labeling under a
0.5-style decision rule, and TabNet's softer, less-confident predictions
(visible in the calibration audit of §8) rank classes correctly more
often than they confidently assign the top-ranked class. The §6.1 table
weighs the latter; the §6.2 table weighs the former. The macro-F1 result
is the one the rubric and the proposal commit to, and it is the one we
treat as the headline finding.

Per-class ROC curves for both models are shown in Figure&nbsp;3 and
Figure&nbsp;4. The class with the lowest one-vs-rest AUC under either
model is Hail (XGB AUC for Hail is recoverable from the per-class
breakdown; TabNet's is 0.927 per `tabnet_diagnostics.roc_auc_ovr_per_class`).
Hail and Thunderstorm Wind are the two classes the proposal predicted
TabNet would excel at; §9 returns to them.

## 6.3 Temporal generalization, 2023 → 2024

We re-trained both models on the 2023 archive only (62,656 rows) and
evaluated on the 2024 archive (55,061 rows), holding all hyperparameters
fixed at the values selected in §5 (with the §5.3 caveat that TabNet on
the temporal split uses the coarse-sweep configuration, not the refined
one). Numbers are from `week4_results.json: temporal_split_2023_to_2024`.

| model | random macro-F1 | temporal macro-F1 | Δ (temporal − random) |
| --- | ---: | ---: | ---: |
| tuned XGBoost | 0.9001 | 0.7556 | −0.1445 |
| TabNet (coarse temporal config) | 0.8231 | 0.7487 | −0.0744 |

The temporal split is harder for both models. The aggregate gap between
the two narrows substantially: from 0.0488 macro-F1 on the random split
(refined TabNet vs tuned XGBoost; using the matched coarse-config TabNet
the gap on the random split is 0.0770) to 0.0069 on the 2023→2024 split.
On the aggregate metric XGBoost still wins, but only just; the narrowing
is driven by XGBoost dropping further under shift than TabNet does. We
do not promote this aggregate narrowing to a thesis — the per-class
breakdown in §9 shows the narrowing concentrates in two specific
classes, and one annual fold is not enough to make a domain-adaptation
claim. But the contrast between the two ∆-columns above (−0.1445 vs
−0.0744) is itself worth flagging: TabNet's softer fit appears to
generalize more gracefully across years on this dataset, and the §11
discussion takes that as a hypothesis-of-mechanism consistent with the
§8 calibration audit.

## 6.4 Figures

- **Figure&nbsp;2.** `reports/figures/macro_f1_with_ci.png` — tuned
  XGBoost vs refined TabNet macro-F1 with 95% bootstrap confidence
  intervals on the random-split test fold. Anchors §7.
- **Figure&nbsp;3.** `reports/figures/model_comparison_macro_f1.png` —
  bar chart of the four random-split configurations in §6.1.
- **Figure&nbsp;4.** `reports/figures/model_comparison_macro_auc.png` —
  bar chart of the four random-split macro AUC values in §6.2.
- **Figure&nbsp;5.** `reports/figures/random_vs_temporal_split.png` —
  random vs 2023→2024 macro-F1 for both finalized models. Anchors §6.3
  and §9.
- **Figure&nbsp;6.** `reports/figures/roc_curves_tuned_xgb.png` and
  **Figure&nbsp;7.** `reports/figures/roc_curves_tabnet.png` — per-class
  one-vs-rest ROC curves for the tuned XGBoost and the refined TabNet.

(Figure numbering in this markdown is local to §6; the final figure
sequence across the paper is fixed during the LaTeX pass.)
