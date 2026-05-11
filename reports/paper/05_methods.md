# §5 Methods

## 5.1 Baseline: tuned XGBoost

We used XGBoost 3.2 with the `hist` tree method and a `multi:softmax`
objective over the ten retained classes (`tool/weather.ipynb`,
data-loading and grid-search cells). Hyperparameters were chosen by a
five-fold stratified grid search over a 192-candidate cube:

| hyperparameter | grid |
| --- | --- |
| `learning_rate` | 0.01, 0.05, 0.1, 0.3 |
| `max_depth` | 3, 5, 7, 9 |
| `n_estimators` | 100, 300, 500, 700 |
| `subsample` | 0.7, 0.8, 1.0 |

Cross-validation used `StratifiedKFold(n_splits=5, shuffle=True,
random_state=42)` with `scoring='f1_macro'`. The selected configuration
was `learning_rate=0.1, max_depth=9, n_estimators=500, subsample=0.7`,
with cross-validated macro-F1 of 0.8949. We refit the selected
configuration on the full random-split training fold (94,173 rows)
before evaluating on the 23,544-row test fold. The same configuration
is used unchanged on the temporal split's 2023 train fold.

## 5.2 TabNet

We used `pytorch_tabnet 4.x` (Arik & Pfister 2021). Two TabNet
configurations appear in the paper, distinguished by which hyperparameter
sweep produced them.

**Baseline TabNet.** Out-of-the-box defaults — `n_d = n_a = 8`,
`n_steps = 3`, `gamma = 1.3`, `lr = 0.02`, `batch_size = 1024`, Adam with
StepLR scheduling (`step_size = 50`, scheduler `gamma = 0.9`), 100 epochs
with early stopping `patience = 15` on validation accuracy. The two
categorical columns `STATE_ENC` and `WFO_ENC` are passed via
`cat_idxs` / `cat_dims` with `cat_emb_dim = 1`. This is the model that
posts the random-split macro-F1 of 0.7828 reported as the TabNet baseline.

**Refined TabNet.** Tuning proceeded in two stages.

1. *Coarse sweep.* A 27-config grid over `n_d = n_a ∈ {8, 16, 32}`,
   `n_steps ∈ {3, 5, 7}`, and `lr ∈ {0.01, 0.02, 0.05}`, scored on a
   single held-out validation split carved from the training fold. The
   notebook is explicit about the choice not to use k-fold here, citing
   runtime. The best coarse configuration was
   `n_d = n_a = 16, n_steps = 7, lr = 0.02` with val macro-F1 0.8358
   (`week4_results.json: tabnet_sweep_best`).
2. *Refined sweep with k-fold CV.* Holding the coarse winner fixed, a
   second 12-config grid varied `gamma ∈ {1.0, 1.3, 1.5, 1.8}` and
   `batch_size ∈ {512, 1024, 2048}`. Each configuration was re-scored
   under `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)`
   on the full training fold. The winner was `gamma = 1.0,
   batch_size = 1024`, with cross-validated macro-F1 of 0.8298 ± 0.0076
   (`week4_results.json: tabnet_refined_tuning`).

The refined TabNet was then refit on the full training fold with the
selected configuration (`n_d = n_a = 16, n_steps = 7, lr = 0.02,
gamma = 1.0, batch_size = 1024`, all other settings unchanged from the
baseline) and evaluated on the random-split test fold, where it scores
macro-F1 of 0.8513. This is the TabNet model used in every random-split
result in §6 through §10.

## 5.3 Protocol differences between splits

We used the tuned XGBoost configuration unchanged on both splits. For
TabNet, the random-split results use the *refined* configuration
(§5.2 stage 2). The temporal-split results use the coarse-sweep
configuration of §5.2 stage 1 (`n_d = n_a = 16, n_steps = 7, lr = 0.02,
gamma = 1.3, batch_size = 1024`). The refined `gamma = 1.0` finding
was locked after the temporal-split fit had already been run, and we
chose not to re-run the temporal-split training to keep the
temporal-split TabNet matched to its pre-registered architecture.
Section §11 discusses how this caveat affects the temporal per-class
qualification: any per-class advantage TabNet shows under shift is a
lower bound on what a re-tuned TabNet might deliver, not an upper bound.

The temporal split also required a defensive re-encoding of the target
because the LabelEncoder fit on the combined dataset cannot be assumed
to produce a contiguous label space when restricted to 2023 alone. The
notebook asserts that 2024 contains no classes absent from 2023 and
otherwise transforms train and test through a fresh encoder fit on the
2023 train labels (`tool/weather.ipynb` temporal-split cell, lines
~3090–3120). All ten classes appeared in both years, so the encoder is
order-preserving in practice.

## 5.4 Reproducibility

All random-seeded operations use `random_state = 42`: the random-split
`train_test_split`, the XGBoost grid-search CV, the refined-sweep CV,
and the XGBoost classifier itself. TabNet does not seed PyTorch globally
in our notebook, so TabNet runs are reproducible up to GPU non-determinism
(cuDNN convolution algorithm selection and floating-point reduction
order). We did not seed PyTorch because the random-split TabNet
macro-F1 was reproduced within rounding across the coarse sweep
(val 0.8358), the standalone refit (test 0.8231 — the tuned-but-not-refined
TabNet), and the refined refit (test 0.8513). All test predictions and
probabilities have been persisted to disk
(`reports/{tabnet,xgb}_test_{preds,proba}.npy` and the
temporal-split counterparts) so every downstream analysis in §7–§10
runs from frozen artifacts and is fully reproducible on CPU without
re-fitting on CUDA.

## 5.5 Evaluation metrics

We report macro-F1 as the primary metric (so the long-tail classes are
not drowned out by Thunderstorm Wind; cf. §4), macro one-versus-rest
ROC-AUC as a secondary discrimination metric, and Expected Calibration
Error, Brier score, and negative log-likelihood for the probabilistic
predictions (definitions are standard; ECE uses ten equal-width bins on
the maximum-class confidence per the `note` field of
`week4_results.json: uncertainty`). Statistical comparisons in §7 use a
paired bootstrap on the macro-F1 difference (1,000 resamples, paired by
test-row index) and a continuity-corrected McNemar test on per-row
correctness.
