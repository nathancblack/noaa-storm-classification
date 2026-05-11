# Final Push: Task Plan

This document is the single source of truth for finishing the CISC 484/684 project. It is written so a fresh context window can be told "we are on Step N" and know exactly what to do, why, and what "done" looks like.

## How to use this document

- Tasks are ordered by priority and (mostly) by dependency. Do them in order unless a step explicitly says it can run in parallel.
- Each task has: **Goal**, **Why it matters** (rubric link), **Inputs**, **Steps**, **Outputs/acceptance criteria**.
- When a step is finished, edit this file: change `Status: TODO` to `Status: DONE` and add a one-line note with the artifact path or commit. Do **not** delete completed tasks — future steps reference earlier outputs.
- The headline numbers, file layout, and rubric this plan targets are summarized in the "Reference" section at the bottom.

## Status board

| # | Task | Status | Notes |
|---|------|--------|-------|
| 0 | Fix `tabnet_recovered_pct` bug in `week4_results.json` | DONE | bug was wrong var (`macro_f1_tabnet` baseline, should be `macro_f1_tabnet_best` tuned); fixed `tool/weather.ipynb` cell `eb9b6f0d` + `90aa206b`; JSON now reads 0.9304 with explanatory `note` field |
| 1 | Decide feature-count framing (low-feature regime vs 27-feature headline) | DONE | Option A′ (audit-strengthened negative result) — calibration audit (Brier/NLL + temperature scaling) showed Framing B's calibration claim was artifactual; thesis is now a robust negative result. Audit results in `reports/calibration_audit.json`; updated thesis at `reports/framing.md` |
| 2 | Persist TabNet test predictions + probs to disk (CUDA run) | DONE | 81-min CUDA run completed; round-trip macro-F1 reproduces locked numbers (TabNet 0.8513, XGB 0.9001) |
| 3 | Statistical testing cell (paired bootstrap + McNemar) | DONE | Diff +0.0489 CI [+0.0444,+0.0536], boot p≈0; McNemar χ²=591.6 p=1.1e-130 |
| 4 | Uncertainty quantification cell (reliability, ECE, entropy) | DONE | TabNet ECE=0.0043 vs XGB ECE=0.0250 — TabNet better calibrated; entropy correlates with errors for both |
| 5 | TabNet-specific figures (confusion matrix, ROC, attention masks) | DONE | CM, ROC, attention + bonus feature-attribution comparison plot all saved. Macro AUC ovr 0.9831 |
| 6 | Cross-model error-overlap analysis | DONE | Agreement 0.8645, Cohen's κ=0.4718, 908 TabNet-only-correct rows — moderate independence, ensembling note |
| 7 | Per-class precision/recall on temporal split | DONE | Both splits saved. **Temporal qualification:** under 2023→2024, TabNet beats XGB on Hail (0.5910 vs 0.5542) and Thunderstorm Wind (0.7768 vs 0.7596) — partial vindication of proposal's pre-registered prediction under distribution shift; framed as §7/§10 qualification, not headline reversal |
| 8 | Scaffold final report (sections + drop in figures) | TODO | |
| 9 | Write final report (8–12 pages) | TODO | |
| 10 | Build 15-minute presentation deck | TODO | |

---

## Step 0 — Fix `tabnet_recovered_pct` bug

**Goal:** The value `cz_type_replacement.tabnet_recovered_pct = 33466001.328803837` in `reports/week4_results.json` is broken (divide-by-near-zero or unit error). It must be correct before any report cites it.

**Why it matters:** Technical Execution (15 pts), reproducibility. A nonsense number in a persisted artifact undermines the whole "locked numbers" claim.

**Inputs:** `reports/week4_results.json`, the cell in `tool/weather.ipynb` that computed the CZ_TYPE-replacement section.

**Steps:**
1. Open `tool/weather.ipynb` and find the cell that writes `cz_type_replacement` to JSON. Search for `tabnet_recovered_pct` or `xgb_recovered_pct`.
2. Inspect the formula. The intended metric is "what fraction of the ablation drop did the priors recover?" i.e. `(priors - no_cz) / (with_cz - no_cz)` expressed as a percent. The bug is almost certainly that for TabNet, `with_cz - no_cz` is tiny (0.0360), and the numerator is being divided by something near zero, or units are wrong.
3. Replace the formula with a guarded version: if `abs(with_cz - no_cz) < 1e-3`, report `"recovered_pct": None` and a note instead of a number.
4. Re-run only that cell (it does not need TabNet retraining — it just reshapes existing numbers).
5. Commit the JSON change.

**Outputs / acceptance criteria:**
- `tabnet_recovered_pct` is either a sensible percent (between -200 and 200) or `null` with a `note` field.
- `xgb_recovered_pct` value (-9.85) is sanity-checked too — it's currently negative because priors *hurt* XGBoost, so a negative recovered-pct is correct, just confusing. Add a one-line note in the JSON or report explaining negative = "priors made it worse."

---

## Step 1 — Decide feature-count framing

**Goal:** Pick one of two framings and document it in a short paragraph that will be reused verbatim in the report's introduction.

**Why it matters:** Scientific Rigor (15 pts) + data requirements. The spec's "≥20 features" bar is an OR'd criterion; the headline 8-feature config does not strictly meet it. Either we own this as a research-question choice, or we promote a different config to the headline.

**Decision options:**
- **Option A (recommended): Low-feature framing.** Position the 8-feature config as the headline and add to the hypothesis: "We further investigate whether TabNet's attention-based feature selection provides advantage in the *low-feature tabular regime*, where there are fewer features for attention to discriminate among." The negative result (TabNet underperforms XGBoost) becomes a *finding*, not a failure.
- **Option B: 27-feature headline.** Promote the CZ_TYPE-replacement-priors variant to the headline. Pro: meets ≥20 features. Con: TabNet barely closes the gap there too, and adding target-derived priors makes the feature engineering story harder to defend (overfitting risk per Week 4 obstacle).

**Recommendation:** Option A. The data set already has 145K instances which independently satisfies the high-volume tabular instance count; framing the feature count as a deliberate choice is defensible.

**Steps:**
1. Pick A or B. Update this file with the decision.
2. Draft a 4–6 sentence paragraph stating the framing. Save it as `reports/framing.md`.
3. The framing paragraph must answer: (a) why this feature count, (b) what the hypothesis becomes under this framing, (c) how the negative TabNet result is interpreted.

**Outputs / acceptance criteria:**
- `reports/framing.md` exists with the chosen framing paragraph.
- This task's row in the status board is marked DONE with `A` or `B` recorded.

---

## Step 2 — Persist TabNet test predictions and probabilities

**Goal:** Save TabNet's test-set predictions and predicted-probability matrices to disk so all downstream analyses (steps 3–7) can run on a CPU without re-running TabNet on CUDA.

**Why it matters:** Unblocks four downstream tasks. Currently TabNet is CUDA-only and re-fits cost ~24 fits per full notebook run.

**Inputs:** Access to a CUDA machine; the existing notebook with the refined TabNet config (n_d=n_a=16, n_steps=7, lr=0.02, gamma=1.0, batch_size=1024).

**Steps:**
1. On the CUDA machine, open `tool/weather.ipynb`.
2. Locate the cell that fits the *refined* TabNet on the full training set and evaluates on the held-out test set (test macro F1 = 0.8513).
3. Immediately after the `.predict()` and `.predict_proba()` calls, add code that saves:
   - `reports/tabnet_test_preds.npy` — integer class predictions, shape `(n_test,)`
   - `reports/tabnet_test_proba.npy` — float probability matrix, shape `(n_test, n_classes)`
   - `reports/tabnet_test_y_true.npy` — ground-truth integer labels, shape `(n_test,)`
   - `reports/tabnet_class_labels.json` — list mapping integer label → class name (Thunderstorm Wind, Hail, …)
   - `reports/tabnet_test_index.npy` — the original DataFrame index for each test row (so XGBoost predictions on the *same* test rows can be aligned in step 6)
4. While on the CUDA machine, *also* save the equivalents for tuned XGBoost: `xgb_test_preds.npy`, `xgb_test_proba.npy`. The truth labels and index are shared.
5. While there, save TabNet's feature-attention masks (`model.explain(X_test)` returns `(M_explain, masks)`): `reports/tabnet_masks.npy`. This is needed for step 5.
6. Commit the new files.

**Outputs / acceptance criteria:**
- All listed `.npy` and `.json` files exist under `reports/`.
- A small CPU-side sanity check: `np.load('reports/tabnet_test_preds.npy').shape` matches `tabnet_test_y_true.npy.shape`.
- Macro F1 recomputed from the saved files (`f1_score(y_true, y_pred, average='macro')`) reproduces 0.8513 ± 1e-4.

---

## Step 3 — Statistical testing

**Goal:** Add a notebook section that produces (a) 95% confidence intervals on each model's test macro-F1 via paired bootstrap, and (b) a McNemar's test p-value for tuned XGBoost vs refined TabNet.

**Why it matters:** Scientific Rigor (15 pts) explicitly calls out "statistical validity." Currently every comparison is a raw delta with no significance signal.

**Depends on:** Step 2.

**Steps:**
1. Add a new section to `tool/weather.ipynb` titled "Statistical Testing".
2. Load `xgb_test_preds.npy`, `tabnet_test_preds.npy`, `*_y_true.npy`.
3. **Paired bootstrap (1000 resamples):**
   - For `b` in `range(1000)`: sample test indices with replacement; compute macro-F1 for each model on the resample; store both, and store `f1_xgb - f1_tabnet`.
   - Report: each model's macro-F1 mean and 2.5/97.5 percentile CI; the percentile CI on the *paired* difference; the bootstrap p-value as `2 * min(P(diff>=0), P(diff<=0))`.
   - Use a fixed RNG seed (e.g., `np.random.default_rng(42)`).
4. **McNemar's test:**
   - Build the 2×2 contingency table over test rows: (XGB correct, TabNet correct), (XGB correct, TabNet wrong), (XGB wrong, TabNet correct), (XGB wrong, TabNet wrong).
   - Use `statsmodels.stats.contingency_tables.mcnemar(table, exact=False, correction=True)` (mid-p with continuity correction).
   - Report the chi-square statistic and p-value.
5. Persist results into `reports/week4_results.json` under a new key `statistical_tests` with sub-keys `bootstrap_xgb_ci`, `bootstrap_tabnet_ci`, `bootstrap_diff_ci`, `bootstrap_p`, `mcnemar_chi2`, `mcnemar_p`, `mcnemar_table`.
6. Save a forest plot (XGB CI, TabNet CI, both side-by-side) to `reports/figures/macro_f1_with_ci.png`.

**Outputs / acceptance criteria:**
- New JSON entry `statistical_tests` with all six fields populated.
- New figure `reports/figures/macro_f1_with_ci.png`.
- A one-sentence written conclusion in a notebook markdown cell stating whether the XGB-vs-TabNet gap is statistically significant at α=0.05.

---

## Step 4 — Uncertainty quantification

**Goal:** Reliability diagrams + Expected Calibration Error (ECE) for both models, plus a predicted-probability entropy histogram split by correct vs incorrect predictions.

**Why it matters:** Analysis Depth (10 pts) explicitly lists "uncertainty quantification" as a Phase 3 deliverable. Currently absent.

**Depends on:** Step 2.

**Steps:**
1. Add a new section to `tool/weather.ipynb` titled "Uncertainty Quantification".
2. Load `xgb_test_proba.npy` and `tabnet_test_proba.npy` (both shape `(n_test, n_classes)`), `*_y_true.npy`.
3. **Reliability diagram (per model):**
   - For each test row, take the predicted class's probability as the model's confidence.
   - Bin into 10 equal-width bins on `[0, 1]`.
   - In each bin compute mean confidence and accuracy.
   - Plot accuracy vs confidence with diagonal reference. One subplot per model.
   - Save to `reports/figures/reliability_diagrams.png`.
4. **Expected Calibration Error:** `ECE = sum_b (|B_b|/N) * |acc(B_b) - conf(B_b)|`. Report for both models.
5. **Entropy histogram:**
   - Compute predicted-probability entropy per test row: `H = -sum p log p` (use `scipy.stats.entropy`).
   - Plot two histograms per model (correct vs incorrect predictions) on the same axes, normalized.
   - Save to `reports/figures/entropy_correct_vs_incorrect.png`.
6. Persist into `week4_results.json` under `uncertainty`: `xgb_ece`, `tabnet_ece`, `xgb_mean_entropy_correct`, `xgb_mean_entropy_wrong`, `tabnet_mean_entropy_correct`, `tabnet_mean_entropy_wrong`.

**Outputs / acceptance criteria:**
- Two new figures under `reports/figures/`.
- Six new JSON values under `uncertainty`.
- One-sentence interpretation in a markdown cell: which model is better calibrated, and whether higher entropy correlates with errors as expected.

---

## Step 5 — TabNet-specific figures

**Goal:** Generate TabNet's own confusion matrix, per-class one-vs-rest ROC curves, and a feature-attention-mask visualization.

**Why it matters:** The proposal explicitly promised attention-mask comparison with XGBoost feature importance. Currently TabNet only appears in aggregate bar charts. Technical Execution + Analysis Depth.

**Depends on:** Step 2.

**Steps:**
1. Add a section "TabNet Diagnostics" to the notebook.
2. **Confusion matrix:** Load `tabnet_test_preds.npy`, `tabnet_test_y_true.npy`, build with `sklearn.metrics.confusion_matrix`, label with class names from `tabnet_class_labels.json`. Save `reports/figures/confusion_matrix_tabnet.png`. Use the same colormap and class ordering as the existing `confusion_matrix_tuned_xgb.png` so the two are visually comparable.
3. **ROC curves:** Load `tabnet_test_proba.npy`, compute one-vs-rest ROC per class with `sklearn.metrics.roc_curve` + `roc_auc_score`. Plot all 10 curves on one axis. Save `reports/figures/roc_curves_tabnet.png` (mirror layout of `roc_curves_tuned_xgb.png`).
4. **Attention masks:** Load `tabnet_masks.npy` (a list with one mask per decision step, each shape `(n_test, n_features)`). For each step, average across test rows to get a per-feature attention vector. Plot a heatmap with rows = decision steps, cols = features. Save `reports/figures/tabnet_attention_masks.png`.
5. **Bonus:** A side-by-side bar chart of XGBoost feature importance vs TabNet's mean attention mask aggregated across steps. Save `reports/figures/feature_attribution_comparison.png`.

**Outputs / acceptance criteria:**
- Four new figures under `reports/figures/`.
- A markdown cell summarizing: which features TabNet attends to most, whether they overlap with XGBoost's importance ranking, and what that says about the two models' inductive biases.

---

## Step 6 — Cross-model error overlap

**Goal:** Answer the spec's explicit Applied-Analysis question: *"Do models fail on the same examples?"*

**Why it matters:** Scientific Rigor + Analysis Depth. Small effort, high rubric value. The spec literally asks this question.

**Depends on:** Step 2.

**Steps:**
1. Add a section "Cross-Model Error Analysis".
2. Load `xgb_test_preds.npy`, `tabnet_test_preds.npy`, `*_y_true.npy`.
3. Build a 2×2 contingency table:
   |  | TabNet correct | TabNet wrong |
   |---|---|---|
   | XGB correct | a | b |
   | XGB wrong | c | d |
4. Report counts and percentages. Compute the *agreement rate* `(a+d)/N` and the *Cohen's kappa* between the two models' correct/incorrect indicator vectors.
5. **Per-class breakdown:** for each of the 10 classes, report (XGB-only-correct, TabNet-only-correct, both-correct, both-wrong) — i.e., where each model uniquely contributes.
6. Save the contingency table as `reports/figures/error_overlap.png` (a small annotated heatmap is fine) and per-class breakdown as `reports/figures/error_overlap_per_class.png`.
7. Persist into `week4_results.json` under `error_overlap`.

**Outputs / acceptance criteria:**
- Two figures + JSON entry.
- A one-paragraph interpretation: are errors correlated (suggesting shared signal limits) or independent (suggesting an ensemble might help)?

---

## Step 7 — Per-class precision/recall on temporal split

**Goal:** The proposal's hypothesis specifically targets Hail vs Thunderstorm Wind separation. Currently we only have aggregate macro-F1 on the temporal split (XGB 0.7556, TabNet 0.7487). We need per-class precision/recall to know whether the gap survives temporally for the *interesting* classes.

**Why it matters:** Scientific Rigor — directly tests the original proposal hypothesis. Without this, the report can only claim aggregate generalization.

**Depends on:** Step 2 (need temporal-split test predictions saved analogously). If the original temporal-split predictions weren't persisted, persist them now.

**Steps:**
1. If `reports/xgb_temporal_test_preds.npy` and `reports/tabnet_temporal_test_preds.npy` don't exist, run the temporal-split fits on CUDA and save them (mirror Step 2).
2. Compute `sklearn.metrics.classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)` for both models.
3. Save the resulting per-class precision/recall/F1 to `reports/temporal_per_class.json`.
4. Build a grouped bar chart: x-axis = class, y-axis = F1, bars = (XGB random, XGB temporal, TabNet random, TabNet temporal). Save `reports/figures/per_class_f1_random_vs_temporal.png`.
5. Pay particular attention to Hail and Thunderstorm Wind — call out their per-class deltas explicitly in the markdown cell.

**Outputs / acceptance criteria:**
- New JSON file + figure.
- One-paragraph interpretation focused on whether the per-class gap on Hail/Thunderstorm Wind matches the proposal's prediction.

---

## Step 8 — Scaffold the final report

**Goal:** Create the report skeleton with section headers, figure placeholders, and a one-line "what goes here" note per section. The intent is that step 9 ("write the report") is just filling in prose against an already-structured doc with all figures attached.

**Why it matters:** Final Report = 45 pts (single largest item). Starting with a skeleton makes the writing tractable.

**Depends on:** Steps 0–7 should all be DONE before this is *finished*, but the skeleton itself can be written immediately and is the safest place to begin in parallel.

**Steps:**
1. Choose tooling. Recommended: LaTeX (matches the proposal). Create `reports/final_report/` with a `main.tex`, `references.bib`, and a `figures/` symlink to `reports/figures/`.
2. Use this section structure (matches the rubric exactly):
   - **Abstract** (≤200 words)
   - **1. Introduction** — domain motivation, hypothesis (use the framing from Step 1), summary of contributions.
   - **2. Related Work** — TabNet (Arik & Pfister 2021), XGBoost (Chen & Guestrin 2016), tabular deep-learning surveys.
   - **3. Data** — NOAA Storm Events, scale, class distribution (Figure: existing class distribution from proposal), preprocessing pipeline, leakage avoidance.
   - **4. Methods** — XGBoost baseline, TabNet, hyperparameter search (grid for XGB, two-stage coarse+refined for TabNet), validation strategy (5-fold CV, held-out test, temporal split), missing-data policy.
   - **5. Experiments** — random-split headline numbers; temporal-split numbers; CZ_TYPE ablation + meteorological-prior replacement; NaN-policy A/B; refined TabNet k-fold CV.
   - **6. Statistical Analysis** — bootstrap CIs, McNemar (Step 3).
   - **7. Error Analysis** — confusion matrices for both models, per-class metrics, cross-model error overlap (Step 6), per-class temporal-split (Step 7), Hail-vs-Thunderstorm Wind focus.
   - **8. Uncertainty Quantification** — reliability diagrams, ECE, entropy analysis (Step 4).
   - **9. Interpretability** — XGBoost feature importance, TabNet attention masks, comparison (Step 5).
   - **10. Discussion** — why TabNet did *not* beat XGBoost in the low-feature regime; what this implies about attention-based feature selection on small-feature tabular tasks.
   - **11. Limitations** — feature count, NaN-subset comparison caveat, single-year test fold.
   - **12. Conclusion**
   - **References**
3. For each section, write a 2–3 sentence "scope note" inside a `% TODO:` LaTeX comment so the next pass knows what bullets to hit.
4. Insert all 8 existing figures + new figures from Steps 3–7 with captions (caption text can be a placeholder for now).
5. Compile to PDF and commit. The PDF should be ~3–5 pages of mostly headers, captions, and TODOs at this stage.

**Outputs / acceptance criteria:**
- `reports/final_report/main.tex` compiles to a PDF without errors.
- All sections present, all figures embedded, every section has a TODO comment describing what prose goes there.
- Page count is in the 8–12 page target range *as a skeleton estimate* (sections roughly sized; each section budgeted explicitly).

---

## Step 9 — Write the final report

**Goal:** Replace every `% TODO:` block in the skeleton with actual prose, hitting the 8–12 page target.

**Why it matters:** 45 points.

**Depends on:** Step 8 (skeleton); all of Steps 0–7 (numbers and figures).

**Steps:**
1. Section budget (rough page targets; adjust as needed):
   - Intro + Related Work: 1.5 pages
   - Data + Methods: 2 pages
   - Experiments: 2 pages
   - Statistical Analysis + Error Analysis + UQ + Interpretability: 3 pages
   - Discussion + Limitations + Conclusion: 1 page
   - References: 0.5 page
2. Writing principles (these come from the rubric):
   - Every quantitative claim cites a figure or `week4_results.json` field.
   - Every model comparison is reported with a CI or p-value (use Step 3 outputs).
   - Negative results are stated honestly. The TabNet-vs-XGBoost gap is part of the contribution.
   - Reproducibility section (subsection of Methods or its own appendix) lists: random seed, CSV file SHA, train/test split rule, exact hyperparameter grids, the regen-cell location.
3. Pass 1: fill all sections. Pass 2: tighten for length. Pass 3: read aloud, fix flow, add transitions.
4. Compile, check page count is 8–12, check all figure references resolve.

**Outputs / acceptance criteria:**
- Compiled PDF in 8–12 page range.
- No `% TODO` left in the source.
- Every claim numbered against a figure, table, or JSON field.

---

## Step 10 — Final presentation deck

**Goal:** 15-minute presentation + 5-minute Q&A.

**Why it matters:** 15 points.

**Depends on:** Step 9 effectively done (so the slides reflect the locked report narrative).

**Steps:**
1. Slide budget for 15 minutes ≈ 12–15 content slides + title + agenda + thanks.
2. Suggested deck:
   - Title (1) — title, team, course, date.
   - Motivation (1) — NOAA Storm Events, why severe-weather classification matters.
   - Hypothesis (1) — TabNet vs XGBoost in the low-feature tabular regime; Hail vs Thunderstorm Wind as the interesting case.
   - Data (1) — class distribution figure, scale, leakage controls.
   - Methods (2) — XGBoost grid; TabNet two-stage tuning + k-fold CV.
   - Headline results (1) — macro-F1 bar chart with CIs (Step 3 figure).
   - Confusion matrices side-by-side (1) — XGB vs TabNet.
   - Error overlap + per-class temporal (1) — Step 6 + Step 7.
   - Statistical significance (1) — bootstrap CI + McNemar.
   - Uncertainty (1) — reliability diagrams + ECE.
   - Interpretability (1) — XGB importance vs TabNet attention.
   - Discussion / negative result framing (1) — why TabNet didn't win in this regime.
   - Limitations + future work (1).
   - Conclusion (1).
   - Q&A backup slides (3–5) — extra figures kept hidden, e.g., temporal-split details, CZ_TYPE ablation, NaN policy.
3. Rule of thumb: ~1 minute per content slide. Practice once end-to-end and time it. If over 15 minutes, cut content slides — do not speak faster.
4. Have one designated presenter per section; ensure all team members present at least one slide (rubric: "all team members must contribute substantially").

**Outputs / acceptance criteria:**
- Deck file (Keynote / PowerPoint / Google Slides / PDF beamer).
- One full timed run-through completed without exceeding 15 minutes.

---

## Reference

### Headline numbers (locked, from `reports/week4_results.json`)

| Config | Macro F1 | Macro ROC-AUC |
|---|---|---|
| Default XGBoost (random split) | 0.8887 | 0.9882 |
| Tuned XGBoost (random split) | **0.9001** | **0.9907** |
| TabNet baseline (random split) | 0.7828 | 0.9760 |
| TabNet refined (random split) | 0.8513 | 0.9792 |
| Tuned XGBoost (2023→2024) | 0.7556 | — |
| TabNet refined (2023→2024) | 0.7487 | — |

Refined TabNet config: `n_d=n_a=16, n_steps=7, lr=0.02, gamma=1.0, batch_size=1024`. CV macro-F1 mean 0.8298, std 0.0076 (3-fold).

Production decision (Week 5): impute missing coordinates; keep CZ_TYPE; ship tuned XGBoost.

### Rubric (from `reports/cisc484_684_final_project.pdf`)

- Proposal: 10
- Preliminary results: 20
- Weekly updates: 10
- Final report: 45 (Scientific Rigor 15, Technical Execution 15, Analysis Depth 10, Clarity 5)
- Final presentation: 15

### File map

```
reports/
├── FINAL_PUSH_TASKS.md          # this file
├── proposal.pdf                  # Phase 1 submission
├── week{1..5}-updates.pdf        # weekly updates
├── week4_results.json            # locked numbers (Step 0 fixes a bug here)
├── framing.md                    # Step 1 output
├── figures/                      # all report figures land here
│   ├── confusion_matrix_tuned_xgb.png
│   ├── feature_importance_tuned_xgb.png
│   ├── roc_curves_tuned_xgb.png
│   ├── model_comparison_macro_f1.png
│   ├── model_comparison_macro_auc.png
│   ├── random_vs_temporal_split.png
│   ├── nan_coord_policy.png
│   ├── cz_type_ablation_and_priors.png
│   ├── macro_f1_with_ci.png                  # Step 3
│   ├── reliability_diagrams.png              # Step 4
│   ├── entropy_correct_vs_incorrect.png      # Step 4
│   ├── confusion_matrix_tabnet.png           # Step 5
│   ├── roc_curves_tabnet.png                 # Step 5
│   ├── tabnet_attention_masks.png            # Step 5
│   ├── feature_attribution_comparison.png    # Step 5
│   ├── error_overlap.png                     # Step 6
│   ├── error_overlap_per_class.png           # Step 6
│   └── per_class_f1_random_vs_temporal.png   # Step 7
├── tabnet_test_preds.npy                     # Step 2
├── tabnet_test_proba.npy                     # Step 2
├── tabnet_test_y_true.npy                    # Step 2
├── tabnet_test_index.npy                     # Step 2
├── tabnet_class_labels.json                  # Step 2
├── tabnet_masks.npy                          # Step 2
├── xgb_test_preds.npy                        # Step 2
├── xgb_test_proba.npy                        # Step 2
├── xgb_temporal_test_preds.npy               # Step 7 (if needed)
├── tabnet_temporal_test_preds.npy            # Step 7 (if needed)
├── temporal_per_class.json                   # Step 7
└── final_report/                              # Step 8
    ├── main.tex
    ├── references.bib
    └── figures -> ../figures/
```

### How to brief a fresh context window

> "Read `reports/FINAL_PUSH_TASKS.md`. We are on Step N. Do that step. When done, mark its row DONE in the status board with a one-line note pointing at the artifact you produced, then stop and report what you did."

That instruction plus this file is enough — every step is self-contained, lists its inputs and dependencies, and defines acceptance criteria.
