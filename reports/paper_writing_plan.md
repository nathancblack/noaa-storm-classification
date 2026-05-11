# Plan: writing the final paper (content-first → LaTeX)

This plan is designed so a fresh context window can pick it up cold and execute
without depending on prior conversation memory. The writing workflow is
deliberately **content-first**: draft every section as plain markdown, get the
prose and numbers right, *then* transcribe to LaTeX as a mechanical pass.

If you are the fresh context window: do not trust any specific number, claim,
section structure, or design choice that you cannot find in the files listed
below. The author of this plan may have hallucinated. Treat the on-disk
artifacts (JSON, PDFs, notebook outputs, figures) as authoritative.

## Status at the time this plan was last revised

- Both CUDA runs are complete. All `.npy` and `.json` artifacts under
  `reports/` should exist; verify with `ls reports/*.npy reports/*.json` and
  `ls reports/figures/*.png`.
- Steps 0 through 7 in `reports/FINAL_PUSH_TASKS.md` are DONE.
- The framing is **A′ with a temporal-split qualification.** The negative
  result is the headline on the random split and the calibration audit, but
  the temporal-split per-class data is genuinely supportive of TabNet on the
  two boundary classes the proposal named (Hail, Thunderstorm Wind). Read
  `reports/framing.md` for the current canonical statement of the thesis;
  do not paraphrase it from memory.

---

## 0. Before you write anything — read these in this order

1. **`reports/final-directions.pdf`** — the official project spec and rubric.
   Note especially the §2.2 data requirement, §7.3 Phase 3 deliverables, and
   the Final Report rubric breakdown (Scientific Rigor 15 / Technical Execution
   15 / Analysis Depth 10 / Clarity 5).
2. **`reports/proposal.pdf`** — the team's pre-registered hypothesis. The paper
   must engage this honestly. Specifically the proposal commits to a
   prediction about TabNet outperforming on subtle boundary classes (Hail vs
   Thunderstorm Wind). Re-read this so you state the prediction correctly.
3. **`reports/framing.md`** — the *current* thesis of the paper. This file has
   gone through revisions; trust only what is in it right now, not any prior
   version you may recall.
4. **`reports/FINAL_PUSH_TASKS.md`** — the status board. The `Status` column
   tells you what is DONE, PARTIAL, or TODO. Believe the board.
5. **`reports/week4_results.json`** — every locked headline number lives here
   under keyed sections. Quote numbers from this file, not from prose.
6. **`reports/calibration_audit.json`** — the post-hoc audit that reframed the
   paper. Contains bootstrap CIs on ECE, Brier, NLL, and temperature-scaling
   results. The conclusion that the calibration "win" for TabNet does not
   survive scrutiny depends on these numbers.
7. **`reports/statistical_tests.json`** — bootstrap CIs and McNemar.
8. **`reports/uncertainty.json`** — raw uncertainty/calibration numbers.
9. **`reports/temporal_per_class.json`** — per-class precision/recall tables.
10. **`reports/figures/`** — list it (`ls reports/figures/`). Every figure
    referenced in the paper must exist here. If a figure named in the outline
    is missing, do not invent it — flag it.

After reading the above, *also* spot-check the notebook to ground methodology
claims:

- `tool/weather.ipynb` — primary code. It is long (~4300 lines, ~30 cells).
  Do not read it end-to-end; jump to specific cells by id when you need to
  cite methodology. The cell-id ↔ topic mapping is in
  `reports/FINAL_PUSH_TASKS.md` and in the file-map at the bottom of that file.

---

## 1. Verify ground-truth before writing

Before drafting any prose, *confirm* the following by direct read. Do not
write any of them from memory. If any answer disagrees with what you expect,
trust the artifact.

1. Open `reports/framing.md` and read the **Thesis** section verbatim. The
   paper's first sentence in the abstract must paraphrase this thesis, not
   contradict it.
2. Open `reports/week4_results.json` and confirm the headline numbers under
   `random_split` and `temporal_split_2023_to_2024`. Use these exact values.
3. Open `reports/calibration_audit.json` and confirm:
   - whether the bootstrap diff CI on `(XGB_ECE − TabNet_ECE)` excludes zero,
   - whether Brier and NLL agree on which model wins,
   - whether temperature scaling brings XGB's post-T ECE below TabNet's.
   If any of these three is **not** consistent with the framing, stop and
   ask the user before proceeding — the framing may need to be re-opened.
4. Open `reports/statistical_tests.json` and confirm the bootstrap p-value
   and McNemar statistic. Use those values verbatim in the Statistical
   Analysis section.
5. Open `reports/temporal_per_class.json` and confirm:
   - the `temporal_split` key exists (it should, both CUDA runs are done),
   - and the per-class F1 entries for `Hail` and `Thunderstorm Wind` under
     `temporal_split.tabnet` are *higher* than the corresponding entries
     under `temporal_split.xgb`. This is the core of the temporal-split
     qualification in §7 of the paper; if the JSON disagrees, stop and
     re-read `reports/framing.md` before writing.
6. Verify all CUDA-dependent artifacts exist:
   - `reports/xgb_test_preds.npy`, `reports/tabnet_test_preds.npy`,
     `reports/xgb_test_proba.npy`, `reports/tabnet_test_proba.npy`,
     `reports/tabnet_test_y_true.npy`, `reports/tabnet_masks.npy`,
     `reports/xgb_feature_importance.npy`, `reports/xgb_feature_cols.json`,
     `reports/xgb_temporal_test_preds.npy`,
     `reports/tabnet_temporal_test_preds.npy`,
     `reports/temporal_test_y_true.npy`.
   - If any of these is missing, stop and ask the user — do not rerun CUDA.

---

## 2. Output structure

Create `reports/paper/` as a working directory. Each section gets its own
markdown file. This isolates revisions and makes a single section easy to
hand off for review.

```
reports/paper/
├── 00_outline.md            # section-by-section outline with figure
│                            # placements, citation needs, and page budgets
├── 01_abstract.md           # ≤200 words, written LAST
├── 02_introduction.md
├── 03_related_work.md
├── 04_data.md
├── 05_methods.md
├── 06_experiments.md
├── 07_statistical_analysis.md
├── 08_uncertainty.md        # this is the audit section — the most novel
├── 09_error_analysis.md
├── 10_interpretability.md
├── 11_discussion.md
├── 12_limitations.md
├── 13_conclusion.md
├── references.md            # one entry per work, full citation, with the
│                            # exact bibkey you intend to use in LaTeX later
└── figure_captions.md       # caption per figure, in one place
```

Do not write any LaTeX until every section in `reports/paper/` has been
reviewed by the user. The LaTeX pass is mechanical and comes after content
is locked.

---

## 3. Drafting order (recommended)

Write sections in this order; each one depends on the previous being
roughly stable.

1. **`00_outline.md`** — section-by-section. Each section gets:
   - 2-3 sentence scope note
   - explicit list of which figures and JSON keys it cites
   - paragraph count and rough word budget (the report target is 8–12 pages
     in `\documentclass[11pt]{article}`)
   - confirmation that the section makes a claim consistent with
     `framing.md`
2. **`04_data.md`** — straightforward and self-contained. Cite the proposal's
   table (145,394 rows; 51 raw columns; 10 retained classes). Include the
   §2.2 footnote noting raw-column-count vs engineered-feature-count.
3. **`05_methods.md`** — XGBoost baseline + grid; TabNet two-stage tuning +
   k-fold CV; validation strategy; preprocessing; reproducibility. Confirm
   all hyperparameters by reading the relevant cells in
   `tool/weather.ipynb` rather than recalling them.
4. **`06_experiments.md`** — the table of locked numbers from
   `week4_results.json`. This is mostly mechanical; do not invent numbers
   not in the JSON.
5. **`07_statistical_analysis.md`** — bootstrap CIs + McNemar. Numbers come
   from `statistical_tests.json`.
6. **`08_uncertainty.md`** — *the audit section*. Lead with the apparent
   ECE finding, then walk through the three controls (Brier/NLL,
   temperature scaling, robustness to binning if you choose to verify).
   This section is the paper's distinctive methodological contribution
   under Framing A′; spend more words here than the original Step 8 outline
   suggested.
7. **`09_error_analysis.md`** — confusion matrices, cross-model overlap
   (Cohen's κ), per-class tables on both random and temporal splits. This is
   where the temporal-split qualification lives: the per-class F1 table must
   include both splits side-by-side and call out explicitly that TabNet
   beats XGBoost on Hail (+0.037) and Thunderstorm Wind (+0.017) under
   temporal shift, while losing both classes on the random split. Read the
   "Temporal-split qualification" section of `reports/framing.md` for the
   tone — supportive but careful not to over-claim from one annual fold.
8. **`10_interpretability.md`** — XGB importance vs TabNet attention. The
   bonus comparison plot may or may not exist on disk; check first.
9. **`11_discussion.md`** — three threads, in this order:
   - *Mechanistic explanation* of the random-split negative result: the
     "low-feature regime saturates gain-based splits" story. With only 8
     informative features XGBoost's gain-based splits already enumerate
     near-optimal partitions; TabNet's attention has nothing irrelevant to
     filter out.
   - *Why the temporal-split qualification is consistent with the
     mechanism.* TabNet's softer, more regularized representations appear
     to degrade more gracefully on the boundary classes when the
     distribution shifts; XGBoost's tight fit on 2023 patterns hurts it
     more on the 2024 hold-out. State this as a hypothesis-of-mechanism,
     not a proven claim — one annual fold is not enough to anchor it.
   - *Cross-model error independence* (Cohen's κ = 0.47) and the ensemble
     follow-up direction. Note this is a future-work bullet, not an
     experiment we ran.
10. **`12_limitations.md`** — feature count, single-year temporal fold,
    no ensemble baseline, ECE-binning sensitivity if relevant.
11. **`13_conclusion.md`** — three to five sentences.
12. **`02_introduction.md`** — write *after* you have the body, so the
    motivation and contribution summary match what was actually delivered.
13. **`03_related_work.md`** — short. Five core citations minimum (TabNet,
    XGBoost, two tabular-DL surveys, calibration). Confirm bibkeys against
    `references.md`.
14. **`01_abstract.md`** — write LAST. ≤200 words.
15. **`figure_captions.md`** — write captions after the sections that cite
    them. Each caption identifies the figure path, the source data, and
    the takeaway in one sentence.

---

## 4. Rules for the drafter

These exist because the paper has been re-framed multiple times. Drift is
the biggest risk.

1. **Quote numbers from JSON, never from memory.** If a number is not in a
   JSON file under `reports/`, do not put it in the paper. If you think a
   number should exist and does not, flag it to the user; do not invent.
2. **Every quantitative claim must point to a figure, table, or JSON key.**
   In markdown, use parenthetical `(week4_results.json: random_split.tuned_xgb_macro_f1)`
   style citations. These get removed in the LaTeX pass but they are how
   reviewers and the user verify each claim during content review.
3. **Do not commit to numbers that depend on the second CUDA run unless
   you have verified the artifacts exist on disk.** Specifically: the
   temporal per-class breakdown and the bonus feature-attribution plot.
4. **Do not soften the negative result, but do not omit the temporal-split
   qualification either.** The proposal made a specific prediction that was
   falsified on the random split and on aggregate. That is the headline.
   But the temporal-split per-class F1 on Hail (TabNet 0.5910 vs XGBoost
   0.5542) and Thunderstorm Wind (TabNet 0.7768 vs XGBoost 0.7596)
   partially vindicates the original prediction under distribution shift.
   §7 (Error Analysis) and §10 (Discussion) must report this honestly. It
   is not the thesis but it is a substantive qualification — do not bury
   it and do not promote it past §7 / §10. The exact placement and tone
   are described in `reports/framing.md` under "Temporal-split
   qualification".
5. **Do not re-introduce Framing B's calibration-tradeoff thesis.** That
   framing was audited and rejected (`calibration_audit.json`). The audit
   itself goes in §8, but the *thesis* is Framing A′ (negative result
   strengthened by the audit). If the user asks why the calibration
   "advantage" is not the headline, point them at the audit JSON.
6. **Match the proposal's voice and team list.** Team is Dhruv Patel,
   Marcos Diaz Vazquez, Nathaniel Black. Title and section headings
   should be in the same plain-article style the proposal used.
7. **Page budget.** Target 10 pages in `\documentclass[11pt]{article}`
   (spec range is 8–12). Suggested section budgets (revise if needed):
   - Intro + Related Work: 1.5 pp
   - Data + Methods: 2.5 pp
   - Experiments: 2 pp
   - Statistical Analysis + Uncertainty (audit-heavy) + Error Analysis +
     Interpretability: 2.5 pp
   - Discussion + Limitations + Conclusion: 1 pp
   - References: 0.5 pp

---

## 5. LaTeX pass (do this only after content review)

1. Confirm the proposal was typeset in plain `\documentclass[11pt]{article}`
   by opening `reports/proposal.pdf` and verifying. Match that template.
2. Create `reports/final_report/` with:
   - `main.tex`
   - `references.bib`
   - `figures/` symlinked to `reports/figures/`
3. Build the document one section at a time, copying prose from
   `reports/paper/*.md` into `\section{...}` blocks. Replace markdown
   parenthetical citations with `\cite{...}` keyed to `references.bib`.
   Replace markdown tables with `tabular` environments. Replace inline
   figure references with `\includegraphics`.
4. Compile with `pdflatex` twice (for cross-references). Verify page count
   is in 8–12.
5. Read the rendered PDF end-to-end before declaring done.

---

## 6. What the fresh context window should ask the user before starting

If any of these is unclear, do not proceed silently — ask:

1. Has the second CUDA run completed? Are the temporal per-class artifacts
   on disk?
2. Has the user reviewed `framing.md` and confirmed Framing A′? (The file
   has been rewritten more than once.)
3. Does the user want the paper to be written as one full markdown draft
   for one round of review, or section-by-section with review in between?
4. Are there any team-internal stylistic conventions (we, the authors;
   present vs past tense; American vs British spelling) the user wants
   applied?
5. Who is the primary author for the cover page? (Proposal lists three
   authors.)

---

## 7. What to *not* do

- Do not refactor `tool/weather.ipynb`.
- Do not re-run the CUDA pipeline.
- Do not modify `week4_results.json`, `calibration_audit.json`,
  `statistical_tests.json`, `uncertainty.json`, or any file under
  `reports/figures/`. These are inputs to the paper, not products of it.
- Do not write LaTeX before the markdown content is reviewed.
- Do not change the project-wide hypothesis framing without an explicit
  user decision.
- Do not add new experiments. The science is locked.

---

## 8. End state

When this plan has been executed:

- `reports/paper/` contains 14 markdown files (one per section + outline +
  references + captions), each reviewed by the user.
- `reports/final_report/main.tex` compiles to a 8–12 page PDF.
- Every quantitative claim in the PDF is traceable to a JSON key or a
  figure under `reports/figures/`.
- No `% TODO` markers remain in the LaTeX source.
- The bibliography contains at minimum: Arik & Pfister 2021 (TabNet),
  Chen & Guestrin 2016 (XGBoost), Grinsztajn et al. 2022 (tabular DL
  survey), Shwartz-Ziv & Armon 2022 (tabular DL critique), Guo et al. 2017
  (calibration / temperature scaling).

When the report PDF is signed off by the user, Step 8 and Step 9 in
`FINAL_PUSH_TASKS.md` are DONE. The remaining work is then Step 10
(presentation deck), which is out of scope for this plan.
