# Running list of paper-drafting concerns

This file tracks open questions, judgment calls, and risks discovered
during the markdown draft. Each entry is one issue. Resolve by editing
the relevant section file and marking the entry **resolved** with a
one-line note on what changed.

---

## C1 — §8.6: reliability-diagram caption was wrong on first pass (resolved)

**Status:** resolved by edit on 2026-05-10. Caption now describes the
diagram accurately (XGBoost noise with a mid-range dip; TabNet tracking
the diagonal) and clarifies that the under-confidence reading depends on
§8.2 / §8.5, not on the diagram itself.

**Reason it happened:** the §8 prose was drafted before opening the
figure. The under-confidence story is real (Brier/NLL/entropy all
support it) but the *reliability diagram alone* does not visualize it —
a reliability diagram with line-only rendering normalizes away the
per-bin prediction *frequencies* that would distinguish "well-calibrated
and confident" from "well-calibrated and soft."

**Open follow-up — bin-frequency plot.** If we want the figure to carry
the under-confidence message visually, the right artifact is a histogram
of `max(P(class))` per row, per model, overlaid. That would show
TabNet's mass shifted left toward lower confidences while XGBoost's mass
is more right-skewed. The existing `entropy_correct_vs_incorrect.png`
partially substitutes for this — flag during §9/§10 review whether to
build a confidence-histogram figure too.

---

## C2 — Temporal-split TabNet uses the coarse, not refined config (disclosed in §5.3)

**Status:** disclosed honestly in §5.3 and reframed as a lower-bound
argument in §11; user chose option 1 (do not retrain on CUDA).

**Risk for the paper:** a careful reviewer will ask why the random-split
TabNet (gamma=1.0, refined) differs from the temporal-split TabNet
(gamma=1.3, coarse). The §5.3 disclosure handles this; the §6.3 narrative
explicitly compares matched-config TabNet across splits to keep the
temporal-narrowing argument honest.

**Open follow-up — none expected**, but the LaTeX pass should double-
check that the table in §6.3 keeps both the cross-config and
matched-config numbers so the reader can see both.

---

## C3 — Bootstrap p-value reported as "< 10⁻³" rather than the raw "0.0" (open, low-risk)

**Status:** wrote `< 10⁻³` in §7.1 with reasoning in chat. None of 1,000
paired resamples favored TabNet, so the empirical p is 0; the honest
upper bound at 95% one-sided is roughly 3/1,000.

**Decision needed:** if a reviewer prefers explicit reporting, switch to
`p = 0/1000 (≤ 3 × 10⁻³ at 95% confidence)` or similar. Either form is
defensible — keep `< 10⁻³` unless the user changes preference.

---

## C4 — Class-distribution figure was built post-hoc (resolved)

**Status:** resolved. `reports/figures/class_distribution.png` is built
from frozen per-class counts; reproducible from
`tool/_make_class_distribution_fig.py`.

**Reason it's worth tracking:** the plan said "do not invent figures."
The judgment call was that *computing* a figure from existing data
artifacts is not the same as *inventing* one, and the user explicitly
approved building it. Noted here so the rule isn't relaxed silently.

---

## C5 — Total-N discrepancy: 145K (framing.md) vs 117,717 (actual) (resolved)

**Status:** resolved. `framing.md` was edited on 2026-05-10 to read
"117,717 records after filtering" instead of "145K records." 145K was
the proposal's number, likely 2023-only or pre-filter raw.

**Open follow-up — none**, but if a reviewer cross-references our paper
against the proposal PDF, the §4 row count will be lower than the
proposal table; we should be ready to explain. The right answer is the
proposal table was a starting estimate; the paper reports what the model
was actually trained on.

---

## C6 — §4 missing-coord rate is quoted from notebook narrative, not recomputed (open)

**Status:** open. §4 says "roughly 39.7% of records have missing
BEGIN_LAT / BEGIN_LON." That number came from the notebook prose at
line 3172. The `nan_coord_policy.native_coord_test_pct = 0.6823` in
`week4_results.json` implies a test-fold missing rate of 31.8%, not
39.7%, so the 39.7% likely refers to a different denominator (full
combined dataset before split? raw 2023-only?).

**Decision needed:** either recompute the missing rate from raw data
and pin the exact value, or hedge to "roughly a third of records" so
the precise denominator does not matter. Low priority — the result
(`delta_xgb ≈ 5e-5`) is the load-bearing number for §4 and that's
independent of the exact missing rate.

---

## C7 — TabNet PyTorch seeding (open, low-risk)

**Status:** disclosed in §5.4 ("TabNet does not seed PyTorch globally").
Argued that the runs reproduce within rounding across coarse / tuned /
refined refits, so GPU non-determinism is not load-bearing.

**Open follow-up — none expected**, but if a reviewer asks for a
bit-exact rerun we should be ready to say it is reproducible up to
cuDNN floating-point ordering, not bit-exact.

---

## How to use this file

- Add a new `C<n>` block when a drafting decision could plausibly become
  a reviewer or co-author concern later.
- Resolve by editing the relevant section and changing **Status** to
  `resolved` with a one-line note on what changed.
- Keep entries terse — this is a punch list, not a discussion log.
