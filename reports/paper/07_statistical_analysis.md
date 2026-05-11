# §7 Statistical Analysis

The point-estimate gap of 0.0488 macro-F1 between tuned XGBoost and the
refined TabNet on the random-split test fold (§6.1) admits two reads: it
could reflect a real difference in model behavior, or it could be within
the noise we should expect from a 23,544-row test fold. We address the
question with two complementary tests: a paired bootstrap on the macro-F1
difference and a McNemar test on per-row correctness. All numbers below
are from `reports/statistical_tests.json`.

## 7.1 Paired bootstrap on macro-F1

We drew 1,000 paired bootstrap resamples from the random-split test
fold (`seed = 42`). Each resample drew test-row indices with replacement
and computed macro-F1 for both models using the *same* sampled indices,
so the resampled distributions of the two models are jointly correlated
in the obvious way (a row that helps one model on resample *b* is the
same row used to score the other model on resample *b*). We report the
bootstrap mean of the per-resample difference and the 95% percentile
interval.

| quantity | value |
| --- | --- |
| XGBoost macro-F1 (point) | 0.9001 |
| XGBoost macro-F1 95% CI | [0.8958, 0.9041] |
| TabNet macro-F1 (point) | 0.8513 |
| TabNet macro-F1 95% CI | [0.8463, 0.8560] |
| paired diff (XGB − TabNet), mean | +0.0489 |
| paired diff 95% CI | [+0.0444, +0.0536] |
| two-sided p-value | < 10⁻³ |

The two model-level confidence intervals are disjoint. The 95% CI on the
paired difference is well to the right of zero, and not a single one of
the 1,000 resamples produced a TabNet-favoring difference, so the
two-sided empirical p-value is 0 to the resolution of our resample count
(reported as < 10⁻³). The bootstrap mean diff (+0.0489) sits within 0.0001
of the point-estimate diff (+0.0488), so the resampling is not introducing
any noticeable bias.

## 7.2 McNemar on per-row correctness

The bootstrap tests whether a single scalar summary (macro-F1) differs;
McNemar tests whether the two models *agree* on a per-row basis under
the null of equal error rates. We tabulate the joint correctness of the
two models against the ground-truth labels on the 23,544-row test fold.

|  | XGB correct | XGB wrong | row total |
| --- | ---: | ---: | ---: |
| **TabNet correct** | 18,419 | 908 | 19,327 |
| **TabNet wrong** | 2,283 | 1,934 | 4,217 |
| **column total** | 20,702 | 2,842 | 23,544 |

The two off-diagonal cells — 2,283 rows where only XGBoost is correct
versus 908 rows where only TabNet is correct — are the McNemar test's
input. With continuity correction the statistic is χ² = 591.6 on 1 d.f.,
p ≈ 1.1 × 10⁻¹³⁰. The asymmetry between the off-diagonals is large and
in XGBoost's favor: there are roughly 2.5 rows that only XGBoost gets
right for every row that only TabNet does.

## 7.3 Reading

Both tests reject the null of equivalence decisively, and both lean in
the same direction. The aggregate macro-F1 gap is not within the noise
that the test fold's size would otherwise tolerate, and the per-row
disagreement is not symmetric between the two models. We treat this
as the central statistical justification for the §6.1 headline.

Two caveats worth naming, both of which appear again later. First, McNemar
treats per-row correctness as the unit of disagreement and is blind to
*which* class each model got right; the 908 TabNet-only-correct rows are
not random — they concentrate in specific classes and the §9 error
analysis recovers that structure. Second, the kappa statistic on the
joint correctness vector is 0.47 (`week4_results.json:
error_overlap.cohen_kappa`), so the two models' errors are moderately —
but not strongly — correlated. That moderate-correlation finding sets up
the ensemble future-work bullet in §11.

Figure 2 (§6.4, `reports/figures/macro_f1_with_ci.png`) plots the two
macro-F1 point estimates with their 95% bootstrap intervals; it is the
figure that anchors this section visually.
