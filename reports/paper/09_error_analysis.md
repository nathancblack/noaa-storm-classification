# §9 Error Analysis

The §6 macro-F1 numbers and the §7 statistical tests settle the aggregate
question. This section opens up the per-row and per-class structure of
those errors. We do three things: present the confusion matrices for
both finalized models on the random-split test fold, examine how their
errors overlap row by row, and read the per-class F1 numbers from the
random split alongside the 2023→2024 temporal split. The temporal
comparison is where the proposal's pre-registered hypothesis is partially
vindicated; we report that vindication carefully and do not let it
overwrite the §6 / §7 headline.

## 9.1 Confusion matrices

Figure&nbsp;10 and Figure&nbsp;11 show the row-normalized confusion
matrices for the tuned XGBoost and the refined TabNet on the 23,544-row
random-split test fold. Three features of those matrices are worth
calling out before the per-class table in §9.3.

First, both models classify Marine Thunderstorm Wind perfectly. Of the
967 test rows in that class, both models are correct on every row
(`error_overlap.per_class: Marine Thunderstorm Wind` —
`both_correct = 967`, `xgb_only = 0`, `tabnet_only = 0`,
`both_wrong = 0`). The class is trivially separable from the others on
the engineered feature set, presumably because `CZ_TYPE_ENC` flags the
event as marine and no other class shares that code in 2023.

Second, the dominant confusion under both models is Hail ↔ Thunderstorm
Wind. These are also the two classes the proposal singled out as
boundary cases. Under XGBoost, Hail's recall is 0.737 (1,053 of 4,140
test rows misclassified, the largest absolute miss-mass on the diagonal);
under TabNet, Hail's recall is 0.630 (1,533 misclassified). The bulk of
that off-diagonal mass lands on Thunderstorm Wind in both matrices,
which is the most common class in the test fold (8,235 rows). The
asymmetry — Hail bleeds into Thunderstorm Wind more than the reverse —
is what we would expect when a dominant class soaks up uncertain
predictions.

Third, TabNet trades recall on Excessive Heat for recall on Heat in a
way XGBoost does not: TabNet's Heat recall is 0.948 (the highest of any
class besides Marine Thunderstorm Wind), but its Excessive Heat recall
drops to 0.675 against XGBoost's 0.846. The two classes are distinguished
in NOAA by a temperature/duration threshold, and TabNet's softer
decision boundary appears to pull threshold-adjacent Excessive Heat rows
back across the boundary into ordinary Heat. This is a small per-class
effect — combined support 2,836 — but it is the most visible single
disagreement between the two confusion matrices.

## 9.2 Cross-model error overlap

A natural question after §7 is whether the two models make the *same*
errors or *different* ones. Same errors imply a shared signal ceiling;
different errors imply that the two models are sensitive to different
parts of the feature space and that an ensemble could improve over
either alone. We computed the four-cell agreement table on per-row
correctness (`week4_results.json: error_overlap`):

| | XGBoost correct | XGBoost wrong | row total |
| --- | ---: | ---: | ---: |
| TabNet correct | 18,419 | 908 | 19,327 |
| TabNet wrong | 2,283 | 1,934 | 4,217 |
| column total | 20,702 | 2,842 | 23,544 |

The agreement rate is 86.4% (the diagonal sum); Cohen's κ on per-row
correctness is 0.472. Two readings.

First, the off-diagonal cells are not symmetric. XGBoost is uniquely
correct on 2,283 rows where TabNet errs; TabNet is uniquely correct on
only 908 rows where XGBoost errs. The ratio is roughly 2.5 : 1 in
XGBoost's favor. This is consistent with §6.1's macro-F1 gap — the
gross row count of XGBoost-only-correct rows minus TabNet-only-correct
rows is +1,375, which translates directly into the accuracy gap on the
random split.

Second, κ = 0.47 sits in the "moderate" agreement band of the
Landis-Koch scale. If the two models made fully independent errors at
their observed individual accuracy rates, the both-wrong cell would
contain roughly (4,225 / 23,544) × (2,842 / 23,544) × 23,544 ≈ 509 rows;
we observe 1,934. The errors are *correlated*, but not nearly redundantly
so: 908 + 2,283 = 3,191 rows are correctable by *one* of the two models
and wrong under the other. That is the headroom an ensemble would draw
on. We do not run that experiment in this paper (it is a separate
modeling pass and the §6 / §7 contribution does not require it), but
§11 flags it explicitly as the most defensible follow-up.

The per-class breakdown of the overlap (Figure&nbsp;13) shows the
headroom is not evenly distributed. Hail has 743 XGB-only-correct rows
against 299 TabNet-only-correct, and 790 both-wrong rows — by far the
largest both-wrong cell of any class, and the place where a confidence-
weighted ensemble would have the least to work with. Thunderstorm Wind
has 795 XGB-only-correct rows against 327 TabNet-only-correct,
re-confirming the §6.1 ordering on the proposal's boundary classes.
The one class where TabNet is uniquely better than XGBoost on net is
Heat: 140 TabNet-only-correct against 29 XGB-only-correct, the only
class with that sign.

## 9.3 Per-class F1: random split vs 2023→2024

Table&nbsp;9.3 sets the random-split per-class F1 alongside the
2023→2024 temporal-split per-class F1 for both models
(`temporal_per_class.json: random_split` and `temporal_split`). The
random-split column resolves the proposal's boundary-class claim
directly; the temporal column resolves the qualification.

| class (support, random / temporal) | XGB random | TabNet random | Δ random (TN − XGB) | XGB temporal | TabNet temporal | Δ temporal (TN − XGB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Drought (1,858 / 3,473) | 0.997 | 0.996 | −0.001 | 0.979 | 0.969 | −0.010 |
| Excessive Heat (1,537 / 2,660) | 0.862 | 0.783 | −0.079 | 0.577 | 0.567 | −0.011 |
| Flash Flood (1,657 / 4,661) | 0.878 | 0.827 | −0.052 | 0.779 | 0.755 | −0.025 |
| Flood (1,064 / 2,472) | 0.845 | 0.781 | −0.065 | 0.697 | 0.637 | −0.060 |
| **Hail** (4,140 / 8,941) | **0.761** | **0.648** | **−0.113** | **0.554** | **0.591** | **+0.037** |
| Heat (1,299 / 3,171) | 0.845 | 0.812 | −0.033 | 0.587 | 0.651 | +0.065 |
| High Wind (1,497 / 3,994) | 0.968 | 0.928 | −0.041 | 0.795 | 0.750 | −0.044 |
| Marine TS Wind (967 / 2,192) | 0.999 | 0.998 | −0.001 | 0.995 | 0.993 | −0.003 |
| **TS Wind** (8,235 / 19,940) | **0.880** | **0.822** | **−0.058** | **0.760** | **0.777** | **+0.017** |
| Winter Weather (1,290 / 3,557) | 0.966 | 0.919 | −0.046 | 0.834 | 0.798 | −0.036 |

On the random split, TabNet loses on every class. The two losses on
the proposal's named boundary classes — Hail −0.113 and Thunderstorm
Wind −0.058 — are not only in the wrong direction relative to the
proposal's prediction, they are the *largest* two losses on the table
in absolute terms (after a tie with Excessive Heat at −0.079). The
random-split per-class evidence is fully consistent with the §6 / §7
headline.

The temporal column tells a different story on exactly the two
boundary classes plus Heat. Hail flips sign: TabNet beats XGBoost by
+0.037 F1 under the 2023→2024 shift. Thunderstorm Wind flips sign:
TabNet beats XGBoost by +0.017 F1. Heat also flips, by a larger
+0.065. On every other class XGBoost retains the lead under shift,
which is why the aggregate temporal macro-F1 still favors XGBoost
(§6.3, 0.7556 vs 0.7487 — a gap of 0.0069 versus the random split's
0.0488).

The asymmetry of the drops is the more diagnostic comparison. Reading
the per-class F1 change from random to temporal:

| class | ΔF1 random→temporal, XGB | ΔF1 random→temporal, TabNet |
| --- | ---: | ---: |
| Hail | −0.207 | **−0.057** |
| Thunderstorm Wind | −0.120 | **−0.045** |
| Excessive Heat | −0.285 | −0.216 |
| Heat | −0.258 | −0.161 |
| Flash Flood | −0.099 | −0.072 |
| Flood | −0.148 | −0.144 |
| High Wind | −0.173 | −0.178 |
| Winter Weather | −0.132 | −0.121 |
| Drought | −0.018 | −0.027 |
| Marine TS Wind | −0.004 | −0.006 |

On the two proposal classes, TabNet's degradation under the shift is
roughly 3–4× smaller than XGBoost's. The pattern is not universal —
High Wind and Drought show the opposite ordering — but the two classes
the proposal *named in advance* are the two where TabNet's softer
representations transfer to the next year visibly better. We are not
claiming this is a vindication of the proposal's mechanism; one annual
fold is far too thin a base for a domain-shift claim, and the §12
limitations make that explicit. We are claiming, with the framing of
§1 and the discussion of §11, that this is a substantive qualification
worth reporting: on the classes the proposal singled out, and only on
those classes, the temporal-shift fold partially recovers the
pre-registered hypothesis.

## 9.4 Figures

- **Figure&nbsp;10.** `reports/figures/confusion_matrix_tuned_xgb.png` —
  row-normalized confusion matrix for the tuned XGBoost on the random-
  split test fold. Anchors §9.1.
- **Figure&nbsp;11.** `reports/figures/confusion_matrix_tabnet.png` —
  row-normalized confusion matrix for the refined TabNet on the same
  fold. Read alongside Figure&nbsp;10 for the Hail ↔ Thunderstorm Wind
  and Heat ↔ Excessive Heat patterns.
- **Figure&nbsp;12.** `reports/figures/error_overlap.png` — aggregate
  agreement table from §9.2 in plot form.
- **Figure&nbsp;13.** `reports/figures/error_overlap_per_class.png` —
  per-class breakdown of the four agreement cells. Anchors the §9.2
  reading on Hail, Thunderstorm Wind, and Heat.
- **Figure&nbsp;14.** `reports/figures/per_class_f1_random_vs_temporal.png` —
  per-class F1 under the random split and the 2023→2024 temporal split
  for both models. The figure anchors §9.3 and is the visual carrier
  of the temporal-split qualification.
