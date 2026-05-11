# §8 Calibration and Uncertainty

The macro-F1 result of §6.1 settles the proposal's primary hypothesis,
but the proposal also commits us to a careful analysis of the
*probability quality* of the two models. The predicted probabilities are
where TabNet's softer, attention-mixed representations could in principle
pay off even when the hard-label decision still favors gradient boosting.
This section is the most novel methodological contribution of the paper:
we report what a surface read of one calibration metric (Expected
Calibration Error) initially suggested, then put that finding through
three controls, and find that it does not survive any of them.

## 8.1 The apparent finding

We computed Expected Calibration Error on the full random-split test
fold (23,544 rows) using ten equal-width bins over the maximum-class
confidence per the standard definition (Guo et al. 2017). The results
strongly favor TabNet (`week4_results.json: uncertainty`,
`reports/uncertainty.json`):

| model | ECE (full test) |
| --- | ---: |
| tuned XGBoost | 0.0250 |
| refined TabNet | 0.0043 |

A 500-resample bootstrap on the same test fold puts the ECE difference
at +0.0191 in TabNet's favor with a 95% interval of [+0.0141, +0.0240]
and P(diff > 0) = 1.0 across resamples
(`calibration_audit.json: bootstrap_ece`). Taken on its own, this would
be a textbook claim that TabNet is roughly five-and-a-half times better
calibrated than XGBoost on this task, with a statistically
unambiguous gap.

We treat that as a *hypothesis to audit*, not a finding to report, for
two reasons. First, ECE is a single scalar that compresses two distinct
properties of a probabilistic classifier — calibration and sharpness —
into one number, and in a way that can reward underconfident predictions.
Second, ECE depends on the binning scheme, and a model that produces
confidences clustered in a narrow band of the [0, 1] interval can score
low ECE by populating few bins with low-variance predictions. The §4
class-imbalance and the §10 attention-mask analysis both suggest that
TabNet might produce exactly such soft predictions on this task. We
therefore run three controls before deciding whether to report the
ECE finding.

## 8.2 Audit 1: proper scoring rules

Brier score and negative log-likelihood are *strictly proper* scoring
rules: under each, the unique optimal forecast is the true conditional
distribution, so a model cannot improve its score by softening its
predictions away from the truth. Both rules reward calibration and
sharpness jointly. We computed both on the same full-test predictions
(`calibration_audit.json: proper_scores`):

| model | Brier ↓ | NLL ↓ |
| --- | ---: | ---: |
| tuned XGBoost | 0.1791 | 0.3123 |
| refined TabNet | 0.2477 | 0.4080 |

Both rules favor XGBoost, and decisively. Brier is 38% higher for TabNet;
NLL is 31% higher. The two rules agree, which is itself a sanity check —
Brier penalizes per-class probability errors quadratically and NLL
penalizes the assigned probability on the true class logarithmically, so
they can disagree when one model is systematically off-target on a single
class while another is mildly off-target across all classes. They do not
disagree here. TabNet's predictions are worse, on the rules that price
in sharpness as well as calibration, by margins that dwarf the §6 macro-F1
gap.

This is the first piece of audit evidence: ECE prefers TabNet, but it
does so because TabNet is *underconfident*, not because it is more
correctly calibrated about events that actually happen. Entropy
diagnostics in §8.5 below sharpen that interpretation.

## 8.3 Audit 2: temperature scaling

The audit's most direct intervention is to fit a single learned scalar
*T* per model and rescale logits as `softmax(z / T)` before scoring
calibration. *T* > 1 softens predictions; *T* < 1 sharpens them. We
followed the standard protocol (Guo et al. 2017): randomly split the
full test fold in half with `seed = 42`, fit *T* on the first half by
minimizing NLL, and reported ECE/Brier/NLL on the held-out second half
(`calibration_audit.json: temperature_scaling`). Numbers below are on
this 50% eval half, *not* the full test fold; reading them against the
§8.1 full-test ECE values directly would be a category error.

| model | learned *T* | ECE pre | ECE post | Brier pre | Brier post | NLL pre | NLL post |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tuned XGBoost | 1.263 | 0.0269 | **0.0074** | 0.1837 | 0.1818 | 0.3230 | 0.3106 |
| refined TabNet | 1.012 | 0.0084 | 0.0097 | 0.2481 | 0.2481 | 0.4106 | 0.4105 |

Three things to note.

First, XGBoost's optimal *T* is 1.26: its native predictions are
*overconfident* on the eval half, and the post-scaling ECE collapses
from 0.0269 to 0.0074. After scaling, XGBoost is *better* calibrated
on this half than TabNet (post-ECE 0.0074 vs 0.0097). The ECE gap
flips sign from +0.0206 (pre) to −0.0022 (post).

Second, TabNet's optimal *T* is 1.012 — essentially the identity. TabNet
has no temperature-scaling headroom on this task; its softmax outputs
are already as well-tempered as a single global scalar can make them.
Whatever "calibration advantage" §8.1's ECE table awarded TabNet is
therefore not a fixable property of XGBoost — it is a one-line post-hoc
correction that the standard calibration toolkit applies for free.

Third, temperature scaling *preserves argmax*: dividing logits by a
positive scalar before softmax does not change which class has the
highest probability, so the macro-F1 numbers of §6 are entirely unaffected.
The audit overturns the apparent calibration win without disturbing the
hard-label result.

## 8.4 Audit 3: reading

Across the three controls — proper scoring rules, temperature scaling,
and argmax invariance — the apparent ECE advantage that motivated this
section disappears. The honest summary is:

- On ECE alone, TabNet looks better-calibrated (§8.1).
- On Brier and NLL, XGBoost is strictly better (§8.2).
- After temperature scaling, XGBoost is also better-calibrated by ECE
  (§8.3), and the macro-F1 result is unchanged.

We therefore decline to report the §8.1 ECE finding as a TabNet advantage.
What it actually measures is TabNet's tendency to produce softer
predictions, and that tendency makes its hard-label decisions worse
(§6.1) and its proper-score-rule probabilities worse (§8.2) without
giving back a real calibration win once a single learned scalar is
applied. The methodological lesson, which we state explicitly because
it is broader than this dataset, is that ECE-only calibration claims
in the tabular-DL literature should be paired with at least one proper
scoring rule and a temperature-scaling baseline before they are believed.

## 8.5 Why TabNet's predictions are soft: entropy diagnostics

A useful sanity check on the under-confidence interpretation is the mean
entropy of each model's predicted distribution, split by whether the
prediction is correct (`reports/uncertainty.json`, entropies in nats):

| model | mean H on correct preds | mean H on wrong preds |
| --- | ---: | ---: |
| tuned XGBoost | 0.192 | 0.522 |
| refined TabNet | 0.352 | 0.633 |

Both models concentrate their uncertainty appropriately: wrong
predictions carry roughly 2.5× to 3× the entropy of correct ones. But
TabNet's *correct*-prediction entropy is 83% higher than XGBoost's
(0.352 vs 0.192) — confirming that TabNet is producing softer
distributions even when it gets the answer right. That is exactly the
profile that yields low ECE without yielding better Brier or NLL: a
model that says "I think it's class X but I'm only 70% sure" will be
well-calibrated on the bin around 0.7, but it will pay for the
underconfidence under any proper scoring rule when class X really was
the answer.

## 8.6 Figures

- **Figure&nbsp;8.** `reports/figures/reliability_diagrams.png` —
  reliability diagrams for both models on the full random-split test
  fold, before temperature scaling. XGBoost's curve is noisy, with the
  largest deviation from the diagonal a downward dip in the
  0.45-confidence bin (empirical accuracy ≈ 0.37). TabNet's curve tracks
  the diagonal closely from its lowest populated bin onward — which is
  what the §8.1 ECE number rewards. The diagram does not directly
  visualize the under-confidence story; that interpretation depends on
  the proper-score-rule numbers of §8.2 and the entropy diagnostics of
  §8.5, because a reliability diagram normalizes away the per-bin
  prediction frequencies that distinguish "well-calibrated and
  confident" from "well-calibrated and soft."
- **Figure&nbsp;9.** `reports/figures/entropy_correct_vs_incorrect.png` —
  predicted-distribution entropy histograms for both models, stratified
  by whether the top-1 prediction matched the label. The shift of
  TabNet's correct-prediction histogram toward higher entropy is the
  graphical complement of §8.5.
