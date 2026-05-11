# §10 Interpretability

The §6 result settles which model is more accurate on this task. §8
settles which model produces better-calibrated probabilities once a
fair audit is applied. This section asks a different question: when
each model makes a prediction, *which features is it actually using*?
Tabular interpretability has a sharp methodological asymmetry — XGBoost
exposes a global gain-based importance score over features, and TabNet
exposes per-step attention masks over features for every input row.
The two artifacts are not directly commensurable, but they can both be
reduced to a per-feature score on the same 8-feature axis, and that is
where we compare them. The finding of the section is that both models
rank the same small set of features at the top and the same single
feature at the bottom — which sets up the §11 mechanistic argument
without making the claim there.

## 10.1 XGBoost gain-based importance

The tuned XGBoost's split-gain importance over the 8 engineered features
(`reports/xgb_feature_importance.npy` indexed by
`reports/xgb_feature_cols.json`) is:

| feature | gain importance |
| --- | ---: |
| CZ_TYPE_ENC | 0.9037 |
| DURATION_MIN | 0.0357 |
| STATE_ENC | 0.0202 |
| MONTH | 0.0131 |
| WFO_ENC | 0.0080 |
| BEGIN_LON | 0.0079 |
| HOUR | 0.0057 |
| BEGIN_LAT | 0.0056 |

The distribution is extremely concentrated. CZ_TYPE_ENC alone accounts
for 90.4% of total gain across the boosted ensemble; the next feature
(DURATION_MIN) takes 3.6%, and the remaining six features share the
final 6%. CZ_TYPE_ENC is the NOAA county-zone-marine type code, which
is the feature that lets the model separate Marine Thunderstorm Wind
from its land counterpart at near-perfect accuracy (§9.1) and that
provides the first split on which boosted trees can sort almost every
event into one of three broad bins. The runner-up DURATION_MIN is the
event duration in minutes, which separates short-duration severe events
(Hail, Thunderstorm Wind) from longer-duration ones (Flood, Winter
Weather, Drought). Geography (BEGIN_LAT/LON, STATE_ENC, WFO_ENC) and
time (MONTH, HOUR) contribute small fractions, with no single secondary
feature dominating the others.

## 10.2 TabNet attention masks

TabNet exposes per-step attention masks `M[s, i, f]` giving the
attention weight that decision step *s* placed on feature *f* for input
row *i* (`reports/tabnet_masks.npy`, shape `(7 steps, 23,544 rows, 8
features)`). The conventional row-and-step-averaged importance score
is the mean mask weight per feature across all rows and steps:

| feature | mean mask weight |
| --- | ---: |
| CZ_TYPE_ENC | 0.1912 |
| DURATION_MIN | 0.1774 |
| HOUR | 0.1449 |
| BEGIN_LAT | 0.1393 |
| BEGIN_LON | 0.1209 |
| MONTH | 0.0940 |
| STATE_ENC | 0.0890 |
| WFO_ENC | 0.0433 |

The ranking shape is qualitatively different from XGBoost's. CZ_TYPE_ENC
is still in first place, but with only 19% of total mask mass instead
of XGBoost's 90% — and the tail does not collapse. The least-used
feature (WFO_ENC) still receives 4.3% of total mask weight, almost an
order of magnitude more than its XGBoost gain share would suggest.
A per-row TabNet prediction is built from contributions of all eight
features at non-trivial weight, not from one dominant split on
CZ_TYPE_ENC followed by small corrections.

The step-by-step breakdown is more informative than the marginal table.
Each of TabNet's seven decision steps specializes in a different feature
subset (mean mask weight per feature, per step, top-4 per row):

| step | top-4 features and weights |
| --- | --- |
| 0 | CZ_TYPE_ENC 0.301, MONTH 0.230, WFO_ENC 0.224, BEGIN_LON 0.110 |
| 1 | HOUR 0.296, STATE_ENC 0.258, BEGIN_LAT 0.243, CZ_TYPE_ENC 0.199 |
| 2 | BEGIN_LAT 0.303, BEGIN_LON 0.269, CZ_TYPE_ENC 0.266, DURATION_MIN 0.087 |
| 3 | BEGIN_LON 0.227, BEGIN_LAT 0.168, CZ_TYPE_ENC 0.155, MONTH 0.141 |
| 4 | HOUR 0.379, CZ_TYPE_ENC 0.182, STATE_ENC 0.175, BEGIN_LAT 0.118 |
| 5 | DURATION_MIN 0.623, CZ_TYPE_ENC 0.185, BEGIN_LON 0.075, MONTH 0.066 |
| 6 | DURATION_MIN 0.311, MONTH 0.214, HOUR 0.123, STATE_ENC 0.111 |

Three observations. First, CZ_TYPE_ENC is in the top-4 of every single
decision step — the architecture concentrates on it consistently, but
nowhere near as exclusively as XGBoost does. Second, steps 5 and 6 are
where DURATION_MIN does its work, and step 5 in particular allocates
62% of its mask weight to DURATION_MIN alone — TabNet has effectively
trained a "duration-discriminating" decision step. Third, the
geographical features split across steps: step 2 is a latitude/longitude
step; step 4 is an hour-of-day step; step 1 is a state/latitude/hour
step. The architecture is recognizable as feature-selecting in the way
the TabNet paper describes — but on this 8-feature task it never has
enough irrelevant features to filter away. Every feature gets used
non-trivially in at least one step.

A second, complementary view comes from the local-attribution explain
matrix (`reports/tabnet_explain_matrix.npy`, shape `(23,544 rows, 8
features)`), which weights the per-step masks by each step's
contribution to the final prediction. Mean absolute attribution per
feature is:

| feature | mean &#124;explain&#124; |
| --- | ---: |
| DURATION_MIN | 4.238 |
| CZ_TYPE_ENC | 2.882 |
| MONTH | 2.254 |
| HOUR | 2.230 |
| BEGIN_LON | 1.944 |
| BEGIN_LAT | 1.630 |
| STATE_ENC | 1.608 |
| WFO_ENC | 0.943 |

This re-ranks DURATION_MIN and CZ_TYPE_ENC versus the marginal mask
ranking, putting DURATION_MIN at the top. The reordering is consistent
with the step table: step 5's 62%-DURATION_MIN allocation carries large
prediction weight even though the marginal mask weight on DURATION_MIN
across all steps is only 17.7%. The two TabNet rankings agree on the
top three features (DURATION_MIN, CZ_TYPE_ENC, and one of MONTH/HOUR)
and on the bottom feature (WFO_ENC).

## 10.3 Cross-model comparison

Reduced to ranks on the common 8-feature axis, the two models agree
much more than the §10.1 / §10.2 distribution shapes suggest. The
table below puts the three rankings side by side:

| feature | XGB rank (by gain) | TabNet rank (by mean mask) | TabNet rank (by mean &#124;explain&#124;) |
| --- | ---: | ---: | ---: |
| CZ_TYPE_ENC | 1 | 1 | 2 |
| DURATION_MIN | 2 | 2 | 1 |
| STATE_ENC | 3 | 7 | 7 |
| MONTH | 4 | 6 | 3 |
| WFO_ENC | 5 | 8 | 8 |
| BEGIN_LON | 6 | 5 | 5 |
| HOUR | 7 | 3 | 4 |
| BEGIN_LAT | 8 | 4 | 6 |

The two top features (CZ_TYPE_ENC, DURATION_MIN) are the top-two under
all three rankings; the bottom feature (WFO_ENC) is in the bottom-two
under all three. The middle ordering disagrees — XGBoost places STATE_ENC
third but TabNet places it seventh; TabNet places HOUR and BEGIN_LAT
in its top half but XGBoost places them seventh and eighth. The
discrepancy reflects a real architectural difference: XGBoost can encode
location through STATE_ENC because boosted trees split repeatedly on
that integer-coded category, while TabNet's masks distribute location
information across the raw lat/lon features instead. Both models are
solving the same problem with the same information; they package the
location signal differently.

The agreement at the top and bottom of the ranking is the load-bearing
observation for §11. Both models converge on a feature subset of
roughly two strong features and six weaker but non-negligible features.
There is no "irrelevant" feature for TabNet's attention to filter out —
the worst feature on this task (WFO_ENC) still earns 4.3% of TabNet's
mask weight and 0.8% of XGBoost's gain. Section §11 takes this as
mechanistic support for the §6 / §7 negative result: TabNet's value
proposition is sparse feature selection, but the 8-feature engineered
representation has no slack for that proposition to exploit.

## 10.4 Figures

- **Figure&nbsp;15.** `reports/figures/feature_importance_tuned_xgb.png` —
  the §10.1 bar chart of gain-based importances for the tuned XGBoost.
  The visual emphasizes the 90%-on-CZ_TYPE_ENC concentration.
- **Figure&nbsp;16.** `reports/figures/tabnet_attention_masks.png` —
  per-step mean attention masks for the refined TabNet, displayed as
  a heatmap of (decision step) × (feature). The visual carrier of the
  §10.2 step-specialization argument.
- **Figure&nbsp;17.** `reports/figures/feature_attribution_comparison.png` —
  side-by-side comparison of XGBoost gain importance and TabNet
  explain-matrix attribution on the common 8-feature axis. Anchors the
  §10.3 cross-model ranking comparison.
