# §11 Discussion

The empirical sections leave three things to explain: why the random-
split macro-F1 result came out against the pre-registered hypothesis,
why the 2023→2024 temporal split partially recovers it on exactly the
two named boundary classes, and what to do with the moderate cross-model
error independence we measured in §9.2. We address those three in
order. The first two are mechanism claims about *this* dataset and
this 8-feature working set; the third is a follow-up flag, not a
conclusion of the paper.

## 11.1 Why XGBoost wins the random split

TabNet's architectural promise is sparse, instance-adaptive feature
selection: at each decision step, the attentive transformer learns to
attend to a small subset of features and to ignore the rest, with the
sparsity prior `gamma` regularizing the masks to be approximately
disjoint across steps (Arik & Pfister 2021). The mechanism is most
valuable when the input has many redundant, noisy, or irrelevant
features — that is, when *selection itself* is the hard part of the
problem. On the NOAA Storm Events working set we use here, selection
is not the hard part. The data ingest enters the modeling pipeline as
51 raw columns, but the §4 feature-engineering step compresses those
to 8 features in which every column carries non-trivial signal: §10.2
shows that even the worst feature (WFO_ENC) receives 4.3% of TabNet's
mask weight and 0.8% of XGBoost's gain, and the next-worst features
(STATE_ENC, MONTH, HOUR, BEGIN_LAT/LON) all clear 1% of XGBoost gain
and 8% of TabNet mask mass. There is no idle feature for attention to
filter away.

In that regime the gradient-boosted baseline is at its strongest.
XGBoost's split-finding enumerates near-optimal partitions of the input
space by greedy gain maximization, and with only 8 features the
greedy search is close to exhaustive. §10.1's gain distribution makes
the corollary visible: 90% of total gain is on CZ_TYPE_ENC and the
next 4% is on DURATION_MIN, so XGBoost finds and exploits the two
features that matter most before TabNet's seven decision steps even
have a chance to specialize. The negative result of §6 / §7 is not
that TabNet's attention failed — §10.2 shows it learned coherent
step-specialized masks — but that the attention mechanism's
comparative advantage requires a feature regime this dataset does
not provide.

## 11.2 Why the temporal qualification is consistent with the mechanism

The §9.3 temporal result — TabNet beating XGBoost on Hail (+0.037) and
Thunderstorm Wind (+0.017) under 2023→2024 shift, while degrading
roughly 3–4× less than XGBoost on those classes year-over-year — is the
one place the pre-registered hypothesis is partially recovered. The
§8 calibration audit makes a mechanistic reading available: TabNet
produces *softer*, less-confident decision boundaries (§8.2 Brier and
NLL, §8.5 entropy diagnostics). A tight, sharply-confident fit to 2023
training data is exactly the kind of fit that pays the largest penalty
under a 2024 distribution shift in the boundary regions of decision
space, and the two boundary classes the proposal singled out — Hail
and Thunderstorm Wind — are visibly the classes where XGBoost's
random-to-temporal drop is most asymmetric (−0.207 and −0.120, against
TabNet's −0.057 and −0.045). We offer this as a hypothesis of
mechanism, not a proven claim. One annual fold is not enough to anchor
a domain-shift result, and §12 makes that limitation explicit. But
the per-class direction is the one the §8 audit predicts, and it is
worth flagging rather than burying.

## 11.3 Cross-model error independence and the ensemble follow-up

§9.2 measured Cohen's κ = 0.47 on per-row correctness between the two
final models. If errors were perfectly correlated, κ would be 1 and
no ensemble could help; if errors were perfectly independent at the
observed individual accuracy rates, κ would be 0 and an oracle
ensemble could push accuracy close to 1 − (both-wrong rate). We
observe moderate correlation: 3,191 of 23,544 test rows are
correctable by exactly one of the two models. A confidence-weighted
ensemble would in principle close some of that gap, with Hail (790
both-wrong rows, the largest such cell) the class where the headroom
is smallest. We do not run that experiment in this paper — it would
be a separate modeling pass, and the §6 / §7 contribution does not
depend on it. We flag it as the most defensible follow-up: the per-
class overlap structure of §9.2 is exactly the diagnostic that would
inform whether the ensemble is worth the engineering cost.
