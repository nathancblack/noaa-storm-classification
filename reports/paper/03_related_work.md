# §3 Related Work

Our work touches four threads in the tabular-ML literature.

**Attention-based tabular deep learning.** TabNet (Arik & Pfister
2021) is the canonical sequential-attention architecture for
structured data. At each of *N* decision steps an attentive
transformer learns a sparse mask over features, and a sparsemax
nonlinearity together with a sparsity prior (`gamma`) push the
masks toward instance-adaptive feature subsets that are approximately
disjoint across steps. The architectural promise is interpretable,
selection-sparse inference; the empirical promise is competitive or
better accuracy than gradient-boosted baselines on tabular tasks.
A range of follow-on attention-based tabular methods (FT-Transformer,
SAINT, NODE) make broadly similar claims but inherit the same
selection-driven motivation.

**Gradient-boosted baselines.** XGBoost (Chen & Guestrin 2016) is
the long-standing reference for tabular classification. Its weighted-
quantile sketch and regularized split-finding heuristics, together
with histogram-based extensions in LightGBM and CatBoost, set the
accuracy bar against which most tabular DL methods are reported.

**The tabular DL versus tree-based debate.** Two recent surveys
synthesize this debate directly. Grinsztajn et al. (2022) benchmark
a wide range of tabular DL methods against tree ensembles on
heterogeneous medium-sized datasets and report that tree-based
methods remain strictly better on most. Shwartz-Ziv and Armon (2022)
reach a complementary conclusion: no tabular DL method consistently
beats well-tuned trees across tasks, and the gap is most pronounced
when the input feature set is small and free of irrelevant features.
Our §11.1 mechanism claim is downstream of this finding — the
8-feature regime we work in is exactly the regime Shwartz-Ziv and
Armon flag as least favorable for attention-based methods.

**Calibration of classifier probabilities.** Guo et al. (2017) is the
standard reference on neural-classifier miscalibration and
temperature scaling. The paper showed that modern deep classifiers
are systematically overconfident, that Expected Calibration Error
(ECE) is a useful but imperfect scalar diagnostic, and that a single
learned temperature is a strong post-hoc correction. Our §8 audit
treats ECE, Brier, NLL, and temperature scaling jointly in the
spirit of that paper, and the methodological caution we issue —
that ECE-only calibration claims are fragile in the presence of
under-confident predictions — generalizes a point that Guo et al.
already make about the *over*-confident case.

We are not aware of prior work that pairs a pre-registered TabNet-vs-
XGBoost comparison on a NOAA Storm Events working set with a
statistical-test-anchored macro-F1 result, a three-control
calibration audit, and a temporal-shift qualification on named
boundary classes. The contribution of this paper is to assemble
those four pieces of evidence into a single auditable case study.
