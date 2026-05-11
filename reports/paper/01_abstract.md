# §1 Abstract

We pre-registered the hypothesis that TabNet's sequential attention
would outperform a tuned gradient-boosted baseline on a high-volume,
low-feature severe-weather classification task (NOAA Storm Events
2023; 117,717 records after filtering to ten event classes; 8
engineered features from 51 raw columns). On the principal random-
split, iid evaluation the hypothesis is falsified along every axis
we tested: tuned XGBoost reaches macro-F1 0.9001 against refined
TabNet's 0.8513, a paired-bootstrap gap of +0.049 (95% CI [+0.044,
+0.054]; McNemar χ² = 591.6, p < 10⁻¹³⁰), with the largest per-class
losses on Hail and Thunderstorm Wind — the boundary classes the
proposal had singled out. A surface read of Expected Calibration
Error initially favored TabNet (0.0043 vs 0.0250), but a three-
control audit overturns that reading: Brier and NLL both favor
XGBoost decisively, and a single learned temperature T = 1.26
brings XGBoost below TabNet on ECE as well, without changing the
macro-F1 result. On a 2023→2024 temporal fold, TabNet's softer fit
degrades 3–4× less than XGBoost's on Hail and Thunderstorm Wind —
the one place the proposal's prediction is partially recovered. We
report this as a robust negative result against an attention-based
tabular DL method in a low-feature regime, with a methodological
caution against ECE-only calibration claims.
