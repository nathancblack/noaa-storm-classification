# §13 Conclusion

We pre-registered the hypothesis that TabNet's sequential attention
would outperform a tuned gradient-boosted baseline on NOAA Storm
Events 2023 — and on the principal random-split, iid evaluation, the
hypothesis is falsified along every axis we tested. Tuned XGBoost
beats refined TabNet by +0.049 macro-F1 (95% bootstrap CI [+0.044,
+0.054]; McNemar χ² = 591.6, p < 10⁻¹³⁰), and the direction is
reversed on exactly the two boundary classes the proposal singled
out. A surface reading of Expected Calibration Error initially
favored TabNet, but a three-control audit — proper scoring rules,
temperature scaling, and argmax invariance — overturned that
reading: XGBoost is strictly better under Brier and NLL, and a
single learned scalar T = 1.26 makes XGBoost the better-calibrated
model on ECE as well. The one nuance we report honestly is the
2023→2024 temporal fold, where TabNet's softer fit degrades 3–4×
less than XGBoost's on Hail and Thunderstorm Wind, partially
recovering the proposal's per-class prediction. The broader
methodological takeaway is the audit itself: ECE-only calibration
claims in the tabular-DL literature should be paired with at least
one proper scoring rule and a temperature-scaling baseline before
they are believed.
