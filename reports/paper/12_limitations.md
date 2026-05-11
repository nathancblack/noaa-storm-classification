# §12 Limitations

We flag four limitations that bound the scope of the claims above.

**Feature count.** The headline comparison is run on an 8-feature
engineered working set derived from the 51 raw columns of the NOAA
Storm Events archive. §11.1 argues that this low-feature regime is
itself a reason TabNet's attention mechanism has nothing to filter
away; the corollary is that our negative result does not transfer
mechanically to wider tabular tasks. A study with 50 or 500 features,
of which only a handful carry signal, would put TabNet's selection
mechanism back in the regime where it is designed to add value. We do
not claim our result generalizes to that setting.

**Single-year temporal fold.** The 2023→2024 split is one annual
shift. The per-class temporal flip on Hail and Thunderstorm Wind
(§9.3) is suggestive of TabNet's softer fit generalizing more
robustly across years on those specific classes, but a single fold is
not enough to support a domain-adaptation claim. A leave-one-year-out
study across multiple years of NOAA archives would be the right design
to harden or falsify the §11.2 hypothesis.

**No ensemble baseline.** §9.2 measured Cohen's κ = 0.47 between the
two models' per-row correctness and §11.3 argued this leaves room for
a confidence-weighted ensemble to improve over either alone. We did
not evaluate one. The ensemble follow-up is the most defensible
extension of this work; we flag it as such rather than running it
incompletely.

**ECE binning sensitivity.** Expected Calibration Error depends on
the choice of bin count and bin spacing. We used 10 equal-width bins
in [0, 1] over the maximum-class confidence (Guo et al. 2017). This
is the most common convention in the calibration literature and was
fixed *before* the §8 audit ran. The audit's load-bearing numbers —
Brier and NLL in §8.2, post-temperature-scaling ECE in §8.3 — are not
materially affected by bin choice, which is part of why the audit is
robust where ECE alone is not.
