# §2 Introduction

A persistent question in applied tabular machine learning is whether
attention-based deep architectures can outperform gradient-boosted
tree ensembles on the kind of structured, finite-feature, high-volume
classification problems that account for most real-world tabular
workloads. The two methodological poles are well-defined: XGBoost
(Chen & Guestrin 2016) is the de-facto baseline, with split-finding
heuristics that have been refined over a decade of competition and
production use; TabNet (Arik & Pfister 2021) is the most prominent
attention-based alternative, with a sequential decision process that
learns sparse per-instance feature masks at each of several steps.
The broader literature reviewing this comparison has reached mixed
verdicts: Grinsztajn et al. (2022) report that tree-based methods
remain strictly better on most tabular benchmarks, while Shwartz-Ziv
and Armon (2022) caution that no single tabular DL method
consistently beats well-tuned trees across heterogeneous tasks.
Our work sits inside that debate and contributes a single high-
volume, low-feature, multi-class operational case study with
statistical testing, a calibration audit, and a temporal-shift
qualification.

The dataset is the NOAA Storm Events database (2023–2024), filtered
to the ten most frequent event-type classes — Drought, Excessive
Heat, Flash Flood, Flood, Hail, Heat, High Wind, Marine Thunderstorm
Wind, Thunderstorm Wind, and Winter Weather — for 117,717 training-
evaluation records and a 55,061-record 2024 hold-out. The raw archive
exposes 51 columns; after feature engineering (§4) the working set
is 8 features mixing categorical event-context codes (CZ_TYPE_ENC,
STATE_ENC, WFO_ENC), continuous time-of-event variables (HOUR,
MONTH, DURATION_MIN), and geographic coordinates (BEGIN_LAT,
BEGIN_LON). We pre-registered the hypothesis that TabNet's sparse
attention would outperform a tuned XGBoost baseline on this task,
with the largest gains expected on the two boundary classes
Hail vs Thunderstorm Wind whose feature distributions overlap most.

The paper makes three contributions.

**A statistically tested comparison.** We compare a tuned XGBoost
baseline against a two-stage tuned TabNet (coarse sweep over
`n_d, n_steps, lr` then a refined sweep over `gamma, batch_size`
validated with k-fold CV) on both a random-split iid test fold and
a 2023→2024 temporal hold-out. We report macro-F1, macro
one-vs-rest AUC, paired bootstrap on the macro-F1 difference, and
McNemar on per-row correctness. The random-split result falsifies
the pre-registered hypothesis decisively: XGBoost 0.9001 vs TabNet
0.8513 (bootstrap diff +0.049, 95% CI [+0.044, +0.054]; McNemar
χ² = 591.6, p < 10⁻¹³⁰), with XGBoost ahead on every per-class F1
including Hail (+0.113) and Thunderstorm Wind (+0.058).

**A calibration audit.** A naive read of Expected Calibration Error
favors TabNet (0.0043 vs 0.0250). We treat that as a hypothesis to
audit rather than a finding to report and run three controls:
proper scoring rules (Brier and NLL, both of which favor XGBoost),
temperature scaling (a single learned T = 1.26 on XGBoost reduces
its ECE below TabNet's on a held-out half), and argmax invariance
(temperature scaling preserves the macro-F1 ordering). The audit
overturns the ECE advantage and yields a methodological caution
that we state explicitly because it generalizes beyond this
dataset: ECE-only calibration claims should be paired with at
least one proper scoring rule and a temperature-scaling baseline.

**A temporal-split qualification on the proposal's boundary
classes.** On the 2023→2024 temporal fold, TabNet beats XGBoost on
Hail (+0.037 F1) and Thunderstorm Wind (+0.017 F1), and TabNet's
per-class F1 drop from random to temporal is 3–4× smaller than
XGBoost's on those two classes. The aggregate temporal macro-F1
still favors XGBoost, but on the *specific* classes the proposal
named, TabNet generalizes more robustly to next-year data. We
report this as a substantive qualification of the headline negative
result, not as a thesis reversal — one annual fold is too thin a
base for a domain-shift claim, and §12 makes that limitation
explicit.

The remainder of the paper is organized as follows. §3 places the
work in context. §4 and §5 describe the data and methods. §6 and §7
report the headline experiments and statistical tests. §8 is the
calibration audit. §9 covers error analysis and the temporal-split
qualification. §10 reads each model's feature attributions. §11
discusses what mechanism is consistent with these results, and §12
states the limitations that bound the claim.
