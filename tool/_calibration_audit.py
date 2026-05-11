"""Four-check audit of the Framing B calibration claim.

  1. Bootstrap CI on ECE for each model and on (XGB - TabNet)
  2. Brier score + NLL for each model
  3. Temperature-scaling control on both models (split test 50/50: fit T on half, evaluate on the other)
  4. Reliability diagrams already saved -> reports/figures/reliability_diagrams.png
"""
import json
import numpy as np
from pathlib import Path
from scipy.optimize import minimize_scalar

reports_dir = Path(__file__).parent.parent / 'reports'
y_true   = np.load(reports_dir / 'tabnet_test_y_true.npy')
proba_x  = np.load(reports_dir / 'xgb_test_proba.npy').astype(np.float64)
proba_t  = np.load(reports_dir / 'tabnet_test_proba.npy').astype(np.float64)
N, K = proba_x.shape
print(f"Loaded {N} test rows, {K} classes\n")

EPS = 1e-12

def ece(probs, y, n_bins=10):
    pred = probs.argmax(1)
    conf = probs.max(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.sum() / len(y) * abs(correct[m].mean() - conf[m].mean())
    return float(e)

def brier(probs, y, K):
    onehot = np.eye(K)[y]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))

def nll(probs, y):
    p = np.clip(probs[np.arange(len(y)), y], EPS, 1.0)
    return float(-np.mean(np.log(p)))

def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def temp_scale(probs, T):
    log_p = np.log(np.clip(probs, EPS, 1.0))
    return softmax(log_p / T)

# ---- 1. Bootstrap CI on ECE ----
print("=" * 72)
print("1. Bootstrap CI on ECE (B=500, seed=42)")
print("=" * 72)
B = 500
rng = np.random.default_rng(42)
ece_x_b = np.empty(B); ece_t_b = np.empty(B); diff_b = np.empty(B)
for b in range(B):
    idx = rng.integers(0, N, N)
    ece_x_b[b] = ece(proba_x[idx], y_true[idx])
    ece_t_b[b] = ece(proba_t[idx], y_true[idx])
    diff_b[b]  = ece_x_b[b] - ece_t_b[b]
ece_x_full = ece(proba_x, y_true); ece_t_full = ece(proba_t, y_true)
print(f"  XGB    ECE: {ece_x_full:.4f}    boot mean {ece_x_b.mean():.4f}    95% CI [{np.percentile(ece_x_b, 2.5):.4f}, {np.percentile(ece_x_b, 97.5):.4f}]")
print(f"  TabNet ECE: {ece_t_full:.4f}    boot mean {ece_t_b.mean():.4f}    95% CI [{np.percentile(ece_t_b, 2.5):.4f}, {np.percentile(ece_t_b, 97.5):.4f}]")
print(f"  Diff (XGB - TabNet): mean {diff_b.mean():+.4f}    95% CI [{np.percentile(diff_b, 2.5):+.4f}, {np.percentile(diff_b, 97.5):+.4f}]")
print(f"  P(diff > 0): {(diff_b > 0).mean():.4f}")

# ---- 2. Brier and NLL ----
print("\n" + "=" * 72)
print("2. Proper-scoring rules (lower is better)")
print("=" * 72)
br_x = brier(proba_x, y_true, K); nll_x = nll(proba_x, y_true)
br_t = brier(proba_t, y_true, K); nll_t = nll(proba_t, y_true)
print(f"  XGB    Brier: {br_x:.4f}    NLL: {nll_x:.4f}")
print(f"  TabNet Brier: {br_t:.4f}    NLL: {nll_t:.4f}")
print(f"  Brier delta (XGB - TabNet): {br_x - br_t:+.4f}    (positive => TabNet better)")
print(f"  NLL   delta (XGB - TabNet): {nll_x - nll_t:+.4f}    (positive => TabNet better)")

# ---- 3. Temperature-scaling control ----
print("\n" + "=" * 72)
print("3. Temperature scaling (50/50 cal/eval split, seed=42)")
print("=" * 72)
rng2 = np.random.default_rng(42)
perm = rng2.permutation(N)
half = N // 2
cal_idx, ev_idx = perm[:half], perm[half:]

def fit_T(probs_cal, y_cal):
    def obj(T):
        return nll(temp_scale(probs_cal, T), y_cal)
    res = minimize_scalar(obj, bounds=(0.05, 10.0), method='bounded')
    return float(res.x), float(res.fun)

T_x, _ = fit_T(proba_x[cal_idx], y_true[cal_idx])
T_t, _ = fit_T(proba_t[cal_idx], y_true[cal_idx])
print(f"  Optimal T (XGB):    {T_x:.3f}")
print(f"  Optimal T (TabNet): {T_t:.3f}")

# Pre- and post-T metrics on the eval half
p_x_ev = proba_x[ev_idx]; p_t_ev = proba_t[ev_idx]; y_ev = y_true[ev_idx]
p_x_ev_T = temp_scale(p_x_ev, T_x)
p_t_ev_T = temp_scale(p_t_ev, T_t)

print("\n  Eval-half metrics (lower is better)")
print("                       ECE         Brier         NLL")
print(f"  XGB    pre-T   {ece(p_x_ev,   y_ev):.4f}      {brier(p_x_ev,   y_ev, K):.4f}        {nll(p_x_ev,   y_ev):.4f}")
print(f"  XGB    post-T  {ece(p_x_ev_T, y_ev):.4f}      {brier(p_x_ev_T, y_ev, K):.4f}        {nll(p_x_ev_T, y_ev):.4f}")
print(f"  TabNet pre-T   {ece(p_t_ev,   y_ev):.4f}      {brier(p_t_ev,   y_ev, K):.4f}        {nll(p_t_ev,   y_ev):.4f}")
print(f"  TabNet post-T  {ece(p_t_ev_T, y_ev):.4f}      {brier(p_t_ev_T, y_ev, K):.4f}        {nll(p_t_ev_T, y_ev):.4f}")

# Headline question: does temperature scaling close the gap?
ece_x_post = ece(p_x_ev_T, y_ev)
ece_t_post = ece(p_t_ev_T, y_ev)
print(f"\n  Post-T ECE gap (XGB - TabNet): {ece_x_post - ece_t_post:+.4f}    "
      f"(was {ece_x_full - ece_t_full:+.4f} pre-T)")

# ---- Persist audit results ----
audit = {
    'bootstrap_ece': {
        'B': B, 'seed': 42,
        'xgb':    {'point': ece_x_full, 'mean': float(ece_x_b.mean()), 'ci': [float(np.percentile(ece_x_b, 2.5)), float(np.percentile(ece_x_b, 97.5))]},
        'tabnet': {'point': ece_t_full, 'mean': float(ece_t_b.mean()), 'ci': [float(np.percentile(ece_t_b, 2.5)), float(np.percentile(ece_t_b, 97.5))]},
        'diff_xgb_minus_tabnet': {'mean': float(diff_b.mean()), 'ci': [float(np.percentile(diff_b, 2.5)), float(np.percentile(diff_b, 97.5))], 'p_diff_positive': float((diff_b > 0).mean())},
    },
    'proper_scores': {
        'xgb':    {'brier': br_x, 'nll': nll_x},
        'tabnet': {'brier': br_t, 'nll': nll_t},
    },
    'temperature_scaling': {
        'split_seed': 42, 'cal_frac': 0.5,
        'xgb':    {'T': T_x, 'pre_ece': ece(p_x_ev, y_ev), 'post_ece': ece_x_post, 'pre_brier': brier(p_x_ev, y_ev, K), 'post_brier': brier(p_x_ev_T, y_ev, K), 'pre_nll': nll(p_x_ev, y_ev), 'post_nll': nll(p_x_ev_T, y_ev)},
        'tabnet': {'T': T_t, 'pre_ece': ece(p_t_ev, y_ev), 'post_ece': ece_t_post, 'pre_brier': brier(p_t_ev, y_ev, K), 'post_brier': brier(p_t_ev_T, y_ev, K), 'pre_nll': nll(p_t_ev, y_ev), 'post_nll': nll(p_t_ev_T, y_ev)},
        'gap_pre':  ece_x_full - ece_t_full,
        'gap_post': ece_x_post - ece_t_post,
    },
    'note': 'Bootstrap is on the full test set. Temperature is fit on a 50% calibration half via NLL minimization on softmax(log p / T); the 50% eval half is reported. T>1 softens, T<1 sharpens.',
}
out = reports_dir / 'calibration_audit.json'
out.write_text(json.dumps(audit, indent=2))
print(f"\nSaved -> {out}")
