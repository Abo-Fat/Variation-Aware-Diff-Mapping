"""
Phase 1.3 — kappa calibration and final N_th^{2b} determination.

Reads Results/col_mc_2bit.npz and Results/calibration_params.npz (initial
kappa estimate from error_prob.py) and:

1. Verifies that combos with the same n_eq = n1 + kappa_2*n2 + kappa_3*n3
   share a common p_err curve ("collapse check").
2. If the collapse is insufficient, minimises the scatter around the master
   curve by a small numerical search over (kappa_2, kappa_3).
3. Re-derives N_th^{2b} from the refined kappa values.
4. Saves Results/kappa_final.npz with kappa_2, kappa_3, N_th^{2b},
   and a p_err master curve.

Collapse metric used for fitting:
  Loss = mean variance of p_err samples at the same n_eq bin.
  Minimised over (kappa_2, kappa_3) using scipy.optimize.minimize.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.optimize import minimize
import config_inno2 as cfg


# ---------------------------------------------------------------------------
# Helper: build p_err curve from (combos, Y, kappa_2, kappa_3)
# ---------------------------------------------------------------------------

def _build_master_curve(combos_arr, Y, kappa_2, kappa_3, epsilon=None):
    """
    Bin combos by rounded n_eq and compute p_err vs n_eq.

    Returns
    -------
    neq_grid : ndarray  unique n_eq bin centres
    p_err    : ndarray  error probability per bin boundary
    N_th_2b  : float
    """
    if epsilon is None:
        epsilon = cfg.EPSILON

    n1 = combos_arr[:, 0].astype(float)
    n2 = combos_arr[:, 1].astype(float)
    n3 = combos_arr[:, 2].astype(float)
    n_eq = n1 + kappa_2 * n2 + kappa_3 * n3

    n_eq_rounded = np.round(n_eq).astype(int)
    bins = {}
    for k, neq_k in enumerate(n_eq_rounded):
        bins.setdefault(neq_k, []).append(Y[k])

    sorted_keys = sorted(bins.keys())
    K_bins = len(sorted_keys)
    neq_grid = np.array(sorted_keys, dtype=float)

    p_err = np.zeros(K_bins - 1, dtype=float)
    for i in range(K_bins - 1):
        Y_curr = np.concatenate(bins[sorted_keys[i]])
        Y_next = np.concatenate(bins[sorted_keys[i + 1]])
        mu_c, mu_n = Y_curr.mean(), Y_next.mean()
        if mu_n <= mu_c:
            p_err[i] = 0.5
            continue
        tau = 0.5 * (mu_c + mu_n)
        p_err[i] = (np.mean(Y_curr > tau) + np.mean(Y_next < tau)) / 2.0

    neq_pairs = neq_grid[:K_bins - 1]
    valid = np.where(p_err <= epsilon)[0]
    N_th_2b = float(neq_pairs[valid[-1]]) if len(valid) > 0 else 0.0

    return neq_pairs, p_err, N_th_2b


# ---------------------------------------------------------------------------
# Collapse quality metric
# ---------------------------------------------------------------------------

def _collapse_loss(kappa_vec, combos_arr, Y, n_samples=None, n_bins=30):
    """
    Measure how well p_err data "collapses" onto a single curve.

    For a given (kappa_2, kappa_3), assign each combo an n_eq value,
    bin them, and compute the intra-bin variance of median Q-sum.
    Lower = better collapse.

    We use a lightweight proxy: for each bin, compute std of mean(Y[k])
    across members; sum these stds weighted by bin size.
    """
    kappa_2, kappa_3 = float(kappa_vec[0]), float(kappa_vec[1])
    if kappa_2 < 1.0 or kappa_3 < kappa_2:
        return 1e9   # enforce physics ordering: kappa_3 > kappa_2 > 1

    n1 = combos_arr[:, 0].astype(float)
    n2 = combos_arr[:, 1].astype(float)
    n3 = combos_arr[:, 2].astype(float)
    n_eq = n1 + kappa_2 * n2 + kappa_3 * n3

    # Only use combos where at least one of n2, n3 > 0 (otherwise invariant)
    mixed_mask = (n2 > 0) | (n3 > 0)
    if mixed_mask.sum() < 10:
        return 1e9

    n_eq_m = n_eq[mixed_mask]
    Y_m    = Y[mixed_mask]

    # Bin into n_bins equal-width bins
    lo, hi = n_eq_m.min(), n_eq_m.max()
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, n_bins + 1)
    bin_idx = np.digitize(n_eq_m, edges[1:])   # 0..n_bins-1

    loss = 0.0
    total_weight = 0
    for b in range(n_bins):
        mask_b = bin_idx == b
        if mask_b.sum() < 2:
            continue
        means_b = Y_m[mask_b].mean(axis=1)   # per-combo mean Q-sum
        loss   += mask_b.sum() * means_b.std()
        total_weight += mask_b.sum()

    return loss / max(total_weight, 1)


# ---------------------------------------------------------------------------
# Main calibration routine
# ---------------------------------------------------------------------------

def calibrate_kappa(combos_arr, Y, kappa_2_init, kappa_3_init,
                    epsilon=None, verbose=True):
    """
    Refine kappa_2, kappa_3 to improve collapse, then compute N_th^{2b}.

    Parameters
    ----------
    combos_arr    : ndarray [K,3]
    Y             : ndarray [K, N_mc]
    kappa_2_init  : float
    kappa_3_init  : float

    Returns
    -------
    kappa_2  : float  refined
    kappa_3  : float  refined
    N_th_2b  : float
    neq_grid : ndarray  n_eq values for master curve
    p_err    : ndarray  master curve
    """
    if epsilon is None:
        epsilon = cfg.EPSILON

    # Initial collapse quality
    loss_init = _collapse_loss([kappa_2_init, kappa_3_init], combos_arr, Y)
    if verbose:
        print(f"  Initial kappa_2={kappa_2_init:.4f}, kappa_3={kappa_3_init:.4f}, "
              f"collapse_loss={loss_init:.6f}")

    # Numerical optimisation with bounds
    # kappa_2 in [1.0, 5.0], kappa_3 in [kappa_2, 10.0]
    bounds = [(1.0, 8.0), (1.0, 15.0)]
    result = minimize(
        _collapse_loss,
        x0=[kappa_2_init, kappa_3_init],
        args=(combos_arr, Y),
        method='Nelder-Mead',
        options={'xatol': 1e-3, 'fatol': 1e-6, 'maxiter': 400, 'disp': False},
    )

    kappa_2_opt = float(result.x[0])
    kappa_3_opt = float(result.x[1])
    loss_opt    = float(result.fun)

    # Enforce ordering
    kappa_2_opt = max(kappa_2_opt, 1.0)
    kappa_3_opt = max(kappa_3_opt, kappa_2_opt)

    if verbose:
        print(f"  Optimised kappa_2={kappa_2_opt:.4f}, kappa_3={kappa_3_opt:.4f}, "
              f"collapse_loss={loss_opt:.6f}")

    # Decide whether to accept the refined values
    if loss_opt < loss_init:
        kappa_2, kappa_3 = kappa_2_opt, kappa_3_opt
        if verbose:
            print("  -> Using optimised kappa values (improved collapse)")
    else:
        kappa_2, kappa_3 = kappa_2_init, kappa_3_init
        if verbose:
            print("  -> Keeping initial kappa values (no improvement)")

    # Build final master curve
    neq_grid, p_err, N_th_2b = _build_master_curve(
        combos_arr, Y, kappa_2, kappa_3, epsilon)

    if verbose:
        print(f"  Final N_th^2b = {N_th_2b:.2f}  (epsilon={epsilon})")

    return kappa_2, kappa_3, N_th_2b, neq_grid, p_err


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'Results')

    # Load MC data
    data2      = np.load(os.path.join(out_dir, 'col_mc_2bit.npz'))
    combos_arr = data2['combos']
    Y_2b       = data2['Y']

    # Load initial kappa from error_prob.py
    cal = np.load(os.path.join(out_dir, 'calibration_params.npz'))
    kappa_2_init = float(cal['kappa_2'])
    kappa_3_init = float(cal['kappa_3'])
    N_th_1b      = int(cal['N_th_1bit'])

    print("=" * 60)
    print("Phase 1.3: kappa calibration")
    kappa_2, kappa_3, N_th_2b, neq_grid, p_err = calibrate_kappa(
        combos_arr, Y_2b, kappa_2_init, kappa_3_init, verbose=True)

    # Save results
    out_kappa = os.path.join(out_dir, 'kappa_final.npz')
    np.savez(out_kappa,
             kappa_2=kappa_2,
             kappa_3=kappa_3,
             N_th_1bit=N_th_1b,
             N_th_2bit=N_th_2b,
             neq_grid=neq_grid,
             p_err_2b=p_err)
    print(f"\nFinal calibration saved -> {out_kappa}")
    print(f"  kappa_2  = {kappa_2:.4f}")
    print(f"  kappa_3  = {kappa_3:.4f}")
    print(f"  N_th^1b  = {N_th_1b}")
    print(f"  N_th^2b  = {N_th_2b:.2f}")

    # Also update calibration_params.npz with refined values
    out_cal = os.path.join(out_dir, 'calibration_params.npz')
    np.savez(out_cal,
             N_th_1bit=N_th_1b,
             N_th_2bit=N_th_2b,
             kappa_2=kappa_2,
             kappa_3=kappa_3)
    print(f"  Updated -> {out_cal}")
    print("Done.")


if __name__ == '__main__':
    main()
