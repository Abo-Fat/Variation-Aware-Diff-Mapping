"""
Proposed1 SDR Mapping Optimizer.

Objective: maximise  sum_c  log P_c
where      log P_c = sum_{bitplane b, side s} log(1 - p_err_{b,s}(column c))

p_err is computed analytically under all-ones input (every row activated),
using Gaussian column-sum distributions derived from FeFET device statistics.
Column independence holds — each column's charge integration is independent —
so maximising sum_c log P_c is equivalent to maximising each P_c independently.

Algorithm: greedy coordinate descent (same structure as mapping_optimizer.py).
  1. Initialise from conventional (non-redundant binary) mapping.
  2. Maintain count arrays: n_on_p/m for 1-bit planes, n_state_p/m for 2-bit.
  3. Repeat until convergence:
       For each row i (shuffled), for each column c:
         Enumerate all SDR candidates for W[i,c].
         For each candidate, compute delta counts (O(1)) and new log P_c (O(KB+KQ)).
         Accept if log P_c strictly increases.
  4. Return compatible result dict (same schema as mapping_optimizer.py output).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import config_inno2 as cfg

from decompose    import (build_sdr_lut,
                          conventional_decompose,
                          sdr_to_planes,
                          verify_reconstruction)
from column_stats import (count_1bit_columns,
                          count_2bit_eq_columns,
                          compute_J_c,
                          compute_S_c,
                          compute_S_c_uniform,
                          compute_objective_corrected,
                          compute_mean_n1b_per_plane,
                          compute_mean_neq_per_plane,
                          load_calibration)
from analytical_cal import compute_device_stats
from col_log_prob   import (build_p_err_table_1bit,
                            col_log_prob)


# ---------------------------------------------------------------------------
# Count array helpers
# ---------------------------------------------------------------------------

def _init_count_arrays(D_B, D_Q, KB, KQ, N):
    """
    Build (n_on_p, n_on_m, n_state_p, n_state_m) from current SDR arrays.

    n_on_p[m, c]       = count of rows i where D_B[m,i,c] > 0  (B_plus active)
    n_on_m[m, c]       = count of rows i where D_B[m,i,c] < 0  (B_minus active)
    n_state_p[t, c, s] = count of rows where Q_plus[t][i,c] == s
    n_state_m[t, c, s] = count of rows where Q_minus[t][i,c] == s
    """
    n_on_p    = np.zeros((KB, N), dtype=np.int32)
    n_on_m    = np.zeros((KB, N), dtype=np.int32)
    n_state_p = np.zeros((KQ, N, 4), dtype=np.int32)
    n_state_m = np.zeros((KQ, N, 4), dtype=np.int32)

    for m in range(KB):
        n_on_p[m] = (D_B[m] > 0).sum(axis=0)
        n_on_m[m] = (D_B[m] < 0).sum(axis=0)

    for t in range(KQ):
        Q_p = np.maximum(D_Q[t], 0)   # [M, N]
        Q_m = np.maximum(-D_Q[t], 0)
        for s in range(4):
            n_state_p[t, :, s] = (Q_p == s).sum(axis=0)
            n_state_m[t, :, s] = (Q_m == s).sum(axis=0)

    return n_on_p, n_on_m, n_state_p, n_state_m


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

def optimize_mapping_proposed1(W, cal=None, lut=None, KB=None, KQ=None,
                               max_iter=10, verbose=True, seed=None):
    """
    Proposed1 column-wise SDR mapping optimizer.

    Parameters
    ----------
    W        : ndarray [M, N]  integer weights in [-W_MAX, W_MAX]
    lut      : pre-built SDR LUT (built if None)
    KB, KQ   : precision config (defaults from cfg)
    max_iter : max coordinate-descent sweeps
    verbose  : print per-iteration progress
    seed     : int, RNG seed for row shuffle order

    Returns
    -------
    result : dict  (same schema as optimize_mapping() output)
    """
    if KB is None: KB = cfg.KB
    if KQ is None: KQ = cfg.KQ

    W = np.asarray(W, dtype=np.int64)
    M, N = W.shape

    if lut is None:
        if verbose: print("  Building SDR LUT...")
        lut = build_sdr_lut(KB, KQ)

    # Analytical device stats — fast, no MC
    if verbose: print("  Computing analytical device statistics...")
    stats_1bit, stats_2bit = compute_device_stats(verbose=False)

    # Precompute 1-bit p_err lookup table
    p_err_1b_table = build_p_err_table_1bit(stats_1bit, M)

    # Initialise from conventional mapping
    D_B, D_Q = conventional_decompose(W, KB, KQ)

    n_on_p, n_on_m, n_state_p, n_state_m = _init_count_arrays(D_B, D_Q, KB, KQ, N)

    # Initial per-column log-probabilities
    log_P = np.array([
        col_log_prob(n_on_p[:, c], n_on_m[:, c],
                     n_state_p[:, c, :], n_state_m[:, c, :],
                     p_err_1b_table, stats_2bit)
        for c in range(N)
    ], dtype=np.float64)

    total_log_P = log_P.sum()
    if verbose:
        print(f"  Init (conventional): sum log_P_c = {total_log_P:.4f}")

    rng = np.random.default_rng(seed if seed is not None else 2026)

    for it in range(max_iter):
        n_improved = 0
        row_order  = rng.permutation(M)

        for i in row_order:
            for c in range(N):
                w_ic  = int(W[i, c])
                cands = lut[w_ic]
                if len(cands) <= 1:
                    continue

                # Current element's SDR coefficients
                cur_d_B = [int(D_B[m, i, c]) for m in range(KB)]
                cur_d_Q = [int(D_Q[t, i, c]) for t in range(KQ)]

                # Current plane contributions for this element
                cur_bp = [max(b, 0) for b in cur_d_B]   # B_plus values {0,1}
                cur_bm = [max(-b, 0) for b in cur_d_B]  # B_minus values {0,1}
                cur_qp = [max(q, 0) for q in cur_d_Q]   # Q_plus values {0..3}
                cur_qm = [max(-q, 0) for q in cur_d_Q]  # Q_minus values {0..3}

                # Snapshot of this column's counts (used as base for all candidates)
                n_on_p_c    = n_on_p[:, c].copy()
                n_on_m_c    = n_on_m[:, c].copy()
                n_state_p_c = n_state_p[:, c, :].copy()
                n_state_m_c = n_state_m[:, c, :].copy()

                best_log_P_c = log_P[c]
                best_cand    = None

                for cand in cands:
                    # Skip if identical to current assignment
                    if (all(cand[m]      == cur_d_B[m] for m in range(KB)) and
                        all(cand[KB + t] == cur_d_Q[t] for t in range(KQ))):
                        continue

                    new_bp = [max(cand[m],       0) for m in range(KB)]
                    new_bm = [max(-cand[m],      0) for m in range(KB)]
                    new_qp = [max(cand[KB + t],  0) for t in range(KQ)]
                    new_qm = [max(-cand[KB + t], 0) for t in range(KQ)]

                    # Incremental count update — O(KB + KQ)
                    n_on_p_try    = n_on_p_c.copy()
                    n_on_m_try    = n_on_m_c.copy()
                    n_state_p_try = n_state_p_c.copy()
                    n_state_m_try = n_state_m_c.copy()

                    for m in range(KB):
                        n_on_p_try[m] += new_bp[m] - cur_bp[m]
                        n_on_m_try[m] += new_bm[m] - cur_bm[m]

                    for t in range(KQ):
                        n_state_p_try[t, cur_qp[t]] -= 1
                        n_state_p_try[t, new_qp[t]] += 1
                        n_state_m_try[t, cur_qm[t]] -= 1
                        n_state_m_try[t, new_qm[t]] += 1

                    try_log_P_c = col_log_prob(
                        n_on_p_try, n_on_m_try,
                        n_state_p_try, n_state_m_try,
                        p_err_1b_table, stats_2bit)

                    if try_log_P_c > best_log_P_c + 1e-12:
                        best_log_P_c   = try_log_P_c
                        best_cand      = cand
                        best_n_on_p    = n_on_p_try
                        best_n_on_m    = n_on_m_try
                        best_n_state_p = n_state_p_try
                        best_n_state_m = n_state_m_try

                if best_cand is not None:
                    for m in range(KB):
                        D_B[m, i, c] = best_cand[m]
                    for t in range(KQ):
                        D_Q[t, i, c] = best_cand[KB + t]
                    n_on_p[:, c]      = best_n_on_p
                    n_on_m[:, c]      = best_n_on_m
                    n_state_p[:, c, :] = best_n_state_p
                    n_state_m[:, c, :] = best_n_state_m
                    log_P[c]          = best_log_P_c
                    n_improved        += 1

        new_total = log_P.sum()
        if verbose:
            print(f"  Iter {it+1}: sum log_P_c = {new_total:.4f}  "
                  f"(delta = {new_total - total_log_P:+.4f}, updates = {n_improved})")

        if n_improved == 0:
            if verbose: print("  Converged.")
            break
        total_log_P = new_total

    # Final planes and reconstruction check
    planes = sdr_to_planes(D_B, D_Q)
    err    = verify_reconstruction(W, D_B, D_Q)
    assert err == 0, f"Accuracy optimizer: reconstruction error = {err}"

    # Legacy stats for compatibility with existing pipeline
    if cal is None:
        cal = load_calibration()
    kappa_2 = cal['kappa_2']
    kappa_3 = cal['kappa_3']
    N_th_1b = cal['N_th_1b']
    N_th_2b = cal['N_th_2b']
    alpha   = [l ** 2 for l in cfg.LAMBDA_B]
    beta    = [l ** 2 for l in cfg.LAMBDA_Q]
    eta     = cfg.ETA

    n1b  = count_1bit_columns(D_B)
    neq  = count_2bit_eq_columns(D_Q, kappa_2, kappa_3)
    J_c  = compute_J_c(n1b, neq, N_th_1b, N_th_2b, alpha, beta)
    S_c  = compute_S_c(n1b, neq, alpha, beta)
    S_raw = compute_S_c_uniform(n1b, neq)
    obj   = compute_objective_corrected(J_c, n1b, neq, N_th_1b, N_th_2b, eta)

    return {
        'planes':       planes,
        'D_B':          D_B,
        'D_Q':          D_Q,
        'J_c':          J_c,
        'S_c':          S_c,
        'S_raw':        S_raw,
        'J_total':      float(J_c.sum()),
        'S_total':      float(S_c.sum()),
        'S_raw_total':  float(S_raw.sum()),
        'obj':          obj,
        'log_P_total':  float(log_P.sum()),
        'mean_n1b':     compute_mean_n1b_per_plane(D_B),
        'mean_neq':     compute_mean_neq_per_plane(D_Q, kappa_2, kappa_3),
        'method':       'proposed1',
    }


def optimize_mapping_accuracy(W, cal=None, lut=None, KB=None, KQ=None,
                              max_iter=10, verbose=True, seed=None):
    """
    Backward-compatible alias of optimize_mapping_proposed1().
    """
    return optimize_mapping_proposed1(
        W=W, cal=cal, lut=lut, KB=KB, KQ=KQ,
        max_iter=max_iter, verbose=verbose, seed=seed)


# ---------------------------------------------------------------------------
# Entry point: quick self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from baseline_mapping  import print_summary
    from column_stats      import load_calibration

    rng = np.random.default_rng(2026)
    W   = rng.integers(-cfg.W_MAX, cfg.W_MAX + 1,
                        size=(cfg.COLUMN_SIZE, cfg.COLUMN_SIZE))

    try:
        cal = load_calibration()
    except FileNotFoundError:
        print("No calibration file — using placeholder thresholds.")
        cal = {'N_th_1b': 32.0, 'N_th_2b': 40.0, 'kappa_2': 1.5, 'kappa_3': 2.5}

    print("Proposed1 optimizer (5 iters)...")
    res = optimize_mapping_proposed1(W, cal=cal, max_iter=5, verbose=True)
    print_summary(res, label='Proposed1')
    print(f"  log_P_total = {res['log_P_total']:.4f}")
