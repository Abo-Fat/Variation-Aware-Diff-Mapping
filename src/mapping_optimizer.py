"""
Proposed SDR Mapping Optimizer — Algorithm 1 (Column-wise SDR Mapping).

Optimization problem:
  min  Σ_c [ J_c(overload_mode)  +  η · S_c(sparsity_mode) ]

J_c  — column overload risk (controlled by OVERLOAD_MODE in config):
  'weighted'       : Σ_m α_m [n1b_m - N_th^1b]_+  + Σ_t β_t [neq_t - N_th^2b]_+
                     α_m = λ_B[m]², β_t = λ_Q[t]²  — penalises MSB overloads more
  'uniform'        : Σ_m [n1b_m - N_th^1b]_+  + Σ_t [neq_t - N_th^2b]_+
  'nth_normalized' : Σ_m [(n1b_m - N_th^1b)/N_th^1b]_+  + …

S_c  — column sparsity cost (controlled by SPARSITY_MODE in config):
  'weighted'       : Σ_m α_m n1b_m + Σ_t β_t neq_t   ← BIASED: mirrors conventional
  'uniform'        : Σ_m n1b_m + Σ_t neq_t
  'nth_normalized' : Σ_m n1b_m/N_th^1b + Σ_t neq_t/N_th^2b  ← recommended

NOTE: Using 'weighted' for S_c biases the optimizer toward zeroing high-bit digits
and filling low-bit digits — which replicates conventional binary decomposition and
eliminates the benefit of SDR redundancy.  Use 'uniform' or 'nth_normalized' for S_c.

Recommended configuration (config_inno2.py):
  OVERLOAD_MODE = 'weighted'       # error-importance-weighted J_c
  SPARSITY_MODE = 'nth_normalized' # unbiased uniform sparsity incentive

Strategy (greedy coordinate descent per Algorithm 1):
  1. Initialise every element with its Min-neq SDR candidate.
  2. Compute column loads n1b[:,c] and neq[:,c].
  3. Repeat until no improvement:
       For each column c, for each row i:
         Enumerate all SDR candidates for W[i,c].
         Temporarily replace; evaluate new column-level objective.
         Keep the candidate that minimises the column objective.
  4. Return the resulting D_B, D_Q and derived statistics.

Since only column c is affected when element (i,c) changes, comparing
per-column obj_c values is sufficient for the acceptance criterion.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import config_inno2 as cfg

from decompose    import (build_sdr_lut,
                          conventional_decompose,
                          min_neq_decompose,
                          sdr_to_planes,
                          verify_reconstruction)
from column_stats import (count_1bit_columns,
                          count_2bit_eq_columns,
                          compute_J_c,
                          compute_S_c,
                          compute_S_c_uniform,
                          compute_objective,
                          compute_objective_corrected,
                          compute_mean_n1b_per_plane,
                          compute_mean_neq_per_plane,
                          load_calibration)


# ---------------------------------------------------------------------------
# Per-column objective helper
# ---------------------------------------------------------------------------

def _sparsity_term(n1b_c, neq_c, N_th_1b, N_th_2b, alpha, beta, mode):
    """
    Column sparsity proxy used by optimizer.

    mode:
      - 'weighted'       : sum alpha*n1b + sum beta*neq   (legacy)
      - 'uniform'        : sum n1b + sum neq
      - 'nth_normalized' : sum(n1b/N_th_1b) + sum(neq/N_th_2b)
    """
    if mode == 'weighted':
        s = 0.0
        for m, a in enumerate(alpha):
            s += a * float(n1b_c[m])
        for t, b in enumerate(beta):
            s += b * float(neq_c[t])
        return s

    if mode == 'uniform':
        return float(np.sum(n1b_c) + np.sum(neq_c))

    if mode == 'nth_normalized':
        d1 = max(float(N_th_1b), 1e-12)
        d2 = max(float(N_th_2b), 1e-12)
        return float(np.sum(n1b_c) / d1 + np.sum(neq_c) / d2)

    raise ValueError(f"Unknown sparsity mode: {mode}")


def _overload_term(n1b_c, neq_c, N_th_1b, N_th_2b, alpha, beta, mode):
    """
    Column overload proxy used by optimizer.

    mode:
      - 'weighted'       : original weighted hinge
      - 'uniform'        : unweighted hinge
      - 'nth_normalized' : hinge on normalized overload
    """
    if mode == 'weighted':
        j = 0.0
        for m, a in enumerate(alpha):
            j += a * max(float(n1b_c[m]) - N_th_1b, 0.0)
        for t, b in enumerate(beta):
            j += b * max(float(neq_c[t]) - N_th_2b, 0.0)
        return j

    if mode == 'uniform':
        j = 0.0
        for m in range(len(n1b_c)):
            j += max(float(n1b_c[m]) - N_th_1b, 0.0)
        for t in range(len(neq_c)):
            j += max(float(neq_c[t]) - N_th_2b, 0.0)
        return j

    if mode == 'nth_normalized':
        d1 = max(float(N_th_1b), 1e-12)
        d2 = max(float(N_th_2b), 1e-12)
        j = 0.0
        for m in range(len(n1b_c)):
            j += max((float(n1b_c[m]) - N_th_1b) / d1, 0.0)
        for t in range(len(neq_c)):
            j += max((float(neq_c[t]) - N_th_2b) / d2, 0.0)
        return j

    raise ValueError(f"Unknown overload mode: {mode}")


def _col_obj(n1b_c, neq_c, N_th_1b, N_th_2b, alpha, beta,
             eta, sparsity_mode, overload_mode):
    """
    Scalar column objective  obj_c = J_c + η S_c  for one column.

    Parameters
    ----------
    n1b_c : [KB]  float array
    neq_c : [KQ]  float array
    """
    total = _overload_term(n1b_c, neq_c, N_th_1b, N_th_2b,
                           alpha, beta, overload_mode)
    s_term = _sparsity_term(n1b_c, neq_c, N_th_1b, N_th_2b,
                            alpha, beta, sparsity_mode)
    return total + eta * s_term


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

def optimize_mapping(W, cal=None, lut=None, KB=None, KQ=None,
                     max_iter=10, verbose=True, seed=None,
                     sparsity_mode=None, overload_mode=None):
    """
    Proposed column-wise SDR mapping optimizer.

    Parameters
    ----------
    W        : ndarray [M, N]  integer weights in [-W_MAX, W_MAX]
    cal      : calibration dict (N_th_1b, N_th_2b, kappa_2, kappa_3)
    lut      : pre-built SDR LUT (built if None)
    KB, KQ   : precision config (defaults from cfg)
    max_iter : max coordinate-descent sweeps
    verbose  : print per-iteration progress
    seed     : int, RNG seed for row shuffle order

    Returns
    -------
    result : dict  (same structure as baseline_mapping functions)
    """
    if KB  is None: KB  = cfg.KB
    if KQ  is None: KQ  = cfg.KQ
    if cal is None: cal = load_calibration()

    N_th_1b = cal['N_th_1b']
    N_th_2b = cal['N_th_2b']
    kappa_2 = cal['kappa_2']
    kappa_3 = cal['kappa_3']
    alpha   = [l ** 2 for l in cfg.LAMBDA_B]
    beta    = [l ** 2 for l in cfg.LAMBDA_Q]
    eta     = cfg.ETA
    if sparsity_mode is None:
        sparsity_mode = getattr(cfg, 'SPARSITY_MODE', 'weighted')
    if overload_mode is None:
        overload_mode = getattr(cfg, 'OVERLOAD_MODE', 'weighted')

    W = np.asarray(W, dtype=np.int64)
    M, N = W.shape

    # Build LUT if not supplied
    if lut is None:
        if verbose: print("  Building SDR LUT...")
        lut = build_sdr_lut(KB, KQ)

    # Pre-build per-element phi lookup for values {0,1,2,3}
    _phi = {0: 0.0, 1: 1.0, 2: kappa_2, 3: kappa_3}

    # ---- Initialise from conventional (non-redundant binary) mapping ----
    D_B, D_Q = conventional_decompose(W, KB, KQ)

    n1b = count_1bit_columns(D_B)   # [KB, N]  mutable
    neq = count_2bit_eq_columns(D_Q, kappa_2, kappa_3)   # [KQ, N]

    J_c = compute_J_c(n1b, neq, N_th_1b, N_th_2b, alpha, beta)
    S_c = compute_S_c(n1b, neq, alpha, beta)
    obj_legacy = compute_objective(J_c, S_c, eta)
    obj = sum(_col_obj(n1b[:, c], neq[:, c],
                       N_th_1b, N_th_2b, alpha, beta, eta,
                       sparsity_mode, overload_mode)
              for c in range(N))

    if verbose:
        print(f"  Init (conventional): J_total={J_c.sum():.4f}, "
              f"S_total={S_c.sum():.4f}, obj_opt={obj:.6f}, "
              f"obj_legacy={obj_legacy:.6f}, "
              f"mode=(overload:{overload_mode}, sparsity:{sparsity_mode})")

    rng = np.random.default_rng(seed if seed is not None else 2026)

    for it in range(max_iter):
        n_improved = 0
        row_order  = rng.permutation(M)

        for i in row_order:
            for c in range(N):
                w_ic     = int(W[i, c])
                cands    = lut[w_ic]
                if len(cands) <= 1:
                    continue   # only one valid decomposition, skip

                # Current element contribution to column c
                cur_d_B = [int(D_B[m, i, c]) for m in range(KB)]
                cur_d_Q = [int(D_Q[t, i, c]) for t in range(KQ)]
                cur_delta_1b = [abs(cur_d_B[m])           for m in range(KB)]
                cur_delta_nq = [_phi[abs(cur_d_Q[t])]     for t in range(KQ)]

                # Column c current counts
                n1b_c = n1b[:, c].copy()   # [KB]
                neq_c = neq[:, c].copy()   # [KQ]

                cur_obj_c = _col_obj(n1b_c, neq_c,
                                     N_th_1b, N_th_2b, alpha, beta, eta,
                                     sparsity_mode, overload_mode)

                best_obj_c  = cur_obj_c
                best_cand   = None
                best_n1b_c  = None
                best_neq_c  = None

                for cand in cands:
                    # Skip if identical to current assignment
                    if all(cand[m] == cur_d_B[m] for m in range(KB)) and \
                       all(cand[KB + t] == cur_d_Q[t] for t in range(KQ)):
                        continue

                    try_delta_1b = [abs(cand[m])            for m in range(KB)]
                    try_delta_nq = [_phi[abs(cand[KB + t])] for t in range(KQ)]

                    n1b_try = np.array([n1b_c[m] - cur_delta_1b[m] + try_delta_1b[m]
                                        for m in range(KB)], dtype=np.float64)
                    neq_try = np.array([neq_c[t] - cur_delta_nq[t] + try_delta_nq[t]
                                        for t in range(KQ)], dtype=np.float64)

                    obj_try = _col_obj(n1b_try, neq_try,
                                       N_th_1b, N_th_2b, alpha, beta, eta,
                                       sparsity_mode, overload_mode)

                    if obj_try < best_obj_c - 1e-12:
                        best_obj_c = obj_try
                        best_cand  = cand
                        best_n1b_c = n1b_try
                        best_neq_c = neq_try

                if best_cand is not None:
                    # Commit update
                    for m in range(KB):
                        D_B[m, i, c] = best_cand[m]
                    for t in range(KQ):
                        D_Q[t, i, c] = best_cand[KB + t]
                    n1b[:, c] = best_n1b_c
                    neq[:, c] = best_neq_c
                    n_improved += 1

        # Full objective after sweep
        J_c  = compute_J_c(n1b, neq, N_th_1b, N_th_2b, alpha, beta)
        S_c  = compute_S_c(n1b, neq, alpha, beta)
        obj_legacy_new = compute_objective(J_c, S_c, eta)
        obj_new = sum(_col_obj(n1b[:, c], neq[:, c],
                               N_th_1b, N_th_2b, alpha, beta, eta,
                               sparsity_mode, overload_mode)
                      for c in range(N))

        if verbose:
            print(f"  Iter {it+1}: J_total={J_c.sum():.4f}, "
                  f"S_total={S_c.sum():.4f}, obj_opt={obj_new:.6f}, "
                  f"obj_legacy={obj_legacy_new:.6f}, "
                  f"updates={n_improved}")

        if obj_new >= obj - 1e-12 and it > 0:
            if verbose:
                print("  Converged.")
            break
        obj = obj_new

    # Final planes and stats
    planes = sdr_to_planes(D_B, D_Q)
    err    = verify_reconstruction(W, D_B, D_Q)
    assert err == 0, f"Proposed: reconstruction error = {err}"

    J_c   = compute_J_c(n1b, neq, N_th_1b, N_th_2b, alpha, beta)
    S_c   = compute_S_c(n1b, neq, alpha, beta)        # weighted (legacy reference)
    S_raw = compute_S_c_uniform(n1b, neq)              # uniform active-digit count
    obj   = compute_objective_corrected(               # corrected: weighted J + η·norm S
        J_c, n1b, neq, N_th_1b, N_th_2b, eta)

    return {
        'planes':   planes,
        'D_B':      D_B,
        'D_Q':      D_Q,
        'J_c':      J_c,
        'S_c':      S_c,
        'S_raw':    S_raw,
        'J_total':  float(J_c.sum()),
        'S_total':  float(S_c.sum()),
        'S_raw_total': float(S_raw.sum()),
        'obj':      obj,
        'overload_mode': overload_mode,
        'sparsity_mode': sparsity_mode,
        'mean_n1b': compute_mean_n1b_per_plane(D_B),
        'mean_neq': compute_mean_neq_per_plane(D_Q, kappa_2, kappa_3),
        'method':   'proposed',
    }


# ---------------------------------------------------------------------------
# Entry point: demo comparison
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from baseline_mapping import conventional_map, minneq_map, print_summary

    rng = np.random.default_rng(2026)
    W   = cfg.generate_weight_matrix(rng)

    try:
        cal = load_calibration()
    except FileNotFoundError:
        print("No calibration — using placeholder thresholds.")
        cal = {'N_th_1b': 32.0, 'N_th_2b': 40.0,
               'kappa_2': 1.5,  'kappa_3': 2.5}

    lut = build_sdr_lut()

    print("Conventional...")
    res_conv = conventional_map(W, cal=cal)
    print_summary(res_conv, label='Conventional')

    print("\nMin-neq SDR...")
    res_mneq = minneq_map(W, cal=cal, lut=lut)
    print_summary(res_mneq, label='Min-neq SDR')

    print("\nProposed optimizer (5 iters)...")
    res_prop = optimize_mapping(W, cal=cal, lut=lut, max_iter=5, verbose=True)
    print_summary(res_prop, label='Proposed')
