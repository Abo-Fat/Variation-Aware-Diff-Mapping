"""
Proposed3 SDR Mapping Optimizer.

Goal: align optimization objective with Phase4 metric as closely as possible:
maximize end-to-end INT8xINT8 MAC exact-match rate under random INT8 inputs.

Approach:
  - Fixed Common Random Numbers (CRN): fixed random INT8 batch and fixed chip
    samples during one optimization run, reducing stochastic ranking noise.
  - Coordinate descent over SDR candidates (same decomposition constraints).
  - Objective per column: log exact-rate averaged over CRN vectors and chips.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from concurrent.futures import ThreadPoolExecutor
import config_inno2 as cfg

from decompose import (build_sdr_lut,
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


def _sample_chip_tables_batch(stats_1bit, stats_2bit, KB, KQ, M, n_chip, rng):
    """Sample per-state cell charge tables for multiple chip instantiations."""
    C = int(n_chip)
    q1b_p = np.zeros((C, KB, M, 2), dtype=np.float64)
    q1b_m = np.zeros((C, KB, M, 2), dtype=np.float64)
    q2b_p = np.zeros((C, KQ, M, 4), dtype=np.float64)
    q2b_m = np.zeros((C, KQ, M, 4), dtype=np.float64)

    for s in (0, 1):
        key = 'on' if s == 1 else 'off'
        mu = stats_1bit[key]['mu']
        sig = max(stats_1bit[key]['var'], 1e-30) ** 0.5
        q1b_p[:, :, :, s] = rng.normal(mu, sig, size=(C, KB, M))
        q1b_m[:, :, :, s] = rng.normal(mu, sig, size=(C, KB, M))

    for s in range(4):
        mu = stats_2bit[s]['mu']
        sig = max(stats_2bit[s]['var'], 1e-30) ** 0.5
        q2b_p[:, :, :, s] = rng.normal(mu, sig, size=(C, KQ, M))
        q2b_m[:, :, :, s] = rng.normal(mu, sig, size=(C, KQ, M))

    return q1b_p, q1b_m, q2b_p, q2b_m


def _col_log_exact_rate_crn_col(DB_col, DQ_col, X_bits, bit_weights, y_ref_col,
                                q1b_p, q1b_m, q2b_p, q2b_m,
                                lambda_B, lambda_Q, clip_1b, clip_2b, row_idx):
    """
    CRN objective for one column decomposition:
      log( mean_{chip, x in CRN batch}[ 1(y_cim==y_ref) ] )
    """
    C = q1b_p.shape[0]
    V = X_bits[0].shape[0]
    KB = len(lambda_B)
    KQ = len(lambda_Q)

    exact_sum = 0.0

    for ch in range(C):
        y_cim = np.zeros(V, dtype=np.float64)

        for k, w_k in enumerate(bit_weights):
            A = X_bits[k]  # [V, M]
            y_cycle = np.zeros(V, dtype=np.float64)

            for m in range(KB):
                sp = (DB_col[m] > 0).astype(np.int64)
                sm = (DB_col[m] < 0).astype(np.int64)

                qp = q1b_p[ch, m, row_idx, sp]  # [M]
                qm = q1b_m[ch, m, row_idx, sm]  # [M]

                S_p_phys = A @ qp
                S_m_phys = A @ qm
                S_p_ideal = A @ sp
                S_m_ideal = A @ sm

                S_p_q = np.clip(np.round(S_p_phys).astype(np.int64), 0, clip_1b)
                S_m_q = np.clip(np.round(S_m_phys).astype(np.int64), 0, clip_1b)
                y_cycle += lambda_B[m] * (S_p_q - S_m_q).astype(np.float64)

            for t in range(KQ):
                sp = np.maximum(DQ_col[t], 0).astype(np.int64)
                sm = np.maximum(-DQ_col[t], 0).astype(np.int64)

                qp = q2b_p[ch, t, row_idx, sp]  # [M]
                qm = q2b_m[ch, t, row_idx, sm]  # [M]

                S_p_phys = A @ qp
                S_m_phys = A @ qm
                S_p_ideal = A @ sp
                S_m_ideal = A @ sm

                S_p_q = np.clip(np.round(S_p_phys).astype(np.int64), 0, clip_2b)
                S_m_q = np.clip(np.round(S_m_phys).astype(np.int64), 0, clip_2b)
                y_cycle += lambda_Q[t] * (S_p_q - S_m_q).astype(np.float64)

            y_cim += float(w_k) * y_cycle

        exact_sum += float(np.mean(np.round(y_cim).astype(np.int64) == y_ref_col))

    exact_rate = exact_sum / max(float(C), 1.0)
    return float(np.log(max(exact_rate, 1e-12)))


def optimize_mapping_proposed3(W, cal=None, lut=None, KB=None, KQ=None,
                               max_iter=3, verbose=True, seed=None,
                               n_vec_obj=128, n_chip_obj=16, rows_per_iter=None,
                               n_jobs=None, parallel_init=True, parallel_candidates=True):
    """
    Proposed3: CRN-aligned end-to-end optimizer for Phase4-like metric.
    """
    if KB is None:
        KB = cfg.KB
    if KQ is None:
        KQ = cfg.KQ

    W = np.asarray(W, dtype=np.int64)
    M, N = W.shape
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 1) - 1)
    n_jobs = int(max(1, n_jobs))

    if lut is None:
        if verbose:
            print("  Building SDR LUT...")
        lut = build_sdr_lut(KB, KQ)

    if verbose:
        print("  Computing analytical device statistics...")
    stats_1bit, stats_2bit = compute_device_stats(verbose=False)

    rng = np.random.default_rng(seed if seed is not None else 2027)
    D_B, D_Q = conventional_decompose(W, KB, KQ)

    # CRN datasets used across the whole optimization.
    X = rng.integers(-128, 128, size=(int(n_vec_obj), M), dtype=np.int8).astype(np.int32)
    X_uint = X % 256
    X_bits = [((X_uint >> k) & 1).astype(np.int64) for k in range(8)]
    bit_weights = [1 << k for k in range(7)] + [-128]
    y_ref_all = X.astype(np.int64) @ W   # [V, N]

    q1b_p, q1b_m, q2b_p, q2b_m = _sample_chip_tables_batch(
        stats_1bit, stats_2bit, KB, KQ, M, int(n_chip_obj), rng)

    lambda_B = list(cfg.LAMBDA_B)
    lambda_Q = list(cfg.LAMBDA_Q)
    clip_1b = int(M)
    clip_2b = int(3 * M)
    row_idx = np.arange(M, dtype=np.int64)

    def _score_col(c):
        return _col_log_exact_rate_crn_col(
            D_B[:, :, c], D_Q[:, :, c], X_bits, bit_weights, y_ref_all[:, c],
            q1b_p, q1b_m, q2b_p, q2b_m,
            lambda_B, lambda_Q, clip_1b, clip_2b, row_idx)

    if parallel_init and n_jobs > 1:
        with ThreadPoolExecutor(max_workers=min(n_jobs, N)) as ex:
            log_P = np.fromiter(ex.map(_score_col, range(N)), dtype=np.float64, count=N)
    else:
        log_P = np.array([_score_col(c) for c in range(N)], dtype=np.float64)

    total_log_P = float(log_P.sum())
    if verbose:
        print(f"  Init (conventional): sum log P_exact_c(CRN) = {total_log_P:.4f}")

    executor = None
    if parallel_candidates and n_jobs > 1:
        executor = ThreadPoolExecutor(max_workers=n_jobs)

    try:
        for it in range(max_iter):
            n_improved = 0
            row_order = rng.permutation(M)
            if rows_per_iter is not None:
                n_rows = int(max(1, min(M, rows_per_iter)))
                row_order = row_order[:n_rows]

            for i in row_order:
                for c in range(N):
                    w_ic = int(W[i, c])
                    cands = lut[w_ic]
                    if len(cands) <= 1:
                        continue

                    DB_col = D_B[:, :, c].copy()
                    DQ_col = D_Q[:, :, c].copy()
                    cur_B = [int(DB_col[m, i]) for m in range(KB)]
                    cur_Q = [int(DQ_col[t, i]) for t in range(KQ)]
                    best_log = log_P[c]
                    best_cand = None

                    def _score_cand(cand):
                        if (all(cand[m] == cur_B[m] for m in range(KB)) and
                            all(cand[KB + t] == cur_Q[t] for t in range(KQ))):
                            return None
                        db_try = DB_col.copy()
                        dq_try = DQ_col.copy()
                        for m in range(KB):
                            db_try[m, i] = int(cand[m])
                        for t in range(KQ):
                            dq_try[t, i] = int(cand[KB + t])
                        val = _col_log_exact_rate_crn_col(
                            db_try, dq_try, X_bits, bit_weights, y_ref_all[:, c],
                            q1b_p, q1b_m, q2b_p, q2b_m,
                            lambda_B, lambda_Q, clip_1b, clip_2b, row_idx)
                        return val, cand

                    if executor is not None:
                        scored = list(executor.map(_score_cand, cands))
                    else:
                        scored = [_score_cand(cand) for cand in cands]

                    for item in scored:
                        if item is None:
                            continue
                        try_log, cand = item
                        if try_log > best_log + 1e-12:
                            best_log = try_log
                            best_cand = cand

                    if best_cand is not None:
                        for m in range(KB):
                            D_B[m, i, c] = int(best_cand[m])
                        for t in range(KQ):
                            D_Q[t, i, c] = int(best_cand[KB + t])
                        log_P[c] = best_log
                        n_improved += 1

            new_total = float(log_P.sum())
            if verbose:
                print(f"  Iter {it+1}: sum log P_exact_c(CRN) = {new_total:.4f}  "
                      f"(delta = {new_total - total_log_P:+.4f}, updates = {n_improved})")

            if n_improved == 0:
                if verbose:
                    print("  Converged.")
                break
            total_log_P = new_total
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    planes = sdr_to_planes(D_B, D_Q)
    err = verify_reconstruction(W, D_B, D_Q)
    assert err == 0, f"Proposed3 optimizer: reconstruction error = {err}"

    if cal is None:
        cal = load_calibration()
    kappa_2 = cal['kappa_2']
    kappa_3 = cal['kappa_3']
    N_th_1b = cal['N_th_1b']
    N_th_2b = cal['N_th_2b']
    alpha = [l ** 2 for l in cfg.LAMBDA_B]
    beta = [l ** 2 for l in cfg.LAMBDA_Q]
    eta = cfg.ETA

    n1b = count_1bit_columns(D_B)
    neq = count_2bit_eq_columns(D_Q, kappa_2, kappa_3)
    J_c = compute_J_c(n1b, neq, N_th_1b, N_th_2b, alpha, beta)
    S_c = compute_S_c(n1b, neq, alpha, beta)
    S_raw = compute_S_c_uniform(n1b, neq)
    obj = compute_objective_corrected(J_c, n1b, neq, N_th_1b, N_th_2b, eta)

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
        'method':       'proposed3',
        'crn_n_vec':    int(n_vec_obj),
        'crn_n_chip':   int(n_chip_obj),
    }


if __name__ == '__main__':
    from baseline_mapping import print_summary

    rng = np.random.default_rng(2027)
    W = cfg.generate_weight_matrix(rng)

    try:
        cal = load_calibration()
    except FileNotFoundError:
        cal = {'N_th_1b': 32.0, 'N_th_2b': 40.0, 'kappa_2': 1.5, 'kappa_3': 2.5}

    res = optimize_mapping_proposed3(W, cal=cal, max_iter=1, verbose=True)
    print_summary(res, label='Proposed3')
    print(f"log_P_total={res['log_P_total']:.4f}")
