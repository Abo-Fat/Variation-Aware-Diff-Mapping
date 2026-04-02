"""
Proposed2 SDR Mapping Optimizer.

Objective: maximise sum_c log P(y_CIM[c] == y_exact[c]) directly
under all-ones INT8 input condition.

Proposed2 here optimises the end-to-end MAC exactness of each output column
for the specific all-ones input scenario (all input bit-cycles activated).
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


def _init_count_arrays(D_B, D_Q, KB, KQ, N):
    n_on_p    = np.zeros((KB, N), dtype=np.int32)
    n_on_m    = np.zeros((KB, N), dtype=np.int32)
    n_state_p = np.zeros((KQ, N, 4), dtype=np.int32)
    n_state_m = np.zeros((KQ, N, 4), dtype=np.int32)

    for m in range(KB):
        n_on_p[m] = (D_B[m] > 0).sum(axis=0)
        n_on_m[m] = (D_B[m] < 0).sum(axis=0)

    for t in range(KQ):
        Q_p = np.maximum(D_Q[t], 0)
        Q_m = np.maximum(-D_Q[t], 0)
        for s in range(4):
            n_state_p[t, :, s] = (Q_p == s).sum(axis=0)
            n_state_m[t, :, s] = (Q_m == s).sum(axis=0)

    return n_on_p, n_on_m, n_state_p, n_state_m


def _sample_chip_tables_batch(stats_1bit, stats_2bit, KB, KQ, M, n_chip, rng):
    """
    Sample device tables for multiple chip instantiations.

    Returns
    -------
    q1b_p : [B, KB, M, 2]
    q1b_m : [B, KB, M, 2]
    q2b_p : [B, KQ, M, 4]
    q2b_m : [B, KQ, M, 4]
    """
    B = int(n_chip)
    q1b_p = np.zeros((B, KB, M, 2), dtype=np.float64)
    q1b_m = np.zeros((B, KB, M, 2), dtype=np.float64)
    q2b_p = np.zeros((B, KQ, M, 4), dtype=np.float64)
    q2b_m = np.zeros((B, KQ, M, 4), dtype=np.float64)

    for s in (0, 1):
        key = 'on' if s == 1 else 'off'
        mu = stats_1bit[key]['mu']
        sig = max(stats_1bit[key]['var'], 1e-30) ** 0.5
        q1b_p[:, :, :, s] = rng.normal(mu, sig, size=(B, KB, M))
        q1b_m[:, :, :, s] = rng.normal(mu, sig, size=(B, KB, M))

    for s in range(4):
        mu = stats_2bit[s]['mu']
        sig = max(stats_2bit[s]['var'], 1e-30) ** 0.5
        q2b_p[:, :, :, s] = rng.normal(mu, sig, size=(B, KQ, M))
        q2b_m[:, :, :, s] = rng.normal(mu, sig, size=(B, KQ, M))

    return q1b_p, q1b_m, q2b_p, q2b_m


def _col_log_exact_rate_allones(c, D_B, D_Q, y_ref_col,
                        q1b_p, q1b_m, q2b_p, q2b_m, lambda_B, lambda_Q,
                        clip_1b, clip_2b, bit_weight_sum, row_idx):
    """
    Monte Carlo estimate of log P(y_CIM[c] == y_exact[c]) for one column
    under all-ones INT8 input (all input bit-cycles active).
    """
    B = q1b_p.shape[0]
    KB = len(lambda_B)
    KQ = len(lambda_Q)

    y_cycle = np.zeros(B, dtype=np.float64)

    # all-ones input: each bit-cycle has all rows active => sum over rows directly
    for m in range(KB):
        sp = (D_B[m, :, c] > 0).astype(np.int64)
        sm = (D_B[m, :, c] < 0).astype(np.int64)

        qp = q1b_p[:, m, row_idx, sp]  # [B, M]
        qm = q1b_m[:, m, row_idx, sm]  # [B, M]
        S_p_phys = qp.sum(axis=1)
        S_m_phys = qm.sum(axis=1)

        S_p_ideal = int(sp.sum())
        S_m_ideal = int(sm.sum())

        S_p_q = np.clip(np.round(S_p_phys).astype(np.int64), 0, clip_1b)
        S_m_q = np.clip(np.round(S_m_phys).astype(np.int64), 0, clip_1b)
        y_cycle += lambda_B[m] * (S_p_q - S_m_q).astype(np.float64)

    for t in range(KQ):
        sp = np.maximum(D_Q[t, :, c], 0).astype(np.int64)
        sm = np.maximum(-D_Q[t, :, c], 0).astype(np.int64)

        qp = q2b_p[:, t, row_idx, sp]  # [B, M]
        qm = q2b_m[:, t, row_idx, sm]  # [B, M]
        S_p_phys = qp.sum(axis=1)
        S_m_phys = qm.sum(axis=1)

        S_p_ideal = int(sp.sum())
        S_m_ideal = int(sm.sum())

        S_p_q = np.clip(np.round(S_p_phys).astype(np.int64), 0, clip_2b)
        S_m_q = np.clip(np.round(S_m_phys).astype(np.int64), 0, clip_2b)
        y_cycle += lambda_Q[t] * (S_p_q - S_m_q).astype(np.float64)

    y_cim = float(bit_weight_sum) * y_cycle

    exact_rate = float(np.mean(np.round(y_cim).astype(np.int64) == y_ref_col))
    return float(np.log(max(exact_rate, 1e-12)))


def optimize_mapping_proposed2(W, cal=None, lut=None, KB=None, KQ=None,
                               max_iter=10, verbose=True, seed=None,
                               act_scale=0.5, n_bit_cycles=8,
                               n_chip_obj=512):
    """
    Proposed2: optimize end-to-end per-column MAC exactness directly.

    Objective:
      max sum_c log P(y_CIM[c] == y_exact[c]),
    where probability is estimated over device-variation MC chips, under
    all-ones INT8 input condition.

    Notes
    -----
    - act_scale / n_bit_cycles are kept for API compatibility and unused.
    """
    _ = act_scale, n_bit_cycles

    if KB is None: KB = cfg.KB
    if KQ is None: KQ = cfg.KQ

    W = np.asarray(W, dtype=np.int64)
    M, N = W.shape

    if lut is None:
        if verbose: print("  Building SDR LUT...")
        lut = build_sdr_lut(KB, KQ)

    if verbose:
        print("  Computing analytical device statistics...")
    stats_1bit, stats_2bit = compute_device_stats(verbose=False)

    rng = np.random.default_rng(seed if seed is not None else 2026)
    D_B, D_Q = conventional_decompose(W, KB, KQ)

    # Device-variation MC tables for objective estimation.
    q1b_p, q1b_m, q2b_p, q2b_m = _sample_chip_tables_batch(
        stats_1bit, stats_2bit, KB, KQ, M, int(n_chip_obj), rng)

    bit_weights = [1 << k for k in range(7)] + [-128]
    bit_weight_sum = int(sum(bit_weights))   # = -1 for INT8 two's complement
    # All-ones INT8 input means x = -1 for every row.
    y_ref_allones = (-W.sum(axis=0)).astype(np.int64)  # [N]
    row_idx = np.arange(M, dtype=np.int64)

    lambda_B = list(cfg.LAMBDA_B)
    lambda_Q = list(cfg.LAMBDA_Q)
    clip_1b = int(M)
    clip_2b = int(3 * M)

    # Initial column objectives.
    log_P = np.array([
        _col_log_exact_rate_allones(
            c, D_B, D_Q, y_ref_allones[c],
            q1b_p, q1b_m, q2b_p, q2b_m, lambda_B, lambda_Q, clip_1b, clip_2b, bit_weight_sum, row_idx)
        for c in range(N)
    ], dtype=np.float64)

    total_log_P = float(log_P.sum())
    if verbose:
        print(f"  Init (conventional): sum log P_exact_c(all-ones) = {total_log_P:.4f}")

    for it in range(max_iter):
        n_improved = 0
        row_order = rng.permutation(M)

        for i in row_order:
            for c in range(N):
                w_ic = int(W[i, c])
                cands = lut[w_ic]
                if len(cands) <= 1:
                    continue

                cur_B = [int(D_B[m, i, c]) for m in range(KB)]
                cur_Q = [int(D_Q[t, i, c]) for t in range(KQ)]

                best_log_P_c = log_P[c]
                best_cand = None

                for cand in cands:
                    if (all(cand[m] == cur_B[m] for m in range(KB)) and
                        all(cand[KB + t] == cur_Q[t] for t in range(KQ))):
                        continue

                    for m in range(KB):
                        D_B[m, i, c] = int(cand[m])
                    for t in range(KQ):
                        D_Q[t, i, c] = int(cand[KB + t])

                    try_log_P_c = _col_log_exact_rate_allones(
                        c, D_B, D_Q, y_ref_allones[c],
                        q1b_p, q1b_m, q2b_p, q2b_m,
                        lambda_B, lambda_Q, clip_1b, clip_2b, bit_weight_sum, row_idx)

                    if try_log_P_c > best_log_P_c + 1e-12:
                        best_log_P_c = try_log_P_c
                        best_cand = cand

                if best_cand is not None:
                    for m in range(KB):
                        D_B[m, i, c] = int(best_cand[m])
                    for t in range(KQ):
                        D_Q[t, i, c] = int(best_cand[KB + t])
                    log_P[c] = best_log_P_c
                    n_improved += 1
                else:
                    for m in range(KB):
                        D_B[m, i, c] = cur_B[m]
                    for t in range(KQ):
                        D_Q[t, i, c] = cur_Q[t]

        new_total = float(log_P.sum())
        if verbose:
            print(f"  Iter {it+1}: sum log P_exact_c = {new_total:.4f}  "
                  f"(delta = {new_total - total_log_P:+.4f}, updates = {n_improved})")

        if n_improved == 0:
            if verbose:
                print("  Converged.")
            break
        total_log_P = new_total

    planes = sdr_to_planes(D_B, D_Q)
    err = verify_reconstruction(W, D_B, D_Q)
    assert err == 0, f"Proposed2 optimizer: reconstruction error = {err}"

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
        'method':       'proposed2',
    }


if __name__ == '__main__':
    from baseline_mapping import print_summary
    from column_stats import load_calibration

    rng = np.random.default_rng(2026)
    W = cfg.generate_weight_matrix(rng)

    try:
        cal = load_calibration()
    except FileNotFoundError:
        print("No calibration file, using placeholder thresholds.")
        cal = {'N_th_1b': 32.0, 'N_th_2b': 40.0, 'kappa_2': 1.5, 'kappa_3': 2.5}

    print("Proposed2 optimizer (5 iters)...")
    res = optimize_mapping_proposed2(W, cal=cal, max_iter=5, verbose=True)
    print_summary(res, label='Proposed2')
    print(f"  log_P_total = {res['log_P_total']:.4f}")
