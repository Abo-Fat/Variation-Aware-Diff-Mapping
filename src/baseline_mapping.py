"""
Three mapping baselines for SDR structural analysis (Innovation 2).

conventional_map  – standard u=0 non-redundant binary decomposition.
                    Serves as the "Conventional mapping" baseline.

minneq_map        – per-element min-neq SDR (independent element optimisation).
                    Serves as the "Min-neq SDR baseline".

baseline_map      – alias for conventional_map (backward compat.).

All functions return a result dict with the same structure:
  'planes'    : dict  B_plus/B_minus/Q_plus/Q_minus/D_B/D_Q
  'D_B'       : [KB, M, N]  int8  signed digit 1-bit
  'D_Q'       : [KQ, M, N]  int8  signed digit 2-bit
  'J_c'       : [N]  float64  per-column risk
  'S_c'       : [N]  float64  per-column sparsity
  'J_total'   : float
  'S_total'   : float
  'obj'       : float   Σ J_c + η Σ S_c
  'mean_n1b'  : [KB]  mean combined 1-bit count per plane (over columns)
  'mean_neq'  : [KQ]  mean combined 2-bit eq count per plane (over columns)
  'method'    : str
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
                          compute_objective,
                          compute_mean_n1b_per_plane,
                          compute_mean_neq_per_plane,
                          load_calibration)


# ---------------------------------------------------------------------------
# Shared result builder
# ---------------------------------------------------------------------------

def _build_result(W, D_B, D_Q, cal, method):
    """Assemble a unified result dict from signed-digit arrays."""
    planes = sdr_to_planes(D_B, D_Q)

    err = verify_reconstruction(W, D_B, D_Q)
    assert err == 0, f"{method}: reconstruction error = {err}"

    n1b = count_1bit_columns(D_B)
    neq = count_2bit_eq_columns(D_Q, cal['kappa_2'], cal['kappa_3'])

    J_c = compute_J_c(n1b, neq, cal['N_th_1b'], cal['N_th_2b'])
    S_c = compute_S_c(n1b, neq)
    obj = compute_objective(J_c, S_c)

    return {
        'planes':   planes,
        'D_B':      D_B,
        'D_Q':      D_Q,
        'J_c':      J_c,
        'S_c':      S_c,
        'J_total':  float(J_c.sum()),
        'S_total':  float(S_c.sum()),
        'obj':      obj,
        'mean_n1b': compute_mean_n1b_per_plane(D_B),
        'mean_neq': compute_mean_neq_per_plane(
            D_Q, cal['kappa_2'], cal['kappa_3']),
        'method':   method,
    }


# ---------------------------------------------------------------------------
# Conventional mapping  (u=0, non-redundant binary)
# ---------------------------------------------------------------------------

def conventional_map(W, cal=None, KB=None, KQ=None):
    """
    Apply conventional (u=0) non-redundant binary decomposition to W.

    Parameters
    ----------
    W   : ndarray [M, N]  integer weights in [-W_MAX, W_MAX]
    cal : calibration dict (N_th_1b, N_th_2b, kappa_2, kappa_3)
    """
    if KB  is None: KB  = cfg.KB
    if KQ  is None: KQ  = cfg.KQ
    if cal is None: cal = load_calibration()

    W    = np.asarray(W, dtype=np.int64)
    D_B, D_Q = conventional_decompose(W, KB, KQ)
    return _build_result(W, D_B, D_Q, cal, 'conventional')


# ---------------------------------------------------------------------------
# Min-neq SDR baseline
# ---------------------------------------------------------------------------

def minneq_map(W, cal=None, lut=None, KB=None, KQ=None):
    """
    Apply min-neq SDR decomposition to W.

    For each element independently, picks the SDR candidate with the
    smallest per-element equivalent active count.

    Parameters
    ----------
    W   : ndarray [M, N]
    cal : calibration dict
    lut : pre-built SDR lookup table (built if None)
    """
    if KB  is None: KB  = cfg.KB
    if KQ  is None: KQ  = cfg.KQ
    if cal is None: cal = load_calibration()

    W = np.asarray(W, dtype=np.int64)

    if lut is None:
        lut = build_sdr_lut(KB, KQ)

    D_B, D_Q = min_neq_decompose(W, lut,
                                  cal['kappa_2'], cal['kappa_3'],
                                  KB, KQ)
    return _build_result(W, D_B, D_Q, cal, 'minneq')


# Backward-compat alias
def baseline_map(W, cal=None, KB=None, KQ=None):
    """Alias for conventional_map (backward compatibility)."""
    return conventional_map(W, cal=cal, KB=KB, KQ=KQ)


# ---------------------------------------------------------------------------
# Print helper
# ---------------------------------------------------------------------------

def print_summary(result, label=None):
    if label is None: label = result.get('method', '?')
    print(f"\n{'='*50}")
    print(f"{label} mapping summary")
    print(f"  J_total = {result['J_total']:.6f}")
    print(f"  S_total = {result['S_total']:.6f}")
    print(f"  obj     = {result['obj']:.6f}")
    print(f"  mean_n1b per plane : {np.round(result['mean_n1b'], 2)}")
    print(f"  mean_neq per plane : {np.round(result['mean_neq'], 2)}")


# ---------------------------------------------------------------------------
# Entry point: demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    rng = np.random.default_rng(2026)
    W   = rng.integers(-cfg.W_MAX, cfg.W_MAX + 1,
                        size=(cfg.COLUMN_SIZE, cfg.COLUMN_SIZE))

    try:
        cal = load_calibration()
    except FileNotFoundError:
        print("No calibration file — using placeholder thresholds.")
        cal = {'N_th_1b': 32.0, 'N_th_2b': 40.0,
               'kappa_2': 1.5,  'kappa_3': 2.5}

    lut = build_sdr_lut()

    res_conv  = conventional_map(W, cal=cal)
    res_mneq  = minneq_map(W, cal=cal, lut=lut)

    print_summary(res_conv,  label='Conventional')
    print_summary(res_mneq,  label='Min-neq SDR')
