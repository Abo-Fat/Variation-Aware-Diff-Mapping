"""
CIM MAC Accuracy Evaluation — Innovation 2 (Quantised-Integer Model).

Computation scheme (1-bit serial input, quantised PE output):
  - INT8 input vector x ∈ [-128, 127]^M decomposed into 8 bit planes
    (two's complement: bit k weight = 2^k for k=0..6, -128 for k=7)
  - FeFET variation is sampled ONCE per weight matrix (fixed-chip model),
    then reused across all N_VEC test vectors.
  - Per bit cycle k:
      input_bit[i] = (unsigned(x[i]) >> k) & 1   ∈ {0, 1}

      For each 1-bit PE plane m  (B_plus[m], B_minus[m]):
        [Step 1 — analogue integration]
        S_p_phys[j] = Σ_i input_bit[i] · Q_eff_1b(B_plus[m][i,j])   (float, ~integer + noise)
        S_m_phys[j] = Σ_i input_bit[i] · Q_eff_1b(B_minus[m][i,j])

        [Step 2 — quantised readout (ADC / sense-amp)]
        S_p_quant[j] = clip( round(S_p_phys[j]), 0, M )              (integer)
        S_m_quant[j] = clip( round(S_m_phys[j]), 0, M )

        [Step 3 — weighted accumulation]
        y_cycle[j]  += λ_B[m] · (S_p_quant[j] − S_m_quant[j])

      For each 2-bit PE plane t  (Q_plus[t], Q_minus[t]):
        (same, with clip to [0, 3M])

    y_CIM[j] += bit_weight_k · y_cycle[j]

PE error event:
  For plane m, column j, bit cycle k:
    error iff S_quant[j] ≠ S_ideal[j]
  This is the per-PE event that the J_c / N_th threshold objective targets.

Accuracy metrics (mean ± std over N_VEC random INT8 vectors):
  Primary:
    - MAC exact match rate  :  fraction of output elements where y_CIM[j] == y_ref[j]
  Secondary:
    - Cosine similarity     :  cos(y_CIM, y_ref)
    - Relative L2 error     :  ‖y_CIM − y_ref‖₂ / ‖y_ref‖₂
  Diagnostic:
    - Per-plane PE error rate (1-bit and 2-bit planes, p/m sides combined)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
import torch
import config_inno2 as cfg

# ---------------------------------------------------------------------------
# INT8 two's complement bit weights: bits 0..6 have weights 1,2,...,64;
# bit 7 (sign bit) has weight -128.
# ---------------------------------------------------------------------------
_BIT_WEIGHTS = [1 << k for k in range(7)] + [-128]


# ---------------------------------------------------------------------------
# FeFET model factory
# ---------------------------------------------------------------------------

def make_fefet_model():
    """
    Create a FeFETVariationModel configured from config_inno2.py.
    Mode '2bit' so that both charge methods are fully initialised.
    """
    from FeFET_model import FeFETVariationModel
    return FeFETVariationModel(
        mode='2bit',
        SS=cfg.FEFET_SS,
        I0=cfg.FEFET_I0,
        R_limit=cfg.FEFET_R_LIMIT,
        Vd=cfg.FEFET_VD,
        vth_states=list(cfg.FEFET_VTH_STATES),
        sigma_vth=list(cfg.FEFET_SIGMA_VTH),
        vg_steps_2bit=list(cfg.FEFET_VG_STEPS_2BIT),
        vg_read_1bit=cfg.FEFET_VG_READ_1BIT,
        t_step=cfg.FEFET_T_STEP,
        sigma_r_rel=cfg.FEFET_SIGMA_R_REL,
    )


# ---------------------------------------------------------------------------
# Pre-compute FeFET charges for all PE planes (one chip instantiation)
# ---------------------------------------------------------------------------

def _precompute_pe_charges(planes, fefet, sigma, device):
    """
    Sample FeFET Vth variation ONCE per weight matrix and compute Q_eff
    for every PE plane.  Fixed-chip model: same variation for all vectors.

    Returns
    -------
    q1b_p : list[KB]  each ndarray [M, N] float64, Q_eff ≈ {0, 1} + noise
    q1b_m : list[KB]
    q2b_p : list[KQ]  each ndarray [M, N] float64, Q_eff ≈ {0,1,2,3} + noise
    q2b_m : list[KQ]
    """
    KB = planes['B_plus'].shape[0]
    KQ = planes['Q_plus'].shape[0]

    q1b_p, q1b_m, q2b_p, q2b_m = [], [], [], []

    for m in range(KB):
        B_p = torch.tensor(planes['B_plus'][m].astype(np.float32),  device=device)
        B_m = torch.tensor(planes['B_minus'][m].astype(np.float32), device=device)
        q1b_p.append(fefet.compute_cell_charge_1bit(B_p, sigma, device)
                          .cpu().numpy().astype(np.float64))
        q1b_m.append(fefet.compute_cell_charge_1bit(B_m, sigma, device)
                          .cpu().numpy().astype(np.float64))

    for t in range(KQ):
        Q_p = torch.tensor(planes['Q_plus'][t].astype(np.int64),  device=device)
        Q_m = torch.tensor(planes['Q_minus'][t].astype(np.int64), device=device)
        q2b_p.append(fefet.compute_cell_charge_2bit(Q_p, sigma, device)
                          .cpu().numpy().astype(np.float64))
        q2b_m.append(fefet.compute_cell_charge_2bit(Q_m, sigma, device)
                          .cpu().numpy().astype(np.float64))

    return q1b_p, q1b_m, q2b_p, q2b_m


# ---------------------------------------------------------------------------
# CIM output — quantised-integer model
# ---------------------------------------------------------------------------

def _cim_output_quantised(x,
                           q1b_p, q1b_m, q2b_p, q2b_m,
                           b_plus_int, b_minus_int, q_plus_int, q_minus_int,
                           lambda_B, lambda_Q,
                           clip_1b, clip_2b):
    """
    Compute quantised-integer CIM MAC output for one INT8 input vector.

    Each PE's analogue column sum is rounded to the nearest integer before
    entering the λ-weighted accumulation — modelling ADC / sense-amp readout.
    PE quantisation errors (S_quant ≠ S_ideal) are counted per plane.

    Parameters
    ----------
    x             : [M] int32
    q1b_p/m       : list[KB] of [M, N] float64  (physical charges, with Vth variation)
    q2b_p/m       : list[KQ] of [M, N] float64
    b_plus_int    : [KB, M, N] int64  (ideal plane, values {0, 1})
    b_minus_int   : [KB, M, N] int64
    q_plus_int    : [KQ, M, N] int64  (ideal plane, values {0, 1, 2, 3})
    q_minus_int   : [KQ, M, N] int64
    lambda_B      : list[KB] int
    lambda_Q      : list[KQ] int
    clip_1b       : int  = M    (max column sum for 1-bit plane)
    clip_2b       : int  = 3·M  (max column sum for 2-bit plane)

    Returns
    -------
    y_CIM       : [N] float64   (integer-valued, exact when no PE errors)
    pe_err_1b   : [KB] int64    total PE errors (p + m sides) across all bit cycles
    pe_err_2b   : [KQ] int64
    pe_total_1b : [KB] int64    total measurements  (denominator for error rate)
    pe_total_2b : [KQ] int64
    """
    KB = len(q1b_p)
    KQ = len(q2b_p)
    N  = q1b_p[0].shape[1]

    x_uint = x.astype(np.int32) % 256  # unsigned reinterpretation for bit extraction

    y_CIM        = np.zeros(N, dtype=np.float64)
    pe_err_1b    = np.zeros(KB, dtype=np.int64)
    pe_err_2b    = np.zeros(KQ, dtype=np.int64)
    pe_total_1b  = np.zeros(KB, dtype=np.int64)
    pe_total_2b  = np.zeros(KQ, dtype=np.int64)

    for k, w_k in enumerate(_BIT_WEIGHTS):
        # [M] float and int views of the k-th input bit plane
        input_f = ((x_uint >> k) & 1).astype(np.float64)
        input_i = ((x_uint >> k) & 1).astype(np.int64)

        y_cycle = np.zeros(N, dtype=np.float64)

        # ── 1-bit PE planes ──────────────────────────────────────────────────
        for m in range(KB):
            # Step 1: physical analogue column sum
            S_p_phys = input_f @ q1b_p[m]   # [N] float64
            S_m_phys = input_f @ q1b_m[m]

            # Ideal integer sum (no variation reference)
            S_p_ideal = input_i @ b_plus_int[m]    # [N] int64
            S_m_ideal = input_i @ b_minus_int[m]

            # Step 2: quantise → round then clip to [0, M]
            S_p_q = np.clip(np.round(S_p_phys).astype(np.int64), 0, clip_1b)
            S_m_q = np.clip(np.round(S_m_phys).astype(np.int64), 0, clip_1b)

            # PE error counting
            pe_err_1b[m]   += int(np.sum(S_p_q != S_p_ideal))
            pe_err_1b[m]   += int(np.sum(S_m_q != S_m_ideal))
            pe_total_1b[m] += 2 * N

            # Step 3: weighted contribution
            y_cycle += lambda_B[m] * (S_p_q - S_m_q).astype(np.float64)

        # ── 2-bit PE planes ──────────────────────────────────────────────────
        for t in range(KQ):
            S_p_phys = input_f @ q2b_p[t]
            S_m_phys = input_f @ q2b_m[t]

            S_p_ideal = input_i @ q_plus_int[t]
            S_m_ideal = input_i @ q_minus_int[t]

            # Clip to [0, 3M]
            S_p_q = np.clip(np.round(S_p_phys).astype(np.int64), 0, clip_2b)
            S_m_q = np.clip(np.round(S_m_phys).astype(np.int64), 0, clip_2b)

            pe_err_2b[t]   += int(np.sum(S_p_q != S_p_ideal))
            pe_err_2b[t]   += int(np.sum(S_m_q != S_m_ideal))
            pe_total_2b[t] += 2 * N

            y_cycle += lambda_Q[t] * (S_p_q - S_m_q).astype(np.float64)

        y_CIM += w_k * y_cycle

    return y_CIM, pe_err_1b, pe_err_2b, pe_total_1b, pe_total_2b


# ---------------------------------------------------------------------------
# Accuracy metrics
# ---------------------------------------------------------------------------

def _compute_metrics(y_cim, y_ref):
    """
    Compute cosine similarity, relative L2 error, and per-element exact match
    between CIM output (integer-valued float) and exact reference.

    Returns
    -------
    cos_sim     : float  (1.0 = perfect)
    rel_l2      : float  (0.0 = perfect)
    exact_rate  : float  fraction of elements where round(y_cim) == y_ref
    """
    y_cim = np.asarray(y_cim, dtype=np.float64)
    y_ref = np.asarray(y_ref, dtype=np.float64)

    dot    = float(np.dot(y_cim, y_ref))
    norm_c = float(np.linalg.norm(y_cim))
    norm_r = float(np.linalg.norm(y_ref))

    cos_sim    = dot / (norm_c * norm_r + 1e-12)
    rel_l2     = float(np.linalg.norm(y_cim - y_ref)) / (norm_r + 1e-12)
    exact_rate = float(np.mean(
        np.round(y_cim).astype(np.int64) == y_ref.astype(np.int64)
    ))

    return cos_sim, rel_l2, exact_rate


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def evaluate_accuracy(W, planes_base, planes_p1, planes_p2, planes_p3=None,
                      sigma=1.0, N_vec=1000, seed=2026,
                      device='cpu', verbose=True):
    """
    Evaluate CIM MAC accuracy for conventional, Proposed1, Proposed2
    (and optionally Proposed3) mappings using
    the quantised-integer model.

    FeFET variation is sampled once (fixed-chip). N_vec random INT8 vectors
    are passed through each mapping and compared against the exact integer MAC.

    Parameters
    ----------
    W           : [M, N] int array  (original weight matrix)
    planes_base : dict  from conventional_map()
    planes_p1   : dict  from optimize_mapping_proposed1()
    planes_p2   : dict  from optimize_mapping_proposed2()
    planes_p3   : dict|None  from optimize_mapping_proposed3()
    sigma       : float  vth_sigma_scale  (1.0 = nominal)
    N_vec       : int    number of random INT8 test vectors  (default 1000)
    seed        : int    RNG seed
    device      : str    torch device
    verbose     : bool

    Returns
    -------
    result : dict
      Primary accuracy:
        'baseline_exact_rate'     mean fraction of output elements exactly correct
        'proposed1_exact_rate'
        'proposed2_exact_rate'
        'proposed3_exact_rate'  (if planes_p3 is provided)
      Continuous metrics (on quantised y_CIM):
        'baseline_cos_mean/std'
        'proposed1_cos_mean/std'
        'proposed2_cos_mean/std'
        'baseline_rel_l2_mean/std'
        'proposed1_rel_l2_mean/std'
        'proposed2_rel_l2_mean/std'
      Per-plane PE error rates (fraction of PE outputs that mis-quantised):
        'baseline_pe_rate_1b'     list[KB] float
        'proposed1_pe_rate_1b'    list[KB] float
        'proposed2_pe_rate_1b'    list[KB] float
        'baseline_pe_rate_2b'     list[KQ] float
        'proposed1_pe_rate_2b'    list[KQ] float
        'proposed2_pe_rate_2b'    list[KQ] float
      Metadata:
        'lambda_B', 'lambda_Q', 'N_vec', 'sigma', 'clip_1b', 'clip_2b'
    """
    W    = np.asarray(W, dtype=np.int64)
    M, N = W.shape
    dev  = torch.device(device)

    clip_1b  = int(M)
    clip_2b  = int(3 * M)
    lambda_B = list(cfg.LAMBDA_B)
    lambda_Q = list(cfg.LAMBDA_Q)
    KB = len(lambda_B)
    KQ = len(lambda_Q)

    # Integer ideal-plane arrays (no variation)
    b_plus_b  = planes_base['B_plus'].astype(np.int64)
    b_minus_b = planes_base['B_minus'].astype(np.int64)
    q_plus_b  = planes_base['Q_plus'].astype(np.int64)
    q_minus_b = planes_base['Q_minus'].astype(np.int64)

    b_plus_p1  = planes_p1['B_plus'].astype(np.int64)
    b_minus_p1 = planes_p1['B_minus'].astype(np.int64)
    q_plus_p1  = planes_p1['Q_plus'].astype(np.int64)
    q_minus_p1 = planes_p1['Q_minus'].astype(np.int64)
    b_plus_p2  = planes_p2['B_plus'].astype(np.int64)
    b_minus_p2 = planes_p2['B_minus'].astype(np.int64)
    q_plus_p2  = planes_p2['Q_plus'].astype(np.int64)
    q_minus_p2 = planes_p2['Q_minus'].astype(np.int64)
    has_p3 = planes_p3 is not None
    if has_p3:
        b_plus_p3 = planes_p3['B_plus'].astype(np.int64)
        b_minus_p3 = planes_p3['B_minus'].astype(np.int64)
        q_plus_p3 = planes_p3['Q_plus'].astype(np.int64)
        q_minus_p3 = planes_p3['Q_minus'].astype(np.int64)

    if verbose:
        print("  Initialising FeFET model (building LUT)...")
    fefet = make_fefet_model()

    if verbose:
        print("  Sampling FeFET charges — baseline mapping...")
    q1bp_b, q1bm_b, q2bp_b, q2bm_b = _precompute_pe_charges(
        planes_base, fefet, sigma, dev)

    if verbose:
        print("  Sampling FeFET charges — proposed1 mapping...")
    q1bp_p1, q1bm_p1, q2bp_p1, q2bm_p1 = _precompute_pe_charges(
        planes_p1, fefet, sigma, dev)

    if verbose:
        print("  Sampling FeFET charges - proposed2 mapping...")
    q1bp_p2, q1bm_p2, q2bp_p2, q2bm_p2 = _precompute_pe_charges(
        planes_p2, fefet, sigma, dev)
    if has_p3:
        if verbose:
            print("  Sampling FeFET charges - proposed3 mapping...")
        q1bp_p3, q1bm_p3, q2bp_p3, q2bm_p3 = _precompute_pe_charges(
            planes_p3, fefet, sigma, dev)

    # Random INT8 test vectors: shape [N_vec, M]
    rng = np.random.default_rng(seed)
    X   = rng.integers(-128, 128, size=(N_vec, M), dtype=np.int8)

    cos_b_list,   l2_b_list,   exact_b_list   = [], [], []
    cos_p1_list,  l2_p1_list,  exact_p1_list  = [], [], []
    cos_p2_list,  l2_p2_list,  exact_p2_list  = [], [], []
    cos_p3_list,  l2_p3_list,  exact_p3_list  = [], [], []

    pe_err_1b_b = np.zeros(KB, dtype=np.int64)
    pe_err_1b_p1 = np.zeros(KB, dtype=np.int64)
    pe_err_1b_p2 = np.zeros(KB, dtype=np.int64)
    pe_err_2b_b = np.zeros(KQ, dtype=np.int64)
    pe_err_2b_p1 = np.zeros(KQ, dtype=np.int64)
    pe_err_2b_p2 = np.zeros(KQ, dtype=np.int64)
    pe_err_1b_p3 = np.zeros(KB, dtype=np.int64)
    pe_err_2b_p3 = np.zeros(KQ, dtype=np.int64)
    pe_total_1b = np.zeros(KB, dtype=np.int64)
    pe_total_2b = np.zeros(KQ, dtype=np.int64)

    if verbose:
        print(f"  Evaluating {N_vec} random INT8 vectors "
              f"(clip_1b=[0,{clip_1b}], clip_2b=[0,{clip_2b}])...")

    for i in range(N_vec):
        x     = X[i].astype(np.int32)
        y_ref = (W.T @ x.astype(np.int64)).astype(np.float64)   # exact [N]

        y_base, err1b_b, err2b_b, tot1, tot2 = _cim_output_quantised(
            x, q1bp_b, q1bm_b, q2bp_b, q2bm_b,
            b_plus_b, b_minus_b, q_plus_b, q_minus_b,
            lambda_B, lambda_Q, clip_1b, clip_2b)

        y_p1, err1b_p1, err2b_p1, _, _ = _cim_output_quantised(
            x, q1bp_p1, q1bm_p1, q2bp_p1, q2bm_p1,
            b_plus_p1, b_minus_p1, q_plus_p1, q_minus_p1,
            lambda_B, lambda_Q, clip_1b, clip_2b)

        y_p2, err1b_p2, err2b_p2, _, _ = _cim_output_quantised(
            x, q1bp_p2, q1bm_p2, q2bp_p2, q2bm_p2,
            b_plus_p2, b_minus_p2, q_plus_p2, q_minus_p2,
            lambda_B, lambda_Q, clip_1b, clip_2b)
        if has_p3:
            y_p3, err1b_p3, err2b_p3, _, _ = _cim_output_quantised(
                x, q1bp_p3, q1bm_p3, q2bp_p3, q2bm_p3,
                b_plus_p3, b_minus_p3, q_plus_p3, q_minus_p3,
                lambda_B, lambda_Q, clip_1b, clip_2b)

        pe_err_1b_b += err1b_b;   pe_err_1b_p1 += err1b_p1; pe_err_1b_p2 += err1b_p2
        pe_err_2b_b += err2b_b;   pe_err_2b_p1 += err2b_p1; pe_err_2b_p2 += err2b_p2
        if has_p3:
            pe_err_1b_p3 += err1b_p3
            pe_err_2b_p3 += err2b_p3
        pe_total_1b += tot1;      pe_total_2b += tot2

        c_b, l_b, e_b = _compute_metrics(y_base, y_ref)
        c_p1, l_p1, e_p1 = _compute_metrics(y_p1, y_ref)
        c_p2, l_p2, e_p2 = _compute_metrics(y_p2, y_ref)
        if has_p3:
            c_p3, l_p3, e_p3 = _compute_metrics(y_p3, y_ref)

        cos_b_list.append(c_b);   l2_b_list.append(l_b);   exact_b_list.append(e_b)
        cos_p1_list.append(c_p1); l2_p1_list.append(l_p1); exact_p1_list.append(e_p1)
        cos_p2_list.append(c_p2); l2_p2_list.append(l_p2); exact_p2_list.append(e_p2)
        if has_p3:
            cos_p3_list.append(c_p3); l2_p3_list.append(l_p3); exact_p3_list.append(e_p3)

    denom_1b = np.maximum(pe_total_1b, 1).astype(np.float64)
    denom_2b = np.maximum(pe_total_2b, 1).astype(np.float64)

    return {
        # Primary accuracy
        'baseline_exact_rate':     float(np.mean(exact_b_list)),
        'proposed1_exact_rate':    float(np.mean(exact_p1_list)),
        'proposed2_exact_rate':    float(np.mean(exact_p2_list)),
        'proposed3_exact_rate':    float(np.mean(exact_p3_list)) if has_p3 else np.nan,
        # Continuous metrics
        'baseline_cos_mean':       float(np.mean(cos_b_list)),
        'baseline_cos_std':        float(np.std(cos_b_list)),
        'baseline_rel_l2_mean':    float(np.mean(l2_b_list)),
        'baseline_rel_l2_std':     float(np.std(l2_b_list)),
        'proposed1_cos_mean':      float(np.mean(cos_p1_list)),
        'proposed1_cos_std':       float(np.std(cos_p1_list)),
        'proposed1_rel_l2_mean':   float(np.mean(l2_p1_list)),
        'proposed1_rel_l2_std':    float(np.std(l2_p1_list)),
        'proposed2_cos_mean':      float(np.mean(cos_p2_list)),
        'proposed2_cos_std':       float(np.std(cos_p2_list)),
        'proposed2_rel_l2_mean':   float(np.mean(l2_p2_list)),
        'proposed2_rel_l2_std':    float(np.std(l2_p2_list)),
        'proposed3_cos_mean':      float(np.mean(cos_p3_list)) if has_p3 else np.nan,
        'proposed3_cos_std':       float(np.std(cos_p3_list)) if has_p3 else np.nan,
        'proposed3_rel_l2_mean':   float(np.mean(l2_p3_list)) if has_p3 else np.nan,
        'proposed3_rel_l2_std':    float(np.std(l2_p3_list)) if has_p3 else np.nan,
        # Per-plane PE error rates
        'baseline_pe_rate_1b':     (pe_err_1b_b / denom_1b).tolist(),
        'proposed1_pe_rate_1b':    (pe_err_1b_p1 / denom_1b).tolist(),
        'proposed2_pe_rate_1b':    (pe_err_1b_p2 / denom_1b).tolist(),
        'proposed3_pe_rate_1b':    (pe_err_1b_p3 / denom_1b).tolist() if has_p3 else [np.nan] * KB,
        'baseline_pe_rate_2b':     (pe_err_2b_b / denom_2b).tolist(),
        'proposed1_pe_rate_2b':    (pe_err_2b_p1 / denom_2b).tolist(),
        'proposed2_pe_rate_2b':    (pe_err_2b_p2 / denom_2b).tolist(),
        'proposed3_pe_rate_2b':    (pe_err_2b_p3 / denom_2b).tolist() if has_p3 else [np.nan] * KQ,
        # Metadata
        'lambda_B': lambda_B,
        'lambda_Q': lambda_Q,
        'N_vec':    N_vec,
        'sigma':    sigma,
        'clip_1b':  clip_1b,
        'clip_2b':  clip_2b,
    }


# ---------------------------------------------------------------------------
# Print helper
# ---------------------------------------------------------------------------

def print_accuracy_comparison(result):
    """Print a formatted accuracy comparison table (3-way or 4-way)."""
    lB = result['lambda_B']
    lQ = result['lambda_Q']
    has_p3 = np.isfinite(result.get('proposed3_exact_rate', np.nan))

    print(f"\n{'='*62}")
    print("CIM MAC Accuracy Comparison  (quantised-integer model)")
    print(f"  N_vec={result['N_vec']},  σ_scale={result['sigma']:.1f},  "
          f"clip_1b=[0,{result['clip_1b']}],  clip_2b=[0,{result['clip_2b']}]")

    if has_p3:
        print(f"\n  {'Metric':<34} {'Conventional':>12} {'Proposed1':>12} {'Proposed2':>12} {'Proposed3':>12}")
        print(f"  {'-'*90}")
    else:
        print(f"\n  {'Metric':<34} {'Conventional':>12} {'Proposed1':>12} {'Proposed2':>12}")
        print(f"  {'-'*76}")

    # Primary
    b_ex = result['baseline_exact_rate']
    p1_ex = result['proposed1_exact_rate']
    p2_ex = result['proposed2_exact_rate']
    if has_p3:
        p3_ex = result['proposed3_exact_rate']
        print(f"  {'MAC exact match rate':<34} {b_ex:>12.4f} {p1_ex:>12.4f} {p2_ex:>12.4f} {p3_ex:>12.4f}")
    else:
        print(f"  {'MAC exact match rate':<34} {b_ex:>12.4f} {p1_ex:>12.4f} {p2_ex:>12.4f}")

    # Continuous
    if has_p3:
        print(f"\n  {'Cosine Similarity  (mean)':<34} "
              f"{result['baseline_cos_mean']:>12.6f} {result['proposed1_cos_mean']:>12.6f} {result['proposed2_cos_mean']:>12.6f} {result['proposed3_cos_mean']:>12.6f}")
        print(f"  {'Cosine Similarity  (std)':<34} "
              f"{result['baseline_cos_std']:>12.6f} {result['proposed1_cos_std']:>12.6f} {result['proposed2_cos_std']:>12.6f} {result['proposed3_cos_std']:>12.6f}")
        print(f"  {'Rel. L2 Error      (mean)':<34} "
              f"{result['baseline_rel_l2_mean']:>12.6f} {result['proposed1_rel_l2_mean']:>12.6f} {result['proposed2_rel_l2_mean']:>12.6f} {result['proposed3_rel_l2_mean']:>12.6f}")
        print(f"  {'Rel. L2 Error      (std)':<34} "
              f"{result['baseline_rel_l2_std']:>12.6f} {result['proposed1_rel_l2_std']:>12.6f} {result['proposed2_rel_l2_std']:>12.6f} {result['proposed3_rel_l2_std']:>12.6f}")
    else:
        print(f"\n  {'Cosine Similarity  (mean)':<34} "
              f"{result['baseline_cos_mean']:>12.6f} {result['proposed1_cos_mean']:>12.6f} {result['proposed2_cos_mean']:>12.6f}")
        print(f"  {'Cosine Similarity  (std)':<34} "
              f"{result['baseline_cos_std']:>12.6f} {result['proposed1_cos_std']:>12.6f} {result['proposed2_cos_std']:>12.6f}")
        print(f"  {'Rel. L2 Error      (mean)':<34} "
              f"{result['baseline_rel_l2_mean']:>12.6f} {result['proposed1_rel_l2_mean']:>12.6f} {result['proposed2_rel_l2_mean']:>12.6f}")
        print(f"  {'Rel. L2 Error      (std)':<34} "
              f"{result['baseline_rel_l2_std']:>12.6f} {result['proposed1_rel_l2_std']:>12.6f} {result['proposed2_rel_l2_std']:>12.6f}")

    # Per-plane PE error rates
    print(f"\n  Per-plane PE error rate  (p-side + m-side combined):")
    if has_p3:
        print(f"  {'Plane':<28} {'lambda':>7}  {'Conv':>10} {'P1':>10} {'P2':>10} {'P3':>10}")
        print(f"  {'-'*86}")
    else:
        print(f"  {'Plane':<28} {'lambda':>7}  {'Conv':>10} {'P1':>10} {'P2':>10}")
        print(f"  {'-'*74}")
    for m in range(len(lB)):
        b_r = result['baseline_pe_rate_1b'][m]
        p1_r = result['proposed1_pe_rate_1b'][m]
        p2_r = result['proposed2_pe_rate_1b'][m]
        if has_p3:
            p3_r = result['proposed3_pe_rate_1b'][m]
            print(f"  {'1-bit plane '+str(m):<28} {lB[m]:>5d}  "
                  f"{b_r:>10.4%} {p1_r:>10.4%} {p2_r:>10.4%} {p3_r:>10.4%}")
        else:
            print(f"  {'1-bit plane '+str(m):<28} {lB[m]:>5d}  "
                  f"{b_r:>10.4%} {p1_r:>10.4%} {p2_r:>10.4%}")
    for t in range(len(lQ)):
        b_r = result['baseline_pe_rate_2b'][t]
        p1_r = result['proposed1_pe_rate_2b'][t]
        p2_r = result['proposed2_pe_rate_2b'][t]
        if has_p3:
            p3_r = result['proposed3_pe_rate_2b'][t]
            print(f"  {'2-bit plane '+str(t):<28} {lQ[t]:>5d}  "
                  f"{b_r:>10.4%} {p1_r:>10.4%} {p2_r:>10.4%} {p3_r:>10.4%}")
        else:
            print(f"  {'2-bit plane '+str(t):<28} {lQ[t]:>5d}  "
                  f"{b_r:>10.4%} {p1_r:>10.4%} {p2_r:>10.4%}")


# ---------------------------------------------------------------------------
# Standalone entry point for quick testing
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from decompose         import decompose_weight_matrix
    from baseline_mapping  import conventional_map
    from accuracy_optimizer import optimize_mapping_proposed1
    from mac_accuracy_optimizer import optimize_mapping_proposed2
    from mac_accuracy_optimizer_p3 import optimize_mapping_proposed3
    from column_stats      import load_calibration

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    rng = np.random.default_rng(2026)
    W   = rng.integers(-cfg.W_MAX, cfg.W_MAX + 1,
                        size=(cfg.COLUMN_SIZE, cfg.COLUMN_SIZE))

    try:
        cal = load_calibration()
    except FileNotFoundError:
        print("No calibration file — using placeholder thresholds.")
        cal = {'N_th_1b': 32.0, 'N_th_2b': 40.0,
               'kappa_2': 1.5,  'kappa_3': 2.5}

    print("Mapping conventional...")
    res_base = conventional_map(W, cal=cal)

    print("Optimising Proposed1 mapping (3 iters)...")
    res_p1 = optimize_mapping_proposed1(W, cal=cal, max_iter=3, verbose=False)
    print("Optimising Proposed2 mapping (3 iters)...")
    res_p2 = optimize_mapping_proposed2(W, cal=cal, max_iter=3, verbose=False)
    print("Optimising Proposed3 mapping (2 iters)...")
    res_p3 = optimize_mapping_proposed3(W, cal=cal, max_iter=2, verbose=False)

    print("\nRunning CIM accuracy evaluation (quantised-integer model)...")
    result = evaluate_accuracy(
        W,
        planes_base=res_base['planes'],
        planes_p1=res_p1['planes'],
        planes_p2=res_p2['planes'],
        planes_p3=res_p3['planes'],
        sigma=cfg.VTH_SIGMA_SCALE,
        N_vec=1000,
        seed=2026,
        verbose=True,
    )
    print_accuracy_comparison(result)
