"""
CIM MAC Accuracy Evaluation — Innovation 2.

Simulates bit-serial CIM matrix-vector multiplication using the FeFET
charge model, comparing against exact integer arithmetic.

Computation scheme (1-bit serial input):
  - INT8 input vector x ∈ [-128, 127]^M decomposed into 8 bit planes
    (two's complement: bit k weight = 2^k for k=0..6, -128 for k=7)
  - FeFET variation is sampled ONCE per weight matrix (fixed-chip model),
    then reused across all N_VEC test vectors.
  - Per bit cycle k:
      input_bit[i] = (unsigned(x[i]) >> k) & 1   ∈ {0, 1}

      For each 1-bit PE plane m  (B_plus[m], B_minus[m]):
        S_p[j] = clip( Σ_i input_bit[i] * Q_eff_1b(B_plus[m][i,j]),  ±M )
        S_m[j] = clip( Σ_i input_bit[i] * Q_eff_1b(B_minus[m][i,j]), ±M )
        y_cycle += λ_B[m] * (S_p − S_m)

      For each 2-bit PE plane t  (Q_plus[t], Q_minus[t]):
        S_p[j] = clip( Σ_i input_bit[i] * Q_eff_2b(Q_plus[t][i,j]),  ±3M )
        S_m[j] = clip( Σ_i input_bit[i] * Q_eff_2b(Q_minus[t][i,j]), ±3M )
        y_cycle += λ_Q[t] * (S_p − S_m)

    y_CIM += bit_weight_k * y_cycle

Clip ranges:
  1-bit PE → ±M     = ±COLUMN_SIZE = ±64
  2-bit PE → ±3*M   = ±3*COLUMN_SIZE = ±192
  (With 1-bit serial input and Q_eff ≈ {0,1,2,3}, the theoretical
   per-cycle max is 3*M=192 for the positive side.)

Accuracy metrics (mean ± std over N_VEC random INT8 vectors):
  - Cosine similarity:  cos(y_CIM, y_ref)
  - Relative L2 error:  ‖y_CIM − y_ref‖₂ / ‖y_ref‖₂
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

    Uses mode='2bit' so that both compute_cell_charge_1bit and
    compute_cell_charge_2bit are fully initialised.
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
    Sample FeFET Vth variation ONCE per weight matrix and compute the
    normalised charge Q_eff for every PE plane.

    This models a fixed physical chip: variation is determined at
    manufacturing time and is the same for all input vectors.

    Parameters
    ----------
    planes : dict from decompose_weight_matrix()
    fefet  : FeFETVariationModel instance
    sigma  : float, vth_sigma_scale
    device : torch.device

    Returns
    -------
    q1b_p : list[KB]  each ndarray [M, N] float64, Q_eff ≈ {0, 1}
    q1b_m : list[KB]  each ndarray [M, N] float64
    q2b_p : list[KQ]  each ndarray [M, N] float64, Q_eff ≈ {0,1,2,3}
    q2b_m : list[KQ]  each ndarray [M, N] float64
    """
    KB = planes['B_plus'].shape[0]
    KQ = planes['Q_plus'].shape[0]

    q1b_p, q1b_m, q2b_p, q2b_m = [], [], [], []

    for m in range(KB):
        B_p = torch.tensor(
            planes['B_plus'][m].astype(np.float32), device=device)
        B_m = torch.tensor(
            planes['B_minus'][m].astype(np.float32), device=device)
        q1b_p.append(
            fefet.compute_cell_charge_1bit(B_p, sigma, device)
                 .cpu().numpy().astype(np.float64))
        q1b_m.append(
            fefet.compute_cell_charge_1bit(B_m, sigma, device)
                 .cpu().numpy().astype(np.float64))

    for t in range(KQ):
        Q_p = torch.tensor(
            planes['Q_plus'][t].astype(np.int64), device=device)
        Q_m = torch.tensor(
            planes['Q_minus'][t].astype(np.int64), device=device)
        q2b_p.append(
            fefet.compute_cell_charge_2bit(Q_p, sigma, device)
                 .cpu().numpy().astype(np.float64))
        q2b_m.append(
            fefet.compute_cell_charge_2bit(Q_m, sigma, device)
                 .cpu().numpy().astype(np.float64))

    return q1b_p, q1b_m, q2b_p, q2b_m


# ---------------------------------------------------------------------------
# CIM output for a single input vector
# ---------------------------------------------------------------------------

def _cim_output_one_vector(x, q1b_p, q1b_m, q2b_p, q2b_m,
                            lambda_B, lambda_Q, clip_1b, clip_2b):
    """
    Compute CIM MAC output for one INT8 input vector using pre-sampled
    FeFET charges.

    Parameters
    ----------
    x       : [M] int array, values in [-128, 127]
    q1b_p/m : list[KB] of [M, N] float64 arrays  (1-bit PE charges)
    q2b_p/m : list[KQ] of [M, N] float64 arrays  (2-bit PE charges)
    lambda_B : list[KB] float  (bit-plane weights from config)
    lambda_Q : list[KQ] float
    clip_1b  : float  column-sum clip for 1-bit PE (= M = 64)
    clip_2b  : float  column-sum clip for 2-bit PE (= 3*M = 192)

    Returns
    -------
    y_CIM : [N] float64 array
    """
    KB = len(q1b_p)
    KQ = len(q2b_p)
    N  = q1b_p[0].shape[1]

    # Map INT8 two's complement to unsigned [0, 255] for bit extraction
    x_uint = x.astype(np.int32) % 256

    y_CIM = np.zeros(N, dtype=np.float64)

    for k, w_k in enumerate(_BIT_WEIGHTS):
        # 1-bit input for cycle k: shape [M], values {0.0, 1.0}
        input_bit = ((x_uint >> k) & 1).astype(np.float64)

        y_cycle = np.zeros(N, dtype=np.float64)

        # --- 1-bit PE planes ---
        for m in range(KB):
            # Column accumulation: [M] @ [M, N] → [N]
            S_p = input_bit @ q1b_p[m]
            S_m = input_bit @ q1b_m[m]
            S_p = np.clip(S_p, -clip_1b, clip_1b)
            S_m = np.clip(S_m, -clip_1b, clip_1b)
            y_cycle += lambda_B[m] * (S_p - S_m)

        # --- 2-bit PE planes ---
        for t in range(KQ):
            S_p = input_bit @ q2b_p[t]
            S_m = input_bit @ q2b_m[t]
            S_p = np.clip(S_p, -clip_2b, clip_2b)
            S_m = np.clip(S_m, -clip_2b, clip_2b)
            y_cycle += lambda_Q[t] * (S_p - S_m)

        y_CIM += w_k * y_cycle

    return y_CIM


# ---------------------------------------------------------------------------
# Accuracy metrics
# ---------------------------------------------------------------------------

def _compute_metrics(y_cim, y_ref):
    """
    Compute cosine similarity and relative L2 error between CIM output
    and exact reference.

    Returns
    -------
    cos_sim : float in [-1, 1]  (1.0 = perfect)
    rel_l2  : float >= 0        (0.0 = perfect)
    """
    y_cim = np.asarray(y_cim, dtype=np.float64)
    y_ref = np.asarray(y_ref, dtype=np.float64)

    dot    = float(np.dot(y_cim, y_ref))
    norm_c = float(np.linalg.norm(y_cim))
    norm_r = float(np.linalg.norm(y_ref))

    cos_sim = dot / (norm_c * norm_r + 1e-12)
    rel_l2  = float(np.linalg.norm(y_cim - y_ref)) / (norm_r + 1e-12)

    return cos_sim, rel_l2


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def evaluate_accuracy(W, planes_base, planes_opt,
                      sigma=1.0, N_vec=100, seed=2026,
                      device='cpu', verbose=True):
    """
    Evaluate CIM MAC accuracy for baseline and proposed mappings.

    For each mapping, FeFET variation is sampled once (fixed-chip model).
    N_vec random INT8 vectors are then passed through each mapping and
    compared against the exact integer matrix-vector product.

    Parameters
    ----------
    W           : [M, N] int array  (original weight matrix)
    planes_base : dict from baseline_map()
    planes_opt  : dict from optimize_mapping()
    sigma       : float, vth_sigma_scale (default 1.0 = nominal σ)
    N_vec       : int, number of random INT8 test vectors (default 100)
    seed        : int, RNG seed for reproducible input generation
    device      : str, torch device string (default 'cpu')
    verbose     : bool

    Returns
    -------
    result : dict
      'baseline_cos_mean'    : mean cosine similarity   (baseline)
      'baseline_cos_std'     : std  cosine similarity   (baseline)
      'baseline_rel_l2_mean' : mean relative L2 error   (baseline)
      'baseline_rel_l2_std'  : std  relative L2 error   (baseline)
      'proposed_cos_mean'    : mean cosine similarity   (proposed)
      'proposed_cos_std'     : std  cosine similarity   (proposed)
      'proposed_rel_l2_mean' : mean relative L2 error   (proposed)
      'proposed_rel_l2_std'  : std  relative L2 error   (proposed)
      'N_vec', 'sigma', 'clip_1b', 'clip_2b'
    """
    W    = np.asarray(W, dtype=np.int64)
    M, N = W.shape
    dev  = torch.device(device)

    clip_1b  = float(M)             # ±M   = ±64
    clip_2b  = 3.0 * float(M)      # ±3M  = ±192
    lambda_B = list(cfg.LAMBDA_B)
    lambda_Q = list(cfg.LAMBDA_Q)

    if verbose:
        print("  Initialising FeFET model (building LUT)...")
    fefet = make_fefet_model()

    if verbose:
        print("  Sampling FeFET charges — baseline mapping...")
    q1bp_b, q1bm_b, q2bp_b, q2bm_b = _precompute_pe_charges(
        planes_base, fefet, sigma, dev)

    if verbose:
        print("  Sampling FeFET charges — proposed mapping...")
    q1bp_o, q1bm_o, q2bp_o, q2bm_o = _precompute_pe_charges(
        planes_opt, fefet, sigma, dev)

    # Random INT8 test vectors: shape [N_vec, M]
    rng = np.random.default_rng(seed)
    X   = rng.integers(-128, 128, size=(N_vec, M), dtype=np.int8)

    cos_b_list, l2_b_list = [], []
    cos_o_list, l2_o_list = [], []

    if verbose:
        print(f"  Evaluating {N_vec} random INT8 vectors "
              f"(clip_1b=±{clip_1b:.0f}, clip_2b=±{clip_2b:.0f})...")

    for i in range(N_vec):
        x     = X[i].astype(np.int32)
        y_ref = (W.T @ x.astype(np.int64)).astype(np.float64)  # exact [N]

        y_base = _cim_output_one_vector(
            x, q1bp_b, q1bm_b, q2bp_b, q2bm_b,
            lambda_B, lambda_Q, clip_1b, clip_2b)

        y_opt = _cim_output_one_vector(
            x, q1bp_o, q1bm_o, q2bp_o, q2bm_o,
            lambda_B, lambda_Q, clip_1b, clip_2b)

        c_b, l_b = _compute_metrics(y_base, y_ref)
        c_o, l_o = _compute_metrics(y_opt,  y_ref)

        cos_b_list.append(c_b);  l2_b_list.append(l_b)
        cos_o_list.append(c_o);  l2_o_list.append(l_o)

    return {
        'baseline_cos_mean':    float(np.mean(cos_b_list)),
        'baseline_cos_std':     float(np.std(cos_b_list)),
        'baseline_rel_l2_mean': float(np.mean(l2_b_list)),
        'baseline_rel_l2_std':  float(np.std(l2_b_list)),
        'proposed_cos_mean':    float(np.mean(cos_o_list)),
        'proposed_cos_std':     float(np.std(cos_o_list)),
        'proposed_rel_l2_mean': float(np.mean(l2_o_list)),
        'proposed_rel_l2_std':  float(np.std(l2_o_list)),
        'N_vec':   N_vec,
        'sigma':   sigma,
        'clip_1b': clip_1b,
        'clip_2b': clip_2b,
    }


# ---------------------------------------------------------------------------
# Print helper
# ---------------------------------------------------------------------------

def print_accuracy_comparison(result):
    """Print a formatted accuracy comparison table."""
    print(f"\n{'='*60}")
    print("CIM MAC Accuracy Comparison")
    print(f"  N_vec={result['N_vec']},  σ_scale={result['sigma']:.1f},  "
          f"clip_1b=±{result['clip_1b']:.0f},  clip_2b=±{result['clip_2b']:.0f}")
    print(f"\n  {'Metric':<30} {'Baseline':>11} {'Proposed':>11}")
    print(f"  {'-'*54}")
    rows = [
        ('Cosine Similarity  (mean)', 'baseline_cos_mean',    'proposed_cos_mean'),
        ('Cosine Similarity  (std)',  'baseline_cos_std',     'proposed_cos_std'),
        ('Rel. L2 Error      (mean)', 'baseline_rel_l2_mean', 'proposed_rel_l2_mean'),
        ('Rel. L2 Error      (std)',  'baseline_rel_l2_std',  'proposed_rel_l2_std'),
    ]
    for label, bk, pk in rows:
        print(f"  {label:<30} {result[bk]:>11.6f} {result[pk]:>11.6f}")

    delta_cos = result['proposed_cos_mean'] - result['baseline_cos_mean']
    delta_l2  = result['proposed_rel_l2_mean'] - result['baseline_rel_l2_mean']
    print(f"\n  Cosine similarity change:  {delta_cos:+.6f} "
          f"({'improved' if delta_cos > 0 else 'degraded'})")
    print(f"  Rel. L2 error change:      {delta_l2:+.6f} "
          f"({'improved' if delta_l2 < 0 else 'degraded'})")


# ---------------------------------------------------------------------------
# Standalone entry point for quick testing
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from decompose      import decompose_weight_matrix
    from baseline_mapping  import baseline_map
    from mapping_optimizer import optimize_mapping
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

    print("Mapping baseline...")
    res_base = baseline_map(W, cal=cal)

    print("Optimising mapping (5 iters)...")
    res_opt  = optimize_mapping(W, cal=cal, max_iter=5, verbose=False)

    print("\nRunning CIM accuracy evaluation...")
    result = evaluate_accuracy(
        W,
        planes_base=res_base['planes'],
        planes_opt=res_opt['planes'],
        sigma=cfg.VTH_SIGMA_SCALE,
        N_vec=100,
        seed=2026,
        verbose=True,
    )
    print_accuracy_comparison(result)
