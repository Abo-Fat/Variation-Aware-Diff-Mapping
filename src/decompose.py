"""
SDR (Signed Digit Representation) decomposition for mixed-precision CIM.

Each INT8 weight w ∈ [-W_MAX, W_MAX] is decomposed as:

  w = Σ_{m=0}^{K_B-1} λ_B[m] · d^B_m  +  Σ_{t=0}^{K_Q-1} λ_Q[t] · d^Q_t

where:
  λ_B[m] = 2^(2*K_Q + m),   d^B_m ∈ {-1, 0, +1}
  λ_Q[t] = 4^t,              d^Q_t ∈ {-3, -2, -1, 0, +1, +2, +3}

Positive / negative plane extraction (non-overlap guaranteed by construction):
  B^+_m = max(d^B_m, 0),    B^-_m = max(-d^B_m, 0)
  Q^+_t = max(d^Q_t, 0),    Q^-_t = max(-d^Q_t, 0)

Three mapping functions are provided:
  conventional_decompose  – u=0 non-redundant binary (unique representation)
  min_neq_decompose       – per-element SDR minimising equiv. active count
  (proposed starts from min_neq; see mapping_optimizer.py)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from itertools import product as iproduct
import config_inno2 as cfg


# ---------------------------------------------------------------------------
# SDR Lookup Table
# ---------------------------------------------------------------------------

def build_sdr_lut(KB=None, KQ=None):
    """
    Pre-enumerate all legal SDR candidates for each w ∈ [-W_MAX, W_MAX].

    Returns
    -------
    lut : dict  {int w: list of tuples}
        Each tuple is (d_B_0, …, d_B_{KB-1}, d_Q_0, …, d_Q_{KQ-1}).
    """
    if KB is None: KB = cfg.KB
    if KQ is None: KQ = cfg.KQ

    lambda_B = [2 ** (2 * KQ + m) for m in range(KB)]
    lambda_Q = [4 ** t             for t in range(KQ)]
    W_MAX    = cfg.W_MAX

    lut = {w: [] for w in range(-W_MAX, W_MAX + 1)}

    max_2b = sum(3 * lambda_Q[t] for t in range(KQ))

    for d_B in iproduct((-1, 0, 1), repeat=KB):
        s_B = sum(lambda_B[m] * d_B[m] for m in range(KB))
        # Prune: even extreme 2-bit values cannot reach any valid w
        if s_B - max_2b > W_MAX or s_B + max_2b < -W_MAX:
            continue
        for d_Q in iproduct((-3, -2, -1, 0, 1, 2, 3), repeat=KQ):
            w = s_B + sum(lambda_Q[t] * d_Q[t] for t in range(KQ))
            if -W_MAX <= w <= W_MAX:
                lut[w].append(d_B + d_Q)

    return lut


# ---------------------------------------------------------------------------
# Conventional mapping  (u = 0, unique non-redundant binary)
# ---------------------------------------------------------------------------

def conventional_decompose(W, KB=None, KQ=None):
    """
    Standard binary decomposition: decomposes |w| into positive digits,
    then applies the sign of w.  Equivalent to the old u=0 approach.

    For w ≥ 0: d^B_m ∈ {0,1}, d^Q_t ∈ {0,1,2,3}.
    For w < 0: all digits negated.

    Returns
    -------
    D_B : ndarray [KB, M, N]  int8, values in {-1, 0, +1}
    D_Q : ndarray [KQ, M, N]  int8, values in {-3,...,+3}
    """
    if KB is None: KB = cfg.KB
    if KQ is None: KQ = cfg.KQ

    W = np.asarray(W, dtype=np.int64)
    M, N = W.shape

    D_B = np.zeros((KB, M, N), dtype=np.int8)
    D_Q = np.zeros((KQ, M, N), dtype=np.int8)

    remainder = np.abs(W).copy()

    # Extract 1-bit MSB planes (highest weight first)
    for m in range(KB - 1, -1, -1):
        w_m = 2 ** (2 * KQ + m)
        D_B[m] = (remainder // w_m).astype(np.int8)
        remainder = remainder % w_m

    # Extract 2-bit LSB planes (highest weight first)
    for t in range(KQ - 1, -1, -1):
        w_t = 4 ** t
        D_Q[t] = (remainder // w_t).astype(np.int8)
        remainder = remainder % w_t

    assert np.all(remainder == 0), "conventional_decompose: non-zero remainder"

    # Apply sign of original weight
    sign = np.sign(W).astype(np.int8)
    for m in range(KB):
        D_B[m] *= sign
    for t in range(KQ):
        D_Q[t] *= sign

    return D_B, D_Q


# ---------------------------------------------------------------------------
# Two's complement mapping  (二补码)
# ---------------------------------------------------------------------------

def twos_complement_decompose(W, KB=None, KQ=None):
    """
    Two's complement decomposition for mixed-precision CIM.

    The MSB 1-bit plane (index KB-1, weight λ_B[KB-1]) acts as the sign bit:
      d^B[KB-1] ∈ {-1, 0}  →  B_minus active when weight is negative.
    All other planes carry non-negative digits:
      d^B[m<KB-1] ∈ {0, 1},   d^Q[t] ∈ {0, 1, 2, 3}.

    Representable TC range: [-λ_B[KB-1], W_MAX]  (= [-64, 127] for KB=3,KQ=2).
    For weights below -λ_B[KB-1] (outside TC range), falls back to
    sign-magnitude (conventional_decompose) so reconstruction is always exact.

    Key difference vs sign-magnitude (差分幅值码):
      Sign-magnitude: for w<0, ALL digits negative → only B_minus/Q_minus active.
      Two's complement: lower digits remain positive → B_plus/Q_plus also active,
      increasing column activity and thus CIM error exposure.

    Returns
    -------
    D_B : ndarray [KB, M, N]  int8, values in {-1, 0, +1}
    D_Q : ndarray [KQ, M, N]  int8, values in {-3,...,+3}
    """
    if KB is None: KB = cfg.KB
    if KQ is None: KQ = cfg.KQ

    lambda_B = [2 ** (2 * KQ + m) for m in range(KB)]
    sign_th  = lambda_B[KB - 1]   # threshold = λ_B[KB-1], e.g. 64

    W = np.asarray(W, dtype=np.int64)
    M, N = W.shape

    D_B = np.zeros((KB, M, N), dtype=np.int8)
    D_Q = np.zeros((KQ, M, N), dtype=np.int8)

    # ── Region masks ──────────────────────────────────────────────────────────
    pos_mask = W >= 0                          # [0, W_MAX]  : standard
    tc_mask  = (W < 0) & (W >= -sign_th)      # [-sign_th, -1]: exact TC
    sm_mask  = W < -sign_th                    # below TC range: SM fallback

    # ── Helper: decompose a non-negative integer array into planes ────────────
    def _decompose_nonneg(V, start_plane_B=0):
        """Fill D_B[start_plane_B..KB-1] and D_Q from non-negative V."""
        rem = V.copy()
        for m in range(KB - 1, start_plane_B - 1, -1):
            D_B[m] += ((rem // lambda_B[m]) * np.where(V > 0, 1, 0)).astype(np.int8)
            rem = rem % lambda_B[m]
        for t in range(KQ - 1, -1, -1):
            D_Q[t] += ((rem // (4 ** t)) * np.where(V > 0, 1, 0)).astype(np.int8)
            rem = rem % (4 ** t)

    # ── Positive weights ──────────────────────────────────────────────────────
    if np.any(pos_mask):
        W_p = np.where(pos_mask, W, np.int64(0))
        rem = W_p.copy()
        for m in range(KB - 1, -1, -1):
            D_B[m] = np.where(pos_mask,
                              (rem // lambda_B[m]).astype(np.int8), D_B[m])
            rem = rem % lambda_B[m]
        for t in range(KQ - 1, -1, -1):
            D_Q[t] = np.where(pos_mask,
                              (rem // (4 ** t)).astype(np.int8), D_Q[t])
            rem = rem % (4 ** t)

    # ── TC region: w in [-sign_th, -1] ───────────────────────────────────────
    if np.any(tc_mask):
        # Lower planes represent (w + sign_th) ∈ [0, sign_th-1] non-negatively
        W_tc = np.where(tc_mask, W + sign_th, np.int64(0))   # in [0, sign_th-1]
        rem  = W_tc.copy()
        for m in range(KB - 2, -1, -1):   # planes 0 .. KB-2 (skip MSB)
            D_B[m] = np.where(tc_mask,
                              (rem // lambda_B[m]).astype(np.int8), D_B[m])
            rem = rem % lambda_B[m]
        for t in range(KQ - 1, -1, -1):
            D_Q[t] = np.where(tc_mask,
                              (rem // (4 ** t)).astype(np.int8), D_Q[t])
            rem = rem % (4 ** t)
        # MSB is the sign bit: d^B[KB-1] = -1
        D_B[KB - 1] = np.where(tc_mask, np.int8(-1), D_B[KB - 1])

    # ── SM fallback: w < -sign_th ─────────────────────────────────────────────
    if np.any(sm_mask):
        W_sm = np.where(sm_mask, np.abs(W), np.int64(0))
        rem  = W_sm.copy()
        D_B_sm = np.zeros((KB, M, N), dtype=np.int8)
        D_Q_sm = np.zeros((KQ, M, N), dtype=np.int8)
        for m in range(KB - 1, -1, -1):
            D_B_sm[m] = (rem // lambda_B[m]).astype(np.int8)
            rem = rem % lambda_B[m]
        for t in range(KQ - 1, -1, -1):
            D_Q_sm[t] = (rem // (4 ** t)).astype(np.int8)
            rem = rem % (4 ** t)
        for m in range(KB):
            D_B[m] = np.where(sm_mask, -D_B_sm[m], D_B[m])
        for t in range(KQ):
            D_Q[t] = np.where(sm_mask, -D_Q_sm[t], D_Q[t])

    return D_B, D_Q


# ---------------------------------------------------------------------------
# Min-neq SDR baseline
# ---------------------------------------------------------------------------

def _element_neq(cand, KB, kappa_2, kappa_3):
    """Per-element equivalent active count: Σ|d^B| + Σ φ(|d^Q|)."""
    total = float(sum(abs(cand[m]) for m in range(KB)))
    for t in range(KB, KB + (len(cand) - KB)):
        v = abs(cand[t])
        if   v == 0: pass
        elif v == 1: total += 1.0
        elif v == 2: total += kappa_2
        else:        total += kappa_3
    return total


def min_neq_decompose(W, lut, kappa_2, kappa_3, KB=None, KQ=None):
    """
    Min-neq SDR: for each element independently, choose the SDR candidate
    from lut[w] that minimises the per-element equivalent active count.

    Operates weight-value-wise (up to 255 distinct values for INT8),
    so this is efficient regardless of matrix size.

    Returns
    -------
    D_B : ndarray [KB, M, N]  int8
    D_Q : ndarray [KQ, M, N]  int8
    """
    if KB is None: KB = cfg.KB
    if KQ is None: KQ = cfg.KQ

    W = np.asarray(W, dtype=np.int64)
    M, N = W.shape

    D_B = np.zeros((KB, M, N), dtype=np.int8)
    D_Q = np.zeros((KQ, M, N), dtype=np.int8)

    # Pre-compute best candidate for each distinct weight value
    best_per_w = {}
    for w_val in np.unique(W):
        w_int  = int(w_val)
        cands  = lut[w_int]
        best   = min(cands, key=lambda c: _element_neq(c, KB, kappa_2, kappa_3))
        best_per_w[w_int] = best

    # Apply to matrix (vectorised per weight value)
    for w_int, best in best_per_w.items():
        mask = (W == w_int)
        for m in range(KB):
            D_B[m][mask] = best[m]
        for t in range(KQ):
            D_Q[t][mask] = best[KB + t]

    return D_B, D_Q


# ---------------------------------------------------------------------------
# SDR → plane matrices
# ---------------------------------------------------------------------------

def sdr_to_planes(D_B, D_Q):
    """
    Convert signed digit arrays to positive/negative plane arrays.

      B^+_m = max(D_B[m], 0),   B^-_m = max(-D_B[m], 0)
      Q^+_t = max(D_Q[t], 0),   Q^-_t = max(-D_Q[t], 0)

    The non-overlap constraint is automatically satisfied: for any cell,
    exactly one of B^+/B^- is nonzero (or both are zero).

    Returns dict with keys: B_plus, B_minus, Q_plus, Q_minus, D_B, D_Q.
    """
    return {
        'B_plus':  np.maximum( D_B, 0).astype(np.int8),
        'B_minus': np.maximum(-D_B, 0).astype(np.int8),
        'Q_plus':  np.maximum( D_Q, 0).astype(np.int8),
        'Q_minus': np.maximum(-D_Q, 0).astype(np.int8),
        'D_B':     D_B,
        'D_Q':     D_Q,
    }


# ---------------------------------------------------------------------------
# Reconstruction check
# ---------------------------------------------------------------------------

def verify_reconstruction(W, D_B, D_Q, KB=None, KQ=None):
    """
    Verify w = Σ_m λ_B[m] D_B[m] + Σ_t λ_Q[t] D_Q[t].
    Returns max absolute reconstruction error (should be 0).
    """
    if KB is None: KB = cfg.KB
    if KQ is None: KQ = cfg.KQ

    lambda_B = [2 ** (2 * KQ + m) for m in range(KB)]
    lambda_Q = [4 ** t             for t in range(KQ)]

    W_recon = np.zeros_like(W, dtype=np.int64)
    for m in range(KB):
        W_recon += lambda_B[m] * D_B[m].astype(np.int64)
    for t in range(KQ):
        W_recon += lambda_Q[t] * D_Q[t].astype(np.int64)

    return int(np.abs(W.astype(np.int64) - W_recon).max())


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

def _self_test():
    KB, KQ = cfg.KB, cfg.KQ
    print(f"Self-test: KB={KB}, KQ={KQ}, W_MAX={cfg.W_MAX}")

    lut = build_sdr_lut(KB, KQ)
    sizes = [len(lut[w]) for w in range(-cfg.W_MAX, cfg.W_MAX + 1)]
    print(f"  LUT: total={sum(sizes)} candidates, "
          f"avg={np.mean(sizes):.1f}/weight, max={max(sizes)}")
    assert all(s > 0 for s in sizes), "Some weight has no candidate!"

    rng = np.random.default_rng(42)
    W   = rng.integers(-cfg.W_MAX, cfg.W_MAX + 1, size=(16, 16))

    D_B, D_Q = conventional_decompose(W, KB, KQ)
    err = verify_reconstruction(W, D_B, D_Q, KB, KQ)
    assert err == 0, f"Conventional recon error = {err}"
    print(f"  conventional_decompose: max_err={err}  OK")

    D_B2, D_Q2 = min_neq_decompose(W, lut, 1.5, 2.5, KB, KQ)
    err2 = verify_reconstruction(W, D_B2, D_Q2, KB, KQ)
    assert err2 == 0, f"Min-neq recon error = {err2}"
    print(f"  min_neq_decompose:      max_err={err2}  OK")

    D_B_tc, D_Q_tc = twos_complement_decompose(W, KB, KQ)
    err_tc = verify_reconstruction(W, D_B_tc, D_Q_tc, KB, KQ)
    assert err_tc == 0, f"TC recon error = {err_tc}"
    print(f"  twos_complement_decompose: max_err={err_tc}  OK")

    planes = sdr_to_planes(D_B2, D_Q2)
    overlap_B = np.any((planes['B_plus'] > 0) & (planes['B_minus'] > 0))
    overlap_Q = np.any((planes['Q_plus'] > 0) & (planes['Q_minus'] > 0))
    assert not overlap_B and not overlap_Q, "Non-overlap violated!"
    print("  Non-overlap constraint:  OK")
    print("All self-tests passed.")


if __name__ == '__main__':
    _self_test()
