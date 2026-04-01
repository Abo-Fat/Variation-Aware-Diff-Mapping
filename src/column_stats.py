"""
Column active-count statistics and objective functions for SDR mapping.

All counts use the COMBINED (positive + negative) definition:

  n_(m,c)^(1b)    = Σ_i |D_B[m, i, c]|
  n_(t,c)^(2b,eq) = Σ_i φ(|D_Q[t, i, c]|)

where φ(0)=0, φ(1)=1, φ(2)=κ_2, φ(3)=κ_3.

Overload penalty (hinge):
  [x]_+ = max(x, 0)

Column risk  (α_m = λ_B[m]^2, β_t = λ_Q[t]^2):
  J_c = Σ_m α_m [n_(m,c)^(1b) - N_th^(1b)]_+
      + Σ_t β_t [n_(t,c)^(2b,eq) - N_th^(2b)]_+

Sparsity term:
  S_c = Σ_m α_m n_(m,c)^(1b)
      + Σ_t β_t n_(t,c)^(2b,eq)

Full objective:
  min  Σ_c J_c  +  η Σ_c S_c
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import config_inno2 as cfg


# ---------------------------------------------------------------------------
# φ function  (2-bit digit magnitude → equivalent active count)
# ---------------------------------------------------------------------------

def phi_eq(v, kappa_2, kappa_3):
    """
    Vectorised φ(v) for v = |d^Q| ∈ {0,1,2,3}.

    Parameters
    ----------
    v        : ndarray, integer magnitudes
    kappa_2  : float
    kappa_3  : float

    Returns
    -------
    ndarray, same shape as v, float64
    """
    v = np.asarray(v, dtype=np.int32)
    return np.where(v == 0, 0.0,
           np.where(v == 1, 1.0,
           np.where(v == 2, float(kappa_2),
                            float(kappa_3))))


# ---------------------------------------------------------------------------
# Column count computation
# ---------------------------------------------------------------------------

def count_1bit_columns(D_B):
    """
    n_(m,c)^(1b) = Σ_i |D_B[m, i, c]|

    Parameters
    ----------
    D_B : ndarray [KB, M, N]  int8, values in {-1, 0, +1}

    Returns
    -------
    n1b : ndarray [KB, N]  float64
    """
    return np.abs(D_B).astype(np.float64).sum(axis=1)   # [KB, N]


def count_2bit_eq_columns(D_Q, kappa_2=None, kappa_3=None):
    """
    n_(t,c)^(2b,eq) = Σ_i φ(|D_Q[t, i, c]|)

    Parameters
    ----------
    D_Q      : ndarray [KQ, M, N]  int8, values in {-3,...,+3}
    kappa_2  : float (default cfg.KAPPA_2)
    kappa_3  : float (default cfg.KAPPA_3)

    Returns
    -------
    neq : ndarray [KQ, N]  float64
    """
    if kappa_2 is None: kappa_2 = cfg.KAPPA_2
    if kappa_3 is None: kappa_3 = cfg.KAPPA_3

    phi = phi_eq(np.abs(D_Q), kappa_2, kappa_3)   # [KQ, M, N]
    return phi.sum(axis=1)                          # [KQ, N]


# ---------------------------------------------------------------------------
# Per-column risk J_c and sparsity S_c
# ---------------------------------------------------------------------------

def compute_J_c(n1b, neq, N_th_1b, N_th_2b, alpha=None, beta=None):
    """
    J_c = Σ_m α_m [n1b[m,c] - N_th_1b]_+
         + Σ_t β_t [neq[t,c]  - N_th_2b]_+

    Parameters
    ----------
    n1b   : ndarray [KB, N]
    neq   : ndarray [KQ, N]
    alpha : list[KB]  defaults to [λ_B[m]^2]
    beta  : list[KQ]  defaults to [λ_Q[t]^2]

    Returns
    -------
    J_c : ndarray [N]  float64
    """
    if alpha is None: alpha = [l ** 2 for l in cfg.LAMBDA_B]
    if beta  is None: beta  = [l ** 2 for l in cfg.LAMBDA_Q]

    N = n1b.shape[1]
    J = np.zeros(N, dtype=np.float64)
    for m, a in enumerate(alpha):
        J += a * np.maximum(n1b[m] - N_th_1b, 0.0)
    for t, b in enumerate(beta):
        J += b * np.maximum(neq[t]  - N_th_2b, 0.0)
    return J


def compute_S_c(n1b, neq, alpha=None, beta=None):
    """
    S_c = Σ_m α_m n1b[m,c]  +  Σ_t β_t neq[t,c]

    Returns
    -------
    S_c : ndarray [N]  float64
    """
    if alpha is None: alpha = [l ** 2 for l in cfg.LAMBDA_B]
    if beta  is None: beta  = [l ** 2 for l in cfg.LAMBDA_Q]

    N = n1b.shape[1]
    S = np.zeros(N, dtype=np.float64)
    for m, a in enumerate(alpha):
        S += a * n1b[m].astype(np.float64)
    for t, b in enumerate(beta):
        S += b * neq[t].astype(np.float64)
    return S


def compute_objective(J_c, S_c, eta=None):
    """Full objective = Σ_c J_c + η Σ_c S_c.  (legacy weighted S_c)"""
    if eta is None: eta = cfg.ETA
    return float(J_c.sum()) + eta * float(S_c.sum())


def compute_S_c_uniform(n1b, neq):
    """
    Uniform (unweighted) active-digit count per column.

      S_c_raw = Σ_m n1b[m,c]  +  Σ_t neq[t,c]

    Unlike compute_S_c, no α_m / β_t weighting is applied, so every
    active digit in every plane contributes equally.  This is the correct
    sparsity proxy for comparing mappings: min-neq correctly scores lower
    (sparser) than conventional under this metric.

    Parameters
    ----------
    n1b : ndarray [KB, N]
    neq : ndarray [KQ, N]

    Returns
    -------
    S_raw : ndarray [N]  float64
    """
    return n1b.sum(axis=0).astype(np.float64) + neq.sum(axis=0).astype(np.float64)


def compute_objective_corrected(J_c, n1b, neq, N_th_1b, N_th_2b, eta=None):
    """
    Corrected full objective = Σ_c J_c(weighted)  +  η Σ_c S_c(nth_normalised).

      obj = Σ_c J_c  +  η · Σ_c (Σ_m n1b[m,c]/N_th_1b + Σ_t neq[t,c]/N_th_2b)

    J_c keeps the error-importance weighting (α_m = λ_B[m]²) so MSB
    overloads are penalised more.  S_c uses N_th-normalised uniform counts
    so the sparsity incentive does not bias toward any particular plane.

    Under this metric min-neq correctly scores lower than conventional
    because J_total(min-neq) < J_total(conventional).

    Parameters
    ----------
    J_c      : ndarray [N]  weighted per-column risk (from compute_J_c)
    n1b      : ndarray [KB, N]
    neq      : ndarray [KQ, N]
    N_th_1b  : float
    N_th_2b  : float
    eta      : float  (defaults to cfg.ETA)

    Returns
    -------
    obj : float
    """
    if eta is None: eta = cfg.ETA
    d1 = max(float(N_th_1b), 1e-12)
    d2 = max(float(N_th_2b), 1e-12)
    S_norm = (n1b.sum(axis=0) / d1 + neq.sum(axis=0) / d2).sum()
    return float(J_c.sum()) + eta * float(S_norm)


# ---------------------------------------------------------------------------
# Mean counts per plane  (for Figure 2(a))
# ---------------------------------------------------------------------------

def compute_mean_n1b_per_plane(D_B):
    """
    Mean n_(m,c)^(1b) averaged over columns c, for each 1-bit plane m.

    Returns
    -------
    mean_n1b : ndarray [KB]  float64
    """
    return count_1bit_columns(D_B).mean(axis=1)   # [KB]


def compute_mean_neq_per_plane(D_Q, kappa_2=None, kappa_3=None):
    """
    Mean n_(t,c)^(2b,eq) averaged over columns c, for each 2-bit plane t.

    Returns
    -------
    mean_neq : ndarray [KQ]  float64
    """
    if kappa_2 is None: kappa_2 = cfg.KAPPA_2
    if kappa_3 is None: kappa_3 = cfg.KAPPA_3
    return count_2bit_eq_columns(D_Q, kappa_2, kappa_3).mean(axis=1)   # [KQ]


# ---------------------------------------------------------------------------
# Load calibration parameters from disk
# ---------------------------------------------------------------------------

def load_calibration(results_dir=None):
    """
    Load N_th_1bit, N_th_2bit, kappa_2, kappa_3 from kappa_final.npz
    (falls back to calibration_params.npz).

    Returns dict with keys: N_th_1b, N_th_2b, kappa_2, kappa_3.
    """
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), 'Results')

    kappa_path = os.path.join(results_dir, 'kappa_final.npz')
    cal_path   = os.path.join(results_dir, 'calibration_params.npz')

    if os.path.exists(kappa_path):
        d = np.load(kappa_path)
    elif os.path.exists(cal_path):
        d = np.load(cal_path)
    else:
        raise FileNotFoundError(
            "No calibration file found. Run Phase 1 first.")

    return {
        'N_th_1b': float(d['N_th_1bit']),
        'N_th_2b': float(d['N_th_2bit']),
        'kappa_2': float(d['kappa_2']),
        'kappa_3': float(d['kappa_3']),
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

def _self_test():
    from decompose import conventional_decompose

    rng = np.random.default_rng(42)
    W   = rng.integers(-cfg.W_MAX, cfg.W_MAX + 1, size=(64, 64))
    D_B, D_Q = conventional_decompose(W)

    N_th_1b = 32.0
    N_th_2b = 40.0
    k2, k3  = 1.5, 2.5

    n1b = count_1bit_columns(D_B)
    neq = count_2bit_eq_columns(D_Q, k2, k3)
    J   = compute_J_c(n1b, neq, N_th_1b, N_th_2b)
    S   = compute_S_c(n1b, neq)
    obj = compute_objective(J, S)

    print(f"Self-test: n1b shape={n1b.shape}, neq shape={neq.shape}")
    print(f"  J_c: mean={J.mean():.4f}, max={J.max():.4f}")
    print(f"  S_c: mean={S.mean():.4f}")
    print(f"  objective={obj:.4f}")

    mn1b = compute_mean_n1b_per_plane(D_B)
    mneq = compute_mean_neq_per_plane(D_Q, k2, k3)
    print(f"  mean_n1b per plane: {np.round(mn1b, 2)}")
    print(f"  mean_neq per plane: {np.round(mneq, 2)}")
    print("column_stats self-test passed.")


if __name__ == '__main__':
    _self_test()
