"""
Fast analytical calibration — replaces Phase 1 Monte Carlo entirely.

All single-device statistics (μ_Q, σ²_Q) are obtained by numerical
integration over the LUT (Gauss-Legendre quadrature on the Vth
distribution), not by sampling.  Column-sum distributions are treated
as Gaussian (exact for sums of independent normals; CLT-justified for
M=64 cells).  Error probabilities are computed via erfc.

Outputs the same calibration_params.npz as the MC pipeline so that
Phases 2 and 3 are completely unaffected.

Derivation recap
----------------
Single device (state s, 2-bit read):
  Vth ~ N(μ_vth_s, (scale·σ_vth_s)²)
  Q   = f_lut(Vth)          [from Q2bit or Q1bit LUT]
  μ_s   = E[Q]              [LUT-weighted integral]
  σ²_s  = Var[Q] + σ_r²·μ_s²   [Vth variation + R variation]

Column sum (independent cells):
  Y = Σ Q_i  ~  N(Σ μ_i,  Σ σ²_i)   (sum of independent normals)

Error probability (midpoint decision boundary τ = (μ_n + μ_{n+1})/2):
  p_err(n) = ½·erfc(Δμ / (2√2·σ_n))
           + ½·erfc(Δμ / (2√2·σ_{n+1}))

  1-bit:  Δμ = μ_on - μ_off  (constant; adding one ON, removing one OFF)
  2-bit:  canonical path n1=k, n2=n3=0  →  Δμ = μ_cv1 - μ_cv0

kappa (charge variance ratio):
  κ_s = σ²_cv_s / σ²_cv1    (s = 2, 3;  cell-value indexing)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.special import erfc
from scipy.optimize import brentq
import config_inno2 as cfg


# ---------------------------------------------------------------------------
# LUT builder (same physics as col_mc_sim._build_lut)
# ---------------------------------------------------------------------------

def _build_lut():
    """Return Q-LUT dict (same as col_mc_sim._build_lut)."""
    SS     = cfg.FEFET_SS
    I0     = cfg.FEFET_I0
    R      = cfg.FEFET_R_LIMIT
    Vd     = cfg.FEFET_VD
    vth    = np.array(cfg.FEFET_VTH_STATES,  dtype=np.float64)
    sigma  = np.array(cfg.FEFET_SIGMA_VTH,   dtype=np.float64)
    Q0     = cfg.FEFET_Q0
    t_step = cfg.FEFET_T_STEP

    def _i_fefet(Vgs, Vds, Vth):
        if Vds <= 0: return 0.0
        return max(I0 * 10.0 ** ((Vgs-Vth)/SS) * (1-np.exp(-Vds/0.02585)), 0.0)

    def _solve(Vg, Vth):
        def res(Vsf): return _i_fefet(Vg-Vsf, Vd-Vsf, Vth) - Vsf/R
        if res(0.0) <= 0: return 0.0
        if res(Vd-1e-9) >= 0: return (Vd-1e-9)/R
        return brentq(res, 0.0, Vd-1e-9, xtol=1e-14) / R

    n_lut  = 3000
    margin = 5.0 * sigma.max() + 0.15
    vth_ax = np.linspace(vth.min()-margin, vth.max()+margin, n_lut)

    Q2bit = np.array([
        sum(_solve(vg, v) for vg in cfg.FEFET_VG_STEPS_2BIT) * t_step / Q0
        for v in vth_ax], dtype=np.float64)

    Q1bit = np.array([
        _solve(cfg.FEFET_VG_READ_1BIT, v) * t_step / Q0
        for v in vth_ax], dtype=np.float64)

    return dict(vth_ax=vth_ax, Q1bit=Q1bit, Q2bit=Q2bit,
                vth=vth, sigma=sigma,
                val2state=np.array([3, 2, 1, 0], dtype=np.int32),
                sig_r=float(cfg.FEFET_SIGMA_R_REL))


# ---------------------------------------------------------------------------
# Single-device statistics via numerical LUT integration
# ---------------------------------------------------------------------------

def _gauss_lut_stats(mu_vth, sig_vth, vth_ax, Q_lut, sig_r, n_quad=400):
    """
    Compute E[Q] and Var[Q] for one device state.

    Vth ~ N(mu_vth, sig_vth²); Q = f_lut(Vth).
    Uses a uniform grid over ±4σ weighted by the Gaussian density
    (equivalent to Gauss-Hermite quadrature).

    R-variation contribution is added analytically:
      Var_total = Var_Q + σ_r² · E[Q]²

    Returns (mu_Q, var_Q_total) as floats.
    """
    lo = mu_vth - 4.0 * sig_vth
    hi = mu_vth + 4.0 * sig_vth
    vth_pts = np.linspace(lo, hi, n_quad)

    # Gaussian weights (unnormalised)
    w = np.exp(-0.5 * ((vth_pts - mu_vth) / sig_vth) ** 2)
    w /= w.sum()

    Q_vals = np.interp(vth_pts, vth_ax, Q_lut)
    mu_Q   = float((w * Q_vals).sum())
    var_Q  = float((w * Q_vals**2).sum() - mu_Q**2)
    var_Q  = max(var_Q, 0.0)    # numerical safety

    # R variation: Q_eff = Q / (1+ε), ε~N(0,σ_r²) independent of Q
    # Var[Q_eff] ≈ Var[Q] + σ_r² · E[Q]²  (first-order, σ_r << 1)
    var_total = var_Q + (sig_r * mu_Q) ** 2

    return mu_Q, var_total


def compute_device_stats(lut=None, sigma_scale=None, n_quad=400, verbose=False):
    """
    Compute single-device output statistics for all relevant states.

    Parameters
    ----------
    lut         : LUT dict from _build_lut(); built internally if None.
    sigma_scale : Vth variation scale (default cfg.VTH_SIGMA_SCALE).
    n_quad      : quadrature points per state (400 is more than sufficient).

    Returns
    -------
    stats_1bit : dict with keys 'on', 'off'
                 each {'mu': float, 'var': float}
    stats_2bit : dict with keys 0,1,2,3 (cell values)
                 each {'mu': float, 'var': float}
    """
    if lut          is None: lut          = _build_lut()
    if sigma_scale  is None: sigma_scale  = cfg.VTH_SIGMA_SCALE

    vth_ax    = lut['vth_ax']
    vth       = lut['vth']          # [4]  nominal Vth per device state
    sigma     = lut['sigma']        # [4]  sigma_vth per device state
    val2state = lut['val2state']    # cell val -> device state
    sig_r     = lut['sig_r']

    # ── 2-bit stats (cell values 0,1,2,3) ──────────────────────────────
    stats_2bit = {}
    for cv in range(4):
        state = int(val2state[cv])
        mu_v  = vth[state]
        sig_v = sigma[state] * sigma_scale
        mu_Q, var_Q = _gauss_lut_stats(mu_v, sig_v, vth_ax, lut['Q2bit'], sig_r, n_quad)
        stats_2bit[cv] = {'mu': mu_Q, 'var': var_Q}
        if verbose:
            print(f"  2-bit cv={cv} (state {state}): "
                  f"μ_Q={mu_Q:.4f}  σ_Q={var_Q**0.5:.4f}")

    # ── 1-bit stats (ON = state 0, OFF = state 3) ───────────────────────
    stats_1bit = {}
    for label, state in [('on', 0), ('off', 3)]:
        mu_v  = vth[state]
        sig_v = sigma[state] * sigma_scale
        mu_Q, var_Q = _gauss_lut_stats(mu_v, sig_v, vth_ax, lut['Q1bit'], sig_r, n_quad)
        stats_1bit[label] = {'mu': mu_Q, 'var': var_Q}
        if verbose:
            print(f"  1-bit {label} (state {state}): "
                  f"μ_Q={mu_Q:.4f}  σ_Q={var_Q**0.5:.4f}")

    return stats_1bit, stats_2bit


# ---------------------------------------------------------------------------
# Analytical error probability
# ---------------------------------------------------------------------------

def _p_err_two_gaussians(mu_curr, var_curr, mu_next, var_next):
    """
    Error probability when distinguishing N(mu_curr, var_curr) from
    N(mu_next, var_next) using midpoint decision boundary.

    p_err = ½·erfc(Δμ/(2√2·σ_curr)) + ½·erfc(Δμ/(2√2·σ_next))
    where Δμ = mu_next - mu_curr.
    """
    delta = mu_next - mu_curr
    if delta <= 0.0:
        return 0.5
    sig_curr = max(var_curr, 0.0) ** 0.5
    sig_next = max(var_next, 0.0) ** 0.5
    p = 0.5 * erfc(delta / (2.0 * np.sqrt(2.0) * sig_curr)) \
      + 0.5 * erfc(delta / (2.0 * np.sqrt(2.0) * sig_next))
    return float(np.clip(p, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 1-bit N_th
# ---------------------------------------------------------------------------

def compute_N_th_1bit(stats_1bit, M=None, epsilon=None):
    """
    Analytical p_err curve for 1-bit PE and threshold N_th^{1b}.

    Compare Y_{n_on} vs Y_{n_on+1} for n_on = 0..M-1.

    Returns
    -------
    n_vals : ndarray [M]    n_on values 0..M-1
    p_err  : ndarray [M]    error probabilities
    N_th   : int            max n_on with p_err <= epsilon
    """
    if M       is None: M       = cfg.COLUMN_SIZE
    if epsilon is None: epsilon = cfg.EPSILON

    mu_on  = stats_1bit['on']['mu'];   var_on  = stats_1bit['on']['var']
    mu_off = stats_1bit['off']['mu'];  var_off = stats_1bit['off']['var']

    n_vals = np.arange(M, dtype=int)
    p_err  = np.zeros(M, dtype=float)

    for n in n_vals:
        mu_n    =  n    * mu_on + (M - n)     * mu_off
        mu_np1  = (n+1) * mu_on + (M - n - 1) * mu_off
        var_n   =  n    * var_on + (M - n)     * var_off
        var_np1 = (n+1) * var_on + (M - n - 1) * var_off
        p_err[n] = _p_err_two_gaussians(mu_n, var_n, mu_np1, var_np1)

    valid = np.where(p_err <= epsilon)[0]
    N_th  = int(n_vals[valid[-1]]) if len(valid) > 0 else 0

    return n_vals, p_err, N_th


# ---------------------------------------------------------------------------
# kappa (analytical)
# ---------------------------------------------------------------------------

def compute_kappa_analytical(stats_2bit):
    """
    kappa_s = σ²_cv_s / σ²_cv_1   (s = 2, 3; cell-value indexing)

    Derived directly from single-device charge variances.
    Guarantees kappa_3 >= kappa_2 >= 1 for physical devices
    (higher cell value = higher current = larger absolute Q variation).
    """
    var_1 = stats_2bit[1]['var']
    kappa_2 = float(stats_2bit[2]['var'] / var_1)
    kappa_3 = float(stats_2bit[3]['var'] / var_1)

    # Enforce physical ordering (safety clamp; should hold naturally)
    kappa_2 = max(kappa_2, 1.0)
    kappa_3 = max(kappa_3, kappa_2)

    return kappa_2, kappa_3


# ---------------------------------------------------------------------------
# 2-bit N_th  (canonical path: n1=k, n2=n3=0)
# ---------------------------------------------------------------------------

def compute_N_th_2bit(stats_2bit, M=None, epsilon=None):
    """
    Analytical p_err curve for 2-bit PE along the canonical path
    (n1 = k, n2 = n3 = 0, n0 = M-k) and threshold N_th^{2b}.

    N_th^{2b} is expressed in units of "state-1 equivalent active cells",
    consistent with n_eq = n1 + κ_2·n2 + κ_3·n3.

    Returns
    -------
    n_eq_vals : ndarray [M]    k values 0..M-1
    p_err     : ndarray [M]    error probabilities
    N_th      : float          max k with p_err <= epsilon
    """
    if M       is None: M       = cfg.COLUMN_SIZE
    if epsilon is None: epsilon = cfg.EPSILON

    # Cell value 0 = off state; cell value 1 = first active state
    mu_0  = stats_2bit[0]['mu'];  var_0 = stats_2bit[0]['var']
    mu_1  = stats_2bit[1]['mu'];  var_1 = stats_2bit[1]['var']

    n_vals = np.arange(M, dtype=int)
    p_err  = np.zeros(M, dtype=float)

    for k in n_vals:
        mu_k    =  k    * mu_1 + (M - k)     * mu_0
        mu_kp1  = (k+1) * mu_1 + (M - k - 1) * mu_0
        var_k   =  k    * var_1 + (M - k)     * var_0
        var_kp1 = (k+1) * var_1 + (M - k - 1) * var_0
        p_err[k] = _p_err_two_gaussians(mu_k, var_k, mu_kp1, var_kp1)

    valid = np.where(p_err <= epsilon)[0]
    N_th  = float(n_vals[valid[-1]]) if len(valid) > 0 else 0.0

    return n_vals.astype(float), p_err, N_th


# ---------------------------------------------------------------------------
# Full analytical calibration pipeline
# ---------------------------------------------------------------------------

def run_analytical_calibration(sigma_scale=None, epsilon=None,
                                results_dir=None, verbose=True):
    """
    Run full analytical calibration and save calibration_params.npz.

    Equivalent output to running col_mc_sim → error_prob → kappa_calibration,
    but completes in seconds with no Monte Carlo sampling.

    Returns
    -------
    cal : dict with keys N_th_1b, N_th_2b, kappa_2, kappa_3
    """
    if sigma_scale  is None: sigma_scale  = cfg.VTH_SIGMA_SCALE
    if epsilon      is None: epsilon      = cfg.EPSILON
    if results_dir  is None:
        results_dir = os.path.join(os.path.dirname(__file__), 'Results')
    os.makedirs(results_dir, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("Analytical calibration (fast mode)")
        print(f"  sigma_scale={sigma_scale}, epsilon={epsilon}, M={cfg.COLUMN_SIZE}")

    # Step 1: build LUT and compute device statistics
    lut = _build_lut()
    stats_1bit, stats_2bit = compute_device_stats(
        lut, sigma_scale, verbose=verbose)

    if verbose:
        print("\n  Device stats (μ_Q):")
        print(f"    1-bit ON  : {stats_1bit['on']['mu']:.4f}  "
              f"σ={stats_1bit['on']['var']**0.5:.4f}")
        print(f"    1-bit OFF : {stats_1bit['off']['mu']:.4f}  "
              f"σ={stats_1bit['off']['var']**0.5:.4f}")
        for cv in range(4):
            print(f"    2-bit cv={cv}: {stats_2bit[cv]['mu']:.4f}  "
                  f"σ={stats_2bit[cv]['var']**0.5:.4f}")

    # Step 2: kappa (analytical)
    kappa_2, kappa_3 = compute_kappa_analytical(stats_2bit)
    if verbose:
        print(f"\n  kappa_2 = {kappa_2:.4f}")
        print(f"  kappa_3 = {kappa_3:.4f}")

    # Step 3: 1-bit N_th
    n_1b, p_err_1b, N_th_1b = compute_N_th_1bit(
        stats_1bit, epsilon=epsilon)
    if verbose:
        print(f"\n  N_th^1b = {N_th_1b}  (epsilon={epsilon})")

    # Step 4: 2-bit N_th
    neq_2b, p_err_2b, N_th_2b = compute_N_th_2bit(
        stats_2bit, epsilon=epsilon)
    if verbose:
        print(f"  N_th^2b = {N_th_2b:.2f}  (epsilon={epsilon})")

    # Step 5: save to same file as MC pipeline
    cal_path = os.path.join(results_dir, 'calibration_params.npz')
    np.savez(cal_path,
             N_th_1bit=N_th_1b,
             N_th_2bit=N_th_2b,
             kappa_2=kappa_2,
             kappa_3=kappa_3)

    # Also save p_err curves for figure generation
    np.savez_compressed(os.path.join(results_dir, 'error_prob_1bit.npz'),
                        n=n_1b, p_err=p_err_1b, N_th=N_th_1b)
    np.savez_compressed(os.path.join(results_dir, 'error_prob_2bit.npz'),
                        n_eq=neq_2b, p_err=p_err_2b, N_th=N_th_2b,
                        kappa_2=kappa_2, kappa_3=kappa_3)

    # Save a kappa_final.npz as well (same format as kappa_calibration.py output)
    np.savez(os.path.join(results_dir, 'kappa_final.npz'),
             kappa_2=kappa_2, kappa_3=kappa_3,
             N_th_1bit=N_th_1b, N_th_2bit=N_th_2b,
             neq_grid=neq_2b, p_err_2b=p_err_2b)

    if verbose:
        print(f"\n  Saved -> {cal_path}")
        print("Analytical calibration done.")

    return {
        'N_th_1b':  N_th_1b,
        'N_th_2b':  N_th_2b,
        'kappa_2':  kappa_2,
        'kappa_3':  kappa_3,
        'stats_1bit': stats_1bit,
        'stats_2bit': stats_2bit,
        'n_1b':     n_1b,
        'p_err_1b': p_err_1b,
        'neq_2b':   neq_2b,
        'p_err_2b': p_err_2b,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    result = run_analytical_calibration(verbose=True)
    print(f"\nSummary:")
    print(f"  N_th^1b  = {result['N_th_1b']}")
    print(f"  N_th^2b  = {result['N_th_2b']:.2f}")
    print(f"  kappa_2  = {result['kappa_2']:.4f}")
    print(f"  kappa_3  = {result['kappa_3']:.4f}")
