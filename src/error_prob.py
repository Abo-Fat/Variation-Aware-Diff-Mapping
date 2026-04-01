"""
Phase 1.2 — Error probability curves and N_th threshold determination.

Reads the Monte Carlo column-sum data from Results/col_mc_1bit.npz and
Results/col_mc_2bit.npz (produced by col_mc_sim.py) and computes:

1-bit PE:
  p_err^{1b}(n) = P(Y_n > tau_n) + P(Y_{n+1} < tau_n)
  where tau_n = midpoint of mean(Y_n) and mean(Y_{n+1}).
  N_th^{1b} = max{ n : p_err^{1b}(n) <= epsilon }

2-bit PE (before kappa calibration):
  For each combo (n1,n2,n3), define n_eq = n1 + kappa_2*n2 + kappa_3*n3
  using the initial kappa estimate kappa_s = sigma_s^2 / sigma_1^2.
  We bin combos by their n_eq value and estimate p_err by comparing
  adjacent bins. Threshold N_th^{2b} is defined similarly.

Outputs:
  Results/error_prob_1bit.npz  — n, p_err arrays
  Results/error_prob_2bit.npz  — n_eq, p_err arrays (using initial kappa)
  Results/calibration_params.npz — N_th_1bit, N_th_2bit, kappa_2_init, kappa_3_init
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import config_inno2 as cfg

# ---------------------------------------------------------------------------
# Initial kappa estimate from device variance ratios
# ---------------------------------------------------------------------------

def compute_initial_kappa():
    """
    kappa_s = sigma_s^2 / sigma_1^2   (s=2,3 in cell-value indexing)

    Cell value 1 -> device state 2  (sigma_vth[2])
    Cell value 2 -> device state 1  (sigma_vth[1])
    Cell value 3 -> device state 0  (sigma_vth[0])

    BUT for the column charge sum, the relevant noise is the charge variance,
    not the Vth variance. Since the Q-LUT is nearly linear in Vth over the
    working range, sigma_Q ≈ |dQ/dVth| * sigma_Vth.  We approximate
    kappa_s = sigma_Q_s^2 / sigma_Q_1^2 ≈ sigma_vth_s^2 / sigma_vth_1^2.

    Cell value convention:
      cell_val 0 -> device state 3 (off)
      cell_val 1 -> device state 2
      cell_val 2 -> device state 1
      cell_val 3 -> device state 0 (most on)
    """
    sigma = np.array(cfg.FEFET_SIGMA_VTH)   # [s0, s1, s2, s3]
    # val2state: cell val s -> device state (3-s)
    # state for cell val 1: device state 2  -> sigma[2]
    # state for cell val 2: device state 1  -> sigma[1]
    # state for cell val 3: device state 0  -> sigma[0]
    sigma_cv1 = sigma[2]   # cell val 1
    sigma_cv2 = sigma[1]   # cell val 2
    sigma_cv3 = sigma[0]   # cell val 3

    kappa_2 = (sigma_cv2 / sigma_cv1) ** 2
    kappa_3 = (sigma_cv3 / sigma_cv1) ** 2
    return float(kappa_2), float(kappa_3)


# ---------------------------------------------------------------------------
# 1-bit error probability
# ---------------------------------------------------------------------------

def compute_error_prob_1bit(n_on, Y, epsilon=None):
    """
    Compute p_err^{1b}(n) for n = 0..M-1 by comparing Y_n vs Y_{n+1}.

    Parameters
    ----------
    n_on : ndarray [M+1]  active cell counts
    Y    : ndarray [M+1, N_mc]  column sum samples
    epsilon : float  threshold (default cfg.EPSILON)

    Returns
    -------
    n_vals  : ndarray [M]   n values (0 to M-1)
    p_err   : ndarray [M]   error probability
    N_th    : int            max n with p_err <= epsilon
    """
    if epsilon is None:
        epsilon = cfg.EPSILON

    M = len(n_on) - 1
    n_vals = n_on[:M]
    p_err  = np.zeros(M, dtype=float)

    for i in range(M):
        Y_n  = Y[i]
        Y_n1 = Y[i + 1]

        # Decision boundary: midpoint of means
        mu_n  = Y_n.mean()
        mu_n1 = Y_n1.mean()
        tau   = 0.5 * (mu_n + mu_n1)

        # Error probability
        p_err[i] = (np.mean(Y_n > tau) + np.mean(Y_n1 < tau)) / 2.0

    # N_th = max n such that p_err(n) <= epsilon
    valid = np.where(p_err <= epsilon)[0]
    N_th = int(n_vals[valid[-1]]) if len(valid) > 0 else 0

    return n_vals, p_err, N_th


# ---------------------------------------------------------------------------
# 2-bit error probability (before kappa calibration)
# ---------------------------------------------------------------------------

def compute_error_prob_2bit_initial(combos_arr, Y, kappa_2, kappa_3,
                                    M=None, epsilon=None, n_eq_bins=None):
    """
    Compute p_err^{2b}(n_eq) using the initial kappa estimate.

    For each combo (n1,n2,n3) compute n_eq = n1 + kappa_2*n2 + kappa_3*n3,
    then bin by n_eq and compare adjacent bins.

    Parameters
    ----------
    combos_arr : ndarray [K,3]   (n1, n2, n3) configurations
    Y          : ndarray [K, N_mc]
    kappa_2    : float
    kappa_3    : float
    M          : int  column size
    epsilon    : float

    Returns
    -------
    neq_vals : ndarray  n_eq grid values
    p_err    : ndarray  error probability
    N_th_2b  : int      threshold
    """
    if M       is None: M       = cfg.COLUMN_SIZE
    if epsilon is None: epsilon = cfg.EPSILON

    n1 = combos_arr[:, 0].astype(float)
    n2 = combos_arr[:, 1].astype(float)
    n3 = combos_arr[:, 2].astype(float)
    n_eq = n1 + kappa_2 * n2 + kappa_3 * n3   # [K]

    # Build a mapping n_eq -> list of sample arrays
    # Bin into integer-rounded n_eq slots for tractability
    n_eq_rounded = np.round(n_eq).astype(int)
    max_neq = int(n_eq_rounded.max())

    # For each bin, pool all Y samples
    bins = {}
    for k, neq_k in enumerate(n_eq_rounded):
        if neq_k not in bins:
            bins[neq_k] = []
        bins[neq_k].append(Y[k])

    # Sort bins
    sorted_keys = sorted(bins.keys())
    neq_vals = np.array(sorted_keys, dtype=float)
    means    = np.array([np.concatenate(bins[k]).mean() for k in sorted_keys])

    # Compute p_err by comparing adjacent bins
    K_bins = len(sorted_keys)
    p_err = np.zeros(K_bins - 1, dtype=float)

    for i in range(K_bins - 1):
        Y_curr = np.concatenate(bins[sorted_keys[i]])
        Y_next = np.concatenate(bins[sorted_keys[i + 1]])

        mu_c = Y_curr.mean()
        mu_n = Y_next.mean()
        if mu_n <= mu_c:
            p_err[i] = 0.5
            continue
        tau = 0.5 * (mu_c + mu_n)
        p_err[i] = (np.mean(Y_curr > tau) + np.mean(Y_next < tau)) / 2.0

    # N_th_2b = max n_eq such that p_err(n_eq) <= epsilon
    neq_pairs = neq_vals[:K_bins - 1]
    valid = np.where(p_err <= epsilon)[0]
    N_th_2b = float(neq_pairs[valid[-1]]) if len(valid) > 0 else 0.0

    return neq_pairs, p_err, N_th_2b


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    out_dir  = os.path.join(os.path.dirname(__file__), 'Results')
    in_1bit  = os.path.join(out_dir, 'col_mc_1bit.npz')
    in_2bit  = os.path.join(out_dir, 'col_mc_2bit.npz')

    # ---- 1-bit -------------------------------------------------------
    print("=" * 60)
    print("Phase 1.2: 1-bit error probability")
    data1 = np.load(in_1bit)
    n_on  = data1['n_on']
    Y_1b  = data1['Y']
    n_vals, p_err_1b, N_th_1b = compute_error_prob_1bit(n_on, Y_1b)
    print(f"  N_th^1b = {N_th_1b}  (epsilon={cfg.EPSILON})")

    out_ep1 = os.path.join(out_dir, 'error_prob_1bit.npz')
    np.savez_compressed(out_ep1, n=n_vals, p_err=p_err_1b, N_th=N_th_1b)
    print(f"  Saved -> {out_ep1}")

    # ---- 2-bit -------------------------------------------------------
    print("=" * 60)
    print("Phase 1.2: 2-bit error probability (initial kappa)")
    data2      = np.load(in_2bit)
    combos_arr = data2['combos']     # [K,3]
    Y_2b       = data2['Y']

    kappa_2_init, kappa_3_init = compute_initial_kappa()
    print(f"  Initial kappa_2={kappa_2_init:.4f}, kappa_3={kappa_3_init:.4f}")

    neq_vals, p_err_2b, N_th_2b = compute_error_prob_2bit_initial(
        combos_arr, Y_2b, kappa_2_init, kappa_3_init)
    print(f"  N_th^2b = {N_th_2b:.2f}  (epsilon={cfg.EPSILON})")

    out_ep2 = os.path.join(out_dir, 'error_prob_2bit.npz')
    np.savez_compressed(out_ep2, n_eq=neq_vals, p_err=p_err_2b,
                        N_th=N_th_2b,
                        kappa_2=kappa_2_init, kappa_3=kappa_3_init)
    print(f"  Saved -> {out_ep2}")

    # ---- Summary -----------------------------------------------------
    out_cal = os.path.join(out_dir, 'calibration_params.npz')
    np.savez(out_cal,
             N_th_1bit=N_th_1b,
             N_th_2bit=N_th_2b,
             kappa_2=kappa_2_init,
             kappa_3=kappa_3_init)
    print(f"\nCalibration params saved -> {out_cal}")
    print(f"  N_th^1b  = {N_th_1b}")
    print(f"  N_th^2b  = {N_th_2b:.2f}")
    print(f"  kappa_2  = {kappa_2_init:.4f}")
    print(f"  kappa_3  = {kappa_3_init:.4f}")
    print("Done.")


if __name__ == '__main__':
    main()
