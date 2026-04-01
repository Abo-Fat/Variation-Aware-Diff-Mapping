"""
Analytical per-column log-probability for CIM accuracy optimisation.

Under all-ones input (maximum-load scenario, every row activated):
  Each PE plane column has a physical analogue sum:
    S_phys ~ N(mu_S, var_S)
  where mu_S and var_S are determined by the cell state composition of
  that column, derived from FeFET single-device statistics.

  Ideal integer readout: S_ideal = integer sum of cell values in that column.
  PE error:  p_err = P(round(S_phys) != S_ideal)
                   = P(|S_phys - S_ideal| > 0.5)

Per-column log-probability (column independence holds):
  log_P_c = sum over all bitplanes b and both p/m sides:
              log(1 - p_err_{b,side}(column c))

This is the objective to MAXIMISE in accuracy_optimizer.py.
"""

import numpy as np
from scipy.special import erfc


# Floor for log(1 - p_err) to avoid -inf when p_err -> 1
_LOG_P_FLOOR = -30.0


# ---------------------------------------------------------------------------
# Core error-probability primitive
# ---------------------------------------------------------------------------

def _p_err_gaussian(mu_S, var_S, S_ideal):
    """
    P(round(S_phys) != S_ideal) for S_phys ~ N(mu_S, var_S).

    Derivation:
      error = P(S_phys > S_ideal + 0.5) + P(S_phys < S_ideal - 0.5)
            = erfc((S_ideal + 0.5 - mu_S) / (sqrt(2)*sigma)) / 2
            + erfc((mu_S   - S_ideal + 0.5) / (sqrt(2)*sigma)) / 2

    Parameters
    ----------
    mu_S    : float  mean of physical column sum
    var_S   : float  variance of physical column sum (>= 0)
    S_ideal : int    ideal integer readout

    Returns
    -------
    p_err : float in [0, 1]
    """
    sigma_S = max(var_S, 1e-30) ** 0.5
    s2 = np.sqrt(2.0) * sigma_S
    p = 0.5 * erfc((S_ideal + 0.5 - mu_S) / s2) \
      + 0.5 * erfc((mu_S - S_ideal + 0.5) / s2)
    return float(np.clip(p, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 1-bit: lookup table (indexed by n_on, precomputed once)
# ---------------------------------------------------------------------------

def build_p_err_table_1bit(stats_1bit, M):
    """
    Precompute p_err for all n_on in 0..M for a 1-bit PE column.

    Under all-ones input with n_on active (ON) cells out of M:
      mu_S  = n_on * mu_on  + (M - n_on) * mu_off
      var_S = n_on * var_on + (M - n_on) * var_off
      S_ideal = n_on

    Parameters
    ----------
    stats_1bit : dict  keys 'on', 'off', each {'mu': float, 'var': float}
    M          : int   column size (COLUMN_SIZE)

    Returns
    -------
    table : ndarray [M+1]  table[n_on] = p_err
    """
    mu_on  = stats_1bit['on']['mu'];   var_on  = stats_1bit['on']['var']
    mu_off = stats_1bit['off']['mu'];  var_off = stats_1bit['off']['var']

    table = np.zeros(M + 1, dtype=np.float64)
    for n in range(M + 1):
        mu_S  = n * mu_on  + (M - n) * mu_off
        var_S = n * var_on + (M - n) * var_off
        table[n] = _p_err_gaussian(mu_S, var_S, n)
    return table


# ---------------------------------------------------------------------------
# 2-bit: computed per-column (state composition varies)
# ---------------------------------------------------------------------------

def p_err_column_2bit(n_state, stats_2bit):
    """
    p_err for a 2-bit PE column under all-ones input.

    Cell states in {0, 1, 2, 3}: each active cell contributes according
    to its physical charge distribution from stats_2bit.

      mu_S    = sum_s n_s * mu_s
      var_S   = sum_s n_s * var_s
      S_ideal = 0*n0 + 1*n1 + 2*n2 + 3*n3

    Parameters
    ----------
    n_state    : array-like [4]  n_state[s] = count of cells in state s
    stats_2bit : dict  keys 0..3, each {'mu': float, 'var': float}

    Returns
    -------
    p_err : float
    """
    n0, n1, n2, n3 = int(n_state[0]), int(n_state[1]), \
                     int(n_state[2]), int(n_state[3])
    mu_S  = (n0 * stats_2bit[0]['mu'] + n1 * stats_2bit[1]['mu']
           + n2 * stats_2bit[2]['mu'] + n3 * stats_2bit[3]['mu'])
    var_S = (n0 * stats_2bit[0]['var'] + n1 * stats_2bit[1]['var']
           + n2 * stats_2bit[2]['var'] + n3 * stats_2bit[3]['var'])
    S_ideal = n1 + 2 * n2 + 3 * n3
    return _p_err_gaussian(mu_S, var_S, S_ideal)


# ---------------------------------------------------------------------------
# Per-column log-probability
# ---------------------------------------------------------------------------

def col_log_prob(n_on_p, n_on_m, n_state_p, n_state_m,
                 p_err_1b_table, stats_2bit):
    """
    log P_c for one column under all-ones input.

    Sums log(1 - p_err) over all bitplanes and both plus/minus sides.
    Column independence assumed (valid for independent charge integration).

    Parameters
    ----------
    n_on_p         : array [KB]     n_on counts, 1-bit plus  planes
    n_on_m         : array [KB]     n_on counts, 1-bit minus planes
    n_state_p      : array [KQ, 4]  state histograms, 2-bit plus  planes
    n_state_m      : array [KQ, 4]  state histograms, 2-bit minus planes
    p_err_1b_table : array [M+1]    precomputed from build_p_err_table_1bit
    stats_2bit     : dict  keys 0..3 from compute_device_stats

    Returns
    -------
    log_p : float  (<= 0; closer to 0 means higher probability = better)
    """
    KB = len(n_on_p)
    KQ = len(n_state_p)
    log_p = 0.0

    for m in range(KB):
        p_p = p_err_1b_table[int(n_on_p[m])]
        p_m = p_err_1b_table[int(n_on_m[m])]
        log_p += max(np.log(max(1.0 - p_p, 1e-30)), _LOG_P_FLOOR)
        log_p += max(np.log(max(1.0 - p_m, 1e-30)), _LOG_P_FLOOR)

    for t in range(KQ):
        p_p = p_err_column_2bit(n_state_p[t], stats_2bit)
        p_m = p_err_column_2bit(n_state_m[t], stats_2bit)
        log_p += max(np.log(max(1.0 - p_p, 1e-30)), _LOG_P_FLOOR)
        log_p += max(np.log(max(1.0 - p_m, 1e-30)), _LOG_P_FLOOR)

    return log_p
