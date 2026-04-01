# ==============================================================================
# Innovation 2 Configuration
# ==============================================================================
# Parameters for mixed-precision CIM mapping optimization.
# The FeFET device parameters are copied from the parent project's config.
# Do NOT import from config/config.py to keep Independence.
# ==============================================================================

import numpy as np

# ------------------------------------------------------------------------------
# FeFET device parameters  (must match config/config.py)
# ------------------------------------------------------------------------------
FEFET_SS           = 0.115          # Subthreshold swing [V/dec]
FEFET_I0           = 1e-9           # Drain current at Vth [A]
FEFET_R_LIMIT      = 1e6            # Current-limiting resistor [Ohm]
FEFET_VD           = 0.1            # Drain voltage [V]

FEFET_VTH_STATES   = [-0.96, -0.53, -0.023, 0.52]          # internal state index 0..3 (0=lowest Vth, 3=highest Vth)
FEFET_SIGMA_VTH    = [0.0546, 0.0505, 0.0619, 0.0614]       # matches FEFET_VTH_STATES index order

# Staircase read (2-bit)
FEFET_VG_STEPS_2BIT = [-0.5, 0.0, 0.5]   # [V]
FEFET_VG_READ_1BIT  = 0.0                 # [V]
FEFET_T_STEP        = 50e-9               # [s]
FEFET_SIGMA_R_REL   = 0.01               # R relative std-dev (1%)

# Derived Q0
FEFET_Q0 = (FEFET_VD / FEFET_R_LIMIT) * FEFET_T_STEP   # = 5 fC

# Vth sigma scale used for calibration simulations (1.0 = nominal)
VTH_SIGMA_SCALE = 1.0

# ------------------------------------------------------------------------------
# Mixed-precision configuration: K_B + 2*K_Q = 7
# K_B: number of 1-bit PEs per side (+/-)
# K_Q: number of 2-bit PEs per side (+/-)
# ------------------------------------------------------------------------------
# Available combinations: (7,0), (5,1), (3,2), (1,3)
KB = 3     # number of 1-bit PEs (MSB side)
KQ = 2     # number of 2-bit PEs (LSB side)

assert KB + 2 * KQ == 7, f"Constraint violated: K_B + 2*K_Q = {KB + 2*KQ}, expected 7"

# Weight range after quantization
W_MAX = 127      # |w_{i,j}| <= W_MAX; a+/a- in [0, W_MAX]

# Column size (number of rows in one CIM array)
COLUMN_SIZE = 64   # M in the paper

# ------------------------------------------------------------------------------
# Calibration mode
# ------------------------------------------------------------------------------
# FAST_MODE = True  : analytical Gaussian approximation (seconds, recommended)
#   - No Monte Carlo on columns; single-device stats from LUT integration
#   - kappa derived analytically from charge variance ratios
#   - p_err computed via erfc
# FAST_MODE = False : full Monte Carlo (col_mc_sim -> error_prob -> kappa_cal)
#   - More accurate tail probabilities; useful for validation
#   - Much slower; requires large memory for full 2-bit coverage
FAST_MODE = True

# ------------------------------------------------------------------------------
# Phase 1: Monte Carlo column simulation (only used when FAST_MODE = False)
# ------------------------------------------------------------------------------
N_MC = 5000        # Monte Carlo trials per configuration
# Error probability threshold for determining N_th
EPSILON = 0.01     # p_err <= epsilon defines the reliable threshold

# ------------------------------------------------------------------------------
# Phase 2: kappa calibration (only used when FAST_MODE = False)
# Number of (n1, n2, n3) samples used for fitting
N_KAPPA_SAMPLES = 500   # random (n1,n2,n3) combos to evaluate

# ------------------------------------------------------------------------------
# Phase 3: Mapping optimizer
# ------------------------------------------------------------------------------
# Objective: min_{u} max_c J_c + gamma * sum_c J_c
GAMMA = 0.1        # trade-off: worst-column vs total risk

# Bit-plane weight coefficients
# lambda_{m,B} = 2^(2*K_Q + m)   for 1-bit PE index m = 0..K_B-1
# lambda_{t,Q} = 4^t              for 2-bit PE index t = 0..K_Q-1
LAMBDA_B = [2 ** (2 * KQ + m) for m in range(KB)]   # [lambda_{0,B}, ..., lambda_{K_B-1,B}]
LAMBDA_Q = [4 ** t             for t in range(KQ)]   # [lambda_{0,Q}, ..., lambda_{K_Q-1,Q}]

# Placeholder thresholds; will be overwritten after Phase 1 calibration
N_TH_1BIT = None   # filled by error_prob.py
N_TH_2BIT = None   # filled by error_prob.py

# Placeholder kappa values; will be overwritten after Phase 1.3
KAPPA_2 = None     # filled by kappa_calibration.py
KAPPA_3 = None     # filled by kappa_calibration.py

# ------------------------------------------------------------------------------
# Sparsity-penalty coefficient η in objective: Σ_c J_c + η Σ_c S_c
# ------------------------------------------------------------------------------
ETA = 0.1

# Optimizer sparsity mode:
#   'weighted'       -> original lambda^2-weighted sparsity (legacy)
#   'uniform'        -> unweighted equivalent active counts
#   'nth_normalized' -> normalize counts by N_th (recommended)
SPARSITY_MODE = 'nth_normalized'

# Optimizer overload mode:
#   'weighted'       -> lambda^2-weighted hinge: penalises MSB overloads more
#                       than LSB overloads, matching their CIM error impact
#   'uniform'        -> unweighted hinge
#   'nth_normalized' -> hinge on normalized overload (discards error weighting)
OVERLOAD_MODE = 'weighted'
