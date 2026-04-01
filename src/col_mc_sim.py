"""
Phase 1.1 — Column-level Monte Carlo simulation.
              (vectorized N_mc + multiprocessing, max 8 workers)

Key accelerations vs. original:
  1. Vectorized N_mc: all N_mc trials for a given (n / combo) are computed
     in a single numpy call — no Python loop over trials.
  2. Multiprocessing: independent n values (1-bit) and combos (2-bit) are
     distributed across up to MAX_WORKERS = 8 processes.

1-bit PE:
  Sweep n_on = 0..M (M+1 configurations).
  For each n_on, sample [N_mc, n_on] Vth values at once via numpy.

2-bit PE (FULL coverage):
  All (n1, n2, n3) with n1+n2+n3 <= M  →  C(M+3,3) = 47905 combos for M=64.
  Memory estimate: 47905 × N_mc × 4 bytes  (~960 MB for N_mc=5000).

Results saved to Results/col_mc_1bit.npz and Results/col_mc_2bit.npz.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.optimize import brentq
import multiprocessing as mp
import config_inno2 as cfg

MAX_WORKERS = 8   # hard cap on parallel processes


# ---------------------------------------------------------------------------
# LUT builder — returns a plain picklable dict (no class, no methods)
# ---------------------------------------------------------------------------

def _build_lut():
    """
    Build Q-LUTs as a plain numpy dict.
    Identical math to the _Physics / FeFETVariationModel LUT approach.
    Returned dict contains only numpy arrays and Python scalars → picklable.
    """
    SS    = cfg.FEFET_SS
    I0    = cfg.FEFET_I0
    R     = cfg.FEFET_R_LIMIT
    Vd    = cfg.FEFET_VD
    vth   = np.array(cfg.FEFET_VTH_STATES,  dtype=np.float64)
    sigma = np.array(cfg.FEFET_SIGMA_VTH,   dtype=np.float64)
    Q0    = cfg.FEFET_Q0
    t_step = cfg.FEFET_T_STEP

    def _i_fefet(Vgs, Vds, Vth):
        if Vds <= 0:
            return 0.0
        return max(I0 * 10.0 ** ((Vgs - Vth) / SS) * (1.0 - np.exp(-Vds / 0.02585)), 0.0)

    def _solve(Vg, Vth):
        def res(Vsf): return _i_fefet(Vg - Vsf, Vd - Vsf, Vth) - Vsf / R
        if res(0.0) <= 0.0: return 0.0
        if res(Vd - 1e-9) >= 0.0: return (Vd - 1e-9) / R
        return brentq(res, 0.0, Vd - 1e-9, xtol=1e-14) / R

    n_lut  = 3000
    margin = 5.0 * sigma.max() + 0.15
    vth_ax = np.linspace(vth.min() - margin, vth.max() + margin, n_lut)

    Q2bit = np.array([
        sum(_solve(vg, v) for vg in cfg.FEFET_VG_STEPS_2BIT) * t_step / Q0
        for v in vth_ax
    ], dtype=np.float64)

    Q1bit = np.array([
        _solve(cfg.FEFET_VG_READ_1BIT, v) * t_step / Q0
        for v in vth_ax
    ], dtype=np.float64)

    return {
        'vth_ax':   vth_ax,   # [n_lut]  Vth axis for LUT interpolation
        'Q1bit':    Q1bit,    # [n_lut]  normalised charge for 1-bit read
        'Q2bit':    Q2bit,    # [n_lut]  normalised charge for 2-bit staircase
        'vth':      vth,      # [4]      nominal Vth per state
        'sigma':    sigma,    # [4]      sigma_vth per state
        'val2state': np.array([3, 2, 1, 0], dtype=np.int32),  # cell value -> internal state index
        'sig_r':    float(cfg.FEFET_SIGMA_R_REL),
    }


# ---------------------------------------------------------------------------
# Vectorized single-config samplers  (N_mc dimension fully batched)
# ---------------------------------------------------------------------------

def _sample_1bit_vec(n_on, M, N_mc, sigma_scale, lut):
    """
    Return Y [N_mc] for a column with n_on ON-cells and (M-n_on) OFF-cells.
    All N_mc trials computed in one numpy call per cell group.
    """
    n_off    = M - n_on
    vth      = lut['vth']
    sigma    = lut['sigma']
    sig_r    = lut['sig_r']
    vth_ax   = lut['vth_ax']
    Q1bit    = lut['Q1bit']

    Y = np.zeros(N_mc, dtype=np.float64)

    if n_on > 0:
        # [N_mc, n_on] Vth samples for ON-cells (state_idx 0, lowest Vth)
        vth_s = vth[0] + np.random.randn(N_mc, n_on) * (sigma_scale * sigma[0])
        Q = np.interp(vth_s.ravel(), vth_ax, Q1bit).reshape(N_mc, n_on)
        if sig_r > 0.0:
            Q /= np.maximum(1.0 + np.random.randn(N_mc, n_on) * sig_r, 0.1)
        Y += Q.sum(axis=1)

    if n_off > 0:
        # [N_mc, n_off] Vth samples for OFF-cells (state_idx 3, highest Vth)
        vth_s = vth[3] + np.random.randn(N_mc, n_off) * (sigma_scale * sigma[3])
        Q = np.interp(vth_s.ravel(), vth_ax, Q1bit).reshape(N_mc, n_off)
        if sig_r > 0.0:
            Q /= np.maximum(1.0 + np.random.randn(N_mc, n_off) * sig_r, 0.1)
        Y += Q.sum(axis=1)

    return Y.astype(np.float32)


def _sample_2bit_vec(counts, N_mc, sigma_scale, lut):
    """
    Return Y [N_mc] for a column with cell-value composition counts=[n0,n1,n2,n3].
    All N_mc trials computed in one numpy call per cell-value group.
    """
    vth       = lut['vth']
    sigma     = lut['sigma']
    sig_r     = lut['sig_r']
    vth_ax    = lut['vth_ax']
    Q2bit     = lut['Q2bit']
    val2state = lut['val2state']

    Y = np.zeros(N_mc, dtype=np.float64)

    for cell_val, n_cells in enumerate(counts):
        if n_cells <= 0:
            continue
        state = int(val2state[cell_val])
        vth_s = vth[state] + np.random.randn(N_mc, n_cells) * (sigma_scale * sigma[state])
        Q = np.interp(vth_s.ravel(), vth_ax, Q2bit).reshape(N_mc, n_cells)
        if sig_r > 0.0:
            Q /= np.maximum(1.0 + np.random.randn(N_mc, n_cells) * sig_r, 0.1)
        Y += Q.sum(axis=1)

    return Y.astype(np.float32)


# ---------------------------------------------------------------------------
# Top-level worker functions  (must be at module level to be picklable)
# ---------------------------------------------------------------------------

def _worker_1bit(args):
    """Process a chunk of n_on values. Returns list of (n, Y[N_mc])."""
    chunk_n, M, N_mc, sigma_scale, lut, seed = args
    np.random.seed(seed)
    results = []
    for n in chunk_n:
        Y = _sample_1bit_vec(n, M, N_mc, sigma_scale, lut)
        results.append((int(n), Y))
    return results


def _worker_2bit(args):
    """
    Process a chunk of (global_idx, n1, n2, n3) tuples.
    Returns list of (global_idx, Y[N_mc]).
    """
    combo_chunk, M, N_mc, sigma_scale, lut, seed = args
    np.random.seed(seed)
    results = []
    for idx, n1, n2, n3 in combo_chunk:
        n0 = M - n1 - n2 - n3
        Y  = _sample_2bit_vec([n0, n1, n2, n3], N_mc, sigma_scale, lut)
        results.append((idx, Y))
    return results


# ---------------------------------------------------------------------------
# Full combo generation  (C(M+3,3) combos)
# ---------------------------------------------------------------------------

def _generate_2bit_combos_full(M):
    """
    All non-negative integer (n1, n2, n3) with n1+n2+n3 <= M.
    Total count = C(M+3, 3) = (M+1)(M+2)(M+3)/6
    For M=64: 47905 combos.
    """
    combos = []
    for n1 in range(M + 1):
        for n2 in range(M + 1 - n1):
            for n3 in range(M + 1 - n1 - n2):
                combos.append((n1, n2, n3))
    return combos


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_1bit_mc(M=None, N_mc=None, sigma_scale=None,
                n_workers=None, verbose=True):
    """
    Sweep n_on = 0..M with vectorized N_mc + multiprocessing.

    Returns
    -------
    n_on_list : ndarray [M+1]
    Y         : ndarray [M+1, N_mc]  float32
    """
    if M          is None: M          = cfg.COLUMN_SIZE
    if N_mc       is None: N_mc       = cfg.N_MC
    if sigma_scale is None: sigma_scale = cfg.VTH_SIGMA_SCALE
    if n_workers  is None: n_workers  = min(MAX_WORKERS, M + 1)

    lut       = _build_lut()
    n_on_list = np.arange(M + 1, dtype=np.int32)
    Y         = np.zeros((M + 1, N_mc), dtype=np.float32)

    # Split n values across workers
    chunks = np.array_split(n_on_list, n_workers)
    seeds  = [42 + i * 997 for i in range(n_workers)]
    args   = [(chunk.tolist(), M, N_mc, sigma_scale, lut, s)
              for chunk, s in zip(chunks, seeds)]

    if verbose:
        print(f"  1-bit MC: M={M}, N_mc={N_mc}, workers={n_workers}, "
              f"configs={M+1}")

    with mp.Pool(processes=n_workers) as pool:
        for batch in pool.imap_unordered(_worker_1bit, args):
            for n, Y_n in batch:
                Y[n] = Y_n

    if verbose:
        print(f"  1-bit MC done.")

    return n_on_list, Y


def run_2bit_mc(M=None, N_mc=None, sigma_scale=None,
                n_workers=None, verbose=True):
    """
    Full-coverage 2-bit MC with vectorized N_mc + multiprocessing.

    Returns
    -------
    combos_arr : ndarray [K, 3]  int32  — (n1, n2, n3) for each row
    Y          : ndarray [K, N_mc]  float32
    """
    if M          is None: M          = cfg.COLUMN_SIZE
    if N_mc       is None: N_mc       = cfg.N_MC
    if sigma_scale is None: sigma_scale = cfg.VTH_SIGMA_SCALE
    if n_workers  is None: n_workers  = MAX_WORKERS

    combos = _generate_2bit_combos_full(M)
    K      = len(combos)
    mem_mb = K * N_mc * 4 / 1e6

    if verbose:
        print(f"  2-bit MC: M={M}, N_mc={N_mc}, workers={n_workers}")
        print(f"  Combos (full coverage): {K}  |  Y array: ~{mem_mb:.0f} MB")

    lut = _build_lut()
    Y   = np.zeros((K, N_mc), dtype=np.float32)

    # Build indexed combo list and split into worker chunks
    indexed = [(i, n1, n2, n3) for i, (n1, n2, n3) in enumerate(combos)]
    chunk_sz = (K + n_workers - 1) // n_workers
    chunks   = [indexed[i: i + chunk_sz] for i in range(0, K, chunk_sz)]
    seeds    = [7 + i * 1009 for i in range(len(chunks))]
    args     = [(ch, M, N_mc, sigma_scale, lut, s)
                for ch, s in zip(chunks, seeds)]

    completed = 0
    with mp.Pool(processes=n_workers) as pool:
        for batch in pool.imap_unordered(_worker_2bit, args):
            for idx, Y_k in batch:
                Y[idx] = Y_k
            completed += len(batch)
            if verbose:
                pct = 100 * completed // K
                print(f"  Progress: {completed}/{K} ({pct}%)", end='\r', flush=True)

    if verbose:
        print(f"\n  2-bit MC done.")

    combos_arr = np.array(combos, dtype=np.int32)
    return combos_arr, Y


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'Results')
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("Phase 1.1: 1-bit column MC simulation")
    print(f"  M={cfg.COLUMN_SIZE}, N_mc={cfg.N_MC}, "
          f"sigma_scale={cfg.VTH_SIGMA_SCALE}")
    n_on_list, Y_1bit = run_1bit_mc(verbose=True)
    out_1bit = os.path.join(out_dir, 'col_mc_1bit.npz')
    np.savez_compressed(out_1bit, n_on=n_on_list, Y=Y_1bit,
                        M=cfg.COLUMN_SIZE, N_mc=cfg.N_MC,
                        sigma_scale=cfg.VTH_SIGMA_SCALE)
    print(f"  Saved -> {out_1bit}")

    print("=" * 60)
    print("Phase 1.1: 2-bit column MC simulation")
    combos_arr, Y_2bit = run_2bit_mc(verbose=True)
    out_2bit = os.path.join(out_dir, 'col_mc_2bit.npz')
    np.savez_compressed(out_2bit, combos=combos_arr, Y=Y_2bit,
                        M=cfg.COLUMN_SIZE, N_mc=cfg.N_MC,
                        sigma_scale=cfg.VTH_SIGMA_SCALE)
    print(f"  Saved -> {out_2bit}")
    print("Done.")


if __name__ == '__main__':
    main()
