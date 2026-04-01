"""
Innovation 2 — Full pipeline entry point.

Phase 1   col_mc_sim / analytical_cal  — device-level calibration
             → N_th^(1b), N_th^(2b), κ_2, κ_3

Phase 2/3  SDR mapping methods on a demo weight block:
             1. Conventional (u=0 non-redundant binary)
             2. Min-neq SDR baseline
             3. Proposed1 (all-ones bit-plane log-prob objective)
             4. Proposed2 (INT8xINT8 MAC-accuracy objective)
             5. Proposed3 (CRN-aligned end-to-end random-INT8 objective)

Phase 4   CIM MAC accuracy evaluation via FeFET charge model
           (Conventional vs Proposed1 vs Proposed2 vs Proposed3)

Usage:
  python run_inno2.py [--skip-mc] [--phase1-only]
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import src.config_inno2 as cfg

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'Results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Phase 1 helpers
# ---------------------------------------------------------------------------

def _mc_done():
    return (os.path.exists(os.path.join(RESULTS_DIR, 'col_mc_1bit.npz')) and
            os.path.exists(os.path.join(RESULTS_DIR, 'col_mc_2bit.npz')))


def _ep_done():
    return os.path.exists(os.path.join(RESULTS_DIR, 'calibration_params.npz'))


def _kappa_done():
    return os.path.exists(os.path.join(RESULTS_DIR, 'kappa_final.npz'))


def phase1(skip_mc=False):
    if cfg.FAST_MODE:
        _phase1_analytical()
    else:
        _phase1_mc(skip_mc=skip_mc)


def _phase1_analytical():
    print("\n" + "=" * 60)
    print("[Phase 1] Analytical calibration (FAST_MODE=True)")
    from src.analytical_cal import run_analytical_calibration
    run_analytical_calibration(results_dir=RESULTS_DIR, verbose=True)
    print("[Phase 1] Done.")


def _phase1_mc(skip_mc=False):
    # 1.1 MC simulation
    if skip_mc and _mc_done():
        print("[Phase 1.1] MC data found, skipping simulation.")
    else:
        print("\n" + "=" * 60)
        print("[Phase 1.1] Column-level Monte Carlo simulation")
        from src.col_mc_sim import run_1bit_mc, run_2bit_mc
        n_on, Y_1b = run_1bit_mc(verbose=True)
        np.savez_compressed(os.path.join(RESULTS_DIR, 'col_mc_1bit.npz'),
                            n_on=n_on, Y=Y_1b,
                            M=cfg.COLUMN_SIZE, N_mc=cfg.N_MC,
                            sigma_scale=cfg.VTH_SIGMA_SCALE)
        combos, Y_2b = run_2bit_mc(verbose=True)
        np.savez_compressed(os.path.join(RESULTS_DIR, 'col_mc_2bit.npz'),
                            combos=np.array(combos, dtype=np.int32), Y=Y_2b,
                            M=cfg.COLUMN_SIZE, N_mc=cfg.N_MC,
                            sigma_scale=cfg.VTH_SIGMA_SCALE)
        print("[Phase 1.1] Done.")

    # 1.2 Error probability
    print("\n" + "=" * 60)
    print("[Phase 1.2] Error probability curves")
    from src.error_prob import (compute_error_prob_1bit,
                            compute_error_prob_2bit_initial,
                            compute_initial_kappa)
    data1 = np.load(os.path.join(RESULTS_DIR, 'col_mc_1bit.npz'))
    n_vals, p_err_1b, N_th_1b = compute_error_prob_1bit(data1['n_on'], data1['Y'])
    np.savez_compressed(os.path.join(RESULTS_DIR, 'error_prob_1bit.npz'),
                        n=n_vals, p_err=p_err_1b, N_th=N_th_1b)

    data2    = np.load(os.path.join(RESULTS_DIR, 'col_mc_2bit.npz'))
    k2i, k3i = compute_initial_kappa()
    neq_v, p2b, N_th_2b = compute_error_prob_2bit_initial(
        data2['combos'], data2['Y'], k2i, k3i)
    np.savez_compressed(os.path.join(RESULTS_DIR, 'error_prob_2bit.npz'),
                        n_eq=neq_v, p_err=p2b, N_th=N_th_2b,
                        kappa_2=k2i, kappa_3=k3i)
    np.savez(os.path.join(RESULTS_DIR, 'calibration_params.npz'),
             N_th_1bit=N_th_1b, N_th_2bit=N_th_2b,
             kappa_2=k2i, kappa_3=k3i)
    print(f"  N_th^1b={N_th_1b}, N_th^2b={N_th_2b:.2f}, "
          f"kappa_2={k2i:.4f}, kappa_3={k3i:.4f}")
    print("[Phase 1.2] Done.")

    # 1.3 Kappa calibration
    print("\n" + "=" * 60)
    print("[Phase 1.3] Kappa calibration")
    from src.kappa_calibration import calibrate_kappa
    kappa_2, kappa_3, N_th_2b_ref, neq_grid, p_err_ref = calibrate_kappa(
        data2['combos'], data2['Y'], k2i, k3i, verbose=True)
    np.savez(os.path.join(RESULTS_DIR, 'kappa_final.npz'),
             kappa_2=kappa_2, kappa_3=kappa_3,
             N_th_1bit=N_th_1b, N_th_2bit=N_th_2b_ref,
             neq_grid=neq_grid, p_err_2b=p_err_ref)
    np.savez(os.path.join(RESULTS_DIR, 'calibration_params.npz'),
             N_th_1bit=N_th_1b, N_th_2bit=N_th_2b_ref,
             kappa_2=kappa_2, kappa_3=kappa_3)
    print("[Phase 1.3] Done.")


# ---------------------------------------------------------------------------
# Result saver
# ---------------------------------------------------------------------------

def _save_result(res, stem):
    """Save a mapping result dict to Results/<stem>.npz."""
    np.savez(os.path.join(RESULTS_DIR, f'{stem}.npz'),
             J_c          = res['J_c'],
             S_c          = res['S_c'],
             S_raw        = res.get('S_raw', res['S_c']),
             J_total      = res['J_total'],
             S_total      = res['S_total'],
             S_raw_total  = res.get('S_raw_total', res['S_total']),
             obj          = res['obj'],
             mean_n1b     = res['mean_n1b'],
             mean_neq     = res['mean_neq'])


# ---------------------------------------------------------------------------
# Phase 2/3: Run SDR mapping methods
# ---------------------------------------------------------------------------

def phase23(W=None):
    """
    Run SDR mapping methods on weight matrix W (random block if None).

    Returns
    -------
    W          : ndarray [M, N]
    res_conv   : result dict for conventional mapping
    res_mneq   : result dict for min-neq SDR
    res_p1     : result dict for Proposed1 optimizer
    res_p2     : result dict for Proposed2 optimizer
    res_p3     : result dict for Proposed3 optimizer
    """
    from src.column_stats          import load_calibration
    from src.baseline_mapping      import conventional_map, minneq_map, print_summary
    from src.accuracy_optimizer    import optimize_mapping_proposed1
    from src.mac_accuracy_optimizer import optimize_mapping_proposed2
    from src.mac_accuracy_optimizer_p3 import optimize_mapping_proposed3
    from src.decompose             import build_sdr_lut

    cal = load_calibration(RESULTS_DIR)
    print(f"\nCalibration: N_th_1b={cal['N_th_1b']:.1f}, "
          f"N_th_2b={cal['N_th_2b']:.2f}, "
          f"kappa_2={cal['kappa_2']:.4f}, kappa_3={cal['kappa_3']:.4f}")

    if W is None:
        rng = np.random.default_rng(2026)
        W   = rng.integers(-cfg.W_MAX, cfg.W_MAX + 1,
                            size=(cfg.COLUMN_SIZE, cfg.COLUMN_SIZE))
        print(f"\nUsing random {W.shape} weight block (seed=2026).")

    # Build LUT once, reuse across all methods
    print("\nBuilding SDR LUT...")
    lut = build_sdr_lut()

    # --- Conventional ---
    print("\n" + "=" * 60)
    print("[Phase 2] Conventional mapping (non-redundant binary)")
    res_conv = conventional_map(W, cal=cal)
    print_summary(res_conv, label='Conventional')
    _save_result(res_conv, 'result_conventional')
    print("[Phase 2] Done.")

    # --- Min-neq SDR ---
    print("\n" + "=" * 60)
    print("[Phase 2] Min-neq SDR baseline")
    res_mneq = minneq_map(W, cal=cal, lut=lut)
    print_summary(res_mneq, label='Min-neq SDR')
    _save_result(res_mneq, 'result_minneq')
    print("[Phase 2] Done.")

    # --- Proposed1 optimizer ---
    print("\n" + "=" * 60)
    print("[Phase 3] Proposed1 SDR optimizer (all-ones bit-plane objective)")
    res_p1 = optimize_mapping_proposed1(W, cal=cal, lut=lut, max_iter=10, verbose=True)
    print_summary(res_p1, label='Proposed1')
    _save_result(res_p1, 'result_proposed1')
    print("[Phase 3] Done.")

    # --- Proposed2 optimizer ---
    print("\n" + "=" * 60)
    print("[Phase 3b] Proposed2 SDR optimizer (INT8xINT8 MAC objective)")
    res_p2 = optimize_mapping_proposed2(W, cal=cal, lut=lut, max_iter=10, verbose=True)
    print_summary(res_p2, label='Proposed2')
    _save_result(res_p2, 'result_proposed2')
    print(f"  log_P_total = {res_p2['log_P_total']:.4f}")
    print("[Phase 3b] Done.")

    # --- Proposed3 optimizer ---
    print("\n" + "=" * 60)
    print("[Phase 3c] Proposed3 SDR optimizer (CRN-aligned E2E objective)")
    n_jobs = max(1, (os.cpu_count() or 1) - 1)
    print(f"  Using multi-core for Proposed3: n_jobs={n_jobs}")
    res_p3 = optimize_mapping_proposed3(
        W, cal=cal, lut=lut, max_iter=3, verbose=True, n_jobs=n_jobs)
    print_summary(res_p3, label='Proposed3')
    _save_result(res_p3, 'result_proposed3')
    print(f"  log_P_total = {res_p3['log_P_total']:.4f}")
    print("[Phase 3c] Done.")

    # Summary table
    print("\n" + "=" * 60)
    print("Summary comparison  (obj = weighted J + η·nth_normalised S)")
    print(f"  {'Metric':<20} {'Conventional':>14} {'Min-neq':>14} "
        f"{'Proposed1':>14} {'Proposed2':>14} {'Proposed3':>14}")
    print(f"  {'-'*78}")
    for key in ('J_total', 'S_raw_total', 'obj'):
        cv = res_conv[key]
        mn = res_mneq[key]
        p1 = res_p1[key]
        p2 = res_p2[key]
        p3 = res_p3[key]
        r1 = 100.0 * (cv - mn) / max(abs(cv), 1e-12)
        r2 = 100.0 * (cv - p1) / max(abs(cv), 1e-12)
        r3 = 100.0 * (cv - p2) / max(abs(cv), 1e-12)
        r4 = 100.0 * (cv - p3) / max(abs(cv), 1e-12)
        print(f"  {key:<20} {cv:>14.4f} {mn:>13.4f} ({r1:+.1f}%) "
              f"{p1:>13.4f} ({r2:+.1f}%) {p2:>13.4f} ({r3:+.1f}%) "
              f"{p3:>13.4f} ({r4:+.1f}%)")
    print(f"  {'log_P_total':<20} {'N/A':>14} {'N/A':>14} "
          f"{res_p1['log_P_total']:>14.2f} {res_p2['log_P_total']:>14.2f} "
          f"{res_p3['log_P_total']:>14.2f}")

    return W, res_conv, res_mneq, res_p1, res_p2, res_p3


# ---------------------------------------------------------------------------
# Phase 4: CIM accuracy
# ---------------------------------------------------------------------------

def phase4(W, res_conv, res_p1, res_p2, res_p3):
    """
    CIM MAC accuracy evaluation.
    Compares conventional, Proposed1, Proposed2 and Proposed3
    against the exact integer MAC reference.
    """
    from src.cim_accuracy import evaluate_accuracy, print_accuracy_comparison

    print("\n" + "=" * 60)
    print("[Phase 4] CIM MAC accuracy evaluation")
    print(f"  N_vec=1000, vth_sigma_scale={cfg.VTH_SIGMA_SCALE:.1f}")

    print("\n  [4] Conventional vs Proposed1 vs Proposed2 vs Proposed3")
    acc = evaluate_accuracy(
        W,
        planes_base = res_conv['planes'],
        planes_p1   = res_p1['planes'],
        planes_p2   = res_p2['planes'],
        planes_p3   = res_p3['planes'],
        sigma       = cfg.VTH_SIGMA_SCALE,
        N_vec       = 1000,
        seed        = 2026,
        device      = 'cpu',
        verbose     = True,
    )
    print_accuracy_comparison(acc)

    base_ex = acc['baseline_exact_rate']
    p1_ex   = acc['proposed1_exact_rate']
    p2_ex   = acc['proposed2_exact_rate']
    p3_ex   = acc['proposed3_exact_rate']
    print("\n" + "=" * 60)
    print("Four-way exact match rate summary:")
    print(f"  Conventional   : {base_ex:.4f}")
    print(f"  Proposed1      : {p1_ex:.4f}  (delta vs conv: {p1_ex - base_ex:+.4f})")
    print(f"  Proposed2      : {p2_ex:.4f}  (delta vs conv: {p2_ex - base_ex:+.4f})")
    print(f"  Proposed3      : {p3_ex:.4f}  (delta vs conv: {p3_ex - base_ex:+.4f})")

    np.savez(os.path.join(RESULTS_DIR, 'cim_accuracy.npz'),
             baseline_exact_rate     = acc['baseline_exact_rate'],
             proposed1_exact_rate    = acc['proposed1_exact_rate'],
             proposed2_exact_rate    = acc['proposed2_exact_rate'],
             proposed3_exact_rate    = acc['proposed3_exact_rate'],
             baseline_cos_mean       = acc['baseline_cos_mean'],
             baseline_cos_std        = acc['baseline_cos_std'],
             baseline_rel_l2_mean    = acc['baseline_rel_l2_mean'],
             baseline_rel_l2_std     = acc['baseline_rel_l2_std'],
             proposed1_cos_mean      = acc['proposed1_cos_mean'],
             proposed1_cos_std       = acc['proposed1_cos_std'],
             proposed1_rel_l2_mean   = acc['proposed1_rel_l2_mean'],
             proposed1_rel_l2_std    = acc['proposed1_rel_l2_std'],
             proposed2_cos_mean      = acc['proposed2_cos_mean'],
             proposed2_cos_std       = acc['proposed2_cos_std'],
             proposed2_rel_l2_mean   = acc['proposed2_rel_l2_mean'],
             proposed2_rel_l2_std    = acc['proposed2_rel_l2_std'],
             proposed3_cos_mean      = acc['proposed3_cos_mean'],
             proposed3_cos_std       = acc['proposed3_cos_std'],
             proposed3_rel_l2_mean   = acc['proposed3_rel_l2_mean'],
             proposed3_rel_l2_std    = acc['proposed3_rel_l2_std'],
             baseline_pe_rate_1b     = acc['baseline_pe_rate_1b'],
             proposed1_pe_rate_1b    = acc['proposed1_pe_rate_1b'],
             proposed2_pe_rate_1b    = acc['proposed2_pe_rate_1b'],
             proposed3_pe_rate_1b    = acc['proposed3_pe_rate_1b'],
             baseline_pe_rate_2b     = acc['baseline_pe_rate_2b'],
             proposed1_pe_rate_2b    = acc['proposed1_pe_rate_2b'],
             proposed2_pe_rate_2b    = acc['proposed2_pe_rate_2b'],
             proposed3_pe_rate_2b    = acc['proposed3_pe_rate_2b'],
             # Metadata
             lambda_B = acc['lambda_B'],
             lambda_Q = acc['lambda_Q'],
             N_vec    = acc['N_vec'],
             sigma    = acc['sigma'],
             clip_1b  = acc['clip_1b'],
             clip_2b  = acc['clip_2b'])
    print("[Phase 4] Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Innovation 2 pipeline')
    parser.add_argument('--skip-mc',      action='store_true',
                        help='Skip MC simulation if data already exists')
    parser.add_argument('--phase1-only',  action='store_true',
                        help='Only run Phase 1 (calibration)')
    args = parser.parse_args()

    phase1(skip_mc=args.skip_mc)
    if not args.phase1_only:
        W, res_conv, res_mneq, res_p1, res_p2, res_p3 = phase23()
        phase4(W, res_conv=res_conv, res_p1=res_p1, res_p2=res_p2, res_p3=res_p3)

    print("\n[run_inno2] All phases complete. Results in:", RESULTS_DIR)


if __name__ == '__main__':
    main()
