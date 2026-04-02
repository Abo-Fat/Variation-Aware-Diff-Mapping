"""
Innovation 2 - Full pipeline entry point.

Phase 1   col_mc_sim / analytical_cal  - device-level calibration
             -> N_th^(1b), N_th^(2b), kappa_2, kappa_3

Phase 2/3  SDR mapping methods on a demo weight block:
             1. Two's complement  (二补码, TC baseline)
             2. Conventional sign-magnitude  (差分幅值码)
             3. Min-neq SDR baseline
             4. Proposed1 (all-ones bit-plane log-prob objective)

Phase 4   CIM MAC accuracy evaluation via FeFET charge model
           (TC / Conventional / Proposed1 — three-way comparison)

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
    W        : ndarray [M, N]
    res_tc   : result dict for two's complement mapping
    res_conv : result dict for conventional sign-magnitude mapping
    res_mneq : result dict for min-neq SDR
    res_p1   : result dict for Proposed1 optimizer
    """
    from src.column_stats          import load_calibration
    from src.baseline_mapping      import twos_complement_map, conventional_map, \
                                          minneq_map, print_summary
    from src.accuracy_optimizer    import optimize_mapping_proposed1
    from src.decompose             import build_sdr_lut

    cal = load_calibration(RESULTS_DIR)
    print(f"\nCalibration: N_th_1b={cal['N_th_1b']:.1f}, "
          f"N_th_2b={cal['N_th_2b']:.2f}, "
          f"kappa_2={cal['kappa_2']:.4f}, kappa_3={cal['kappa_3']:.4f}")

    if W is None:
        rng = np.random.default_rng(2026)
        W   = cfg.generate_weight_matrix(rng)
        print(f"\nUsing Gaussian weight block {W.shape} (seed=2026, σ≈{cfg.W_SIGMA:.0f}).")

    # Build LUT once, reuse across all methods
    print("\nBuilding SDR LUT...")
    lut = build_sdr_lut()

    # --- Two's complement ---
    print("\n" + "=" * 60)
    print("[Phase 2] Two's complement mapping (二补码)")
    res_tc = twos_complement_map(W, cal=cal)
    print_summary(res_tc, label="TC (二补码)")
    _save_result(res_tc, 'result_tc')
    print("[Phase 2] Done.")

    # --- Conventional ---
    print("\n" + "=" * 60)
    print("[Phase 2] Conventional mapping (差分幅值码, sign-magnitude)")
    res_conv = conventional_map(W, cal=cal)
    print_summary(res_conv, label='Conv (差分幅值码)')
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

    # Summary table
    print("\n" + "=" * 60)
    print("Summary comparison  (obj = weighted J + nth_normalised S)")
    print(f"  {'Metric':<20} {'TC':>12} {'Conventional':>14} {'Min-neq':>12} {'Proposed1':>12}")
    print(f"  {'-'*82}")
    for key in ('J_total', 'S_raw_total', 'obj'):
        tc = res_tc[key]
        cv = res_conv[key]
        mn = res_mneq[key]
        p1 = res_p1[key]
        r_cv = 100.0 * (tc - cv) / max(abs(tc), 1e-12)
        r_mn = 100.0 * (tc - mn) / max(abs(tc), 1e-12)
        r_p1 = 100.0 * (tc - p1) / max(abs(tc), 1e-12)
        print(f"  {key:<20} {tc:>12.4f} {cv:>13.4f} ({r_cv:+.1f}%) "
              f"{mn:>11.4f} ({r_mn:+.1f}%) {p1:>11.4f} ({r_p1:+.1f}%)")
    print(f"  {'log_P_total':<20} {'N/A':>12} {'N/A':>14} {'N/A':>12} {res_p1['log_P_total']:>12.2f}")

    return W, res_tc, res_conv, res_mneq, res_p1


# ---------------------------------------------------------------------------
# Phase 4: CIM accuracy
# ---------------------------------------------------------------------------

def phase4(W, res_tc, res_conv, res_p1):
    """
    CIM MAC accuracy evaluation.
    Three-way comparison: TC (二补码) / Conventional (差分幅值码) / Proposed1.
    """
    from src.cim_accuracy import evaluate_accuracy, print_accuracy_comparison

    print("\n" + "=" * 60)
    print("[Phase 4] CIM MAC accuracy evaluation  (TC / Conv / Proposed)")
    print(f"  N_vec=500, vth_sigma_scale={cfg.VTH_SIGMA_SCALE:.1f}")

    acc = evaluate_accuracy(
        W,
        planes_tc       = res_tc['planes'],
        planes_conv     = res_conv['planes'],
        planes_proposed = res_p1['planes'],
        sigma           = cfg.VTH_SIGMA_SCALE,
        N_vec           = 500,
        seed            = 2026,
        device          = 'cpu',
        verbose         = True,
    )

    print_accuracy_comparison(acc)

    np.savez(os.path.join(RESULTS_DIR, 'cim_accuracy.npz'),
             tc_exact_rate          = acc['tc_exact_rate'],
             conv_exact_rate        = acc['conv_exact_rate'],
             proposed_exact_rate    = acc['proposed_exact_rate'],
             tc_cos_mean            = acc['tc_cos_mean'],
             tc_cos_std             = acc['tc_cos_std'],
             tc_rel_l2_mean         = acc['tc_rel_l2_mean'],
             tc_rel_l2_std          = acc['tc_rel_l2_std'],
             conv_cos_mean          = acc['conv_cos_mean'],
             conv_cos_std           = acc['conv_cos_std'],
             conv_rel_l2_mean       = acc['conv_rel_l2_mean'],
             conv_rel_l2_std        = acc['conv_rel_l2_std'],
             proposed_cos_mean      = acc['proposed_cos_mean'],
             proposed_cos_std       = acc['proposed_cos_std'],
             proposed_rel_l2_mean   = acc['proposed_rel_l2_mean'],
             proposed_rel_l2_std    = acc['proposed_rel_l2_std'],
             tc_pe_rate_1b          = acc['tc_pe_rate_1b'],
             conv_pe_rate_1b        = acc['conv_pe_rate_1b'],
             proposed_pe_rate_1b    = acc['proposed_pe_rate_1b'],
             tc_pe_rate_2b          = acc['tc_pe_rate_2b'],
             conv_pe_rate_2b        = acc['conv_pe_rate_2b'],
             proposed_pe_rate_2b    = acc['proposed_pe_rate_2b'],
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
        W, res_tc, res_conv, _, res_p1 = phase23()
        phase4(W, res_tc=res_tc, res_conv=res_conv, res_p1=res_p1)

    print("\n[run_inno2] All phases complete. Results in:", RESULTS_DIR)


if __name__ == '__main__':
    main()
