"""
Innovation 2 — Full pipeline entry point.

Phase 1   col_mc_sim / analytical_cal  — device-level calibration
             → N_th^(1b), N_th^(2b), κ_2, κ_3

Phase 2/3  Three SDR mapping methods on a demo weight block:
             1. Conventional (u=0 non-redundant binary)
             2. Min-neq SDR baseline
             3. Proposed column-wise SDR optimizer

Phase 4   CIM MAC accuracy evaluation via FeFET charge model
           (uses conventional as baseline, proposed as optimised)

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
# Phase 2/3: Run all three SDR mapping methods
# ---------------------------------------------------------------------------

def phase23(W=None):
    """
    Run three SDR mapping methods on weight matrix W (random block if None).

    Returns
    -------
    W          : ndarray [M, N]
    res_conv   : result dict for conventional mapping
    res_mneq   : result dict for min-neq SDR
    res_prop   : result dict for proposed optimizer
    """
    from src.column_stats      import load_calibration
    from src.baseline_mapping  import conventional_map, minneq_map, print_summary
    from src.mapping_optimizer import optimize_mapping
    from src.decompose         import build_sdr_lut

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

    # --- Proposed optimizer ---
    print("\n" + "=" * 60)
    print("[Phase 3] Proposed column-wise SDR optimizer")
    res_prop = optimize_mapping(W, cal=cal, lut=lut, max_iter=10, verbose=True)
    print_summary(res_prop, label='Proposed')
    _save_result(res_prop, 'result_proposed')
    print("[Phase 3] Done.")

    # Summary table
    print("\n" + "=" * 60)
    print("Summary comparison  (obj = weighted J + η·nth_normalised S)")
    print(f"  {'Metric':<20} {'Conventional':>14} {'Min-neq':>14} {'Proposed':>14}")
    print(f"  {'-'*64}")
    for key in ('J_total', 'S_raw_total', 'obj'):
        cv = res_conv[key]
        mn = res_mneq[key]
        pr = res_prop[key]
        r1 = 100.0 * (cv - mn) / max(abs(cv), 1e-12)
        r2 = 100.0 * (cv - pr) / max(abs(cv), 1e-12)
        print(f"  {key:<20} {cv:>14.4f} {mn:>13.4f} ({r1:+.1f}%) "
              f"{pr:>13.4f} ({r2:+.1f}%)")
    print(f"  {'(S_total legacy)':<20} {res_conv['S_total']:>14.1f} "
          f"{res_mneq['S_total']:>14.1f} {res_prop['S_total']:>14.1f}")

    return W, res_conv, res_mneq, res_prop


# ---------------------------------------------------------------------------
# Phase 4: CIM accuracy
# ---------------------------------------------------------------------------

def phase4(W, res_base, res_opt):
    """CIM MAC accuracy: conventional as base, proposed as opt."""
    from src.cim_accuracy import evaluate_accuracy, print_accuracy_comparison

    print("\n" + "=" * 60)
    print("[Phase 4] CIM MAC accuracy evaluation")
    print(f"  N_vec=100, vth_sigma_scale={cfg.VTH_SIGMA_SCALE:.1f}")

    acc = evaluate_accuracy(
        W,
        planes_base = res_base['planes'],
        planes_opt  = res_opt['planes'],
        sigma       = cfg.VTH_SIGMA_SCALE,
        N_vec       = 1000,
        seed        = 2026,
        device      = 'cpu',
        verbose     = True,
    )

    print_accuracy_comparison(acc)

    np.savez(os.path.join(RESULTS_DIR, 'cim_accuracy.npz'),
             baseline_exact_rate     = acc['baseline_exact_rate'],
             proposed_exact_rate     = acc['proposed_exact_rate'],
             baseline_cos_mean       = acc['baseline_cos_mean'],
             baseline_cos_std        = acc['baseline_cos_std'],
             baseline_rel_l2_mean    = acc['baseline_rel_l2_mean'],
             baseline_rel_l2_std     = acc['baseline_rel_l2_std'],
             proposed_cos_mean       = acc['proposed_cos_mean'],
             proposed_cos_std        = acc['proposed_cos_std'],
             proposed_rel_l2_mean    = acc['proposed_rel_l2_mean'],
             proposed_rel_l2_std     = acc['proposed_rel_l2_std'],
             baseline_pe_rate_1b     = acc['baseline_pe_rate_1b'],
             proposed_pe_rate_1b     = acc['proposed_pe_rate_1b'],
             baseline_pe_rate_2b     = acc['baseline_pe_rate_2b'],
             proposed_pe_rate_2b     = acc['proposed_pe_rate_2b'],
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
        W, res_conv, res_mneq, res_prop = phase23()
        phase4(W, res_base=res_conv, res_opt=res_prop)

    print("\n[run_inno2] All phases complete. Results in:", RESULTS_DIR)


if __name__ == '__main__':
    main()
