"""
Innovation 2 plotting script aligned with current pipeline.

Figure 1: Device calibration
Figure 2: Mapping structural analysis (TC / Conventional / Min-neq / Proposed1)
Figure 3: CIM MAC accuracy (TC / Conventional / Proposed1 — three-way)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
import src.config_inno2 as cfg

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'Results')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'legend.fontsize': 7.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'lines.linewidth': 1.6,
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

C_TC   = '#D62728'   # red    — two's complement (二补码)
C_CONV = '#E07B54'   # orange — conventional sign-magnitude (差分幅值码)
C_MNEQ = '#F5A623'   # yellow — min-neq SDR
C_P1   = '#4A90D9'   # blue   — Proposed1
C_1BIT = '#2CA02C'
C_2BIT = '#9467BD'
C_EPS  = '#7F7F7F'


def _load(fname):
    return np.load(os.path.join(RESULTS_DIR, fname))


def _savefig(fig, stem):
    for ext in ('pdf', 'png'):
        path = os.path.join(RESULTS_DIR, f'{stem}.{ext}')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f'  Saved -> {path}')
    plt.close(fig)


def plot_fig1():
    from src.analytical_cal import compute_device_stats, _build_lut

    lut = _build_lut()
    stats_1bit, stats_2bit = compute_device_stats(lut, verbose=False)

    ep1 = _load('error_prob_1bit.npz')
    ep2 = _load('error_prob_2bit.npz')
    cal = _load('calibration_params.npz')

    n_th_1b = float(cal['N_th_1bit'])
    n_th_2b = float(cal['N_th_2bit'])

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.17, top=0.87, wspace=0.40)

    ax = axes[0]
    palette_2b = ['#AEC7E8', '#6BAED6', '#2166AC', '#08306B']

    for key, col, ls, lbl in [
        ('on', '#2CA02C', '-', '1-bit ON'),
        ('off', '#8C564B', '--', '1-bit OFF'),
    ]:
        mu = stats_1bit[key]['mu']
        sig = max(stats_1bit[key]['var'], 1e-30) ** 0.5
        x = np.linspace(mu - 5 * sig, mu + 5 * sig, 400)
        ax.plot(x, norm.pdf(x, mu, sig), color=col, linestyle=ls, label=lbl)

    for cv in (3, 2, 1, 0):
        mu = stats_2bit[cv]['mu']
        sig = max(stats_2bit[cv]['var'], 1e-30) ** 0.5
        x = np.linspace(mu - 5 * sig, mu + 5 * sig, 400)
        ax.plot(x, norm.pdf(x, mu, sig), color=palette_2b[cv], linestyle='-.', label=f'2-bit cv={cv}')

    ax.set_xlabel('Charge output Q (norm. units Q0)')
    ax.set_ylabel('Probability density')
    ax.set_title('(a) Single-device output statistics')
    ax.legend(ncol=2, loc='upper left', handlelength=1.5)
    ax.set_xlim(left=0)
    ax.yaxis.grid(True, linestyle=':', alpha=0.4)
    ax.set_axisbelow(True)

    ax = axes[1]
    n_1b = ep1['n'].astype(float)
    p1b = np.maximum(ep1['p_err'], 1e-320)
    n_eq = ep2['n_eq'].astype(float)
    p2b = np.maximum(ep2['p_err'], 1e-320)

    ax.semilogy(n_1b, p1b, color=C_1BIT, label=r'1-bit: $p_{err}^{1b}(n)$')
    ax.semilogy(n_eq, p2b, color=C_2BIT, linestyle='--', label=r'2-bit: $p_{err}^{2b}(n_{eq})$')

    ax.axhline(cfg.EPSILON, color=C_EPS, linestyle=':', linewidth=1.2, label=f'$\\varepsilon={cfg.EPSILON}$')
    ax.axvline(n_th_1b, color=C_1BIT, linestyle=':', linewidth=1.2, label=f'$N_{{th}}^{{1b}}={int(n_th_1b)}$')
    ax.axvline(n_th_2b, color=C_2BIT, linestyle=':', linewidth=1.2, label=f'$N_{{th}}^{{2b}}={n_th_2b:.1f}$')

    ax.set_xlabel(r'Active count $n$ / equiv. active count $n_{eq}$')
    ax.set_ylabel(r'Error probability $p_{err}$')
    ax.set_title('(b) Column-level error probability vs activation')
    ax.legend(loc='upper left', handlelength=1.8)
    ax.set_xlim(0, cfg.COLUMN_SIZE)
    ax.set_ylim(1e-20, 1.5)
    ax.yaxis.grid(True, linestyle=':', alpha=0.4)
    ax.set_axisbelow(True)

    fig.suptitle('Figure 1  Device statistics and calibration', fontsize=9, y=0.99)
    _savefig(fig, 'fig1_calibration')


def plot_fig2():
    rtc  = _load('result_tc.npz')
    rc   = _load('result_conventional.npz')
    rm   = _load('result_minneq.npz')
    rp1  = _load('result_proposed1.npz')

    kb = cfg.KB
    kq = cfg.KQ

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.22, top=0.87, wspace=0.40)

    ax = axes[0]
    plane_labels = []
    for m in range(kb):
        plane_labels.append(f'$B_{{{m}}}$\n($\\lambda$={cfg.LAMBDA_B[m]})')
    for t in range(kq):
        plane_labels.append(f'$Q_{{{t}}}$\n($\\lambda$={cfg.LAMBDA_Q[t]})')

    n_tc   = np.concatenate([rtc['mean_n1b'],  rtc['mean_neq']])
    n_conv = np.concatenate([rc['mean_n1b'],   rc['mean_neq']])
    n_mneq = np.concatenate([rm['mean_n1b'],   rm['mean_neq']])
    n_p1   = np.concatenate([rp1['mean_n1b'],  rp1['mean_neq']])

    x = np.arange(len(plane_labels))
    width = 0.20

    ax.bar(x - 1.5*width, n_tc,   width, color=C_TC,   alpha=0.88, label='TC (二补码)',      zorder=3)
    ax.bar(x - 0.5*width, n_conv, width, color=C_CONV,  alpha=0.88, label='Conv (差分幅值码)', zorder=3)
    ax.bar(x + 0.5*width, n_mneq, width, color=C_MNEQ,  alpha=0.88, label='Min-neq SDR',      zorder=3)
    ax.bar(x + 1.5*width, n_p1,   width, color=C_P1,    alpha=0.88, label='Proposed1',         zorder=3)

    cal = _load('calibration_params.npz')
    n_th_1b = float(cal['N_th_1bit'])
    n_th_2b = float(cal['N_th_2bit'])
    ax.axhline(n_th_1b, color=C_1BIT, linestyle='--', linewidth=1.0, alpha=0.7, label=f'$N_{{th}}^{{1b}}={int(n_th_1b)}$')
    ax.axhline(n_th_2b, color=C_2BIT, linestyle='--', linewidth=1.0, alpha=0.7, label=f'$N_{{th}}^{{2b}}={n_th_2b:.0f}$')

    ax.set_xticks(x)
    ax.set_xticklabels(plane_labels, fontsize=7.5)
    ax.set_ylabel(r'Mean active count $\bar{n}$ / $\bar{n}_{eq}$')
    ax.set_title('(a) Per-plane mean active count')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=6.5, loc='upper right', ncol=2)
    ax.yaxis.grid(True, linestyle='--', alpha=0.40, zorder=0)
    ax.set_axisbelow(True)

    sep = kb - 0.5
    ax.axvline(sep, color='gray', linestyle='--', linewidth=0.8, zorder=2)
    n_total = len(plane_labels)
    ax.text(sep / 2 - 0.5, 1.02, '1-bit PEs', ha='center', va='bottom',
            fontsize=7.5, color='gray', transform=ax.get_xaxis_transform())
    ax.text(sep + (n_total - sep) / 2, 1.02, '2-bit PEs', ha='center', va='bottom',
            fontsize=7.5, color='gray', transform=ax.get_xaxis_transform())

    ax = axes[1]
    j_tc   = rtc['J_c']
    j_conv = rc['J_c']
    j_mneq = rm['J_c']
    j_p1   = rp1['J_c']

    # Sort by TC risk descending so the "worst-case" baseline is on the left
    order = np.argsort(j_tc)[::-1]
    cidx  = np.arange(len(j_tc))

    jt_tc = float(rtc['J_total'])
    jt_c  = float(rc['J_total'])
    jt_m  = float(rm['J_total'])
    jt_p1 = float(rp1['J_total'])

    ax.plot(cidx, j_tc[order],   color=C_TC,   label=f'TC ($\\Sigma J$={jt_tc:.0f})')
    ax.plot(cidx, j_conv[order], color=C_CONV,  linestyle='--',
            label=f'Conv ($\\Sigma J$={jt_c:.0f})')
    ax.plot(cidx, j_mneq[order], color=C_MNEQ,  linestyle=':',
            label=f'Min-neq ($\\Sigma J$={jt_m:.0f})')
    ax.plot(cidx, j_p1[order],   color=C_P1,    linestyle='-.',
            label=f'Proposed1 ($\\Sigma J$={jt_p1:.0f})')

    ax.fill_between(cidx, j_p1[order], j_tc[order], color=C_P1, alpha=0.07, zorder=1)

    ax.set_xlabel('Column index (sorted by TC $J_c$ desc)')
    ax.set_ylabel('Column risk $J_c$')
    ax.set_title('(b) Per-column risk $J_c$ distribution')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlim(0, len(j_tc) - 1)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, linestyle='--', alpha=0.40)
    ax.set_axisbelow(True)

    fig.suptitle('Figure 2  SDR mapping structural analysis', fontsize=9, y=0.99)
    _savefig(fig, 'fig2_mapping')


def plot_fig3():
    acc = _load('cim_accuracy.npz')

    tc_ex   = float(acc['tc_exact_rate'])
    cv_ex   = float(acc['conv_exact_rate'])
    pr_ex   = float(acc['proposed_exact_rate'])

    pe2b_tc   = np.asarray(acc['tc_pe_rate_2b'])
    pe2b_conv = np.asarray(acc['conv_pe_rate_2b'])
    pe2b_pr   = np.asarray(acc['proposed_pe_rate_2b'])
    kq = len(pe2b_conv)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.18, top=0.87, wspace=0.42)

    # ── (a) Exact match rate bar chart ───────────────────────────────────────
    ax = axes[0]
    names  = ['TC\n(二补码)', 'Conv\n(差分幅值码)', 'Proposed1']
    values = [tc_ex, cv_ex, pr_ex]
    colors = [C_TC, C_CONV, C_P1]
    x = np.arange(len(names))

    bars = ax.bar(x, [v * 100 for v in values], color=colors, alpha=0.88,
                  width=0.55, zorder=3)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f'{v*100:.1f}%', ha='center', va='bottom',
                fontsize=8.5, fontweight='bold')

    # Annotate improvement: Proposed vs TC
    delta_pr_tc = (pr_ex - tc_ex) * 100
    ax.annotate(
        f'{delta_pr_tc:+.1f} pp vs TC',
        xy=(x[2], pr_ex * 100),
        xytext=(x[2] + 0.25, pr_ex * 100 * 0.70),
        fontsize=8, color=C_P1, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=C_P1, lw=1.2),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.0)
    ax.set_ylabel('Exact match rate (%)')
    ax.set_title('(a) CIM MAC exact match rate')
    ax.set_ylim(0, min(max(tc_ex, cv_ex, pr_ex) * 100 * 1.40 + 1.0, 100))
    ax.yaxis.grid(True, linestyle='--', alpha=0.40, zorder=0)
    ax.set_axisbelow(True)

    # ── (b) Per-plane 2-bit PE error rate ────────────────────────────────────
    ax = axes[1]
    x_pe  = np.arange(kq)
    width = 0.24

    ax.bar(x_pe - width, pe2b_tc   * 100, width, color=C_TC,   alpha=0.88, label='TC',        zorder=3)
    ax.bar(x_pe,          pe2b_conv * 100, width, color=C_CONV,  alpha=0.88, label='Conv',       zorder=3)
    ax.bar(x_pe + width,  pe2b_pr   * 100, width, color=C_P1,    alpha=0.88, label='Proposed1',  zorder=3)

    ax.axhline(cfg.EPSILON * 100, color=C_EPS, linestyle=':', linewidth=1.2,
               label=f'$\\varepsilon={cfg.EPSILON}$')

    xtick_labels = [f'$Q_{{{t}}}$\n($\\lambda$={cfg.LAMBDA_Q[t]})' for t in range(kq)]
    ax.set_xticks(x_pe)
    ax.set_xticklabels(xtick_labels, fontsize=8.5)
    ax.set_ylabel('PE error rate (%)')
    ax.set_title('(b) Per-plane 2-bit PE error rate')
    ax.legend(fontsize=7.5, loc='upper right')
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, linestyle='--', alpha=0.40, zorder=0)
    ax.set_axisbelow(True)

    for offset, arr in [(-width, pe2b_tc), (0, pe2b_conv), (width, pe2b_pr)]:
        for t, v in enumerate(arr):
            ax.text(x_pe[t] + offset, v * 100 + 0.3, f'{v*100:.1f}%',
                    ha='center', va='bottom', fontsize=6.5)

    fig.suptitle('Figure 3  CIM MAC accuracy evaluation  (TC / Conv / Proposed)',
                 fontsize=9, y=0.99)
    _savefig(fig, 'fig3_accuracy')


if __name__ == '__main__':
    print('=== Figure 1: Device calibration ===')
    plot_fig1()
    print('=== Figure 2: Mapping structural analysis ===')
    plot_fig2()
    print('=== Figure 3: CIM accuracy results ===')
    plot_fig3()
    print('Done. Check Results/fig*.pdf')
