"""
Innovation 2 — Paper figure plotting script.

Figure 1: Device statistics → calibration
  (a) Single-device charge output distributions (1-bit ON/OFF, 2-bit cv=0..3)
  (b) Column-level error probability vs active count, with N_th marked

Figure 2: Mapping structural analysis (4 methods)
  (a) Per-plane mean active / equiv-active count: Conventional, Min-neq, Proposed1, Proposed2
  (b) Per-column risk J_c (sorted by conventional): shows J_c is a poor accuracy proxy

Figure 3: CIM MAC accuracy results (3 methods evaluated)
  (a) Exact match rate — headline accuracy metric
  (b) Per-PE-plane error rate (2-bit planes)

Usage:
  cd VADM
  python plot_inno2.py

Output (saved to Results/):
  fig1_calibration.{pdf,png}
  fig2_mapping.{pdf,png}
  fig3_accuracy.{pdf,png}
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm
import src.config_inno2 as cfg

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'Results')

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.size':        9,
    'axes.titlesize':   9,
    'axes.labelsize':   9,
    'legend.fontsize':  7.5,
    'xtick.labelsize':  8,
    'ytick.labelsize':  8,
    'lines.linewidth':  1.6,
    'figure.dpi':       150,
    'axes.spines.top':  False,
    'axes.spines.right': False,
})

# ── Color palette ─────────────────────────────────────────────────────────────
C_CONV  = '#E07B54'   # orange-red  – conventional
C_MNEQ  = '#F5A623'   # amber       – min-neq SDR
C_P1    = '#4A90D9'   # blue        – proposed1
C_P2    = '#27AE60'   # green       – proposed2
C_1BIT  = '#2ca02c'   # green       – 1-bit curve
C_2BIT  = '#9467bd'   # purple      – 2-bit curve
C_EPS   = '#7f7f7f'   # grey        – epsilon line


def _load(fname):
    return np.load(os.path.join(RESULTS_DIR, fname))


def _savefig(fig, stem):
    for ext in ('pdf', 'png'):
        path = os.path.join(RESULTS_DIR, f'{stem}.{ext}')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f'  Saved -> {path}')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 1 — Device calibration
# ══════════════════════════════════════════════════════════════════════════════

def plot_fig1():
    """
    (a) Single-device charge output distributions
    (b) Column-level error probability vs activation count
    """
    from src.analytical_cal import compute_device_stats, _build_lut

    lut = _build_lut()
    stats_1bit, stats_2bit = compute_device_stats(lut, verbose=False)

    ep1 = _load('error_prob_1bit.npz')
    ep2 = _load('error_prob_2bit.npz')
    cal = _load('calibration_params.npz')

    N_th_1b = float(cal['N_th_1bit'])
    N_th_2b = float(cal['N_th_2bit'])

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.17, top=0.87, wspace=0.40)

    # ── (a) Single-device charge distributions ────────────────────────────
    ax = axes[0]
    PALETTE_2B = ['#aec7e8', '#6baed6', '#2166ac', '#08306b']  # cv 0→3

    for key, col, ls, lbl in [
        ('on',  '#2ca02c', '-',  '1-bit ON'),
        ('off', '#8c564b', '--', '1-bit OFF'),
    ]:
        mu  = stats_1bit[key]['mu']
        sig = max(stats_1bit[key]['var'], 1e-30) ** 0.5
        x   = np.linspace(mu - 5*sig, mu + 5*sig, 400)
        ax.plot(x, norm.pdf(x, mu, sig), color=col, linestyle=ls, label=lbl)

    for cv in (3, 2, 1, 0):
        mu  = stats_2bit[cv]['mu']
        sig = max(stats_2bit[cv]['var'], 1e-30) ** 0.5
        x   = np.linspace(mu - 5*sig, mu + 5*sig, 400)
        ax.plot(x, norm.pdf(x, mu, sig), color=PALETTE_2B[cv],
                linestyle='-.', label=f'2-bit cv={cv}')

    ax.set_xlabel('Charge output $Q$ (norm. units $Q_0$)')
    ax.set_ylabel('Probability density')
    ax.set_title('(a) Single-device output statistics')
    ax.legend(ncol=2, loc='upper left', handlelength=1.5)
    ax.set_xlim(left=0)
    ax.yaxis.grid(True, linestyle=':', alpha=0.4)
    ax.set_axisbelow(True)

    # ── (b) Error probability vs activation count ─────────────────────────
    ax = axes[1]

    n_1b = ep1['n'].astype(float)
    p1b  = np.maximum(ep1['p_err'], 1e-320)
    n_eq = ep2['n_eq'].astype(float)
    p2b  = np.maximum(ep2['p_err'], 1e-320)

    ax.semilogy(n_1b, p1b, color=C_1BIT,
                label=r'1-bit: $p_{\rm err}^{1b}(n)$')
    ax.semilogy(n_eq, p2b, color=C_2BIT, linestyle='--',
                label=r'2-bit: $p_{\rm err}^{2b}(n_{\rm eq})$')

    ax.axhline(cfg.EPSILON, color=C_EPS, linestyle=':', linewidth=1.2,
               label=f'$\\varepsilon = {cfg.EPSILON}$')
    ax.axvline(N_th_1b, color=C_1BIT, linestyle=':', linewidth=1.2,
               label=f'$N_{{\\rm th}}^{{1b}} = {int(N_th_1b)}$')
    ax.axvline(N_th_2b, color=C_2BIT, linestyle=':', linewidth=1.2,
               label=f'$N_{{\\rm th}}^{{2b}} = {N_th_2b:.1f}$')

    ax.set_xlabel(r'Active count $n$ / equiv. active count $n_{\rm eq}$')
    ax.set_ylabel(r'Error probability $p_{\rm err}$')
    ax.set_title('(b) Column-level error probability vs. activation')
    ax.legend(loc='upper left', handlelength=1.8)
    ax.set_xlim(0, cfg.COLUMN_SIZE)
    ax.set_ylim(1e-20, 1.5)
    ax.yaxis.grid(True, linestyle=':', alpha=0.4)
    ax.set_axisbelow(True)

    fig.suptitle(
        'Figure 1  Device statistics and equivalent-active-count calibration',
        fontsize=9, y=0.99)
    _savefig(fig, 'fig1_calibration')


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 2 — Mapping structural analysis (4 methods)
# ══════════════════════════════════════════════════════════════════════════════

def plot_fig2():
    """
    (a) Per-plane mean active / equiv-active count – 4 methods
    (b) Per-column risk J_c sorted descending – 4 methods
        Key message: lower J_c does not always imply better CIM accuracy.
    """
    rc  = _load('result_conventional.npz')
    rm  = _load('result_minneq.npz')
    rp1 = _load('result_proposed1.npz')
    rp2 = _load('result_proposed2.npz')

    KB = cfg.KB
    KQ = cfg.KQ

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.22, top=0.87, wspace=0.40)

    # ── (a) Per-plane mean active count ──────────────────────────────────
    ax = axes[0]

    plane_labels = []
    for m in range(KB):
        lam = cfg.LAMBDA_B[m]
        plane_labels.append(f'$B_{{{m}}}$\n($\\lambda$={lam})')
    for t in range(KQ):
        lam = cfg.LAMBDA_Q[t]
        plane_labels.append(f'$Q_{{{t}}}$\n($\\lambda$={lam})')

    n_conv = np.concatenate([rc['mean_n1b'], rc['mean_neq']])
    n_mneq = np.concatenate([rm['mean_n1b'], rm['mean_neq']])
    n_p1 = np.concatenate([rp1['mean_n1b'], rp1['mean_neq']])
    n_p2 = np.concatenate([rp2['mean_n1b'], rp2['mean_neq']])

    x     = np.arange(len(plane_labels))
    width = 0.20

    ax.bar(x - 1.5*width, n_conv, width, color=C_CONV, alpha=0.88,
           label='Conventional', zorder=3)
    ax.bar(x - 0.5*width, n_mneq, width, color=C_MNEQ, alpha=0.88,
           label='Min-neq SDR',  zorder=3)
    ax.bar(x + 0.5*width, n_p1, width, color=C_P1, alpha=0.88,
           label='Proposed1',    zorder=3)
    ax.bar(x + 1.5*width, n_p2, width, color=C_P2, alpha=0.88,
           label='Proposed2',    zorder=3)

    # N_th reference lines
    cal = _load('calibration_params.npz')
    N_th_1b = float(cal['N_th_1bit'])
    N_th_2b = float(cal['N_th_2bit'])
    ax.axhline(N_th_1b, color=C_1BIT, linestyle='--', linewidth=1.0,
               alpha=0.7, label=f'$N_{{\\rm th}}^{{1b}}={int(N_th_1b)}$', zorder=2)
    ax.axhline(N_th_2b, color=C_2BIT, linestyle='--', linewidth=1.0,
               alpha=0.7, label=f'$N_{{\\rm th}}^{{2b}}={N_th_2b:.0f}$', zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(plane_labels, fontsize=7.5)
    ax.set_ylabel(r'Mean active count $\bar{n}$ / $\bar{n}_{\rm eq}$')
    ax.set_title('(a) Per-plane mean active count')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.yaxis.grid(True, linestyle='--', alpha=0.40, zorder=0)
    ax.set_axisbelow(True)

    # Divider between 1-bit and 2-bit PE regions
    sep = KB - 0.5
    ax.axvline(sep, color='gray', linestyle='--', linewidth=0.8, zorder=2)
    n_total = len(plane_labels)
    ax.text(sep / 2 - 0.5,           1.02, '1-bit PEs', ha='center', va='bottom',
            fontsize=7.5, color='gray', transform=ax.get_xaxis_transform())
    ax.text(sep + (n_total - sep) / 2, 1.02, '2-bit PEs', ha='center', va='bottom',
            fontsize=7.5, color='gray', transform=ax.get_xaxis_transform())

    # ── (b) Per-column risk J_c ───────────────────────────────────────────
    ax = axes[1]

    J_conv = rc['J_c']
    J_mneq = rm['J_c']
    J_p1 = rp1['J_c']
    J_p2 = rp2['J_c']

    order = np.argsort(J_conv)[::-1]
    cidx  = np.arange(len(J_conv))

    Jt_c = float(rc['J_total'])
    Jt_m = float(rm['J_total'])
    Jt_p1 = float(rp1['J_total'])
    Jt_p2 = float(rp2['J_total'])

    ax.plot(cidx, J_conv[order], color=C_CONV,
            label=f'Conventional  ($\\Sigma J$={Jt_c:.0f})')
    ax.plot(cidx, J_mneq[order], color=C_MNEQ, linestyle='--',
            label=f'Min-neq SDR   ($\\Sigma J$={Jt_m:.0f})')
    ax.plot(cidx, J_p1[order], color=C_P1, linestyle='-.',
            label=f'Proposed1     ($\\Sigma J$={Jt_p1:.0f})')
    ax.plot(cidx, J_p2[order], color=C_P2, linestyle=':',
            linewidth=2.0,
            label=f'Proposed2     ($\\Sigma J$={Jt_p2:.0f})')

    ax.fill_between(cidx, J_p1[order], J_conv[order],
                    color=C_P1, alpha=0.08, zorder=1)

    ax.set_xlabel('Column index (sorted by conventional $J_c$ ↓)')
    ax.set_ylabel('Column risk $J_c$')
    ax.set_title('(b) Per-column risk $J_c$ distribution')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlim(0, len(J_conv) - 1)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, linestyle='--', alpha=0.40)
    ax.set_axisbelow(True)

    # Annotation: Proposed2 can improve accuracy even when J_c is not minimal
    ax.annotate(
        'Proposed2 may have higher $J_c$\nthan Proposed1 yet better CIM accuracy\n→ $J_c$ is not a full accuracy proxy',
        xy=(cidx[len(cidx)//4], J_p2[order][len(cidx)//4]),
        xytext=(0.55, 0.62), textcoords='axes fraction',
        fontsize=7, color=C_P2,
        arrowprops=dict(arrowstyle='->', color=C_P2, lw=1.0),
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_P2, alpha=0.9, lw=0.8),
    )

    fig.suptitle('Figure 2  SDR mapping structural analysis', fontsize=9, y=0.99)
    _savefig(fig, 'fig2_mapping')


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 3 — CIM MAC accuracy results
# ══════════════════════════════════════════════════════════════════════════════

def plot_fig3():
    """
    (a) Exact match rate — headline CIM accuracy metric (3 methods)
    (b) Per-PE-plane error rate on 2-bit planes (1-bit planes are error-free)
    """
    acc = _load('cim_accuracy.npz')

    # ── Unpack metrics ────────────────────────────────────────────────────
    exact = {
        'Conventional': float(acc['baseline_exact_rate']),
        'Proposed1':    float(acc['proposed1_exact_rate']),
        'Proposed2':    float(acc['proposed2_exact_rate']),
    }
    cos = {
        'Conventional': (float(acc['baseline_cos_mean']),  float(acc['baseline_cos_std'])),
        'Proposed1':    (float(acc['proposed1_cos_mean']), float(acc['proposed1_cos_std'])),
        'Proposed2':    (float(acc['proposed2_cos_mean']), float(acc['proposed2_cos_std'])),
    }

    # 2-bit PE error rates: shape (KQ,) per method
    pe2b_conv = acc['baseline_pe_rate_2b']
    pe2b_p1 = acc['proposed1_pe_rate_2b']
    pe2b_p2 = acc['proposed2_pe_rate_2b']
    KQ = len(pe2b_conv)

    colors_3 = [C_CONV, C_P1, C_P2]
    labels_3 = ['Conventional', 'Proposed1', 'Proposed2']

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.18, top=0.87, wspace=0.42)

    # ── (a) Exact match rate ──────────────────────────────────────────────
    ax = axes[0]

    names  = list(exact.keys())
    values = [exact[k] for k in names]
    x      = np.arange(len(names))
    bars   = ax.bar(x, [v * 100 for v in values],
                    color=colors_3, alpha=0.88, width=0.50, zorder=3)

    # Value labels on top of each bar
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f'{v*100:.1f}%',
                ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    # Improvement annotation: Proposed2 vs Conventional
    p2_val   = values[2] * 100
    conv_val   = values[0] * 100
    improvement = p2_val / max(conv_val, 1e-12)
    ax.annotate(
        f'$\\times${improvement:.1f}',
        xy=(x[2], p2_val), xytext=(x[2] + 0.35, p2_val * 0.60),
        fontsize=9, color=C_P2, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=C_P2, lw=1.2),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel('Exact match rate (%)')
    ax.set_title('(a) CIM MAC exact match rate')
    ax.set_ylim(0, min(p2_val * 1.35, 100))
    ax.yaxis.grid(True, linestyle='--', alpha=0.40, zorder=0)
    ax.set_axisbelow(True)

    # ── (b) Per-2-bit-plane PE error rate ─────────────────────────────────
    ax = axes[1]

    x_pe   = np.arange(KQ)
    width  = 0.25
    lam_q  = cfg.LAMBDA_Q

    ax.bar(x_pe - width, pe2b_conv * 100, width, color=C_CONV, alpha=0.88,
           label='Conventional', zorder=3)
    ax.bar(x_pe,         pe2b_p1 * 100, width, color=C_P1, alpha=0.88,
           label='Proposed1',    zorder=3)
    ax.bar(x_pe + width, pe2b_p2 * 100, width, color=C_P2, alpha=0.88,
           label='Proposed2',    zorder=3)

    # epsilon reference line
    ax.axhline(cfg.EPSILON * 100, color=C_EPS, linestyle=':', linewidth=1.2,
               label=f'$\\varepsilon={cfg.EPSILON}$')

    xtick_labels = [f'$Q_{{{t}}}$\n($\\lambda$={lam_q[t]})' for t in range(KQ)]
    ax.set_xticks(x_pe)
    ax.set_xticklabels(xtick_labels, fontsize=8.5)
    ax.set_ylabel('PE error rate (%)')
    ax.set_title('(b) Per-plane 2-bit PE error rate')
    ax.legend(fontsize=7.5, loc='upper right')
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, linestyle='--', alpha=0.40, zorder=0)
    ax.set_axisbelow(True)

    # Value labels
    for offset, arr in [(-width, pe2b_conv), (0, pe2b_p1), (width, pe2b_p2)]:
        for t, v in enumerate(arr):
            ax.text(x_pe[t] + offset, v * 100 + 0.3,
                    f'{v*100:.1f}%', ha='center', va='bottom', fontsize=7)

    fig.suptitle('Figure 3  CIM MAC accuracy evaluation', fontsize=9, y=0.99)
    _savefig(fig, 'fig3_accuracy')


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('=== Figure 1: Device calibration ===')
    plot_fig1()
    print('=== Figure 2: Mapping structural analysis ===')
    plot_fig2()
    print('=== Figure 3: CIM accuracy results ===')
    plot_fig3()
    print('\nDone. Check Results/fig*.pdf')
