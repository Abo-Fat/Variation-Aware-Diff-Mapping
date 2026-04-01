"""
Innovation 2 — Paper figure plotting script.

Figure 1: Device statistics → "equivalent active-count" calibration figure
  (a) Single-device charge output distributions (1-bit ON/OFF, 2-bit shown as cv=3..0)
  (b) Column error probability vs active count / equivalent active count,
      with N_th^{1b} and N_th^{2b} marked

Figure 2: Mapping structural analysis figure
  (a) PE/plane non-zero occupancy rate: baseline vs proposed
  (b) Per-column risk J_c (sorted): baseline vs proposed

Usage:
  cd Innovation2
  python plot_inno2.py

Output (saved to Results/):
  fig1_calibration.pdf / .png
  fig2_mapping.pdf     / .png
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
import src.config_inno2 as cfg

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'Results')

# ── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':     'sans-serif',
    'font.size':       9,
    'axes.titlesize':  9,
    'axes.labelsize':  9,
    'legend.fontsize': 7.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'lines.linewidth': 1.6,
    'figure.dpi':      150,
})

C_CONV  = '#E07B54'   # orange-red  – conventional
C_MNEQ  = '#F5A623'   # amber       – min-neq SDR
C_PROP  = '#4A90D9'   # blue        – proposed
C_1BIT  = '#2ca02c'   # green       – 1-bit curve
C_2BIT  = '#9467bd'   # purple      – 2-bit curve
C_EPS   = '#7f7f7f'   # grey        – epsilon line

# Backward-compat alias used by plot_fig2 baseline reference
C_BASE  = C_CONV


def _load(fname):
    return np.load(os.path.join(RESULTS_DIR, fname))


# ════════════════════════════════════════════════════════════════════════════
#  Figure 1 — Calibration
# ════════════════════════════════════════════════════════════════════════════

def plot_fig1():
    """
    Figure 1(a): single-device charge distributions (Gaussian approximation)
    Figure 1(b): column-level error probability vs activation count
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
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.17, top=0.87, wspace=0.38)

    # ── (a) Single-device distributions ──────────────────────────────────
    ax = axes[0]

    PALETTE_2B = ['#aec7e8', '#6baed6', '#2166ac', '#08306b']  # cv 0..3 color map (higher cv is darker)

    # 1-bit states
    for key, col, ls, lbl in [
        ('on',  '#2ca02c', '-',  '1-bit ON'),
        ('off', '#8c564b', '--', '1-bit OFF'),
    ]:
        mu  = stats_1bit[key]['mu']
        sig = max(stats_1bit[key]['var'], 1e-30) ** 0.5
        x   = np.linspace(mu - 5*sig, mu + 5*sig, 400)
        ax.plot(x, norm.pdf(x, mu, sig), color=col, linestyle=ls, label=lbl)

    # 2-bit cell values displayed in physical-strength order: 3 -> 0
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

    n_1b  = ep1['n'].astype(float)
    p1b   = np.maximum(ep1['p_err'], 1e-320)
    n_eq  = ep2['n_eq'].astype(float)
    p2b   = np.maximum(ep2['p_err'], 1e-320)

    ax.semilogy(n_1b, p1b, color=C_1BIT,
                label=r'1-bit: $p_{\rm err}^{1b}(n)$')
    ax.semilogy(n_eq, p2b, color=C_2BIT, linestyle='--',
                label=r'2-bit: $p_{\rm err}^{2b}(n_{\rm eq})$')

    # epsilon threshold
    ax.axhline(cfg.EPSILON, color=C_EPS, linestyle=':', linewidth=1.2,
               label=f'$\\varepsilon = {cfg.EPSILON}$')

    # N_th vertical lines
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
        'Figure 1  Device statistics → equivalent-active-count model calibration',
        fontsize=9, y=0.98)

    _savefig(fig, 'fig1_calibration')


# ════════════════════════════════════════════════════════════════════════════
#  Figure 2 — Mapping structural analysis
# ════════════════════════════════════════════════════════════════════════════

def plot_fig2():
    """
    Figure 2(a): per-plane mean equivalent active count  (3 methods)
    Figure 2(b): per-column risk J_c sorted descending   (3 methods)
    """
    rc  = _load('result_conventional.npz')
    rm  = _load('result_minneq.npz')
    rp  = _load('result_proposed.npz')

    KB = cfg.KB
    KQ = cfg.KQ

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.22, top=0.87, wspace=0.38)

    # ── (a) Per-plane mean equivalent active count ────────────────────────
    ax = axes[0]

    # Plane labels: B_0…B_{KB-1} (LSB→MSB), then Q_0…Q_{KQ-1}
    plane_labels = []
    for m in range(KB):
        lam = cfg.LAMBDA_B[m]
        plane_labels.append(f'$B_{{{m}}}$\n(λ={lam})')
    for t in range(KQ):
        lam = cfg.LAMBDA_Q[t]
        plane_labels.append(f'$Q_{{{t}}}$\n(λ={lam})')

    # mean_n1b: [KB], mean_neq: [KQ]  — concatenate for plotting
    n_conv = np.concatenate([rc['mean_n1b'], rc['mean_neq']])
    n_mneq = np.concatenate([rm['mean_n1b'], rm['mean_neq']])
    n_prop = np.concatenate([rp['mean_n1b'], rp['mean_neq']])

    x     = np.arange(len(plane_labels))
    width = 0.26

    ax.bar(x - width,     n_conv, width, color=C_CONV, alpha=0.85,
           label='Conventional', zorder=3)
    ax.bar(x,             n_mneq, width, color=C_MNEQ, alpha=0.85,
           label='Min-neq SDR',  zorder=3)
    ax.bar(x + width,     n_prop, width, color=C_PROP, alpha=0.85,
           label='Proposed',     zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(plane_labels, fontsize=7.5)
    ax.set_ylabel(r'Mean equiv. active count $\bar{n}_{\rm eq}$')
    ax.set_title('(a) Per-plane mean equiv. active count')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7.5, loc='upper right')
    ax.yaxis.grid(True, linestyle='--', alpha=0.45, zorder=0)
    ax.set_axisbelow(True)

    # Divider between 1-bit and 2-bit PE regions
    sep = KB - 0.5
    ax.axvline(sep, color='gray', linestyle='--', linewidth=0.8, zorder=2)
    n_total = len(plane_labels)
    ax.text(sep / 2 - 0.5,         1.035, '1-bit PEs', ha='center', va='bottom',
            fontsize=7.5, color='gray', transform=ax.get_xaxis_transform())
    ax.text(sep + (n_total - sep) / 2, 1.035, '2-bit PEs', ha='center', va='bottom',
            fontsize=7.5, color='gray', transform=ax.get_xaxis_transform())

    # ── (b) Column risk J_c distribution ─────────────────────────────────
    ax = axes[1]

    J_conv = rc['J_c']
    J_mneq = rm['J_c']
    J_prop = rp['J_c']

    # Sort by conventional risk descending (worst-first view)
    order = np.argsort(J_conv)[::-1]
    cidx  = np.arange(len(J_conv))

    Jt_c = float(rc['J_total'])
    Jt_m = float(rm['J_total'])
    Jt_p = float(rp['J_total'])

    ax.plot(cidx, J_conv[order], color=C_CONV,
            label=f'Conventional  (Σ={Jt_c:.1f})')
    ax.plot(cidx, J_mneq[order], color=C_MNEQ, linestyle='--',
            label=f'Min-neq SDR   (Σ={Jt_m:.1f})')
    ax.plot(cidx, J_prop[order], color=C_PROP, linestyle='-.',
            label=f'Proposed      (Σ={Jt_p:.1f})')

    ax.fill_between(cidx, J_prop[order], J_conv[order],
                    color=C_PROP, alpha=0.10, zorder=1)

    ax.set_xlabel('Column index (sorted by conventional risk ↓)')
    ax.set_ylabel('Column risk $J_c$')
    ax.set_title('(b) Per-column risk distribution')
    ax.legend(fontsize=7.5, loc='upper right')
    ax.set_xlim(0, len(J_conv) - 1)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, linestyle='--', alpha=0.45)
    ax.set_axisbelow(True)

    # Reduction annotation (conventional → proposed)
    red_total = 100.0 * (Jt_c - Jt_p) / max(Jt_c, 1e-12)
    ax.text(0.97, 0.96,
            f'Proposed vs. Conventional\n$J_{{\\rm total}}$: '
            f'$-${abs(red_total):.0f}%',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#aaaaaa', alpha=0.9))

    fig.suptitle('Figure 2  SDR Mapping structural analysis', fontsize=9, y=0.98)

    _savefig(fig, 'fig2_mapping')


# ════════════════════════════════════════════════════════════════════════════
#  Helper
# ════════════════════════════════════════════════════════════════════════════

def _savefig(fig, stem):
    for ext in ('pdf', 'png'):
        path = os.path.join(RESULTS_DIR, f'{stem}.{ext}')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f'  Saved -> {path}')
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('=== Figure 1: Calibration ===')
    plot_fig1()
    print('=== Figure 2: Mapping analysis ===')
    plot_fig2()
    print('Done. Check Results/fig1_*.pdf and Results/fig2_*.pdf')
