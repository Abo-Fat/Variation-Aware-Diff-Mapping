"""
2-bit & 1-bit FeFET-CIM Monte Carlo Simulation
Circuit: Vd=0.1V --[FeFET]-- Vsf --[R_limit=1MΩ]-- GND

2-bit staircase: 3 steps × 50 ns, progressively turning on Weight=3, 2, 1
  Step 1  Vg=VG_STEPS[0]  → Weight=3 on
  Step 2  Vg=VG_STEPS[1]  → Weight=3,2 on
  Step 3  Vg=VG_STEPS[2]  → Weight=3,2,1 on
  (Weight=0, Vth=+0.52V, stays off throughout)

1-bit staircase: 1 step × 50 ns, at VG_STEPS[1] (middle step of 2-bit)
  States used: 0 (Weight=1, lowest Vth) and 3 (Weight=0, highest Vth)

CIM result = total accumulated charge / Q0,  Q0 = (Vd/R_limit) × T_STEP
Monte Carlo: 1000 trials, Vth ~ N(μ_vth, σ_vth²) per state.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy import stats

# ── Parameters (aligned with config/config.py) ────────────────────────────────
SS          = 0.2              # Subthreshold swing [V/dec]
I0_DEVICE   = 1e-9               # Drain current at Vgs=Vth [A]
R_LIMIT     = 1e6                # Current-limiting resistor [Ω]
VD          = 0.1                # FeFET drain voltage [V]
KT_Q        = 0.02585            # kT/q at 300 K [V]

# 4 programmed Vth states: index 0 = lowest Vth (highest current)
VTH_NOMINAL = np.array([-0.96, -0.53, -0.023, 0.52])    # [V]  μ_vth
SIGMA_VTH   = np.array([0.0546, 0.0505, 0.0619, 0.0614]) # [V]  σ_vth

# 2-bit staircase voltages
VG_STEPS = np.array([-0.2, 0.3, 0.8])   # [V]  step 1, 2, 3
T_STEP   = 50e-9                          # [s]  step width (50 ns)

# 1-bit read voltage = middle step of 2-bit
VG_1BIT  = VG_STEPS[1]

# R variation (normal distribution, per-device)
SIGMA_R_REL = 0.01   # relative σ of R_LIMIT (e.g. 0.05 = 5%)

# Normalization reference (uses nominal R_LIMIT)
I0_CIRCUIT = VD / R_LIMIT          # [A]  = 100 nA
Q0         = I0_CIRCUIT * T_STEP   # [C]  = 5 fC


# ── 1F1R solver ───────────────────────────────────────────────────────────────
def solve_1F1R(Vg, Vth, R=None):
    if R is None:
        R = R_LIMIT

    def I_FeFET(Vgs, Vds):
        if Vds <= 0:
            return 0.0
        return I0_DEVICE * 10.0 ** ((Vgs - Vth) / SS) * (1.0 - np.exp(-Vds / KT_Q))

    def residual(Vsf):
        return I_FeFET(Vg - Vsf, VD - Vsf) - Vsf / R

    r_low  = residual(0.0)
    r_high = residual(VD - 1e-9)
    if r_low <= 0:
        return 0.0
    if r_high >= 0:
        return (VD - 1e-9) / R
    return brentq(residual, 0.0, VD - 1e-9, xtol=1e-14) / R


# ── Figure 1: IV curves (2-bit + 1-bit read voltages) ────────────────────────
Vg_sweep     = np.linspace(-1.5, 1.2, 1000)
I_hard_limit = VD / R_LIMIT

# 2-bit states: all 4; 1-bit states: 0 and 3 → solid lines; 1,2 → dashed
colors_iv    = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db']
iv_labels    = ['State 0 / W=3 (Vth=-0.96 V)  [2-bit & 1-bit]',
                'State 1 / W=2 (Vth=-0.53 V)  [2-bit only]',
                'State 2 / W=1 (Vth=-0.023 V) [2-bit only]',
                'State 3 / W=0 (Vth=+0.52 V)  [2-bit & 1-bit]']
iv_ls        = ['-', '--', '--', '-']   # solid = used in both modes

fig_iv, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for Vth, label, color, ls in zip(VTH_NOMINAL, iv_labels, colors_iv, iv_ls):
    I_curve = np.array([solve_1F1R(vg, Vth) for vg in Vg_sweep])
    ax1.plot(Vg_sweep, I_curve * 1e9, color=color, lw=2, ls=ls, label=label)
    ax2.semilogy(Vg_sweep, np.maximum(I_curve, 1e-25) * 1e9,
                 color=color, lw=2, ls=ls, label=label)

for ax in (ax1, ax2):
    ax.axhline(I_hard_limit * 1e9, color='k', ls='--', lw=1.2,
               label=f'$I_{{limit}}$ = {I_hard_limit*1e9:.0f} nA')
    # 2-bit staircase markers
    step_colors = ['#8e44ad', '#16a085', '#d35400']
    for k, (vg_s, sc) in enumerate(zip(VG_STEPS, step_colors)):
        ax.axvline(vg_s, color=sc, ls=':', lw=1.5,
                   label=f'2-bit step{k+1}: {vg_s} V')
    # 1-bit read voltage marker (bold)
    ax.axvline(VG_1BIT, color='#c0392b', ls='-', lw=2.5, alpha=0.55,
               label=f'1-bit read: {VG_1BIT} V')
    ax.set_xlabel('$V_G$ (V)', fontsize=12)
    ax.set_ylabel('$I_{cell}$ (nA)', fontsize=12)
    ax.legend(fontsize=7.5, loc='upper left')
    ax.grid(True, alpha=0.3)

ax1.set_title('Linear Scale', fontsize=12)
ax1.set_ylim(bottom=0)
ax2.set_title('Log Scale', fontsize=12)
ax2.set_ylim(1e-8, I_hard_limit * 1e9 * 5)

fig_iv.suptitle(
    f'1F1R FeFET-CIM IV  |  SS={SS*1000:.0f} mV/dec,  '
    f'$I_{{0,dev}}$={I0_DEVICE*1e9:.0f} nA,  '
    f'$R_{{limit}}$={R_LIMIT/1e3:.0f} kΩ,  $V_d$={VD} V\n'
    f'Solid IV lines = states used in both modes; dashed = 2-bit only',
    fontsize=10)
fig_iv.tight_layout()
fig_iv.savefig('1F1R_FeFET_IV.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 1F1R_FeFET_IV.png")

# Print read currents
print(f"\nCell currents at staircase voltages:")
print(f"{'State':>35}  {'Vg='+str(VG_STEPS[0])+'V':>12}  "
      f"{'Vg='+str(VG_STEPS[1])+'V':>12}  {'Vg='+str(VG_STEPS[2])+'V':>12}")
for Vth, label in zip(VTH_NOMINAL, iv_labels):
    row = [solve_1F1R(vg, Vth) * 1e9 for vg in VG_STEPS]
    print(f"{label:>35}  {row[0]:>12.3f}  {row[1]:>12.3f}  {row[2]:>12.3f}  nA")
print(f"{'I_limit':>35}  {I_hard_limit*1e9:>12.1f} nA")


# ── 2-bit Monte Carlo ─────────────────────────────────────────────────────────
def run_trial_2bit(vth_sample, R):
    results = np.zeros(4)
    for weight in range(4):
        state = 3 - weight
        vth   = vth_sample[state]
        charges = np.array([solve_1F1R(vg, vth, R) for vg in VG_STEPS]) * T_STEP
        results[weight] = np.sum(charges) / Q0
    return results


# ── 1-bit Monte Carlo ─────────────────────────────────────────────────────────
def run_trial_1bit(vth_sample, R):
    # weight_1bit=1 → state 0 (lowest Vth, ON);  weight_1bit=0 → state 3 (OFF)
    results = np.zeros(2)
    results[1] = solve_1F1R(VG_1BIT, vth_sample[0], R) * T_STEP / Q0
    results[0] = solve_1F1R(VG_1BIT, vth_sample[3], R) * T_STEP / Q0
    return results


N_MC = 1000
np.random.seed(42)

all_2bit = np.zeros((N_MC, 4))
all_1bit = np.zeros((N_MC, 2))

for i in range(N_MC):
    vth_s  = np.random.normal(VTH_NOMINAL, SIGMA_VTH)
    R_trial = np.random.normal(R_LIMIT, SIGMA_R_REL * R_LIMIT)
    R_trial = max(R_trial, R_LIMIT * 0.01)   # clamp to avoid negative/zero R
    all_2bit[i] = run_trial_2bit(vth_s, R_trial)
    all_1bit[i] = run_trial_1bit(vth_s, R_trial)

# ── Statistics ────────────────────────────────────────────────────────────────
mu_2b  = np.mean(all_2bit, axis=0)
sig_2b = np.std(all_2bit,  axis=0, ddof=1)
mu_1b  = np.mean(all_1bit, axis=0)
sig_1b = np.std(all_1bit,  axis=0, ddof=1)

print(f"\n{'='*52}")
print(f"2-bit CIM  (N={N_MC},  Q0={Q0*1e15:.1f} fC)")
print(f"{'Weight':>8}  {'Ideal':>6}  {'μ':>10}  {'σ':>10}  {'σ/μ':>8}")
for w in range(4):
    ratio = sig_2b[w]/mu_2b[w]*100 if mu_2b[w] > 0 else float('nan')
    print(f"{w:>8}  {float(w):>6.0f}  {mu_2b[w]:>10.4f}  {sig_2b[w]:>10.4f}  {ratio:>7.2f}%")

print(f"\n{'='*52}")
print(f"1-bit CIM  (N={N_MC},  Vg_read={VG_1BIT} V)")
print(f"{'Weight':>8}  {'Ideal':>6}  {'μ':>10}  {'σ':>10}  {'σ/μ':>8}")
for w in range(2):
    ratio = sig_1b[w]/mu_1b[w]*100 if mu_1b[w] > 0 else float('nan')
    print(f"{w:>8}  {float(w):>6.0f}  {mu_1b[w]:>10.4f}  {sig_1b[w]:>10.4f}  {ratio:>7.2f}%")


# ── Figure 2: combined distribution (2-bit left, 1-bit right) ────────────────
colors_2b = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c']   # w=0,1,2,3
colors_1b = ['#3498db', '#e74c3c']                          # w=0,1

fig, (ax_2b, ax_1b) = plt.subplots(1, 2, figsize=(16, 6))

# ── 2-bit panel ───────────────────────────────────────────────────────────────
for w in range(4):
    ax_2b.hist(all_2bit[:, w], bins=25, alpha=0.65, color=colors_2b[w],
               edgecolor='white', linewidth=0.5,
               label=f'W={w}  μ={mu_2b[w]:.3f}, σ={sig_2b[w]:.4f}')
    ax_2b.axvline(mu_2b[w], color=colors_2b[w], lw=1.5, ls='--')

for w in range(1, 4):
    ax_2b.axvline(w, color='k', lw=0.8, ls=':', alpha=0.4)

x2 = np.linspace(all_2bit[:, 1:].min() - 0.3, all_2bit[:, 1:].max() + 0.3, 500)
for w in range(1, 4):
    bw = (all_2bit[:, w].max() - all_2bit[:, w].min()) / 25
    ax_2b.plot(x2, stats.norm.pdf(x2, mu_2b[w], sig_2b[w]) * N_MC * bw,
               color=colors_2b[w], lw=2, ls='--', alpha=0.9)

for w in range(4):
    data = all_2bit[:, w]
    if w == 0:
        ax_2b.annotate(f'$\\mu$={mu_2b[w]:.4f}\n$\\sigma$={sig_2b[w]:.4f}',
                       xy=(mu_2b[w], 0),
                       xytext=(mu_2b[w] + 0.1, ax_2b.get_ylim()[1] * 0.5),
                       color=colors_2b[w], fontsize=9, ha='left',
                       arrowprops=dict(arrowstyle='->', color=colors_2b[w], lw=1.2))
    else:
        bw = (data.max() - data.min()) / 25
        peak = stats.norm.pdf(mu_2b[w], mu_2b[w], sig_2b[w]) * N_MC * bw
        ax_2b.annotate(f'$\\mu$={mu_2b[w]:.4f}\n$\\sigma$={sig_2b[w]:.4f}',
                       xy=(mu_2b[w], peak),
                       xytext=(mu_2b[w] + 0.06, peak * 1.08),
                       color=colors_2b[w], fontsize=9, ha='left',
                       arrowprops=dict(arrowstyle='->', color=colors_2b[w], lw=1.2))

handles, labels_leg = ax_2b.get_legend_handles_labels()
ax_2b.legend(handles, labels_leg, fontsize=9, loc='upper left',
             title='dashed = normal fit', title_fontsize=8)
ax_2b.set_xlabel('Normalized CIM Result  $Q / Q_0$', fontsize=12)
ax_2b.set_ylabel('Count', fontsize=12)
ax_2b.set_title(f'2-bit CIM  (3 steps, W=0~3)', fontsize=12)
ax_2b.grid(True, alpha=0.3)

# ── 1-bit panel ───────────────────────────────────────────────────────────────
for w in range(2):
    ax_1b.hist(all_1bit[:, w], bins=25, alpha=0.65, color=colors_1b[w],
               edgecolor='white', linewidth=0.5,
               label=f'W={w}  μ={mu_1b[w]:.3f}, σ={sig_1b[w]:.4f}')
    ax_1b.axvline(mu_1b[w], color=colors_1b[w], lw=1.5, ls='--')

ax_1b.axvline(1, color='k', lw=0.8, ls=':', alpha=0.4)

x1 = np.linspace(-0.2, all_1bit[:, 1].max() + 0.2, 500)
bw1 = (all_1bit[:, 1].max() - all_1bit[:, 1].min()) / 25
ax_1b.plot(x1, stats.norm.pdf(x1, mu_1b[1], sig_1b[1]) * N_MC * bw1,
           color=colors_1b[1], lw=2, ls='--', alpha=0.9)

# annotate w=1 (w=0 is near 0, label in legend is enough)
peak1 = stats.norm.pdf(mu_1b[1], mu_1b[1], sig_1b[1]) * N_MC * bw1
ax_1b.annotate(f'$\\mu$={mu_1b[1]:.4f}\n$\\sigma$={sig_1b[1]:.4f}',
               xy=(mu_1b[1], peak1),
               xytext=(mu_1b[1] + 0.03, peak1 * 1.08),
               color=colors_1b[1], fontsize=9, ha='left',
               arrowprops=dict(arrowstyle='->', color=colors_1b[1], lw=1.2))
ax_1b.annotate(f'$\\mu$={mu_1b[0]:.4f}\n$\\sigma$={sig_1b[0]:.4f}',
               xy=(mu_1b[0], 0),
               xytext=(mu_1b[0] + 0.05, ax_1b.get_ylim()[1] * 0.5),
               color=colors_1b[0], fontsize=9, ha='left',
               arrowprops=dict(arrowstyle='->', color=colors_1b[0], lw=1.2))

handles1, labels_leg1 = ax_1b.get_legend_handles_labels()
ax_1b.legend(handles1, labels_leg1, fontsize=9, loc='upper right',
             title='dashed = normal fit', title_fontsize=8)
ax_1b.set_xlabel('Normalized CIM Result  $Q / Q_0$', fontsize=12)
ax_1b.set_ylabel('Count', fontsize=12)
ax_1b.set_title(f'1-bit CIM  (1 step @ Vg={VG_1BIT} V, W=0~1)', fontsize=12)
ax_1b.grid(True, alpha=0.3)

fig.suptitle(
    f'FeFET-CIM Monte Carlo  N={N_MC}  |  '
    f'SS={SS*1000:.0f} mV/dec,  $R_{{limit}}$={R_LIMIT/1e3:.0f} kΩ '
    f'($\\sigma_R$={SIGMA_R_REL*100:.0f}%),  '
    f'$V_d$={VD} V,  $t_0$={T_STEP*1e9:.0f} ns',
    fontsize=11)
fig.tight_layout()
fig.savefig('CIM_MC_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: CIM_MC_distribution.png")
