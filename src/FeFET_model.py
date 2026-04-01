"""
FeFET Device Variation Model for 1F1R CIM Structures

Physical model:
  Circuit:  Vg ──[FeFET gate], Vd ──[FeFET drain→source]── Vsf ──[R_limit]── 0 V
  Regime:   FeFET operates in subthreshold (R_limit limits current)
  Variation: Vth ~ N(μ_i, σ_i²) per programmed state i

2-bit FeFET weight states (high weight = low Vth = high current):
  weight=3 (state 0): μ_vth=-0.96 V, σ=54.6 mV  → highest current (on)
  weight=2 (state 1): μ_vth=-0.53 V, σ=50.5 mV
  weight=1 (state 2): μ_vth=-0.023 V, σ=61.9 mV
  weight=0 (state 3): μ_vth= 0.52 V, σ=61.4 mV  → lowest current (off / Ioff)

Input gate voltages:
  2-bit: input {0,1,2,3} → Vg ∈ {-1.2, -0.74, -0.275, 0.25} V
  1-bit: input {0,1}     → Vg ∈ {-1.2, -0.275} V

CIM current per cell: I_cell = dig_input × dig_weight × I_real(Vg[input], Vth_sampled)

Variation is captured through Vth_sampled ~ N(μ_state, σ_state²).
I_ratio = I_real(Vg_ref, Vth_sampled) / I_real(Vg_ref, Vth_nominal)
is applied to the bit-plane MAC.

Supported modes (FEFET_MODE):
  '1bit'  – every bit plane is an independent 1-bit cell.
              On-state (bit=1): state 0 (Vth=-0.96 V); Vg_ref = vg_input_levels[-1] = -0.275 V.
  '2bit'  – adjacent bit-plane pairs (2j, 2j+1) share one physical 2-bit cell.
              Cell value 0-3 → device state 3-0; Vg_ref = vg_input_levels[-1] = 0.25 V.
  'mixed' – LSB planes (index < split_bit) use 2-bit paired cells (Vg_ref = 0.25 V);
              MSB planes (index ≥ split_bit) use individual 1-bit cells
              (on-state = state 0, Vg_ref = mixed_1bit_vg_on ≈ -0.275 V).
              Motivation: MSB cells are more critical for accuracy, so using only the
              two extreme states (highest contrast) gives better noise margin.
"""

import numpy as np
import torch
from scipy.optimize import brentq


class FeFETVariationModel:
    """
    Physics-based FeFET variation model with multi-level input support.

    Builds a 2-D LUT (Vg_level × Vth → I_cell [A]) at initialisation by
    solving the 1F1R self-consistent operating point for every input Vg level.

    Three variation modes are supported via the `mode` parameter:
      '1bit', '2bit', 'mixed'  (see module docstring for details)
    """

    def __init__(self,
                 mode='2bit',
                 SS=0.080,
                 I0=1e-9,
                 R_limit=500e3,
                 Vd=0.1,
                 vg_input_levels=None,
                 vth_states=None,
                 sigma_vth=None,
                 mixed_split_bit=4,
                 mixed_1bit_vg_on=-0.275,
                 n_lut=2000,
                 vg_steps_2bit=None,
                 vg_read_1bit=0.0,
                 t_step=50e-9,
                 sigma_r_rel=0.01):
        """
        Args:
            mode              : '1bit', '2bit', or 'mixed'
            SS                : subthreshold swing [V/dec]
            I0                : drain current at Vgs = Vth [A]
            R_limit           : current-limiting resistor [Ω]
            Vd                : FeFET drain voltage [V]
            vg_input_levels   : list of Vgate voltages for each digital input level.
                                  '2bit' / 'mixed' default: [-1.2, -0.74, -0.275, 0.25]
                                  '1bit' default:           [-1.2, -0.275]
            vth_states        : list of 4 nominal Vth [V], ordered state 0→3
                                (state 0 = lowest Vth / highest current)
            sigma_vth         : list of 4 Vth std-deviations [V]
            mixed_split_bit   : (mixed mode only) bit-plane index where the transition
                                from 2-bit cells to 1-bit cells occurs.
                                Planes [0, split_bit) → 2-bit pairs.
                                Planes [split_bit, n_planes) → 1-bit individual.
                                Default 4 splits 8-bit weights evenly (4 LSB / 4 MSB).
            mixed_1bit_vg_on  : (mixed mode only) Vgate [V] used as the reference for
                                computing I_ratio on the 1-bit MSB section.
                                Should correspond to the '1-bit on' input voltage.
                                Default -0.275 V (index 2 in the default 2-bit Vg levels).
            n_lut             : number of LUT grid points per Vg level
            vg_steps_2bit     : staircase read gate voltages [V] for charge-based 2-bit MAC.
                                Default [-0.5, 0.0, 0.5] (aligned with test.py VG_STEPS).
            vg_read_1bit      : single read gate voltage [V] for charge-based 1-bit MAC.
                                Default 0.0 V (aligned with test.py VG_1BIT).
            t_step            : staircase pulse width [s]. Default 50 ns.
            sigma_r_rel       : relative std-dev of R_limit (per cell, normal distribution).
                                0.01 = 1%.  Set to 0.0 to disable R variation.
        """
        self.mode    = mode
        self.SS      = SS
        self.I0      = I0
        self.R_limit = R_limit
        self.Vd      = Vd
        self._mixed_split_bit = mixed_split_bit

        # Default device parameters (same 4 states for all modes)
        if vth_states is None:
            vth_states = [-0.96, -0.53, -0.023, 0.52]
        if sigma_vth is None:
            sigma_vth  = [0.0546, 0.0505, 0.0619, 0.0614]

        # Default input gate voltages
        if vg_input_levels is None:
            if mode == '1bit':
                vg_input_levels = [-1.2, -0.275]
            else:  # '2bit' or 'mixed'
                vg_input_levels = [-1.2, -0.74, -0.275, 0.25]

        self._vth_np       = np.array(vth_states,      dtype=np.float64)  # [n_states]
        self._sigma_np     = np.array(sigma_vth,        dtype=np.float64)  # [n_states]
        self._vg_levels_np = np.array(vg_input_levels,  dtype=np.float64)  # [n_levels]

        self._n_levels = len(vg_input_levels)
        self._n_states = len(vth_states)

        # For mixed mode: find the LUT row index closest to mixed_1bit_vg_on
        self._mixed_1bit_vg_on_idx = int(
            np.argmin(np.abs(self._vg_levels_np - mixed_1bit_vg_on))
        )

        # CIM integer weight value → device state index
        # High weight value = low Vth (high current) = low state index
        if mode == '1bit':
            # value {0,1} → state {3,0}
            self._val2state_np = np.array([3, 0], dtype=np.int64)
        else:  # '2bit' or 'mixed'
            # value {0,1,2,3} → state {3,2,1,0}
            self._val2state_np = np.array([3, 2, 1, 0], dtype=np.int64)

        # Build 2-D LUT: _lut_I_np[vg_idx, vth_idx] = I_cell [A]
        margin  = 5 * float(np.max(self._sigma_np)) + 0.15
        vth_min = float(np.min(self._vth_np)) - margin
        vth_max = float(np.max(self._vth_np)) + margin

        self._lut_vth_np = np.linspace(vth_min, vth_max, n_lut)
        self._lut_I_np   = np.zeros((self._n_levels, n_lut), dtype=np.float64)
        for vi, Vg in enumerate(vg_input_levels):
            for ti, Vth in enumerate(self._lut_vth_np):
                self._lut_I_np[vi, ti] = self._solve_1F1R(Vg, Vth)

        # Nominal reference currents: I_ref_np[vg_idx, state_idx]
        self._I_ref_np = np.zeros((self._n_levels, self._n_states), dtype=np.float64)
        for vi, Vg in enumerate(vg_input_levels):
            for s in range(self._n_states):
                self._I_ref_np[vi, s] = self._solve_1F1R(Vg, self._vth_np[s])

        # Lazy device-tensor cache  {str(device): dict}
        self._cache = {}

        # ── Staircase / charge-based parameters ──────────────────────────────
        self._vg_steps_2bit  = list(vg_steps_2bit) if vg_steps_2bit is not None \
                               else [-0.5, 0.0, 0.5]
        self._vg_read_1bit   = float(vg_read_1bit)
        self._t_step         = float(t_step)
        self._sigma_r_rel    = float(sigma_r_rel)
        # Q0 = I_limit × T_step  (normalization charge)
        self._Q0 = (self.Vd / self.R_limit) * self._t_step

        # 2-bit staircase Q-LUT: Q_lut_2bit[vth_idx] =
        #   Σ_k I(vg_steps[k], Vth) * T_step / Q0   ≈ weight value {0..3}
        self._lut_Q2bit_np = np.zeros(n_lut, dtype=np.float64)
        for ti, Vth in enumerate(self._lut_vth_np):
            q = sum(self._solve_1F1R(vg, Vth) for vg in self._vg_steps_2bit)
            self._lut_Q2bit_np[ti] = q * self._t_step / self._Q0

        # 1-bit Q-LUT: Q_lut_1bit[vth_idx] =
        #   I(vg_read_1bit, Vth) * T_step / Q0   ≈ {0, 1}
        self._lut_Q1bit_np = np.zeros(n_lut, dtype=np.float64)
        for ti, Vth in enumerate(self._lut_vth_np):
            self._lut_Q1bit_np[ti] = (
                self._solve_1F1R(self._vg_read_1bit, Vth) * self._t_step / self._Q0
            )

    # ── 1F1R physics ──────────────────────────────────────────────────────────

    def _i_fefet(self, Vgs, Vds, Vth):
        """Subthreshold FeFET drain current [A]."""
        if Vds <= 0.0:
            return 0.0
        return max(
            self.I0 * 10.0 ** ((Vgs - Vth) / self.SS)
            * (1.0 - np.exp(-Vds / 0.02585)),
            0.0
        )

    def _solve_1F1R(self, Vg, Vth):
        """Self-consistent 1F1R operating point at gate voltage Vg → I_cell [A]."""
        Vd, R = self.Vd, self.R_limit

        def res(Vsf):
            return self._i_fefet(Vg - Vsf, Vd - Vsf, Vth) - Vsf / R

        r_lo = res(0.0)
        r_hi = res(Vd - 1e-9)

        if r_lo <= 0.0:            # device off at this Vg/Vth
            return 0.0
        if r_hi >= 0.0:            # current at hard limit
            return (Vd - 1e-9) / R

        Vsf = brentq(res, 0.0, Vd - 1e-9, xtol=1e-14)
        return Vsf / R

    # ── Tensor helpers ────────────────────────────────────────────────────────

    def _tensors(self, device):
        """Return (or create) device-resident tensors."""
        key = str(device)
        if key not in self._cache:
            kw = dict(dtype=torch.float32, device=device)
            self._cache[key] = dict(
                lut_vth    = torch.tensor(self._lut_vth_np,   **kw),         # [n_lut]
                lut_I      = torch.tensor(self._lut_I_np,     **kw),         # [n_levels, n_lut]
                I_ref      = torch.tensor(self._I_ref_np,     **kw),         # [n_levels, n_states]
                vth_states = torch.tensor(self._vth_np,       **kw),         # [n_states]
                sigma_vth  = torch.tensor(self._sigma_np,     **kw),         # [n_states]
                val2state  = torch.tensor(self._val2state_np,
                                          dtype=torch.long, device=device),  # [n_levels]
                lut_Q2bit  = torch.tensor(self._lut_Q2bit_np, **kw),         # [n_lut]
                lut_Q1bit  = torch.tensor(self._lut_Q1bit_np, **kw),         # [n_lut]
            )
        return self._cache[key]

    @staticmethod
    def _lut_interp(x, xp, fp):
        """
        Vectorised 1-D linear LUT interpolation.
        x : query tensor (arbitrary shape)
        xp: 1-D LUT x-axis [n] (sorted ascending)
        fp: 1-D LUT y-axis [n]
        Returns tensor same shape as x.
        """
        x_c  = x.clamp(xp[0], xp[-1])
        idx  = torch.searchsorted(xp.contiguous(), x_c.contiguous())
        idx  = idx.clamp(1, xp.shape[0] - 1)
        x0, x1 = xp[idx - 1], xp[idx]
        y0, y1 = fp[idx - 1], fp[idx]
        return y0 + (y1 - y0) * (x_c - x0) / (x1 - x0 + 1e-30)

    # ── Internal helpers for apply_variation ──────────────────────────────────

    def _apply_2bit_pairs(self, w_bits, plane_slice, lut_vth, lut_I, vth_states,
                          sigma_vth, val2state, fp_ref, vth_sigma_scale, result):
        """
        Apply 2-bit paired variation to bit planes in plane_slice (a range object).
        Pairs are formed as (plane_slice[0], plane_slice[1]), (plane_slice[2], plane_slice[3]), ...
        The last plane is treated individually if the count is odd.
        Modifies `result` in-place.
        """
        planes = list(plane_slice)
        n = len(planes)
        n_pairs = n // 2

        for k in range(n_pairs):
            i0, i1 = planes[2 * k], planes[2 * k + 1]
            lsb = w_bits[i0]
            msb = w_bits[i1]

            cell_val  = lsb.long() + 2 * msb.long()   # {0,1,2,3}
            state_idx = val2state[cell_val]

            vth_nom = vth_states[state_idx]
            sigma   = sigma_vth[state_idx]

            delta  = torch.randn_like(vth_nom) * (vth_sigma_scale * sigma)
            g_act  = self._lut_interp(vth_nom + delta, lut_vth, fp_ref)
            g_nom  = self._lut_interp(vth_nom,          lut_vth, fp_ref)
            ratio  = torch.where(g_nom > 1e-30, g_act / g_nom, torch.ones_like(g_act))

            result[i0] = lsb.float() * ratio
            result[i1] = msb.float() * ratio

        # Odd trailing plane (treated as a 1-bit cell using the 2-bit on-state logic)
        if n % 2 == 1:
            ip = planes[-1]
            plane = w_bits[ip]
            # Use the same state as val2state[1] (one-valued bit → next-to-lowest Vth in 2-bit)
            s_on   = val2state[min(1, val2state.shape[0] - 1)]
            vth_on = vth_states[s_on]
            sig_on = sigma_vth[s_on]
            delta  = torch.randn_like(plane) * (vth_sigma_scale * sig_on)
            g_act  = self._lut_interp(vth_on + delta, lut_vth, fp_ref)
            g_nom  = self._lut_interp(vth_on.view(1, 1).expand_as(plane), lut_vth, fp_ref)
            result[ip] = plane.float() * (g_act / (g_nom + 1e-30))

    def _apply_1bit_individual(self, w_bits, plane_slice, lut_vth, lut_I, vth_states,
                               sigma_vth, fp_ref_1bit, vth_sigma_scale, result):
        """
        Apply 1-bit individual variation to bit planes in plane_slice.
        Every plane is independent; on-state always uses device state 0 (Vth=-0.96 V).
        Modifies `result` in-place.
        """
        # State 0: lowest Vth = strongest on-state, best noise margin
        vth_on = vth_states[0]
        sig_on = sigma_vth[0]

        for ip in plane_slice:
            plane = w_bits[ip]
            delta = torch.randn_like(plane) * (vth_sigma_scale * sig_on)
            g_act = self._lut_interp(vth_on + delta, lut_vth, fp_ref_1bit)
            g_nom = self._lut_interp(
                vth_on.view(1, 1).expand_as(plane), lut_vth, fp_ref_1bit
            )
            result[ip] = plane.float() * (g_act / (g_nom + 1e-30))

    # ── Public API ────────────────────────────────────────────────────────────

    def apply_variation(self, w_bits, vth_sigma_scale):
        """
        Apply Vth variation to weight bit planes for the bit-plane MAC.

        Mode '1bit':
          Every bit plane is independent. On-state (bit=1) uses device state 0
          (Vth=-0.96 V). Vg_ref = vg_input_levels[-1] = -0.275 V.

        Mode '2bit':
          Adjacent bit-plane pairs (2j, 2j+1) share one physical 2-bit cell.
          Cell value 0-3 determines the device state. Vg_ref = 0.25 V.
          An odd trailing plane is treated individually (see _apply_2bit_pairs).

        Mode 'mixed':
          Bit planes [0, split_bit) → 2-bit paired cells (Vg_ref = 0.25 V).
          Bit planes [split_bit, n_planes) → 1-bit individual cells using
            device state 0 exclusively (Vg_ref = mixed_1bit_vg_on ≈ -0.275 V).

        Args:
            w_bits         : float tensor [n_planes, out_ch, in_feat], values in {0, 1}
            vth_sigma_scale: float ≥ 0  (0 → no variation, 1 → paper σ)

        Returns:
            Perturbed float tensor, same shape as w_bits.
        """
        if vth_sigma_scale <= 0.0:
            return w_bits.float()

        device = w_bits.device
        t = self._tensors(device)
        lut_vth, lut_I = t['lut_vth'], t['lut_I']
        vth_states, sigma_vth, val2state = (
            t['vth_states'], t['sigma_vth'], t['val2state']
        )

        n_planes = w_bits.shape[0]
        result   = w_bits.float().clone()

        # Reference Vg row in LUT: highest input voltage (most "on")
        fp_ref_high = lut_I[self._n_levels - 1]          # e.g. 0.25 V for 2-bit
        fp_ref_1bit = lut_I[self._mixed_1bit_vg_on_idx]  # e.g. -0.275 V

        if self.mode == '1bit':
            self._apply_1bit_individual(
                w_bits, range(n_planes), lut_vth, lut_I,
                vth_states, sigma_vth, fp_ref_high, vth_sigma_scale, result
            )

        elif self.mode == '2bit':
            self._apply_2bit_pairs(
                w_bits, range(n_planes), lut_vth, lut_I,
                vth_states, sigma_vth, val2state, fp_ref_high, vth_sigma_scale, result
            )

        elif self.mode == 'mixed':
            split = min(self._mixed_split_bit, n_planes)
            # Round split down to even so the 2-bit section contains only complete pairs
            split_even = (split // 2) * 2

            # LSB section: 2-bit paired cells
            if split_even > 0:
                self._apply_2bit_pairs(
                    w_bits, range(split_even), lut_vth, lut_I,
                    vth_states, sigma_vth, val2state, fp_ref_high, vth_sigma_scale, result
                )

            # MSB section: 1-bit individual cells (state 0 = highest contrast)
            if split_even < n_planes:
                self._apply_1bit_individual(
                    w_bits, range(split_even, n_planes), lut_vth, lut_I,
                    vth_states, sigma_vth, fp_ref_1bit, vth_sigma_scale, result
                )

        else:
            raise ValueError(f"Unknown FeFET mode: '{self.mode}'. "
                             "Expected '1bit', '2bit', or 'mixed'.")

        return result

    def compute_I_ratio_multilevel(self, w_int, vth_sigma_scale, device):
        """
        Compute per-input-level, per-cell I_ratio for the 2-bit multilevel MAC.

        I_ratio[vg_idx, out_ch, in_feat] =
            I_real(Vg[vg_idx], Vth_sampled[out_ch, in_feat])
            / I_ref[vg_idx, state(w_int[out_ch, in_feat])]

        Args:
            w_int           : long tensor [out_ch, in_feat], values in {0..n_levels-1}
                              (the 2-bit weight cell values, e.g. from paired bit planes)
            vth_sigma_scale : float ≥ 0
            device          : torch device

        Returns:
            I_ratio : float tensor [n_levels, out_ch, in_feat]
                      All 1.0 when vth_sigma_scale == 0.
        """
        t = self._tensors(device)
        lut_vth, lut_I = t['lut_vth'], t['lut_I']
        vth_states, sigma_vth, val2state = (
            t['vth_states'], t['sigma_vth'], t['val2state']
        )
        I_ref = t['I_ref']   # [n_levels, n_states]

        w_int   = w_int.to(device)
        states  = val2state[w_int]        # [out_ch, in_feat]
        vth_nom = vth_states[states]      # [out_ch, in_feat]
        sigma   = sigma_vth[states]       # [out_ch, in_feat]

        if vth_sigma_scale > 0.0:
            vth_samp = vth_nom + torch.randn_like(vth_nom) * (vth_sigma_scale * sigma)
        else:
            vth_samp = vth_nom

        I_ratio = torch.ones(
            (self._n_levels,) + tuple(w_int.shape),
            dtype=torch.float32, device=device
        )
        for vi in range(self._n_levels):
            fp_vi  = lut_I[vi]
            I_act  = self._lut_interp(vth_samp, lut_vth, fp_vi)
            I_nom  = I_ref[vi][states]
            I_ratio[vi] = torch.where(
                I_nom > 1e-30, I_act / I_nom, torch.ones_like(I_act)
            )

        return I_ratio

    def get_I_ref_table(self):
        """Return nominal I_ref [A] as ndarray [n_levels, n_states]."""
        return self._I_ref_np.copy()

    def get_vg_input_levels(self):
        """Return the input Vgate levels [V] as a list."""
        return self._vg_levels_np.tolist()

    def get_mixed_split_bit(self):
        """Return the split bit index (mixed mode only)."""
        return self._mixed_split_bit

    # ── Charge-based CIM output (test.py physics) ─────────────────────────────

    def compute_cell_charge_2bit(self, cell_val, vth_sigma_scale, device):
        """
        Charge-based 2-bit CIM cell output using staircase read protocol.

        Simulates the 1F1R circuit with a 3-step staircase gate voltage
        (self._vg_steps_2bit), integrates current over T_step per step,
        and normalises by Q0 = (Vd/R_limit) * T_step.

        Ideal (no variation): Q_eff ≈ cell_val ∈ {0, 1, 2, 3}.

        Args:
            cell_val        : long tensor [out_ch, in_feat], values in {0,1,2,3}
            vth_sigma_scale : float ≥ 0  (0 → no variation)
            device          : torch device

        Returns:
            Q_eff : float tensor [out_ch, in_feat]
        """
        t = self._tensors(device)
        lut_vth   = t['lut_vth']
        lut_Q2bit = t['lut_Q2bit']
        val2state = t['val2state']
        vth_states = t['vth_states']
        sigma_vth  = t['sigma_vth']

        states  = val2state[cell_val.to(device)]    # [out_ch, in_feat]
        vth_nom = vth_states[states]                # [out_ch, in_feat]
        sigma   = sigma_vth[states]                 # [out_ch, in_feat]

        if vth_sigma_scale > 0.0:
            vth_samp = vth_nom + torch.randn_like(vth_nom) * (vth_sigma_scale * sigma)
        else:
            vth_samp = vth_nom

        Q_eff = self._lut_interp(vth_samp, lut_vth, lut_Q2bit)

        # Per-cell R variation: Q ∝ I ∝ 1/R  →  Q_eff *= R_nom / R_samp
        # R_samp = R_nom * (1 + ε),  ε ~ N(0, sigma_r_rel²)
        # → scale factor = 1 / (1 + ε)
        if self._sigma_r_rel > 0.0 and vth_sigma_scale > 0.0:
            eps = torch.randn_like(Q_eff) * self._sigma_r_rel
            Q_eff = Q_eff / (1.0 + eps).clamp(min=0.1)

        return Q_eff

    def compute_cell_charge_1bit(self, bit_plane, vth_sigma_scale, device):
        """
        Charge-based 1-bit CIM cell output using a single read voltage.

        Each cell stores one bit:
          bit = 1  →  device programmed to state 0 (Vth ≈ -0.96 V, ON)
          bit = 0  →  device programmed to state 3 (Vth ≈ +0.52 V, OFF)

        Output normalised by Q0: ideal Q_eff ≈ {0, 1}.
        The small leakage for bit=0 (off-state current of state 3) is
        physically modelled — unlike the ratio approach which gives exactly 0.

        Args:
            bit_plane       : float tensor [out_ch, in_feat], values in {0.0, 1.0}
            vth_sigma_scale : float ≥ 0  (0 → no variation)
            device          : torch device

        Returns:
            Q_eff : float tensor [out_ch, in_feat]
        """
        t = self._tensors(device)
        lut_vth   = t['lut_vth']
        lut_Q1bit = t['lut_Q1bit']
        vth_states = t['vth_states']
        sigma_vth  = t['sigma_vth']

        bit_plane = bit_plane.to(device)
        mask_on   = bit_plane > 0.5          # True where bit == 1

        # State 0: lowest Vth = ON state (bit=1)
        vth_on  = vth_states[0]              # scalar
        sig_on  = sigma_vth[0]
        # State 3: highest Vth = OFF state (bit=0)
        vth_off = vth_states[3]              # scalar
        sig_off = sigma_vth[3]

        if vth_sigma_scale > 0.0:
            noise_on  = torch.randn_like(bit_plane) * (vth_sigma_scale * sig_on)
            noise_off = torch.randn_like(bit_plane) * (vth_sigma_scale * sig_off)
            vth_samp_on  = (vth_on  + noise_on)
            vth_samp_off = (vth_off + noise_off)
        else:
            vth_samp_on  = vth_on.expand_as(bit_plane)
            vth_samp_off = vth_off.expand_as(bit_plane)

        Q_on  = self._lut_interp(vth_samp_on,  lut_vth, lut_Q1bit)
        Q_off = self._lut_interp(vth_samp_off, lut_vth, lut_Q1bit)

        Q_eff = torch.where(mask_on, Q_on, Q_off)

        if self._sigma_r_rel > 0.0 and vth_sigma_scale > 0.0:
            eps = torch.randn_like(Q_eff) * self._sigma_r_rel
            Q_eff = Q_eff / (1.0 + eps).clamp(min=0.1)

        return Q_eff
