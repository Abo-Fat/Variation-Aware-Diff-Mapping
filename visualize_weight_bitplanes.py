"""
Visualize a weight matrix and its mapped bit-planes.

Usage examples
--------------
python visualize_weight_bitplanes.py
python visualize_weight_bitplanes.py --method minneq
python visualize_weight_bitplanes.py --method proposed --seed 2026
python visualize_weight_bitplanes.py --weight-npy path\\to\\W.npy
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import src.config_inno2 as cfg
from src.decompose import build_sdr_lut
from src.baseline_mapping import conventional_map, minneq_map
from src.accuracy_optimizer import optimize_mapping_proposed1
from src.column_stats import load_calibration

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'Results')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'figure.dpi': 150,
})


def _parse_args():
    parser = argparse.ArgumentParser(description='Visualize a weight matrix and every mapped bit-plane.')
    parser.add_argument(
        '--method',
        choices=['conventional', 'minneq', 'proposed'],
        default='conventional',
        help='Mapping method used to generate bit-planes. proposed=Proposed1 optimizer.',
    )
    parser.add_argument('--seed', type=int, default=2026, help='Random seed for generated weight matrix.')
    parser.add_argument('--size', type=int, default=cfg.COLUMN_SIZE, help='Weight matrix size (if generated).')
    parser.add_argument('--weight-npy', type=str, default=None, help='Optional path to a saved weight matrix .npy file.')
    parser.add_argument('--max-iter', type=int, default=10, help='Max iterations for proposed optimizer.')
    return parser.parse_args()


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _load_or_make_weight(args):
    if args.weight_npy is not None:
        w = np.load(args.weight_npy)
        if w.ndim != 2:
            raise ValueError(f'Expected a 2D weight matrix, got shape {w.shape}.')
        return np.asarray(w, dtype=np.int64)

    rng = np.random.default_rng(args.seed)
    return rng.integers(-cfg.W_MAX, cfg.W_MAX + 1, size=(args.size, args.size))


def _load_calibration_or_fallback():
    try:
        return load_calibration(RESULTS_DIR)
    except FileNotFoundError:
        return {
            'N_th_1b': 32.0,
            'N_th_2b': 40.0,
            'kappa_2': 1.5,
            'kappa_3': 2.5,
        }


def _run_mapping(w, method, max_iter):
    cal = _load_calibration_or_fallback()
    lut = build_sdr_lut()

    if method == 'conventional':
        return conventional_map(w, cal=cal)
    if method == 'minneq':
        return minneq_map(w, cal=cal, lut=lut)
    return optimize_mapping_proposed1(w, cal=cal, lut=lut, max_iter=max_iter, verbose=False)


def _hide_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def _plot_weight(ax, w):
    vmax = max(abs(int(w.min())), abs(int(w.max())))
    im = ax.imshow(w, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax.set_title(f'Weight Matrix ({w.shape[0]}x{w.shape[1]})')
    _hide_ticks(ax)
    return im


def _plot_signed_digit(ax, arr, title):
    vmax = max(1, int(np.max(np.abs(arr))))
    im = ax.imshow(arr, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax.set_title(title)
    _hide_ticks(ax)
    return im


def _plot_positive_plane(ax, arr, title):
    vmax = max(1, int(np.max(arr)))
    im = ax.imshow(arr, cmap='YlGn', vmin=0, vmax=vmax, interpolation='nearest')
    ax.set_title(title)
    _hide_ticks(ax)
    return im


def _plot_negative_plane(ax, arr, title):
    vmax = max(1, int(np.max(arr)))
    im = ax.imshow(arr, cmap='OrRd', vmin=0, vmax=vmax, interpolation='nearest')
    ax.set_title(title)
    _hide_ticks(ax)
    return im


def _save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def _make_weight_figure(w, out_dir):
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.8))
    im = _plot_weight(ax, w)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Weight value')
    _save_figure(fig, os.path.join(out_dir, 'weight_matrix.png'))


def _make_signed_digit_figure(w, result, out_dir):
    d_b = result['D_B']
    d_q = result['D_Q']
    n_panels = 1 + d_b.shape[0] + d_q.shape[0]
    fig, axes = plt.subplots(1, n_panels, figsize=(3.4 * n_panels, 4.0))
    axes = np.atleast_1d(axes)

    im0 = _plot_weight(axes[0], w)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    idx = 1
    for m in range(d_b.shape[0]):
        im = _plot_signed_digit(axes[idx], d_b[m], f'D_B[{m}] (lambda={cfg.LAMBDA_B[m]})')
        fig.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        idx += 1

    for t in range(d_q.shape[0]):
        im = _plot_signed_digit(axes[idx], d_q[t], f'D_Q[{t}] (lambda={cfg.LAMBDA_Q[t]})')
        fig.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        idx += 1

    fig.suptitle(f'Signed-Digit Decomposition ({result["method"]})', y=1.02)
    _save_figure(fig, os.path.join(out_dir, 'signed_digits.png'))


def _make_mapped_plane_figure(result, out_dir):
    planes = result['planes']
    kb = planes['B_plus'].shape[0]
    kq = planes['Q_plus'].shape[0]
    n_cols = max(kb, kq)

    fig, axes = plt.subplots(4, n_cols, figsize=(3.1 * n_cols, 10.6))
    axes = np.atleast_2d(axes)

    for c in range(n_cols):
        if c < kb:
            im = _plot_positive_plane(axes[0, c], planes['B_plus'][c], f'B_plus[{c}]')
            fig.colorbar(im, ax=axes[0, c], fraction=0.046, pad=0.04)
            im = _plot_negative_plane(axes[1, c], planes['B_minus'][c], f'B_minus[{c}]')
            fig.colorbar(im, ax=axes[1, c], fraction=0.046, pad=0.04)
        else:
            axes[0, c].axis('off')
            axes[1, c].axis('off')

        if c < kq:
            im = _plot_positive_plane(axes[2, c], planes['Q_plus'][c], f'Q_plus[{c}]')
            fig.colorbar(im, ax=axes[2, c], fraction=0.046, pad=0.04)
            im = _plot_negative_plane(axes[3, c], planes['Q_minus'][c], f'Q_minus[{c}]')
            fig.colorbar(im, ax=axes[3, c], fraction=0.046, pad=0.04)
        else:
            axes[2, c].axis('off')
            axes[3, c].axis('off')

    fig.suptitle(f'Mapped Planes ({result["method"]})', y=1.01)
    _save_figure(fig, os.path.join(out_dir, 'mapped_planes.png'))


def _make_all_in_one_figure(w, result, out_dir):
    planes = result['planes']
    entries = [('Weight', w, 'weight')]
    for m in range(result['D_B'].shape[0]):
        entries.append((f'D_B[{m}]', result['D_B'][m], 'signed'))
    for t in range(result['D_Q'].shape[0]):
        entries.append((f'D_Q[{t}]', result['D_Q'][t], 'signed'))
    for m in range(planes['B_plus'].shape[0]):
        entries.append((f'B_plus[{m}]', planes['B_plus'][m], 'pos'))
        entries.append((f'B_minus[{m}]', planes['B_minus'][m], 'neg'))
    for t in range(planes['Q_plus'].shape[0]):
        entries.append((f'Q_plus[{t}]', planes['Q_plus'][t], 'pos'))
        entries.append((f'Q_minus[{t}]', planes['Q_minus'][t], 'neg'))

    n_cols = 4
    n_rows = (len(entries) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.0 * n_rows))
    axes = np.atleast_2d(axes)

    for ax, (title, arr, mode) in zip(axes.flat, entries):
        if mode == 'weight':
            im = _plot_weight(ax, arr)
        elif mode == 'signed':
            im = _plot_signed_digit(ax, arr, title)
        elif mode == 'pos':
            im = _plot_positive_plane(ax, arr, title)
        else:
            im = _plot_negative_plane(ax, arr, title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes.flat[len(entries):]:
        ax.axis('off')

    fig.suptitle(f'Weight and All Bit-Planes ({result["method"]})', y=1.0)
    _save_figure(fig, os.path.join(out_dir, 'all_in_one.png'))


def main():
    args = _parse_args()
    w = _load_or_make_weight(args)
    result = _run_mapping(w, args.method, args.max_iter)

    out_dir = os.path.join(RESULTS_DIR, f'bitplane_viz_{args.method}')
    _ensure_dir(out_dir)

    _make_weight_figure(w, out_dir)
    _make_signed_digit_figure(w, result, out_dir)
    _make_mapped_plane_figure(result, out_dir)
    _make_all_in_one_figure(w, result, out_dir)

    print(f'Method: {args.method}')
    print(f'Weight shape: {w.shape}')
    print(f'Output directory: {out_dir}')
    print('Saved:')
    print('  weight_matrix.png')
    print('  signed_digits.png')
    print('  mapped_planes.png')
    print('  all_in_one.png')


if __name__ == '__main__':
    main()
