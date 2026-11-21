#!/usr/bin/env python3
"""Diagnose shallow axial pressure maxima in the V3.1 headband array model.

Place next to ultrasound_layout_sweep_v3_headband.py and run:
    python investigate_axial_peak.py

This diagnostic reuses the EXACT V3.1 geometry, phase convention, element
far-field directivity, and 1/r propagator. It does not model the skull,
attenuation, coupled transducers, calibrated pressure, or human safety.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

try:
    import ultrasound_layout_sweep_v3_headband as model
except ImportError as e:
    raise SystemExit(
        "Place investigate_axial_peak.py in the same directory as "
        "ultrasound_layout_sweep_v3_headband.py\n"
        f"Original import error: {e}"
    )


def get_array(n: int, rows: int, band_mm: float, layout: str):
    usable = band_mm - 2 * model.BAND_EDGE_MARGIN_MM
    pitch = (usable - model.PISTON_DIAMETER_MM) / (rows - 1) if rows > 1 else 0.0
    xy = model.centers(n, rows, layout, pitch)
    diagnostics = model.geometry_diagnostics(xy, band_mm)
    return xy, diagnostics


def axial_data(xy, target_z, z):
    """Decompose axial pressure amplitude into envelope and coherence.

    At each z, amplitude_i = piston_directivity(theta_i)/r_i.
    |p(z)| = sum_i(amplitude_i) * coherence(z).
    coherence(z) is between 0 and 1 (when directivity is nonnegative).
    At the requested focus, coherent phase compensation gives coherence=1.
    """
    r_focus = np.sqrt((xy**2).sum(axis=1) + target_z**2)
    phases = np.exp(1j * model.K_MM * r_focus)
    xyz = np.column_stack((np.zeros_like(z), np.zeros_like(z), z))
    p = np.abs(model.field(xy, phases, xyz))

    rr = np.sqrt(z[:, None]**2 + (xy[None, :, :]**2).sum(axis=2))
    transverse = np.sqrt((xy**2).sum(axis=1))[None, :]
    u = model.K_MM * model.RADIUS_MM * transverse / rr
    element_amplitudes = model.piston_directivity(u) / rr
    # Current V3.1 has ka << 1, so each directivity is strictly positive.
    envelope = element_amplitudes.sum(axis=1)
    if np.any(envelope <= 0):
        raise ValueError('Unexpected nonpositive envelope')
    coherence = p / envelope
    return p, envelope, coherence, phases


def interpolated_at(z, values, at):
    return float(np.interp(at, z, values))


def no_spreading_control(xy, target_z, z):
    """NONPHYSICAL control: remove 1/r from the V3.1 propagator.

    Keep EXACTLY the same phases and piston directivity. If the shallow
    maximum disappears, that is strong evidence that geometric spreading
    explains its location within THIS simplified model.
    """
    radial = np.sqrt(np.sum(xy**2, axis=1))
    rr = np.sqrt(z[:, None]**2 + radial[None, :]**2)
    focus_rr = np.sqrt(target_z**2 + radial**2)
    directivity = model.piston_directivity(
        model.K_MM * model.RADIUS_MM * radial[None, :] / rr)
    return np.abs(np.sum(
        directivity * np.exp(1j*model.K_MM*(focus_rr[None, :] - rr)),
        axis=1))


def describe(xy, band, n, rows, layout, target, z):
    p, env, coh, phase = axial_data(xy, target, z)
    ref = interpolated_at(z, p, target)
    eref = interpolated_at(z, env, target)
    idx = int(np.argmax(p))
    j = int(np.argmax(env))
    peak_z = float(z[idx])
    ratio = float(p[idx] / ref)
    return dict(
        band_width_mm=band, n_elements=n, n_rows=rows, layout=layout,
        target_depth_mm=target,
        axis_peak_z_mm=peak_z,
        axis_peak_to_target_db=float(20 * np.log10(ratio)),
        axis_peak_pressure_relative_to_target=ratio,
        target_coherence=interpolated_at(z, coh, target),
        peak_coherence=float(coh[idx]),
        peak_envelope_relative_to_target=float(env[idx] / eref),
        max_envelope_z_in_roi_mm=float(z[j]),
    ), (p, env, coh, phase)


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--elements', type=int, default=64)
    a.add_argument('--rows', type=int, default=3)
    a.add_argument('--band-mm', type=float, default=17.0)
    a.add_argument('--layout', choices=model.LAYOUTS, default='aligned')
    a.add_argument('--focus-mm', type=float, default=40.0)
    a.add_argument('--z-min-mm', type=float, default=5.0)
    a.add_argument('--z-max-mm', type=float, default=125.0)
    a.add_argument('--step-mm', type=float, default=0.1)
    a.add_argument('--output-dir', default='axial_focus_results')
    args = a.parse_args()
    if args.step_mm <= 0 or args.z_max_mm <= args.z_min_mm:
        a.error('Invalid z range or step')
    if not args.z_min_mm < args.focus_mm < args.z_max_mm:
        a.error('Focus must lie within the scanned z range')
    if args.rows == 1 and args.layout != 'aligned':
        a.error('A one-row layout must be aligned')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    xy, geo = get_array(args.elements, args.rows, args.band_mm, args.layout)
    if not geo['geometry_feasible']:
        raise SystemExit(f"Geometry invalid: {geo['geometry_status']}")

    z = np.arange(args.z_min_mm, args.z_max_mm + args.step_mm/2, args.step_mm)
    # Include target exactly to test coherence=1 without interpolation artifacts.
    z = np.unique(np.r_[z, args.focus_mm])
    summary, (p, envelope, coherence, phases) = describe(
        xy, args.band_mm, args.elements, args.rows, args.layout, args.focus_mm, z)
    ref_p = interpolated_at(z, p, args.focus_mm)
    ref_env = interpolated_at(z, envelope, args.focus_mm)
    control = no_spreading_control(xy, args.focus_mm, z)
    control_ref = interpolated_at(z, control, args.focus_mm)
    control_peak = int(np.argmax(control))
    summary['control_no_spreading_peak_z_mm'] = float(z[control_peak])
    summary['control_no_spreading_peak_to_target_db'] = float(
        20*np.log10(control[control_peak]/control_ref))
    series = []
    for zi, pi, ei, ci, ni in zip(z, p, envelope, coherence, control):
        series.append(dict(
            z_mm=float(zi),
            pressure_relative_to_target=float(pi/ref_p),
            pressure_relative_to_target_db=float(20*np.log10(max(pi/ref_p, 1e-15))),
            geometric_envelope_relative_to_target=float(ei/ref_env),
            phase_coherence=float(ci),
            no_1_over_r_control_relative_to_target=float(ni/control_ref),
        ))
    save_csv(output/'axial_profile.csv', series)
    save_csv(output/'axial_summary.csv', [summary])

    zpk = summary['axis_peak_z_mm']
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.plot(z, p/ref_p, label='Actual modeled pressure / target pressure', linewidth=2)
    ax.plot(z, envelope/ref_env, label='Geometric amplitude envelope (sum D/r)', linestyle='--')
    ax.plot(z, coherence, label='Phase coherence (0 to 1)', linestyle=':')
    ax.plot(z, control/control_ref, label='Artificial control: remove 1/r spreading',
            linestyle='-', alpha=.7, linewidth=1.2)
    ax.axvline(args.focus_mm, color='black', linestyle='--', alpha=.6,
               label=f'Requested phase focus ({args.focus_mm:g} mm)')
    ax.axvline(zpk, color='tab:red', linestyle='-.', alpha=.85,
               label=f'Peak along axis ({zpk:.1f} mm)')
    ax.set(xlabel='Depth z (mm)', ylabel='Relative amplitude / coherence',
           title='Why can pressure peak before the requested phase focus?')
    ax.set_xlim(max(5, args.z_min_mm), min(args.z_max_mm, 90))
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    fig.savefig(output/'axial_decomposition.png', dpi=180)
    plt.close(fig)

    # XZ field slice at Y = 0. This is NOT a 3D global-peak search.
    xs = np.arange(-20.0, 20.01, 0.5)
    zs = np.arange(8.0, 80.01, 0.25)
    xx, zz = np.meshgrid(xs, zs, indexing='xy')
    xyz = np.column_stack([xx.ravel(), np.zeros(xx.size), zz.ravel()])
    magnitudes = np.abs(model.field(xy, phases, xyz)).reshape(xx.shape)
    norm = magnitudes / ref_p
    fig, ax = plt.subplots(figsize=(9, 5.3))
    img = ax.pcolormesh(xs, zs, 20*np.log10(np.maximum(norm,1e-8)),
                        shading='auto', cmap='inferno', vmin=-30, vmax=3)
    ax.scatter([0], [args.focus_mm], marker='x', color='cyan', s=85,
               label='Requested focus')
    ax.scatter([0], [zpk], marker='o', facecolors='none', edgecolors='white',
               s=65, label='Axial pressure maximum')
    ax.set(xlabel='X (mm)', ylabel='Z depth (mm)',
           title='XZ pressure slice at Y=0 (dB relative to target)')
    ax.legend(fontsize=8, loc='upper right')
    fig.colorbar(img, ax=ax, label='Relative pressure (dB)')
    fig.tight_layout()
    fig.savefig(output/'xz_pressure_slice.png', dpi=180)
    plt.close(fig)

    # Geometry sensitivity: same n=64, focus=40, aligned across widths/rows.
    # Some cases, e.g. N64/1 row, are mechanically infeasible in this model.
    sweep_rows = []
    for band in model.BAND_WIDTHS_MM:
        for rows in model.ROWS:
            arr, diag = get_array(args.elements, rows, band, 'aligned')
            rec = dict(band_width_mm=band, n_elements=args.elements,
                       n_rows=rows, layout='aligned', target_depth_mm=args.focus_mm,
                       geometry_feasible=diag['geometry_feasible'],
                       geometry_status=diag['geometry_status'])
            if diag['geometry_feasible']:
                stats, _ = describe(arr, band, args.elements, rows,
                                    'aligned', args.focus_mm, z)
                rec.update({k: stats[k] for k in (
                    'axis_peak_z_mm', 'axis_peak_to_target_db',
                    'peak_coherence', 'peak_envelope_relative_to_target')})
            else:
                rec.update({k: '' for k in (
                    'axis_peak_z_mm', 'axis_peak_to_target_db',
                    'peak_coherence', 'peak_envelope_relative_to_target')})
            sweep_rows.append(rec)
    save_csv(output/'band_row_sensitivity.csv', sweep_rows)

    print('Requested focus (mm):', args.focus_mm)
    print('On-axis peak depth (mm):', f"{zpk:.2f}")
    print('Peak/target relative pressure (dB):',
          f"{summary['axis_peak_to_target_db']:+.3f}")
    print('Phase coherence at requested target:',
          f"{summary['target_coherence']:.6f}")
    print('Control (without 1/r) peak depth (mm):',
          f"{summary['control_no_spreading_peak_z_mm']:.2f}")
    print('NOTE: no-1/r control is deliberately nonphysical, for diagnosis only.')
    print('Geometric envelope at peak / envelope at target:',
          f"{summary['peak_envelope_relative_to_target']:.3f}")
    print('Results saved to:', output.resolve())
    for name in ('axial_profile.csv', 'axial_summary.csv',
                 'band_row_sensitivity.csv', 'axial_decomposition.png',
                 'xz_pressure_slice.png'):
        print(' ', name)
    print('LIMITATIONS: Uses the V3.1 flat, homogeneous-water, ideal-drive model.')
    print('This tests a model mechanism; it does not establish in-vivo focal shift.')


if __name__ == '__main__':
    main()
