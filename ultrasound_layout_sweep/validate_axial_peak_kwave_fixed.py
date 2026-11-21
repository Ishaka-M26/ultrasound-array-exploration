#!/usr/bin/env python3
"""3D k-Wave independent check of the shallow on-axis pressure peak.

FIXED: 0.6.x unified Python backend uses C-order for source and sensor masks.
The previous version incorrectly sorted masks in Fortran order and must NOT
be used to interpret focal peak position.

Place this file beside ultrasound_layout_sweep_v3_headband.py.

Initial check (does not require k-Wave):
    python validate_axial_peak_kwave_fixed.py --dry-run

3D propagation (k-wave-python == 0.6.2):
    python validate_axial_peak_kwave_fixed.py

Geometry is 64 elements / 3 aligned rows / 17 mm band / 40 mm focus.
This is a homogeneous-water, grid-source diagnostic, NOT a calibrated finite-
size transducer experiment. It does not prove through-skull focusing or safety.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

try:
    import ultrasound_layout_sweep_v3_headband as v3
except ImportError as err:
    raise SystemExit(
        'Please put this script next to ultrasound_layout_sweep_v3_headband.py.\n'
        f'Import error: {err}'
    )


def setup_geometry(grid_mm):
    n, rows, band, focus = 64, 3, 17.0, 40.0
    pitch = (band - 2 * v3.BAND_EDGE_MARGIN_MM - v3.PISTON_DIAMETER_MM) / (rows - 1)
    xy = v3.centers(n, rows, 'aligned', pitch)
    info = v3.geometry_diagnostics(xy, band)
    if not info['geometry_feasible']:
        raise ValueError(f'V3.1 layout not feasible: {info}')

    # Keep X=Y=0 exactly on a grid point. The source plane is 6 mm from z=0
    # (the geometric position Z=0 for the V3 model).
    nx = int(2 * np.ceil(24 / grid_mm) + 1)
    ny = int(2 * np.ceil(19 / grid_mm) + 1)
    nz = int(np.ceil(96 / grid_mm) + 1)
    cx, cy = nx // 2, ny // 2
    sz = int(round(6.0 / grid_mm))
    shape = (nx, ny, nz)

    ix = np.rint(xy[:, 0] / grid_mm).astype(int) + cx
    iy = np.rint(xy[:, 1] / grid_mm).astype(int) + cy
    iz = np.full(n, sz, dtype=int)
    idxs = np.column_stack((ix, iy, iz))
    assert np.all(idxs >= 0) and np.all(idxs < np.array(shape))
    linear = np.ravel_multi_index(idxs.T, shape, order='C')
    if len(np.unique(linear)) != n:
        raise RuntimeError('Two elements mapped to one grid voxel. Reduce --dx-mm.')

    snapped_xy = np.column_stack(((ix - cx) * grid_mm, (iy - cy) * grid_mm))
    displacement = np.sqrt(np.sum((snapped_xy - xy) ** 2, axis=1))
    return dict(shape=shape, center=(cx, cy), source_z_index=sz,
                original_xy=xy, snapped_xy=snapped_xy, source_indexes=idxs,
                source_linear=linear, max_snap_mm=float(displacement.max()),
                focus_mm=focus, grid_mm=grid_mm)


def sensor_layout(shape, center, source_z_index, dx_mm):
    """Midplane Y=0, every 1 mm in X, every grid voxel in Z."""
    nx, ny, nz = shape
    cx, cy = center
    x_indices = np.arange(nx)
    x_indices = x_indices[(np.abs((x_indices - cx) * dx_mm) <= 20) &
                          ((x_indices - cx) % 2 == 0)]
    z_indices = np.arange(nz)
    depths = (z_indices - source_z_index) * dx_mm
    z_indices = z_indices[(depths >= 8) & (depths <= 80)]
    mask = np.zeros(shape, dtype=bool)
    mask[np.ix_(x_indices, [cy], z_indices)] = True
    source_coords = np.argwhere(mask)
    # The 0.6.x unified Python solver returns sensor traces in NumPy/C order.
    # Sorting in Fortran order (as in the former script) scrambles the Z trace.
    order = np.argsort(np.ravel_multi_index(source_coords.T, shape, order='C'))
    coords = source_coords[order]
    return mask, coords, x_indices, z_indices


def pressure_amplitude(traces, times, frequency, start, end):
    """Complex tone projection of pressure during a specified late window."""
    keep = (times >= start) & (times < end)
    t = times[keep]
    if len(t) < 20:
        raise ValueError('Insufficient samples for steady-state analysis')
    w = np.hanning(len(t))
    # Spectral projection yields amplitude up to a common factor. Normalized
    # spatial ratios do not depend on this scale.
    return np.abs(traces[:, keep] @ (w * np.exp(-2j * np.pi * frequency * t))) * (2 / np.sum(w))


def analytical_profile(xy, z_mm, focus_mm):
    r_focus = np.sqrt((xy**2).sum(axis=1) + focus_mm**2)
    phases = np.exp(1j * v3.K_MM * r_focus)
    xyz = np.column_stack((np.zeros_like(z_mm), np.zeros_like(z_mm), z_mm))
    return np.abs(v3.field(xy, phases, xyz))


def write_rows(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dx-mm', type=float, default=0.5,
                        help='Grid spacing in mm; 0.5 gives 6 cells/wavelength')
    parser.add_argument('--end-us', type=float, default=85,
                        help='Time duration in microseconds')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate geometry, no wave solver')
    parser.add_argument('--out', default='kwave_axial_validation_results_fixed')
    args = parser.parse_args()
    if not 0.25 <= args.dx_mm <= 0.75:
        parser.error('--dx-mm must be between 0.25 and 0.75 mm')
    if args.end_us < 75:
        parser.error('--end-us should be >= 75 us for late steady state at 80 mm')
    geo = setup_geometry(args.dx_mm)
    nx, ny, nz = geo['shape']
    mask, coords, xs, zs = sensor_layout(
        geo['shape'], geo['center'], geo['source_z_index'], args.dx_mm)
    print('k-Wave validation (CORRECTED C-order mask indexing): 64 elements / 3 rows / 17 mm / focus 40 mm')
    print('Grid:', geo['shape'], 'voxels', nx*ny*nz)
    print('Spatial sampling:', args.dx_mm, 'mm; cells per wavelength:',
          round((v3.C_M_S * 1000 / v3.FREQ_HZ) / args.dx_mm, 2))
    print('Source voxels:', len(geo['source_linear']),
          '; sensor voxels:', int(mask.sum()))
    print('Largest element-center rasterization error:',
          round(geo['max_snap_mm'], 3), 'mm')
    # Check index-order mapping even during --dry-run (no k-Wave required).
    check_source = np.zeros(geo['shape'], dtype=bool)
    check_idx = geo['source_indexes']
    check_source[check_idx[:, 0], check_idx[:, 1], check_idx[:, 2]] = True
    if not np.array_equal(np.flatnonzero(check_source.ravel(order='C')),
                          np.sort(geo['source_linear'])):
        raise AssertionError('Source mask index ordering is not C')
    check_sensor = np.zeros(geo['shape'], dtype=bool)
    check_sensor[tuple(coords.T)] = True
    if not np.array_equal(np.flatnonzero(check_sensor.ravel(order='C')),
                          np.ravel_multi_index(coords.T, geo['shape'], order='C')):
        raise AssertionError('Sensor mask index ordering is not C')
    print('Source/sensor C-order index consistency: PASS')
    if args.dx_mm >= v3.PISTON_DIAMETER_MM / 2:
        print('WARNING: 0.6 mm piston is NOT resolved into a proper circular surface.')
        print('         k-Wave here uses one voxel source per element, not finite pistons.')

    out = Path(args.out)
    if args.dry_run:
        print('DRY RUN passed. No propagation has been computed.')
        return

    try:
        from kwave.kgrid import kWaveGrid
        from kwave.kmedium import kWaveMedium
        from kwave.ksource import kSource
        from kwave.ksensor import kSensor
        from kwave.kspaceFirstOrder import kspaceFirstOrder
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise SystemExit('Install k-wave-python==0.6.2 and matplotlib in your active .venv312. '
                         f'Details: {err}')

    out.mkdir(parents=True, exist_ok=True)
    dx_m = args.dx_mm * 1e-3
    grid = kWaveGrid([nx, ny, nz], [dx_m, dx_m, dx_m])
    grid.makeTime(v3.C_M_S, cfl=0.3, t_end=args.end_us*1e-6)
    Nt, dt = int(grid.Nt), float(grid.dt)
    times = np.arange(Nt) * dt
    frequency = v3.FREQ_HZ
    print('Time steps:', Nt, '; dt:', round(dt*1e9, 2), 'ns')

    medium = kWaveMedium(sound_speed=v3.C_M_S, density=v3.RHO_KG_M3)
    source = kSource()
    src_mask = np.zeros(geo['shape'], dtype=bool)
    idx = geo['source_indexes']
    src_mask[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    expected_linear = np.flatnonzero(src_mask.ravel(order='C'))
    actual_linear = np.sort(geo['source_linear'])
    if not np.array_equal(expected_linear, actual_linear):
        raise AssertionError('Source mask/order mismatch before assigning phases')
    source.p_mask = src_mask
    # Unified kWave-python 0.6.x backend='python' reads mask points in C order.
    # CRITICAL: phase time series MUST have this same ordering.
    ordering = np.argsort(geo['source_linear'])
    snapped = geo['snapped_xy'][ordering]
    lengths_m = 1e-3*np.sqrt(np.sum(snapped**2, axis=1)+geo['focus_mm']**2)
    phases = 2*np.pi*frequency*lengths_m / v3.C_M_S
    ramp_time = 8 / frequency
    ramp = np.sin(0.5*np.pi*np.minimum(times/ramp_time, 1.0))**2
    # Arbitrary excitation amplitude, NOT an actual transducer pressure rating.
    source.p = (1e4 * ramp[None, :] *
                np.sin(2*np.pi*frequency*times[None, :] + phases[:, None]))
    # Additive local pressure sources are closer to a discrete monopole model
    # than specifying a physical pressure boundary across a measured piston.
    source.p_mode = 'additive'
    sensor = kSensor(mask=mask, record=['p'])

    print('Starting independent 3D wave propagation...')
    data = kspaceFirstOrder(
        grid, medium, source, sensor,
        backend='python', device='cpu',
        pml_size=10, pml_inside=False, quiet=True,
    )
    if 'p' not in data:
        raise RuntimeError(f'k-Wave did not return p; result keys: {list(data)}')
    p = np.asarray(data['p'])
    if p.shape == (Nt, len(coords)):
        p = p.T
    if p.shape != (len(coords), Nt):
        raise RuntimeError(f'Unexpected pressure shape: {p.shape}; '
                           f'expected ({len(coords)}, {Nt})')
    # Compare two successive late windows. If amplitudes still change, the
    # simulation has not reached a reliable steady harmonic regime.
    end = float(times[-1])
    window = 5 / frequency
    recent = pressure_amplitude(p, times, frequency, end-window, end+dt/2)
    prior = pressure_amplitude(p, times, frequency, end-2*window, end-window)
    rel_change = np.abs(recent-prior) / np.maximum(recent, 1e-30)
    # Verify the recorded sensor traces have the exact C-order mapping expected
    # by the k-Wave-python 0.6.x unified Python solver.
    expected_sensor_indices = np.flatnonzero(mask.ravel(order='C'))
    actual_sensor_indices = np.ravel_multi_index(coords.T, geo['shape'], order='C')
    if not np.array_equal(expected_sensor_indices, actual_sensor_indices):
        raise AssertionError('Sensor mask / C-order trace index mismatch')
    # Fill focal-plane maps in known coordinate order.
    xlookup = {v:i for i,v in enumerate(xs)}
    zlookup = {v:i for i,v in enumerate(zs)}
    plane = np.full((len(xs), len(zs)), np.nan)
    change_plane = np.full_like(plane, np.nan)
    for row, (ix, iy, iz) in enumerate(coords):
        plane[xlookup[ix], zlookup[iz]] = recent[row]
        change_plane[xlookup[ix], zlookup[iz]] = rel_change[row]
    if np.isnan(plane).any():
        raise RuntimeError('Sensor-to-grid mapping produced missing pixels')

    center_i = xlookup[geo['center'][0]]
    observed = plane[center_i]
    z_mm = (zs - geo['source_z_index'])*args.dx_mm
    ref_i = int(np.argmin(abs(z_mm-geo['focus_mm'])))
    if abs(z_mm[ref_i]-geo['focus_mm']) > args.dx_mm/2 + 1e-8:
        raise RuntimeError('Focus not in scan grid')
    if not np.isfinite(observed[ref_i]) or observed[ref_i] == 0:
        raise RuntimeError('Zero / invalid target pressure in simulation')
    obs_ratio = observed / observed[ref_i]
    anal_original = analytical_profile(geo['original_xy'], z_mm, geo['focus_mm'])
    anal_snapped = analytical_profile(geo['snapped_xy'], z_mm, geo['focus_mm'])
    anal_original /= anal_original[ref_i]
    anal_snapped /= anal_snapped[ref_i]

    # Report local region around target; do not conflate with global nearfield.
    roi = (z_mm >= 18) & (z_mm <= 60)
    idx_obs = np.where(roi)[0][int(np.argmax(obs_ratio[roi]))]
    idx_anal = np.where(roi)[0][int(np.argmax(anal_snapped[roi]))]
    idx_global = int(np.argmax(obs_ratio))
    in_focus = (z_mm >= 25) & (z_mm <= 55)
    change_focus = float(np.median(change_plane[center_i, in_focus]))
    # Diagnostic only: correlation against the independent analytic profile
    # (unity is perfect agreement; lower values call for investigating model
    # differences, resolution, and source approximations).
    profile_correlation = float(np.corrcoef(obs_ratio[roi], anal_snapped[roi])[0, 1])
    summary = {
        'axial_profile_correlation_kwave_vs_snapped_analytic_18_to_60_mm': profile_correlation,
        'mask_index_order': 'C',
        'requested_focus_mm': geo['focus_mm'],
        'kwave_local_peak_18_to_60_mm': float(z_mm[idx_obs]),
        'kwave_global_axis_peak_8_to_80_mm': float(z_mm[idx_global]),
        'kwave_peak_vs_target_db': float(20*np.log10(obs_ratio[idx_obs])),
        'v3_snapped_local_peak_mm': float(z_mm[idx_anal]),
        'median_relative_change_last_2_windows_25_to_55_mm': change_focus,
        'dx_mm': args.dx_mm,
        'nominal_piston_diameter_mm': v3.PISTON_DIAMETER_MM,
        'max_raster_error_mm': geo['max_snap_mm'],
        'grid_voxels': nx*ny*nz,
        'time_steps': Nt,
        'model': '3D k-Wave homogeneous water, one source voxel per element',
    }
    rows = []
    for zi, val, orig, snapped_val, unst in zip(z_mm, obs_ratio, anal_original,
                                                  anal_snapped, change_plane[center_i]):
        rows.append({'z_mm': float(zi),
                     'kwave_pressure_relative_to_target': float(val),
                     'kwave_pressure_relative_db': float(20*np.log10(max(val,1e-12))),
                     'v3_original_relative_to_target': float(orig),
                     'v3_snapped_relative_to_target': float(snapped_val),
                     'window_relative_change': float(unst)})
    write_rows(out/'kwave_axial_comparison.csv', rows)
    write_rows(out/'kwave_summary.csv', [summary])
    (out/'simulation_config.json').write_text(json.dumps({
        **summary,
        'shape_xyz': geo['shape'],
        'time_end_us': float(times[-1]*1e6),
        'frequency_hz': frequency,
        'wave_speed_m_s': v3.C_M_S,
        'source_elements': 64,
        'rows': 3,
        'band_mm': 17,
        'plane': 'XZ at Y=0',
    }, indent=2), encoding='utf-8')

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(z_mm, obs_ratio, label='k-Wave 3D (grid sources)', lw=2)
    ax.plot(z_mm, anal_original, label='V3.1 analytic (original centers)', ls='--')
    ax.plot(z_mm, anal_snapped, label='V3.1 analytic (snapped centers)', ls=':')
    ax.axvline(geo['focus_mm'], color='black', ls='--', label='Intended phase focus: 40 mm')
    ax.axvline(z_mm[idx_obs], color='red', ls='-.',
               label=f'k-Wave local axial peak: {z_mm[idx_obs]:.1f} mm')
    ax.set(xlabel='Distance from element plane (mm)', ylabel='Pressure amplitude / at 40 mm',
           title='Independent 3D k-Wave axial focusing check')
    ax.set_xlim(8, 80)
    ax.grid(alpha=.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out/'kwave_vs_v3_axial.png', dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    x_mm = (xs - geo['center'][0]) * args.dx_mm
    pcm = ax.pcolormesh(x_mm, z_mm,
                         20*np.log10(np.maximum(plane.T/observed[ref_i],1e-6)),
                         cmap='inferno', shading='auto', vmin=-30, vmax=3)
    ax.scatter([0], [geo['focus_mm']], marker='x', s=80, color='cyan',
               label='Requested focus')
    ax.scatter([0], [z_mm[idx_obs]], marker='o', s=70,
               facecolors='none', edgecolors='white', label='Local axial peak')
    ax.set(xlabel='X (mm)', ylabel='Distance from source plane (mm)',
           title='3D k-Wave: XZ slice at Y=0')
    ax.legend(fontsize=9)
    fig.colorbar(pcm, ax=ax, label='Relative pressure amplitude (dB)')
    fig.tight_layout()
    fig.savefig(out/'kwave_xz_slice.png', dpi=170)
    plt.close(fig)

    print('k-Wave local axial peak depth:', f'{z_mm[idx_obs]:.1f} mm')
    print('V3.1 snapped-center local peak:', f'{z_mm[idx_anal]:.1f} mm')
    print('Local peak / target:', f'{summary["kwave_peak_vs_target_db"]:+.2f} dB')
    print('Steady-state change (median 25-55 mm):', f'{change_focus*100:.1f}%')
    print('Profile correlation vs snapped analytic (18-60 mm):', f'{profile_correlation:.3f}')
    if not np.isfinite(profile_correlation) or profile_correlation < 0.75:
        print('WARNING: Poor agreement with analytic trend. Do not interpret the')
        print('         peak depth as an independently confirmed physical result.')
    if change_focus > 0.05:
        print('WARNING: >5% time-window change; increase --end-us before interpreting.')
    print('Saved to:', out.resolve())
    for f in ('kwave_axial_comparison.csv','kwave_summary.csv',
              'kwave_vs_v3_axial.png','kwave_xz_slice.png'):
        print(' ',f)
    print('LIMITATIONS: source represented by grid voxels, not calibrated pistons;')
    print('homogeneous water, no skull, attenuation, thermal or electrical power.')


if __name__ == '__main__':
    main()
