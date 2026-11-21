#!/usr/bin/env python3
"""Targeted 3-D k-Wave check of analytical off-target peaks, N=21 vs N=60.

For the 80 x 17 mm flat headband array in homogeneous water:
  * 2 mm nominal circular elements, 3 rows at y = -6.5, 0, +6.5 mm
  * 21 elements: 7 columns from x=-38 to +38 mm (12.667 mm pitch)
  * 60 elements: 20 columns from x=-38 to +38 mm (4 mm pitch)
  * 500 kHz, nominal phase focus (0,0,40) mm

Uses 0.5-mm k-Wave grid, each nominal disk represented by 13 *additive
pressure source cells* driven at one element phase. All elements use
identical source-sum normalization 10000/sqrt(N); this does NOT mean
identical radiated watts or actual electrical consumption.

Samples (a) the focal point and local focus region, (b) a central axial
trace, and (c) fine local 3-D neighborhoods around ANALYTICALLY predicted
off-target maxima. The study does NOT compute a global 3-D peak.

Usage:
  python validate_offtarget_kwave.py --dry-run --elements both
  python -u validate_offtarget_kwave.py --elements 21
  python -u validate_offtarget_kwave.py --elements 60
  python validate_offtarget_kwave.py --report-only

Separate N21 and N60 outputs avoid overwriting previous experiments.
Requires: numpy, matplotlib, k-wave-python==0.6.2 for propagation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

FREQUENCY_HZ = 500_000.0
SOUND_SPEED = 1500.0
DENSITY = 1000.0
FOCUS_Z = 40.0
DIAMETER = 2.0
SOURCE_STRENGTH = 10_000.0
PHASE_K_MM = 2 * math.pi * FREQUENCY_HZ / (SOUND_SPEED * 1000.0)
AXIAL_Z_MIN = 8.0
AXIAL_Z_MAX = 60.0
SOURCE_OFFSET = 6.0  # mm from the near simulation-domain boundary

# Predictions from the 0.5-mm LOCAL REFINEMENT of the analytical model.
# They are starting points only, NOT verified k-Wave maxima.
CANDIDATES = {
    21: [(-14.0, 0.0, 21.5), (14.0, 0.0, 21.5)],
    60: [(0.0, -20.5, 34.5), (0.0, 20.5, 34.5)],
}


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def layout(n: int) -> np.ndarray:
    if n not in (21, 60):
        raise ValueError('Choose 21 or 60 elements')
    cols = n // 3
    xs = np.linspace(-38.0, 38.0, cols)
    ys = (-6.5, 0.0, 6.5)
    return np.array([(x, y) for y in ys for x in xs], dtype=float)


def make_geometry(n: int, dx: float, hotspot_radius: float,
                  focus_radius: float):
    if dx <= 0 or abs((0.5 / dx) - round(0.5 / dx)) > 1e-8:
        raise ValueError('Choose a grid spacing dividing 0.5 mm, such as 0.5 or 0.25')
    # Keep all source cells and hotspot neighborhoods inside the nominal domain.
    # pml_inside=False adds absorbing cells beyond it.
    xmax = 45.0
    ymax = max(27.0, 20.5 + hotspot_radius + 3.0)
    zmax = max(66.0, AXIAL_Z_MAX + 6.0)
    nx = 2 * int(math.ceil(xmax / dx)) + 1
    ny = 2 * int(math.ceil(ymax / dx)) + 1
    nz = int(math.ceil((SOURCE_OFFSET + zmax) / dx)) + 1
    shape = (nx, ny, nz)
    cx, cy = nx // 2, ny // 2
    source_iz = int(round(SOURCE_OFFSET / dx))

    xy = layout(n)
    ix = np.rint(xy[:, 0] / dx).astype(int) + cx
    iy = np.rint(xy[:, 1] / dx).astype(int) + cy
    snapped_xy = np.column_stack(((ix - cx) * dx, (iy - cy) * dx))
    error = float(np.max(np.linalg.norm(snapped_xy - xy, axis=1)))

    disk_owners: dict[tuple[int, int, int], int] = {}
    for eid, (x0, y0) in enumerate(zip(ix, iy)):
        r_cells = int(math.ceil((DIAMETER / 2) / dx))
        for i in range(x0 - r_cells, x0 + r_cells + 1):
            for j in range(y0 - r_cells, y0 + r_cells + 1):
                if ((i - x0) * dx) ** 2 + ((j - y0) * dx) ** 2 <= (DIAMETER/2) ** 2 + 1e-9:
                    key = (int(i), int(j), source_iz)
                    if key in disk_owners:
                        raise ValueError(f'Overlapping disk cells at {key}')
                    disk_owners[key] = eid
    counts = np.bincount(list(disk_owners.values()), minlength=n)
    assert np.all(counts > 0)
    assert all(0 <= v < s for coord in disk_owners for v, s in zip(coord, shape))

    regions = {'focus': (0.0, 0.0, FOCUS_Z)}
    for i, pos in enumerate(CANDIDATES[n], 1):
        regions[f'offtarget_{i}'] = pos
    centers_idx = {}
    for name, (x, y, z) in regions.items():
        cent = (int(round(x/dx)) + cx,
                int(round(y/dx)) + cy,
                int(round(z/dx)) + source_iz)
        centers_idx[name] = cent

    sensor_mask = np.zeros(shape, dtype=bool)
    # Focus point and central axial profile, for continuity with older runs.
    zlo = source_iz + int(round(AXIAL_Z_MIN / dx))
    zhi = source_iz + int(round(AXIAL_Z_MAX / dx))
    sensor_mask[cx, cy, zlo:zhi + 1] = True
    # 3D voxel cubes, e.g. 13x13x13 at a 3 mm radius and 0.5 mm spacing.
    region_indices = {}
    for name, center in centers_idx.items():
        radius = focus_radius if name == 'focus' else hotspot_radius
        rr = int(round(radius / dx))
        slices = tuple(slice(k - rr, k + rr + 1) for k in center)
        if any(s.start < 0 or s.stop > dim for s, dim in zip(slices, shape)):
            raise ValueError(f'{name}: sensor cube outside grid. Reduce radius or enlarge grid.')
        sensor_mask[slices] = True
        region_indices[name] = slices
    sensor_coords = np.argwhere(sensor_mask)  # C-order; same order as k-Wave-python 0.6.2
    # Verify the requested focus is exactly included.
    assert sensor_mask[centers_idx['focus']]
    return dict(shape=shape, center=(cx, cy), source_iz=source_iz,
                dx=dx, xy=xy, snapped_xy=snapped_xy, disk_owners=disk_owners,
                counts=counts, max_raster_error_mm=error,
                sensor_mask=sensor_mask, sensor_coords=sensor_coords,
                regions=regions, center_indices=centers_idx,
                region_slices=region_indices, axial_z_indices=(zlo, zhi))


def amplitude_from_window(pressures: np.ndarray, times: np.ndarray,
                          start: float, stop: float) -> np.ndarray:
    choose = (times >= start) & (times < stop)
    ts = times[choose]
    if len(ts) < 6:
        raise ValueError('Window has too few time samples')
    win = np.hanning(len(ts))
    demod = win * np.exp(-2j * np.pi * FREQUENCY_HZ * ts)
    return 2.0 * np.abs(pressures[:, choose] @ demod) / float(win.sum())


def summarize_case(n: int, geo: dict, amplitudes: np.ndarray,
                   stability: np.ndarray, out: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dx = geo['dx']
    coords = geo['sensor_coords']
    n_sensors = len(coords)
    # Mask coordinates are in C-flattening order, verified with direct lookup.
    amplitude_lookup = {tuple(idx): float(amplitudes[j]) for j, idx in enumerate(coords)}
    stability_lookup = {tuple(idx): float(stability[j]) for j, idx in enumerate(coords)}
    focal_idx = geo['center_indices']['focus']
    focus_pressure = amplitude_lookup[focal_idx]
    if not np.isfinite(focus_pressure) or focus_pressure <= 0:
        raise RuntimeError('No valid pressure at the requested focus')

    point_rows = []
    region_rows = []
    region_pictures = {}
    target_world = np.array((0.0, 0.0, FOCUS_Z))
    for name, cube in geo['region_slices'].items():
        selected = []
        for idx in np.ndindex(*(s.stop-s.start for s in cube)):
            cell = tuple(idx[j] + cube[j].start for j in range(3))
            amp = amplitude_lookup[cell]
            xyz = np.array(((cell[0] - geo['center'][0]) * dx,
                            (cell[1] - geo['center'][1]) * dx,
                            (cell[2] - geo['source_iz']) * dx))
            off_target = bool(np.linalg.norm(xyz - target_world) >= 10.0 - 1e-10)
            selected.append((cell, xyz, amp, off_target))
            point_rows.append(dict(region=name, x_mm=xyz[0], y_mm=xyz[1],
                                   z_mm=xyz[2], normalized_pressure=amp/focus_pressure,
                                   relative_db=20*math.log10(max(amp/focus_pressure, 1e-20)),
                                   outside_10mm_sphere=off_target,
                                   late_window_relative_change=stability_lookup[cell]))
        # Use the off-target restriction in hotspot cubes; not relevant in the focus cube.
        eligible = [x for x in selected if x[3]] if name != 'focus' else selected
        best = max(eligible, key=lambda q: q[2])
        cell, xyz, amp, _ = best
        touches_edge = any(cell[j] in (cube[j].start, cube[j].stop-1) for j in range(3))
        center_amp = amplitude_lookup[geo['center_indices'][name]]
        region_rows.append(dict(elements=n, region=name,
                                center_pressure_relative_db=20*math.log10(max(center_amp/focus_pressure,1e-20)),
                                center_x_mm=geo['regions'][name][0],
                                center_y_mm=geo['regions'][name][1],
                                center_z_mm=geo['regions'][name][2],
                                local_max_x_mm=xyz[0],local_max_y_mm=xyz[1],
                                local_max_z_mm=xyz[2],
                                local_max_over_focus_db=20*math.log10(max(amp/focus_pressure,1e-20)),
                                local_max_normalized_pressure=amp/focus_pressure,
                                window_boundary_peak=touches_edge,
                                spatial_window_only=True))
        mid_z = geo['center_indices'][name][2]
        # XY image at the grid plane nearest the predicted center depth.
        xx = sorted(set(int(c[0]) for c, _, _, _ in selected))
        yy = sorted(set(int(c[1]) for c, _, _, _ in selected))
        im = np.empty((len(yy), len(xx)), dtype=float)
        for j, y in enumerate(yy):
            for i, x in enumerate(xx):
                im[j, i] = amplitude_lookup[(x, y, mid_z)] / focus_pressure
        extent = ((xx[0]-geo['center'][0])*dx,(xx[-1]-geo['center'][0])*dx,
                  (yy[0]-geo['center'][1])*dx,(yy[-1]-geo['center'][1])*dx)
        region_pictures[name] = (im, extent)

    zlo, zhi = geo['axial_z_indices']
    axial = []
    for z_idx in range(zlo, zhi+1):
        idx = (geo['center'][0], geo['center'][1], z_idx)
        amp = amplitude_lookup[idx]
        axial.append(dict(z_mm=(z_idx-geo['source_iz'])*dx,
                          pressure_arbitrary=amp,
                          relative_to_focus=amp/focus_pressure,
                          relative_db=20*math.log10(max(amp/focus_pressure,1e-20))))
    write_csv(out/'sampled_3d_pressure.csv', point_rows)
    write_csv(out/'region_summary.csv', region_rows)
    write_csv(out/'axial.csv', axial)

    # The local max can land on the local measurement-window boundary.
    hotspot_rows = [r for r in region_rows if r['region'].startswith('offtarget')]
    strongest = max(hotspot_rows, key=lambda r: r['local_max_normalized_pressure'])
    focus_region = next(r for r in region_rows if r['region'] == 'focus')
    final = dict(elements=n, n_columns=n//3, source_model='rasterized_additive_pressure_disk',
                 grid_dx_mm=dx, frequency_hz=FREQUENCY_HZ,
                 requested_focus_z_mm=FOCUS_Z, focus_pressure_arbitrary=focus_pressure,
                 strongest_sampled_offtarget_db=strongest['local_max_over_focus_db'],
                 strongest_sampled_offtarget_x_mm=strongest['local_max_x_mm'],
                 strongest_sampled_offtarget_y_mm=strongest['local_max_y_mm'],
                 strongest_sampled_offtarget_z_mm=strongest['local_max_z_mm'],
                 off_target_peak_on_local_window_boundary=strongest['window_boundary_peak'],
                 focus_cube_local_max_db=focus_region['local_max_over_focus_db'],
                 scope='only analytically selected local 3D hotspot cubes, NOT a whole-volume max')
    write_csv(out/'summary.csv', [final])

    fig, ax = plt.subplots(figsize=(9,4.6))
    zz = np.array([r['z_mm'] for r in axial])
    pp = np.array([r['relative_to_focus'] for r in axial])
    ax.plot(zz, pp, label=f'k-Wave N={n}, center axis')
    ax.axvline(FOCUS_Z, ls='--', color='black', label='Requested phase focus')
    ax.set(xlabel='Z from source plane (mm)', ylabel='Pressure / 40 mm focus',
           title='Axial pressure, 3D k-Wave')
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out/'axial_profile.png', dpi=160); plt.close(fig)

    fig, axs = plt.subplots(1, len(region_pictures), figsize=(4.4*len(region_pictures),4.4))
    if len(region_pictures) == 1: axs = [axs]
    for ax, (name, (field, extent)) in zip(axs, region_pictures.items()):
        db = 20*np.log10(np.maximum(field,1e-7))
        im = ax.imshow(db, extent=extent, origin='lower', cmap='inferno',
                       vmin=-15, vmax=3, interpolation='nearest', aspect='equal')
        ax.set(title=name+' (XY at predicted Z)', xlabel='X (mm)',ylabel='Y (mm)')
        fig.colorbar(im, ax=ax, shrink=.77, label='dB relative to focus')
    fig.tight_layout(); fig.savefig(out/'local_xy_maps.png',dpi=155);plt.close(fig)

    print(f'[N={n}] Focus pressure (noncalibrated): {focus_pressure:.5g}', flush=True)
    for row in region_rows:
        print(f'  {row["region"]}: local maximum {row["local_max_over_focus_db"]:+.3f} dB at '
              f'({row["local_max_x_mm"]:.1f}, {row["local_max_y_mm"]:.1f}, '
              f'{row["local_max_z_mm"]:.1f}) mm; '
              f'boundary={row["window_boundary_peak"]}',flush=True)
    return final


def run_case(n: int, args, out_root: Path):
    geo = make_geometry(n, args.dx_mm, args.radius_mm, args.focus_radius_mm)
    nx, ny, nz = geo['shape']
    print(f'[N={n}] Grid {geo["shape"]} ({nx*ny*nz:,} voxels), dx={args.dx_mm} mm',flush=True)
    print(f'[N={n}] {len(geo["disk_owners"])} source cells, '
          f'{len(geo["sensor_coords"])} sensor cells, '
          f'{geo["max_raster_error_mm"]:.3f} mm max center rasterization error',flush=True)
    print(f'[N={n}] Candidate windows at {CANDIDATES[n]} mm',flush=True)
    if args.dry_run:
        print('[dry-run] Geometry and source/sensor C-order coordinates: PASS. No k-Wave call.',flush=True)
        return

    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksource import kSource
    from kwave.ksensor import kSensor
    from kwave.kspaceFirstOrder import kspaceFirstOrder

    grid = kWaveGrid([nx,ny,nz], [args.dx_mm*1e-3]*3)
    grid.makeTime(SOUND_SPEED, cfl=0.3, t_end=args.end_us*1e-6)
    nt, dt = int(grid.Nt), float(grid.dt)
    t = np.arange(nt) * dt

    owners = geo['disk_owners']
    source_mask = np.zeros(geo['shape'], dtype=bool)
    for pos in owners:
        source_mask[pos] = True
    sc = np.argwhere(source_mask)  # C-order source rows
    element_ids = np.array([owners[tuple(p)] for p in sc],dtype=int)
    per_cell_strength = SOURCE_STRENGTH / (math.sqrt(n) * geo['counts'][element_ids])
    r_focus = np.sqrt(np.sum(geo['snapped_xy']**2, axis=1) + FOCUS_Z**2)
    phases = PHASE_K_MM * r_focus
    ramp_duration = 8.0/FREQUENCY_HZ
    ramp = np.sin(0.5*np.pi*np.minimum(t/ramp_duration,1.0))**2
    signals = (per_cell_strength[:,None] * ramp[None,:] *
               np.sin(2*np.pi*FREQUENCY_HZ*t[None,:] + phases[element_ids,None]))
    source = kSource()
    source.p_mask = source_mask
    source.p = signals
    source.p_mode = 'additive'
    sensor = kSensor(mask=geo['sensor_mask'], record=['p'])
    medium = kWaveMedium(sound_speed=SOUND_SPEED,density=DENSITY)

    print(f'[N={n}] Starting k-Wave: {nt} time steps; '
          f'{len(geo["sensor_coords"])} sensor traces; progress below',flush=True)
    result = kspaceFirstOrder(grid,medium,source,sensor,
                              backend='python',device='cpu',pml_size=10,
                              pml_inside=False,quiet=False)
    p = np.asarray(result['p'])
    if p.shape == (nt,len(geo['sensor_coords'])):
        p = p.T
    if p.shape != (len(geo['sensor_coords']),nt):
        raise RuntimeError(f'Unexpected k-Wave sensor shape {p.shape}')
    tend = t[-1]
    period_window = 5.0 / FREQUENCY_HZ
    recent = amplitude_from_window(p, t, tend - period_window, tend + dt/2)
    prior = amplitude_from_window(p, t, tend - 2*period_window, tend - period_window)
    relative_change = np.abs(recent-prior) / np.maximum(recent,1e-20)
    out = out_root/f'N{n}'
    out.mkdir(parents=True,exist_ok=True)
    summary = summarize_case(n,geo,recent,relative_change,out)
    stability = float(np.median(relative_change))
    summary['late_window_median_relative_change'] = stability
    # Rewrite summary so it includes the stability diagnostic.
    write_csv(out/'summary.csv',[summary])
    with (out/'config.json').open('w',encoding='utf-8') as f:
        json.dump(dict(n_elements=n, element_positions=geo['xy'].tolist(),
                       focus_mm=[0,0,FOCUS_Z], predicted_hotspots_mm=CANDIDATES[n],
                       dx_mm=args.dx_mm, grid_shape=geo['shape'], source_diameter_mm=DIAMETER,
                       source_voxels=len(owners), sensors=len(geo['sensor_coords']),
                       max_raster_error_mm=geo['max_raster_error_mm'],
                       source_norm='10000/sqrt(N)/disk_source_cell_count',
                       source_type='additive pressure disk voxels; NOT physical piston velocity',
                       length_mm=80,band_mm=17,frequency_hz=FREQUENCY_HZ,
                       time_steps=nt,dt_seconds=dt,end_us=args.end_us,
                       scope='only preselected local hotspot cubes + axial trace; not whole-volume'),
                  f,indent=2)
    if stability > 0.05:
        print(f'[N={n}] WARNING: late-window median relative change {stability*100:.1f}% > 5%; '
              'consider increasing --end-us',flush=True)
    print(f'[N={n}] Saved {out.resolve()}',flush=True)
    refresh_report(out_root)


def refresh_report(out_root: Path):
    rows=[]
    for n in (21,60):
        path=out_root/f'N{n}'/'summary.csv'
        if path.exists():
            with path.open(newline='',encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    rows.append(row)
    if rows:
        write_csv(out_root/'completed_comparison.csv',rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--elements',choices=('21','60','both'),default='21')
    parser.add_argument('--dx-mm',type=float,default=0.5)
    parser.add_argument('--radius-mm',type=float,default=3.0,
                        help='Half-width of local hotspot observation cubes (default: 3 mm)')
    parser.add_argument('--focus-radius-mm',type=float,default=1.0)
    parser.add_argument('--end-us',type=float,default=85.0)
    parser.add_argument('--out',type=Path,default=Path('kwave_offtarget_validation'))
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    if args.report_only:
        refresh_report(args.out)
        print(f'Updated {args.out}/completed_comparison.csv from available completed runs')
        return
    if args.end_us < 70.0:
        parser.error('--end-us must be >= 70 us for late-window analysis')
    for value, name in ((args.radius_mm,'--radius-mm'),(args.focus_radius_mm,'--focus-radius-mm')):
        if value < 0.5 or abs(value / args.dx_mm - round(value / args.dx_mm)) > 1e-8:
            parser.error(f'{name} must be >=0.5 and divisible by --dx-mm')
    cases = (21,60) if args.elements == 'both' else (int(args.elements),)
    for n in cases:
        run_case(n,args,args.out)


if __name__ == '__main__':
    main()
