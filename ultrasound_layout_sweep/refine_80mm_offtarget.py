#!/usr/bin/env python3
"""Refine 3D off-target acoustic peaks in an 80 x 17 mm array.

Extends compare_80mm_element_counts.py without changing the acoustic model:
- same 2 mm nominal disk, 500 kHz, 40 mm phase focus in water
- same source quadrature and element-wise 1/sqrt(N) normalization
- N=21 (7 x 3) and N=60 (20 x 3)

Step A: rescan a finite 3D ROI at 1 mm spacing (configurable).
Step B: select separated off-target peak candidates, refine each locally
        with a 0.5 mm grid (configurable), and report the strongest.

Outputs are placed in a NEW directory by default, never rewriting V1/V2/V3
or earlier comparison folders.

This is not a k-Wave simulation. It uses the SAME analytical quadrature as
compare_80mm_element_counts.py and cannot independently validate that model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _fast_point_amplitudes(points, sx, sy, amp_re, amp_im, wave_number, normalization):
        results = np.empty(len(points), dtype=np.float32)
        for i in prange(len(points)):
            x, y, z = points[i]
            real = 0.0
            imag = 0.0
            for j in range(len(sx)):
                dx, dy = x - sx[j], y - sy[j]
                r = math.sqrt(dx*dx + dy*dy + z*z)
                angle = wave_number * r
                c, s = math.cos(angle), math.sin(angle)
                real += (amp_re[j]*c + amp_im[j]*s)/r
                imag += (amp_im[j]*c - amp_re[j]*s)/r
            results[i] = math.sqrt(real*real + imag*imag)/normalization
        return results

from compare_80mm_element_counts import (
    ARRAY_MM, BAND_MM, DIAM_MM, F_HZ, FOCUS_MM, SPEED_M_S,
    analytical_field, circular_sources, element_centers, K_MM
)

ROI = (-40.0, 40.0, -24.0, 24.0, 8.0, 80.0)
OFF_RADIUS_MM = 10.0


def csv_write(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def grid(lo: float, hi: float, step: float) -> np.ndarray:
    count = round((hi - lo) / step)
    if abs(lo + count * step - hi) > 1e-6:
        raise ValueError(f'Grid extent {(lo, hi)} is not divisible by {step}')
    return np.linspace(lo, hi, count + 1)


def grid_points(x, y, z):
    # For N-dimensional arrays: C order, z changes fastest.
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    return np.stack((xx.ravel(), yy.ravel(), zz.ravel()), axis=1)


def eval_normalized(points: np.ndarray, coords: np.ndarray,
                    amps: np.ndarray, target_amp: float,
                    point_chunk: int) -> np.ndarray:
    if HAS_NUMBA:
        return _fast_point_amplitudes(points, coords[:,0], coords[:,1],
                                      amps.real, amps.imag, K_MM, target_amp)
    out = np.empty(len(points), dtype=np.float32)
    # All calculations use the original quadrature, in small batches.
    for start in range(0, len(points), point_chunk):
        end = min(start + point_chunk, len(points))
        val = analytical_field(points[start:end], coords, amps, chunk=point_chunk)
        out[start:end] = np.abs(val / target_amp)
    return out


def dB(val: float) -> float:
    return float(20 * np.log10(max(val, 1e-20)))


def is_roi_boundary(loc, step):
    xlo, xhi, ylo, yhi, zlo, zhi = ROI
    return any((loc[0] <= xlo + step / 2,
                loc[0] >= xhi - step / 2,
                loc[1] <= ylo + step / 2,
                loc[1] >= yhi - step / 2,
                loc[2] <= zlo + step / 2,
                loc[2] >= zhi - step / 2))


def candidates_from_volume(p: np.ndarray, x, y, z,
                           top: int, min_sep_mm: float):
    from scipy.ndimage import maximum_filter
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij', sparse=True)
    eligible = xx * xx + yy * yy + (zz - FOCUS_MM) ** 2 >= OFF_RADIUS_MM ** 2
    local = (p >= maximum_filter(p, size=3, mode='constant', cval=-np.inf) - 1e-8)
    # Include the boundaries in the maxima search; mark them explicitly later.
    ii, jj, kk = np.where(eligible & local)
    values = p[ii, jj, kk]
    order = np.argsort(values)[::-1]
    seeds = []
    for idx in order:
        point = np.array([x[ii[idx]], y[jj[idx]], z[kk[idx]]], dtype=float)
        if any(np.linalg.norm(point - prev['point']) < min_sep_mm for prev in seeds):
            continue
        seeds.append(dict(point=point, value=float(values[idx])))
        if len(seeds) >= top:
            break
    return seeds


def local_refine(seed: np.ndarray, coords, amps, target_amp: float,
                 radius: float, step: float, point_chunk: int):
    # Allow local grid to extend beyond the original ROI to reveal a missing
    # boundary maximum. This does NOT certify the extended ROI global max.
    offset = grid(-radius, radius, step)
    x, y, z = (seed[dim] + offset for dim in range(3))
    pts = grid_points(x, y, z)
    vals = eval_normalized(pts, coords, amps, target_amp, point_chunk)
    eligible = ((pts[:, 0] ** 2 + pts[:, 1] ** 2 +
                 (pts[:, 2] - FOCUS_MM) ** 2) >= OFF_RADIUS_MM ** 2)
    vals = np.where(eligible, vals, -np.inf)
    idx = int(np.argmax(vals))
    pos = pts[idx]
    on_refine_edge = bool(np.any(np.abs(pos - seed) >= radius - step * .25))
    return dict(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]),
                amplitude=float(vals[idx]), relative_db=dB(vals[idx]),
                on_local_edge=on_refine_edge,
                outside_original_roi=is_roi_boundary(pos, 0.0))


def run_one(ncols: int, args, output_dir: Path):
    n = ncols * 3
    xy = element_centers(ncols)
    coords, amps = circular_sources(xy)
    target_amp = float(abs(analytical_field(
        np.array([[0.0, 0.0, FOCUS_MM]]), coords, amps))[0])
    x = grid(ROI[0], ROI[1], args.step_mm)
    y = grid(ROI[2], ROI[3], args.step_mm)
    z = grid(ROI[4], ROI[5], args.step_mm)
    dims = (len(x), len(y), len(z))
    point_count = int(np.prod(dims))
    print(f'[N={n}] {point_count:,} sampled 3D points, {len(coords)} disk quadrature sub-sources', flush=True)

    # Scan each X plane to limit RAM and provide a deterministic progress cue.
    plane_points = grid_points(np.array([0.]), y, z)
    values = np.empty(dims, dtype=np.float32)
    start = time.monotonic()
    progress_chunk = max(1, len(x) // 10)
    for xi, xx in enumerate(x):
        plane_points[:, 0] = xx
        values[xi] = eval_normalized(plane_points, coords, amps,
                                     target_amp, args.chunk).reshape(len(y), len(z))
        if (xi+1) % progress_chunk == 0 or xi == len(x)-1:
            elapsed = time.monotonic() - start
            print(f'  [N={n}] {xi+1}/{len(x)} X-planes completed ({(xi+1)/len(x)*100:.0f}%)',flush=True)

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij', sparse=True)
    off = xx**2 + yy**2 + (zz - FOCUS_MM)**2 >= OFF_RADIUS_MM**2
    masked = np.where(off, values, -np.inf)
    idx = np.unravel_index(int(np.argmax(masked)), values.shape)
    coarse_loc = np.array([x[idx[0]], y[idx[1]], z[idx[2]]], dtype=float)
    coarse_amp = float(masked[idx])
    seeds = candidates_from_volume(values, x, y, z, args.top,
                                   args.min_separation_mm)
    local_rows = []
    for i, item in enumerate(seeds, start=1):
        result = local_refine(item['point'], coords, amps, target_amp,
                              args.refine_radius_mm, args.refine_step_mm,
                              args.chunk)
        local_rows.append(dict(elements=n, candidate=i,
                               seed_x_mm=float(item['point'][0]),
                               seed_y_mm=float(item['point'][1]),
                               seed_z_mm=float(item['point'][2]),
                               seed_db=dB(item['value']),
                               peak_x_mm=result['x'],
                               peak_y_mm=result['y'],
                               peak_z_mm=result['z'],
                               peak_relative_to_target=result['amplitude'],
                               peak_db=result['relative_db'],
                               peak_on_refinement_boundary=result['on_local_edge'],
                               peak_on_or_beyond_original_roi_boundary=result['outside_original_roi'],
                               note='candidate refinement, not a proof of global maximum'))
        print(f"  [N={n}] candidate {i}: {result['relative_db']:+.2f} dB at ({result['x']:.1f}, {result['y']:.1f}, {result['z']:.1f}) mm", flush=True)

    best = max(local_rows, key=lambda v: v['peak_relative_to_target'])
    csv_write(output_dir / f'candidates_N{n}.csv', local_rows)
    output = dict(
        elements=n, x_columns=ncols, coarse_step_mm=args.step_mm,
        refinement_step_mm=args.refine_step_mm,
        off_target_radius_mm=OFF_RADIUS_MM,
        coarse_offtarget_peak_db=dB(coarse_amp),
        coarse_offtarget_x_mm=float(coarse_loc[0]),
        coarse_offtarget_y_mm=float(coarse_loc[1]),
        coarse_offtarget_z_mm=float(coarse_loc[2]),
        coarse_peak_on_roi_boundary=is_roi_boundary(coarse_loc,args.step_mm),
        refined_best_db=float(best['peak_db']),
        refined_best_x_mm=float(best['peak_x_mm']),
        refined_best_y_mm=float(best['peak_y_mm']),
        refined_best_z_mm=float(best['peak_z_mm']),
        refined_peak_on_local_edge=bool(best['peak_on_refinement_boundary']),
        refined_peak_on_or_beyond_roi_boundary=bool(best['peak_on_or_beyond_original_roi_boundary']),
        target_pressure_proxy=target_amp,
        model='Analytical disk quadrature in water; 3x16 sub-sources per 2 mm disk',
        caution='NOT k-Wave; not calibrated Pa or equal measured watts; ROI limited; 10mm sphere exclusion is NOT true sidelobe segmentation'
    )

    # Persist only two cross-sections, not a huge 3D data cube.
    idx_y0 = int(np.argmin(np.abs(y)))
    idx_z40 = int(np.argmin(np.abs(z - FOCUS_MM)))
    np.savez_compressed(output_dir / f'slices_N{n}.npz',
                        x_mm=x, y_mm=y, z_mm=z,
                        xz_at_y0=values[:,idx_y0,:],
                        xy_at_z40=values[:,:,idx_z40])
    return output


def plot_results(out: Path, rows: list[dict]):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2,len(rows),figsize=(6*len(rows),9), constrained_layout=True, squeeze=False)
    for j, r in enumerate(rows):
        n = r['elements']
        with np.load(out/f'slices_N{n}.npz') as d:
            x,y,z = d['x_mm'],d['y_mm'],d['z_mm']
            xz, xy = d['xz_at_y0'], d['xy_at_z40']
        axs[0,j].imshow(20*np.log10(np.maximum(xz.T,1e-8)),origin='lower',
                        extent=[x[0],x[-1],z[0],z[-1]],vmin=-30,vmax=3,
                        cmap='inferno',aspect='auto')
        axs[0,j].scatter([0],[40],marker='x',c='cyan')
        axs[0,j].set(title=f'N={n} XZ at Y=0',xlabel='X (mm)',ylabel='Z (mm)')
        axs[1,j].imshow(20*np.log10(np.maximum(xy.T,1e-8)),origin='lower',
                        extent=[x[0],x[-1],y[0],y[-1]],vmin=-30,vmax=3,
                        cmap='inferno',aspect='auto')
        axs[1,j].scatter([0],[0],marker='x',c='cyan')
        axs[1,j].set(title=f'N={n} XY at Z=40',xlabel='X (mm)',ylabel='Y (mm)')
    fig.suptitle('1 mm 3D analytical field slices (NOT k-Wave)')
    fig.savefig(out/'fine_3d_slices.png',dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--step-mm',type=float,default=1.0)
    p.add_argument('--refine-step-mm',type=float,default=0.5)
    p.add_argument('--refine-radius-mm',type=float,default=3.0)
    p.add_argument('--top',type=int,default=6)
    p.add_argument('--min-separation-mm',type=float,default=5.0)
    p.add_argument('--chunk',type=int,default=64)
    p.add_argument('--elements',type=int,nargs='+',default=[21,60],choices=[21,60])
    p.add_argument('--out',type=Path,default=Path('element_count_80mm_refined_3d'))
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if args.step_mm <= 0 or args.refine_step_mm <= 0:
        p.error('steps must be positive')
    if args.top < 1 or args.refine_radius_mm <= 0:
        p.error('top and radius must be positive')
    if not HAS_NUMBA:
        print('NOTE: numba is not installed. Calculation will be substantially slower. Install with: python -m pip install numba',flush=True)
    else:
        print('Using Numba-accelerated computation; identical analytical source model.',flush=True)
    if args.out.exists() and any(args.out.iterdir()) and not args.dry_run:
        p.error(f'Output directory {args.out} already has files. Choose a fresh --out directory to prevent overwriting.')
    if args.dry_run:
        dims = [len(grid(lo,hi,args.step_mm)) for lo,hi in zip(ROI[::2],ROI[1::2])]
        print('Output directory:',args.out)
        print('Grid dimensions:',dims, 'points:', math.prod(dims))
        print('Element counts:',args.elements)
        print('Dry run complete; no files written')
        return
    args.out.mkdir(parents=True,exist_ok=True)
    results=[]
    for n in args.elements:
        results.append(run_one(n // 3,args,args.out))
    csv_write(args.out/'fine_3d_summary.csv',results)
    baseline_file = Path('element_count_comparison_80mm/analytic_3d_summary.csv')
    old_rows = {}
    if baseline_file.is_file():
        with baseline_file.open(newline='',encoding='utf-8') as f:
            for record in csv.DictReader(f):
                old_rows[int(record['elements'])] = record
    comparison = []
    for r in results:
        old = old_rows.get(r['elements'])
        comparison.append(dict(
            elements=r['elements'],
            previous_2mm_max_offtarget_db=(old['off_target_3d_max_db'] if old else ''),
            current_1mm_max_offtarget_db=r['coarse_offtarget_peak_db'],
            current_0p5mm_local_refined_max_offtarget_db=r['refined_best_db'],
            refined_peak_x_mm=r['refined_best_x_mm'],
            refined_peak_y_mm=r['refined_best_y_mm'],
            refined_peak_z_mm=r['refined_best_z_mm'],
            peak_on_refinement_boundary=r['refined_peak_on_local_edge'],
            peak_on_original_roi_boundary=r['refined_peak_on_or_beyond_roi_boundary'],
            note='Analytical disk quadrature; not k-Wave'))
    csv_write(args.out/'resolution_comparison.csv',comparison)
    plot_results(args.out,results)
    (args.out/'config.json').write_text(json.dumps(dict(
        step_mm=args.step_mm,refine_step_mm=args.refine_step_mm,
        refine_radius_mm=args.refine_radius_mm,top=args.top,
        min_separation_mm=args.min_separation_mm,
        roi=dict(x_mm=[ROI[0],ROI[1]],y_mm=[ROI[2],ROI[3]],z_mm=[ROI[4],ROI[5]]),
        frequency_hz=F_HZ,sound_speed_m_s=SPEED_M_S,focus_mm=FOCUS_MM,
        array_length_mm=ARRAY_MM,band_width_mm=BAND_MM,element_diameter_mm=DIAM_MM,
        source_model='identical to compare_80mm_element_counts.py',
        limitations='Analytical 3D not k-Wave; sphere exclusion is not mainlobe segmentation; clipped finite ROI; no calibrated power/skull/thermal'
    ),indent=2),encoding='utf-8')
    print('Finished. Saved:',args.out.resolve(),flush=True)

if __name__=='__main__':
    main()
