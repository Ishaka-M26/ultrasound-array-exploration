#!/usr/bin/env python3
"""Validate the analytical N48 stagger vs N60 aligned shortlist with 3D k-Wave.

This script reuses the previously tested k-Wave pressure-source implementation
in validate_offtarget_kwave.py, including C-order source/sensor indexing and
steady-state window amplitude estimation. The source geometry and observation
hotspots come directly from sparse_array_screen_results_final CSVs.

This is targeted validation of preselected off-target candidate regions, NOT a
whole-volume search or a physically calibrated transducer model.

Usage (from ultrasound_layout_sweep/):
  python kwave_validate_sparse_48_vs_60.py --dry-run
  python -u kwave_validate_sparse_48_vs_60.py --elements both
  python -u kwave_validate_sparse_48_vs_60.py --elements 48
  python -u kwave_validate_sparse_48_vs_60.py --elements 60

Requires: existing validate_offtarget_kwave.py and sparse-array CSV outputs.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import validate_offtarget_kwave as base

CASES = {48: 'stagger', 60: 'aligned'}


def discover_results_dir(explicit: str | None) -> Path:
    if explicit:
        root = Path(explicit)
        if not root.is_dir():
            raise SystemExit(f'Cannot find results directory: {root}')
        return root
    for folder in ('sparse_array_screen_results_final', 'sparse_array_screen_results'):
        root = Path(folder)
        if (root / 'array_centers.csv').exists() and (root / 'offtarget_peak_candidates.csv').exists():
            return root
    raise SystemExit('Cannot locate sparse-array CSV results. Pass --screen-dir YOUR_RESULTS_FOLDER')


def load_layouts_and_candidates(root: Path, max_candidates: int):
    geometries = {}
    candidates = {}
    centers_csv = root / 'array_centers.csv'
    hotspots_csv = root / 'offtarget_peak_candidates.csv'
    for n, layout_name in CASES.items():
        rows = []
        with centers_csv.open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if int(row['elements']) == n and row['layout'] == layout_name:
                    rows.append(row)
        rows.sort(key=lambda r: int(r['element_id']))
        if len(rows) != n or [int(r['element_id']) for r in rows] != list(range(n)):
            raise ValueError(f'N={n}/{layout_name}: expected exactly {n} sequential centers, got {len(rows)}')
        positions = np.asarray([(float(r['x_mm']), float(r['y_mm'])) for r in rows], dtype=float)
        if not np.isfinite(positions).all():
            raise ValueError('Nonfinite geometry')
        geometries[n] = positions
        found = []
        with hotspots_csv.open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if (int(row['elements']) == n and row['layout'] == layout_name
                    and row['stage'] == 'local_refined'):
                    found.append(row)
        found.sort(key=lambda r: int(r['rank']))
        found = found[:max_candidates]
        if len(found) < 2:
            raise ValueError(f'N={n}/{layout_name}: need at least two refined candidate hotspots')
        candidates[n] = [tuple(float(r[k]) for k in ('x_mm','y_mm','z_mm')) for r in found]
    return geometries, candidates


def rebuild_combined_report(out_dir: Path):
    results = []
    for n, layout in CASES.items():
        p = out_dir / f'N{n}' / 'summary.csv'
        if not p.exists():
            continue
        with p.open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                results.append(dict(layout=layout, **row))
    if results:
        base.write_csv(out_dir / 'completed_comparison.csv', results)
        print(f'Wrote combined results: {out_dir / "completed_comparison.csv"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--elements', choices=('48','60','both'), default='both')
    parser.add_argument('--screen-dir', default=None,
                        help='Directory containing array_centers.csv and offtarget_peak_candidates.csv')
    parser.add_argument('--candidate-count', type=int, default=3,
                        help='Number of top refined off-target candidates per layout (default 3)')
    parser.add_argument('--dx-mm', type=float, default=0.5)
    parser.add_argument('--radius-mm', type=float, default=3.0)
    parser.add_argument('--focus-radius-mm', type=float, default=1.0)
    parser.add_argument('--end-us', type=float, default=85.0)
    parser.add_argument('--out', type=Path, default=Path('kwave_sparse_48_vs_60'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--overwrite', action='store_true',
                        help='Explicitly permit overwriting results for a case')
    args = parser.parse_args()
    if args.report_only:
        rebuild_combined_report(args.out)
        return
    if args.candidate_count < 2 or args.candidate_count > 5:
        parser.error('Choose 2..5 candidate hotspots')
    if args.end_us < 70:
        parser.error('--end-us must be >=70 us')
    if args.dx_mm <= 0 or abs(0.5/args.dx_mm-round(0.5/args.dx_mm))>1e-8:
        parser.error('Choose grid spacing 0.5 or 0.25 mm (must divide 0.5 mm)')
    for val, name in ((args.radius_mm,'--radius-mm'),(args.focus_radius_mm,'--focus-radius-mm')):
        if val<0.5 or abs(val/args.dx_mm -round(val/args.dx_mm))>1e-8:
            parser.error(f'{name} must be >=0.5 mm and divisible by dx')

    root = discover_results_dir(args.screen_dir)
    geom, candidates = load_layouts_and_candidates(root, args.candidate_count)
    cases = (48, 60) if args.elements == 'both' else (int(args.elements),)
    for n in cases:
        if not args.dry_run and not args.overwrite and (args.out / f'N{n}').exists():
            parser.error(f'{args.out / f"N{n}"} already exists: refusing to overwrite. Choose --out other_dir.')

    # Use exact screened layouts rather than approximating/reconstructing stagger.
    base.CANDIDATES = candidates
    base.layout = lambda n: geom[n].copy()
    base.refresh_report = rebuild_combined_report

    for n in cases:
        print(f'\n[N={n}] Layout={CASES[n]}, {len(geom[n])} coordinates loaded from {root}',flush=True)
        print(f'[N={n}] Hotspot candidates: {candidates[n]}',flush=True)
        run_args = SimpleNamespace(dx_mm=args.dx_mm, radius_mm=args.radius_mm,
                                   focus_radius_mm=args.focus_radius_mm,
                                   end_us=args.end_us, dry_run=args.dry_run)
        base.run_case(n, run_args, args.out)
    if not args.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)
        manifest = dict(source_screen_dir=str(root), cases={str(n):CASES[n] for n in CASES},
                        candidate_coordinates_mm={str(n):candidates[n] for n in CASES},
                        geometry_read_from='array_centers.csv', candidates_read_from='offtarget_peak_candidates.csv',
                        source_strength='10000/sqrt(N) divided by disk cells per element',
                        source='rasterized 2mm additive-pressure source, not physical piston',
                        validation_scope='only selected 3D candidate windows and axial traces, no full-volume search',
                        compare_caution='same artificial driving convention is NOT equal measured acoustic power')
        (args.out/'study_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        rebuild_combined_report(args.out)


if __name__ == '__main__':
    main()
