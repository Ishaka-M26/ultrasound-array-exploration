#!/usr/bin/env python3
"""80 x 17 mm headband array: 21 vs 60 finite-disk source footprints.

Computes a 3D coarse analytical acoustic-pressure proxy with matched per-element
integrated source-strength RMS (each transducer weighted 1/sqrt(N)); compares
existing k-Wave axial traces WITHOUT re-running the time-domain wave solver.

The 3D analytical model is a sum of finite disk quadrature monopoles in uniform
water, NOT a velocity-boundary-condition piston, equal radiated acoustic watts,
or tissue safety/performance model. The 3D off-target metric is the maximum
pressure at least 10 mm from the intended focus inside the stated bounding ROI.
It is NOT a rigorously segmented sidelobe peak.

Usage from the ultrasound_layout_sweep folder:
    python compare_80mm_element_counts.py
    python compare_80mm_element_counts.py --mode analytic
    python compare_80mm_element_counts.py --mode existing
    python compare_80mm_element_counts.py --mode both --out element_count_comparison_80mm

Existing input defaults:
    aperture_length_results/kwave/fixed_count_L80/axial.csv  (N21)
    finite_transducer_results/disk_axial.csv                (N60)

Expected environment: Python 3.12, numpy, matplotlib (no k-wave-python needed).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

F_HZ = 500_000.0
SPEED_M_S = 1500.0
K_MM = 2 * np.pi * F_HZ / (SPEED_M_S * 1_000.0)
FOCUS_MM = 40.0
BAND_MM = 17.0
ARRAY_MM = 80.0
DIAM_MM = 2.0
Y_CENTERS_MM = np.array([-6.5, 0.0, 6.5])


def element_centers(n_cols: int):
    # Both models use the same endpoint centers, +/-38 mm. The 60-element
    # configuration then has pitch 4 mm and 21-element has 12.6667 mm.
    xx = np.linspace(-38.0, 38.0, n_cols)
    xy = np.array([(x, y) for y in Y_CENTERS_MM for x in xx], dtype=float)
    for x, y in xy:
        assert abs(x) + DIAM_MM / 2 + 1.0 <= ARRAY_MM / 2 + 1e-10
        assert abs(y) + DIAM_MM / 2 + 1.0 <= BAND_MM / 2 + 1e-10
    return xy


def circular_sources(xy: np.ndarray):
    """3 radial Gauss nodes x 16 angles; equal AREA-AVERAGED disk strength.

    Reproduces the quadrature method of aperture_length_study.py.
    Each element is one independently phase-controlled unit; the quadrature
    points within one disk share its phase and their weights sum to 1/N**0.5.
    """
    nodes, ww = np.polynomial.legendre.leggauss(3)
    rr = (nodes + 1) / 2
    rw = ww / 2
    angles = np.arange(16) * (2 * np.pi / 16)
    offsets = []
    weights = []
    for r, w in zip(rr, rw):
        for theta in angles:
            offsets.append((r * math.cos(theta), r * math.sin(theta)))
            weights.append(2 * r * w / len(angles))
    offsets = np.array(offsets) * DIAM_MM / 2
    weights = np.array(weights)
    assert abs(weights.sum() - 1.0) < 1e-12
    coords = (xy[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
    ph = np.exp(1j * K_MM * np.sqrt((xy**2).sum(axis=1) + FOCUS_MM**2))
    amplitudes = (ph[:, None] * weights[None, :] / math.sqrt(len(xy))).ravel()
    return coords, amplitudes


def analytical_field(points: np.ndarray, coords: np.ndarray, amplitudes: np.ndarray,
                     chunk: int = 96) -> np.ndarray:
    p = np.empty(len(points), dtype=np.complex128)
    sx = coords[:, 0]
    sy = coords[:, 1]
    for i in range(0, len(points), chunk):
        pos = points[i:i + chunk]
        r = np.sqrt((pos[:, None, 0] - sx[None, :])**2 +
                    (pos[:, None, 1] - sy[None, :])**2 +
                    pos[:, None, 2]**2)
        p[i:i + chunk] = (np.exp(-1j * K_MM * r) / r) @ amplitudes
    return p


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fwhm_around_target(x, p):
    """Crossing of pressure amplitude peak/sqrt(2) around x==0 or z~peak."""
    center = int(np.argmin(abs(x)))
    # The chosen axial peak is expected in [18,60], and for lateral cuts the
    # center focus is the relevant main-lobe peak, not off-axis grating lobes.
    threshold = p[center] / math.sqrt(2)
    left = center
    while left > 0 and p[left - 1] >= threshold:
        left -= 1
    right = center
    while right < len(x) - 1 and p[right + 1] >= threshold:
        right += 1
    if left == 0 or right == len(x) - 1:
        return float('nan')
    def root(a, b):
        if abs(p[b] - p[a]) < 1e-20:
            return float(x[a])
        return float(x[a] + (threshold - p[a]) * (x[b] - x[a])/(p[b] - p[a]))
    return root(right, right + 1) - root(left - 1, left)


def run_analytic(out: Path, step: float):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    summaries=[]
    axis_profiles={}
    plane_profiles={}
    geometry_rows=[]
    three_d_rows=[]
    slice_fields={}
    xfine=np.arange(-15, 15.0001, 0.25)
    yfine=np.arange(-12, 12.0001, 0.25)
    axialz=np.arange(8, 80.0001, 0.25)
    xcoarse=np.arange(-40, 40.0001, step)
    ycoarse=np.arange(-24, 24.0001, step)
    zcoarse=np.arange(8, 80.0001, step)

    for ncols in (7, 20):
        xy = element_centers(ncols)
        n = len(xy)
        coords, amps = circular_sources(xy)
        pitch=76.0/(ncols - 1)
        print(f'[N={n}] source cells (disk quadrature): {len(coords)}; X pitch={pitch:.3f} mm',flush=True)
        for eid,(xx,yy) in enumerate(xy):
            geometry_rows.append(dict(elements=n,element_id=eid,x_mm=float(xx),y_mm=float(yy)))

        f0 = abs(analytical_field(np.array([[0,0,FOCUS_MM]]), coords, amps))[0]
        axispts=np.column_stack((np.zeros_like(axialz),np.zeros_like(axialz),axialz))
        axial=np.abs(analytical_field(axispts,coords,amps))/f0
        roi=(axialz>=18)&(axialz<=60)
        peakz=float(axialz[np.where(roi)[0][np.argmax(axial[roi])]])
        axis_profiles[n]=(axialz,axial)

        # fine X/Y target-plane cuts for main-lobe FWHM
        xfpts=np.column_stack((xfine,np.zeros_like(xfine),np.full_like(xfine,FOCUS_MM)))
        yfpts=np.column_stack((np.zeros_like(yfine),yfine,np.full_like(yfine,FOCUS_MM)))
        pxfine=np.abs(analytical_field(xfpts,coords,amps))/f0
        pyfine=np.abs(analytical_field(yfpts,coords,amps))/f0
        wx=fwhm_around_target(xfine,pxfine)
        wy=fwhm_around_target(yfine,pyfine)
        plane_profiles[n]=(xfine,pxfine,yfine,pyfine)

        # 3D region evaluated at coarse resolution; cell counts depend on step.
        X,Y,Z=np.meshgrid(xcoarse,ycoarse,zcoarse,indexing='ij')
        targets=np.column_stack((X.ravel(),Y.ravel(),Z.ravel()))
        res=np.empty(len(targets),dtype=np.float32)
        for i in range(0,len(targets),2560):
            res[i:i+2560]=np.abs(analytical_field(targets[i:i+2560],coords,amps,chunk=96)/f0)
        volume=res.reshape(X.shape)
        slice_fields[n]=(volume[:, np.argmin(abs(ycoarse)), :].copy(),
                          volume[:, :, np.argmin(abs(zcoarse-FOCUS_MM))].copy())
        radius=np.sqrt(X**2+Y**2+(Z-FOCUS_MM)**2)
        off_mask=radius>=10.0
        p_off=np.where(off_mask,volume,-np.inf)
        pmaxidx=np.unravel_index(np.argmax(p_off),volume.shape)
        amp=float(p_off[pmaxidx])
        mx,my,mz=(float(X[pmaxidx]),float(Y[pmaxidx]),float(Z[pmaxidx]))
        gpidx=np.unravel_index(np.argmax(volume),volume.shape)
        gx,gy,gz=(float(X[gpidx]),float(Y[gpidx]),float(Z[gpidx]))
        pglobal=float(volume[gpidx])
        on_boundary=(abs(mx-xcoarse[0])<step/2 or abs(mx-xcoarse[-1])<step/2 or
                     abs(my-ycoarse[0])<step/2 or abs(my-ycoarse[-1])<step/2 or
                     abs(mz-zcoarse[0])<step/2 or abs(mz-zcoarse[-1])<step/2)
        summaries.append(dict(elements=n,columns=ncols,array_length_mm=ARRAY_MM,
                              band_width_mm=BAND_MM,element_diameter_mm=DIAM_MM,
                              pitch_x_mm=pitch,target_pressure_proxy=float(f0),
                              axial_peak_depth_mm=peakz,
                              peak_vs_target_db=float(20*np.log10(np.max(axial[roi]))),
                              x_fwhm_mm=float(wx),y_fwhm_mm=float(wy),
                              off_target_radius_mm=10.0,
                              off_target_3d_max_db=float(20*np.log10(amp)),
                              off_target_3d_max_x_mm=mx,off_target_3d_max_y_mm=my,
                              off_target_3d_max_z_mm=mz,
                              off_target_peak_on_domain_boundary=on_boundary,
                              global_3d_max_db=float(20*np.log10(pglobal)),
                              global_3d_max_x_mm=gx,global_3d_max_y_mm=gy,
                              global_3d_max_z_mm=gz,
                              roi_step_mm=step,
                              caution='coarse 3D disk quadrature proxy; off-target is outside 10mm sphere, NOT mainlobe segmentation or Watts'))
        for zz,pr in zip(axialz,axial):
            three_d_rows.append(dict(elements=n,z_mm=float(zz),normalized_pressure=float(pr),
                                     relative_db=float(20*np.log10(max(pr,1e-12)))))
        print(f'  N={n}: axial peak {peakz:.2f}mm; 3D outside 10mm max {20*np.log10(amp):+.2f} dB at ({mx:.1f},{my:.1f},{mz:.1f}) mm',flush=True)

    base=next(row['target_pressure_proxy'] for row in summaries if row['elements']==21)
    for row in summaries:
        row['target_gain_vs_N21_db']=float(20*np.log10(row['target_pressure_proxy']/base))
    write_csv(out/'analytic_3d_summary.csv',summaries)
    write_csv(out/'analytic_axial_profiles.csv',three_d_rows)
    write_csv(out/'element_centers.csv',geometry_rows)

    fig,axs=plt.subplots(1,2,figsize=(12,4.4))
    for n in (21,60):
        z,pp=axis_profiles[n]
        axs[0].plot(z,pp,label=f'N={n}')
        xx,px,yy,py=plane_profiles[n]
        axs[1].plot(xx,px,label=f'N={n}')
    axs[0].axvline(40,color='gray',ls='--');axs[0].set(xlabel='Depth Z (mm)',ylabel='Pressure / target',title='Axial pressure (analytic)')
    axs[1].set(xlabel='X at Z=40 mm (mm)',ylabel='Pressure / target',title='Lateral pressure (analytic)')
    for ax in axs: ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(out/'analytic_axial_and_lateral.png',dpi=160);plt.close(fig)

    fig,axs=plt.subplots(2,2,figsize=(12,9),sharex='row')
    last=None
    for j,n in enumerate((21,60)):
        xz,xy=slice_fields[n]
        last=axs[0,j].imshow(20*np.log10(np.maximum(xz.T,1e-9)),origin='lower',
            extent=(xcoarse[0],xcoarse[-1],zcoarse[0],zcoarse[-1]),
            aspect='auto',vmin=-30,vmax=3,cmap='inferno')
        axs[0,j].scatter([0],[FOCUS_MM],marker='x',c='cyan',s=40)
        axs[0,j].set(title=f'N={n}: XZ at Y=0',xlabel='X (mm)',ylabel='Depth Z (mm)')
        axs[1,j].imshow(20*np.log10(np.maximum(xy.T,1e-9)),origin='lower',
            extent=(xcoarse[0],xcoarse[-1],ycoarse[0],ycoarse[-1]),
            aspect='auto',vmin=-30,vmax=3,cmap='inferno')
        axs[1,j].scatter([0],[0],marker='x',c='cyan',s=40)
        axs[1,j].set(title=f'N={n}: XY at Z=40 mm',xlabel='X (mm)',ylabel='Y (mm)')
    fig.suptitle('Analytical 3D acoustic field slices (coarse sampling; not k-Wave)')
    fig.subplots_adjust(left=.07,right=.86,top=.91,bottom=.08,wspace=.25,hspace=.30)
    cax=fig.add_axes([.895,.19,.018,.63])
    fig.colorbar(last,cax=cax,label='Relative pressure (dB)')
    fig.savefig(out/'analytic_3d_slices.png',dpi=160);plt.close(fig)
    # The coarse 3D scan itself is intentionally not persisted to keep size small.
    print(f'[analytical] Saved {out}/analytic_3d_summary.csv',flush=True)
    return summaries


def load_csv(path: Path):
    with path.open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))


def existing_kwave_comparison(out: Path, a21: Path, a60: Path, summary21: Path, summary60: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not a21.exists() or not a60.exists():
        print('[existing k-Wave] Skipped: required axial.csv inputs were not found.',flush=True)
        print('  N21:',a21,flush=True)
        print('  N60:',a60,flush=True)
        return
    r21=load_csv(a21)
    r60=load_csv(a60)
    z21=np.array([float(r['z_mm']) for r in r21]); p21=np.array([float(r['relative_to_target']) for r in r21]);
    z60=np.array([float(r['z_mm']) for r in r60]); p60=np.array([float(r['normalized_pressure']) for r in r60]);
    z=np.arange(max(z21.min(),z60.min()),min(z21.max(),z60.max())+.00001,.5)
    interp21=np.interp(z,z21,p21)
    interp60=np.interp(z,z60,p60)
    rows=[dict(z_mm=float(zz),N21_relative_to_target=float(p),N60_relative_to_target=float(q),
               N21_relative_db=float(20*np.log10(max(p,1e-12))),
               N60_relative_db=float(20*np.log10(max(q,1e-12))))
          for zz,p,q in zip(z,interp21,interp60)]
    write_csv(out/'kwave_existing_axial_comparison.csv',rows)
    fig,ax=plt.subplots(figsize=(9,5))
    ax.plot(z,interp21,label='N=21, pitch 12.67 mm (existing k-Wave)')
    ax.plot(z,interp60,label='N=60, pitch 4 mm (existing k-Wave)')
    ax.axvline(40,ls='--',color='k',label='Requested 40 mm focus')
    ax.set(xlabel='Depth from source plane (mm)',ylabel='Pressure / pressure at 40 mm',
           title='Previously computed k-Wave axial profiles (same 80 mm aperture)')
    ax.grid(alpha=.3);ax.legend();fig.tight_layout()
    fig.savefig(out/'kwave_existing_axial_comparison.png',dpi=170);plt.close(fig)

    rr21=load_csv(summary21)[0] if summary21.exists() else None
    rr60=load_csv(summary60)[0] if summary60.exists() else None
    gain=float('nan')
    # IMPORTANT: old N21 run had 1/sqrt(21) per-element source amplitude.
    # Old N60 run did not have 1/sqrt(60), so the raw reported pressures
    # are NOT directly comparable. Reconstruct the same nominal 1/sqrt(N)
    # convention by dividing N60 raw target amplitude by sqrt(60).
    if rr21 is not None and rr60 is not None:
        q21=float(rr21['target_pressure_arbitrary'])
        q60=float(rr60['unscaled_target_pressure_arbitrary'])/math.sqrt(60.0)
        gain=20*math.log10(q60/q21)
    peak21=float(z[np.argmax(np.where((z>=18)&(z<=60),interp21,-np.inf))])
    peak60=float(z[np.argmax(np.where((z>=18)&(z<=60),interp60,-np.inf))])
    outputs=[dict(elements=21,peak_depth_mm=peak21,
                  pressure_at_target_arbitrary_before_crossrun_scaling=rr21['target_pressure_arbitrary'] if rr21 else '',
                  source_scaling_used='1/sqrt(N)',
                  note='Comparing relative axial shape only; 3D off-target not available from saved traces'),
             dict(elements=60,peak_depth_mm=peak60,
                  pressure_at_target_arbitrary_before_crossrun_scaling=rr60['unscaled_target_pressure_arbitrary'] if rr60 else '',
                  source_scaling_used='unity (per element) in original run',
                  note='Raw amplitude cannot directly be compared with N21; 3D off-target not available')]
    write_csv(out/'kwave_existing_run_notes.csv',outputs)
    print(f'[existing k-Wave] N21 peak {peak21:.1f} mm, N60 peak {peak60:.1f} mm.',flush=True)
    if math.isfinite(gain):
        print(f'[existing k-Wave] N60 vs N21 nominally rescaled target-pressure gain = {gain:+.2f} dB; proxy only, NOT radiated power.',flush=True)
        (out/'kwave_rescaled_gain_note.txt').write_text(
            'Original N21 drive: 10000/sqrt(21) per element.\n'
            'Original N60 drive: 10000 per element.\n'
            'To put them on the same artificial 10000/sqrt(N) convention, '
            'divide the old N60 pressure by sqrt(60).\n'
            f'Reconstructed N60 vs N21 target-gain proxy: {gain:+.3f} dB.\n'
            'This is NOT a comparison at equal measured acoustic power, '
            'equal total electrical power, or equal physical transducer efficiency.\n',
            encoding='utf-8')
    print('[existing k-Wave] Saved plot and CSV without rerunning wave solver.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['both','analytic','existing'],default='both')
    parser.add_argument('--out',default='element_count_comparison_80mm')
    parser.add_argument('--roi-step-mm',type=float,default=2.0,
                        help='3D off-target ROI sampling step. Default 2 mm (coarse).')
    parser.add_argument('--kwave21',default='aperture_length_results/kwave/fixed_count_L80/axial.csv')
    parser.add_argument('--kwave60',default='finite_transducer_results/disk_axial.csv')
    parser.add_argument('--kwave21-summary',default='aperture_length_results/kwave/fixed_count_L80/summary.csv')
    parser.add_argument('--kwave60-summary',default='finite_transducer_results/disk_summary.csv')
    args=parser.parse_args()
    if not 0.5 <= args.roi_step_mm <= 4.0:
        parser.error('Use --roi-step-mm between 0.5 and 4.0')
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):
        print(f'WARNING: {out} already contains files. This run WILL OVERWRITE same-named files there.',flush=True)
    out.mkdir(parents=True,exist_ok=True)
    if args.mode in ('analytic','both'):
        run_analytic(out,args.roi_step_mm)
    if args.mode in ('existing','both'):
        existing_kwave_comparison(out,Path(args.kwave21),Path(args.kwave60),
                                  Path(args.kwave21_summary),Path(args.kwave60_summary))
    (out/'study_config.json').write_text(json.dumps({
        'aperture_mm':ARRAY_MM,'band_mm':BAND_MM,'nominal_disk_diameter_mm':DIAM_MM,
        'element_counts':[21,60],'columns':[7,20],'rows':3,
        'pitch_mm':[76/6,4],'focus_mm':FOCUS_MM,
        'frequency_hz':F_HZ,'medium':'homogeneous water',
        'analytical_source':'disk subsource quadrature, 3 radial x 16 azimuth',
        'normalization':'equal sum of squared integrated element source amplitudes; NOT equal measured watts',
        'offtarget':'max amplitude >=10mm Euclidean distance from focus in -40..40 x -24..24 x 8..80mm ROI',
        'offtarget_3d_grid_step_mm':args.roi_step_mm,
        'existing_kwave':'already computed 1D axial traces; k-Wave 3D off-target was NOT computed',
        'limitations':'water, flat array, pressure sources, no skull, power, safety, piston velocity, thermal, array mutual coupling'
    },indent=2),encoding='utf-8')
    print('Done. Output:',out.resolve(),flush=True)


if __name__=='__main__':
    main()
