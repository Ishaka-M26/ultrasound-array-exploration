#!/usr/bin/env python3
"""Narrow-headband ultrasound: separate aperture-length and element-count effects.

Six configurations, one 2 mm disk source model, fixed 17 mm band width:
  fixed_count:  7 columns x 3 rows at each length (N=21), pitch changes
  fixed_pitch:  4 mm pitch and 3 rows, column count changes with length
  lengths:      30, 50, 80 mm

Analytical mode is a homogeneous-water Rayleigh-like uniformly weighted
circular-aperture superposition, not a verified baffled-piston model.
Optional k-Wave 3D mode uses rasterized additive pressure cells per disk,
not surface velocity or a physical acoustic power calibration.

First run:
  python aperture_length_study.py --mode analytic
Then select one case for an optional expensive numerical validation:
  python -u aperture_length_study.py --mode kwave --study fixed_count --length-mm 30
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

# All geometry values are millimeters unless specified otherwise.
FREQUENCY_HZ = 500_000.0
SOUND_SPEED_M_S = 1500.0
DENSITY_KG_M3 = 1000.0
FOCUS_Z_MM = 40.0
BAND_WIDTH_MM = 17.0
ELEMENT_DIAMETER_MM = 2.0
MIN_EDGE_MARGIN_MM = 1.0
ROWS = 3
MIN_PHYSICAL_CLEARANCE_MM = 0.0
PITCH_FIXED_MM = 4.0
FIXED_COLUMNS = 7
LENGTHS_MM = (30, 50, 80)
SOURCE_PRESSURE = 10_000.0  # Noncalibrated additive-pressure-source parameter.
K_RAD_MM = 2.0 * np.pi * FREQUENCY_HZ / (SOUND_SPEED_M_S * 1000.0)


def make_layout(study: str, length_mm: int):
    if study not in ('fixed_count', 'fixed_pitch') or length_mm not in LENGTHS_MM:
        raise ValueError(f'Invalid study/length: {study}, {length_mm}')
    usable_span = length_mm - 2 * (MIN_EDGE_MARGIN_MM + ELEMENT_DIAMETER_MM / 2)
    if study == 'fixed_count':
        cols = FIXED_COLUMNS
        pitch = usable_span / (cols - 1)
    else:
        cols = int(np.floor(usable_span / PITCH_FIXED_MM + 1e-9)) + 1
        pitch = PITCH_FIXED_MM
    xx = (np.arange(cols) - (cols - 1) / 2) * pitch
    outer_y = BAND_WIDTH_MM / 2 - MIN_EDGE_MARGIN_MM - ELEMENT_DIAMETER_MM / 2
    yy = np.array([-outer_y, 0.0, outer_y])
    xy = np.array([(x, y) for y in yy for x in xx], dtype=float)
    if not np.all(np.abs(xy[:, 0]) + ELEMENT_DIAMETER_MM / 2 + MIN_EDGE_MARGIN_MM <= length_mm / 2 + 1e-9):
        raise ValueError('Element outside X boundary')
    if not np.all(np.abs(xy[:, 1]) + ELEMENT_DIAMETER_MM / 2 + MIN_EDGE_MARGIN_MM <= BAND_WIDTH_MM / 2 + 1e-9):
        raise ValueError('Element outside Y boundary')
    dr = xy[:, None, :] - xy[None, :, :]
    sep = np.linalg.norm(dr, axis=2)
    np.fill_diagonal(sep, np.inf)
    clearance = float(sep.min() - ELEMENT_DIAMETER_MM)
    if clearance <= MIN_PHYSICAL_CLEARANCE_MM:
        raise ValueError(f'Disks intersect: {clearance} mm clearance')
    return dict(study=study, length_mm=length_mm, columns=cols, rows=ROWS,
                elements=len(xy), pitch_mm=float(pitch), xy=xy,
                min_clearance_mm=clearance,
                occupied_x_mm=float(np.ptp(xy[:, 0]) + ELEMENT_DIAMETER_MM),
                occupied_y_mm=float(np.ptp(xy[:, 1]) + ELEMENT_DIAMETER_MM))


def subsource_layout(layout):
    # Area-average over a disk: 3-point Gauss quadrature in radius, 16 azimuths.
    nodes, w = np.polynomial.legendre.leggauss(3)
    radii = (nodes + 1) / 2
    radial_weights = w / 2
    angles = np.arange(16) * (2 * np.pi / 16)
    offsets = []
    weights = []
    for r, wr in zip(radii, radial_weights):
        for theta in angles:
            offsets.append((r*np.cos(theta), r*np.sin(theta)))
            weights.append(2*r*wr/len(angles))
    offsets = np.asarray(offsets) * (ELEMENT_DIAMETER_MM / 2)
    weights = np.asarray(weights)
    assert np.isclose(weights.sum(), 1.0)

    xy = layout['xy']
    points = (xy[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
    # One phase per element, not an independently focused subsource.
    rf = np.sqrt(np.sum(xy**2, axis=1) + FOCUS_Z_MM**2)
    phase = np.exp(1j * K_RAD_MM * rf)
    amps = (phase[:, None] * weights[None, :] / np.sqrt(len(xy))).ravel()
    return points, amps


def analytical_field(layout, targets, chunk=192):
    """Analytical complex pressure proxy, equal sum-of-squared element drive.

    Not equal physical radiated power, and not a full piston velocity boundary.
    """
    points, amps = subsource_layout(layout)
    targets = np.asarray(targets, dtype=float).reshape(-1, 3)
    p = np.empty(len(targets), dtype=np.complex128)
    sx = points[:, 0]
    sy = points[:, 1]
    for start in range(0, len(targets), chunk):
        target = targets[start:start+chunk]
        R = np.sqrt((target[:, None, 0] - sx[None, :])**2 +
                    (target[:, None, 1] - sy[None, :])**2 +
                    target[:, None, 2]**2)
        p[start:start+chunk] = (np.exp(-1j*K_RAD_MM*R) / R) @ amps
    return p


def fwhm_axis(xs, vals, center_idx=None):
    """First -3dB half-power main-lobe width around the selected peak.

    Returns NaN if either boundary is outside the observed interval.
    """
    x = np.asarray(xs, dtype=float)
    p = np.asarray(vals, dtype=float)
    ci = int(np.argmax(p)) if center_idx is None else int(center_idx)
    level = float(p[ci]) / np.sqrt(2.0)
    i = ci
    while i > 0 and p[i-1] >= level:
        i -= 1
    if i == 0:
        return float('nan'), 'left_truncated'
    j = ci
    while j < len(x)-1 and p[j+1] >= level:
        j += 1
    if j == len(x)-1:
        return float('nan'), 'right_truncated'
    def cross(a, b):
        ya, yb = p[a], p[b]
        return float(x[a] + (level - ya) * (x[b] - x[a]) / (yb - ya)) if abs(yb-ya)>1e-15 else float(x[a])
    xl = cross(i-1, i)
    xr = cross(j, j+1)
    return xr-xl, 'ok'


def calc_analytic(layout):
    z = np.arange(8.0, 80.00001, 0.25)
    axis_pts = np.column_stack([np.zeros(len(z)), np.zeros(len(z)), z])
    pz = np.abs(analytical_field(layout, axis_pts))
    target_amp = float(np.abs(analytical_field(layout, [[0, 0, FOCUS_Z_MM]])[0]))
    roi = (z>=18.0) & (z<=60.0)
    indices = np.where(roi)[0]
    pkidx = int(indices[np.argmax(pz[roi])])
    axial_fwhm, axial_status = fwhm_axis(z, pz, pkidx)

    # Target plane: deliberately includes a broad area to detect off-axis peaks.
    x = np.arange(-42., 42.0001, 1.)
    y = np.arange(-25., 25.0001, 1.)
    X, Y = np.meshgrid(x, y, indexing='xy')
    targets = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, FOCUS_Z_MM)])
    pxy = np.abs(analytical_field(layout, targets)).reshape(X.shape)
    ix0, iy0 = int(np.argmin(abs(x))), int(np.argmin(abs(y)))
    xfwhm, xstatus = fwhm_axis(x, pxy[iy0], ix0)
    yfwhm, ystatus = fwhm_axis(y, pxy[:, ix0], iy0)
    mask_far = np.hypot(X, Y) >= 10.0
    offtarget = float(np.max(pxy[mask_far]))
    peak_iy, peak_ix = np.unravel_index(np.argmax(pxy), pxy.shape)
    # Disk source area is identical across conditions. This is an approximate
    # source-normalized target pressure proxy, not an actual power efficiency.
    row = dict(study=layout['study'], length_mm=layout['length_mm'],
               band_width_mm=BAND_WIDTH_MM, columns=layout['columns'],
               rows=ROWS, elements=layout['elements'], pitch_x_mm=layout['pitch_mm'],
               element_diameter_mm=ELEMENT_DIAMETER_MM,
               occupied_x_mm=layout['occupied_x_mm'],
               occupied_y_mm=layout['occupied_y_mm'],
               min_clearance_mm=layout['min_clearance_mm'],
               target_pressure_proxy=target_amp,
               axial_peak_depth_mm=float(z[pkidx]),
               axial_peak_vs_target_db=float(20*np.log10(pz[pkidx]/target_amp)),
               axial_fwhm_mm=axial_fwhm, axial_fwhm_status=axial_status,
               focalplane_x_fwhm_mm=xfwhm, focalplane_x_fwhm_status=xstatus,
               focalplane_y_fwhm_mm=yfwhm, focalplane_y_fwhm_status=ystatus,
               focalplane_offtarget_10mm_peak_db=float(20*np.log10(offtarget/target_amp)),
               focalplane_global_peak_x_mm=float(x[peak_ix]),
               focalplane_global_peak_y_mm=float(y[peak_iy]),
               focalplane_global_peak_vs_target_db=float(20*np.log10(pxy[peak_iy,peak_ix]/target_amp)),
               model='homogeneous_water_disk_surface_quadrature')
    return row, z, pz, (x, y, pxy)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def config_json(layout, args):
    return dict(study=layout['study'], length_mm=layout['length_mm'],
                columns=layout['columns'], elements=layout['elements'],
                pitch_x_mm=layout['pitch_mm'], band_width_mm=BAND_WIDTH_MM,
                diameter_mm=ELEMENT_DIAMETER_MM, focus_mm=FOCUS_Z_MM,
                frequency_hz=FREQUENCY_HZ, sound_speed_m_s=SOUND_SPEED_M_S,
                dx_mm=args.dx_mm, end_us=args.end_us,
                model_note='homogeneous water, additive pressure disk cell approximation',
                source_norm='1/sqrt(N) element integrated source sum; not equal measured watts')


def plot_analytic(out, rows, profiles, planes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Axial curve normalized to its own target; separate plot for raw proxy levels.
    fig, axes = plt.subplots(1, 2, figsize=(14,5))
    for (study, length), (z, p) in profiles.items():
        if study == 'fixed_count':
            axes[0].plot(z, p/p[np.argmin(abs(z-FOCUS_Z_MM))], label=f'{length} mm, N=21')
        else:
            axes[1].plot(z, p/p[np.argmin(abs(z-FOCUS_Z_MM))], label=f'{length} mm, N={next(r["elements"] for r in rows if r["study"]==study and r["length_mm"]==length)}')
    for i, name in enumerate(['Fixed element count', 'Fixed 4 mm pitch']):
        axes[i].axvline(FOCUS_Z_MM, ls='--',c='k',alpha=.8)
        axes[i].set(xlabel='Depth from array (mm)', ylabel='Pressure / target pressure', title=name)
        axes[i].grid(alpha=.3);axes[i].legend(fontsize=9)
    fig.suptitle('Axial pressure by headband array length (2 mm disks; analytic)')
    fig.tight_layout();fig.savefig(out/'axial_profiles.png', dpi=160);plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(13,5))
    for i, study in enumerate(['fixed_count','fixed_pitch']):
        sub = [r for r in rows if r['study']==study]
        lengths = [r['length_mm'] for r in sub]
        axes[i].plot(lengths, [r['target_pressure_gain_vs_baseline_db'] for r in sub], 'o-', label='Target pressure gain')
        axes[i].set(xlabel='Array length (mm)', ylabel='Relative target pressure (dB)',title=study)
        axes[i].grid(alpha=.3)
    fig.suptitle('Source-normalized focal pressure proxy (not electrical efficiency)')
    fig.tight_layout();fig.savefig(out/'target_gain.png',dpi=160);plt.close(fig)

    # Three 2D pressure maps from each study, normalized within each map at target.
    fig, axs = plt.subplots(2,3,figsize=(13,8),sharex=True,sharey=True)
    im = None
    for i, study in enumerate(['fixed_count','fixed_pitch']):
        for j, length in enumerate(LENGTHS_MM):
            xx, yy, p = planes[(study,length)]
            base = p[np.argmin(abs(yy)),np.argmin(abs(xx))]
            db = 20*np.log10(np.maximum(p/base,1e-6))
            im=axs[i,j].imshow(db,origin='lower',aspect='equal',
                               extent=(xx[0]-.5, xx[-1]+.5, yy[0]-.5,yy[-1]+.5),
                               cmap='inferno',vmin=-30,vmax=8)
            axs[i,j].scatter([0],[0],marker='x',c='cyan',s=25)
            n = next(r['elements'] for r in rows if r['study']==study and r['length_mm']==length)
            axs[i,j].set_title(f'{length} mm / N={n}')
            if j==0: axs[i,j].set_ylabel(study+'\nY (mm)')
            if i==1: axs[i,j].set_xlabel('X (mm)')
    fig.suptitle('Pressure distribution in the 40 mm focal plane (analytic)')
    fig.subplots_adjust(left=.095,right=.86,bottom=.09,top=.90,wspace=.16,hspace=.22)
    cax = fig.add_axes([.89,.16,.02,.67])
    fig.colorbar(im,cax=cax,label='Pressure relative to target (dB)')
    fig.savefig(out/'focal_plane_maps.png',dpi=160);plt.close(fig)


def run_analytic(out):
    rows=[]; profiles={}; planes={}; elements=[]; axialrows=[]
    for study in ('fixed_count','fixed_pitch'):
        for L in LENGTHS_MM:
            layout=make_layout(study,L)
            print(f'Analytical: {study}, {L}mm, N={layout["elements"]}, pitch={layout["pitch_mm"]:.3f}mm',flush=True)
            row,z,p,plane=calc_analytic(layout)
            rows.append(row); profiles[(study,L)]=(z,p); planes[(study,L)]=plane
            for eid, (x,y) in enumerate(layout['xy']):
                elements.append(dict(study=study,length_mm=L,element_id=eid,x_mm=x,y_mm=y))
            for zi,pi in zip(z,p):
                axialrows.append(dict(study=study,length_mm=L,z_mm=float(zi),
                                      pressure_proxy=float(pi),relative_to_focus=float(pi/row['target_pressure_proxy'])))
    reference = next(r['target_pressure_proxy'] for r in rows if r['study']=='fixed_count' and r['length_mm']==30)
    for row in rows:
        row['target_pressure_gain_vs_baseline_db'] = float(20*np.log10(row['target_pressure_proxy']/reference))
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out/'geometry_sweep_summary.csv',rows)
    write_csv(out/'geometry_sweep_axial_profiles.csv',axialrows)
    write_csv(out/'array_elements.csv',elements)
    plot_analytic(out,rows,profiles,planes)
    print('\nSummary:',flush=True)
    for r in rows:
        print(f'  {r["study"]:11} {r["length_mm"]:2}mm N={r["elements"]:2} '
              f'pitch={r["pitch_x_mm"]:5.2f} peak_z={r["axial_peak_depth_mm"]:5.2f} '
              f'gain={r["target_pressure_gain_vs_baseline_db"]:+6.2f}dB '
              f'offtarget={r["focalplane_offtarget_10mm_peak_db"]:+6.2f}dB',flush=True)
    print('Saved:',out.resolve(),flush=True)


def grid_geometry(layout, dx):
    # Same spatial extents across all configurations to aid direct comparison.
    nx = 2*int(np.ceil(45.0/dx))+1
    ny = 2*int(np.ceil(15.5/dx))+1
    nz = int(np.ceil(94.0/dx))+1
    cx, cy = nx//2, ny//2
    iz = int(round(6.0/dx))
    shape = (nx,ny,nz)
    ix = np.rint(layout['xy'][:,0]/dx).astype(int)+cx
    iy = np.rint(layout['xy'][:,1]/dx).astype(int)+cy
    snapped = np.column_stack(((ix-cx)*dx,(iy-cy)*dx))
    err = float(np.max(np.linalg.norm(snapped-layout['xy'],axis=1)))
    diskmap={};rad=ELEMENT_DIAMETER_MM/2
    radcells=int(np.ceil(rad/dx))
    for eid,(xc,yc) in enumerate(zip(ix,iy)):
        for x in range(xc-radcells,xc+radcells+1):
            for y in range(yc-radcells,yc+radcells+1):
                if ((x-xc)*dx)**2+((y-yc)*dx)**2<=rad**2+1e-9:
                    key=(int(x),int(y),iz)
                    if key in diskmap: raise ValueError('Rasterized source footprints overlap')
                    diskmap[key]=eid
    if not diskmap: raise ValueError('Empty source map')
    if any(not(0<=x<nx and 0<=y<ny and 0<=z<nz) for x,y,z in diskmap):
        raise ValueError('Source outside grid')
    counts=np.bincount(list(diskmap.values()),minlength=layout['elements'])
    if np.any(counts==0): raise ValueError('Some elements have no source cells')
    return dict(shape=shape, center=(cx,cy), source_z=iz, snapped=snapped, source_map=diskmap,
                counts=counts, dx=dx, max_raster_error_mm=err)


def run_kwave(layout, args, out):
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksource import kSource
    from kwave.ksensor import kSensor
    from kwave.kspaceFirstOrder import kspaceFirstOrder

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    geo=grid_geometry(layout,args.dx_mm)
    dx=geo['dx']
    nx,ny,nz=geo['shape'];cx,cy=geo['center']; iz=geo['source_z']
    print('Grid:',geo['shape'],'voxels:',nx*ny*nz,'raster error:',geo['max_raster_error_mm'],'mm',flush=True)
    grid = kWaveGrid([nx,ny,nz], [dx*1e-3]*3)
    grid.makeTime(SOUND_SPEED_M_S,cfl=0.3,t_end=args.end_us*1e-6)
    Nt = int(grid.Nt)
    dt = float(grid.dt)
    t=np.arange(Nt)*dt
    srcmap=geo['source_map']
    srcmask=np.zeros(geo['shape'],dtype=bool)
    for k in srcmap: srcmask[k]=True
    src_coords=np.argwhere(srcmask)
    ids=np.array([srcmap[tuple(c)] for c in src_coords],dtype=int)
    # Every source voxel belonging to one disk uses the SAME per-element phase.
    rf=np.sqrt(np.sum(geo['snapped']**2,axis=1)+FOCUS_Z_MM**2)*1e-3
    phase=2*np.pi*FREQUENCY_HZ*rf/SOUND_SPEED_M_S
    n_per_element=geo['counts'][ids]
    amplitude=SOURCE_PRESSURE/(np.sqrt(layout['elements'])*n_per_element)
    ramp_t=8.0/FREQUENCY_HZ
    ramp=np.sin(0.5*np.pi*np.minimum(t/ramp_t,1.))**2
    signals = amplitude[:,None]*ramp[None,:]*np.sin(2*np.pi*FREQUENCY_HZ*t[None,:]+phase[ids,None])
    source=kSource();source.p_mask=srcmask;source.p=signals;source.p_mode='additive'

    sx=np.arange(cx-int(np.floor(30/dx)),cx+int(np.floor(30/dx))+1)
    stride=max(1,int(round(1.0/dx)))
    sx=sx[((sx-cx)%stride)==0]
    zs=np.arange(nz)
    zs=zs[((zs-iz)*dx>=8) & ((zs-iz)*dx<=80)]
    sensor_mask=np.zeros(geo['shape'],dtype=bool)
    sensor_mask[np.ix_(sx,[cy],zs)]=True
    sensor_coords=np.argwhere(sensor_mask)
    sensor=kSensor(mask=sensor_mask,record=['p'])
    print(f'Sources: {int(srcmask.sum())} voxels; sensors: {len(sensor_coords)}; steps: {Nt}',flush=True)
    data=kspaceFirstOrder(grid,kWaveMedium(sound_speed=SOUND_SPEED_M_S,density=DENSITY_KG_M3),
                          source,sensor,backend='python',device='cpu',pml_size=10,
                          pml_inside=False,quiet=args.quiet)
    p=np.asarray(data['p'])
    if p.shape==(Nt,len(sensor_coords)): p=p.T
    if p.shape!=(len(sensor_coords),Nt):
        raise RuntimeError(f'Unexpected k-Wave pressure shape {p.shape}')
    end=t[-1];window=5.0/FREQUENCY_HZ
    def amplitude_estimate(start,stop):
        use=(t>=start)&(t<stop)
        ts=t[use]
        w=np.hanning(len(ts))
        demod=w*np.exp(-2j*np.pi*FREQUENCY_HZ*ts)
        return 2.0/w.sum()*np.abs(p[:,use]@demod)
    recent=amplitude_estimate(end-window,end+dt/2)
    prior=amplitude_estimate(end-2*window,end-window)
    changes=np.abs(recent-prior)/np.maximum(recent,1e-20)
    ix_lookup={v:i for i,v in enumerate(sx)}
    iz_lookup={v:i for i,v in enumerate(zs)}
    plane=np.full((len(sx),len(zs)),np.nan)
    stability=np.full_like(plane,np.nan)
    for n,(a,b,c) in enumerate(sensor_coords):
        plane[ix_lookup[a],iz_lookup[c]]=recent[n]
        stability[ix_lookup[a],iz_lookup[c]]=changes[n]
    if np.isnan(plane).any(): raise RuntimeError('C-order sensor mapping failed')
    z=(zs-iz)*dx
    axis=plane[ix_lookup[cx]]
    focusi=np.argmin(np.abs(z-FOCUS_Z_MM))
    focval=float(axis[focusi])
    if focval<=0: raise RuntimeError('Zero pressure at requested focus')
    roi=(z>=18)&(z<=60)
    pi=int(np.where(roi)[0][np.argmax(axis[roi])])
    axial_fwhm, axial_status=fwhm_axis(z,axis,pi)
    xs=(sx-cx)*dx
    x_fwhm,x_status=fwhm_axis(xs,plane[:,focusi],int(np.argmin(abs(xs))))
    win=(z>=25)&(z<=55)
    steady=float(np.median(stability[ix_lookup[cx],win]))
    res=dict(study=layout['study'],length_mm=layout['length_mm'],
             elements=layout['elements'],columns=layout['columns'],pitch_x_mm=layout['pitch_mm'],
             dx_mm=dx,source_voxels=int(srcmask.sum()),target_pressure_arbitrary=focval,
             axial_peak_depth_mm=float(z[pi]),
             peak_vs_target_db=float(20*np.log10(axis[pi]/focval)),
             axial_fwhm_mm=axial_fwhm,axial_fwhm_status=axial_status,
             x_fwhm_at_focus_mm=x_fwhm,x_fwhm_status=x_status,
             median_late_window_change=steady, max_raster_error_mm=geo['max_raster_error_mm'])
    case_dir=out/'kwave'/f'{layout["study"]}_L{layout["length_mm"]}'
    case_dir.mkdir(parents=True,exist_ok=True)
    write_csv(case_dir/'summary.csv',[res])
    write_csv(case_dir/'axial.csv',[dict(z_mm=float(zz),pressure_arbitrary=float(pp),relative_to_target=float(pp/focval)) for zz,pp in zip(z,axis)])
    (case_dir/'config.json').write_text(json.dumps(config_json(layout,args),indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    axes[0].plot(z,axis/focval)
    axes[0].axvline(FOCUS_Z_MM,ls='--',color='k')
    axes[0].axvline(z[pi],ls=':',color='k')
    axes[0].set(xlabel='Depth (mm)',ylabel='Pressure / target',title='Axial pressure')
    axes[0].grid(alpha=.3)
    img=axes[1].pcolormesh(xs,z,20*np.log10(np.maximum(plane.T/focval,1e-6)),
                           shading='auto',cmap='inferno',vmin=-30,vmax=6)
    axes[1].scatter([0],[FOCUS_Z_MM],marker='x',c='cyan',s=40)
    axes[1].set(xlabel='X (mm)',ylabel='Depth (mm)',title='XZ pressure map (Y=0)')
    fig.colorbar(img,ax=axes[1],label='Relative pressure (dB)')
    fig.tight_layout();fig.savefig(case_dir/'axial_and_xz.png',dpi=160);plt.close(fig)
    print('Saved k-Wave result:',case_dir.resolve(),flush=True)
    print('Peak:',res['axial_peak_depth_mm'],'mm; target pressure proxy:',focval,
          'late-window relative change:',steady,flush=True)
    if steady>0.05:
        print('WARNING: >5% late-window difference. Increase --end-us.',flush=True)
    # Aggregate only available cases; never pretend the other cases were run.
    rows=[]
    for study in ('fixed_count','fixed_pitch'):
        for length in LENGTHS_MM:
            f=out/'kwave'/f'{study}_L{length}'/'summary.csv'
            if f.exists():
                with f.open(newline='',encoding='utf-8') as stream:
                    rows.extend(csv.DictReader(stream))
    write_csv(out/'kwave_completed_summary.csv',rows)


def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--mode',choices=('analytic','kwave'),default='analytic')
    pa.add_argument('--study',choices=('fixed_count','fixed_pitch'),default='fixed_count')
    pa.add_argument('--length-mm',type=int,choices=LENGTHS_MM,default=30)
    pa.add_argument('--dx-mm',type=float,default=.5)
    pa.add_argument('--end-us',type=float,default=85.)
    pa.add_argument('--out',default='aperture_length_results')
    pa.add_argument('--dry-run',action='store_true')
    pa.add_argument('--quiet',action='store_true')
    args=pa.parse_args()
    if not .35<=args.dx_mm<=.6: pa.error('Use --dx-mm from 0.35 to 0.6')
    if args.end_us<80: pa.error('--end-us must be at least 80 us for 80 mm recording')
    out=Path(args.out)
    if args.dry_run:
        for study in ('fixed_count','fixed_pitch'):
            for L in LENGTHS_MM:
                a=make_layout(study,L)
                print(f'{study}: {L}mm, N={a["elements"]}, columns={a["columns"]}, pitch={a["pitch_mm"]:.3f}mm, clearance={a["min_clearance_mm"]:.3f}mm')
        chosen=make_layout(args.study,args.length_mm)
        geo=grid_geometry(chosen,args.dx_mm)
        print('Selected grid:',geo['shape'],'voxels:',np.prod(geo['shape']),
              'source cells:',len(geo['source_map']),
              'max center raster error:',geo['max_raster_error_mm'],'mm')
        print('DRY RUN: no propagation performed')
        return
    if args.mode=='analytic':
        run_analytic(out)
    else:
        run_kwave(make_layout(args.study,args.length_mm),args,out)


if __name__ == '__main__':
    main()
