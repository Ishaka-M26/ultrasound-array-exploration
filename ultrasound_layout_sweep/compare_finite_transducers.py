#!/usr/bin/env python3
"""Finite circular source vs grid-point source, k-Wave-python 0.6.2.

Same 60 centers (20 columns x 3 rows), same per-element drive phases,
same source-strength sum per element. Unlike the earlier V3.1 study,
both models use one fixed 80 x 17 mm geometry. The 'disk' case uses a
rasterized 2 mm diameter surface of additive pressure sources, NOT a
physical baffled piston velocity boundary condition.

Commands:
  python compare_finite_transducers.py --dry-run
  python compare_finite_transducers.py --mode analytic
  python -u compare_finite_transducers.py --mode disk
  python -u compare_finite_transducers.py --mode point

The last two runs can be performed in either order. A joint comparison is
created automatically when both results exist.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

FREQUENCY = 500_000.0
SPEED = 1500.0
DENSITY = 1000.0
FOCUS_MM = 40.0
BAND_MM = 17.0
MODULE_LENGTH_MM = 80.0
DIAMETER_MM = 2.0
N_COLUMNS = 20
N_ROWS = 3
PITCH_MM = 4.0
EDGE_MARGIN_MM = 1.0
SOURCE_PRESSURE = 10_000.0  # arbitrary additive-source parameter, not a transducer rating


def centers_mm():
    x = (np.arange(N_COLUMNS, dtype=float) - (N_COLUMNS - 1) / 2) * PITCH_MM
    y = np.array([-BAND_MM / 2 + EDGE_MARGIN_MM + DIAMETER_MM / 2,
                  0.0,
                  BAND_MM / 2 - EDGE_MARGIN_MM - DIAMETER_MM / 2])
    return np.array([(xx, yy) for yy in y for xx in x], dtype=float)


def validate_geometry(xy):
    assert len(xy) == N_COLUMNS * N_ROWS == 60
    assert np.ptp(xy[:, 0]) + DIAMETER_MM <= MODULE_LENGTH_MM + 1e-8
    assert np.ptp(xy[:, 1]) + DIAMETER_MM <= BAND_MM - 2 * EDGE_MARGIN_MM + 1e-8
    dr = xy[:, None, :] - xy[None, :, :]
    d = np.linalg.norm(dr, axis=-1) + np.eye(len(xy)) * 1e6
    min_clearance = d.min() - DIAMETER_MM
    assert min_clearance > 0, 'Two physical transducers overlap'
    return float(min_clearance)


def build_geometry(dx_mm):
    # Domain: X = +/-45mm, Y = +/-15.5mm; sources 6mm inside Z domain.
    nx = 2 * int(np.ceil(45.0 / dx_mm)) + 1
    ny = 2 * int(np.ceil(15.5 / dx_mm)) + 1
    nz = int(np.ceil(94.0 / dx_mm)) + 1
    cx, cy = nx // 2, ny // 2
    iz = int(round(6.0 / dx_mm))
    shape = (nx, ny, nz)
    xy = centers_mm()
    min_clearance = validate_geometry(xy)
    ix = np.rint(xy[:, 0] / dx_mm).astype(int) + cx
    iy = np.rint(xy[:, 1] / dx_mm).astype(int) + cy
    snapped = np.column_stack(((ix - cx) * dx_mm, (iy - cy) * dx_mm))
    center_error = np.linalg.norm(snapped - xy, axis=1)
    # Preserve each element identity; all cells in a disk share the phase
    # determined by its CENTER, rather than receiving individual delays.
    element_maps = {'point': {}, 'disk': {}}
    rr = DIAMETER_MM / 2
    r_cells = int(np.ceil(rr / dx_mm))
    for eid, (a, b) in enumerate(zip(ix, iy)):
        element_maps['point'][(int(a), int(b), iz)] = eid
        for i in range(a-r_cells, a+r_cells+1):
            for j in range(b-r_cells, b+r_cells+1):
                if (i-a)**2 * dx_mm**2 + (j-b)**2 * dx_mm**2 <= rr**2 + 1e-10:
                    key = (int(i), int(j), iz)
                    if key in element_maps['disk']:
                        raise RuntimeError('Rasterized disks overlap: reduce diameter or change geometry')
                    element_maps['disk'][key] = eid
    for mode, assignments in element_maps.items():
        for (i, j, k), eid in assignments.items():
            if not 0 <= i < nx or not 0 <= j < ny or not 0 <= k < nz:
                raise RuntimeError(f'{mode} source outside domain')
        counts = np.bincount(list(assignments.values()), minlength=len(xy))
        if not np.all(counts > 0):
            raise RuntimeError(f'{mode}: some elements have no voxels')
    return dict(shape=shape, center=(cx,cy), source_z=iz, xy=xy,
                snapped_xy=snapped, max_raster_error_mm=float(center_error.max()),
                source_maps=element_maps, min_clearance_mm=min_clearance,
                dx_mm=dx_mm)


def sensor_geometry(geo):
    dx = geo['dx_mm']
    nx, ny, nz = geo['shape']
    cx,cy = geo['center']
    stride = max(1, int(round(1.0/dx)))
    xs = np.arange(cx-int(np.floor(13/dx)), cx+int(np.floor(13/dx))+1)
    xs = xs[((xs-cx) % stride) == 0]
    zz = np.arange(nz)
    depths = (zz - geo['source_z'])*dx
    zz = zz[(depths>=8) & (depths<=80)]
    mask = np.zeros(geo['shape'],dtype=bool)
    mask[np.ix_(xs, [cy], zz)] = True
    # np.argwhere is C-order. This matters: a prior experiment's F-order
    # mismatch produced spurious pressure spikes and an invalid focal depth.
    coords = np.argwhere(mask)
    actual_idx = np.ravel_multi_index(coords.T, geo['shape'], order='C')
    if not np.array_equal(np.flatnonzero(mask.ravel(order='C')), actual_idx):
        raise AssertionError('Sensor mask order is not C order')
    return mask, coords, xs, zz


def make_source_mask(geo, mode, Nt, times):
    src = geo['source_maps'][mode]
    shape = geo['shape']
    mask = np.zeros(shape, dtype=bool)
    for key in src:
        mask[key] = True
    coords = np.argwhere(mask)
    # Center phase, using actual rasterized element centers; sign chosen to
    # match the already validated old k-Wave point-source experiment.
    d = np.sqrt((geo['snapped_xy']**2).sum(axis=1) + FOCUS_MM**2) * 1e-3
    phases = 2*np.pi*FREQUENCY * d / SPEED
    count = np.bincount(list(src.values()), minlength=len(geo['xy']))
    ids = np.array([src[tuple(c)] for c in coords], dtype=int)
    amps = SOURCE_PRESSURE / count[ids]
    ramp_t = 8.0 / FREQUENCY
    ramp = np.sin(0.5*np.pi * np.minimum(times/ramp_t, 1.0))**2
    signals = (amps[:, None]*ramp[None,:] *
               np.sin(2*np.pi*FREQUENCY*times[None,:] + phases[ids,None]))
    return mask, signals, count


def tone_amplitude(traces, times, start, end):
    sel = (times >= start) & (times < end)
    ts = times[sel]
    if len(ts) < 20:
        raise RuntimeError('Too few samples in steady-state window')
    window = np.hanning(len(ts))
    demod = window * np.exp(-2j*np.pi*FREQUENCY*ts)
    return 2.0 / window.sum() * np.abs(traces[:,sel] @ demod)


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def surface_field_profiles(geo, depths):
    """Fast *analytical* comparison: monopole kernel integrated over disks.

    Quadrature is independent of rasterization and the wave solver. This is
    a free-field surface-source approximation, NOT a calibrated piston model.
    Equal element-integrated source strengths, no skull or backing structure.
    """
    xy = geo['xy']
    z = np.asarray(depths)
    k = 2*np.pi*FREQUENCY/(SPEED*1000.)
    distances = np.linalg.norm(xy, axis=1)**2 + FOCUS_MM**2
    phase = np.exp(1j*k*np.sqrt(distances))
    ppoint = np.zeros(len(z), dtype=complex)
    for center, ph in zip(xy, phase):
        r = np.sqrt(center[0]**2 + center[1]**2 + z**2)
        ppoint += ph * np.exp(-1j*k*r) / r
    # Uniform area integration: 4 radial Gauss points x 20 azimuth points.
    rr, ww = np.polynomial.legendre.leggauss(4)
    rr = (rr+1)/2
    ww = ww / 2
    phi = np.arange(20)*2*np.pi/20
    radius = DIAMETER_MM/2
    offsets=[]
    weights=[]
    for q, w in zip(rr,ww):
        for theta in phi:
            offsets.append([radius*q*np.cos(theta), radius*q*np.sin(theta)])
            # uniform disk area average: integral 2r dr dphi / 2pi
            weights.append(2*q*w/len(phi))
    offsets=np.asarray(offsets)
    weights=np.asarray(weights)
    assert abs(weights.sum()-1) < 1e-12
    pdisk=np.zeros(len(z),dtype=complex)
    for center, ph in zip(xy,phase):
        pxy=center[None,:]+offsets
        R=np.sqrt((pxy[:,0,None]**2 + pxy[:,1,None]**2) + z[None,:]**2)
        pdisk += ph * (weights[:,None]*np.exp(-1j*k*R)/R).sum(axis=0)
    return np.abs(ppoint),np.abs(pdisk)


def load_profile(out, mode):
    f=out/f'{mode}_axial.csv'
    if not f.exists():
        return None
    data = np.genfromtxt(f, delimiter=',', names=True)
    return np.atleast_1d(data)


def compare_outputs(out, analytic_only=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    if analytic_only:
        geo=build_geometry(0.5)
        z=np.arange(8.,80.01,.25)
        point,disk=surface_field_profiles(geo,z)
        i=np.argmin(abs(z-FOCUS_MM))
        point/=point[i]
        disk/=disk[i]
        write_csv(out/'analytic_surface_comparison.csv', [
            {'z_mm':float(zi),'point_relative_to_40mm':float(a),
             'disk_relative_to_40mm':float(b)} for zi,a,b in zip(z,point,disk)
        ])
        fig,ax=plt.subplots(figsize=(9,5))
        ax.plot(z,point,label='Analytical point sources')
        ax.plot(z,disk,label='Analytical disk surface quadrature')
        ax.axvline(FOCUS_MM,ls='--',c='k',label='Target 40 mm')
        ax.set(xlabel='Z from source (mm)', ylabel='Pressure amplitude / at 40 mm',
               title='Preliminary surface-integral comparison (not k-Wave)')
        ax.grid(alpha=.3);ax.legend();fig.tight_layout()
        fig.savefig(out/'analytic_surface_comparison.png',dpi=170)
        plt.close(fig)
        print('Analytic point peak:',float(z[np.argmax(point)]),'mm')
        print('Analytic disk peak:',float(z[np.argmax(disk)]),'mm')
        print('Saved:',out/'analytic_surface_comparison.csv')
        return
    a=load_profile(out,'point')
    b=load_profile(out,'disk')
    if a is None or b is None:
        return
    if len(a) != len(b) or not np.allclose(a['z_mm'],b['z_mm']):
        raise RuntimeError('Point and disk profiles have incompatible depth sampling')
    z=a['z_mm']
    pa=a['normalized_pressure']
    pb=b['normalized_pressure']
    rows=[{'z_mm':float(zi),'point_normalized_pressure':float(p),
           'disk_normalized_pressure':float(q),'point_relative_db':float(20*np.log10(max(p,1e-12))),
           'disk_relative_db':float(20*np.log10(max(q,1e-12)))} for zi,p,q in zip(z,pa,pb)]
    write_csv(out/'point_vs_disk_axial.csv',rows)
    fig,ax=plt.subplots(figsize=(9,5))
    ax.plot(z,pa,label='k-Wave point-like sources')
    ax.plot(z,pb,label='k-Wave 2 mm disk footprints')
    ax.axvline(FOCUS_MM,ls='--',c='k',label='Target focus')
    ax.set(xlabel='Depth from source plane (mm)',ylabel='Pressure / pressure at 40 mm',
           title='Same 60 centers: point vs finite disk source footprints')
    ax.grid(alpha=.3);ax.legend();fig.tight_layout()
    fig.savefig(out/'point_vs_disk_axial.png',dpi=170);plt.close(fig)
    roi=(z>=18)&(z<=60)
    local_point=np.where(roi)[0][int(np.argmax(pa[roi]))]
    local_disk=np.where(roi)[0][int(np.argmax(pb[roi]))]
    write_csv(out/'combined_summary.csv',[
        {'comparison':'point','local_peak_18_to_60_mm':float(z[local_point]),
         'local_peak_vs_40mm_db':float(20*np.log10(pa[local_point]))},
        {'comparison':'disk','local_peak_18_to_60_mm':float(z[local_disk]),
         'local_peak_vs_40mm_db':float(20*np.log10(pb[local_disk]))}
    ])
    print('Generated joint comparison: point_vs_disk_axial.png, point_vs_disk_axial.csv, combined_summary.csv')


def run_kwave(mode, geo, out, end_us, quiet):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksource import kSource
    from kwave.ksensor import kSensor
    from kwave.kspaceFirstOrder import kspaceFirstOrder

    nx,ny,nz=geo['shape']
    dx_m=geo['dx_mm']*1e-3
    grid=kWaveGrid([nx,ny,nz],[dx_m]*3)
    grid.makeTime(SPEED,cfl=0.3,t_end=end_us*1e-6)
    Nt=int(grid.Nt);dt=float(grid.dt)
    times=np.arange(Nt)*dt
    mask,signals,counts=make_source_mask(geo,mode,Nt,times)
    s_mask,coords,xs,zs=sensor_geometry(geo)
    print(f'[{mode}] {len(geo["xy"])} elements, {int(mask.sum())} source voxels, '
          f'{int(s_mask.sum())} sensors, {Nt} time steps',flush=True)
    source=kSource()
    source.p_mask=mask
    source.p=signals
    source.p_mode='additive'
    sensor=kSensor(mask=s_mask,record=['p'])
    medium=kWaveMedium(sound_speed=SPEED,density=DENSITY)
    data=kspaceFirstOrder(grid,medium,source,sensor,
                          backend='python',device='cpu',pml_size=10,
                          pml_inside=False,quiet=quiet)
    p=np.asarray(data['p'])
    if p.shape == (Nt,len(coords)):
        p=p.T
    if p.shape != (len(coords),Nt):
        raise RuntimeError(f'Bad pressure output shape {p.shape}, expected {(len(coords),Nt)}')
    end=times[-1]
    window=5./FREQUENCY
    recent=tone_amplitude(p,times,end-window,end+dt/2)
    prior=tone_amplitude(p,times,end-2*window,end-window)
    change=np.abs(recent-prior)/np.maximum(recent,1e-20)
    xlookup={v:i for i,v in enumerate(xs)}
    zlookup={v:i for i,v in enumerate(zs)}
    plane=np.full((len(xs),len(zs)),np.nan)
    stability=np.full_like(plane,np.nan)
    for n,(ix,iy,iz) in enumerate(coords):
        plane[xlookup[ix],zlookup[iz]]=recent[n]
        stability[xlookup[ix],zlookup[iz]]=change[n]
    if np.isnan(plane).any():
        raise RuntimeError('Failed mapping C-order traces to sensor positions')
    z_mm=(zs-geo['source_z'])*geo['dx_mm']
    axis=plane[xlookup[geo['center'][0]]]
    target_i=int(np.argmin(abs(z_mm-FOCUS_MM)))
    relative=axis/axis[target_i]
    roi=(z_mm>=18)&(z_mm<=60)
    peak_i=np.where(roi)[0][np.argmax(relative[roi])]
    stable_roi=(z_mm>=25)&(z_mm<=55)
    stable_med=float(np.median(stability[xlookup[geo['center'][0]],stable_roi]))
    print(f'[{mode}] local axial peak: {z_mm[peak_i]:.2f} mm; '
          f'peak vs 40 mm: {20*np.log10(relative[peak_i]):+.2f} dB; '
          f'median window change {stable_med*100:.2f}%',flush=True)
    if stable_med>0.05:
        print(f'[{mode}] WARNING: >5% steady-state-window change. Increase --end-us.',flush=True)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/f'{mode}_axial.csv',[
        {'z_mm':float(z),'normalized_pressure':float(pr),
         'relative_db':float(20*np.log10(max(pr,1e-12)))}
        for z,pr in zip(z_mm,relative)
    ])
    global_i=int(np.argmax(relative))
    write_csv(out/f'{mode}_summary.csv', [{
        'model':mode,
        'source_voxels':int(mask.sum()),
        'elements':len(geo['xy']),
        'dx_mm':geo['dx_mm'],
        'local_peak_depth_mm':float(z_mm[peak_i]),
        'global_axial_peak_depth_mm':float(z_mm[global_i]),
        'target_depth_mm':FOCUS_MM,
        'peak_vs_target_db':float(20*np.log10(relative[peak_i])),
        'median_late_window_change':stable_med,
        'unscaled_target_pressure_arbitrary':float(axis[target_i]),
        'min_cells_per_element':int(counts.min()),
        'max_cells_per_element':int(counts.max()),
    }])
    xx=(xs-geo['center'][0])*geo['dx_mm']
    fig,ax=plt.subplots(figsize=(9,5))
    image=ax.pcolormesh(xx,z_mm,20*np.log10(np.maximum(plane.T/axis[target_i],1e-6)),
                       shading='auto',cmap='inferno',vmin=-30,vmax=3)
    ax.scatter([0],[FOCUS_MM],c='cyan',marker='x',s=70,label='Requested focus')
    ax.scatter([0],[z_mm[peak_i]],edgecolors='white',facecolors='none',s=60,label='Axial peak')
    ax.legend(fontsize=9)
    ax.set(xlabel='X (mm)',ylabel='Depth (mm)',title=f'k-Wave {mode} XZ slice (Y=0)')
    fig.colorbar(image,ax=ax,label='dB relative to target axial pressure')
    fig.tight_layout();fig.savefig(out/f'{mode}_xz.png',dpi=170);plt.close(fig)
    print(f'[{mode}] Saved {out.resolve()}',flush=True)


def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--mode',choices=['analytic','point','disk','both'],default='analytic')
    pa.add_argument('--dx-mm',type=float,default=0.5)
    pa.add_argument('--end-us',type=float,default=85.)
    pa.add_argument('--out',default='finite_transducer_results')
    pa.add_argument('--dry-run',action='store_true')
    pa.add_argument('--quiet',action='store_true')
    args=pa.parse_args()
    if not 0.3<=args.dx_mm<=0.6:
        pa.error('--dx-mm must be 0.3 to 0.6 mm')
    if args.end_us<80:
        pa.error('--end-us must be >=80 microseconds for 80mm depth observation')
    out=Path(args.out)
    geo=build_geometry(args.dx_mm)
    snum={mode:len(mp) for mode,mp in geo['source_maps'].items()}
    print('60 elements, 20 columns x 3 rows, center pitch 4mm along X',flush=True)
    print('80 x 17 mm module, 2mm circular source footprints, 500 kHz, focus 40mm',flush=True)
    print('Grid:',geo['shape'],'voxels:',int(np.prod(geo['shape'])),flush=True)
    print('Source cells:',snum,'; minimum element clearance:',geo['min_clearance_mm'],'mm',flush=True)
    print('Largest center raster error:',geo['max_raster_error_mm'],'mm',flush=True)
    print('Comparison normalization: equal SUM OF ADDITIVE SOURCE STRENGTH per element,',flush=True)
    print('NOT equal radiated watts or a calibrated surface-velocity model.',flush=True)
    for mode, mp in geo['source_maps'].items():
        cnt=np.bincount(list(mp.values()),minlength=len(geo['xy']))
        print(f'  {mode}: {cnt.min()}..{cnt.max()} grid source cells per element',flush=True)
    print('WARNING: finite disk is a rasterized additive pressure source, not a physical baffled piston.',flush=True)
    if args.dry_run:
        sensor_mask,_,_,_=sensor_geometry(geo)
        print('Sensor cells:',int(sensor_mask.sum()),'; dry run successful: no wave propagation',flush=True)
        return
    out.mkdir(parents=True,exist_ok=True)
    xy=geo['xy']
    k=2*np.pi*FREQUENCY/(SPEED*1000.)
    phases=k*np.sqrt((geo['snapped_xy']**2).sum(axis=1)+FOCUS_MM**2)
    write_csv(out/'array_elements.csv',[
        {'element_id':int(i), 'x_mm':float(x), 'y_mm':float(y),
         'center_phase_rad':float(ph)}
        for i,((x,y),ph) in enumerate(zip(xy,phases))
    ])
    if args.mode=='analytic':
        compare_outputs(out,analytic_only=True)
        return
    try:
        import kwave  # noqa: F401
    except ImportError as e:
        raise SystemExit(f'Install k-wave-python==0.6.2 in the active environment: {e}')
    for mode in (['point','disk'] if args.mode=='both' else [args.mode]):
        run_kwave(mode,geo,out,args.end_us,args.quiet)
    compare_outputs(out)
    (out/'config.json').write_text(json.dumps({
        'frequency_hz':FREQUENCY,'sound_speed_m_s':SPEED,'array_length_mm':MODULE_LENGTH_MM,
        'band_width_mm':BAND_MM,'element_count':len(geo['xy']),
        'rows':N_ROWS,'columns':N_COLUMNS,'element_diameter_mm':DIAMETER_MM,
        'center_pitch_mm':PITCH_MM,'requested_focus_mm':FOCUS_MM,
        'grid_spacing_mm':args.dx_mm,'source_type':'additive_pressure_cells',
        'source_strength_normalization':'equal_sum_per_element',
        'caveat':'NOT a measured/calibrated baffled piston, no skull/coupling/thermal/power',
    },indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
