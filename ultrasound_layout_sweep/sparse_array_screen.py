#!/usr/bin/env python3
"""Screen narrow-band ultrasonic array layouts under fixed 80 x 17 mm geometry.

Motivation: revisit original aligned/staggered/random array study using finite
2 mm disk footprints. Four element counts N=21,36,48,60 are compared.

Model: homogeneous-water linear analytical Green's function, each 2 mm disk
approximated by 3 x 16 polar quadrature samples. Independent elements are
steered (phase only) to (0,0,40) mm; all samples within each disk share that
phase. Each element has weight 1/sqrt(N), for an idealized fixed sum of
squared element source weights. NOT equal measured acoustic/electrical watts.

Screened 3D domain: x[-40,40], y[-24,24], z[8,80] mm. 'Off target' is defined
as Euclidean distance >=10 mm from the intended focus. It is not a physically
segmented sidelobe map. Boundary maxima and coarse-grid limitations are
explicitly flagged; rank only as a provisional shortlist.

Requires numpy, matplotlib and numba; Python 3.12 supported.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import time
import numpy as np
from numba import njit, prange

SPEED_M_S=1500.0
FREQ_HZ=500_000.0
K_MM=2*math.pi*FREQ_HZ/(SPEED_M_S*1000.0)
FOCUS_MM=40.0
LENGTH_MM=80.0
BAND_MM=17.0
DIAM_MM=2.0
Y_ROWS=np.array([-6.5,0.0,6.5],dtype=np.float64)
HALF_X=38.0  # center extents; 1mm side clearance for 2mm disks
OFF_RADIUS=10.0
ROI=(-40.0,40.0,-24.0,24.0,8.0,80.0)
COUNTS=(21,36,48,60)
MIN_CENTER_PITCH=2.5  # 2mm nominal diameter + .5mm minimum clearance

@njit(parallel=True, cache=True)
def pressure_magnitude_normalized(points,sx,sy,amp_r,amp_i,wavenumber,norm):
    out=np.empty(points.shape[0],dtype=np.float32)
    for i in prange(points.shape[0]):
        x=points[i,0];y=points[i,1];z=points[i,2]
        p_r=0.0;p_i=0.0
        for j in range(sx.shape[0]):
            dx=x-sx[j];dy=y-sy[j]
            r=math.sqrt(dx*dx+dy*dy+z*z)
            angle=wavenumber*r
            c=math.cos(angle);s=math.sin(angle)
            p_r+=(amp_r[j]*c+amp_i[j]*s)/r
            p_i+=(amp_i[j]*c-amp_r[j]*s)/r
        out[i]=math.sqrt(p_r*p_r+p_i*p_i)/norm
    return out

@njit(cache=True)
def target_amplitude(sx,sy,amp_r,amp_i,wavenumber):
    p_r=0.0;p_i=0.0
    for j in range(sx.shape[0]):
        r=math.sqrt(sx[j]*sx[j]+sy[j]*sy[j]+FOCUS_MM*FOCUS_MM)
        a=wavenumber*r
        c=math.cos(a);s=math.sin(a)
        p_r+=(amp_r[j]*c+amp_i[j]*s)/r
        p_i+=(amp_i[j]*c-amp_r[j]*s)/r
    return math.sqrt(p_r*p_r+p_i*p_i)

def layout_positions(ncols:int,kind:str,seed:int=0)->np.ndarray:
    x=np.linspace(-HALF_X,HALF_X,ncols)
    rows=[]
    for r,y in enumerate(Y_ROWS):
        if kind=='aligned':
            xx=x.copy()
        elif kind=='stagger':
            # Same N and SAME full x extent. Middle row's interior is shifted
            # smoothly up to ~half a pitch, with endpoints anchored; this is
            # a constrained stagger, NOT a perfect infinite hexagonal lattice.
            xx=x.copy()
            if r==1:
                dx=(x[-1]-x[0])/(ncols-1)
                t=np.linspace(0,math.pi,ncols)
                xx[1:-1]+=0.48*dx*np.sin(t[1:-1])
        elif kind=='random':
            # Different seed per layout AND row; endpoints remain fixed.
            rng=np.random.default_rng(seed*31+1009*r+7)
            span=2*HALF_X
            nspaces=ncols-1
            slack=span-nspaces*MIN_CENTER_PITCH
            if slack<0:raise ValueError('Too many elements for the minimum pitch')
            gaps=MIN_CENTER_PITCH+slack*rng.dirichlet(np.ones(nspaces)*2.0)
            xx=np.r_[-HALF_X,-HALF_X+np.cumsum(gaps)]
        else:
            raise ValueError(kind)
        rows.extend((float(a),float(y)) for a in xx)
    pos=np.asarray(rows,dtype=np.float64)
    # Disk footprints are inside 80x17mm band and no elements overlap.
    if np.max(np.abs(pos[:,0]))+DIAM_MM/2 > LENGTH_MM/2+1e-10:
        raise ValueError('Element footprint exceeds length')
    if np.max(np.abs(pos[:,1]))+DIAM_MM/2 > BAND_MM/2+1e-10:
        raise ValueError('Element footprint exceeds band')
    dr=pos[:,None,:]-pos[None,:,:]
    dst=np.sqrt(np.sum(dr*dr,axis=-1));np.fill_diagonal(dst,np.inf)
    if dst.min()<MIN_CENTER_PITCH-1e-9:
        raise ValueError(f'Center spacing {dst.min():.3f}mm violates minimum {MIN_CENTER_PITCH}mm')
    return pos

def disk_quadrature(xy):
    roots,gw=np.polynomial.legendre.leggauss(3)
    rr=(roots+1)/2;rw=gw/2
    offsets=[];ww=[]
    for r,w in zip(rr,rw):
        for j in range(16):
            theta=2*math.pi*j/16
            offsets.append((r*math.cos(theta),r*math.sin(theta)))
            ww.append(2*r*w/16)
    offsets=np.array(offsets,dtype=np.float64)*(DIAM_MM/2)
    w=np.array(ww,dtype=np.float64)
    assert abs(w.sum()-1)<1e-12
    src=(xy[:,None,:]+offsets[None,:,:]).reshape(-1,2)
    phases=np.exp(1j*K_MM*np.sqrt(np.sum(xy*xy,axis=1)+FOCUS_MM*FOCUS_MM))
    amps=(phases[:,None]*w[None,:]/math.sqrt(len(xy))).ravel()
    return src,amps

def points_grid(x,y,z):
    xx,yy,zz=np.meshgrid(x,y,z,indexing='ij')
    return np.column_stack((xx.ravel(),yy.ravel(),zz.ravel()))

def line_fwhm(xx,amp,center):
    mid=int(np.argmin(abs(xx-center)))
    threshold=amp[mid]/math.sqrt(2)
    left=mid;right=mid
    while left>0 and amp[left-1]>=threshold:left-=1
    while right<len(xx)-1 and amp[right+1]>=threshold:right+=1
    if left==0 or right==len(xx)-1:return math.nan
    def cross(i,j):
        a,b=amp[i],amp[j]
        return xx[i]+(threshold-a)*(xx[j]-xx[i])/(b-a)
    return float(cross(right,right+1)-cross(left-1,left))

def db(x):return 20*math.log10(max(float(x),1e-20))

def save_csv(path,rows):
    if not rows:return
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def evaluate_one(n,kind,seed,step,top=4, refine_step=0.5, refine_radius=2.0):
    from scipy.ndimage import maximum_filter
    xy=layout_positions(n//3,kind,seed)
    sources,amps=disk_quadrature(xy)
    sx=np.ascontiguousarray(sources[:,0]);sy=np.ascontiguousarray(sources[:,1])
    ar=np.ascontiguousarray(amps.real);ai=np.ascontiguousarray(amps.imag)
    norm=target_amplitude(sx,sy,ar,ai,K_MM)
    def calc(pts):return pressure_magnitude_normalized(np.ascontiguousarray(pts,dtype=np.float64),sx,sy,ar,ai,K_MM,norm)
    # 3D coarse scan for off-target candidates
    x=np.arange(ROI[0],ROI[1]+step*.1,step)
    y=np.arange(ROI[2],ROI[3]+step*.1,step)
    z=np.arange(ROI[4],ROI[5]+step*.1,step)
    # process each x-plane to keep memory modest
    out=np.empty((len(x),len(y),len(z)),dtype=np.float32)
    for i,xi in enumerate(x):
        p=points_grid(np.array([xi]),y,z)
        out[i,:,:]=calc(p).reshape(len(y),len(z))
    xx,yy,zz=np.meshgrid(x,y,z,indexing='ij',sparse=True)
    mask=xx*xx+yy*yy+(zz-FOCUS_MM)**2>=OFF_RADIUS*OFF_RADIUS
    masked=np.where(mask,out,-np.inf)
    extrema=(out>=maximum_filter(out,size=3,mode='constant',cval=-np.inf)-1e-6)&mask
    ix,iy,iz=np.where(extrema)
    order=np.argsort(out[ix,iy,iz])[::-1]
    candidates=[]
    for idx in order:
        pt=np.array([x[ix[idx]],y[iy[idx]],z[iz[idx]]])
        if any(np.linalg.norm(pt-c['xyz'])<6 for c in candidates):continue
        candidates.append({'xyz':pt,'coarse_amp':float(out[ix[idx],iy[idx],iz[idx]])})
        if len(candidates)==top:break
    best=max(candidates,key=lambda c:c['coarse_amp'])
    loc=best['xyz']
    boundary=(np.any(np.isclose(loc,[ROI[0],ROI[2],ROI[4]])) or np.any(np.isclose(loc,[ROI[1],ROI[3],ROI[5]])))
    # Locally sample multiple strong off-target candidates at 0.5mm resolution.
    # Restrict to the same ROI; boundary winners are explicitly flagged.
    refined=[]
    offsets=np.arange(-refine_radius,refine_radius+refine_step*.1,refine_step)
    for cand in candidates:
        xyz=cand['xyz']
        lx=xyz[0]+offsets;ly=xyz[1]+offsets;lz=xyz[2]+offsets
        lx=lx[(lx>=ROI[0])&(lx<=ROI[1])]
        ly=ly[(ly>=ROI[2])&(ly<=ROI[3])]
        lz=lz[(lz>=ROI[4])&(lz<=ROI[5])]
        ppts=points_grid(lx,ly,lz)
        pp=calc(ppts)
        eligible=np.sum((ppts-np.array([0.0,0.0,FOCUS_MM]))**2,axis=1)>=OFF_RADIUS**2
        pp=np.where(eligible,pp,-np.inf)
        k=int(np.argmax(pp))
        candidate_position=ppts[k]
        at_boundary=any(abs(candidate_position[i]-edge)<refine_step/2 for i,edges in enumerate(((ROI[0],ROI[1]),(ROI[2],ROI[3]),(ROI[4],ROI[5]))) for edge in edges)
        refined.append(dict(xyz=candidate_position,amp=float(pp[k]),roi_edge=bool(at_boundary),seed_xyz=xyz))
    refined_best=max(refined,key=lambda r:r['amp'])
    # Fine 1D cuts around requested target independent of coarse scan
    xfine=np.arange(-15,15.0001,.25)
    yfine=np.arange(-12,12.0001,.25)
    zfine=np.arange(8,80.0001,.25)
    xf=calc(np.column_stack([xfine,np.zeros_like(xfine),np.full_like(xfine,FOCUS_MM)]))
    yf=calc(np.column_stack([np.zeros_like(yfine),yfine,np.full_like(yfine,FOCUS_MM)]))
    zf=calc(np.column_stack([np.zeros_like(zfine),np.zeros_like(zfine),zfine]))
    zwindow=(zfine>=18)&(zfine<=60)
    zlocal=zfine[zwindow][np.argmax(zf[zwindow])]
    return {
        'row':dict(elements=n,columns=n//3,layout=kind,random_seed=(seed if kind=='random' else ''),
            length_mm=LENGTH_MM,band_mm=BAND_MM,diameter_mm=DIAM_MM,min_pitch_mm=MIN_CENTER_PITCH,
            focus_mm=FOCUS_MM,focus_pressure_proxy=float(norm),
            axial_peak_z_mm=float(zlocal),x_fwhm_mm=line_fwhm(xfine,xf,0),
            y_fwhm_mm=line_fwhm(yfine,yf,0),
            axial_fwhm_mm=line_fwhm(zfine,zf,FOCUS_MM),
            off_target_peak_db_coarse=db(best['coarse_amp']),
            off_target_peak_x_mm=float(loc[0]),off_target_peak_y_mm=float(loc[1]),off_target_peak_z_mm=float(loc[2]),
            off_target_on_roi_boundary=bool(boundary),
            scan_step_mm=step,refine_step_mm=refine_step,
            off_target_peak_db_refined=db(refined_best['amp']),
            refined_peak_x_mm=float(refined_best['xyz'][0]),
            refined_peak_y_mm=float(refined_best['xyz'][1]),
            refined_peak_z_mm=float(refined_best['xyz'][2]),
            refined_peak_on_roi_boundary=refined_best['roi_edge'],
            interpretation='Screening only: max >=10mm from focus in finite ROI, not true full sidelobe metric'),
        'xy':xy,'src':(sx,sy,ar,ai,norm),'off_candidates':candidates,'refined_candidates':refined,
        'line':(zfine,zf)
    }

def run(args):
    out=args.out
    if out.exists() and any(out.iterdir()) and not args.dry_run:
        raise SystemExit(f'ERROR: {out} already contains files. Choose --out another_folder to avoid overwriting.')
    kinds=['aligned','stagger']+[f'random_{seed}' for seed in range(args.random_seeds)]
    combos=[(n,k,0) if not k.startswith('random_') else (n,'random',int(k.split('_')[1])) for n in args.counts for k in kinds]
    print(f'{len(combos)} layouts, ROI={ROI}, scan spacing={args.step_mm}mm, 3x16 disk quadrature',flush=True)
    if args.dry_run:
        for n,k,s in combos:
            xy=layout_positions(n//3,k,s)
            print(f'  N={n:2d} {k:8s} seed={s} centers={xy.shape[0]}')
        return
    out.mkdir(parents=True,exist_ok=True)
    rows=[];centers=[];seeds=[];profiles={}
    start=time.monotonic()
    for idx,(n,kind,seed) in enumerate(combos,1):
        t0=time.monotonic()
        res=evaluate_one(n,kind,seed,args.step_mm,args.candidates,args.refine_step_mm,args.refine_radius_mm)
        row=res['row']
        rows.append(row)
        for i,(x,y) in enumerate(res['xy']):
            centers.append(dict(elements=n,layout=kind,random_seed=(seed if kind=='random' else ''),element_id=i,x_mm=float(x),y_mm=float(y)))
        for i,c in enumerate(res['off_candidates']):
            p=c['xyz']
            seeds.append(dict(elements=n,layout=kind,random_seed=(seed if kind=='random' else ''),rank=i+1,
                stage='coarse',coarse_amp_ratio=c['coarse_amp'],coarse_db=db(c['coarse_amp']),x_mm=float(p[0]),y_mm=float(p[1]),z_mm=float(p[2])))
        for j,c in enumerate(res['refined_candidates']):
            p=c['xyz']
            seeds.append(dict(elements=n,layout=kind,random_seed=(seed if kind=='random' else ''),rank=j+1,
                stage='local_refined',coarse_amp_ratio=c['amp'],coarse_db=db(c['amp']),
                x_mm=float(p[0]),y_mm=float(p[1]),z_mm=float(p[2])))
        profiles[(n,kind,seed)]=res['line']
        print(f'[{idx:02}/{len(combos):02}] N={n:2} {kind:8s} seed={seed:2} off-target {row["off_target_peak_db_refined"]:+6.2f}dB at ({row["off_target_peak_x_mm"]:.1f},{row["off_target_peak_y_mm"]:.1f},{row["off_target_peak_z_mm"]:.1f}); {time.monotonic()-t0:.1f}s',flush=True)
        save_csv(out/'screening_summary.csv',rows)
        save_csv(out/'array_centers.csv',centers)
        save_csv(out/'offtarget_peak_candidates.csv',seeds)
    # Common reference for gain, ensures comparable source model within this simulation.
    ref=next(row['focus_pressure_proxy'] for row in rows if row['elements']==21 and row['layout']=='aligned')
    for row in rows:row['target_gain_vs_N21_aligned_db']=db(row['focus_pressure_proxy']/ref)
    save_csv(out/'screening_summary.csv',rows)
    # Proposed shortlist per N: aligned baseline + best (lowest off-target sampled peak)
    # Does NOT assert global optimum, or adequate biological selectivity.
    shortlist=[]
    for n in args.counts:
        subset=[r for r in rows if r['elements']==n]
        base=next(r for r in subset if r['layout']=='aligned')
        best=min(subset,key=lambda r:r['off_target_peak_db_refined'])
        for reason,r in [('aligned_baseline',base),('lowest_sampled_offtarget',best)]:
            shortlist.append(dict(reason=reason,**r))
    save_csv(out/'provisional_shortlist.csv',shortlist)
    (out/'study_config.json').write_text(json.dumps(dict(
        frequency_hz=FREQ_HZ,water_sound_speed_m_s=SPEED_M_S,
        module_mm=[LENGTH_MM,BAND_MM],diameter_mm=DIAM_MM,
        target=[0,0,FOCUS_MM],counts=args.counts,random_seeds=args.random_seeds,
        grid_mm=args.step_mm,local_refine_step_mm=args.refine_step_mm,local_refine_radius_mm=args.refine_radius_mm,ROI_xyz_mm=ROI,offtarget_exclusion_sphere_radius_mm=OFF_RADIUS,
        numerical_model='3 radial Gauss quadrature x 16 azimuth of finite circular footprint, phase at element center',
        normalization='one element amplitude scaled 1/sqrt(N); not calibrated watts',
        warnings=['ROI and grid limited: may miss true maximum', 'off-target outside 10 mm sphere is not true sidelobe segmentation',
                  'fixed 2 mm source footprint is additive monopole quadrature not calibrated baffled piston',
                  'no skull, thermal, transducer efficiency, circuit power, or coupling',
                  'stagger has anchored endpoints and gradually shifted center-row interiors; not a perfect hexagonal lattice']),indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(10,7),sharex=True)
    for i,n in enumerate(args.counts):
        ax=axs.flat[i]
        for r in rows:
            if r['elements'] != n:continue
            marker='o' if r['layout']=='aligned' else ('s' if r['layout']=='stagger' else '.')
            lab=('random' if r['layout']=='random' else r['layout'])
            ax.scatter(r['x_fwhm_mm'],r['off_target_peak_db_refined'],s=56,marker=marker,label=lab,
                       alpha=.8)
        ax.axhline(0,color='gray',ls=':',lw=.8)
        ax.set(title=f'N={n}',xlabel='X mainlobe FWHM (mm)',ylabel='Max sampled off-target (dB)')
        ax.grid(alpha=.2)
        handles,labels=ax.get_legend_handles_labels()
        uniq=dict(zip(labels,handles));ax.legend(uniq.values(),uniq.keys(),fontsize=8)
    fig.suptitle(f'Preliminary 3D layout screening, {args.step_mm} mm grid (NOT k-Wave)')
    fig.tight_layout();fig.savefig(out/'screening_tradeoffs.png',dpi=150);plt.close(fig)
    # Comparison for reproducibility: layout map for N=36 aligned / stagger / first random
    fig,axs=plt.subplots(1,3,figsize=(11,2.8),sharex=True,sharey=True)
    for ax,(kind,seed) in zip(axs,[('aligned',0),('stagger',0),('random',0)]):
        xy=layout_positions(12,kind,seed)
        ax.scatter(xy[:,0],xy[:,1],s=15)
        ax.set(title=f'N=36 {kind}',xlim=(-40,40),ylim=(-8.5,8.5),xlabel='X (mm)')
        ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.1)
    axs[0].set_ylabel('Y (mm)');fig.tight_layout()
    fig.savefig(out/'example_layouts_N36.png',dpi=150);plt.close(fig)
    print(f'Finished {len(rows)} layouts in {time.monotonic()-start:.1f}s; output in {out}',flush=True)
    print('NOTE: Screening only. 3D grid and ROI limited, finite-disk analytical model; not k-Wave or measured acoustic power.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--counts',nargs='+',type=int,choices=COUNTS,default=list(COUNTS))
    parser.add_argument('--random-seeds',type=int,default=3)
    parser.add_argument('--step-mm',type=float,default=2.0,
        help='2mm initial 3D screen. 4mm may miss strong grating lobes; not a convergence check.')
    parser.add_argument('--candidates',type=int,default=5)
    parser.add_argument('--refine-step-mm',type=float,default=0.5)
    parser.add_argument('--refine-radius-mm',type=float,default=2.0)
    parser.add_argument('--out',type=Path,default=Path('sparse_array_screen_results'))
    parser.add_argument('--dry-run',action='store_true')
    a=parser.parse_args()
    if a.random_seeds<1 or a.candidates<1:parser.error('Positive --random-seeds and --candidates required')
    for lo,hi in zip(ROI[::2],ROI[1::2]):
        if abs((hi-lo)/a.step_mm-round((hi-lo)/a.step_mm))>1e-9:
            parser.error('--step-mm must divide ROI side lengths 80,48,72')
    run(a)
