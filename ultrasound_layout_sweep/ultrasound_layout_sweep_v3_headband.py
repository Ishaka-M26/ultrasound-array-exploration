#!/usr/bin/env python3
"""Headband-constrained array screen V3.1: planar LOCAL patch only.

Run: python ultrasound_layout_sweep_v3_headband.py
Dependencies: numpy, scipy, matplotlib. No MATLAB, k-Wave binary, or GPU required.

MODEL: Phase-steered, uniform circular baffled pistons in infinite, homogeneous
water, using each piston's far-field directivity (2 J1(u)/u) and a spherical
1/r propagator. Valid only when observation distances are much greater than a
single piston's radius. Elements share one common plane, no coupling, skull,
attenuation, driver limitations, or calibrated source measurements. For equal
modeled far-field hemispherical power, integrate the coherent array pattern.
A 1 W acoustic normalization is a THEORETICAL model reference, not a claim a
real device can radiate 1 W. This is not a safety/dosimetry calculation.

28 unique layouts x 3 band widths x 4 focal depths = 336 CSV rows.
Unbuildable arrangements retain CSV rows but have NaN acoustics and a reason.

V3 improvements over V2:
* Fixed-size circular radiator with piston directivity (not a point monopole).
* Mechanical collision and package-footprint checks.
* Physically motivated equal-forward-hemisphere-radiated-power normalization.
* 3-D -3 dB intensity connected focal volume and xyz/global peak metrics.
* X, Y and axial Z FWHM with explicit failure/truncation status.
* Explicit geometrical 3-D off-target guard region, not a validated sidelobe.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import label
from scipy.spatial.distance import pdist

ELEMENT_COUNTS = (16, 32, 64, 128)
ROWS = (1, 2, 3)
LAYOUTS = ('aligned', 'horizontal_stagger', 'vertical_stagger')
DEPTHS_MM = (20.0, 40.0, 60.0, 80.0)

# All of these may be changed independently for your design study.
FREQ_HZ = 500_000.0
C_M_S = 1500.0
RHO_KG_M3 = 1000.0
PACKAGE_WIDTH_MM = 30.0     # LOCAL active module length ALONG band, not full headband arc
BAND_WIDTHS_MM = (12.0, 17.0, 26.0) # commercial illustrative widths, not universal standards
BAND_EDGE_MARGIN_MM = 1.0   # margin from both band width edges
HEADBAND_ARC_LENGTH_MM = 400.0  # packaging metadata ONLY; not simulated acoustically
HEADBAND_OUTER_HEIGHT_MM = 155.0 # illustrative packaging metadata, not head anatomy
HEADBAND_OUTER_WIDTH_MM = 130.0  # illustrative packaging metadata, not head anatomy
PISTON_DIAMETER_MM = 0.60   # hypothetical, not an off-the-shelf specification
EDGE_GAP_MM = 0.08          # desired minimum edge-to-edge clearance
THEORETICAL_ACOUSTIC_W = 1.0

K_MM = 2 * np.pi * FREQ_HZ / (C_M_S * 1000.)
RADIUS_MM = PISTON_DIAMETER_MM / 2
LAMBDA_MM = C_M_S * 1000 / FREQ_HZ

# Whole-volume screening map. The z-domain is a plane at least 8 mm from disks.
GLOBAL_X = np.arange(-45., 45.01, 3.)
GLOBAL_Y = np.arange(-45., 45.01, 3.)
GLOBAL_Z = np.arange(8., 108.01, 4.)
# Higher-resolution volume around the requested target, for -3 dB volume.
LOCAL_X = np.arange(-18., 18.01, 1.)
LOCAL_Y = np.arange(-45., 45.01, 2.)
LOCAL_REL_Z = np.arange(-30., 30.01, 2.)
# 1-D high-resolution cut domains; FWHM is interpolated.
CUT_X = np.arange(-45., 45.01, .5)
CUT_Y = np.arange(-90., 90.01, .5)
CUT_Z = np.arange(5., 125.01, .5)


def centers(n: int, rows: int, layout: str, row_pitch_mm: float) -> np.ndarray:
    """(N,2) piston CENTER coordinates. Every layout uses same total x footprint.

    aligned: all rows share regular x positions.
    horizontal_stagger: odd-numbered rows shift in x by half the local pitch.
    vertical_stagger: odd-numbered COLUMN positions shift in y by +/- quarter
        of the row pitch. The y-span is renormalized to the aligned row span.
    For rows>1, x-span and y-span are kept constant across layout families.
    """
    if rows == 1 and layout != 'aligned':
        raise ValueError('No stagger is defined for a single row')
    # Distribute remainders symmetrically to avoid an artificial outer-row bias.
    if rows == 3:
        counts = [(n - n // 3) // 2, n // 3 + ((n - n // 3) % 2),
                  (n - n // 3) // 2]
        # Reconcile any integer remainder to the center row.
        counts[1] += n - sum(counts)
    else:
        counts = [n // rows + int(r < n % rows) for r in range(rows)]
    points = []
    usable_width = PACKAGE_WIDTH_MM - PISTON_DIAMETER_MM
    for row, count in enumerate(counts):
        x = np.linspace(-usable_width / 2, usable_width / 2, count)
        y = np.full(count, (row - (rows - 1) / 2) * row_pitch_mm)
        pitch = usable_width / (count - 1) if count > 1 else 0.
        if layout == 'horizontal_stagger' and (row % 2):
            x = x + pitch / 2
        if layout == 'vertical_stagger':
            y = y + (2 * (np.arange(count) % 2) - 1) * row_pitch_mm / 4
        points.extend(zip(x, y))
    xy = np.array(points, dtype=np.float64)
    xy[:, 0] -= (xy[:, 0].max() + xy[:, 0].min()) / 2
    xy[:, 0] *= usable_width / np.ptp(xy[:, 0])
    if rows > 1:
        xy[:, 1] -= (xy[:, 1].max() + xy[:, 1].min()) / 2
        xy[:, 1] *= (rows - 1) * row_pitch_mm / np.ptp(xy[:, 1])
    else:
        xy[:, 1] = 0.0
    assert len(xy) == n
    return xy


def geometry_diagnostics(xy: np.ndarray, band_width_mm: float) -> dict:
    distances = pdist(xy)
    min_pitch = float(np.min(distances))
    min_clearance = min_pitch - PISTON_DIAMETER_MM
    span_x = np.ptp(xy[:, 0]) + PISTON_DIAMETER_MM
    span_y = np.ptp(xy[:, 1]) + PISTON_DIAMETER_MM
    # Treat spacing very close to min as feasible, up to floating-point noise.
    band_usable_width = band_width_mm - 2*BAND_EDGE_MARGIN_MM
    feasible = (min_clearance >= EDGE_GAP_MM - 1e-8
                and span_x <= PACKAGE_WIDTH_MM + 1e-8
                and span_y <= band_usable_width + 1e-8)
    if feasible:
        reason = 'ok'
    elif min_clearance < -1e-8:
        reason = 'overlapping_disks'
    elif span_y > band_usable_width + 1e-8:
        reason = 'exceeds_band_width'
    else:
        reason = 'clearance_below_design_minimum'
    return {
        'geometry_feasible': bool(feasible),
        'geometry_status': reason,
        'min_center_distance_mm': min_pitch,
        'min_edge_clearance_mm': min_clearance,
        'physical_width_mm': float(span_x),
        'physical_height_mm': float(span_y),
        'usable_band_width_mm': float(band_usable_width),
        'band_fit': bool(span_y <= band_usable_width + 1e-8),
    }


def piston_directivity(u: np.ndarray) -> np.ndarray:
    """Far field of a uniformly vibrating circular piston in a rigid baffle.

    For present ka<=0.629, 4-term Taylor expansion has <1e-10 relative error
    over the whole forward hemisphere, saving a costly special-function call
    inside the 3-D wavefield loop. Test verified against 2*J1(u)/u.
    """
    w = u * u
    return 1.0 - w/8.0 + w*w/192.0 - w*w*w/9216.0 + w*w*w*w/737280.0


def radiated_power_integral(xy: np.ndarray, phase: np.ndarray) -> float:
    """Integral over FORWARD hemisphere of |far-field array factor|^2 dOmega.

    Uses 48-point Gauss-Legendre in mu=cos(theta), and 128 azimuth samples.
    This includes interference between phase-steered elements and finite-piston
    directivity. Baffle, no back-radiation and no element coupling assumed.
    """
    mu, wmu = np.polynomial.legendre.leggauss(48)
    mu = 0.5 * (mu + 1)
    wmu = wmu * 0.5
    az = np.arange(128) * (2*np.pi/128)
    sinth = np.sqrt(1 - mu**2)
    ux = (sinth[:, None] * np.cos(az)[None, :]).ravel()
    uy = (sinth[:, None] * np.sin(az)[None, :]).ravel()
    d = piston_directivity(K_MM * RADIUS_MM * np.repeat(sinth, len(az)))
    angles = (K_MM * (ux[:, None] * xy[None, :, 0] +
                       uy[:, None] * xy[None, :, 1]) + phase[None, :])
    af = np.exp(1j * angles).sum(axis=1) * d
    weights = np.repeat(wmu, len(az)) * (2*np.pi/len(az))
    integral = float(np.dot(weights, np.abs(af)**2))
    if integral <= 0 or not np.isfinite(integral):
        raise RuntimeError('Invalid radiated power integral')
    return integral


def field(xy: np.ndarray, phase_factors: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Pressure field: finite piston FAR-FIELD directivity x spherical propagator.

    xyz is (M, 3) in mm; returned is complex wave coefficient in 1/mm.
    Actual pressure field is obtained by normalized model scale below.
    Each disk is assumed small relative to ALL source-observation distances.
    """
    p = np.zeros(len(xyz), dtype=np.complex128)
    X = xyz[:, 0]
    Y = xyz[:, 1]
    Z = xyz[:, 2]
    for (xi, yi), phase in zip(xy, phase_factors):
        dx, dy = X-xi, Y-yi
        transverse2 = dx*dx + dy*dy
        rr = np.sqrt(transverse2 + Z*Z)
        u = K_MM * RADIUS_MM * np.sqrt(transverse2) / rr
        d = piston_directivity(u)
        p += phase * d * np.exp(-1j*K_MM*rr) / rr
    return p


def coordinate_grid(xs, ys, zs):
    a, b, c = np.meshgrid(xs, ys, zs, indexing='ij')
    return np.stack((a.ravel(), b.ravel(), c.ravel()), axis=1), a.shape


def width_from_center(coord, line, center_coord):
    """Intensity -3dB FWHM from the pressure AMPLITUDE, relative to target.

    Handles peaks elsewhere, missing crossings, and clipped ROI explicitly.
    """
    k0 = int(np.argmin(abs(coord-center_coord)))
    if abs(coord[k0]-center_coord) > 1e-7:
        raise ValueError('Target absent from line grid')
    a0 = float(line[k0])
    if a0 <= 0 or not np.isfinite(a0):
        return np.nan, 'invalid_target'
    threshold = a0 / np.sqrt(2)
    edges = []
    for sign in (-1, 1):
        i = k0
        crossed = False
        while 0 <= i + sign < len(coord):
            j = i + sign
            if line[j] < threshold <= line[i]:
                x = coord[i] + (threshold-line[i]) * (coord[j]-coord[i]) / (line[j]-line[i])
                edges.append(float(x))
                crossed = True
                break
            i = j
        if not crossed:
            return np.nan, 'roi_truncated_or_nonfocusing'
    return edges[1]-edges[0], 'ok'


def evaluate(xy: np.ndarray, depth: float):
    """All acoustic metrics at one depth, normalized to model radiated 1 W."""
    focus_r = np.sqrt(np.sum(xy**2, axis=1) + depth**2)
    # Sign MUST be positive with propagator exp(-1j*k*r): phase compensates r.
    phases = K_MM*focus_r
    phase_factors = np.exp(1j*phases)
    power_integral = radiated_power_integral(xy, phases)
    # p_model = sqrt(2*rho*c*1W/integral) * 1000 * field_in_inverse_mm
    model_pa_scale = np.sqrt(2*RHO_KG_M3*C_M_S*THEORETICAL_ACOUSTIC_W /
                             power_integral) * 1000.
    target = np.array([[0., 0., depth]])
    focal_pa = float(abs(field(xy, phase_factors, target)[0]) * model_pa_scale)

    cx = np.column_stack((CUT_X, np.zeros_like(CUT_X), np.full_like(CUT_X,depth)))
    cy = np.column_stack((np.zeros_like(CUT_Y), CUT_Y, np.full_like(CUT_Y,depth)))
    cz = np.column_stack((np.zeros_like(CUT_Z), np.zeros_like(CUT_Z), CUT_Z))
    px = np.abs(field(xy, phase_factors, cx))
    py = np.abs(field(xy, phase_factors, cy))
    pz = np.abs(field(xy, phase_factors, cz))
    wx, sx = width_from_center(CUT_X, px, 0.0)
    wy, sy = width_from_center(CUT_Y, py, 0.0)
    wz, sz = width_from_center(CUT_Z, pz, depth)

    # Local target neighborhood: connected component containing requested focus.
    clipped_local_z = depth + LOCAL_REL_Z
    clipped_local_z = clipped_local_z[(clipped_local_z >= 8) & (clipped_local_z <= 110)]
    local_xyz, local_shape = coordinate_grid(LOCAL_X, LOCAL_Y, clipped_local_z)
    local_pa = np.abs(field(xy, phase_factors, local_xyz)).reshape(local_shape)
    local_intensity = (local_pa / (focal_pa / model_pa_scale))**2
    component_labels, _ = label(local_intensity >= 0.5)
    center_idx = (int(np.argmin(abs(LOCAL_X))),
                  int(np.argmin(abs(LOCAL_Y))),
                  int(np.argmin(abs(clipped_local_z-depth))))
    center_label = int(component_labels[center_idx])
    mainlobe = ((component_labels == center_label) if center_label else
                np.zeros(local_shape, dtype=bool))
    truncated = bool(
        mainlobe[0,:,:].any() or mainlobe[-1,:,:].any() or
        mainlobe[:,0,:].any() or mainlobe[:,-1,:].any() or
        mainlobe[:,:,0].any() or mainlobe[:,:,-1].any()
    )
    voxel_mm3 = (LOCAL_X[1]-LOCAL_X[0]) * (LOCAL_Y[1]-LOCAL_Y[0]) * (
        clipped_local_z[1]-clipped_local_z[0])
    observed_volume = float(mainlobe.sum() * voxel_mm3)
    focal_volume = np.nan if truncated else observed_volume

    global_xyz, global_shape = coordinate_grid(GLOBAL_X, GLOBAL_Y, GLOBAL_Z)
    g_amp = np.abs(field(xy, phase_factors, global_xyz))
    g_max_idx = int(np.argmax(g_amp))
    g_peak_xyz = global_xyz[g_max_idx]
    g_peak_pa = float(g_amp[g_max_idx] * model_pa_scale)
    peak_offset = float(np.linalg.norm(g_peak_xyz - target[0]))

    # Deterministically defined exclusion ellipsoid: semi-axes 1.5 x each
    # FWHM. This is NOT an algorithm detecting the true first-null sidelobe.
    if all(s == 'ok' for s in (sx, sy, sz)):
        xrad, yrad, zrad = [max(3., 1.5*v) for v in (wx,wy,wz)]
        delta = global_xyz - target[0]
        ellipsoid_dist = ((delta[:,0]/xrad)**2 + (delta[:,1]/yrad)**2 +
                          (delta[:,2]/zrad)**2)
        outside = ellipsoid_dist > 1
        observed_off = float(np.max(g_amp[outside])) if outside.any() else np.nan
        off_db = (float(20*np.log10(observed_off/(focal_pa/model_pa_scale)))
                  if observed_off > 0 else np.nan)
        off_status = 'ok_guard_ellipsoid_not_true_sidelobe'
    else:
        off_db = np.nan
        off_status = 'not_defined_missing_fwhm'

    return {
        'model_hemisphere_power_integral': power_integral,
        'focus_pressure_pa_at_model_1w': focal_pa,
        'fwhm_x_mm': wx, 'fwhm_x_status': sx,
        'fwhm_y_mm': wy, 'fwhm_y_status': sy,
        'fwhm_z_mm': wz, 'fwhm_z_status': sz,
        'focal_halfpower_volume_mm3': focal_volume,
        'focal_volume_observed_lower_bound_mm3': observed_volume,
        'focal_volume_status': 'roi_truncated' if truncated else 'ok',
        'global_peak_pressure_pa_at_model_1w': g_peak_pa,
        'global_peak_x_mm': float(g_peak_xyz[0]),
        'global_peak_y_mm': float(g_peak_xyz[1]),
        'global_peak_z_mm': float(g_peak_xyz[2]),
        'global_peak_offset_from_target_mm': peak_offset,
        'global_peak_to_target_db': float(20*np.log10(g_peak_pa/focal_pa)),
        'offtarget_peak_outside_1p5_fwhm_ellipsoid_db': off_db,
        'offtarget_status': off_status,
    }


def make_plots(output_dir: Path, records: list[dict]):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # N64, 3 rows, aligned and staggered compare at 40mm
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for layout in LAYOUTS:
        selected = sorted([r for r in records
            if r['n_elements']==64 and r['n_rows']==3 and
               r['layout']==layout and r['depth_mm']==40 and
               r['geometry_feasible']], key=lambda r:r['band_width_mm'])
        if selected:
            x=[r['band_width_mm'] for r in selected]
            axes[0].plot(x,[r['fwhm_y_mm'] for r in selected],marker='o',label=layout)
            axes[1].plot(x,[r['focus_gain_vs_16_1row_db'] for r in selected],marker='o',label=layout)
    axes[0].set_ylabel('Focal Y FWHM (mm)')
    axes[1].set_ylabel('Focal gain vs 16-element 1-row (dB)')
    for a in axes:
        a.set_xlabel('Band width (mm)')
        a.set_xticks(BAND_WIDTHS_MM)
        a.grid(alpha=.25)
        a.legend(fontsize=8)
    fig.suptitle('Headband width screen: N64 / 3 rows / 40mm depth (local flat patch)')
    fig.tight_layout()
    fig.savefig(output_dir/'band_width_comparison_N64_R3_40mm.png',dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true', help='12mm band, 16 elements, 20mm depth only')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    counts = (16,) if args.quick else ELEMENT_COUNTS
    bands = (12.,) if args.quick else BAND_WIDTHS_MM
    depths = (20.,) if args.quick else DEPTHS_MM
    out = Path(__file__).resolve().parent / 'layout_sweep_results_v3_headband'
    out.mkdir(exist_ok=True)
    all_rows = []
    all_centers = []
    t_start = time.monotonic()
    for band in bands:
        usable = band - 2*BAND_EDGE_MARGIN_MM
        if usable < PISTON_DIAMETER_MM:
            raise ValueError('Band width is less than radiator diameter plus margins')
        for n in counts:
            for nr in ROWS:
                # Fill available transverse band width to test the BEST-CASE
                # effect of added Y aperture. This is not a comfort specification.
                row_pitch = ((usable-PISTON_DIAMETER_MM)/(nr-1)) if nr > 1 else 0.0
                for layout in (('aligned',) if nr==1 else LAYOUTS):
                    xy = centers(n,nr,layout,row_pitch)
                    diag = geometry_diagnostics(xy,band)
                    case = f'B{int(band):02d}_N{n:03d}_R{nr}_{layout}'
                    for idx,(x,y) in enumerate(xy):
                        all_centers.append({'case_id':case,'band_width_mm':band,
                                            'element_id':idx+1,'x_mm':float(x),
                                            'y_mm':float(y),'diameter_mm':PISTON_DIAMETER_MM})
                    for depth in depths:
                        row = {'case_id':case,'band_width_mm':band,
                            'band_edge_margin_mm':BAND_EDGE_MARGIN_MM,
                            'headband_arc_length_mm_metadata':HEADBAND_ARC_LENGTH_MM,
                            'headband_outer_height_mm_metadata':HEADBAND_OUTER_HEIGHT_MM,
                            'headband_outer_width_mm_metadata':HEADBAND_OUTER_WIDTH_MM,
                            'local_active_module_length_mm':PACKAGE_WIDTH_MM,
                            'row_pitch_mm':row_pitch,
                            'n_elements':n,'n_rows':nr,'layout':layout,
                            'depth_mm':depth,'frequency_hz':FREQ_HZ,
                            'wavelength_mm':LAMBDA_MM,
                            'piston_diameter_mm':PISTON_DIAMETER_MM,
                            'min_required_edge_gap_mm':EDGE_GAP_MM,
                            **diag}
                        if diag['geometry_feasible']:
                            row.update(evaluate(xy,depth))
                        all_rows.append(row)
                    print(f'{case}: {"OK" if diag["geometry_feasible"] else diag["geometry_status"]}', flush=True)
    baseline={(r['band_width_mm'],r['depth_mm']):r['focus_pressure_pa_at_model_1w']
        for r in all_rows if r['n_elements']==16 and r['n_rows']==1}
    for row in all_rows:
        pa=row.get('focus_pressure_pa_at_model_1w',np.nan)
        ref=baseline[(row['band_width_mm'],row['depth_mm'])]
        if np.isfinite(pa):
            row['focus_gain_vs_16_1row_db']=float(20*np.log10(pa/ref))
            row['relative_radiated_power_for_equal_target']=float((ref/pa)**2)
        else:
            row['focus_gain_vs_16_1row_db']=np.nan
            row['relative_radiated_power_for_equal_target']=np.nan
    keys=[
        'case_id','band_width_mm','band_edge_margin_mm',
        'headband_arc_length_mm_metadata','headband_outer_height_mm_metadata',
        'headband_outer_width_mm_metadata','local_active_module_length_mm',
        'row_pitch_mm','n_elements','n_rows','layout','depth_mm','frequency_hz',
        'wavelength_mm','piston_diameter_mm','min_required_edge_gap_mm',
        'geometry_feasible','geometry_status','physical_width_mm',
        'physical_height_mm','usable_band_width_mm','band_fit',
        'min_center_distance_mm','min_edge_clearance_mm',
        'model_hemisphere_power_integral','focus_pressure_pa_at_model_1w',
        'focus_gain_vs_16_1row_db','relative_radiated_power_for_equal_target',
        'fwhm_x_mm','fwhm_x_status','fwhm_y_mm','fwhm_y_status',
        'fwhm_z_mm','fwhm_z_status','focal_halfpower_volume_mm3',
        'focal_volume_observed_lower_bound_mm3','focal_volume_status',
        'global_peak_pressure_pa_at_model_1w','global_peak_x_mm',
        'global_peak_y_mm','global_peak_z_mm','global_peak_offset_from_target_mm',
        'global_peak_to_target_db',
        'offtarget_peak_outside_1p5_fwhm_ellipsoid_db','offtarget_status'
    ]
    for name, subset in [('results_all_depths_v3_headband.csv',all_rows),
                         ('results_40mm_v3_headband.csv',[r for r in all_rows if r['depth_mm']==40])]:
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=keys)
            writer.writeheader()
            writer.writerows(subset)
    with (out/'array_positions_v3_headband.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['case_id','band_width_mm','element_id',
                                            'x_mm','y_mm','diameter_mm'])
        writer.writeheader()
        writer.writerows(all_centers)
    if not args.no_plots and not args.quick:
        make_plots(out,all_rows)
    good=sum(r['geometry_feasible'] for r in all_rows)
    print(f'Finished: {len(all_rows)} rows, {good} feasible, {len(all_rows)-good} invalid')
    print('Output:',out)
    print('Elapsed:',round(time.monotonic()-t_start,1),'seconds')
    print('LIMITATIONS: Band modeled as a locally FLAT 30mm long patch; full 400mm curved headband')
    print('not included in acoustic propagation. 1 W normalization is theoretical, not clinical or measured.')
    print('No skull, thermal, attenuation, coupling or actual electrical consumption.')


if __name__ == '__main__':
    main()
