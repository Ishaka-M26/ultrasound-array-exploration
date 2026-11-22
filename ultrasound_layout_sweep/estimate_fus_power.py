#!/usr/bin/env python3
"""Illustrative electrical power budget based on 48/60-element k-Wave results.

WARNING: The k-Wave additive-pressure source is NOT an electrical transducer model.
No electrical watts can be inferred without assumptions for the transducer,
voltage-to-pressure transfer, driver, and duty cycle.

Run:
    python estimate_fus_power.py
    python estimate_fus_power.py --comparison path/to/completed_comparison.csv
    python estimate_fus_power.py --cap-pf 100 --r-motion-kohm 20 --vrms60 10
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def find_comparison(requested: str | None) -> Path:
    candidates = [Path(requested)] if requested else [
        Path('kwave_sparse_48_vs_60/completed_comparison.csv'),
        Path('kwave_sparse_48_vs60/completed_comparison.csv'),
        Path('completed_comparison.csv'),
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise SystemExit('Could not locate completed_comparison.csv; use --comparison PATH')


def load_pressures(path: Path) -> dict[int, float]:
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    try:
        found = {int(r['elements']): float(r['focus_pressure_arbitrary']) for r in rows}
        if 48 not in found or 60 not in found:
            raise ValueError('48 and 60 rows are required')
        if not (math.isfinite(found[48]) and math.isfinite(found[60]) and found[48] > 0 and found[60] > 0):
            raise ValueError('focus pressures must be positive and finite')
        return found
    except (KeyError, ValueError) as exc:
        raise SystemExit(f'Invalid comparison CSV: {exc}')


def budget(n: int, vrms: float, args: argparse.Namespace, duty: float | None = None,
           channel_mw: float | None = None) -> dict:
    """Very simple parallel capacitance+loss tangent+real motional branch model."""
    omega = 2 * math.pi * args.frequency_hz
    c_f = args.cap_pf * 1e-12
    # Dissipative part of capacitive current, NOT all capacitive reactive VA.
    dielectric_w = n * omega * c_f * vrms**2 * args.tan_delta
    # R_motion is a *parallel equivalent real load* representing mechanical
    # conversion and other motional real power. Not inferred from k-Wave.
    motional_w = n * vrms**2 / (args.r_motion_kohm * 1000)
    active_w = (dielectric_w + motional_w) / args.driver_eff
    use_duty = args.duty if duty is None else duty
    use_channel_mw = args.channel_mw if channel_mw is None else channel_mw
    standby_w = n * use_channel_mw / 1000 + args.fixed_mw / 1000
    mean_w = active_w * use_duty + standby_w
    reactive_va = n * omega * c_f * vrms**2
    return {
        'elements': n,
        'vrms_per_element': vrms,
        'dielectric_loss_w_while_on': dielectric_w,
        'motional_real_power_w_while_on': motional_w,
        'capacitive_reactive_VAr_while_on': reactive_va,
        'driver_input_w_while_on': active_w,
        'duty': use_duty,
        'channel_standby_mw_each': use_channel_mw,
        'standby_w': standby_w,
        'average_device_w': mean_w,
    }


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    ap = argparse.ArgumentParser(description='Illustrative 48 vs 60 FUS power budget; no k-Wave rerun')
    ap.add_argument('--comparison', help='Location of completed_comparison.csv')
    ap.add_argument('--out', default='fus_power_estimates', help='Output directory')
    ap.add_argument('--frequency-hz', type=float, default=500000)
    ap.add_argument('--vrms60', type=float, default=10, help='Illustrative per-element RMS voltage of 60-element array')
    ap.add_argument('--cap-pf', type=float, default=100, help='Assumed capacitance per element, pF')
    ap.add_argument('--tan-delta', type=float, default=0.02, help='Assumed dielectric loss tangent')
    ap.add_argument('--r-motion-kohm', type=float, default=20, help='Assumed real motional impedance in parallel, kOhm')
    ap.add_argument('--driver-eff', type=float, default=0.70, help='Assumed driver efficiency, 0-1')
    ap.add_argument('--duty', type=float, default=0.10, help='Assumed fraction of time actively transmitting, 0-1')
    ap.add_argument('--channel-mw', type=float, default=1.0, help='Assumed always-on electronics power per channel, mW')
    ap.add_argument('--fixed-mw', type=float, default=20, help='Assumed always-on shared electronics power, mW')
    args = ap.parse_args()

    for name in ['frequency_hz', 'vrms60', 'cap_pf', 'r_motion_kohm']:
        if getattr(args, name) <= 0 or not math.isfinite(getattr(args, name)):
            ap.error(f'--{name.replace("_", "-")} must be positive')
    for name in ['tan_delta', 'channel_mw', 'fixed_mw']:
        if getattr(args, name) < 0 or not math.isfinite(getattr(args, name)):
            ap.error(f'--{name.replace("_", "-")} must be nonnegative')
    if not 0 < args.driver_eff <= 1:
        ap.error('--driver-eff must be in (0,1]')
    if not 0 <= args.duty <= 1:
        ap.error('--duty must be in [0,1]')

    source = find_comparison(args.comparison)
    pressures = load_pressures(source)
    # The previous k-Wave runs used 1/sqrt(N) nominal source amplitude.
    # We assume additive-pressure source amplitude scales LINEARLY with
    # transducer voltage (this is not validated by k-Wave).
    # Per-element drive at same voltage: p_N * sqrt(N), up to common factor.
    v_ratio = (pressures[60] * math.sqrt(60)) / (pressures[48] * math.sqrt(48))
    power_ratio = 48 / 60 * v_ratio**2
    v48 = args.vrms60 * v_ratio
    fixedv_focus_ratio = 1 / v_ratio

    print(f'Read: {source}')
    print('48 vs 60 ELEMENT COMPARISON | SAME TARGET FOCAL PRESSURE')
    print(f'  k-Wave focus amplitudes (arbitrary, normalized by 1/sqrt(N)): '
          f'48={pressures[48]:.4f}, 60={pressures[60]:.4f}')
    print(f'  At the SAME per-element voltage, 48-element focal pressure = '
          f'{fixedv_focus_ratio:.4f}x the 60-element pressure')
    print(f'  To MATCH focal pressure: V48/V60 = {v_ratio:.4f}x')
    print(f'  With identical per-element real conductance: '
          f'total ACTIVE LOAD POWER 48/60 = {power_ratio:.4f}x '
          f'({(power_ratio-1)*100:+.1f}%)')
    print('\nILLUSTRATIVE BUDGET (NOT MEASURED, NO VALIDATED ELECTRICAL MODEL)')
    print(f'  Assumptions: 500 kHz; C={args.cap_pf:g} pF/element; '
          f'tan(delta)={args.tan_delta:g}; parallel real load={args.r_motion_kohm:g} kOhm; '
          f'driver eff={args.driver_eff:g}; duty={args.duty:g}; '
          f'{args.channel_mw:g} mW/channel idle; {args.fixed_mw:g} mW shared')
    budgets = [budget(48, v48, args), budget(60, args.vrms60, args)]
    for b in budgets:
        print(f'  N={b["elements"]:2d}: V_rms={b["vrms_per_element"]:6.2f} V; '
              f'active real load={1000*(b["dielectric_loss_w_while_on"] + b["motional_real_power_w_while_on"]):7.1f} mW; '
              f'average budget={1000*b["average_device_w"]:7.1f} mW; '
              f'capacitive reactive={b["capacitive_reactive_VAr_while_on"]:.3f} VAr')
    print(f'  Delta average (N48-N60) = '
          f'{1000*(budgets[0]["average_device_w"]-budgets[1]["average_device_w"]):+.2f} mW')

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'power_budget_example.csv', budgets)
    sensitivity = []
    for duty in [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]:
        for channel_mw in [0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]:
            b48 = budget(48, v48, args, duty=duty, channel_mw=channel_mw)
            b60 = budget(60, args.vrms60, args, duty=duty, channel_mw=channel_mw)
            sensitivity.append({
                'duty': duty,
                'standby_per_channel_mw': channel_mw,
                'avg_48_mw': 1000*b48['average_device_w'],
                'avg_60_mw': 1000*b60['average_device_w'],
                '48_minus_60_mw': 1000*(b48['average_device_w']-b60['average_device_w']),
                'less_average_power': 48 if b48['average_device_w'] < b60['average_device_w'] else (60 if b48['average_device_w'] > b60['average_device_w'] else 'tie'),
            })
    write_csv(out / 'duty_and_channel_sensitivity.csv', sensitivity)
    with (out / 'model_assumptions.txt').open('w') as f:
        f.write('This is a parameterized ILLUSTRATION, not an independently validated electrical power simulation.\n'
                'The k-Wave model uses additive-pressure voxels, not a calibrated piezoelectric transfer function.\n'
                'Source drive used 1/sqrt(N). We assume pressure proportional to drive voltage for both arrays.\n'
                'Same frequency, per-element C, tan(delta), parallel real motional conductance, and driver efficiency.\n'
                'Dielectric loss = N * 2*pi*f*C*Vrms**2*tan(delta).\n'
                'Real motional power = N*Vrms**2/R_parallel_real.\n'
                'Reactive VAr = N*2*pi*f*C*Vrms**2; it is NOT automatically consumed real power.\n'
                'Average device budget = duty * (dielectric + motional)/driver_efficiency\n'
                '                      + N*always_on_channel_mW + shared_always_on_mW.\n'
                'This omits coupling, skull attenuation, coil/PCB parasitics, charge recovery, switching topology,\n'
                'phase-dependent coupling, nonlinear behavior, thermal constraints, and sensing power.\n'
                f'Source CSV: {source}\n')
    print(f'\nSaved: {out}/power_budget_example.csv')
    print(f'       {out}/duty_and_channel_sensitivity.csv')
    print(f'       {out}/model_assumptions.txt')


if __name__ == '__main__':
    main()
