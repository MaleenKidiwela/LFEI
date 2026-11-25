#!/usr/bin/env python3
"""Stage 6: collect receiver-function time-lapse results -> results_all.csv, station_table.csv, summary.json,
figures/sci_rf_timelapse_overview.png.
Network combination: inverse-variance weighted mean over stations of each contrast/quantity (weights 1/null_sd^2),
z against the combined null SD (stations treated as independent).
Expected signal from the shallow dv/v decline: a uniform Vs change dv/v in the top H km delays a conversion from
depth H by dt = -dv/v * t_Ps(H) and changes the radial/vertical direct-P amplitude by ~ +dv/v (Svenningsen &
Jacobsen 2007 apparent-velocity relation, d ln Vs_app / d ln R0 ~ 0.9-1).
"""
import io
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sci_rf_common import OUT, FIG, STATIONS, QUAKES, coords  # noqa: E402

KEYQ = ['P0_amp', 'Ppol_deg', 'sh_dt', 'sh_amp', 'sh_a5_dt', 'w1_dt', 'w2_dt', 'w2_amp', 'w3_dt', 'w3_amp', 'stretch_eps']
CONS = ['step_Petrolia', 'step_Ferndale', 'step_M7', 'longterm_post2021', 'trend_per_yr']
QL = {'P0_amp': 'direct-P amplitude (%)', 'Ppol_deg': 'direct-P polarisation (deg)', 'sh_dt': 'shallow 0.15-1 s delay (ms)',
      'sh_amp': 'shallow 0.15-1 s amplitude (%)', 'sh_a5_dt': 'shallow delay, a=5 (ms)', 'w1_dt': '1-2.5 s delay (ms)',
      'w2_dt': '2.5-4 s delay (ms)', 'w2_amp': '2.5-4 s amplitude (%)', 'w3_dt': '4-6 s delay (ms)',
      'w3_amp': '4-6 s amplitude (%)', 'stretch_eps': 'stretch 0.15-6 s (%)'}
CL = {'step_Petrolia': 'Petrolia step', 'step_Ferndale': 'Ferndale step', 'step_M7': 'M7 step',
      'longterm_post2021': 'after vs before 2021-12-20', 'trend_per_yr': 'trend per year'}


def expected_signal(dvv=-0.2e-2, H=0.8, vs=1.5, vp=3.0):
    t = H * (1 / vs - 1 / vp)
    return dict(dvv_pct=100 * dvv, depth_km=H, tPs_s=t, delay_ms=-1000 * dvv * t, p0_amp_pct=100 * dvv)


def main():
    files = sorted(glob.glob(os.path.join(OUT, 'timelapse_*.csv')))
    T = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    T.to_csv(os.path.join(OUT, 'results_all.csv'), index=False)
    Q = pd.read_csv(os.path.join(OUT, 'qc_all.csv'))
    P = pd.read_csv(os.path.join(OUT, 'static_picks.csv')) if os.path.exists(os.path.join(OUT, 'static_picks.csv')) else None
    C = coords()
    stas = [s for s in STATIONS if s in set(T.sta)]
    # station table
    rows = []
    for s in stas:
        q = Q[Q.sta == s]
        d = dict(sta=s, depth_m=C[s][3], n_windows=len(q), n_rf=int((q.status.isin(['ok', 'lowsnr'])).sum()),
                 n_deconvolved=int((q.status == 'ok').sum()))
        fn = os.path.join(OUT, f'rf_{s}.npz')
        z = np.load(fn)
        meta = pd.read_json(io.StringIO(str(z['meta'])))
        from sci_rf_common import load_rf
        m, *_ = load_rf(s)
        d['n_pass'] = len(m)
        d['first'] = str(pd.Timestamp(m.time.min()).date()) if len(m) else ''
        d['last'] = str(pd.Timestamp(m.time.max()).date()) if len(m) else ''
        for k, v in QUAKES.items():
            d[f'n_post_{k}'] = int((m.time >= pd.Timestamp(v)).sum())
        if P is not None:
            for r in P[P.sta == s].itertuples():
                d[f'pick_{r.phase}_t'] = r.t_obs; d[f'pick_{r.phase}_se'] = r.t_se; d[f'pick_{r.phase}_amp'] = r.amp
                if r.phase in 'tcm':
                    d[f'pred_{r.phase}_t'] = r.t_pred; d[f'model_{r.phase}_z'] = r.z_model
        tt = T[(T.sta == s)]
        for c in ['step_M7', 'longterm_post2021']:
            for qq in ['P0_amp', 'sh_dt', 'sh_amp', 'w2_dt', 'w2_amp', 'stretch_eps', 'Ppol_deg']:
                r = tt[(tt.contrast == c) & (tt.quantity == qq)]
                if len(r):
                    d[f'{c}_{qq}_est'] = r.est.iloc[0]; d[f'{c}_{qq}_det2s'] = r.det2sigma.iloc[0]; d[f'{c}_{qq}_z'] = r.z.iloc[0]
        rows.append(d)
    ST = pd.DataFrame(rows)
    ST.to_csv(os.path.join(OUT, 'station_table.csv'), index=False)
    # network combination
    net = []
    for c in CONS:
        for q in KEYQ:
            r = T[(T.contrast == c) & (T.quantity == q) & np.isfinite(T.est) & (T.null_sd > 0)]
            if len(r) < 2:
                continue
            w = 1 / r.null_sd ** 2
            mu = np.sum(w * r.est) / w.sum(); sd = 1 / np.sqrt(w.sum())
            net.append(dict(contrast=c, quantity=q, n_sta=len(r), est=mu, sd=sd, z=mu / sd,
                            n_abs_z_gt2=int((r.z.abs() > 2).sum()), chi2=float(np.sum((r.est / r.null_sd) ** 2)),
                            det2sigma=2 * sd))
    NET = pd.DataFrame(net)
    NET.to_csv(os.path.join(OUT, 'network_combined.csv'), index=False)
    # multiple testing overview
    sub = T[T.contrast.isin(CONS) & T.quantity.isin(KEYQ) & np.isfinite(T.z)]
    frac2 = float((sub.z.abs() > 2).mean()); frac3 = float((sub.z.abs() > 3).mean())
    # injection recovery (median over stations) for M7 / long-term
    inj = {}
    for c in ['step_M7', 'longterm_post2021']:
        for q in KEYQ:
            r = T[(T.contrast == c) & (T.quantity == q)]
            if len(r) and 'inj_delay20ms' in r:
                inj[f'{c}:{q}'] = dict(delay20ms=float(r['inj_delay20ms'].median()), amp10pct=float(r['inj_amp10pct'].median()),
                                       stretch0p5pct=float(r['inj_stretch0.5pct'].median()))
    S = dict(stations=ST.to_dict(orient='records'), network=NET.to_dict(orient='records'),
             frac_abs_z_gt2=frac2, frac_abs_z_gt3=frac3, n_tests=len(sub),
             strong=sub[sub.z.abs() > 3][['sta', 'contrast', 'quantity', 'est', 'det2sigma', 'z']].to_dict(orient='records'),
             injection_median=inj, expected_shallow_dvv=expected_signal(),
             qc=dict(snr_z=3.0, fit_pct=80, note='see sci_rf_common.QC'))
    # spectral-power (Herath & Audet) results, PyRaysum modelling and RfPy cross-check, if present
    for key, fn in [('power_contrasts', 'power_contrasts_all.csv'), ('power_corr', 'power_corr.csv'),
                    ('model_raysum', 'model_raysum.csv'), ('model_flat', 'model_sensitivity_flat.csv')]:
        f = os.path.join(OUT, fn)
        if os.path.exists(f):
            d = pd.read_csv(f)
            if key == 'power_contrasts':
                d = d[d.window.isin(['slab', 'noise'])]
            if key == 'model_raysum':
                d = d[d.contrast.isin(['longterm_post2021', 'step_M7'])]
            S[key] = d.to_dict(orient='records')
    for f in glob.glob(os.path.join(OUT, 'rfpy_check_*.csv')):
        d = pd.read_csv(f)
        S.setdefault('rfpy_check', {})[os.path.basename(f)[11:-4]] = dict(
            n=len(d), median_event_cc=float(d.cc_0_8s.median()), median_power_offset_dB=float(d.power_dB_rfpy_minus_ours.median()),
            sd_power_offset_dB=float(d.power_dB_rfpy_minus_ours.std()))
    json.dump(S, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1, default=float)
    figure(T, ST, NET, Q)
    print(ST.to_string())
    print(NET[NET.contrast.isin(['step_M7', 'longterm_post2021', 'trend_per_yr'])].round(3).to_string())
    print('frac |z|>2', frac2, '|z|>3', frac3, 'n', len(sub))


def figure(T, ST, NET, Q):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sci_rf_common import load_rf
    stas = list(ST.sta)
    fig = plt.figure(figsize=(19, 14))
    qcol = {'Petrolia': 'darkorange', 'Ferndale': 'm', 'M7': 'r'}
    ax = fig.add_axes([0.06, 0.58, 0.36, 0.36])
    for i, s in enumerate(stas):
        m, *_ = load_rf(s)
        ax.plot(pd.to_datetime(m.time), np.full(len(m), i), '|', ms=6, color='k', alpha=0.6)
        ax.text(pd.Timestamp('2026-12-15'), i, str(len(m)), fontsize=7, va='center')
    for k, v in QUAKES.items():
        ax.axvline(pd.Timestamp(v), color=qcol[k], ls='--', lw=1)
    ax.set_yticks(range(len(stas))); ax.set_yticklabels(stas, fontsize=8); ax.invert_yaxis()
    ax.set_xlim(pd.Timestamp('2008-01-01'), pd.Timestamp('2027-06-01'))
    ax.set_title('Accepted receiver functions per station (count at right);\n'
                 'dashed: Petrolia (orange), Ferndale (magenta), M7 (red) earthquakes', fontsize=10)
    # detection limits
    ax = fig.add_axes([0.50, 0.58, 0.22, 0.36])
    for j, (q, c, mk) in enumerate([('sh_dt', 'step_M7', 'o'), ('sh_dt', 'longterm_post2021', 's'),
                                    ('w2_dt', 'longterm_post2021', '^')]):
        v = [T[(T.sta == s) & (T.contrast == c) & (T.quantity == q)].det2sigma.mean() for s in stas]
        ax.plot(v, np.arange(len(stas)) + 0.15 * (j - 1), mk, label=f'{QL[q]}, {CL[c]}')
    ax.axvline(20, color='0.5', ls=':', lw=1)
    ax.set_xscale('log'); ax.set_yticks(range(len(stas))); ax.set_yticklabels(stas, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel('2-sigma detection limit, delay (ms)'); ax.legend(fontsize=7, loc='lower right')
    ax.set_title('Timing detection limits (dotted: 20 ms injection)', fontsize=10)
    ax = fig.add_axes([0.77, 0.58, 0.21, 0.36])
    for j, (q, c, mk) in enumerate([('P0_amp', 'step_M7', 'o'), ('P0_amp', 'longterm_post2021', 's'),
                                    ('w2_amp', 'longterm_post2021', '^')]):
        v = [T[(T.sta == s) & (T.contrast == c) & (T.quantity == q)].det2sigma.mean() for s in stas]
        ax.plot(v, np.arange(len(stas)) + 0.15 * (j - 1), mk, label=f'{QL[q]}, {CL[c]}')
    ax.axvline(10, color='0.5', ls=':', lw=1)
    ax.set_xscale('log'); ax.set_yticks(range(len(stas))); ax.set_yticklabels([]); ax.invert_yaxis()
    ax.set_xlabel('2-sigma detection limit, amplitude (%)'); ax.legend(fontsize=7, loc='lower right')
    ax.set_title('Amplitude detection limits (dotted: 10% injection)', fontsize=10)
    # z heatmap: stations x (contrast, quantity)
    ax = fig.add_axes([0.10, 0.06, 0.80, 0.42])
    cols = [(c, q) for q in ['P0_amp', 'Ppol_deg', 'sh_dt', 'sh_amp', 'w2_dt', 'w2_amp', 'stretch_eps'] for c in CONS]
    Z = np.full((len(stas) + 1, len(cols)), np.nan)
    for j, (c, q) in enumerate(cols):
        for i, s in enumerate(stas):
            r = T[(T.sta == s) & (T.contrast == c) & (T.quantity == q)]
            if len(r):
                Z[i, j] = r.z.iloc[0]
        r = NET[(NET.contrast == c) & (NET.quantity == q)]
        if len(r):
            Z[-1, j] = r.z.iloc[0]
    im = ax.imshow(Z, cmap='RdBu_r', vmin=-4, vmax=4, aspect='auto')
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if np.isfinite(Z[i, j]) and abs(Z[i, j]) >= 2:
                ax.text(j, i, f'{Z[i, j]:.1f}', ha='center', va='center', fontsize=6)
    ax.set_yticks(range(len(stas) + 1)); ax.set_yticklabels(stas + ['network (weighted mean)'], fontsize=8)
    ax.axhline(len(stas) - 0.5, color='k', lw=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([CL[c] for c, q in cols], rotation=90, fontsize=7)
    for k, q in enumerate(['P0_amp', 'Ppol_deg', 'sh_dt', 'sh_amp', 'w2_dt', 'w2_amp', 'stretch_eps']):
        ax.axvline(k * len(CONS) - 0.5, color='k', lw=1)
        ax.text(k * len(CONS) + len(CONS) / 2 - 0.5, -1.0, QL[q], ha='center', fontsize=8)
    plt.colorbar(im, ax=ax, pad=0.01, fraction=0.03).set_label('z = estimate / null SD (|z|>=2 printed)')
    fig.text(0.5, 0.505, 'Null-calibrated z-scores of receiver-function changes (steps at the three earthquakes, '
             'after vs before 2021-12-20, linear trend)', ha='center', fontsize=10)
    fig.suptitle('Time-lapse teleseismic P receiver functions near the Mendocino triple junction, 2008-2026: '
                 'data coverage, detection limits and null-calibrated changes', fontsize=12)
    fig.savefig(os.path.join(FIG, 'sci_rf_timelapse_overview.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    main()
