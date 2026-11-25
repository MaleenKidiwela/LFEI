#!/usr/bin/env python3
"""Stage 8b -- RUN WITH THE 'rf' CONDA ENV (~/anaconda3/envs/rf/bin/python; PyRaysum 1.0.0).
PyRaysum dipping-layer modelling of the receiver-function spectral power, after Herath & Audet (2024).
Inputs: data/sci/rf_timelapse/raysum/input_{STA}.json (sci_rf_raysum_prep.py, lfe env) and rf_{STA}.npz.
Per station:
  - dipping layered model (Bloch et al. 2023 depths / dips; overriding plate, 2-km layer above the slab
    low-velocity layer (LVL), LVL, oceanic crust, mantle), synthetic seismograms (PyRaysum, all first-order
    free-surface multiples, 80 Hz) for the observed back-azimuth / slowness of every power-set event, deconvolved
    here (R/Z, T/Z, water level 1e-3) and decimated to 20 Hz
  - synthetics processed like the data: cosine low-pass 4-6 Hz, multitaper PSD in 1/8-octave bands, 1.25-3 Hz
    total power in the station's slab window (and crustal window)
  - geometry-only synthetic power series: the same 1.5-yr windows / events / back-azimuth reweighting as the data
    with an unchanged model -> how much apparent power variation the changing event geometry produces
  - sensitivity: Vs of the LVL and of the 2-km layer above it perturbed by -10..+10 % at fixed Vp (Vp/Vs changes),
    change of the all-event slab-window power (dB), diluted by the station's noise
    (dB_obs = 10 log10(1 + dP/P * signal_fraction)); slope from +-2 %
  - fit of the observed power as in Herath & Audet: dVs(t) = (observed dB(t) - geometry-only synthetic dB(t)) /
    slope, 1-sigma from the event-permutation null envelope; step / long-term contrasts converted the same way;
    2-sigma minimum detectable Vs change = 2 null SD / |slope|
-> raysum/result_{STA}.json, model_raysum.csv, figures/sci_rf_power_model_{STA}.png, figures/sci_rf_power_model.png
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/seismic/LFE'
OUT = os.environ.get('RF_OUT', f'{ROOT}/data/sci/rf_timelapse')
FIG = os.environ.get('RF_FIG', f'{ROOT}/figures')
RDIR = os.path.join(OUT, 'raysum')
DT = 0.05
NPTS = 800                      # 40 s; P at sample NPTS/2 in PyRaysum RFs
FB = 0.25 * 2 ** (np.arange(0, 8 * np.log2(5 / 0.25) + 1) / 8)
YR = 365.25
WIN_LEN = 1.5 * YR
PERT = np.array([-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10])
KINDS = {'lvl': (2, 'slab low-velocity layer'), 'above': (1, '2-km layer above the low-velocity layer')}


def mt_psd(x, dt, nw=2.0, k=3, nfft=512):
    from scipy.signal.windows import dpss
    tap = dpss(len(x), nw, k)
    X = np.fft.rfft((x - x.mean())[None, :] * tap, nfft, axis=1)
    return np.fft.rfftfreq(nfft, dt), 2 * dt * np.mean(np.abs(X) ** 2, 0)


def band_power(x, t, win, band):
    w = (t >= win[0]) & (t <= win[1])
    f, p = mt_psd(x[w], DT)
    b = np.array([p[(f >= c * 2 ** (-1 / 16)) & (f < c * 2 ** (1 / 16))].mean()
                  if np.any((f >= c * 2 ** (-1 / 16)) & (f < c * 2 ** (1 / 16))) else np.interp(c, f, p) for c in FB])
    return b[(FB >= band[0]) & (FB <= band[1])].sum()


def lowpass(x):
    nf = 2 * len(x)
    f = np.fft.rfftfreq(nf, DT)
    lp = np.clip((6.0 - f) / 2.0, 0, 1); lp = 0.5 - 0.5 * np.cos(np.pi * lp)
    return np.fft.irfft(np.fft.rfft(x, nf) * lp, nf)[:len(x)]


def build_model(layers, kind=None, d=0.0):
    from pyraysum import Model
    L = [dict(l) for l in layers]
    if kind:
        L[KINDS[kind][0]]['vs'] *= (1 + d)
    return Model([l['thick_km'] * 1000 for l in L], [l['rho'] for l in L], [l['vp'] * 1000 for l in L],
                 [l['vs'] * 1000 for l in L], strike=[l['strike'] for l in L], dip=[l['dip'] for l in L])


def synth_rfs(model, baz, p, dtf=0.0125):
    """PyRaysum synthetic seismograms (R, T, Z; all first-order multiples) at a fine sampling interval (so that
    arrival times are not quantised to the 20-Hz grid), deconvolved here by Z with a 1e-3 water level, low-passed
    (cosine 4-6 Hz, as the data) and decimated to 20 Hz; P at sample NPTS/2.  Radial sign flipped to the data
    convention (PyRaysum radial is positive towards the source)."""
    from pyraysum import prs, Geometry, Control
    n = int(round(NPTS * DT / dtf))
    rc = Control(npts=n, dt=dtf, mults=2, rot=1, verbose=0, shift=1.0)   # shift=None dropped the direct P for dipping models (fixed 2026-11-25)
    nf = 2 * n
    f = np.fft.rfftfreq(nf, dtf)
    lp = np.clip((6.0 - f) / 2.0, 0, 1); lp = 0.5 - 0.5 * np.cos(np.pi * lp)
    step = int(round(DT / dtf))
    idx = (np.arange(-NPTS // 2, NPTS // 2) * step) % nf
    R = np.zeros((len(baz), NPTS)); T = np.zeros((len(baz), NPTS))
    for i0 in range(0, len(baz), 400):
        res = prs.run(model, Geometry(list(baz[i0:i0 + 400]), list(p[i0:i0 + 400])), rc, rf=False, mode='bare')
        for j, st in enumerate(res.streams):
            Z = np.fft.rfft(st[2].data, nf)
            den = np.abs(Z) ** 2
            den = den + 1e-3 * den.max()
            rz = np.fft.irfft(lp * Z * np.conj(Z) / den, nf)
            k0 = int(np.argmax(np.abs(rz))); nrm = rz[k0]
            for X, comp in ((R, 0), (T, 1)):
                y = np.fft.irfft(lp * np.fft.rfft(st[comp].data, nf) * np.conj(Z) / den, nf)
                X[i0 + j] = np.roll(y, -k0)[idx] / nrm
    return -R, T


def event_power(R, T, win, band):
    t = np.arange(NPTS) * DT - NPTS / 2 * DT
    out = np.zeros((len(R), 2, 2))           # event, window (slab, crust), comp
    for i in range(len(R)):
        for j, w in enumerate(['slab', 'crust']):
            out[i, j, 0] = band_power(R[i], t, win[w], band)
            out[i, j, 1] = band_power(T[i], t, win[w], band)
    return out


def series(Pev, ev, tmid):
    """geometry-reweighted 1.5-yr window power (dB re all-event mean), as sci_rf_power.PStation.series"""
    tt = np.array(ev['time']); geo = np.array(ev['geo'])
    ref = np.bincount(geo, minlength=24) / len(geo)
    allP = Pev.mean(0)
    out = np.full((len(tmid),) + Pev.shape[1:], np.nan)
    for i, c in enumerate(tmid):
        k = np.where(np.abs(tt - c) <= WIN_LEN / 2)[0]
        if len(k) < 8:
            continue
        g = geo[k]
        if np.sum(ref[np.unique(g)]) < 0.5:
            continue
        cnt = np.bincount(g, minlength=24)
        w = ref[g] / cnt[g]
        out[i] = np.tensordot(w / w.sum(), Pev[k], axes=(0, 0))
    return 10 * np.log10(out / allP[None])


def dilute(ratio, f):
    return 10 * np.log10(np.maximum(1 + (ratio - 1) * f, 1e-6))


def run_station(fn):
    I = json.load(open(fn))
    code = I['sta']
    ev = I['events']
    baz, p = np.array(ev['baz']), np.array(ev['p'])
    win = {k: tuple(v) for k, v in I['windows'].items()}
    band = I['band']
    f_sig = I['signal_fraction']
    tmid = np.array(I['series']['tmid'])
    R0, T0 = synth_rfs(build_model(I['layers']), baz, p)
    P0 = event_power(R0, T0, win, band)
    geo_series = series(P0, ev, tmid)                       # (nwin, window, comp), noise-free dB
    sens = {}
    for kind in KINDS:
        curve = []
        for d in PERT:
            if d == 0:
                curve.append(np.zeros((2, 2))); continue
            R, T = synth_rfs(build_model(I['layers'], kind, d), baz, p)
            Pd = event_power(R, T, win, band)
            ratio = Pd.mean(0) / P0.mean(0)                  # (window, comp)
            curve.append(np.array([[dilute(ratio[w, c], f_sig[c]) for c in range(2)] for w in range(2)]))
        sens[kind] = np.array(curve)                          # (npert, window, comp)
    # slopes (dB per % Vs) from +-2 %
    i2m, i2p = list(PERT).index(-0.02), list(PERT).index(0.02)
    slope = {k: (sens[k][i2p] - sens[k][i2m]) / 4.0 for k in KINDS}   # (window, comp)
    # fit of the observed series and contrasts (slab window)
    S = I['series']
    res = dict(sta=code, model_src=I['model_src'], layers=I['layers'], windows=I['windows'],
               signal_fraction=f_sig, pert=PERT.tolist(),
               sens={k: v.tolist() for k, v in sens.items()}, slope={k: v.tolist() for k, v in slope.items()},
               tmid=tmid.tolist(), geo_series=geo_series.tolist())
    rows = []
    fits = {}
    for c, cn in enumerate(['R', 'T']):
        obs = np.array(S[f'slab_{cn}_dB'], float)
        lo, hi = np.array(S[f'slab_{cn}_null_lo'], float), np.array(S[f'slab_{cn}_null_hi'], float)
        sig = (hi - lo) / 3.92
        geo_d = dilute(10 ** (geo_series[:, 0, c] / 10), f_sig[c])
        for kind in KINDS:
            s = slope[kind][0, c]
            dvs = (obs - geo_d) / s if abs(s) > 1e-6 else np.full(len(obs), np.nan)
            fits[(kind, cn)] = (dvs, np.abs(sig / s) if abs(s) > 1e-6 else np.full(len(obs), np.nan))
        for r in I['contrasts']:
            if r['comp'] != cn:
                continue
            for kind in KINDS:
                s = slope[kind][0, c]
                rows.append(dict(sta=code, comp=cn, contrast=r['contrast'], layer=kind, est_dB=r['est_dB'],
                                 null_sd_dB=r['null_sd'], slope_dB_per_pct=s,
                                 dVs_pct=r['est_dB'] / s if abs(s) > 1e-6 else np.nan,
                                 dVs_sd_pct=r['null_sd'] / abs(s) if abs(s) > 1e-6 else np.nan,
                                 min_detect_dVs_2sigma_pct=2 * r['null_sd'] / abs(s) if abs(s) > 1e-6 else np.nan,
                                 geometry_series_range_dB=float(np.nanmax(geo_d) - np.nanmin(geo_d)),
                                 signal_fraction=f_sig[c], model_src=I['model_src'],
                                 dip_t=I['layers'][1]['dip'], strike_t=I['layers'][1]['strike']))
    res['fits'] = {f'{k}_{c}': [v[0].tolist(), v[1].tolist()] for (k, c), v in fits.items()}
    # mean synthetic vs observed RF (shape comparison)
    d = np.load(os.path.join(OUT, f'rf_{code}.npz'))
    meta = pd.DataFrame(json.loads(str(d['meta'])))
    tt = np.array(ev['time'])
    tobs = pd.to_datetime(meta.time).values.astype('datetime64[s]').astype(np.int64) / 86400.0
    sel = np.isin(np.round(tobs, 4), np.round(tt, 4))
    res['obs_mean_R'] = d['rfw'][sel, 0].mean(0).tolist(); res['obs_mean_T'] = d['rfw'][sel, 1].mean(0).tolist()
    res['obs_t'] = d['t'].tolist()
    res['syn_mean_R'] = R0.mean(0).tolist(); res['syn_mean_T'] = T0.mean(0).tolist()
    json.dump(res, open(os.path.join(RDIR, f'result_{code}.json'), 'w'), default=float)
    figure(res, I)
    print(code, 'slopes LVL R/T', np.round(slope['lvl'][0], 3), 'above R/T', np.round(slope['above'][0], 3), flush=True)
    return rows


def figure(res, I):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    code = res['sta']
    fig = plt.figure(figsize=(17, 11))
    ax = fig.add_axes([0.05, 0.57, 0.27, 0.36])
    ts = np.arange(NPTS) * DT - NPTS / 2 * DT
    to = np.array(res['obs_t'])
    for k, (cn, off) in enumerate([('R', 0), ('T', -0.4)]):
        o = np.array(res[f'obs_mean_{cn}']); s = np.array(res[f'syn_mean_{cn}'])
        w = (to >= 0.3) & (to <= 8)
        so = np.interp(to, ts, s)
        sc = np.dot(o[w], so[w]) / max(np.dot(so[w], so[w]), 1e-12)
        ax.plot(to, off + o, 'k', lw=1, label='observed mean (modified Wiener)' if k == 0 else None)
        ax.plot(to, off + sc * so, 'C3', lw=1, label='PyRaysum synthetic mean (scaled)' if k == 0 else None)
        ax.text(9.5, off + 0.05, 'radial' if cn == 'R' else 'transverse', fontsize=8, ha='right')
    ax.axvspan(*I['windows']['slab'], color='C0', alpha=0.15)
    ax.set_xlim(-2, 10); ax.set_xlabel('time after P (s)'); ax.legend(fontsize=7)
    ax.set_title('Mean RF, observed vs dipping-layer synthetic\n(blue: slab window)', fontsize=9)
    ax = fig.add_axes([0.38, 0.57, 0.25, 0.36])
    L = I['layers']; z = 0
    for l in L[:-1]:
        ax.barh(-(z + l['thick_km'] / 2), l['vs'], height=l['thick_km'], color='0.8', edgecolor='k')
        ax.text(l['vs'] + 0.05, -(z + l['thick_km'] / 2), f"{l['name']}: Vs {l['vs']:.2f}, Vp/Vs {l['vp'] / l['vs']:.2f}, "
                f"dip {l['dip']:.0f} deg", fontsize=7, va='center')
        z += l['thick_km']
    ax.set_xlim(0, 9); ax.set_xlabel('Vs (km/s)'); ax.set_ylabel('depth below station (km)')
    ax.set_title(f"Model ({res['model_src']})", fontsize=9)
    ax = fig.add_axes([0.70, 0.57, 0.27, 0.36])
    for kind, col in [('lvl', 'C0'), ('above', 'C1')]:
        s = np.array(res['sens'][kind])
        for c, ls in [(0, '-'), (1, '--')]:
            ax.plot(100 * np.array(res['pert']), s[:, 0, c], color=col, ls=ls,
                    label=f"{KINDS[kind][1]}, {'radial' if c == 0 else 'transverse'}")
    ax.axhline(0, color='k', lw=0.5); ax.set_xlabel('Vs change at fixed Vp (%)')
    ax.set_ylabel('predicted slab-window power change (dB, noise-diluted)'); ax.legend(fontsize=7)
    ax.set_title('Sensitivity of 1.25-3 Hz slab-window power', fontsize=9)
    tm = pd.to_datetime(np.array(res['tmid']) * 86400e9)
    S = I['series']
    ax = fig.add_axes([0.05, 0.31, 0.92, 0.19])
    ax.fill_between(tm, S['slab_R_null_lo'], S['slab_R_null_hi'], color='k', alpha=0.12, lw=0)
    ax.plot(tm, S['slab_R_dB'], 'k', lw=1.5, label='observed radial (shading: 95% permutation range)')
    g = np.array(res['geo_series'])
    ax.plot(tm, dilute(10 ** (g[:, 0, 0] / 10), res['signal_fraction'][0]), 'C3', lw=1.2,
            label='synthetic radial, unchanged model (geometry-only variation)')
    ax.plot(tm, dilute(10 ** (g[:, 0, 1] / 10), res['signal_fraction'][1]), 'C3', lw=0.8, ls='--', label='synthetic transverse, unchanged model')
    for q, col in [('2021-12-20', 'darkorange'), ('2022-12-20', 'm'), ('2024-12-05', 'r')]:
        ax.axvline(pd.Timestamp(q), color=col, ls='--', lw=1)
    ax.set_xlim(pd.Timestamp('2008-01-01'), pd.Timestamp('2026-12-31'))
    ax.axhline(0, color='k', lw=0.5); ax.set_ylabel('slab-window power (dB)'); ax.legend(fontsize=7, loc='lower left')
    ax.set_title('1.5-yr windows (95% overlap), back-azimuth reweighted', fontsize=9)
    ax = fig.add_axes([0.05, 0.05, 0.92, 0.19])
    for kind, col in [('lvl', 'C0'), ('above', 'C1')]:
        dv, sd = (np.array(x) for x in res['fits'][f'{kind}_R'])
        ax.fill_between(tm, dv - 2 * sd, dv + 2 * sd, color=col, alpha=0.15, lw=0)
        ax.plot(tm, dv, col, lw=1.2, label=f'Vs change of the {KINDS[kind][1]} needed (radial; band: 2 sigma)')
    for q, col in [('2021-12-20', 'darkorange'), ('2022-12-20', 'm'), ('2024-12-05', 'r')]:
        ax.axvline(pd.Timestamp(q), color=col, ls='--', lw=1)
    ax.set_xlim(pd.Timestamp('2008-01-01'), pd.Timestamp('2026-12-31')); ax.set_ylim(-30, 30)
    ax.axhline(0, color='k', lw=0.5); ax.set_ylabel('Vs change (%)'); ax.legend(fontsize=7, loc='lower left')
    ax.set_title('Observed power changes expressed as Vs perturbations (Herath & Audet-style fit; one layer at a time)', fontsize=9)
    fig.suptitle(f'{code}: PyRaysum dipping-layer modelling of receiver-function power (dashed lines: Petrolia, Ferndale, M7)', fontsize=12)
    fig.savefig(os.path.join(FIG, f'sci_rf_power_model_{code.split(".")[1]}.png'), dpi=100)
    plt.close(fig)


def overview(D):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    stas = list(dict.fromkeys(D.sta))
    fig, axs = plt.subplots(1, 3, figsize=(18, max(6, 0.35 * len(stas) + 2)), sharey=True)
    y = np.arange(len(stas))
    for ax, (q, lab) in zip(axs, [('slope_dB_per_pct', 'sensitivity (dB per % Vs), radial'),
                                  ('min_detect_dVs_2sigma_pct', 'minimum detectable Vs change, 2 sigma (%),\n'
                                                                 'after vs before 2021-12-20, radial'),
                                  ('dVs_pct', 'Vs change, after vs before 2021-12-20 (%, ±2 sigma), radial')]):
        for k, (kind, col) in enumerate([('lvl', 'C0'), ('above', 'C1')]):
            d = D[(D.contrast == 'longterm_post2021') & (D.comp == 'R') & (D.layer == kind)].set_index('sta').reindex(stas)
            if q == 'dVs_pct':
                ax.errorbar(d[q], y + 0.15 * (k - 0.5), xerr=2 * d.dVs_sd_pct, fmt='o', color=col, label=KINDS[kind][1])
                ax.set_xlim(-40, 40)
            else:
                ax.plot(d[q].abs() if q.startswith('min') else d[q], y + 0.15 * (k - 0.5), 'o', color=col, label=KINDS[kind][1])
        ax.axvline(0, color='k', lw=0.5); ax.set_xlabel(lab, fontsize=9)
        if q.startswith('min'):
            ax.set_xscale('log')
    axs[0].set_yticks(y); axs[0].set_yticklabels(stas, fontsize=8); axs[0].invert_yaxis(); axs[0].legend(fontsize=8)
    fig.suptitle('PyRaysum dipping-layer modelling of receiver-function power: sensitivity, detection limits and inferred Vs changes',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(FIG, 'sci_rf_power_model.png'), dpi=100)


def main():
    from concurrent.futures import ProcessPoolExecutor
    files = sorted(glob.glob(os.path.join(RDIR, 'input_*.json')))
    if len(sys.argv) > 1:
        files = [f for f in files if any(s in f for s in sys.argv[1].split(','))]
    rows = []
    with ProcessPoolExecutor(4) as ex:
        for r in ex.map(run_station, files):
            rows += r
    D = pd.DataFrame(rows)
    fn = os.path.join(OUT, 'model_raysum.csv')
    if len(sys.argv) > 1 and os.path.exists(fn):
        old = pd.read_csv(fn)
        D = pd.concat([old[~old.sta.isin(D.sta)], D], ignore_index=True)
    D.to_csv(fn, index=False)
    overview(D)
    print(D[D.comp == 'R'].round(3).to_string())


if __name__ == '__main__':
    main()
