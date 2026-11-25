#!/usr/bin/env python3
"""EQ-above figures and summary.json.
figures/sci_eq_above_map.png     : earthquake families vs LFE families (map + W-E section with Bloch 2023 t/m)
figures/sci_eq_above_lapse.png   : (a) EQ-only S-P steps per station (M7), (b) layer dv/v estimates (joint LFE-minus-EQ)
                                   with honest errors vs existing bounds, (c) yearly layer dv/v series
figures/sci_eq_above_mf.png      : matched-filter detections (amplitude distribution, effective stack gain)
data/sci/eq_above/summary.json
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = '/home/seismic/LFE'
os.chdir(ROOT)
OUT = f'{ROOT}/data/sci/eq_above'
sys.path.insert(0, f'{ROOT}/scripts')


def bloch_profile(lat0, lons):
    import h5py
    from scipy.interpolate import RegularGridInterpolator
    out = {}
    for k in ('t', 'm'):
        f = h5py.File(f'data/slab/bloch2023/grids_masked/{k}_depth_masked.nc', 'r')
        la, lo, z = f['lat'][:], f['lon'][:], f['z'][:]
        I = RegularGridInterpolator((la, lo), np.abs(z), bounds_error=False, fill_value=np.nan)
        out[k] = I(np.c_[np.full(len(lons), lat0), lons])
    return out


def fig_map():
    c = pd.read_csv(f'{OUT}/families.csv')
    sh = pd.read_csv('data/families/shelly_family_locations.csv')
    ev = pd.read_csv(f'{OUT}/reloc_events.csv')
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.2))
    a = ax[0]
    bg = c[c.efam == 0]
    a.scatter(bg.Longitude, bg.Latitude, s=6, c='0.75', lw=0, label='unclustered NCSS 14-35 km')
    fam = c[c.efam > 0]
    sc = a.scatter(fam.Longitude, fam.Latitude, s=14, c=fam.depth, cmap='plasma', vmin=15, vmax=31, lw=0,
                   label='earthquake families')
    a.scatter(sh.lon, sh.lat, s=40, marker='^', c=sh.depth_km, cmap='plasma', vmin=15, vmax=31, edgecolor='k',
              label='LFE families (Shelly)')
    for E, g in fam.groupby('efam'):
        a.text(g.Longitude.median(), g.Latitude.median(), str(E), fontsize=7)
    a.set_xlim(-124.0, -123.25)
    a.set_ylim(39.85, 40.35)
    a.set_aspect(1 / np.cos(np.radians(40.1)))
    a.legend(fontsize=7, loc='lower left')
    plt.colorbar(sc, ax=a, label='depth (km)')
    a.set_title('Earthquake families (CC>=0.62) and LFE families')
    a = ax[1]
    sel = c[(c.Latitude > 39.95) & (c.Latitude < 40.2)]
    a.scatter(sel.Longitude, sel.depth, s=6, c=np.where(sel.efam > 0, 'C1', '0.7'), lw=0)
    a.scatter(sh.lon, sh.depth_km, s=40, marker='^', c='k', label='LFE families (Shelly depth)')
    lons = np.linspace(-124.0, -123.3, 80)
    pr = bloch_profile(40.06, lons)
    a.plot(lons, pr['t'], 'b-.', label='Bloch 2023 t (top LVL), 40.06N')
    a.plot(lons, pr['m'], 'b-', label='Bloch 2023 m (slab Moho)')
    a.axhline(22.0, color='r', ls=':', label='Z_TOP 22 km (layer top for the differential kernel)')
    a.invert_yaxis()
    a.set_xlabel('longitude')
    a.set_ylabel('depth (km)')
    a.set_title('W-E section 39.95-40.20N (catalog depths)')
    a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig('figures/sci_eq_above_map.png', dpi=130)


def fig_lapse(summ):
    L = pd.read_csv(f'{OUT}/lapse_station.csv')
    R = pd.read_csv(f'{OUT}/diff_results.csv')
    if os.path.exists(f'{OUT}/diff_results_Kmean.csv'):
        K = pd.read_csv(f'{OUT}/diff_results_Kmean.csv')
        R = pd.concat([R, K[K.variant == 'joint_Kmean']], ignore_index=True)
    Y = pd.read_csv(f'{OUT}/diff_yearly.csv') if os.path.exists(f'{OUT}/diff_yearly.csv') else None
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.5))
    a = ax[0]
    x = L[L.phase == 'SmP'].copy()
    x['se'] = x[['m7_se_boot', 'm7_se_plac']].max(axis=1)
    x = x[np.isfinite(x.m7) & (x.se < 25)].sort_values('sta')
    a.errorbar(np.arange(len(x)), x.m7, yerr=2 * x.se, fmt='o', ms=4)
    a.axhline(0, color='k', lw=0.7)
    a.set_xticks(np.arange(len(x)))
    a.set_xticklabels(x.sta, rotation=90, fontsize=7)
    a.set_ylabel('M7 step in S-P residual (ms), +-2 SE')
    a.set_title('(a) earthquake families only (after relocation)')
    a = ax[1]
    qs = ['m7', 'restart', 'fer', 'pet', 'D']
    vs = [v for v in ['joint', 'lfe_beta', 'joint_steps', 'lfe_nobeta', 'joint_Kmean'] if v in set(R.variant)]
    for k, v in enumerate(vs):
        r = R[R.variant == v].iloc[0]
        a.errorbar(np.arange(len(qs)) + 0.12 * (k - 2), [r[q] for q in qs], yerr=[2 * r[f'{q}_se'] for q in qs],
                   fmt='os^dv'[k], ms=4,
                   label=v + (' (EQ contrast only)' if v == 'joint_Kmean' else ''))
    a.axhline(0, color='k', lw=0.7)
    a.axhspan(-0.36, 0.21, color='0.9', zorder=0, label='existing near-source S-P 95% (10-km sphere, M7)')
    a.set_xticks(range(len(qs)))
    a.set_xticklabels(qs)
    a.set_ylabel('dv/v of 22 km - LFE layer (%), +-2 honest SE')
    a.legend(fontsize=7)
    a.set_title('(b) LFE-minus-earthquake differential')
    a = ax[2]
    if Y is not None:
        for v, g in Y.groupby('variant'):
            a.errorbar(g.year + (0.1 if v == 'joint' else -0.1), g.eps_pct, yerr=2 * g.se_formal_pct, fmt='o-', ms=3,
                       label=f'{v} (pre-2021 RMS {g.rms_pre2021_pct.iloc[0]:.3f}%)')
        a.axhline(0, color='k', lw=0.7)
        for d in (2021.97, 2022.97, 2024.93):
            a.axvline(d, color='r', ls='--', lw=0.7)
        a.set_ylabel('layer dv/v vs 2008-09 (%), +-2 formal SE')
        a.legend(fontsize=7)
    a.set_title('(c) yearly layer dv/v')
    fig.tight_layout()
    fig.savefig('figures/sci_eq_above_lapse.png', dpi=130)


def fig_mf(summ):
    import glob
    fs = sorted(glob.glob(f'{OUT}/mf/det_*.csv'))
    if not fs:
        return
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    a = ax[0]
    a.hist(d.lamp.dropna(), bins=np.arange(-2, 1.5, 0.1), color='C0', alpha=0.7, label='all detections')
    a.hist(d[d.cc / d.thr >= 1.2].lamp.dropna(), bins=np.arange(-2, 1.5, 0.1), color='C1', alpha=0.7,
           label='cc/thr >= 1.2')
    a.set_xlabel('log10 amplitude relative to family template (~ dM)')
    a.set_ylabel('count')
    a.set_yscale('log')
    a.legend(fontsize=8)
    a.set_title('Matched-filter detections ' + ', '.join(os.path.basename(f)[4:8] for f in fs))
    a = ax[1]
    mf = summ.get('mf', {})
    if mf:
        a.bar(range(len(mf['gain'])), list(mf['gain'].values()))
        a.set_xticks(range(len(mf['gain'])))
        a.set_xticklabels(list(mf['gain'].keys()), rotation=30, fontsize=8)
        a.set_ylabel('stack SNR^2 with detections / catalog only')
        a.set_title('effective stack gain (sum of amplitude^2)')
    fig.tight_layout()
    fig.savefig('figures/sci_eq_above_mf.png', dpi=130)


def write_summary():
    S = {'what': 'Earthquakes above/near the LFEs as a second source to isolate temporal change between ~22 km and '
                 'the LFE depths (25-31 km); scripts/sci_eq_above*.py',
         'units': 'dv/v in %, positive = faster; se = max(formal, bootstrap, placebo) unless stated'}
    c = pd.read_csv(f'{OUT}/families.csv')
    S['catalog'] = dict(n_selected=int(len(c)), n_in_families=int((c.efam > 0).sum()), n_families=int(c.efam.max()),
                        cc_cut=0.62, region='epicentre < 30 km of 40.06N 123.62W, 14-35 km, NCSS all M, 2008-2026')
    ev = pd.read_csv(f'{OUT}/reloc_events.csv')
    rr = pd.read_parquet(f'{OUT}/reloc_resid.parquet')
    S['relocation'] = dict(n_events=int(len(ev)), median_abs_res_ms={ph: float(1e3 * rr[rr.phase == ph].res.abs().median())
                                                                     for ph in 'PS'},
                           median_dz_se_m=float(1e3 * ev.dz_se.median()))
    L = pd.read_csv(f'{OUT}/lapse_station.csv')
    x = L[L.phase == 'SmP'].copy()
    net = {}
    for q in ('pet', 'fer', 'm7', 'ramp'):
        x['se'] = x[[q + '_se_boot', q + '_se_plac']].max(axis=1)
        y = x[np.isfinite(x[q]) & np.isfinite(x.se) & (x.se < 20)]
        w = 1 / y.se ** 2
        m = float((w * y[q]).sum() / w.sum())
        net[q] = dict(ms=m, se_ms=float(1 / np.sqrt(w.sum())), n_sta=int(len(y)),
                      chi2r=float((w * (y[q] - m) ** 2).sum() / max(len(y) - 1, 1)))
    S['eq_only_SminusP_network_ms'] = net
    for t, key in (('', 'diff'), ('_Kmean', 'diff_Kmean'), ('_shelly', 'diff_shelly')):
        f = f'{OUT}/diff_results{t}.csv'
        if os.path.exists(f):
            R = pd.read_csv(f)
            S[key] = {r.variant: {q: dict(val=getattr(r, q), se=getattr(r, q + '_se'),
                                          se_formal=getattr(r, q + '_se_formal'))
                                  for q in ('m7', 'restart', 'fer', 'pet', 'D')} for r in R.itertuples()}
    if os.path.exists(f'{OUT}/diff_yearly.csv'):
        Y = pd.read_csv(f'{OUT}/diff_yearly.csv')
        S['yearly'] = {v: dict(rms_pre2021=float(g.rms_pre2021_pct.iloc[0]), median_formal=float(g.se_formal_pct.median()),
                               max_abs=float(g.eps_pct.abs().max()))
                       for v, g in Y.groupby('variant')}
    V = pd.read_csv(f'{OUT}/vpvs.csv')
    S['vpvs_LinShearer'] = V[V.efam.astype(str) == 'all'].replace({np.nan: None}).to_dict('records')
    for nm in ('seis_rates', 'seis_bvalue', 'seis_stress', 'seis_stress_change'):
        S[nm + '_file'] = f'data/sci/eq_above/{nm}.csv'
    st = pd.read_csv(f'{OUT}/seis_stress_change.csv')
    S['stress_change'] = st.to_dict('records')
    if os.path.exists(f'{OUT}/mf/eval.json'):
        e = json.load(open(f'{OUT}/mf/eval.json'))
        e.pop('per_family_year', None)
        S['mf'] = e
    json.dump(S, open(f'{OUT}/summary.json', 'w'), indent=1, default=float)
    return S


if __name__ == '__main__':
    summ = write_summary()
    fig_map()
    fig_lapse(summ)
    fig_mf(summ)
