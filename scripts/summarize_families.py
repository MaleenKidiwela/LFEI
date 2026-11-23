#!/usr/bin/env python3
"""Cross-family summary of data/reports/fam*_summary.json.

Outputs:
  data/reports/all_families_summary.csv     one row per family (all metrics)
  data/reports/all_families_report.md       key table + combined tests + per-family sections
  figures/all_families_map.png              Shelly vs our 3D locations, coloured by M7 gap
  figures/all_families_m7.png               M7 response: gap, 30-d rate, dv/v transient, source shift
Combined tests (families other than 23 = independent of where the tau=60 d model came from):
  weighted mean of the post-M7 dv/v transient amplitude, its chi2 consistency, sign counts;
  same for dv/v steps at each quake and fixed-station bin shifts at each quake.
"""
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def md_table(df, index=True):
    d = df.reset_index() if index else df
    fmt = lambda x: '' if (isinstance(x, float) and not np.isfinite(x)) else (f'{x:.4g}' if isinstance(x, float) else str(x))  # noqa: E731
    lines = ['| ' + ' | '.join(map(str, d.columns)) + ' |', '|' + '---|' * len(d.columns)]
    lines += ['| ' + ' | '.join(fmt(x) for x in r) + ' |' for r in d.itertuples(index=False)]
    return '\n'.join(lines)


def wmean(v, e):
    m = np.isfinite(v) & np.isfinite(e) & (e > 0)
    v, e = v[m], e[m]
    if len(v) == 0:
        return np.nan, np.nan, np.nan, 0
    w = 1 / e ** 2
    mu = np.sum(w * v) / np.sum(w)
    se = 1 / np.sqrt(np.sum(w))
    chi2 = np.sum(w * (v - mu) ** 2) / max(len(v) - 1, 1)
    return mu, se * np.sqrt(max(chi2, 1)), chi2, len(v)


def main():
    rows = [json.load(open(f)) for f in sorted(glob.glob('data/reports/fam*_summary.json'))]
    s = pd.DataFrame(rows).set_index('family').sort_index()
    s.to_csv('data/reports/all_families_summary.csv')
    num = lambda c: pd.to_numeric(s.get(c), errors='coerce') if c in s else pd.Series(np.nan, index=s.index)  # noqa: E731
    L = ['# LFE families, southern Cascadia / MTJ: per-family results (Shelly v2 catalog)\n']

    # ---- key table
    cols = dict(n_events='N', loc3d_lat='lat', loc3d_lon='lon', loc3d_depth_km='z_km', loc3d_horiz_offset_from_shelly_km='dH_shelly_km',
                reloc_events_good='reloc_good', patch_len_km_5_95='patch_L_km', patch_azimuth_deg='patch_az',
                m7_gapfull_d='M7_gap_d', m7_gap_rank='M7_gap_rank', m7_ratio30='M7_rate30', fer_ratio30='Fern_rate30', pet_ratio30='Pet_rate30',
                dvv_channels_pass='dvv_ch', dvv_annual_amp_pct='annual_pct', dvv_annual_peak_doy='annual_peak_doy',
                cw_nch='cw_ch', cw_annual_amp_pct='cw_annual_pct', cw_annual_peak_doy='cw_peak_doy',
                cw_step_m7_pct='cw_M7_step', cw_step_m7_pct_se='cw_M7_se', cw_transient60_pct='cw_M7_trans', cw_transient60_pct_se='cw_trans_se',
                rev_rate_silence_per_day='rev_sil_per_d', rev_rate_pre_per_day='rev_pre_per_d')
    tab = pd.DataFrame({v: num(k) for k, v in cols.items()})
    L.append('## Key numbers per family\n')
    L.append(md_table(tab))
    L.append('')

    # ---- combined tests
    oth = s.index > 0   # all families (CWIRE tests have fixed tau=60 d and model chosen before seeing other families)
    L.append('## Combined tests across families (inverse-variance mean; se inflated by sqrt(reduced chi2))\n')
    tests = []
    for k, lab in (('cw', 'CWIRE all ch'), ('cwPB', 'CWIRE borehole'), ('cwSurf', 'CWIRE surface')):
        for p, plab in (('step_pet_pct', 'step Petrolia'), ('step_fer_pct', 'step Ferndale'), ('step_m7_pct', 'step M7'),
                        ('transient60_pct', 'M7 60-d transient'), ('trend_pct_per_yr', 'trend /yr')):
            tests.append((f'{lab}: {plab} (%)', f'{k}_{p}', f'{k}_{p}_se'))
    tests += [('post-M7 dv/v transient, tau 60 d (%)', 'transient_A_pct', 'transient_A_se_pct'),
             ('dv/v step Petrolia (%)', 'dvv_pet_pct', 'dvv_pet_se'),
             ('dv/v step Ferndale (%)', 'dvv_fer_pct', 'dvv_fer_se'),
             ('dv/v step M7 (%)', 'dvv_m7_pct', 'dvv_m7_se'),
             ('dv/v trend (%/yr)', 'dvv_trend_pct', 'dvv_trend_se')]
    for k in ('pet', 'fer', 'm7'):
        for c in ('dx', 'dy', 'dz'):
            tests.append((f'fixed-station bin shift {k} {c} (m)', f'step_core_{k}_{c}_m', f'step_core_{k}_{c}_se'))
    tests.append(('relocation depth trend since 2020 (m/yr)', 'trend2020_dz_m_per_yr', 'trend2020_dz_se'))
    out = []
    for name, v, e in tests:
        vv, ee = num(v), num(e)
        mu, se, chi2, n = wmean(vv[oth].values, ee[oth].values)
        npos = int((vv[oth] > 0).sum())
        nsig = int((np.abs(vv[oth] / ee[oth]) > 2).sum())
        f23 = f'{vv.get(23, np.nan):+.4g} +- {ee.get(23, np.nan):.2g}' if 23 in vv.index else 'n/a'
        out.append(dict(test=name, n_fam=n, weighted_mean=mu, se=se, sigma=mu / se if se > 0 else np.nan,
                        red_chi2=chi2, n_positive=npos, n_beyond_2sigma=nsig, family23=f23))
    ct = pd.DataFrame(out)
    L.append(md_table(ct, index=False))
    L.append('')
    # annual phase coherence
    ph = num('dvv_annual_peak_doy')[oth].dropna()
    if len(ph):
        ang = 2 * np.pi * ph / 365.25
        R = np.hypot(np.cos(ang).mean(), np.sin(ang).mean())
        mean_doy = (np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360) / 360 * 365.25
        L.append(f'- Annual dv/v peak day across {len(ph)} families: circular mean day {mean_doy:.0f}, '
                 f'resultant length R = {R:.2f} (1 = all in phase; ~{1 / np.sqrt(len(ph)):.2f} expected for random phases).')
    L.append('')

    # ---- activity response to all M>=5.5 within 200 km (catalog only)
    fq = 'data/reports/quake_response_catalog.csv'
    if os.path.exists(fq):
        r = pd.read_csv(fq)
        qt = r.groupby(['quake', 'dist_km'], sort=False).agg(
            n_fam=('family', 'size'), fam_gap_top10=('gap_rank', lambda x: int((x <= 10).sum())),
            median_gap_d=('gap_d', 'median'), max_gap_d=('gap_d', 'max'), median_ratio30=('ratio30', 'median'),
            post30=('post30', 'sum'), pre365=('pre365', 'sum'))
        qt['all_ratio30'] = qt.post30 / (qt.pre365 * 30 / 365)
        L.append('## LFE activity after every M>=5.5 within 200 km, 2008-2026 (all 27 families)\n')
        L.append(md_table(qt.drop(columns=['post30', 'pre365']).reset_index(), index=False))
        L.append('')
    # ---- CWIRE layout: per station-channel, all families pooled
    fp = 'data/dvv/pool/params_by_channel.csv'
    if os.path.exists(fp):
        d = pd.read_csv(fp)
        d['type'] = np.where(d.chan.str.startswith('PB'), 'borehole PB', 'surface BK/NC')
        L.append('## dv/v per station-channel, all families pooled (CWIRE layout)\n')
        rows = []
        for t, g in d.groupby('type'):
            row = dict(type=t, channels=len(g))
            for c in ('step_pet_pct', 'step_fer_pct', 'step_m7_pct', 'transient60_pct'):
                w = 1 / g[c + '_se'] ** 2
                m = (g[c] * w).sum() / w.sum()
                chi = ((g[c] - m) ** 2 * w).sum() / max(len(g) - 1, 1)
                row[c] = f'{m:+.4f} +- {np.sqrt(max(chi, 1)) / np.sqrt(w.sum()):.4f}'
            row['annual_amp_median'] = f'{g.annual_amp_pct.median():.4f}'
            row['annual_peak_doy_median'] = f'{g.annual_peak_doy.median():.0f}'
            rows.append(row)
        L.append(md_table(pd.DataFrame(rows), index=False))
        L.append('')
        cols = ['chan', 'n_fam', 'annual_amp_pct', 'annual_peak_doy', 'step_pet_pct', 'step_pet_pct_se', 'step_fer_pct',
                'step_fer_pct_se', 'step_m7_pct', 'step_m7_pct_se', 'transient60_pct', 'transient60_pct_se', 'dist_km']
        L.append(md_table(d[[c for c in cols if c in d]].sort_values('chan'), index=False))
        L.append('')

    # ---- per-family sections
    L.append('## Per-family reports\n')
    for F in s.index:
        try:
            L.append(open(f'data/reports/fam{F}_report.md').read())
        except OSError:
            pass
    open('data/reports/all_families_report.md', 'w').write('\n'.join(L))
    print('\n'.join(L[:12]))
    print(ct.round(4).to_string(index=False))

    # ---- figures
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    g = num('m7_gapfull_d')
    for F in s.index:
        ax[0].plot([num('shelly_lon')[F], num('loc3d_lon')[F]], [num('shelly_lat')[F], num('loc3d_lat')[F]], '-', color='0.7', lw=0.8)
        ax[0].annotate(str(F), (num('loc3d_lon')[F], num('loc3d_lat')[F]), fontsize=7, xytext=(3, 3), textcoords='offset points')
    ax[0].scatter(num('shelly_lon'), num('shelly_lat'), s=18, facecolor='none', edgecolor='k', label='Shelly')
    sc = ax[0].scatter(num('loc3d_lon'), num('loc3d_lat'), c=g, cmap='viridis', s=40, label='this study (3D)')
    plt.colorbar(sc, ax=ax[0], label='gap spanning the M7 (days)')
    ax[0].set_xlabel('lon')
    ax[0].set_ylabel('lat')
    ax[0].legend(fontsize=8)
    ax[0].set_title('Family locations')
    ax[1].scatter(num('loc3d_lon'), num('loc3d_depth_km'), c=g, cmap='viridis', s=40)
    ax[1].scatter(num('shelly_lon'), num('shelly_depth_km'), s=18, facecolor='none', edgecolor='k')
    for F in s.index:
        ax[1].annotate(str(F), (num('loc3d_lon')[F], num('loc3d_depth_km')[F]), fontsize=7, xytext=(3, 3), textcoords='offset points')
    ax[1].invert_yaxis()
    ax[1].set_xlabel('lon')
    ax[1].set_ylabel('depth (km; ours ~2.5 km too deep per earthquake test)')
    ax[1].set_title('Depth vs longitude')
    fig.tight_layout()
    fig.savefig('figures/all_families_map.png', dpi=130)

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    x = num('loc3d_lon')
    ax[0, 0].scatter(x, g)
    for F in s.index:
        ax[0, 0].annotate(str(F), (x[F], g[F]), fontsize=7)
    ax[0, 0].set_ylabel('gap spanning the M7 (d)')
    ax[0, 1].scatter(x, num('m7_ratio30'), label='M7')
    ax[0, 1].scatter(x, num('fer_ratio30'), marker='s', label='Ferndale')
    ax[0, 1].scatter(x, num('pet_ratio30'), marker='^', label='Petrolia')
    ax[0, 1].axhline(1, color='k', lw=0.5)
    ax[0, 1].set_ylabel('30-d count / expected')
    ax[0, 1].legend(fontsize=8)
    A, E = num('cw_transient60_pct'), num('cw_transient60_pct_se')
    ax[1, 0].errorbar(s.index, A, yerr=E, fmt='o')
    ax[1, 0].axhline(0, color='k', lw=0.5)
    mu = ct.set_index('test').loc['CWIRE all ch: M7 60-d transient (%)']
    ax[1, 0].axhspan(mu.weighted_mean - mu.se, mu.weighted_mean + mu.se, color='r', alpha=0.2,
                     label=f'weighted mean (excl. 23) {mu.weighted_mean:+.4f}+-{mu.se:.4f}')
    ax[1, 0].set_xlabel('family')
    ax[1, 0].set_ylabel('post-M7 dv/v transient amplitude (%)')
    ax[1, 0].legend(fontsize=8)
    for c, m in (('dx', 'o'), ('dy', 's'), ('dz', '^')):
        ax[1, 1].errorbar(s.index, num(f'step_core_m7_{c}_m'), yerr=num(f'step_core_m7_{c}_se'), fmt=m, label=c, ms=4)
    ax[1, 1].axhline(0, color='k', lw=0.5)
    ax[1, 1].set_xlabel('family')
    ax[1, 1].set_ylabel('source shift at M7, fixed-station bins (m)')
    ax[1, 1].legend(fontsize=8)
    for a in ax[0]:
        a.set_xlabel('family longitude (our 3D)')
    fig.suptitle('Response of all LFE families to the Dec 2024 M7.0')
    fig.tight_layout()
    fig.savefig('figures/all_families_m7.png', dpi=130)


if __name__ == '__main__':
    main()
