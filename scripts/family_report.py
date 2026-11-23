#!/usr/bin/env python3
"""Collect one family's results into data/reports/fam{F}_summary.json (flat metrics) and
data/reports/fam{F}_report.md (human-readable). Every section is optional (missing outputs -> NaN).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

Q = {'Petrolia M6.1': '2021-12-20T20:10', 'Ferndale M6.4': '2022-12-20T10:34', 'M7.0': '2024-12-05T18:44'}
QK = {'Petrolia M6.1': 'pet', 'Ferndale M6.4': 'fer', 'M7.0': 'm7'}


def ep(x):
    return np.asarray((pd.DatetimeIndex(x) - pd.Timestamp('1970-01-01')) / pd.Timedelta(seconds=1))


def safe(f):
    def w(*a, **k):
        try:
            return f(*a, **k)
        except Exception as e:  # noqa: BLE001
            return {'error_' + f.__name__: repr(e)[:200]}
    return w


@safe
def activity(F):
    c = pd.read_csv('data/merged/LFEcat_MTJ_merged_v1v2.csv')
    c = c[(c.source != 'v1_only') & (c.idnum == F)]
    t = pd.to_datetime(c.time_utc, format='mixed').sort_values().reset_index(drop=True)
    gaps = t.diff().dt.total_seconds().values[1:] / 86400
    tg = t.values[1:]
    sel = t.values[1:] >= np.datetime64('2009-01-01')
    out = dict(n_events=len(t), first=str(t.iloc[0]), last=str(t.iloc[-1]),
               rate_per_yr=len(t) / ((t.iloc[-1] - t.iloc[0]).days / 365.25),
               median_interevent_min=float(np.median(gaps) * 1440))
    longest = np.sort(gaps[sel])[::-1]
    out['longest_gap_since_2009_d'] = float(longest[0]) if len(longest) else np.nan
    out['longest_gap_since_2009_start'] = str(pd.Timestamp(tg[sel][np.argmax(gaps[sel])]) - pd.Timedelta(days=float(longest[0]))) if len(longest) else ''
    for k, q in Q.items():
        q = pd.Timestamp(q)
        after = t[t >= q]
        g = (after.iloc[0] - q).total_seconds() / 86400 if len(after) else np.nan
        before365 = ((t >= q - pd.Timedelta(days=365)) & (t < q)).sum()
        a30 = ((t >= q) & (t < q + pd.Timedelta(days=30))).sum()
        a90 = ((t >= q) & (t < q + pd.Timedelta(days=90))).sum()
        exp30 = before365 * 30 / 365
        # gap percentile: fraction of all post-2009 inter-event gaps shorter than this one
        out[f'{QK[k]}_gap_d'] = g
        # rank (1 = longest since 2009) of the inter-event gap that contains the quake
        if len(after) and (t < q).any():
            gfull = (after.iloc[0] - t[t < q].iloc[-1]).total_seconds() / 86400
            out[f'{QK[k]}_gapfull_d'] = gfull
            out[f'{QK[k]}_gap_rank'] = int((gaps[sel] > gfull + 1e-9).sum() + 1)
        out[f'{QK[k]}_n30_after'] = int(a30)
        out[f'{QK[k]}_n30_expected'] = float(exp30)
        out[f'{QK[k]}_ratio30'] = float(a30 / exp30) if exp30 > 0 else np.nan
        out[f'{QK[k]}_ratio90'] = float(a90 / (before365 * 90 / 365)) if before365 > 0 else np.nan
        after10 = after.iloc[9] if len(after) >= 10 else None
        out[f'{QK[k]}_t10_d'] = (after10 - q).total_seconds() / 86400 if after10 is not None else np.nan
    return out


@safe
def location(F):
    out = {}
    for m in ('3d', '1d'):
        f = f'data/locations/fam{F}_{m}.json'
        if os.path.exists(f):
            j = json.load(open(f))
            for k in ('lat', 'lon', 'depth_km', 'sd_x_km', 'sd_y_km', 'sd_z_km', 'n_S', 'n_P', 'l1_misfit_s',
                      'horiz_offset_from_shelly_km', 'dz_from_shelly_km', 's_scale'):
                out[f'loc{m}_{k}'] = j.get(k)
    sh = pd.read_csv('data/families/shelly_family_locations.csv').set_index('idnum').loc[F]
    out.update(shelly_lat=sh.lat, shelly_lon=sh.lon, shelly_depth_km=sh.depth_km)
    return out


@safe
def detections(F):
    s = pd.read_csv(f'data/detections/fam{F}_station_summary.csv')
    d = pd.read_parquet(f'data/detections/fam{F}_station_detections.parquet', columns=['ev', 'detected'])
    g = d.groupby('ev').detected
    top = s.sort_values('frac_detected', ascending=False).head(5)
    return dict(det_median_sta_with_data=float(g.size().median()), det_median_sta_detecting=float(g.sum().median()),
                det_frac_events_ge3=float((g.sum() >= 3).mean()),
                det_top=';'.join(f'{r.sta} {r.frac_detected:.0%}' for r in top.itertuples()))


def pca_patch(x, y):
    X = np.c_[x, y] - np.c_[x, y].mean(0)
    w, v = np.linalg.eigh(np.cov(X.T))
    major = v[:, 1]
    az = (np.degrees(np.arctan2(major[0], major[1])) + 360) % 180   # x=east, y=north
    p = X @ v
    L = np.percentile(p[:, 1], 95) - np.percentile(p[:, 1], 5)
    W = np.percentile(p[:, 0], 95) - np.percentile(p[:, 0], 5)
    return az, L, W


def step_bins(b, q, win=365):
    q = pd.Timestamp(q)
    t = pd.to_datetime(b.t_mid, format='mixed')
    bb, aa = b[(t >= q - pd.Timedelta(days=win)) & (t < q)], b[(t >= q) & (t < q + pd.Timedelta(days=win))]
    if len(bb) < 2 or len(aa) < 2:
        return {}
    o = {}
    for c in ('dx', 'dy', 'dz'):
        v = (aa[c].mean() - bb[c].mean()) * 1000
        e = np.hypot(aa[c].std(ddof=1) / np.sqrt(len(aa)), bb[c].std(ddof=1) / np.sqrt(len(bb))) * 1000
        o[c] = (v, e)
    return o


@safe
def relocation(F):
    out = {}
    e = pd.read_csv(f'data/relocations/fam{F}_events_3d.csv')
    g = e[(e.sd_x < 0.5) & (e.sd_y < 0.5) & (e.sd_z < 1.0)]
    out.update(reloc_events=len(e), reloc_events_good=len(g))
    if len(g) > 30:
        az, L, W = pca_patch(g.dx.values, g.dy.values)
        out.update(patch_azimuth_deg=az, patch_len_km_5_95=L, patch_wid_km_5_95=W,
                   patch_depth_5_95_km=float(np.percentile(g.dz, 95) - np.percentile(g.dz, 5)))
    b = pd.read_csv(f'data/relocations/fam{F}_bins_3d.csv')
    out.update(reloc_bins=len(b), bin_std_m=';'.join(f'{1000 * b[c].std():.0f}' for c in ('dx', 'dy', 'dz')))
    t = pd.to_datetime(b.t_mid, format='mixed')
    s20 = t >= pd.Timestamp('2020-01-01')
    if s20.sum() > 5:
        yr = (t[s20] - t[s20].min()).dt.total_seconds() / (365.25 * 86400)
        for c in ('dx', 'dy', 'dz'):
            p = np.polyfit(yr, b[c][s20] * 1000, 1, cov=True)
            out[f'trend2020_{c}_m_per_yr'] = p[0][0]
            out[f'trend2020_{c}_se'] = float(np.sqrt(p[1][0, 0]))
    fc = f'data/relocations/fam{F}_bins_3d_core.csv'
    for name, bf in (('all', b), ('core', pd.read_csv(fc) if os.path.exists(fc) and os.path.getsize(fc) > 50 else None)):
        if bf is None:
            continue
        for k, q in Q.items():
            for c, (v, er) in step_bins(bf, q).items():
                out[f'step_{name}_{QK[k]}_{c}_m'] = v
                out[f'step_{name}_{QK[k]}_{c}_se'] = er
    ec = f'data/relocations/fam{F}_events_3d_core.csv'
    if os.path.exists(ec) and os.path.getsize(ec) > 50:
        c = pd.read_csv(ec)
        out.update(reloc_core_events=len(c),
                   reloc_core_events_good=int(((c.sd_x < 0.5) & (c.sd_y < 0.5) & (c.sd_z < 1.0)).sum()))
        m = e.merge(c, on='ev', suffixes=('_a', '_c'))
        m = m[(m.sd_x_a < 0.5) & (m.sd_y_a < 0.5) & (m.sd_z_a < 1) & (m.sd_x_c < 0.5) & (m.sd_y_c < 0.5) & (m.sd_z_c < 1)]
        if len(m) > 30:
            out['core_vs_all_r'] = ';'.join(f"{np.corrcoef(m[k + '_a'], m[k + '_c'])[0, 1]:.2f}" for k in ('dx', 'dy', 'dz'))
    return out


@safe
def dvv(F):
    out = {}
    fs = f'data/dvv/fam{F}_summary_iso5_n150.csv'
    if os.path.exists(fs):
        sm = pd.read_csv(fs)
        good = sm[(sm.mean_cc_coda >= 0.7) & (sm.dvv_noise_std >= 5 * sm.dvv_ref_std)]
        out.update(dvv_channels_pass=len(good), dvv_channels_tested=len(sm))
    ff = f'data/dvv/fam{F}_dvv_fit_iso5_n150.csv'
    if os.path.exists(ff):
        f = pd.read_csv(ff).set_index('param')
        se = f[['se_lsq_pct', 'se_boot_pct']].max(axis=1)
        for p, k in (('step M6.1 Petrolia', 'pet'), ('step M6.4 Ferndale', 'fer'), ('step M7.0 Mendocino', 'm7'),
                     ('trend per yr', 'trend')):
            out[f'dvv_{k}_pct'] = f.loc[p, 'value_pct']
            out[f'dvv_{k}_se'] = se[p]
        s, c = f.loc['annual sin', 'value_pct'], f.loc['annual cos', 'value_pct']
        out['dvv_annual_amp_pct'] = float(np.hypot(s, c))
        out['dvv_annual_amp_se'] = float(np.sqrt(se['annual sin'] ** 2 + se['annual cos'] ** 2) / np.sqrt(2))
        out['dvv_annual_peak_doy'] = float(((np.pi / 2 - np.arctan2(c, s)) % (2 * np.pi)) / (2 * np.pi) * 365.25)
    fn = f'data/dvv/fam{F}_network_dvv_iso5_n150.csv'
    if os.path.exists(fn):
        out['dvv_bins'] = len(pd.read_csv(fn))
    ft = f'data/dvv/fam{F}_dvv_transient_iso5_n150.json'
    if os.path.exists(ft):
        j = json.load(open(ft))
        for k in ('A_pct', 'A_se_pct', 'A_sigma', 'best_tau', 'best_A_pct', 'best_dchi2', 'n_bins_0_120d', 'note'):
            if k in j:
                out[f'transient_{k}'] = j[k]
    return out


@safe
def cwire(F):
    """CWIRE-style dv/v (dvv_cwire.py): all channels, borehole PB only, surface BK/NC only."""
    out = {}
    for suf, k in (('', 'cw'), ('_PB', 'cwPB'), ('_BKNC', 'cwSurf')):
        f = f'data/dvv/fam{F}_cwire{suf}_params.json'
        if not os.path.exists(f):
            continue
        j = json.load(open(f))
        out[f'{k}_nch'] = j.get('n_channels', 0)
        out[f'{k}_measurable'] = bool(j.get('n_channels', 0) >= 5)
        if not out[f'{k}_measurable']:
            continue
        out[f'{k}_annual_amp_pct'] = j['annual_amp_pct']
        out[f'{k}_annual_peak_doy'] = j['annual_peak_doy']
        for p in ('step_pet_pct', 'step_fer_pct', 'step_m7_pct', 'transient60_pct', 'trend_pct_per_yr'):
            out[f'{k}_{p}'] = j[p]['value']
            out[f'{k}_{p}_se'] = j[p]['se']
    return out


@safe
def silence(F):
    out = {}
    fw = f'data/scan/fam{F}_silence_window.json'
    if os.path.exists(fw):
        j = json.load(open(fw))
        out.update(silence_end=j['end'], silence_days=j['silence_days'])
    fs = f'data/scan/fam{F}_scan_all.csv'
    if os.path.exists(fs) and 'silence_end' in out:
        d = pd.read_csv(fs)
        d['t'] = pd.to_datetime(d.time_utc.str.replace('Z', ''), format='mixed')
        M7, END = pd.Timestamp(Q['M7.0']), pd.Timestamp(out['silence_end'])
        S0 = pd.Timestamp('2024-11-10')
        iso_neg = d[(d.sign == -1) & ~d.catfam & ~d.catoth]
        pre_days = (M7 - S0).total_seconds() / 86400
        sil_days = max((END - M7).total_seconds() / 86400, 1e-3)
        out['rev_rate_pre_per_day'] = (iso_neg.t < M7).sum() / pre_days
        out['rev_n_silence'] = int(((iso_neg.t >= M7) & (iso_neg.t < END)).sum())
        out['rev_rate_silence_per_day'] = out['rev_n_silence'] / sil_days
        pos = d[d.sign == 1]
        out['scan_new_pos_silence'] = int(((pos.t >= M7) & (pos.t < END) & ~pos.catfam & ~pos.catoth).sum())
        c = pd.read_csv('data/merged/LFEcat_MTJ_merged_v1v2.csv')
        c = c[(c.source != 'v1_only') & (c.idnum == F)]
        ct = pd.to_datetime(c.time_utc, format='mixed')
        ncat = ((ct >= S0) & (ct < M7)).sum()
        out['scan_recovery_pre'] = float(pos[pos.t < M7].catfam.sum() / ncat) if ncat else np.nan
        out['scan_n_catalog_pre'] = int(ncat)
    fc = f'data/scan/fam{F}_coherence_check.csv'
    if os.path.exists(fc):
        r = pd.read_csv(fc)
        g = r.groupby('group')[['frac_same_sign', 'frac_strong']].median()
        for grp, k in (('silence_reversed_isolated', 'sil_rev'), ('ref_noise (pre-M7 reversed isolated)', 'noise'),
                       ('ref_real_LFE (catalog, normal)', 'real')):
            if grp in g.index:
                out[f'coh_{k}_same_sign'] = g.loc[grp, 'frac_same_sign']
                out[f'coh_{k}_strong'] = g.loc[grp, 'frac_strong']
        sr = r[r.group == 'silence_reversed_isolated']
        if len(sr) and 'coh_real_same_sign' in out:
            thr = (out['coh_real_same_sign'] + out.get('coh_noise_same_sign', 0.6)) / 2
            out['sil_rev_coherent'] = int((sr.frac_same_sign >= thr).sum())
            if 'eq_nearby' in sr:
                out['sil_rev_with_eq'] = int(sr.eq_nearby.fillna('').astype(str).str.len().gt(0).sum())
    for tag, k in (('', 'm7'), ('_control', 'ctl')):
        fm = f'data/silent/fam{F}{tag}_relative_magnitudes.csv'
        if os.path.exists(fm):
            m = pd.read_csv(fm)
            pre = m[m.group == 'pre'].rel_amp.mean()
            rs = m[m.group == 'restart'].sort_values('time_utc')
            out[f'amp_{k}_restart_over_pre'] = rs.rel_amp.mean() / pre
            out[f'amp_{k}_first25_over_pre'] = rs.rel_amp.head(25).mean() / pre
            out[f'amp_{k}_n'] = len(rs)
        fd = f'data/silent/fam{F}{tag}_dSP.csv'
        if os.path.exists(fd):
            x = pd.read_csv(fd)
            x = x[x.group == 'restart'].d_SminusP_ms
            if len(x):
                out[f'dSP_{k}_ms'] = x.mean()
                out[f'dSP_{k}_se'] = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
                out[f'dSP_{k}_nsta'] = len(x)
        fr = f'data/silent/fam{F}{tag}_amplitude_ratios.csv'
        if os.path.exists(fr):
            r = pd.read_csv(fr)
            p, s = r[r.group == 'pre'].set_index('sta'), r[r.group == 'restart'].set_index('sta')
            j = p.add_suffix('_p').join(s.add_suffix('_r'), how='inner')
            if len(j):
                z = (j.sh_sv_r - j.sh_sv_p) / np.hypot(j.sh_sv_sd_p, j.sh_sv_sd_r)
                out[f'shsv_{k}_n_z2'] = f'{int((z.abs() > 2).sum())}/{len(z)}'
    return out


def fmt(v, f='{:.2f}'):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return 'n/a'
    return f.format(v)


def markdown(F, s):
    L = [f'## Family {F}']
    g = lambda k, f='{:.2f}': fmt(s.get(k), f)  # noqa: E731
    L.append(f"- **Catalog (v2):** {g('n_events', '{:d}')} events, {s.get('first', '')[:10]} to {s.get('last', '')[:10]} "
             f"({g('rate_per_yr', '{:.0f}')}/yr; median inter-event {g('median_interevent_min', '{:.0f}')} min)")
    L.append(f"- **Shelly location:** {g('shelly_lat', '{:.4f}')}N {g('shelly_lon', '{:.4f}')}E, {g('shelly_depth_km', '{:.1f}')} km")
    L.append(f"- **Our 3D location:** {g('loc3d_lat', '{:.4f}')}N {g('loc3d_lon', '{:.4f}')}E, {g('loc3d_depth_km', '{:.1f}')} km "
             f"(sd {g('loc3d_sd_x_km')}/{g('loc3d_sd_y_km')}/{g('loc3d_sd_z_km')} km E/N/Z; {g('loc3d_n_S', '{:d}')} S + {g('loc3d_n_P', '{:d}')} P picks); "
             f"{g('loc3d_horiz_offset_from_shelly_km')} km from Shelly, dz {g('loc3d_dz_from_shelly_km', '{:+.1f}')} km")
    L.append(f"- **Per-station detection:** median {g('det_median_sta_detecting', '{:.0f}')} of {g('det_median_sta_with_data', '{:.0f}')} stations "
             f"detect an event individually; best: {s.get('det_top', 'n/a')}")
    L.append(f"- **Relocation:** {g('reloc_events', '{:d}')} events ({g('reloc_events_good', '{:d}')} well constrained), {g('reloc_bins', '{:d}')} bins; "
             f"patch {g('patch_len_km_5_95')} x {g('patch_wid_km_5_95')} km (5-95%), long axis {g('patch_azimuth_deg', '{:.0f}')} deg; "
             f"bin scatter E/N/Z {s.get('bin_std_m', 'n/a')} m; trend since 2020 dz {g('trend2020_dz_m_per_yr', '{:+.1f}')} +- {g('trend2020_dz_se', '{:.1f}')} m/yr")
    st = []
    for q, k in QK.items():
        st.append(f"{q}: dE {g(f'step_core_{k}_dx_m', '{:+.0f}')}+-{g(f'step_core_{k}_dx_se', '{:.0f}')}, "
                  f"dN {g(f'step_core_{k}_dy_m', '{:+.0f}')}+-{g(f'step_core_{k}_dy_se', '{:.0f}')}, "
                  f"dZ {g(f'step_core_{k}_dz_m', '{:+.0f}')}+-{g(f'step_core_{k}_dz_se', '{:.0f}')} m")
    L.append('- **Source shift at quakes (fixed-station bins, 1 yr before vs after):** ' + '; '.join(st))
    L.append(f"- **Quake response (activity):** " + '; '.join(
        f"{q}: first event after {g(f'{k}_gap_d', '{:.1f}')} d (enclosing gap {g(f'{k}_gapfull_d', '{:.1f}')} d, rank {s.get(f'{k}_gap_rank', 'n/a')} since 2009), "
        f"30-d rate {g(f'{k}_ratio30')}x normal" for q, k in QK.items()))
    L.append(f"- **dv/v (iso 5 s, 150/bin):** {g('dvv_channels_pass', '{:d}')} channels, {g('dvv_bins', '{:d}')} bins; annual "
             f"{g('dvv_annual_amp_pct', '{:.4f}')}% (fastest ~day {g('dvv_annual_peak_doy', '{:.0f}')}); steps Petrolia "
             f"{g('dvv_pet_pct', '{:+.4f}')}+-{g('dvv_pet_se', '{:.4f}')}, Ferndale {g('dvv_fer_pct', '{:+.4f}')}+-{g('dvv_fer_se', '{:.4f}')}, "
             f"M7 {g('dvv_m7_pct', '{:+.4f}')}+-{g('dvv_m7_se', '{:.4f}')} %; post-M7 transient (tau 60 d) "
             f"{g('transient_A_pct', '{:+.4f}')}+-{g('transient_A_se_pct', '{:.4f}')} % ({g('transient_A_sigma', '{:+.1f}')} sigma, "
             f"{s.get('transient_n_bins_0_120d', 'n/a')} bins within 120 d)")
    for k, lab in (('cw', 'all channels'), ('cwPB', 'borehole PB'), ('cwSurf', 'surface BK/NC')):
        if f'{k}_nch' not in s:
            continue
        if not s.get(f'{k}_measurable'):
            L.append(f"- **dv/v CWIRE-style ({lab}):** not measurable ({s.get(f'{k}_nch')} channels pass QC)")
            continue
        L.append(f"- **dv/v CWIRE-style ({lab}, {s[f'{k}_nch']} ch):** annual {g(f'{k}_annual_amp_pct', '{:.3f}')}% "
                 f"(fastest ~day {g(f'{k}_annual_peak_doy', '{:.0f}')}); steps Petrolia {g(f'{k}_step_pet_pct', '{:+.3f}')}+-{g(f'{k}_step_pet_pct_se', '{:.3f}')}, "
                 f"Ferndale {g(f'{k}_step_fer_pct', '{:+.3f}')}+-{g(f'{k}_step_fer_pct_se', '{:.3f}')}, M7 {g(f'{k}_step_m7_pct', '{:+.3f}')}+-{g(f'{k}_step_m7_pct_se', '{:.3f}')}; "
                 f"post-M7 60-d transient {g(f'{k}_transient60_pct', '{:+.3f}')}+-{g(f'{k}_transient60_pct_se', '{:.3f}')} %")
    L.append(f"- **Post-M7 silence:** window {g('silence_days', '{:.1f}')} d to 10th event; scan recovers {g('scan_recovery_pre', '{:.0%}')} of "
             f"{s.get('scan_n_catalog_pre', 'n/a')} pre-M7 catalog events; isolated reversed detections {g('rev_rate_silence_per_day')}/d in silence vs "
             f"{g('rev_rate_pre_per_day')}/d before (noise); {s.get('sil_rev_coherent', 'n/a')} of {s.get('rev_n_silence', 'n/a')} coherent; "
             f"{s.get('scan_new_pos_silence', 'n/a')} new normal-polarity detections in silence")
    L.append(f"- **Ideas 1/3/5 (restart vs pre; control in brackets):** amplitude {g('amp_m7_restart_over_pre')} [{g('amp_ctl_restart_over_pre')}], "
             f"first 25 {g('amp_m7_first25_over_pre')} [{g('amp_ctl_first25_over_pre')}]; d(S-P) {g('dSP_m7_ms', '{:+.1f}')}+-{g('dSP_m7_se', '{:.1f}')} ms "
             f"[{g('dSP_ctl_ms', '{:+.1f}')}+-{g('dSP_ctl_se', '{:.1f}')}]; SH/SV |z|>2 {s.get('shsv_m7_n_z2', 'n/a')} [{s.get('shsv_ctl_n_z2', 'n/a')}]")
    errs = [f'{k}: {v}' for k, v in s.items() if k.startswith('error_')]
    if errs:
        L.append('- **Missing/failed sections:** ' + '; '.join(errs))
    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', type=int, required=True)
    a = ap.parse_args()
    F = a.family
    s = dict(family=F)
    for fn in (activity, location, detections, relocation, dvv, cwire, silence):
        s.update(fn(F))
    os.makedirs('data/reports', exist_ok=True)
    json.dump({k: (None if isinstance(v, float) and not np.isfinite(v) else
                   (v.item() if hasattr(v, 'item') else v)) for k, v in s.items()},
              open(f'data/reports/fam{F}_summary.json', 'w'), indent=1, default=str)
    md = markdown(F, s)
    open(f'data/reports/fam{F}_report.md', 'w').write(md)
    print(md)


if __name__ == '__main__':
    main()
