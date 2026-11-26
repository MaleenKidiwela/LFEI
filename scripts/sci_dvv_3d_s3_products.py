#!/usr/bin/env python3
"""Stage 3 products: viewer.json (coordinator's spec), station-symbol maps, predicted-vs-observed, validation figure.

viewer.json:
  meta {description, units, kappa_runs, time functions, deep statement}
  stations [{sta, lat, lon, kind, sensor_depth_m}], lfe [{fam, lat, lon, depth}]
  components {key: {label, time_desc, units, signed, kappa_runs: {k1|kprof: {sta: {amp, amp_sd, depth_param,
              depth_hyp_dchi2, verdict, n_series, bands}}}}}
     amp = apparent dv/v the station's 1-3 Hz single-station-noise series would show (kernel-weighted 0-3 km), sd;
     depth_param = {bin0_0.3: [mean, sd], bin0.3_3: [mean, sd], corr};  verdict from depth_verdicts.csv
  dates {date: {sta: [mean, sd]}} (kappa k1; dates_kprof likewise)
  timeseries {sta: {label, t: [...], obs: [...], model: [...]}}  (10-day means; 1-3 Hz SC band, else best band)
  deep [{target, quantity, est, se, ci95, text}] from stage 1 (clean subset) - no deep field is presented
  map (optional; only if the variogram supports a correlation length)
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
sys.path.insert(0, f'{ROOT}/scripts')
import sci_dvv_3d_s3_inv as S3  # noqa: E402

OUT = S3.OUT
BIG = S3.BIG
FIG = f'{ROOT}/figures'
LABEL = {'ENV_cos': ('Seasonal environmental, cos part (peak 1 Jan)', 'ENV_cos x cos(2 pi doy / 365.25)', True),
         'ENV_sin': ('Seasonal environmental, sin part (peak 1 Apr)', 'ENV_sin x sin(2 pi doy / 365.25)', True),
         'ENV_amp': ('Seasonal environmental annual amplitude (positively biased magnitude)', 'hypot(ENV_cos, ENV_sin)', False),
         'drop0_Petrolia': ('Petrolia 2021-12-20: drop at t_q', 'drop0 x exp(-(t - t_q)/tau)', True),
         'level_Petrolia': ('Petrolia: permanent level', 'level x (H(t - t_q) - exp(-(t - t_q)/tau))', True),
         'drop0_Ferndale': ('Ferndale 2022-12-20: drop at t_q', 'drop0 x exp(-(t - t_q)/tau)', True),
         'level_Ferndale': ('Ferndale: permanent level', 'level x (H(t - t_q) - exp(-(t - t_q)/tau))', True),
         'drop0_M7': ('M7 2024-12-05: drop at t_q', 'drop0 x exp(-(t - t_q)/tau)', True),
         'level_M7': ('M7: permanent level', 'level x (H(t - t_q) - exp(-(t - t_q)/tau))', True),
         'ramp21': ('Post-2021-07 ramp', 'ramp21 x (t - 2021-07-01) [yr], t > hinge', True),
         'trend': ('Secular trend', 'trend x (t - 2018-01-01) [yr]', True)}


def fsp_outline(path):
    """surface projection outline (convex hull, [lon, lat]) of the finite-fault patches in a USGS .fsp file."""
    from scipy.spatial import ConvexHull
    pts = []
    for line in open(path):
        if line.startswith('%') or not line.strip():
            continue
        p = line.split()
        try:
            lat, lon = float(p[0]), float(p[1])
        except (ValueError, IndexError):
            continue
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            pts.append((lon, lat))
    if len(pts) < 3:
        return None
    P = np.array(pts)
    h = ConvexHull(P)
    ring = P[h.vertices].tolist() + [P[h.vertices[0]].tolist()]
    return [[round(a, 4), round(b, 4)] for a, b in ring]


def quakes():
    base = f'{ROOT}/data/sci/dose_response/usgs_raw'
    Q = {'Petrolia': ('2021-12-20T20:10:31', 6.2, 40.3902, -124.2980, None),
         'Ferndale': ('2022-12-20T10:34:24', 6.4, 40.5250, -124.4230, 'ferndale.fsp'),
         'M7': ('2024-12-05T18:44:21', 7.0, 40.3740, -125.0217, 'm7.fsp')}
    out = {}
    for k, (t, m, la, lo, f) in Q.items():
        rup = fsp_outline(f'{base}/{f}') if f and os.path.exists(f'{base}/{f}') else None
        out[k] = dict(time=t, mag=m, lat=la, lon=lo, rupture=rup,
                      note=None if rup else 'epicentre only (no finite-fault model)')
    return out


def r4(v):
    if v is None or not np.isfinite(v):
        return None
    return round(float(v), 4)


def main(smoke=False):
    sfx = '_smoke' if smoke else ''
    T = pd.read_csv(f'{OUT}/station_terms{sfx}.csv')
    H = pd.read_csv(f'{OUT}/depth_hypotheses{sfx}.csv')
    V = pd.read_csv(f'{OUT}/depth_verdicts{sfx}.csv')
    P = pd.read_csv(f'{OUT}/predictions{sfx}.csv')
    VG = json.load(open(f'{OUT}/variogram{sfx}.json')) if os.path.exists(f'{OUT}/variogram{sfx}.json') else None
    KF = pd.read_csv(f'{OUT}/kernel_fracs.csv').set_index(['dataset', 'sta', 'band', 'variant'])
    ST = pd.read_csv(f'{ROOT}/data/sci/dvv_3d/stations.csv').set_index('ns')
    taus = json.load(open(f'{OUT}/tau_net{sfx}.json'))['tau_q']
    Z = np.load(f'{BIG}/solutions{sfx}.npz', allow_pickle=False)
    dat = S3.Data(smoke=smoke, calib=False)
    comps = {}
    PA = P[P.date.str.startswith('amp:')]
    for ktag in ('k1', 'kprof'):
        for r in T[(T.kappa == ktag) & (T.cfg == 'primary')].itertuples():
            a = PA[(PA.kappa == ktag) & (PA.sta == r.sta) & (PA.date == f'amp:{r.amp}')]
            if a.empty:
                continue
            h = H[(H.run == f'{ktag}|x1') & (H.sta == r.sta) & (H.amp == r.amp)]
            vd = V[(V.sta == r.sta) & (V.amp == r.amp)]
            rows = dat.D[(dat.D.sta == r.sta) & (dat.D.amp == r.amp)]
            hyp = None
            if len(h):
                h = h.iloc[0]
                deep = min(h['tot_ge1km'], h['tot_below3'])
                best = min(h['tot_top0.3'], h['tot_uniform0_3'], deep)
                hyp = {'top0.3': r4(h['tot_top0.3'] - best), 'uniform0_3': r4(h['tot_uniform0_3'] - best),
                       'deep_ge1km': r4(deep - best), 'H0': r4(h['tot_H0'] - best), 'free': r4(h['tot_free'] - best)}
            split = bool(r.split_ok) if r.split_ok == r.split_ok else False
            comps.setdefault(r.amp, {}).setdefault(ktag, {})[r.sta] = dict(
                amp=r4(a['mean'].iloc[0]), amp_sd=r4(a['sd'].iloc[0]),
                depth_param=({'bin0_0.3': [r4(r.a_0_0p3), r4(r.a_sd)], 'bin0.3_3': [r4(r.b_0p3_3), r4(r.b_sd)], 'corr': r4(r.corr_ab)}
                             if split else 'split prior-dominated'),
                depth_hyp_dchi2=hyp, verdict=vd.verdict.iloc[0] if len(vd) else 'not resolved', n_series=int(len(rows)),
                bands=sorted(set(('SC ' + c) if d_ == 'SC10' else ('LFE ' + c.split('.')[-1]) if d_ == 'LFE' else d_
                                 for d_, c in zip(rows.dataset, rows.chan))),
                level_without_transient=bool(getattr(r, 'level_without_transient', False)))
        # annual amplitude (unsigned; positively biased) from the cos / sin amplitudes of the 1-3 Hz prediction
        for sta in set(PA[PA.kappa == ktag].sta):
            c = PA[(PA.kappa == ktag) & (PA.sta == sta) & (PA.date == 'amp:ENV_cos')]
            s_ = PA[(PA.kappa == ktag) & (PA.sta == sta) & (PA.date == 'amp:ENV_sin')]
            if c.empty or s_.empty:
                continue
            cv, sv = c['mean'].iloc[0], s_['mean'].iloc[0]
            amp = np.hypot(cv, sv)
            sd = np.sqrt((cv * c['sd'].iloc[0]) ** 2 + (sv * s_['sd'].iloc[0]) ** 2) / max(amp, 1e-12)
            comps.setdefault('ENV_amp', {}).setdefault(ktag, {})[sta] = dict(amp=r4(amp), amp_sd=r4(sd),
                                                                            peak_doy=r4((np.degrees(np.arctan2(sv, cv)) % 360) / 360 * 365.25),
                                                                            verdict='not resolved')


    components = {k: dict(label=LABEL[k][0], time_desc=LABEL[k][1], units='% (apparent dv/v)' if k not in ('ramp21', 'trend') else '%/yr (apparent)',
                          signed=LABEL[k][2], kappa_runs=v) for k, v in comps.items()}
    dates, dates_kp = {}, {}
    for r in P[~P.date.str.startswith('amp:')].itertuples():
        (dates if r.kappa == 'k1' else dates_kp).setdefault(r.date, {})[r.sta] = [r4(r.mean), r4(r.sd)]
    # time series: 1-3 Hz SC (else the series with most samples), observed and model, 10-day means
    F = np.load(f'{BIG}/fit{sfx}.npz', allow_pickle=False)
    M = S3.Model(dat, 'k1', KF=KF)
    x = Z['k1|primary|x']
    pred = M.A @ x
    import sci_dvv_attr_lib as L
    ts = {}
    for sta in M.STA:
        cands = [k.split('||')[0] for k in F.files if k.endswith('||y') and k.split('|')[1] == sta and k.split('|')[0] in ('SC10', 'LFE')]
        if not cands:
            continue
        pick = next((c for c in cands if c.endswith('|1-3Hz')), None) or max(cands, key=lambda c: len(F[f'{c}||y']))
        y, fit, day, sh = F[f'{pick}||y'], F[f'{pick}||fit'], F[f'{pick}||day'], F[f'{pick}||shapes']
        rows = dat.D[dat.D.series == pick]
        model = fit.copy()
        for i in rows.index:
            k = S3.AMPS.index(rows.loc[i, 'amp'])
            model += (pred[i] - rows.loc[i, 'est']) * sh[:, k]
        b = day // 10
        ub = np.unique(b)
        tt = [str(L.DAYS[int(u * 10 + 5)].date()) for u in ub]
        ob = [r4(np.mean(y[b == u])) for u in ub]
        mb = [r4(np.mean(model[b == u])) for u in ub]
        lab = pick.split('|')
        ts[sta] = dict(label=(f'{lab[2]} single-station noise' if lab[0] == 'SC10' else f'{lab[0]} {lab[2]}'), t=tt, obs=ob, model=mb)
    # deep statement from stage 1 (clean subset)
    R = pd.read_csv(f'{ROOT}/data/sci/dvv_3d/tt_results.csv')
    deep = []
    for run, vol, lab in [('clean_lay_3d', 'lay3', '20-31 km layer contrast (3-D rays)'), ('clean_lay_1d', 'lay', '20-31 km layer contrast (1-D rays)'),
                          ('clean_box_3d', 'box3', '20-31 km, 20-km box at the LFEs'), ('clean_sph_3d', 'sph3', '10-km spheres around the LFE families'),
                          ('clean_slab_3d', 'slab3', '5-km fault slab'), ('clean_R_lay_3d', 'lay3', '20-31 km layer, absolute (with 1-10, 10-20 km terms)')]:
        for q in ('m7', 'pet', 'fer', 'D'):
            g = R[(R.run == run) & (R.vol == vol) & (R.quantity == q)]
            if len(g):
                r = g.iloc[0]
                deep.append(dict(target=lab, quantity=q, est=r4(r.est_pct), se=r4(r.se), ci95=[r4(r.ci95_lo), r4(r.ci95_hi)],
                                 text=f'{lab}: {q} {r.est_pct:+.2f} +- {r.se:.2f} % (honest SE; stage-1 travel times, clean subset)'))
    fam = pd.read_csv(f'{ROOT}/data/families/shelly_family_locations.csv')
    lfe = []
    for F_ in range(1, 28):
        f = f'{ROOT}/data/locations/fam{F_}_3d.json'
        if os.path.exists(f):
            d = json.load(open(f))
            lfe.append(dict(fam=F_, lat=r4(d['lat']), lon=r4(d['lon']), depth=r4(d['depth_km'])))
    stations = [dict(sta=s, lat=r4(ST.loc[s, 'lat']), lon=r4(ST.loc[s, 'lon']),
                     kind='borehole' if s.startswith('PB.') else 'surface', sensor_depth_m=r4(ST.loc[s, 'depth_m']))
                for s in M.STA if s in ST.index]
    meta = dict(description='Stage-3 unified dv/v model: station-local shallow apparent dv/v (two depth bins 0-0.3 and 0.3-3 km per '
                            'station and component) + deep bound from stage-1 travel times. Below 3 km coda data only feed a nuisance; '
                            'no deep field is estimated. Amplitudes are APPARENT (coupling of coda to its kernels uncertain x2.5-10): '
                            'k1 = coupling 1, kprof = stage-2 profile (SC 10, LFE 5, REP 2.5).',
                units='%', tau_q_days=taus, time_functions={k: v[1] for k, v in LABEL.items()},
                depth_verdict_rule='best one-parameter depth hypothesis beats both others by dchi2 >= 4 in k1 and kprof and under prior x0.3/x3',
                variogram=VG)
    # GP maps (sci_dvv_3d_s3_map.py): per component x kappa run and per date
    if os.path.exists(f'{BIG}/maps{sfx}.npz'):
        Mz = np.load(f'{BIG}/maps{sfx}.npz', allow_pickle=False)
        MS = pd.read_csv(f'{OUT}/maps_summary{sfx}.csv')
        lat, lon = [r4(v) for v in Mz['lat']], [r4(v) for v in Mz['lon']]

        def lay(key, names=('apparent', '0-0.3', '0.3-3')):
            o = {}
            for L_ in names:
                if f'{key}|{L_}|mean' not in Mz.files:
                    continue
                m, sd_, mk = Mz[f'{key}|{L_}|mean'], Mz[f'{key}|{L_}|sd'], Mz[f'{key}|{L_}|mask']
                o[L_] = dict(mean=[[r4(v) for v in row] for row in m], sd=[[(None if not np.isfinite(v) else round(float(v), 3)) for v in row] for row in sd_],
                             mask=[[int(v > 0) for v in row] for row in mk])
            return o or None
        for comp, c in components.items():
            for ktag in c['kappa_runs']:
                lo = lay(f'{ktag}|{comp}')
                if lo is None:
                    continue
                ms = MS[(MS.kappa == ktag) & (MS.comp == comp)]
                if ms.empty:
                    continue
                m_ = ms.iloc[0]
                c.setdefault('map', {})[ktag] = dict(
                    lat=lat, lon=lon, layers=lo, units='% apparent (1-3 Hz kernel-weighted functional, kappa of the run)',
                    corr_len_km=r4(m_.L_km), corr_len_ci=[r4(m_.L_lo), r4(m_.L_hi)], L_not_bounded=bool(m_.L_not_bounded),
                    coherent_fraction=r4(m_.phi), coherent_fraction_ci=[r4(m_.phi_lo), r4(m_.phi_hi)],
                    depth_ratio_r=r4(m_.r), depth_ratio_ci=[r4(m_.r_lo), r4(m_.r_hi)], perm_p=r4(m_.perm_p),
                    verdict='coherent field' if bool(m_.coherent) else 'station-specific', text=m_.text,
                    mask_definition='1 = farther than L_lo from any station (fade by SD elsewhere)',
                    net_mean=r4(m_.net_mean), net_se=r4(m_.net_se),
                    depth_split='separate bins mapped' if bool(m_.split_maps) else 'depth split prior-dominated')
        date_maps = {}
        for ktag in ('k1', 'kprof'):
            for lab in dates:
                lo = lay(f'{ktag}|date:{lab}', ('apparent',))
                if lo is not None:
                    used = [str(x) for x in Mz[f'{ktag}|date:{lab}|used']]
                    sp_ = [str(x) for x in Mz[f'{ktag}|date:{lab}|station_specific']]
                    date_maps.setdefault(ktag, {})[lab] = dict(lat=lat, lon=lon, layers=lo, used=[u for u in used if u != 'none'],
                                                               station_specific=[u for u in sp_ if u != 'none'])
    else:
        date_maps = {}
    VS = pd.read_csv(f'{OUT}/validation_summary{sfx}.csv') if os.path.exists(f'{OUT}/validation_summary{sfx}.csv') else pd.DataFrame()
    meta['validation_summary'] = VS.astype(str).to_dict('records')
    meta['notes'] = ['Predictions: relative 0-3 km part only; ENV = annual part only; no dose / microseism terms; the 2024-12-20 '
                     'value is tau-sensitive (network tau; see the tau x0.5 / x2 runs).',
                     'ENV_amp is a magnitude (positively biased where cos / sin are noisy).',
                     'Depth bins are exported only where both are data-constrained (posterior/prior < 0.7) and agree with the '
                     'inverted-prior run; otherwise "split prior-dominated" and only the kernel-weighted 0-3 km amplitude.',
                     'Verdicts rank shallow hypotheses only; deep changes are indistinguishable from ">= 1 km"; the deep crust '
                     'is bounded by stage 1 only. Repeaters reuse the LFE kernel fractions.',
                     'LFE coda from unregularised monthly bins (data/sci/decline/monthly_bins.csv); the regularised pool_clean '
                     'curves are only a variant.']
    fl = f'{OUT}/depth_verdicts_env_nolowbands{sfx}.csv'
    if os.path.exists(fl):
        E = pd.read_csv(fl)
        deep = E[E.verdict_all_bands.str.startswith('>=1')]
        meta['env_lowband_test'] = deep[['sta', 'amp', 'verdict_all_bands', 'verdict_no_lowbands']].to_dict('records')
        gone = deep[~deep.verdict_no_lowbands.str.startswith('>=1')]
        if len(gone):
            meta['notes'].append('Annual (ENV) signal at >= 1 km at ' + ', '.join(sorted(set(gone.sta))) + ' disappears without the '
                                 '0.1-0.6 Hz SC bands and LP: it likely reflects seasonal noise-source (microseism) changes in the '
                                 'low bands, not a deep hydrological signal.')
        kept = deep[deep.verdict_no_lowbands.str.startswith('>=1')]
        if len(kept):
            meta['notes'].append('Annual (ENV) >= 1 km verdict survives without the low bands at ' + ', '.join(sorted(set(kept.sta)))
                                 + ' (microseism-power regressor already in step 1); treat with caution.')
    out = dict(meta=meta, stations=stations, lfe=lfe, components=components, dates=dates, dates_kprof=dates_kp,
               date_maps=date_maps, timeseries=ts, deep=deep, quakes=quakes())
    js = json.dumps(out, allow_nan=False, separators=(',', ':'), default=lambda o: None)
    open(f'{OUT}/viewer{sfx}.json', 'w').write(js)
    print('viewer.json', len(js) / 1e6, 'MB', flush=True)
    # ------------------------------------------------------------------ figures
    figs(T, V, sfx, ST)


def figs(T, V, sfx, ST):
    vis = ['ENV_cos', 'level_M7', 'drop0_M7', 'ramp21', 'level_Petrolia', 'level_Ferndale']
    fig, axs = plt.subplots(2, len(vis), figsize=(4 * len(vis), 8))
    for j, nm in enumerate(vis):
        for i, (col, sdc, lab) in enumerate((('a_0_0p3', 'a_sd', '0-0.3 km'), ('b_0p3_3', 'b_sd', '0.3-3 km'))):
            ax = axs[i, j]
            g = T[(T.kappa == 'k1') & (T.cfg == 'primary') & (T.amp == nm) & T.sta.isin(ST.index)]
            if g.empty:
                continue
            vm = np.nanpercentile(np.abs(g[col]), 95) or 0.1
            z = np.abs(g[col] / g[sdc])
            sc = ax.scatter(ST.loc[g.sta, 'lon'], ST.loc[g.sta, 'lat'], c=g[col], cmap='RdBu', vmin=-vm, vmax=vm,
                            s=20 + 25 * np.clip(z, 0, 4), edgecolors=np.where(g[f'{col[0]}_post_over_prior'] > 0.7, '0.6', 'k'))
            for _, r in g.iterrows():
                vd = V[(V.sta == r.sta) & (V.amp == nm)]
                tag = '' if vd.empty or vd.verdict.iloc[0] == 'not resolved' else '*'
                ax.annotate(r.sta.split('.')[1] + tag, (ST.loc[r.sta, 'lon'], ST.loc[r.sta, 'lat']), fontsize=5)
            ax.plot(-123.56, 40.05, 'k*', ms=8)
            plt.colorbar(sc, ax=ax, label='apparent dv/v (%)')
            ax.set_title(f'{nm} {lab} (k=1)', fontsize=8)
    fig.suptitle('Stage-3 station-local shallow terms (symbol size ~ |z|; grey edge = posterior/prior SD > 0.7; * = depth verdict '
                 'resolved). Below 3 km: not estimated - see the stage-1 deep bound.', fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_3d_s3_stations{sfx}.png', dpi=120)
    plt.close(fig)
    Vd = pd.read_csv(f'{OUT}/validation{sfx}.csv')
    labs = ['top ~0.3 km', 'uniform 0-3 km', '>=1 km (lower limit only)', 'not resolved', 'no significant signal']
    D = Vd.groupby(['truth', 'amp_true', 'kernel']).verdict.agg(lambda s_: [np.mean(s_ == l_) for l_ in labs])
    D = pd.DataFrame(D.tolist(), index=D.index, columns=labs)
    fig, ax = plt.subplots(figsize=(7, 9))
    im = ax.imshow(D.values, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(D.shape[1]))
    ax.set_xticklabels(D.columns, rotation=30, fontsize=7)
    ax.set_yticks(range(len(D)))
    ax.set_yticklabels([f'{a} {b:g} | {c}' for a, b, c in D.index], fontsize=6)
    plt.colorbar(im, ax=ax, label='fraction of stations x noise realisations with this final verdict')
    ax.set_title('Injected depth truths (rows: truth, amplitude %, generating kernel) -> final verdict (full rule)', fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_3d_s3_validation{sfx}.png', dpi=120)
    plt.close(fig)


if __name__ == '__main__':
    main(len(sys.argv) > 1 and sys.argv[1] == 'smoke')
