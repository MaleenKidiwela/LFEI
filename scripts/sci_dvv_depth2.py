#!/usr/bin/env python3
"""How deep does the post-2021 dv/v decline (and the M7 step) extend?   (dv/v depth study, second attempt)

All LFE-coda numbers use the instrument-cleaned windows and the free-offset scheme V2 of
scripts/sci_dvv_instrument_check.py (windows excluded on all components, free reference offsets at the ends of
the B933 / BRIC / RVIT / BJES windows and at the KMPB 2023-10-17 maintenance).  Every error is calibrated with
placebo (shifted-date / random-date) fits; formal errors are not used for inference.

Stages
  depthreg  Test 1. sensor depth of each PB borehole (data/stations metadata); per-channel V2 decline
            (2021->2025, steps+ramp model, free offsets) and M7 step (param fit) with per-channel placebo nulls;
            station combination with the placebo vectors (keeps inter-channel correlation); regression on
            sensor depth (surface = 0), intrinsic site scatter, placebo-calibrated slope null.
  lfefit    Test 2/3 LFE coda. pair rows from scripts/sci_dvv_depth2_measure.py (6 families; 2-4 / 4-8 / 2-8 Hz
            x full / early 3-8 s / late 8 s-end) -> V2 masks/segments -> per (channel, config) parametric fit
            (annual + Petrolia/Ferndale/M7 steps + 60-d M7 transient + ramp from 2021-06 + reference offsets per
            family x segment) -> decline D(2021->2025) and M7 step; injection (-0.1 % after the M7, identical
            stacks) gives a per-(channel, config) recovery factor; placebo = the same fit with all model dates
            shifted back 5.25..12.25 yr on pre-2021-06 rows (15 shifts) -> covariance across configs.
  noise     Test 2/3 ambient noise. data/sci/noise_dvv/ccf_*.npz (cached daily CCFs) -> line notch (as
            sci_noise_dvv) -> sub-band filtering of the whitened CCFs (1-2, 1-4 | 2-4, 2-8, 4-8 Hz) -> stretching
            in early (2-8 s) / late (8-20 s) / full (2-20 s) lapse windows -> 10-d stacks -> fit (free offsets at
            the instrument breaks, BRIC 2022-12..2024-05 masked as in the instrument check, annual, 3 steps)
            -> D = sum of steps, M7 step; placebo = 3 fake steps at random pre-2021-11 dates; injection -0.1 %.
  kernels   depth kernels (scripts/sci_dvv_depth2_kernels.py) for every observable and model variant
  invert    joint Bayesian grid inversion for dv/v(z): uniform-to-H, exponential(D), two layers; nuisance
            surface-wave fractions; per-station amplitudes; z50 / z90 posteriors; model-variant sensitivity
  figs      figures/sci_dvv_depth2_*.png
Outputs: data/sci/dvv_depth2/
"""
import argparse
import json
import os
import pickle
import sys
import zlib
from multiprocessing import Pool

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import numpy as np
import pandas as pd

ROOT = '/home/seismic/LFE'
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
OUT = 'data/sci/dvv_depth2'
FIG = 'figures'
os.makedirs(OUT, exist_ok=True)
NPROC = 8
SHIFTS = np.arange(5.25, 12.3, 0.5)
Q = {'pet': pd.Timestamp('2021-12-20T20:10'), 'fer': pd.Timestamp('2022-12-20T10:34'), 'm7': pd.Timestamp('2024-12-05T18:44')}


def jdump(obj, path):
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    json.dump(obj, open(path, 'w'), indent=1, default=conv)


def summary_update(key, val):
    p = f'{OUT}/summary.json'
    S = json.load(open(p)) if os.path.exists(p) else {}
    S[key] = val
    jdump(S, p)


def sta_of(ch):
    return ch.rsplit('.', 2)[0]


# ================================================================================================ metadata
def sensor_depths():
    rows = []
    for f, src in [('data/stations/cha_iris_PB_2008_2026.txt', 'IRIS'), ('data/stations/cha_ncedc_2008_2026.txt', 'NCEDC')]:
        m = pd.read_csv(f, sep='|', dtype=str)
        m.columns = [c.strip().lstrip('#').strip() for c in m.columns]
        for c in m.columns:
            m[c] = m[c].fillna('').astype(str).str.strip()
        m = m[m.Channel.str.match(r'^(HH|EH)[ZNE12]$')]
        m['src'] = src
        rows.append(m)
    M = pd.concat(rows)
    M['ns'] = M.Network + '.' + M.Station
    M['Depth'] = pd.to_numeric(M.Depth, errors='coerce')
    M['Elevation'] = pd.to_numeric(M.Elevation, errors='coerce')
    want = ['PB.B045', 'PB.B046', 'PB.B047', 'PB.B049', 'PB.B932', 'PB.B933', 'PB.B934', 'PB.B935', 'NC.KMPB', 'NC.KMR',
            'BK.BRIC', 'BK.HAYF', 'BK.RBOW', 'BK.RVIT', 'BK.BJES']
    M = M[M.ns.isin(want)]
    # depth used for 2008-2026 epochs before the 2025-26 KMR changes
    M = M[M.StartTime.str[:4] < '2025']
    T = M.groupby('ns').agg(depth_m=('Depth', 'median'), depth_min=('Depth', 'min'), depth_max=('Depth', 'max'),
                            elev_m=('Elevation', 'median'), lat=('Latitude', 'first'), lon=('Longitude', 'first'),
                            source=('src', 'first')).reset_index()
    T['borehole'] = T.ns.str.startswith('PB.')
    T.to_csv(f'{OUT}/sensor_depths.csv', index=False)
    return T


# ================================================================================================ Test 1
def wls_scatter(y, s, X):
    """WLS with intrinsic scatter (chi2/dof = 1); returns beta, se, sig_int."""
    def fit(si):
        w = 1 / (s ** 2 + si ** 2)
        A = X * np.sqrt(w)[:, None]
        b = np.linalg.lstsq(A, y * np.sqrt(w), rcond=None)[0]
        cov = np.linalg.pinv(A.T @ A)
        chi = np.sum(w * (y - X @ b) ** 2) / max(len(y) - X.shape[1], 1)
        return b, cov, chi
    b, cov, chi0 = fit(0.0)
    si = 0.0
    if chi0 > 1:
        lo, hi = 0.0, 1.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if fit(mid)[2] > 1 else (lo, mid)
        si = hi
        b, cov, _ = fit(si)
    return b, np.sqrt(np.diag(cov)), si


def stage_depthreg(args):
    import sci_dvv_instrument_check as ic
    ic.OUT = 'data/sci/dvv_instrument_check'
    T = sensor_depths()
    dep = dict(zip(T.ns, np.where(T.borehole, T.depth_m, 0.0)))
    B = ic.get_breaks()
    V = ic.load_variant_bins()
    Vd = {ch: ic.deseason(b) for ch, b in V['V1'].items()}
    # ---- decline per channel (V2 free offsets, steps+ramp) + placebo shifts
    rows = []
    for ch, b in Vd.items():
        br = B['V2'].get(sta_of(ch), [])
        for kind in ['steps+ramp', 'steps']:
            obs = ic.model_decline_channel(b, br, 0.0, kind)
            pl = np.array([ic.model_decline_channel(b, br, s, kind) for s in SHIFTS])
            rows.append(dict(chan=ch, sta=sta_of(ch), kind=kind, D=obs, **{f'pl{i}': v for i, v in enumerate(pl)}))
    DC = pd.DataFrame(rows)
    DC.to_csv(f'{OUT}/t1_decline_per_channel_placebo.csv', index=False, float_format='%.5g')
    # ---- M7 step per channel: param fits V2 (placebo_steptr at 200 random pre-M7 dates)
    # project-internal cache written by sci_dvv_instrument_check.py (trusted)
    PF = pickle.load(open('data/sci/dvv_instrument_check/param_fits.pkl', 'rb'))
    srow = []
    for (v, ch), r in PF['res'].items():
        if v != 'V2' or r is None or not r['ident']['m7']:
            continue
        pl = r['placebo_steptr'] * 100
        srow.append(dict(chan=ch, sta=sta_of(ch), step=r['m'][5] * 100, **{f'pl{i}': x for i, x in enumerate(pl)}))
    SC = pd.DataFrame(srow)
    SC.to_csv(f'{OUT}/t1_m7step_per_channel_placebo.csv', index=False, float_format='%.5g')

    def station_table(C, val, npl):
        out = []
        plc = [f'pl{i}' for i in range(npl)]
        for sta, g in C.groupby('sta'):
            g = g[np.isfinite(g[val])]
            P = g[plc].values
            ok = np.isfinite(P).all(axis=0)
            if len(g) == 0:
                continue
            if ok.sum() < 8:          # short record (starts 2019-20): placebo SD scaled from the M7-step placebo
                out.append(dict(sta=sta, depth_m=dep.get(sta, np.nan), n_chan=len(g), value=float(g[val].mean()),
                                placebo_mean=np.nan, placebo_sd=np.nan, z=np.nan, n_placebo=0, chans=';'.join(g.chan)))
                continue
            # channel weights from the channel placebo SDs; station placebo = same weighted mean per shift
            sd = np.nanstd(P[:, ok], axis=1, ddof=1)
            w = 1 / np.maximum(sd, 1e-4) ** 2
            w /= w.sum()
            v = float(np.sum(w * g[val].values))
            nul = (w[:, None] * P[:, ok]).sum(0)
            out.append(dict(sta=sta, depth_m=dep.get(sta, np.nan), n_chan=len(g), value=v, placebo_mean=nul.mean(),
                            placebo_sd=nul.std(ddof=1), z=(v - nul.mean()) / nul.std(ddof=1), n_placebo=int(ok.sum()),
                            chans=';'.join(g.chan)))
        return pd.DataFrame(out)

    res = {}
    SM = station_table(SC, 'step', len(PF['dates'])).set_index('sta')
    for name, C, val, npl in [('m7step', SC, 'step', len(PF['dates'])),
                              ('decline', DC[DC.kind == 'steps+ramp'], 'D', len(SHIFTS)),
                              ('decline_steps', DC[DC.kind == 'steps'], 'D', len(SHIFTS))]:
        S = station_table(C, val, npl).dropna(subset=['depth_m'])
        # stations without a shifted-date placebo: SD = (median station ratio decline-SD / M7-step-SD) x own M7 SD
        both = S[S.placebo_sd.notna() & S.sta.isin(SM.index)]
        rr = (both.placebo_sd.values / SM.loc[both.sta, 'placebo_sd'].values)
        ratio = float(np.median(rr)) if len(rr) else np.nan
        miss = S.placebo_sd.isna() & S.sta.isin(SM.index)
        S.loc[miss, 'placebo_sd'] = ratio * SM.loc[S.sta[miss], 'placebo_sd'].values
        S.loc[miss, 'z'] = S.value[miss] / S.placebo_sd[miss]
        S['sd_source'] = np.where(miss, f'scaled_from_m7_placebo_x{ratio:.2f}', 'shifted_date_placebo')
        S = S.dropna(subset=['placebo_sd'])
        S.to_csv(f'{OUT}/t1_{name}_per_station.csv', index=False, float_format='%.5g')
        y, s, d = S.value.values, S.placebo_sd.values, S.depth_m.values / 1000.0
        one = np.ones(len(S))
        bore = (d > 0).astype(float)
        r = {}
        for mn, X in {'linear_depth': np.c_[one, d], 'borehole_indicator': np.c_[one, bore],
                      'boreholes_only_linear': None}.items():
            if X is None:
                m = d > 0
                X, yy, ss = np.c_[one[m], d[m]], y[m], s[m]
            else:
                yy, ss = y, s
            b, se, si = wls_scatter(yy, ss, X)
            # station bootstrap of the slope
            rng = np.random.default_rng(1)
            bs = []
            for _ in range(2000):
                ii = rng.integers(0, len(yy), len(yy))
                if np.ptp(X[ii, 1]) == 0:
                    continue
                bs.append(wls_scatter(yy[ii], ss[ii], X[ii])[0][1])
            # placebo null of the slope (noise only): same regression on the station placebo realisations
            r[mn] = dict(n=len(yy), intercept=b[0], intercept_se=se[0], slope=b[1], slope_se=se[1], sig_int=si,
                         slope_boot_se=float(np.std(bs)), slope_boot_p16_p84=np.percentile(bs, [16, 84]).tolist(),
                         ratio_slope_over_intercept=b[1] / b[0] if b[0] != 0 else np.nan)
        # placebo slope null without site scatter (pure measurement noise): regress placebo vectors
        plc = None
        # ratio surface / borehole means (weighted, with site scatter)
        sm = S[S.depth_m == 0]
        bm = S[S.depth_m > 0]
        r['surface_mean'] = dict(value=float(sm.value.mean()), se=float(sm.value.std(ddof=1) / np.sqrt(len(sm))), n=len(sm))
        r['borehole_mean'] = dict(value=float(bm.value.mean()), se=float(bm.value.std(ddof=1) / np.sqrt(len(bm))), n=len(bm))
        r['ratio_surface_over_borehole'] = float(sm.value.mean() / bm.value.mean())
        # exponential sensitivity decay with sensor depth: y = A exp(-d / L) (profile likelihood over L)
        Ls = np.r_[np.logspace(-2, 1.5, 120)]
        ll = []
        for L in Ls:
            X = np.exp(-d / L)[:, None]
            b, se_, si = wls_scatter(y, s, X)
            w = 1 / (s ** 2 + si ** 2)
            ll.append(-0.5 * np.sum(w * (y - X[:, 0] * b[0]) ** 2) + 0.5 * np.sum(np.log(w)))
        ll = np.array(ll)
        p = np.exp(ll - ll.max())
        p /= p.sum()
        cdf = np.cumsum(p)
        r['exp_sensor_depth_scale_km'] = dict(map=float(Ls[np.argmax(ll)]), p05=float(np.interp(0.05, cdf, Ls)),
                                              p50=float(np.interp(0.5, cdf, Ls)), p95=float(np.interp(0.95, cdf, Ls)),
                                              note='log-uniform prior 0.01-30 km; L > 30 km = no decay')
        r['stations'] = S[['sta', 'depth_m', 'value', 'placebo_sd', 'z', 'n_chan', 'sd_source']].round(4).to_dict('records')
        r['placebo_sd_ratio_decline_over_m7'] = ratio
        res[name] = r
        print(name, json.dumps({k: v for k, v in r.items() if k != 'stations'}, default=float, indent=0)[:1500], flush=True)
    summary_update('test1_sensor_depth', dict(sensor_depths=T[['ns', 'depth_m', 'elev_m', 'borehole']].to_dict('records'), **res))



# ================================================================================================ noise (tests 2/3)
NBANDS = [('1-2Hz', 0, (1.0, 2.0)), ('1-4Hz', 0, None), ('2-4Hz', 1, (2.0, 4.0)), ('2-8Hz', 1, None), ('4-8Hz', 1, (4.0, 8.0))]
NWIN = {'early': (2.0, 8.0), 'late': (8.0, 20.0), 'full': (2.0, 20.0)}
NEPS = np.arange(-0.02, 0.02 + 1e-12, 0.0001)
NOISE_STA = ['KMPB', 'BRIC', 'B045', 'HAYF']
INJ = -0.001


def _stretch_win(ref, X, lags_u, mask_u):
    from scipy.signal import resample
    X = np.atleast_2d(X)
    oe, oc = np.full(len(X), np.nan), np.full(len(X), np.nan)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(ref).all()
    if not ok.any():
        return oe, oc
    t = lags_u[mask_u]
    refu = resample(ref, len(lags_u))
    R = np.array([np.interp(t * (1 + e), lags_u, refu) for e in NEPS])
    R -= R.mean(axis=1, keepdims=True)
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    Y = resample(X[ok], len(lags_u), axis=1)[:, mask_u]
    Y = Y - Y.mean(axis=1, keepdims=True)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    C = R @ Y.T
    j = np.argmax(C, axis=0)
    jj = np.clip(j, 1, len(NEPS) - 2)
    k = np.arange(C.shape[1])
    y0, y1, y2 = C[jj - 1, k], C[jj, k], C[jj + 1, k]
    den = y0 - 2 * y1 + y2
    off = np.clip(np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / den, 0.0), -1, 1)
    e = NEPS[jj] + off * (NEPS[1] - NEPS[0])
    e[(j == 0) | (j == len(NEPS) - 1)] = np.nan
    oe[ok] = e
    oc[ok] = y1 - 0.25 * (y0 - y2) * off
    return oe, oc


def _weaver(cc, band, win):
    f1, f2 = band
    T = 1.0 / (f2 - f1)
    wc = 2 * np.pi * np.sqrt(f1 * f2)
    t1, t2 = win
    cc = np.clip(cc, 1e-3, 0.999)
    return np.sqrt(1 - cc ** 2) / (2 * cc) * np.sqrt(6 * np.sqrt(np.pi / 2) * T / (wc ** 2 * (t2 ** 3 - t1 ** 3)))


def noise_measure(name):
    """per station: (inj in {0,1}) x band x window x pair 10-day dv/v series + quality; returns DataFrame rows."""
    import sci_noise_dvv as nd
    import sci_dvv_instrument_check as ic
    days, ccf = nd.load_ccf(name)
    keep, lines = nd.detect_lines(days, ccf)
    ccf = nd.notch(ccf, keep)
    for a, b in ic.NOISE_MASK[name]:
        ccf[(days >= a) & (days < b)] = np.nan
    L = ccf.shape[-1]
    lags = nd.LAGS
    UP = 4
    lags_u = lags[0] + np.arange(L * UP) / (nd.FS * UP)
    freqs = np.fft.rfftfreq(nd.NF, 1 / nd.FS)
    # sub-band CCFs (filter the whitened, notched CCF along lag)
    sub = np.full((len(days), len(NBANDS), 6, L), np.nan, dtype='float32')
    for ib, (bn, src, fb) in enumerate(NBANDS):
        if fb is None:
            sub[:, ib] = ccf[:, src]
            continue
        W = nd._cos_taper_band(freqs, *fb)
        for i in range(0, len(days), 300):
            c = ccf[i:i + 300, src]
            X = np.fft.rfft(np.nan_to_num(c), nd.NF, axis=-1) * W
            o = np.fft.irfft(X, nd.NF, axis=-1)[..., :L]
            o[~np.isfinite(c)] = np.nan
            sub[i:i + 300, ib] = o
    del ccf
    post = np.asarray(days > Q['m7'])
    k10 = ((days - pd.Timestamp(nd.START)).days // 10).values
    rows = []
    for inj in (0, 1):
        S = sub
        if inj:
            S = sub.copy()
            for i in np.where(post)[0]:
                # stretch each trace: cur(t) = orig(t (1 + INJ))  -> dv/v = INJ
                for ib in range(len(NBANDS)):
                    for ip in range(6):
                        x = sub[i, ib, ip]
                        S[i, ib, ip] = np.interp(lags * (1 + INJ), lags, x) if np.isfinite(x).all() else np.nan
        ref = np.nanmean(S[np.asarray(days <= pd.Timestamp(nd.REF_END))], axis=0)
        labs, S10, N10 = nd.stack_groups(days, S, k10, 5)
        t10 = pd.Timestamp(nd.START) + pd.to_timedelta(labs * 10 + 5, unit='D')
        for ib, (bn, src, fb) in enumerate(NBANDS):
            band = fb if fb is not None else nd.BANDS[src]
            for wn, win in NWIN.items():
                mask_u = (np.abs(lags_u) >= win[0]) & (np.abs(lags_u) <= win[1])
                for ip in range(6):
                    dv, cc = _stretch_win(ref[ib, ip], S10[:, ib, ip], lags_u, mask_u)
                    er = _weaver(cc, band, win)
                    for g in range(len(t10)):
                        rows.append((name, inj, bn, wn, nd.PAIR_NAMES[ip], t10[g], dv[g], cc[g], er[g], N10[g, ib, ip]))
        print(name, 'inj', inj, 'measured', flush=True)
    return pd.DataFrame(rows, columns=['sta', 'inj', 'band', 'win', 'pair', 't', 'dvv', 'cc', 'err', 'ndays'])


def stage_noise(args):
    out = []
    for n in NOISE_STA:                      # sequential: each CCF array is ~0.6 GB in float64
        out.append(noise_measure(n))
    D = pd.concat(out, ignore_index=True)
    D['dvv'] = D.dvv.astype('float32')
    D['cc'] = D.cc.astype('float32')
    D['err'] = D.err.astype('float32')
    D.to_parquet(f'{OUT}/noise_10day_pairs.parquet', compression='zstd')


NCOMBOS = {'ALL': ['ZZ', 'NN', 'EE', 'EN', 'EZ', 'NZ'], 'ZZ': ['ZZ'], 'AC': ['ZZ', 'NN', 'EE'], 'SC': ['EN', 'EZ', 'NZ']}


def noise_series(D):
    """combine pairs (inverse variance, pairs with 10-d reference-period median CC >= 0.3 in the FULL window of
    that band, so early/late use the same pairs)."""
    q = D[(D.inj == 0) & (D.win == 'full') & (D.t <= '2021-11-30')].groupby(['sta', 'band', 'pair']).cc.median()
    good = set(q[q >= 0.3].index)
    D = D[[(s, b, p) in good for s, b, p in zip(D.sta, D.band, D.pair)]]
    rows = []
    for cmb, prs in NCOMBOS.items():
        d = D[D.pair.isin(prs) & np.isfinite(D.dvv) & np.isfinite(D.err)]
        w = 1 / d.err.astype(float) ** 2
        g = d.assign(w=w, wd=w * d.dvv.astype(float)).groupby(['sta', 'inj', 'band', 'win', 't'])
        s = g[['w', 'wd']].sum()
        s['dvv_pct'] = s.wd / s.w * 100
        s['err_pct'] = 1 / np.sqrt(s.w) * 100
        s['npair'] = g.size()
        rows.append(s.reset_index().assign(combo=cmb)[['sta', 'inj', 'band', 'win', 'combo', 't', 'dvv_pct', 'err_pct', 'npair']])
    return pd.concat(rows, ignore_index=True), sorted(good)


def noise_design(t, breaks, steps_at, extra_steps=()):
    """offset per segment + annual + steps (identifiable only) + extra (placebo) steps."""
    seg = (np.searchsorted(np.array([pd.Timestamp(b) for b in breaks], dtype='datetime64[ns]'), t.values, side='right')
           if breaks else np.zeros(len(t), int))
    ty = (t - pd.Timestamp('2019-01-01')).dt.days.values / 365.25
    cols, names = [], []
    for s_ in np.unique(seg):
        cols.append((seg == s_).astype(float)); names.append(f'off{s_}')   # noqa: E702
    cols += [np.cos(2 * np.pi * ty), np.sin(2 * np.pi * ty)]
    names += ['acos', 'asin']
    for k, q in steps_at.items():
        okq = any(((seg == s_) & (t.values < np.datetime64(q)) & (t.values > np.datetime64(q - pd.Timedelta(days=365)))).sum() >= 3
                  and ((seg == s_) & (t.values > np.datetime64(q)) & (t.values < np.datetime64(q + pd.Timedelta(days=365)))).sum() >= 3
                  for s_ in np.unique(seg))
        if okq:
            cols.append((t.values > np.datetime64(q)).astype(float)); names.append('step_' + k)   # noqa: E702
    for i, q in enumerate(extra_steps):
        cols.append((t.values > np.datetime64(q)).astype(float)); names.append(f'pl{i}')   # noqa: E702
    return np.array(cols).T, names


def stage_noisefit(args):
    import sci_dvv_instrument_check as ic
    D = pd.read_parquet(f'{OUT}/noise_10day_pairs.parquet')
    D['t'] = pd.to_datetime(D.t)
    S, good = noise_series(D)
    S.to_csv(f'{OUT}/noise_10day_series.csv.gz', index=False, float_format='%.5g')
    rng = np.random.default_rng(42)
    NPL = 300
    rows = []
    for sta in NOISE_STA:
        br = ic.NOISE_BREAK[sta]
        s0 = S[(S.sta == sta) & (S.inj == 0)]
        first = s0.t.min()
        # placebo: 3 random step dates (>= 60 d apart) in [first + 90 d, 2021-10-15]; same dates for all configs
        lo, hi = first + pd.Timedelta(days=90), pd.Timestamp('2021-10-15')
        PLD = []
        while len(PLD) < NPL:
            x = np.sort(lo + (hi - lo) * rng.random(3))
            if np.min(np.diff(x)) > pd.Timedelta(days=60):
                PLD.append(x)
        for (band, win, cmb), g in s0.groupby(['band', 'win', 'combo']):
            g = g.dropna(subset=['dvv_pct']).sort_values('t')
            if len(g) < 60:
                continue
            t = g.t.reset_index(drop=True)
            y = g.dvv_pct.values
            X, names = noise_design(t, br, Q)
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            co = dict(zip(names, beta))
            stp = [n for n in names if n.startswith('step_')]
            Dm = sum(co[n] for n in stp)
            gi = S[(S.sta == sta) & (S.inj == 1) & (S.band == band) & (S.win == win) & (S.combo == cmb)].dropna(subset=['dvv_pct']).sort_values('t')
            Xi, ni = noise_design(gi.t.reset_index(drop=True), br, Q)
            bi = dict(zip(ni, np.linalg.lstsq(Xi, gi.dvv_pct.values, rcond=None)[0]))
            Di = sum(bi[n] for n in ni if n.startswith('step_'))
            # placebo
            pD, pM = [], []
            pre = (t < pd.Timestamp('2021-11-01')).values
            for dts in PLD:
                Xp, npn = noise_design(t[pre].reset_index(drop=True), [b for b in br if pd.Timestamp(b) < pd.Timestamp('2021-11-01')], {}, dts)
                if np.linalg.matrix_rank(Xp) < Xp.shape[1]:
                    pD.append(np.nan); pM.append(np.nan)   # noqa: E702
                    continue
                bp = np.linalg.lstsq(Xp, y[pre], rcond=None)[0]
                pD.append(bp[-3:].sum())
                pM.append(bp[-1])
            pD, pM = np.array(pD), np.array(pM)
            rows.append(dict(sta=sta, band=band, win=win, combo=cmb, n=len(y), steps=';'.join(stp), D=Dm,
                             step_m7=co.get('step_m7', np.nan), step_fer=co.get('step_fer', np.nan), step_pet=co.get('step_pet', np.nan),
                             D_inj_minus=Di - Dm, rec_D=(Di - Dm) / (INJ * 100),
                             rec_m7=(bi.get('step_m7', np.nan) - co.get('step_m7', np.nan)) / (INJ * 100),
                             annual_amp=np.hypot(co['acos'], co['asin']),
                             D_pl_sd=np.nanstd(pD, ddof=1), m7_pl_sd=np.nanstd(pM, ddof=1),
                             **{f'pD{i}': v for i, v in enumerate(pD)}, **{f'pM{i}': v for i, v in enumerate(pM)}))
        print(sta, 'fitted', flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(f'{OUT}/noise_fits_placebo.csv.gz', index=False, float_format='%.5g')
    base = ['sta', 'band', 'win', 'combo', 'n', 'steps', 'D', 'D_pl_sd', 'rec_D', 'step_m7', 'm7_pl_sd', 'rec_m7', 'annual_amp']
    T = R[base].copy()
    T['D_z'] = T.D / T.D_pl_sd
    T['m7_z'] = T.step_m7 / T.m7_pl_sd
    T.to_csv(f'{OUT}/noise_band_lapse_table.csv', index=False, float_format='%.4g')
    print(T[T.combo == 'ALL'].round(3).to_string(), flush=True)
    summary_update('noise_pairs_used', [list(x) for x in good])


# ================================================================================================ LFE coda (tests 2/3)
LFE_FAMS = [6, 22, 19, 8, 23, 27]
CFGS = ['b28_full', 'b28_early', 'b28_late', 'b24_full', 'b24_early', 'b24_late', 'b48_full', 'b48_early', 'b48_late']
LSHIFTS = np.arange(5.25, 12.3, 0.25)
DAY = 86400.0
MIN_CCREF = 0.25
INJ_LFE = -0.1          # % (sci_dvv_depth2_measure.py --inject ... -0.001)


def _ep(x):
    from dvv_cwire import epoch_s
    return epoch_s([pd.Timestamp(x)])[0]


T_Q = {k: _ep(v) for k, v in Q.items()}
T_RAMP = _ep('2021-06-01')
Y08 = _ep('2008-01-01')
YR = 365.25 * DAY


def lfe_load(tag=''):
    out, meta = [], []
    for F in LFE_FAMS:
        f = f'{OUT}/pairs/fam{F}_pairs{tag}.parquet'
        if not os.path.exists(f):
            continue
        p = pd.read_parquet(f)
        p['chan'] = p.chan.astype(str)
        p['cfg'] = p.cfg.astype(str)
        T = pd.to_datetime(pd.read_csv(f'data/dvv/fam{F}_cwire_stacks.csv').t, format='mixed')
        from dvv_cwire import epoch_s
        T = epoch_s(T)
        p['fam'] = F
        p['tb'] = T[p.b.values.astype(int)]
        p['ta'] = np.where(p.a >= 0, T[np.clip(p.a.values.astype(int), 0, None)], np.nan)
        out.append(p)
        m = pd.read_csv(f'{OUT}/pairs/fam{F}_channels{tag}.csv')
        meta.append(m.assign(fam=F))
    return pd.concat(out, ignore_index=True), pd.concat(meta, ignore_index=True)


def lfe_cols(t, s):
    """model columns at times t (epoch s) with all event dates shifted back by s years."""
    sh = s * YR
    yr = (t - Y08) / YR
    cols = [np.sin(2 * np.pi * yr), np.cos(2 * np.pi * yr)]
    for k in ['pet', 'fer', 'm7']:
        cols.append((t >= T_Q[k] - sh).astype(float))
    dt = (t - (T_Q['m7'] - sh)) / DAY
    cols.append(np.where(dt >= 0, np.exp(-np.clip(dt, 0, None) / 60.0), 0.0))
    cols.append(np.clip((t - (T_RAMP - sh)) / YR, 0, None))
    return np.array(cols).T


LNAMES = ['sin', 'cos', 'pet', 'fer', 'm7', 'tr', 'ramp']


class LfeFit:
    def __init__(self, p):
        self.has_a = (p.a.values >= 0)
        key = p.fam.values * 100 + p.segb.values
        self.k, self.keys = pd.factorize(key)
        self.nk = len(self.keys)
        self.tb, self.ta = p.tb.values, np.where(self.has_a, p.ta.values, p.tb.values)
        self.seg = p.segb.values
        self.d = p.eps.values.astype(float)
        cc = np.clip(p.cc.values.astype(float), 0, 0.99)
        w0 = cc ** 2 / (1 - cc ** 2)
        self.w0 = w0 / w0.mean()
        self.ref = ~self.has_a
        self.sid_b = p.fam.values * 100000 + p.b.values.astype(int)
        self.sid_a = np.where(self.has_a, p.fam.values * 100000 + p.a.values.astype(int), -1)
        self.w = self.w0.copy()

    def design(self, s, rows=None):
        tb, ta = self.tb, self.ta
        F = lfe_cols(tb, s) - np.where(self.has_a[:, None], lfe_cols(ta, s), 0.0)
        # identifiability: each step needs >= 3 stacks within 1 yr on both sides in one segment (rows used)
        r = np.ones(len(tb), bool) if rows is None else rows
        use = np.ones(F.shape[1], bool)
        sh = s * YR
        for j, k in [(2, 'pet'), (3, 'fer'), (4, 'm7')]:
            tq = T_Q[k] - sh
            ok = False
            for sg in np.unique(self.seg[r]):
                x = np.unique(tb[r & (self.seg == sg)])
                if ((x < tq) & (x > tq - YR)).sum() >= 3 and ((x >= tq) & (x < tq + YR)).sum() >= 3:
                    ok = True
            use[j] = ok
        use[5] = use[4]
        x = np.unique(tb[r])
        use[6] = (x < T_RAMP - sh).sum() >= 3 and (x > T_RAMP - sh + YR).sum() >= 3
        return F, use

    def solve(self, w, F, use, irls=0):
        Fu = F[:, use]
        p = Fu.shape[1]
        d, k, ref, nk = self.d, self.k, self.ref, self.nk
        for it in range(irls + 1):
            A11 = (Fu * w[:, None]).T @ Fu
            wr = np.where(ref, w, 0.0)
            A12 = np.zeros((p, nk))
            for j in range(p):
                A12[j] = -np.bincount(k, weights=wr * Fu[:, j], minlength=nk)
            A22 = np.diag(np.bincount(k, weights=wr, minlength=nk)) + 1e-9 * np.eye(nk)
            b1 = Fu.T @ (w * d)
            b2 = -np.bincount(k, weights=wr * d, minlength=nk)
            N = np.block([[A11, A12], [A12.T, A22]])
            x = np.linalg.lstsq(N, np.r_[b1, b2], rcond=None)[0]
            if it < irls:
                r = d - Fu @ x[:p] + np.where(ref, x[p:][k], 0.0)
                z = np.abs(r) * np.sqrt(self.w0) / (1.4826 * np.median(np.abs(r) * np.sqrt(self.w0)) + 1e-12)
                w = self.w0 * np.where(z <= 2.0, 1.0, 2.0 / np.maximum(z, 1e-12))
                self.w = w
        m = np.full(F.shape[1], np.nan)
        m[use] = x[:p]
        return m


def lfe_decline(m, s, use):
    """2025 mean - 2021 mean of the steps + ramp part (shifted frame), %."""
    sh = s * YR
    g0 = np.array([_ep(f'2021-{mm:02d}-15') for mm in range(1, 13)]) - sh
    g1 = np.array([_ep(f'2025-{mm:02d}-15') for mm in range(1, 13)]) - sh
    if not (use[6] or use[2] or use[3] or use[4]):
        return np.nan
    C0, C1 = lfe_cols(g0, s), lfe_cols(g1, s)
    mm = np.where(use, np.nan_to_num(m), 0.0)
    mm[[0, 1, 5]] = 0.0
    return float((C1 @ mm).mean() - (C0 @ mm).mean()) * 100


def lfe_job(args):
    ch, cfg, p, pi, nboot = args
    rng = np.random.default_rng(zlib.crc32((ch + cfg).encode()))
    out = dict(chan=ch, cfg=cfg, n_rows=len(p))
    f = LfeFit(p)
    tb = f.tb
    # need data in 2021 and 2025 (same rule as the instrument check's yearly/model decline)
    in21 = ((tb >= _ep('2021-01-01')) & (tb < _ep('2022-01-01'))).sum()
    in25 = ((tb >= _ep('2025-01-01')) & (tb < _ep('2026-01-01'))).sum()
    F, use = f.design(0.0)
    m = f.solve(f.w0, F, use, irls=3)
    w = f.w.copy()
    out['D'] = lfe_decline(m, 0.0, use) if (in21 >= 3 and in25 >= 3 and use[6]) else np.nan
    out['m7'] = m[4] * 100 if use[4] else np.nan
    out['t_first'] = float(tb.min())
    # injection (identical stacks, -0.1 % after the M7): recovery.  PAIRED: same rows and the same final IRLS
    # weights (only eps differs); 'unpaired' = separate fit of the separately cc-filtered injected rows (the way
    # the earlier sci_dvv_depth recovery was computed) for comparison.
    if pi is not None and len(pi) > 100:
        mg = p[['fam', 'a', 'b']].reset_index().merge(pi[['fam', 'a', 'b', 'eps']].rename(columns={'eps': 'eps_i'}),
                                                       on=['fam', 'a', 'b'], how='left')
        ei = mg.sort_values('index').eps_i.values.astype(float)
        okr = np.isfinite(ei)
        out['inj_rows_matched'] = float(okr.mean())
        d0 = f.d.copy()
        f.d = np.where(okr, ei, d0)
        mi = f.solve(w * okr, F, use)
        f.d = d0
        m0 = f.solve(w * okr, F, use)
        out['rec_D'] = (lfe_decline(mi, 0.0, use) - lfe_decline(m0, 0.0, use)) / INJ_LFE if use[6] else np.nan
        out['rec_m7'] = (mi[4] - m0[4]) * 100 / INJ_LFE if use[4] else np.nan
        fi = LfeFit(pi)
        Fi, ui = fi.design(0.0)
        mu = fi.solve(fi.w0, Fi, ui, irls=3)
        Du = lfe_decline(mu, 0.0, ui) if ui[6] else np.nan
        out['rec_D_unpaired'] = (Du - out['D']) / INJ_LFE if np.isfinite(out['D']) else np.nan
        out['rec_m7_unpaired'] = (mu[4] * 100 - out['m7']) / INJ_LFE if (ui[4] and np.isfinite(out['m7'])) else np.nan
    # placebo: model dates shifted back s yr, rows before 2021-06-01 only, IRLS weights held
    pre = (f.tb < T_RAMP) & (f.ta < T_RAMP)
    for i, s in enumerate(LSHIFTS):
        Fp, up = f.design(s, rows=pre)
        if not (up[4] and up[6]) or pre.sum() < 300:
            out[f'pD{i}'] = np.nan
            out[f'pM{i}'] = np.nan
            continue
        sh = s * YR
        n21 = ((tb[pre] >= _ep('2021-01-01') - sh) & (tb[pre] < _ep('2022-01-01') - sh)).sum()
        n25 = ((tb[pre] >= _ep('2025-01-01') - sh) & (tb[pre] < _ep('2026-01-01') - sh)).sum()
        mp = f.solve(w * pre, Fp, up)
        out[f'pD{i}'] = lfe_decline(mp, s, up) if (n21 >= 3 and n25 >= 3) else np.nan
        out[f'pM{i}'] = mp[4] * 100
    # stack block bootstrap (family, stack), weights held
    ids = np.unique(np.r_[f.sid_b, f.sid_a[f.sid_a >= 0]])
    ib = np.searchsorted(ids, f.sid_b)
    ia = np.searchsorted(ids, np.where(f.sid_a >= 0, f.sid_a, ids[0]))
    bD, bM = [], []
    for _ in range(nboot):
        c = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids)).astype(float)
        mb = f.solve(w * c[ib] * np.where(f.sid_a >= 0, c[ia], 1.0), F, use)
        bD.append(lfe_decline(mb, 0.0, use))
        bM.append(mb[4] * 100)
    out['D_boot_se'] = float(np.nanstd(bD)) if np.isfinite(out['D']) else np.nan
    out['m7_boot_se'] = float(np.nanstd(bM)) if use[4] else np.nan
    return out


def stage_lfefit(args):
    import sci_dvv_instrument_check as ic
    ic.OUT = 'data/sci/dvv_instrument_check'
    B = ic.get_breaks()
    P, Mt = lfe_load('')
    Pi, _ = lfe_load('_inj')
    fams = sorted(P.fam.unique())
    print('families', fams, len(P), len(Pi), flush=True)
    # channel-config QC: reference-row median cc >= MIN_CCREF in each family (from the measurement meta)
    okc = set()
    for c in CFGS:
        col = f'{c}_ccref'
        mm = Mt[Mt[col] >= MIN_CCREF] if col in Mt else Mt.iloc[:0]
        okc |= {(F, ch, c) for F, ch in zip(mm.fam, mm.chan)}
    jobs = []
    for ch, pc in P.groupby('chan'):
        if pc.tb.min() > _ep('2020-12-31'):
            continue                              # no 2021 data: decline undefined
        pci = Pi[Pi.chan == ch]
        for cfg in CFGS:
            p = pc[pc.cfg == cfg]
            p = p[[(F, ch, cfg) in okc for F in p.fam]]
            if len(p) < 1000:
                continue
            q = ic.prep_channel(ch, p.reset_index(drop=True), 'V2', B)
            qi = pci[(pci.cfg == cfg)]
            qi = qi[[(F, ch, cfg) in okc for F in qi.fam]]
            qi = ic.prep_channel(ch, qi.reset_index(drop=True), 'V2', B) if len(qi) else None
            if len(q) >= 1000:
                jobs.append((ch, cfg, q, qi, args.nboot))
    print(len(jobs), 'lfe jobs', flush=True)
    res = []
    with Pool(NPROC) as pool:
        for r in pool.imap_unordered(lfe_job, jobs, chunksize=1):
            res.append(r)
            if len(res) % 50 == 0:
                print(len(res), flush=True)
    R = pd.DataFrame(res)
    R['sta'] = R.chan.map(sta_of)
    R.to_csv(f'{OUT}/lfe_fits_placebo.csv.gz', index=False, float_format='%.5g')
    print('saved', len(R), flush=True)


# ================================================================================================ kernels
ZG = np.r_[0.0, np.logspace(-3, np.log10(40.0), 400)]          # depth grid edges (km)
FREQS = np.geomspace(1.0, 8.0, 13)
BORE_D = {'PB.B045': 0.2176, 'PB.B046': 0.1875, 'PB.B047': 0.1753, 'PB.B049': 0.592, 'PB.B932': 0.412,
          'PB.B933': 0.139, 'PB.B934': 0.492, 'PB.B935': 0.1524}
DEPTHS = np.array([0.0] + sorted(set(BORE_D.values())))
LFE_WIN = {'early': (3.0, 8.0), 'late': (8.0, 14.0), 'full': (3.0, 12.0)}
LFE_L = ['20', '50', '100', '10/5km/100']
NOISE_LSTAR = [1.0, 5.0, 20.0]
VEL_VARIANTS = ['ref', 'soft', 'stiff', 'thin', 'none']


def _kern_variant(v):
    import sci_dvv_depth2_kernels as K
    vm = K.VModel(v)
    out = dict(R=np.zeros((len(FREQS), len(ZG) - 1)), L=np.zeros((len(FREQS), len(ZG) - 1)),
               ampR=np.zeros((len(FREQS), len(DEPTHS))), ampL=np.zeros((len(FREQS), len(DEPTHS))),
               cR=np.zeros(len(FREQS)), cL=np.zeros(len(FREQS)), UR=np.zeros(len(FREQS)), UL=np.zeros(len(FREQS)))
    for i, f in enumerate(FREQS):
        for nm, fn in [('R', K.rayleigh), ('L', K.love)]:
            r = fn(vm, f)
            out[nm][i] = K.on_grid(r, ZG)
            out['amp' + nm][i] = [K.amp_at(r, d) for d in DEPTHS]
            out['c' + nm][i] = r['c']
            out['U' + nm][i] = r['U']
    # LFE body kernels (window x MC scattering variant), plus buried-receiver residual-shift variant
    for wn, win in LFE_WIN.items():
        for l in LFE_L:
            out[f'lfe_{wn}_{l}'] = K.lfe_body_kernel(vm, ZG, win, l_km=l)
            for d in DEPTHS[1:]:
                out[f'lfe_{wn}_{l}_bh{d:.4f}'] = K.lfe_body_kernel(vm, ZG, win, l_km=l, bh_depth=d, f_shift=0.2)
    # vertical S travel time profile (for reporting)
    zc = 0.5 * (ZG[1:] + ZG[:-1])
    out['vs'] = vm(zc)[1]
    out['vp'] = vm(zc)[0]
    return v, out


def stage_kernels(args):
    import sci_dvv_depth2_kernels as K
    K.furlong_profile()
    with Pool(min(NPROC, len(VEL_VARIANTS))) as pool:
        res = dict(pool.map(_kern_variant, VEL_VARIANTS))
    NB = {}
    for wn, win in {'early': (2.0, 8.0), 'late': (8.0, 20.0), 'full': (2.0, 20.0)}.items():
        for ls in NOISE_LSTAR:
            for d in [0.0, BORE_D['PB.B045']]:
                NB[f'noise_{wn}_{ls:g}_z{d:.4f}'] = K.noise_body_kernel(ZG, win, lstar=ls, z0=d)
    flat = {'ZG': ZG, 'FREQS': FREQS, 'DEPTHS': DEPTHS}
    for v, o in res.items():
        for k, x in o.items():
            flat[f'{v}|{k}'] = x
    for k, x in NB.items():
        flat[k] = x
    np.savez_compressed(f'{OUT}/kernels.npz', **flat)
    # summary: depth above which 50 / 90 % of each kernel sits
    zc = ZG[1:]
    rows = []
    for v, o in res.items():
        for i, f in enumerate(FREQS):
            for nm in ['R', 'L']:
                c = np.cumsum(o[nm][i]) / o[nm][i].sum()
                rows.append(dict(variant=v, kind=nm, f=f, c=o['c' + nm][i], U=o['U' + nm][i], z50_km=np.interp(0.5, c, zc),
                                 z90_km=np.interp(0.9, c, zc), amp_at_0p2km=o['amp' + nm][i][np.argmin(abs(DEPTHS - 0.2176))],
                                 amp_at_0p5km=o['amp' + nm][i][np.argmin(abs(DEPTHS - 0.492))]))
        for k, x in o.items():
            if k.startswith('lfe_') and '_bh' not in k:
                c = np.cumsum(x) / x.sum()
                rows.append(dict(variant=v, kind=k, z50_km=np.interp(0.5, c, zc), z90_km=np.interp(0.9, c, zc),
                                 frac_top_0p2km=float(x[zc <= 0.2].sum()), frac_top_1km=float(x[zc <= 1.0].sum())))
    for k, x in NB.items():
        c = np.cumsum(x) / x.sum()
        rows.append(dict(variant='-', kind=k, z50_km=np.interp(0.5, c, zc), z90_km=np.interp(0.9, c, zc),
                         frac_top_0p2km=float(x[zc <= 0.2].sum()), frac_top_1km=float(x[zc <= 1.0].sum())))
    T = pd.DataFrame(rows)
    T.to_csv(f'{OUT}/kernel_summary.csv', index=False, float_format='%.4g')
    print(T.round(3).to_string(), flush=True)


# ================================================================================================ inversion
SHAPE_CFGS = ['b24_early', 'b24_late', 'b48_early', 'b48_late']
NOISE_OBS = [(b, w) for b in ['1-2Hz', '2-4Hz', '4-8Hz'] for w in ['early', 'late']]
BAND_F = {'b24': (2.0, 4.0), 'b48': (4.0, 8.0), 'b28': (2.0, 8.0), '1-2Hz': (1.0, 2.0), '2-4Hz': (2.0, 4.0),
          '4-8Hz': (4.0, 8.0), '1-4Hz': (1.0, 4.0), '2-8Hz': (2.0, 8.0)}
ALPHA = np.linspace(0, 1, 11)
NOISE_STA_FULL = {'KMPB': 'NC.KMPB', 'BRIC': 'BK.BRIC', 'B045': 'PB.B045', 'HAYF': 'BK.HAYF'}


def shrink_cov(Pmat, sd, lam=0.3):
    """covariance from placebo realisations (rows = realisations); correlation shrunk toward identity."""
    X = Pmat[np.isfinite(Pmat).all(axis=1)]
    if len(X) < 8:
        return np.diag(sd ** 2)
    R = np.corrcoef(X.T)
    R = (1 - lam) * R + lam * np.eye(len(sd))
    return R * np.outer(sd, sd)


def load_obs(kind):
    """kind 'D' (decline 2021->2025) or 'm7'.  Returns dict of blocks."""
    blocks = {}
    # --- T1: cross-station full-band LFE (27-family pool, V2)
    T = pd.read_csv(f'{OUT}/t1_{"decline" if kind == "D" else "m7step"}_per_station.csv')
    blocks['t1'] = dict(sta=T.sta.values, d=T.depth_m.values / 1000.0, y=T.value.values, s=T.placebo_sd.values)
    # --- LFE sub-band / lapse shape vectors per station (6 families, V2)
    R = pd.read_csv(f'{OUT}/lfe_fits_placebo.csv.gz')
    val = 'D' if kind == 'D' else 'm7'
    pk = 'pD' if kind == 'D' else 'pM'
    pc = [c for c in R if c.startswith(pk) and c[len(pk):].isdigit()]
    bootc = 'D_boot_se' if kind == 'D' else 'm7_boot_se'
    lst = []
    # placebo/bootstrap inflation per config from channels that have both
    kap = {}
    for cfg, g in R.groupby('cfg'):
        sdp = g[pc].std(axis=1, ddof=1)
        ok = (g[pc].notna().sum(axis=1) >= 10) & (g[bootc] > 0)
        kap[cfg] = float(np.nanmedian(sdp[ok] / g[bootc][ok])) if ok.sum() >= 5 else 1.5
    for sta, g in R.groupby('sta'):
        g = g[np.isfinite(g[val])]
        cfgs_late = [c for c in SHAPE_CFGS]
        cfgs_early = ['b24_early', 'b48_early']
        chosen = None
        for cset in (cfgs_late, cfgs_early):
            chans = [ch for ch, gc in g.groupby('chan') if set(cset) <= set(gc.cfg)]
            if chans:
                chosen = (cset, chans)
                break
        if chosen is None:
            continue
        cset, chans = chosen
        gg = g[g.chan.isin(chans) & g.cfg.isin(cset)]
        y = np.array([gg[gg.cfg == c][val].mean() for c in cset])
        rec = np.array([gg[gg.cfg == c]['rec_D' if kind == 'D' else 'rec_m7'].mean() for c in cset])
        Pm = np.array([gg[gg.cfg == c][pc].mean(axis=0).values for c in cset]).T          # (shift, cfg)
        npl = np.isfinite(Pm).all(axis=1).sum()
        if npl >= 10:
            sd = np.nanstd(Pm, axis=0, ddof=1)
            C = shrink_cov(Pm, sd)
            src = 'placebo'
        else:
            sd = np.array([np.sqrt(np.mean(gg[gg.cfg == c][bootc] ** 2) / max(len(chans), 1)) * kap.get(c, 1.5) for c in cset])
            C = np.diag(sd ** 2)
            src = 'bootstrap_x_kappa'
        lst.append(dict(sta=sta, d=BORE_D.get(sta, 0.0), cfgs=cset, y=y, C=C, sd=sd, rec=rec, n_chan=len(chans), src=src))
    blocks['lfe'] = lst
    blocks['kappa'] = kap
    # --- noise
    N = pd.read_csv(f'{OUT}/noise_fits_placebo.csv.gz')
    N = N[N.combo == 'ALL']
    nv = 'D' if kind == 'D' else 'step_m7'
    npk = 'pD' if kind == 'D' else 'pM'
    npc = [c for c in N if c.startswith(npk) and c[len(npk):].isdigit()]
    nl = []
    for sta, g in N.groupby('sta'):
        rows = [g[(g.band == b) & (g.win == w)] for b, w in NOISE_OBS]
        if any(len(r) == 0 for r in rows) or any(not np.isfinite(r[nv].values[0]) for r in rows):
            continue
        y = np.array([r[nv].values[0] for r in rows])
        Pm = np.array([r[npc].values[0] for r in rows]).T
        sd = np.nanstd(Pm, axis=0, ddof=1)
        rec = np.array([r['rec_D' if kind == 'D' else 'rec_m7'].values[0] for r in rows])
        nl.append(dict(sta=NOISE_STA_FULL[sta], d=BORE_D.get(NOISE_STA_FULL[sta], 0.0), obs=NOISE_OBS, y=y / np.where(np.abs(rec) > 0.3, rec, 1.0),
                       y_raw=y, rec=rec, C=shrink_cov(Pm, sd) / np.outer(np.where(np.abs(rec) > 0.3, rec, 1.0), np.where(np.abs(rec) > 0.3, rec, 1.0)), sd=sd))
    blocks['noise'] = nl
    return blocks


class Kern:
    def __init__(self, vel='ref', lfe_l='20', lstar=5.0, sw='RL', bh_shift=False):
        Z = np.load(f'{OUT}/kernels.npz')
        self.zg = Z['ZG']
        self.zc = 0.5 * (self.zg[1:] + self.zg[:-1])
        self.dz = np.diff(self.zg)
        self.f = Z['FREQS']
        self.depths = Z['DEPTHS']
        R, L = Z[f'{vel}|R'], Z[f'{vel}|L']
        aR, aL = Z[f'{vel}|ampR'], Z[f'{vel}|ampL']
        if sw == 'RL':
            self.Ksw, self.amp = 0.5 * (R + L), 0.5 * (aR + aL)
        elif sw == 'R':
            self.Ksw, self.amp = R, aR
        else:
            self.Ksw, self.amp = L, aL
        self.Z, self.vel, self.lfe_l, self.lstar, self.bh = Z, vel, lfe_l, lstar, bh_shift

    def fidx(self, band):
        f1, f2 = BAND_F[band]
        return np.where((self.f >= f1 * 0.999) & (self.f <= f2 * 1.001))[0]

    def r_at(self, d):
        j = int(np.argmin(np.abs(self.depths - d)))
        return self.amp[:, j]

    def lfe_body(self, win, d):
        key = f'{self.vel}|lfe_{win}_{self.lfe_l}'
        if d > 0 and self.bh:
            key += f'_bh{self.depths[int(np.argmin(np.abs(self.depths - d)))]:.4f}'
        return self.Z[key]

    def noise_body(self, win, d):
        return self.Z[f'noise_{win}_{self.lstar:g}_z{(0.2176 if d > 0 else 0.0):.4f}']


def obs_components(K, typ, band, win, d, M):
    """Gsw (nf, nshape), r (nf,), Gbw (nshape,) for one observable."""
    fi = K.fidx(band)
    Gsw = K.Ksw[fi] @ M.T
    r = K.r_at(d)[fi] if d > 0 else np.ones(len(fi))
    Kb = K.lfe_body(win, d) if typ == 'lfe' else K.noise_body(win, d)
    return Gsw, r, Kb @ M.T


def G_of_alpha(comp, alpha):
    """comp from obs_components; alpha (na,) -> G (na, nshape)"""
    Gsw, r, Gbw = comp
    a = alpha[:, None]
    ae = a * r[None, :] / (a * r[None, :] + 1 - a + 1e-12)                  # (na, nf)
    return (ae[:, :, None] * Gsw[None] + (1 - ae)[:, :, None] * Gbw[None, None]).mean(axis=1)


def loglik_block(y, C, G, sigA):
    """y (n,), C (n,n), G (..., n, nshape) -> log marginal likelihood (..., nshape), amplitude ~ N(0, sigA^2)."""
    Ci = np.linalg.inv(C)
    yCy = y @ Ci @ y
    CiY = Ci @ y
    gCy = np.einsum('...ns,n->...s', G, CiY)
    gCg = np.einsum('...ns,nm,...ms->...s', G, Ci, G)
    s2 = sigA ** 2
    return -0.5 * (yCy - s2 * gCy ** 2 / (1 + s2 * gCg)) - 0.5 * np.log1p(s2 * gCg) - 0.5 * np.linalg.slogdet(C)[1]


def amp_post(y, C, g, sigA):
    Ci = np.linalg.inv(C)
    prec = g @ Ci @ g + 1 / sigA ** 2
    return float((g @ Ci @ y) / prec), float(1 / np.sqrt(prec))


def shapes(param):
    zc = None
    Z = np.load(f'{OUT}/kernels.npz')
    zg = Z['ZG']
    lo, hi = zg[:-1], zg[1:]
    dz = hi - lo
    out, P = [], []
    if param == 'uniform':
        for H in np.geomspace(0.003, 30, 70):
            out.append(np.clip((np.minimum(hi, H) - lo) / dz, 0, 1)); P.append(dict(H=H))   # noqa: E702
    elif param == 'exponential':
        for D in np.geomspace(0.003, 30, 70):
            out.append(D * (np.exp(-lo / D) - np.exp(-hi / D)) / dz); P.append(dict(D=D))   # noqa: E702
    elif param == 'twolayer':
        for H1 in np.geomspace(0.005, 3, 16):
            for H2 in np.geomspace(0.02, 30, 16):
                if H2 <= H1 * 1.5:
                    continue
                for b in [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
                    m = np.clip((np.minimum(hi, H1) - lo) / dz, 0, 1) + b * np.clip((np.minimum(hi, H2) - np.maximum(lo, H1)) / dz, 0, 1)
                    out.append(m); P.append(dict(H1=H1, H2=H2, beta=b))   # noqa: E702
    M = np.array(out)
    # depth above which 50 / 90 % of the integrated change sits
    cum = np.cumsum(M * dz[None], axis=1)
    cum /= cum[:, -1:]
    z50 = np.array([np.interp(0.5, c, hi) for c in cum])
    z90 = np.array([np.interp(0.9, c, hi) for c in cum])
    # prior: log-uniform in each depth parameter (uniform over the log grids), uniform in beta
    return M, pd.DataFrame(P).assign(z50=z50, z90=z90)


def t1_loglik(b, Gt, sigA, tau_grid):
    """cross-station block: Gt (nsta, ns); multiplicative site scatter tau (plug-in amplitude), marginal over tau."""
    y, s2 = b['y'], b['s'] ** 2
    gCy = (Gt * (y / s2)[:, None]).sum(0)
    gCg = (Gt ** 2 / s2[:, None]).sum(0)
    A0 = gCy / (gCg + 1 / sigA ** 2)
    lt = []
    for tau in tau_grid:
        v = s2[:, None] + (tau * A0[None, :] * Gt) ** 2
        yCy = (y[:, None] ** 2 / v).sum(0)
        gy = (Gt * y[:, None] / v).sum(0)
        gg = (Gt ** 2 / v).sum(0)
        S2 = sigA ** 2
        lt.append(-0.5 * (yCy - S2 * gy ** 2 / (1 + S2 * gg)) - 0.5 * np.log1p(S2 * gg) - 0.5 * np.log(v).sum(0))
    lt = np.array(lt)
    mx = lt.max(0)
    return np.log(np.mean(np.exp(lt - mx), axis=0)) + mx


def invert_one(blocks, K, M, use=('t1', 'lfe', 'noise'), sigA=5.0, tie=False, order_noise=False, tau_grid=(0, 0.25, 0.5, 1.0)):
    """log posterior over shapes (marginal over alphas, amplitudes, site scatter)."""
    ns = len(M)
    na = len(ALPHA)
    detail = {}
    lfe_terms = []
    if 't1' in use:
        b = blocks['t1']
        comps = [obs_components(K, 'lfe', 'b28', 'full', d, M) for d in b['d']]
        Gfull = np.array([G_of_alpha(c, ALPHA) for c in comps])          # (nsta, na, ns): alpha_full on ALPHA grid
        ll = np.zeros((na, na, ns))
        for i in range(na):
            for j in range(na):
                af = 0.5 * (ALPHA[i] + ALPHA[j])
                k0 = int(np.floor(af * (na - 1) + 1e-9))
                k1 = min(k0 + 1, na - 1)
                fr = af * (na - 1) - k0
                Gt = (1 - fr) * Gfull[:, k0] + fr * Gfull[:, k1]
                ll[i, j] = t1_loglik(b, Gt, sigA, tau_grid)
        lfe_terms.append(ll)
        detail['t1'] = ll
    if 'lfe' in use:
        for blk in blocks['lfe']:
            wins = [c.split('_')[1] for c in blk['cfgs']]
            GA = [G_of_alpha(obs_components(K, 'lfe', c[:3], w, blk['d'], M), ALPHA) for c, w in zip(blk['cfgs'], wins)]
            y, C = blk['y'] / np.where(np.abs(blk['rec']) > 0.3, blk['rec'], 1.0), blk['C']
            ll = np.zeros((na, na, ns))
            for i in range(na):
                for j in range(na):
                    G = np.array([GA[o][i if w == 'early' else j] for o, w in enumerate(wins)])
                    ll[i, j] = loglik_block(y, C, G, sigA)
            lfe_terms.append(ll)
            detail['lfe_' + blk['sta']] = ll
    noise_terms = []
    if 'noise' in use:
        for blk in blocks['noise']:
            GA = [G_of_alpha(obs_components(K, 'noise', b_, w_, blk['d'], M), ALPHA) for b_, w_ in blk['obs']]
            ll = np.zeros((na, na, ns))
            for i in range(na):
                for j in range(na):
                    G = np.array([GA[o][i if w_ == 'early' else j] for o, (b_, w_) in enumerate(blk['obs'])])
                    ll[i, j] = loglik_block(blk['y'], blk['C'], G, sigA)
            noise_terms.append(ll)
            detail['noise_' + blk['sta']] = ll

    def marg(terms, ordered=False):
        if not terms:
            return np.zeros(ns)
        T = np.sum(terms, axis=0)
        mask = np.ones((na, na), bool)
        if ordered:
            mask = ALPHA[None, :] <= ALPHA[:, None] + 1e-9       # alpha_late <= alpha_early
        T = np.where(mask[..., None], T, -np.inf)
        mx = T.max(axis=(0, 1))
        return np.log(np.exp(T - mx).sum(axis=(0, 1)) / mask.sum()) + mx
    if tie and 'lfe' in use and 'noise' in use:
        LL = (marg([detail['t1']]) if 't1' in detail else 0.0) + tied_loglik(blocks, K, M, sigA, order_noise)
    else:
        LL = marg(lfe_terms) + marg(noise_terms, order_noise)
    return LL, detail


def tied_loglik(blocks, K, M, sigA, order_noise):
    """LFE shape + noise blocks with ONE amplitude per station (stations in both), alphas (aLE, aLL, aNE, aNL)."""
    ns = len(M)
    na = len(ALPHA)
    lf = {b['sta']: b for b in blocks['lfe']}
    nz = {b['sta']: b for b in blocks['noise']}
    tot = None
    sub = np.arange(0, na, 2)                     # coarser alpha grid (6 values) to keep 6^4 combos
    A = ALPHA[sub]
    for sta in sorted(set(lf) | set(nz)):
        parts = []
        for typ, blk in (('lfe', lf.get(sta)), ('noise', nz.get(sta))):
            if blk is None:
                continue
            if typ == 'lfe':
                comps = [obs_components(K, 'lfe', c[:3], c.split('_')[1], blk['d'], M) for c in blk['cfgs']]
                wins = [c.split('_')[1] for c in blk['cfgs']]
                y = blk['y'] / np.where(np.abs(blk['rec']) > 0.3, blk['rec'], 1.0)
            else:
                comps = [obs_components(K, 'noise', b_, w_, blk['d'], M) for b_, w_ in blk['obs']]
                wins = [w_ for b_, w_ in blk['obs']]
                y = blk['y']
            parts.append((typ, comps, wins, y, blk['C']))
        ll = np.zeros((len(A),) * 4 + (ns,))
        for i1, a1 in enumerate(A):
            for i2, a2 in enumerate(A):
                for i3, a3 in enumerate(A):
                    for i4, a4 in enumerate(A):
                        Gs, ys, Cs = [], [], []
                        for typ, comps, wins, y, C in parts:
                            al = {'lfe': (a1, a2), 'noise': (a3, a4)}[typ]
                            Gs.append(np.array([G_of_alpha(c, np.array([al[0] if w == 'early' else al[1]]))[0] for c, w in zip(comps, wins)]))
                            ys.append(y)
                            Cs.append(C)
                        from scipy.linalg import block_diag
                        ll[i1, i2, i3, i4] = loglik_block(np.concatenate(ys), block_diag(*Cs), np.concatenate(Gs, axis=0), sigA)
        tot = ll if tot is None else tot + ll
    mask = np.ones((len(A),) * 4, bool)
    if order_noise:
        mask &= (A[None, None, None, :] <= A[None, None, :, None] + 1e-9)
    T = np.where(mask[..., None], tot, -np.inf)
    mx = T.max(axis=(0, 1, 2, 3))
    return np.log(np.exp(T - mx).sum(axis=(0, 1, 2, 3)) / mask.sum()) + mx


def post_stats(LL, P):
    w = np.exp(LL - LL.max())
    w /= w.sum()

    def q(x, qq):
        o = np.argsort(x)
        c = np.cumsum(w[o])
        return float(np.interp(qq, c, x[o]))
    out = {}
    for k in [c for c in P.columns]:
        x = P[k].values.astype(float)
        out[k] = dict(p05=q(x, 0.05), p16=q(x, 0.16), p50=q(x, 0.5), p84=q(x, 0.84), p95=q(x, 0.95), map=float(x[np.argmax(w)]))
    # probabilities of the three hypotheses on z90
    z90 = P.z90.values
    out['P_z90_lt_0.2km'] = float(w[z90 < 0.2].sum())
    out['P_z90_0.2_1km'] = float(w[(z90 >= 0.2) & (z90 < 1.0)].sum())
    out['P_z90_ge_1km'] = float(w[z90 >= 1.0].sum())
    z50 = P.z50.values
    out['P_z50_lt_0.2km'] = float(w[z50 < 0.2].sum())
    out['P_z50_0.2_1km'] = float(w[(z50 >= 0.2) & (z50 < 1.0)].sum())
    out['P_z50_ge_1km'] = float(w[z50 >= 1.0].sum())
    out['logZ'] = float(np.log(np.mean(np.exp(LL - LL.max()))) + LL.max())
    return out, w


VARIANTS_INV = {
    'reference': dict(),
    'vel_soft': dict(vel='soft'), 'vel_stiff': dict(vel='stiff'), 'vel_thin': dict(vel='thin'), 'vel_noLVZ': dict(vel='none'),
    'lfe_l50': dict(lfe_l='50'), 'lfe_l100': dict(lfe_l='100'), 'lfe_shallowscat': dict(lfe_l='10/5km/100'),
    'noise_lstar1': dict(lstar=1.0), 'noise_lstar20': dict(lstar=20.0),
    'rayleigh_only': dict(sw='R'), 'love_only': dict(sw='L'), 'bh_shift': dict(bh_shift=True),
    'sigA20': dict(sigA=20.0), 'sigA2': dict(sigA=2.0), 'noise_alpha_ordered': dict(order_noise=True),
    'tied_amplitudes': dict(tie=True),
    'only_t1': dict(use=('t1',)), 'only_lfe_shape': dict(use=('lfe',)), 'only_noise': dict(use=('noise',)),
    'lfe_all': dict(use=('t1', 'lfe')), 'no_t1': dict(use=('lfe', 'noise')),
}


def inv_job(args):
    kind, param, vname, opts = args
    blocks = load_obs(kind)
    kk = {k: opts[k] for k in ['vel', 'lfe_l', 'lstar', 'sw', 'bh_shift'] if k in opts}
    K = Kern(**kk)
    M, P = shapes(param)
    io = {k: opts[k] for k in ['use', 'sigA', 'tie', 'order_noise'] if k in opts}
    LL, detail = invert_one(blocks, K, M, **io)
    st, w = post_stats(LL, P)
    return (kind, param, vname), dict(stats=st, w=w, LL=LL, P=P)


def stage_invert(args):
    jobs = []
    for kind in ['D', 'm7']:
        for param in ['uniform', 'exponential', 'twolayer']:
            for vname, opts in VARIANTS_INV.items():
                if param == 'twolayer' and vname not in ('reference', 'tied_amplitudes', 'only_noise', 'lfe_all', 'vel_soft',
                                                          'vel_thin', 'lfe_shallowscat', 'sigA20'):
                    continue
                if opts.get('tie') and param == 'twolayer':
                    continue
                jobs.append((kind, param, vname, opts))
    print(len(jobs), 'inversion jobs', flush=True)
    res = {}
    with Pool(NPROC) as pool:
        for k, r in pool.imap_unordered(inv_job, jobs):
            res[k] = r
            print(k, {kk: (round(v['p50'], 3), round(v['p05'], 3), round(v['p95'], 3)) for kk, v in r['stats'].items()
                      if kk in ('z50', 'z90')}, 'P(z50<0.2, 0.2-1, >1 km)', round(r['stats']['P_z50_lt_0.2km'], 2),
                  round(r['stats']['P_z50_0.2_1km'], 2), round(r['stats']['P_z50_ge_1km'], 2), flush=True)
    pickle.dump(res, open(f'{OUT}/inversion_results.pkl', 'wb'))   # own output, re-read only by this script
    rows = []
    for (kind, param, vname), r in res.items():
        st = r['stats']
        row = dict(kind=kind, param=param, variant=vname, logZ=st['logZ'])
        for k in ['z50', 'z90', 'H', 'D', 'H1', 'H2', 'beta']:
            if k in st:
                for qq in ['p05', 'p16', 'p50', 'p84', 'p95', 'map']:
                    row[f'{k}_{qq}'] = st[k][qq]
        for k in st:
            if k.startswith('P_'):
                row[k] = st[k]
        rows.append(row)
    T = pd.DataFrame(rows).sort_values(['kind', 'param', 'variant'])
    T.to_csv(f'{OUT}/inversion_summary.csv', index=False, float_format='%.4g')
    print(T[['kind', 'param', 'variant', 'z50_p05', 'z50_p50', 'z50_p95', 'z90_p05', 'z90_p50', 'z90_p95', 'P_z50_lt_0.2km', 'P_z50_ge_1km', 'logZ']].round(3).to_string())


# ================================================================================================ station tables
def stage_tables(args):
    """station x config tables for LFE (6 families) and noise, with placebo z; group summaries; ratios."""
    R = pd.read_csv(f'{OUT}/lfe_fits_placebo.csv.gz')
    out = {}
    for kind, val, pk in [('D', 'D', 'pD'), ('m7', 'm7', 'pM')]:
        pc = [c for c in R if c.startswith(pk) and c[len(pk):].isdigit()]
        rows = []
        for (sta, cfg), g in R.groupby(['sta', 'cfg']):
            g = g[np.isfinite(g[val])]
            if len(g) == 0:
                continue
            v = g[val].mean()
            pl = g[pc].mean(axis=0).values
            ok = np.isfinite(pl)
            sd = np.std(pl[ok], ddof=1) if ok.sum() >= 10 else np.nan
            rows.append(dict(kind=kind, sta=sta, depth_m=BORE_D.get(sta, 0.0) * 1000, cfg=cfg, n_chan=len(g), value=v,
                             placebo_sd=sd, z=v / sd if sd > 0 else np.nan, rec_paired=g['rec_' + val].mean(),
                             rec_unpaired=g['rec_' + val + '_unpaired'].mean(), boot_se=g[val + '_boot_se'].mean()))
        T = pd.DataFrame(rows)
        T.to_csv(f'{OUT}/lfe_station_config_{kind}.csv', index=False, float_format='%.4g')
        # group (surface / borehole) paired contrasts with the placebo: same channels in both configs
        grp = {}
        for gname, gsel in [('surface', lambda x: ~x.str.startswith('PB.')), ('borehole', lambda x: x.str.startswith('PB.'))]:
            for c1, c2 in [('b24_full', 'b48_full'), ('b28_early', 'b28_late'), ('b24_early', 'b48_early'), ('b24_late', 'b48_late'),
                           ('b24_early', 'b24_late'), ('b48_early', 'b48_late')]:
                a = R[(R.cfg == c1) & gsel(R.sta)].set_index('chan')
                b = R[(R.cfg == c2) & gsel(R.sta)].set_index('chan')
                ch = a.index.intersection(b.index)
                ch = [c for c in ch if np.isfinite(a.loc[c, val]) and np.isfinite(b.loc[c, val])]
                if len(ch) < 2:
                    continue
                # station-equal weights: mean over channels per station, then over stations
                sta = pd.Series([sta_of(c) for c in ch], index=ch)
                def gm(X, col):
                    return X.loc[ch, col].groupby(sta).mean().mean()
                v1, v2 = gm(a, val), gm(b, val)
                p1 = np.array([gm(a, c) for c in pc])
                p2 = np.array([gm(b, c) for c in pc])
                ok = np.isfinite(p1) & np.isfinite(p2)
                sd1, sd2 = np.std(p1[ok], ddof=1), np.std(p2[ok], ddof=1)
                sdd = np.std((p1 - p2)[ok], ddof=1)
                # station bootstrap of the ratio
                st_u = sta.unique()
                rng = np.random.default_rng(3)
                rb = []
                per = {s_: (a.loc[sta.index[sta == s_], val].mean(), b.loc[sta.index[sta == s_], val].mean()) for s_ in st_u}
                for _ in range(2000):
                    pick = rng.choice(st_u, len(st_u))
                    x1 = np.mean([per[s_][0] for s_ in pick])
                    x2 = np.mean([per[s_][1] for s_ in pick])
                    rb.append(x2 / x1 if x1 != 0 else np.nan)
                grp[f'{gname}:{c1}_vs_{c2}'] = dict(n_sta=len(st_u), n_chan=len(ch), v1=v1, z1=v1 / sd1, v2=v2, z2=v2 / sd2,
                                                     diff=v1 - v2, diff_placebo_sd=sdd, diff_z=(v1 - v2) / sdd,
                                                     ratio_2_over_1=v2 / v1 if v1 != 0 else np.nan,
                                                     ratio_boot_p16_p84=np.nanpercentile(rb, [16, 84]).tolist(),
                                                     stations=';'.join(st_u))
        out[kind] = grp
        print(kind, json.dumps(grp, indent=0, default=float)[:3000], flush=True)
    # noise table summary (ALL combo)
    N = pd.read_csv(f'{OUT}/noise_band_lapse_table.csv')
    out['noise_ALL'] = N[N.combo == 'ALL'][['sta', 'band', 'win', 'D', 'D_pl_sd', 'D_z', 'rec_D', 'step_m7', 'm7_pl_sd', 'm7_z', 'rec_m7']].round(4).to_dict('records')
    out['noise_ZZ'] = N[N.combo == 'ZZ'][['sta', 'band', 'win', 'D', 'D_pl_sd', 'D_z', 'step_m7', 'm7_pl_sd', 'm7_z']].round(4).to_dict('records')
    # noise paired band / lapse contrasts with placebo (common fake dates)
    F = pd.read_csv(f'{OUT}/noise_fits_placebo.csv.gz')
    F = F[F.combo == 'ALL']
    nc = {}
    for kind, val, pk in [('D', 'D', 'pD'), ('m7', 'step_m7', 'pM')]:
        pc = [c for c in F if c.startswith(pk) and c[len(pk):].isdigit()]
        for sta, g in F.groupby('sta'):
            gi = g.set_index(['band', 'win'])
            for (b1, w1), (b2, w2) in [(('1-2Hz', 'full'), ('4-8Hz', 'full')), (('1-2Hz', 'full'), ('2-4Hz', 'full')),
                                       (('2-4Hz', 'full'), ('4-8Hz', 'full')), (('1-4Hz', 'full'), ('2-8Hz', 'full')),
                                       (('2-8Hz', 'early'), ('2-8Hz', 'late')), (('1-4Hz', 'early'), ('1-4Hz', 'late'))]:
                if (b1, w1) not in gi.index or (b2, w2) not in gi.index:
                    continue
                v1, v2 = gi.loc[(b1, w1), val], gi.loc[(b2, w2), val]
                d = (gi.loc[(b1, w1), pc].values - gi.loc[(b2, w2), pc].values).astype(float)
                sdd = np.nanstd(d, ddof=1)
                nc[f'{kind}:{sta}:{b1}_{w1}_vs_{b2}_{w2}'] = dict(v1=v1, v2=v2, diff=v1 - v2, diff_placebo_sd=sdd,
                                                                    diff_z=(v1 - v2) / sdd, ratio=v2 / v1 if v1 else np.nan)
    out['noise_contrasts'] = nc
    summary_update('tests2_3_tables', out)


# ================================================================================================ figures
def stage_figs(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    S = json.load(open(f'{OUT}/summary.json'))
    # ---- 1. sensor depth
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, name, lab in [(axs[0], 'decline', 'decline 2021->2025 (%)'), (axs[1], 'm7step', 'M7 step (%)')]:
        T = pd.read_csv(f'{OUT}/t1_{name}_per_station.csv')
        ax.errorbar(T.depth_m, T.value, yerr=T.placebo_sd, fmt='o', color='C0', capsize=3)
        for _, r in T.iterrows():
            ax.annotate(r.sta.split('.')[1], (r.depth_m, r.value), xytext=(4, 3), textcoords='offset points', fontsize=7)
        r = S['test1_sensor_depth'][name]['linear_depth']
        x = np.linspace(0, 620, 10)
        ax.plot(x, r['intercept'] + r['slope'] * x / 1000, 'k--', lw=1, label=f"slope {r['slope']:.3f}+/-{r['slope_se']:.3f} %/km (site scatter {r['sig_int']:.3f})")
        ax.axhline(0, color='0.7', lw=0.8)
        ax.set_xlabel('sensor depth (m; surface = 0)')
        ax.set_ylabel(lab)
        ax.legend(fontsize=7, loc='lower right')
    axs[0].set_title('LFE coda, V2 cleaned, placebo SD')
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_depth2_sensor_depth.png', dpi=140)
    plt.close(fig)
    # ---- 2. frequency / lapse
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    N = pd.read_csv(f'{OUT}/noise_band_lapse_table.csv')
    N = N[N.combo == 'ALL']
    bands = ['1-2Hz', '1-4Hz', '2-4Hz', '2-8Hz', '4-8Hz']
    fc = {'1-2Hz': 1.41, '1-4Hz': 2.0, '2-4Hz': 2.83, '2-8Hz': 4.0, '4-8Hz': 5.66}
    for j, (kind, col, sdc) in enumerate([('D', 'D', 'D_pl_sd'), ('m7', 'step_m7', 'm7_pl_sd')]):
        ax = axs[0, j]
        for k, sta in enumerate(['KMPB', 'BRIC', 'B045', 'HAYF']):
            for wn, mk in [('early', 'o'), ('late', 's')]:
                g = N[(N.sta == sta) & (N.win == wn) & N.band.isin(['1-2Hz', '2-4Hz', '4-8Hz'])].set_index('band').reindex(['1-2Hz', '2-4Hz', '4-8Hz'])
                x = np.array([fc[b] for b in g.index]) * (1.03 if wn == 'late' else 0.97)
                ax.errorbar(x, g[col], yerr=g[sdc], fmt=mk + ('-' if wn == 'early' else ':'), color=f'C{k}', capsize=2,
                            label=f'{sta} {wn}', ms=4)
        ax.set_xscale('log')
        ax.set_xlabel('band centre (Hz)')
        ax.set_ylabel(('decline 2021->2025' if kind == 'D' else 'M7 step') + ' (%)')
        ax.set_title('ambient noise (all pairs), placebo SD')
        ax.axhline(0, color='0.7', lw=0.8)
        ax.legend(fontsize=6, ncol=2)
        ax = axs[1, j]
        T = pd.read_csv(f'{OUT}/lfe_station_config_{kind}.csv')
        cf = ['b24_early', 'b24_late', 'b48_early', 'b48_late']
        for k, (gname, sel) in enumerate([('surface', ~T.sta.str.startswith('PB.')), ('borehole', T.sta.str.startswith('PB.'))]):
            for i, sta in enumerate(sorted(T[sel].sta.unique())):
                g = T[(T.sta == sta)].set_index('cfg').reindex(cf)
                if g.value.notna().sum() < 2:
                    continue
                xx = np.arange(4) + (i - 4) * 0.04
                ax.errorbar(xx, g.value, yerr=g.placebo_sd, fmt='o-' if gname == 'surface' else 's--', ms=3, lw=0.8, capsize=1.5,
                            label=sta.split('.')[1], color=plt.cm.tab20(i % 20))
        ax.set_xticks(range(4))
        ax.set_xticklabels(['2-4 early', '2-4 late', '4-8 early', '4-8 late'])
        ax.axhline(0, color='0.7', lw=0.8)
        ax.set_ylabel(('decline' if kind == 'D' else 'M7 step') + ' (%)')
        ax.set_title('LFE coda (6 families, V2), solid = surface, dashed = borehole')
        ax.legend(fontsize=5, ncol=3)
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_depth2_freq_lapse.png', dpi=140)
    plt.close(fig)
    # ---- 3. kernels
    Z = np.load(f'{OUT}/kernels.npz')
    zg = Z['ZG']
    zc = 0.5 * (zg[1:] + zg[:-1])
    dz = np.diff(zg)
    fig, axs = plt.subplots(1, 3, figsize=(13, 5))
    ax = axs[0]
    for v in VEL_VARIANTS:
        ax.plot(Z[f'{v}|vs'], zc, label=v)
    ax.set_ylim(3, 0.001)
    ax.set_yscale('log')
    ax.set_xlabel('Vs (km/s)')
    ax.set_ylabel('depth (km)')
    ax.legend(fontsize=7)
    ax.set_title('1-D models (Furlong + shallow LVZ)')
    ax = axs[1]
    for i in [0, 3, 6, 9, 12]:
        f = Z['FREQS'][i]
        ax.plot(Z['ref|R'][i] / dz, zc, label=f'R {f:.1f} Hz', color=plt.cm.viridis(i / 12))
        ax.plot(Z['ref|L'][i] / dz, zc, ':', color=plt.cm.viridis(i / 12))
    ax.set_ylim(3, 0.001)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(1e-3, 100)
    ax.set_xlabel('coda kernel (U/c) dlnc/dlnVs per km')
    ax.set_title('fundamental Rayleigh (solid) / Love (dotted), ref')
    ax.legend(fontsize=7)
    ax = axs[2]
    for k, lab in [('ref|lfe_early_20', 'LFE body early l=20'), ('ref|lfe_late_20', 'LFE body late l=20'),
                   ('ref|lfe_full_10/5km/100', 'LFE body, strong shallow scat.'), ('noise_early_5_z0.0000', 'noise body early l*=5'),
                   ('noise_late_5_z0.0000', 'noise body late l*=5'), ('noise_full_1_z0.0000', 'noise body l*=1')]:
        ax.plot(np.cumsum(Z[k]), zg[1:], label=lab)
    ax.set_yscale('log')
    ax.set_ylim(40, 0.001)
    ax.set_xlabel('cumulative sensitivity above depth')
    ax.legend(fontsize=7)
    ax.set_title('body-wave kernels (cumulative)')
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_depth2_kernels.png', dpi=140)
    plt.close(fig)
    # ---- 4. posteriors
    res = pickle.load(open(f'{OUT}/inversion_results.pkl', 'rb'))     # own output of stage invert (trusted)
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    for i, kind in enumerate(['D', 'm7']):
        for j, param in enumerate(['uniform', 'exponential', 'twolayer']):
            ax = axs[i, j]
            for vname, col in [('reference', 'k'), ('only_noise', 'C0'), ('lfe_all', 'C1'), ('only_t1', 'C2'), ('tied_amplitudes', 'C3')]:
                r = res.get((kind, param, vname))
                if r is None:
                    continue
                w, z50, z90 = r['w'], r['P'].z50.values, r['P'].z90.values
                bins = np.geomspace(0.001, 40, 50)
                h, _ = np.histogram(z50, bins=bins, weights=w)
                ax.step(bins[:-1], h, where='post', color=col, label=f'{vname} z50')
                h, _ = np.histogram(z90, bins=bins, weights=w)
                ax.step(bins[:-1], h, where='post', color=col, ls=':', lw=0.8)
            ax.set_xscale('log')
            ax.set_xlabel('depth (km): z50 solid, z90 dotted')
            ax.set_title(f'{"decline" if kind == "D" else "M7 step"}: {param}')
            ax.axvspan(0.1, 0.2, color='0.9')
            ax.axvspan(0.3, 1.0, color='0.95')
            if i == 0 and j == 0:
                ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_depth2_posterior.png', dpi=140)
    plt.close(fig)
    # ---- 5. variant sensitivity
    T = pd.read_csv(f'{OUT}/inversion_summary.csv')
    fig, axs = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    for ax, kind in zip(axs, ['D', 'm7']):
        t = T[(T.kind == kind)].reset_index(drop=True)
        lab = [f'{p[:4]}:{v}' for p, v in zip(t.param, t.variant)]
        y = np.arange(len(t))
        ax.errorbar(t.z50_p50, y, xerr=[t.z50_p50 - t.z50_p05, t.z50_p95 - t.z50_p50], fmt='o', ms=3, label='z50 (5-95 %)')
        ax.errorbar(t.z90_p50, y + 0.3, xerr=[t.z90_p50 - t.z90_p05, t.z90_p95 - t.z90_p50], fmt='s', ms=3, label='z90')
        ax.set_yticks(y)
        ax.set_yticklabels(lab, fontsize=6)
        ax.set_xscale('log')
        ax.set_xlabel('depth (km)')
        ax.set_title('decline' if kind == 'D' else 'M7 step')
        ax.axvline(0.2, color='0.6', ls='--')
        ax.axvline(1.0, color='0.6', ls='--')
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_depth2_variants.png', dpi=140)
    plt.close(fig)

# ================================================================================================ main
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage')
    ap.add_argument('--nboot', type=int, default=60)
    a = ap.parse_args()
    {'depthreg': stage_depthreg, 'noise': stage_noise, 'noisefit': stage_noisefit, 'lfefit': stage_lfefit, 'kernels': stage_kernels, 'invert': stage_invert, 'tables': stage_tables, 'figs': stage_figs}[a.stage](a)
