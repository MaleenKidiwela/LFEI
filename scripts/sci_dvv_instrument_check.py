#!/usr/bin/env python3
"""Instrument-epoch robustness check of the main LFE-coda and ambient-noise dv/v claims.

The time-lapse shear-wave-splitting work (scripts/sci_sws.py, data/sci/sws/) found horizontal-component
instrument problems (B933 2017-01..2021-11 and from 2024-08; B935 from 2024-05; BRIC degraded 2022-12 ..
2024-05-21 reinstall; RVIT Ferndale .. 2024-06-20 reinstall; BJES 2023 .. 2024-05-22 reinstall; KMR before
2009-09 and from the 2025-09 swap; KMPB before 2010-11, NCEDC gap Feb-Oct 2023 + 2023-10-17 maintenance;
B045 EH1 2009-10 polarity).  Here every dv/v claim is recomputed with those windows removed and/or with free
offsets across the reinstall boundaries.  Existing outputs are only read; nothing is re-downloaded.

Variants (applied to ALL components of a flagged station, B045 only EH1):
  V0  original (no masks; reproduces data/sci/decline/monthly_bins.csv and data/sci/dvv_placebo)
  V1  instrument windows excluded (pair rows with either stack inside a window dropped); records still
      connected across a window by pairs/reference rows that straddle it
  V2  V1 + FREE OFFSETS: connectivity broken at the end of every interior window (B933 2021-11-01, BRIC
      2024-05-22, RVIT 2024-06-21, BJES 2024-05-23) and at the KMPB 2023-10-17 maintenance epoch: no pair
      crosses a boundary, separate reference offsets per (family, segment), so nothing is compared across it
  V3  V2 + free offsets at every other sensor/orientation/gain metadata epoch 2019-2026 (RBOW, HAYF, KMPB 2019 ...)

Stages:
  monthly   unregularised monthly bins per channel (as scripts/sci_decline_monthly.py) for V0..V3
  param     parametric per-channel placebo fits (as scripts/sci_dvv_placebo.py channel fits) for V0..V3
  noise     ambient-noise dv/v (data/sci/noise_dvv/ccf_*.npz): BRIC offset at the reinstall, vertical-only
            autocorrelation (ZZ), post-reinstall reference; KMPB 2023-10 offset; B045, HAYF
  analyze   decline (yearly, model-based with free offsets, placebo), timing, quake steps (monthly bins,
            group-median curve and per-channel), param steps/annual vs placebo, metadata epochs, channel QC,
            centroid B933 P check; summary.json + figures/sci_dvv_instrument_check*.png
Outputs: data/sci/dvv_instrument_check/
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import argparse
import json
import pickle
import sys
import zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = '/home/seismic/LFE'
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from dvv_cwire import epoch_s  # noqa: E402

OUT = 'data/sci/dvv_instrument_check'
FIG = 'figures'
NPROC = 8
os.makedirs(OUT, exist_ok=True)
Q = {'pet': pd.Timestamp('2021-12-20T20:10'), 'fer': pd.Timestamp('2022-12-20T10:34'), 'm7': pd.Timestamp('2024-12-05T18:44')}
MONTHS = pd.date_range('2008-01-01', '2026-07-01', freq='MS')
VARIANTS = ['V0', 'V1', 'V2', 'V3']

# instrument windows (as scripts/sci_sws.py EXCLUDE, B045 restricted to EH1); key matched with str.startswith
EXCL = {'PB.B045..EH1': [('2000-01-01', '2011-01-01')],
        'PB.B933': [('2017-01-01', '2021-11-01'), ('2024-08-01', '2030-01-01')],
        'PB.B935': [('2024-05-01', '2030-01-01')],
        'PB.B047': [('2000-01-01', '2009-03-01')],
        'NC.KMPB': [('2000-01-01', '2010-11-03')],
        'NC.KMR': [('2000-01-01', '2009-09-17'), ('2025-09-17', '2030-01-01')],
        'BK.BRIC': [('2022-12-01', '2024-05-22')],
        'BK.RVIT': [('2022-12-20T10:34', '2024-06-21')],
        'BK.BJES': [('2022-12-20T10:34', '2024-05-23')]}
BREAKS2 = {'PB.B933': ['2021-11-01'], 'BK.BRIC': ['2024-05-22'], 'BK.RVIT': ['2024-06-21'],
           'BK.BJES': ['2024-05-23'], 'NC.KMPB': ['2023-10-17T17:00']}


def sta_of(ch):
    return ch.rsplit('.', 2)[0]


# ------------------------------------------------------------------------------------------------ metadata
def read_meta():
    """all channel epochs of the pooled stations from data/stations (NCEDC + IRIS PB) text metadata."""
    rows = []
    for f in ['data/stations/cha_ncedc_2008_2026.txt', 'data/stations/cha_iris_PB_2008_2026.txt']:
        m = pd.read_csv(f, sep='|', dtype=str)
        m.columns = [c.strip().lstrip('#').strip() for c in m.columns]
        m = m.rename(columns={'SensorDescription': 'Instrument'})
        rows.append(m)
    M = pd.concat(rows, ignore_index=True)
    for c in M.columns:
        M[c] = M[c].fillna('').astype(str).str.strip()
    M = M[M.Channel.str.match(r'^(HH|EH|BH)[ZNE12]$')]
    M['ns'] = M.Network + '.' + M.Station
    M['Location'] = M.Location.replace({'nan': '', 'None': ''})
    M['start'] = pd.to_datetime(M.StartTime.str[:19], errors='coerce')
    M['end'] = pd.to_datetime(M.EndTime.str[:19].replace({'': '2100-01-01', 'nan': '2100-01-01'})
                              .str.replace('3000-01-01', '2100-01-01'), errors='coerce')
    for c in ['Scale', 'Azimuth', 'Dip', 'Depth', 'SampleRate']:
        M[c] = pd.to_numeric(M[c], errors='coerce')
    return M.drop_duplicates(['ns', 'Location', 'Channel', 'start'])


def epoch_changes(M, stations):
    """per station: each epoch start with what changed relative to the previous epoch of the same channel."""
    out = []
    for (ns, loc, cha), g in M[M.ns.isin(stations)].groupby(['ns', 'Location', 'Channel']):
        g = g.sort_values('start')
        prev = None
        for _, r in g.iterrows():
            if prev is not None:
                ch = []
                if r.Instrument != prev.Instrument:
                    ch.append(f'sensor {prev.Instrument.split(",")[0]} -> {r.Instrument.split(",")[0]}')
                if abs(r.Scale - prev.Scale) > 1e-3 * abs(prev.Scale):
                    ch.append(f'gain {prev.Scale:.4g} -> {r.Scale:.4g}')
                if abs(r.Azimuth - prev.Azimuth) > 0.05:
                    ch.append(f'az {prev.Azimuth:g} -> {r.Azimuth:g}')
                if abs(r.Depth - prev.Depth) > 0.05:
                    ch.append(f'depth {prev.Depth:g} -> {r.Depth:g}')
                if r.SampleRate != prev.SampleRate:
                    ch.append(f'sps {prev.SampleRate:g} -> {r.SampleRate:g}')
                gap = (r.start - prev.end).days if pd.notna(prev.end) else 0
                if gap > 1:
                    ch.append(f'gap {gap} d')
                out.append(dict(ns=ns, loc=loc, cha=cha, start=r.start, change='; '.join(ch) if ch else 'new epoch, no listed change'))
            else:
                out.append(dict(ns=ns, loc=loc, cha=cha, start=r.start, change='first epoch: ' + r.Instrument.split(',')[0]))
            prev = r
    E = pd.DataFrame(out)
    # one row per station and epoch start
    E = E.groupby(['ns', 'start']).agg(channels=('cha', lambda x: ','.join(sorted(set(x)))),
                                       change=('change', lambda x: ' | '.join(sorted(set(x))))).reset_index()
    return E


def breaks3(E):
    """V3 extra boundaries: sensor / orientation / gain epochs 2019-01..2026-07 not already a V2 boundary or
    inside an exclusion window."""
    B = {}
    e = E[(E.start >= '2019-01-01') & (E.start <= '2026-07-01') & ~E.change.str.startswith('first epoch')
          & ~(E.change == 'new epoch, no listed change')]
    for ns, g in e.groupby('ns'):
        for t in sorted(set(g.start)):
            if any(abs((t - pd.Timestamp(b)).days) <= 3 for b in BREAKS2.get(ns, [])):
                continue
            inside = any(pd.Timestamp(a) <= t < pd.Timestamp(b) for k, w in EXCL.items() if k.startswith(ns) and len(k) <= len(ns) + 1
                         for a, b in w)
            if inside:
                continue
            B.setdefault(ns, []).append(str(t))
    return B


def get_breaks():
    p = f'{OUT}/breaks.json'
    if os.path.exists(p):
        return json.load(open(p))
    M = read_meta()
    E = epoch_changes(M, sorted(M.ns.unique()))
    b3 = breaks3(E)
    B = {'V0': {}, 'V1': {}, 'V2': BREAKS2, 'V3': {k: sorted(set(BREAKS2.get(k, []) + b3.get(k, []))) for k in set(BREAKS2) | set(b3)}}
    json.dump(B, open(p, 'w'), indent=1)
    return B


def windows_for(ch):
    w = []
    for k, ww in EXCL.items():
        if ch.startswith(k):
            w += ww
    return [(epoch_s([pd.Timestamp(a)])[0], epoch_s([pd.Timestamp(b)])[0]) for a, b in w]


def bad_mask(ch, t):
    t = np.asarray(t, float)
    m = np.zeros(len(t), bool)
    for x0, x1 in windows_for(ch):
        m |= (t >= x0) & (t < x1)
    return m


def seg_of(ch, t, variant, B, shift_s=0.0):
    br = sorted(epoch_s([pd.Timestamp(x) for x in B[variant].get(sta_of(ch), [])]) - shift_s)
    t = np.asarray(t, float)
    return np.searchsorted(br, np.nan_to_num(t, nan=-1e18), side='right')


# ------------------------------------------------------------------------------------------------ pairs
def load_pairs():
    from dvv_cwire_pool import load_all
    P, T = load_all(0.4)
    P = P[P.cc >= 0.2].reset_index(drop=True)
    tb = np.empty(len(P))
    ta = np.full(len(P), np.nan)
    for F, idx in P.groupby('fam').indices.items():
        a, b = P.a.values[idx], P.b.values[idx]
        tb[idx] = T[F][b]
        ta[idx] = np.where(a >= 0, T[F][np.clip(a, 0, None)], np.nan)
    P['tb'], P['ta'] = tb, ta
    return P


def prep_channel(ch, p, variant, B):
    """apply masks / segments of a variant to one channel's rows; returns rows with segb, sega."""
    has_a = p.a.values >= 0
    if variant != 'V0':
        bad = bad_mask(ch, p.tb.values) | (has_a & bad_mask(ch, p.ta.values))
        p = p[~bad]
        has_a = p.a.values >= 0
    sb = seg_of(ch, p.tb.values, variant, B)
    sa = np.where(has_a, seg_of(ch, p.ta.values, variant, B), -1)
    keep = ~has_a | (sa == sb)
    p = p[keep].assign(segb=sb[keep], sega=sa[keep])
    return p.reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ monthly
def month_index(t):
    d = pd.to_datetime(t, unit='s')
    return ((d.year - 2008) * 12 + d.month - 1).values


def fit_monthly(args):
    ch, variant, p = args
    has_a = p.a.values >= 0
    mb = month_index(p.tb.values)
    ma = np.where(has_a, month_index(np.where(has_a, p.ta.values, p.tb.values)), -1)
    ok = (mb >= 0) & (mb < len(MONTHS)) & (~has_a | ((ma >= 0) & (ma < len(MONTHS))))
    p, mb, ma, has_a = p[ok], mb[ok], ma[ok], has_a[ok]
    segb, sega = p.segb.values, p.sega.values
    ub = mb * 100 + segb
    ua = ma * 100 + sega
    units = np.unique(np.r_[ub, ua[has_a]])
    ib, ia = np.searchsorted(units, ub), np.searchsorted(units, ua[has_a])
    kk = p.fam.values * 100 + segb
    keys = np.unique(kk[~has_a])
    ik = np.searchsorted(keys, kk[~has_a])
    nu, nk, n = len(units), len(keys), len(p)
    rows = np.r_[np.arange(n), np.where(has_a)[0], np.where(~has_a)[0]]
    cols = np.r_[ib, ia, nu + ik]
    vals = np.r_[np.ones(n), -np.ones(has_a.sum()), -np.ones((~has_a).sum())]
    G = sp.csr_matrix((vals, (rows, cols)), shape=(n, nu + nk))
    d = p.eps.values
    cc = np.clip(p.cc.values, 0, 0.99)
    w0 = cc ** 2 / (1 - cc ** 2)
    w0 /= w0.mean()
    w = w0.copy()
    useg = units % 100
    ridge = np.zeros((nu + nk, nu + nk))
    for s in np.unique(useg):
        u = np.r_[(useg == s).astype(float), np.zeros(nk)]
        ridge += np.outer(u, u) / u.sum()
    for _ in range(5):
        N = (G.T @ (sp.diags(w) @ G)).toarray() + ridge
        x = np.linalg.lstsq(N, G.T @ (w * d), rcond=1e-12)[0]
        r = d - G @ x
        s = 1.4826 * np.median(np.abs(r)) + 1e-12
        z = np.abs(r) / (2 * s)
        w = w0 * np.where(z <= 1, 1.0, 1.0 / z)
    s2 = np.sum(w * r ** 2) / max(n - nu - nk, 1)
    xm = x[:nu].copy()
    for s_ in np.unique(useg):
        xm[useg == s_] -= xm[useg == s_].mean()
    Ninv = np.linalg.pinv(N)
    se = np.sqrt(np.clip(np.diag(Ninv)[:nu] * s2, 0, None))
    cnt = pd.Series(1, index=ub).groupby(level=0).size().reindex(units).fillna(0).values
    # reference-row median cc per unit (channel QC)
    ccr = pd.Series(p.cc.values[~has_a], index=ub[~has_a]).groupby(level=0).median().reindex(units).values
    return pd.DataFrame(dict(variant=variant, chan=ch, month=MONTHS[units // 100], seg=useg, dvv_pct=xm * 100,
                             se_pct=se * 100, n_rows=cnt, ref_cc_median=ccr))


def stage_monthly(args):
    B = get_breaks()
    P = load_pairs()
    jobs = []
    for ch, p in P.groupby('chan'):
        if len(p) < 3000:
            continue
        for v in VARIANTS:
            q = prep_channel(ch, p, v, B)
            if len(q) >= 1000:
                jobs.append((ch, v, q[['fam', 'a', 'b', 'eps', 'cc', 'tb', 'ta', 'segb', 'sega']]))
    print(len(jobs), 'monthly jobs', flush=True)
    with Pool(NPROC) as pool:
        res = pool.map(fit_monthly, jobs, chunksize=2)
    M = pd.concat(res, ignore_index=True)
    M.to_csv(f'{OUT}/monthly_bins_variants.csv.gz', index=False, float_format='%.6g')
    print(M.groupby('variant').chan.nunique(), flush=True)


# ------------------------------------------------------------------------------------------------ param
def param_job(args):
    ch, variant, p, dates, pers, nboot = args
    from sci_dvv_placebo_fit import run_fit
    p = p.copy()
    sb = p.fam.values * 100000 + p.b.values
    sa = np.where(p.a.values >= 0, p.fam.values * 100000 + p.a.values, -1)
    p['chan'] = p.fam.astype(str) + '_' + p.segb.astype(str)    # reference offset per (family, segment)
    try:
        r = run_fit(p.reset_index(drop=True), dates, pers, nboot, zlib.crc32(ch.encode()), (sb, sa))
    except Exception as e:          # noqa: BLE001
        print('param fail', ch, variant, e, flush=True)
        return (variant, ch), None
    # identifiability of each quake step: some segment with >= 3 stacks within 1 yr on both sides
    ident = {}
    tb, sg = p.tb.values, p.segb.values
    for k, q in Q.items():
        tq = epoch_s([q])[0]
        ok = False
        for s in np.unique(sg):
            x = np.unique(tb[sg == s])
            if ((x < tq) & (x > tq - 365 * 86400)).sum() >= 3 and ((x >= tq) & (x < tq + 365 * 86400)).sum() >= 3:
                ok = True
        ident[k] = ok
    r['ident'] = ident
    r['n_seg'] = int(len(np.unique(sg)))
    return (variant, ch), r


def stage_param(args):
    from sci_transient_all import placebo_dates
    from sci_dvv_placebo import periods
    B = get_breaks()
    P = load_pairs()
    dates, pers = placebo_dates(), periods()
    jobs = []
    for ch, p in P.groupby('chan'):
        if len(p) < 3000:
            continue
        for v in VARIANTS:
            q = prep_channel(ch, p, v, B)
            if len(q) >= 3000:
                jobs.append((ch, v, q, dates, pers, args.nboot))
    print(len(jobs), 'param jobs', flush=True)
    res = {}
    with Pool(NPROC) as pool:
        for k, r in pool.imap_unordered(param_job, jobs):
            res[k] = r
    pickle.dump(dict(dates=dates, periods=pers, res=res), open(f'{OUT}/param_fits.pkl', 'wb'))


# ------------------------------------------------------------------------------------------------ noise
NOISE_BREAK = {'BRIC': ['2024-05-22'], 'KMPB': ['2023-10-17'], 'HAYF': ['2025-05-06'], 'B045': []}
NOISE_MASK = {'BRIC': [('2022-12-01', '2024-05-22')], 'KMPB': [], 'HAYF': [], 'B045': []}


def noise_station(name):
    import sci_noise_dvv as nd
    nd.COMBOS = {'ALL': list(range(6)), 'SC': [3, 4, 5], 'AC': [0, 1, 2], 'ZZ': [0], 'HAC': [1, 2]}
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    days, ccf = nd.load_ccf(name)
    keep, _ = nd.detect_lines(days, ccf)
    ccf = nd.notch(ccf, keep)
    out = dict(rows10=[], steps=[], fits=[])
    for variant in ['orig', 'masked']:
        c = ccf.copy() if variant == 'masked' else ccf
        if variant == 'masked':
            for a, b in NOISE_MASK[name]:
                c[(days >= a) & (days < b)] = np.nan
        ref, _ = nd.reference(days, c)
        k10 = ((days - pd.Timestamp(nd.START)).days // 10).values
        labs, S10, N10 = nd.stack_groups(days, c, k10, 5)
        t10 = pd.Timestamp(nd.START) + pd.to_timedelta(labs * 10 + 5, unit='D')
        dv10, cc10, er10 = nd.measure_stacks(ref, S10)
        good = nd.pair_quality(cc10, t10) >= 0.3
        er10g = np.where(good[None], er10, np.nan)
        refs = {'ref2019_21': ref}
        if name == 'BRIC' and variant == 'masked':
            m2 = (days >= '2024-06-01') & (days < '2024-11-05')
            with np.errstate(invalid='ignore'):
                refs['ref_post_reinstall'] = np.nanmean(c[m2], axis=0)
        for cmb, ix in nd.COMBOS.items():
            m, e = nd.combine(dv10[:, :, ix], er10g[:, :, ix])
            for b in range(len(nd.BANDS)):
                for g in range(len(t10)):
                    out['rows10'].append(dict(sta=name, variant=variant, band=nd.BAND_NAMES[b], combo=cmb, t=t10[g],
                                              dvv_pct=m[g, b] * 100, err_pct=e[g, b] * 100))
        # 30-d event steps vs placebo (pre-2021-11 dates, 5-d spacing) for each reference
        first = days[np.isfinite(c[:, 0, :, 0]).any(axis=1)][0]
        pl_dates = pd.date_range(first + pd.Timedelta(days=31), pd.Timestamp('2021-11-15'), freq='5D')
        for rname, R in refs.items():
            ev = {k: nd.event_step(days, c, R, te, good, nboot=50, rng=rng) for k, te in nd.EVENTS.items()}
            pl = {cm: [] for cm in nd.COMBOS}
            for td in pl_dates:
                r = nd.event_step(days, c, R, td, good)
                if r is not None:
                    for cm in nd.COMBOS:
                        pl[cm].append(r[cm][0])
            for k, r in ev.items():
                for cm in nd.COMBOS:
                    for b in range(len(nd.BANDS)):
                        pv = np.array(pl[cm])[:, b] if len(pl[cm]) else np.array([np.nan])
                        pv = pv[np.isfinite(pv)]
                        st = r[cm][0, b] if r is not None else np.nan
                        out['steps'].append(dict(sta=name, variant=variant, ref=rname, event=k, combo=cm,
                                                 band=nd.BAND_NAMES[b], step_pct=st,
                                                 se_mbb_pct=np.nanstd(r[cm][1:, b]) if r is not None else np.nan,
                                                 placebo_sd_pct=np.std(pv) if len(pv) > 5 else np.nan,
                                                 z_placebo=st / np.std(pv) if len(pv) > 5 else np.nan, n_placebo=len(pv)))
    return out


def noise_fit(D, name, breaks, rng):
    """10-day series fits: c (+ segment offsets) + annual + steps; D_2021_2025 from yearly means (connected) and
    model-based with free offsets; moving-block bootstrap (block 9 stacks ~ 90 d)."""
    rows = []
    for (variant, band, cmb), g in D.groupby(['variant', 'band', 'combo']):
        g = g.dropna(subset=['dvv_pct']).sort_values('t')
        t = pd.to_datetime(g.t)
        y = g.dvv_pct.values
        if len(y) < 60:
            continue
        ty = (t - pd.Timestamp('2019-01-01')).dt.days.values / 365.25
        for mode in ['connected', 'free_offset']:
            br = breaks if mode == 'free_offset' else []
            seg = (np.searchsorted(np.array([pd.Timestamp(b) for b in br], dtype='datetime64[ns]'), t.values, side='right')
                   if br else np.zeros(len(t), int))
            cols, names = [], []
            for s in np.unique(seg):
                cols.append((seg == s).astype(float))
                names.append(f'off{s}')
            cols += [np.cos(2 * np.pi * ty), np.sin(2 * np.pi * ty)]
            names += ['acos', 'asin']
            for k, q in Q.items():
                h = (t.values > np.datetime64(q)).astype(float)
                # identifiable only if some segment has >= 3 stacks within 1 yr on both sides
                okq = any(((seg == s) & (t.values < np.datetime64(q)) & (t.values > np.datetime64(q - pd.Timedelta(days=365)))).sum() >= 3
                          and ((seg == s) & (t.values > np.datetime64(q)) & (t.values < np.datetime64(q + pd.Timedelta(days=365)))).sum() >= 3
                          for s in np.unique(seg))
                if okq:
                    cols.append(h)
                    names.append('step_' + k)
            X = np.array(cols).T
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            res = y - X @ beta
            bb = []
            for _ in range(300):
                st0 = rng.integers(0, len(res) - 9 + 1, size=int(np.ceil(len(res) / 9)))
                rr = np.concatenate([res[s:s + 9] for s in st0])[:len(res)]
                bb.append(np.linalg.lstsq(X, X @ beta + rr, rcond=None)[0])
            bse = np.std(bb, axis=0)
            co = dict(zip(names, beta))
            ce = dict(zip(names, bse))
            # model-based 2021 -> 2025 change excluding offsets: sum of quake steps active in 2025 minus 2021
            stp = [n for n in names if n.startswith('step_')]
            Dm = sum(co[n] for n in stp)
            Dm_se = np.std([sum(b_[names.index(n)] for n in stp) for b_ in bb]) if stp else np.nan
            # yearly-mean difference (deseasoned), only if 2021 and 2025 in the same segment
            yd = y - X[:, names.index('acos')] * co['acos'] - X[:, names.index('asin')] * co['asin']
            yr = t.dt.year.values
            same = len(set(seg[yr == 2021]) | set(seg[yr == 2025])) == 1 and (yr == 2021).sum() >= 9 and (yr == 2025).sum() >= 9
            Dy = yd[yr == 2025].mean() - yd[yr == 2021].mean() if same else np.nan
            rows.append(dict(sta=name, variant=variant, band=band, combo=cmb, mode=mode, n=len(y),
                             breaks=';'.join(br), steps_identifiable=';'.join(stp),
                             D_yearly_2021_2025_pct=Dy, D_model_steps_sum_pct=Dm, D_model_se_pct=Dm_se,
                             **{f'{n}_pct': co.get(n, np.nan) for n in ['step_pet', 'step_fer', 'step_m7']},
                             **{f'{n}_se_pct': ce.get(n, np.nan) for n in ['step_pet', 'step_fer', 'step_m7']},
                             annual_amp_pct=np.hypot(co['acos'], co['asin'])))
    return rows


def stage_noise(args):
    names = ['BRIC', 'KMPB', 'B045', 'HAYF']
    R10, ST = [], []
    for n in names:              # sequential: each CCF array is ~0.6 GB
        print('noise', n, flush=True)
        o = noise_station(n)
        R10 += o['rows10']
        ST += o['steps']
    D = pd.DataFrame(R10)
    D.to_csv(f'{OUT}/noise_10day_variants.csv.gz', index=False, float_format='%.6g')
    pd.DataFrame(ST).to_csv(f'{OUT}/noise_steps_variants.csv', index=False, float_format='%.6g')
    stage_noisefit(args)


def stage_noisefit(args):
    names = ['BRIC', 'KMPB', 'B045', 'HAYF']
    D = pd.read_csv(f'{OUT}/noise_10day_variants.csv.gz', parse_dates=['t'])
    rng = np.random.default_rng(7)
    F = []
    for n in names:
        F += noise_fit(D[D.sta == n], n, NOISE_BREAK[n], rng)
    pd.DataFrame(F).to_csv(f'{OUT}/noise_longterm_fits.csv', index=False, float_format='%.6g')



# ================================================================================================ analyze
GRP = {'borehole': lambda c: c.startswith('PB.'), 'surface': lambda c: not c.startswith('PB.'),
       'surface_long': lambda c: sta_of(c) in ('NC.KMPB', 'NC.KMR'),
       'surface_early': lambda c: (not c.startswith('PB.')) and c in EARLY}
EARLY = set()          # channels whose (V0) record starts before 2020-07 (as sci_decline.timing 'early')
PLACEBO_Y = list(range(2009, 2017))
SUMMARY = {}


def zp(x, null):
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if len(null) < 3 or not np.isfinite(x):
        return dict(value=float(x) if np.isfinite(x) else None, z=None, n_null=int(len(null)))
    return dict(value=float(x), null_mean=float(null.mean()), null_sd=float(null.std(ddof=1)),
                z=float((x - null.mean()) / null.std(ddof=1)), n_null=int(len(null)),
                null_min=float(null.min()), null_max=float(null.max()))


def load_variant_bins():
    B = pd.read_csv(f'{OUT}/monthly_bins_variants.csv.gz', parse_dates=['month'])
    B = B[np.isfinite(B.dvv_pct) & (B.n_rows >= 20)]
    V = {}
    for (v, ch), b in B.groupby(['variant', 'chan']):
        parts = []
        for s, bs in b.groupby('seg'):
            x = bs.set_index('month').dvv_pct
            dev = x - x.rolling(13, center=True, min_periods=5).median()
            ok = (np.abs(dev) <= 6 * 1.4826 * np.nanmedian(np.abs(dev))) | dev.isna()
            parts.append(bs[ok.values])
        V.setdefault(v, {})[ch] = pd.concat(parts).sort_values(['month', 'seg']).reset_index(drop=True)
    EARLY.update(ch for ch, b in V['V0'].items() if b.month.min() < pd.Timestamp('2020-07-01'))
    return V


def deseason(b):
    """annual sin/cos from pre-Petrolia months (segment offsets free); short records: all months with
    identifiable quake steps + trend.  Returns b with column y (deseasoned dv/v, %)."""
    t = b.month + pd.Timedelta(days=14)
    yr = (t - pd.Timestamp('2008-01-01')).dt.total_seconds().values / (365.25 * 86400)
    seg = b.seg.values
    segs = np.unique(seg)
    D = np.array([(seg == s).astype(float) for s in segs]).T
    SC = np.c_[np.sin(2 * np.pi * yr), np.cos(2 * np.pi * yr)]
    v = b.dvv_pct.values
    pre = (t < Q['pet']).values
    if pre.sum() >= 24:
        A = np.c_[D, SC][pre]
        keep = A.any(axis=0)
        m = np.linalg.lstsq(A[:, keep], v[pre], rcond=None)[0]
        coef = np.zeros(A.shape[1])
        coef[keep] = m
        sc = coef[-2:]
    else:
        ex = [(t >= q).astype(float).values for q in Q.values()] + [yr]
        A = np.c_[D, SC, np.array(ex).T]
        m = np.linalg.lstsq(A, v, rcond=None)[0]
        sc = m[len(segs):len(segs) + 2]
    return b.assign(y=v - SC @ sc, t=t)


def yearly_table(Vd, min_n=6):
    """per channel: deseasoned yearly means per (seg, year) with >= min_n months."""
    out = {}
    for ch, b in Vd.items():
        g = b.groupby([b.seg, b.month.dt.year]).y.agg(['mean', 'size'])
        out[ch] = g[g['size'] >= min_n]['mean']
    return out


def D_years(Y, ch, y0, y1):
    s = Y[ch]
    best = np.nan
    nbest = -1
    for sg in s.index.get_level_values(0).unique():
        if (sg, y0) in s.index and (sg, y1) in s.index:
            best = s[(sg, y1)] - s[(sg, y0)]
            nbest = 1
    return best


def decline_yearly(V):
    rows, res = [], {}
    for v in VARIANTS:
        Vd = {ch: deseason(b) for ch, b in V[v].items()}
        Y = yearly_table(Vd)
        D = pd.Series({ch: D_years(Y, ch, 2021, 2025) for ch in Y})
        P = pd.DataFrame({y: pd.Series({ch: D_years(Y, ch, y, y + 4) for ch in Y}) for y in PLACEBO_Y})
        for ch in Y:
            rows.append(dict(variant=v, chan=ch, sta=sta_of(ch), D_2021_2025_pct=D[ch]))
        for g, f in GRP.items():
            c = D[[f(ch) for ch in D.index]].dropna()
            nl = [np.median(P.loc[c.index, y].dropna()) for y in PLACEBO_Y if P.loc[c.index, y].notna().sum() >= 3]
            res[(v, g)] = dict(n_chan=len(c), chans=';'.join(c.index), **zp(np.median(c), nl))
    T = pd.DataFrame(rows)
    T.to_csv(f'{OUT}/decline_per_channel.csv', index=False, float_format='%.5g')
    st = T.groupby(['variant', 'sta']).D_2021_2025_pct.median().unstack(0)
    st.to_csv(f'{OUT}/decline_per_station.csv', float_format='%.5g')
    SUMMARY['decline_yearly'] = {f'{v}_{g}': r for (v, g), r in res.items()}
    return res, st


def model_decline_channel(b, breaks, shift_yr, kind):
    """b deseasoned (V1, connected); free offsets at `breaks` (Timestamps); quake steps + optional ramp from
    2021-06; all dates shifted by -shift_yr years (placebo: data restricted to before 2021-06-01)."""
    sh = pd.Timedelta(days=365.25 * shift_yr)
    t = b.t
    y = b.y.values
    if shift_yr > 0:
        m = (t < pd.Timestamp('2021-06-01')).values
        t, y = t[m], y[m]
    y0a, y0b = pd.Timestamp('2021-01-01') - sh, pd.Timestamp('2022-01-01') - sh
    y1a, y1b = pd.Timestamp('2025-01-01') - sh, pd.Timestamp('2026-01-01') - sh
    if ((t >= y0a) & (t < y0b)).sum() < 6 or ((t >= y1a) & (t < y1b)).sum() < 6:
        return np.nan
    brs = sorted(pd.Timestamp(x) - sh for x in breaks)
    seg = np.searchsorted(np.array(brs, dtype='datetime64[ns]'), t.values, side='right') if brs else np.zeros(len(t), int)
    segs = np.unique(seg)
    cols = [(seg == s).astype(float) for s in segs]
    fns = []
    for q in Q.values():
        qs = q - sh
        ok = any(((seg == s) & (t < qs) & (t > qs - pd.Timedelta(days=365))).sum() >= 3 and
                 ((seg == s) & (t > qs) & (t < qs + pd.Timedelta(days=365))).sum() >= 3 for s in segs)
        if ok:
            fns.append(lambda tt, qs=qs: (tt >= qs).astype(float))
    if kind == 'steps+ramp':
        h = pd.Timestamp('2021-06-01') - sh
        fns.append(lambda tt, h=h: np.clip((tt - h) / pd.Timedelta(days=365.25), 0, None))
    X = np.array(cols + [np.asarray(f(t)) for f in fns]).T
    m = np.linalg.lstsq(X, y, rcond=None)[0]
    beta = m[len(segs):]
    grid0 = pd.date_range(y0a, y0b, freq='MS', inclusive='left') + pd.Timedelta(days=14)
    grid1 = pd.date_range(y1a, y1b, freq='MS', inclusive='left') + pd.Timedelta(days=14)
    pr = lambda g: np.array([np.asarray(f(pd.Series(g))) for f in fns]).T @ beta  # noqa: E731
    return float(pr(grid1).mean() - pr(grid0).mean())


def decline_model(V, B):
    Vd = {ch: deseason(b) for ch, b in V['V1'].items()}
    shifts = np.arange(5.25, 12.3, 0.5)
    rows, res = [], {}
    for mode, brk in [('V1_connected', {}), ('V2_free_offsets', B['V2']), ('V3_free_offsets', B['V3'])]:
        for kind in ['steps+ramp', 'steps']:
            obs = {ch: model_decline_channel(b, brk.get(sta_of(ch), []), 0.0, kind) for ch, b in Vd.items()}
            pl = {s: {ch: model_decline_channel(b, brk.get(sta_of(ch), []), s, kind) for ch, b in Vd.items()} for s in shifts}
            for ch in obs:
                rows.append(dict(mode=mode, kind=kind, chan=ch, D_model_pct=obs[ch]))
            for g, f in GRP.items():
                c = pd.Series({ch: v for ch, v in obs.items() if f(ch)}).dropna()
                nl = [np.nanmedian([pl[s][ch] for ch in c.index]) for s in shifts]
                res[f'{mode}_{kind}_{g}'] = dict(n_chan=len(c), **zp(np.median(c), nl))
    pd.DataFrame(rows).to_csv(f'{OUT}/decline_model_per_channel.csv', index=False, float_format='%.5g')
    SUMMARY['decline_model'] = res
    return res


def to_C(bins, baseline_only=False):
    C = {}
    for ch, b in bins.items():
        if baseline_only:
            mb = b[(b.month >= '2019-01-01') & (b.month <= '2021-06-30')]
            if len(mb) == 0:
                continue
            b = b[b.seg == mb.seg.value_counts().idxmax()]
        C[ch] = pd.DataFrame(dict(t=b.month + pd.Timedelta(days=14), dvv_pct=b.dvv_pct.values, err_pct=b.se_pct.values)).reset_index(drop=True)
    return C


def timing_all(V):
    import sci_decline as sd
    sd.OUT = OUT
    out = {}
    curves = {}
    for v in VARIANTS:
        C = to_C(V[v], baseline_only=(v in ('V2', 'V3')))
        cv, _ = sd.timing(C, f'timing_{v}')
        curves[v] = cv
        T = sd.S[f'timing_{v}']
        r = {}
        for g, x in T.items():
            hf = x['fits']['hinge']
            r[g] = dict(n_chan=x['n_chan'], hinge_t0=hf['t0'], hinge_t0_ci=hf['t0_ci'], hinge_boot=hf['t0_boot_p05_p50_p95'],
                        frac_boot_before_pet=hf['frac_boot_t0_before_pet'], hinge_ramp_pct_per_yr=hf['ramp'],
                        aic_hinge=hf['aic'], aic_steps=x['fits']['steps']['aic'], aic_hinge_steps=x['fits']['hinge+steps']['aic'],
                        h2021H2=x['2021H2_minus_prev2yr'], h2021H1=x['2021H1_minus_prev2yr'],
                        slope_2021=x['slope_2021Jan_Nov_pct_per_yr'],
                        **{f'level_{q}': x[f'level_change_{q}'] for q in Q},
                        **{f'slope_{k}': x[f'slope_{k}_pct_per_yr'] for k in ['pet_to_fer', 'fer_to_m7', 'after_m7']},
                        steps_fit={k: x['fits']['steps'][f'step_{k}'] for k in Q})
        out[v] = r
    SUMMARY['timing'] = out
    return curves


def level_steps_channel(V):
    """per channel: 11-month mean after minus 11 before (quake month excluded), same segment, >= 6 months each;
    group median; placebo = every month 2010-01..2019-06."""
    res, rows = {}, []
    pl_months = pd.date_range('2010-01-01', '2019-06-01', freq='MS')
    for v in VARIANTS:
        Vd = {ch: deseason(b) for ch, b in V[v].items()}
        S = {}
        for ch, b in Vd.items():
            ser = {s: bs.set_index('month').y for s, bs in b.groupby('seg')}
            S[ch] = ser

        def L(ch, m0):
            best = np.nan
            for s, x in S[ch].items():
                bf = x[(x.index >= m0 - pd.DateOffset(months=11)) & (x.index < m0)]
                af = x[(x.index > m0) & (x.index <= m0 + pd.DateOffset(months=11))]
                if len(bf) >= 6 and len(af) >= 6:
                    best = af.mean() - bf.mean()
            return best
        obs = {q: {ch: L(ch, pd.Timestamp(t.year, t.month, 1)) for ch in S} for q, t in Q.items()}
        pl = {m0: {ch: L(ch, m0) for ch in S} for m0 in pl_months}
        for q in Q:
            for ch in S:
                rows.append(dict(variant=v, quake=q, chan=ch, level_change_pct=obs[q][ch]))
            for g, f in GRP.items():
                c = pd.Series({ch: x for ch, x in obs[q].items() if f(ch)}).dropna()
                nl = [np.nanmedian([pl[m0][ch] for ch in S if f(ch)]) for m0 in pl_months]
                res[f'{v}_{g}_{q}'] = dict(n_chan=len(c), **zp(np.median(c) if len(c) else np.nan, nl))
    pd.DataFrame(rows).to_csv(f'{OUT}/level_steps_per_channel.csv', index=False, float_format='%.5g')
    SUMMARY['level_steps_channel_median'] = res
    return res


def param_pool():
    import sci_dvv_placebo as spl
    D = pickle.load(open(f'{OUT}/param_fits.pkl', 'rb'))
    res = D['res']
    rows, units = [], []
    for v in VARIANTS:
        rv = {('chan', ch): r for (vv, ch), r in res.items() if vv == v and r is not None}
        U, PL = spl.unit_rows(rv, 'chan')
        drop = [k for k in PL if k[1].startswith('step_') and not rv[('chan', k[2])]['ident'][k[1][5:]]]
        for k in drop:
            del PL[k]
        U['variant'] = v
        units.append(U)
        for g in ['surface', 'borehole']:
            for q in Q:
                r = spl.pool_steps(PL, g, q)
                if r:
                    rows.append(dict(variant=v, **r))
            rows.append(dict(variant=v, **spl.pool_annual(PL, g)))
    P = pd.DataFrame(rows)
    P.to_csv(f'{OUT}/param_pooled.csv', index=False, float_format='%.5g')
    pd.concat(units).to_csv(f'{OUT}/param_per_channel.csv', index=False, float_format='%.5g')
    SUMMARY['param_pooled'] = P[['variant', 'group', 'quantity', 'n_units', 'pooled_pct', 'z_placebo', 'p_emp', 'se_placebo_pct',
                                 'median_unit_pct', 'pooled_peak_doy']].round(4).to_dict('records')
    return P


def metadata_steps(V):
    """all metadata epoch starts 2008-2026 of the pooled stations; distance to the quakes; monthly dv/v jump
    (5 months after vs before, V1 deseasoned, same segment) vs the channel's own placebo dates."""
    M = read_meta()
    pool_sta = sorted({sta_of(c) for c in V['V1']})
    E = epoch_changes(M, pool_sta)
    E = E[(E.start >= '2008-01-01') & ~E.change.str.startswith('first epoch')]
    Vd = {ch: deseason(b) for ch, b in V['V1'].items()}
    rows = []
    for _, e in E.iterrows():
        dq = {k: (e.start - q).days for k, q in Q.items()}
        kq = min(dq, key=lambda k: abs(dq[k]))
        for ch, b in Vd.items():
            if sta_of(ch) != e.ns:
                continue
            x = b.set_index('t').y

            def jump(tt):
                a = x[(x.index >= tt) & (x.index < tt + pd.Timedelta(days=150))]
                bb = x[(x.index < tt) & (x.index >= tt - pd.Timedelta(days=150))]
                return a.mean() - bb.mean() if len(a) >= 3 and len(bb) >= 3 else np.nan
            j = jump(e.start)
            cand = x.index[(x.index > x.index.min() + pd.Timedelta(days=200)) & (x.index < x.index.max() - pd.Timedelta(days=200))]
            nul = np.array([jump(tt) for tt in cand if min(abs((tt - q).days) for q in Q.values()) > 200 and abs((tt - e.start).days) > 200])
            nul = nul[np.isfinite(nul)]
            rows.append(dict(sta=e.ns, epoch_start=str(e.start), change=e.change, nearest_quake=kq, days_from_quake=dq[kq],
                             chan=ch, jump_pct=j, null_sd=nul.std(ddof=1) if len(nul) > 8 else np.nan,
                             z=(j - nul.mean()) / nul.std(ddof=1) if len(nul) > 8 and np.isfinite(j) else np.nan))
        if not any(sta_of(ch) == e.ns for ch in Vd):
            rows.append(dict(sta=e.ns, epoch_start=str(e.start), change=e.change, nearest_quake=kq, days_from_quake=dq[kq]))
    T = pd.DataFrame(rows)
    T.to_csv(f'{OUT}/metadata_epochs_vs_dvv.csv', index=False, float_format='%.4g')
    E.assign(start=E.start.astype(str)).to_csv(f'{OUT}/metadata_epochs_all.csv', index=False)
    close = T[T.days_from_quake.abs() <= 90].drop_duplicates(['sta', 'epoch_start'])[['sta', 'epoch_start', 'change', 'nearest_quake', 'days_from_quake']]
    SUMMARY['metadata'] = dict(n_epochs=int(len(E)), within_90d_of_quake=close.to_dict('records'),
                               max_abs_z=T.dropna(subset=['z']).sort_values('z', key=np.abs, ascending=False).head(8)
                               [['chan', 'epoch_start', 'change', 'jump_pct', 'z']].round(4).to_dict('records'))
    return T


def channel_qc():
    """reference-row coda CC per month (V0): inside vs outside the instrument windows, per channel."""
    B = pd.read_csv(f'{OUT}/monthly_bins_variants.csv.gz', parse_dates=['month'])
    B = B[(B.variant == 'V0') & B.ref_cc_median.notna()]
    rows = []
    for ch, b in B.groupby('chan'):
        w = windows_for(ch)
        if not w:
            continue
        t = epoch_s(b.month + pd.Timedelta(days=14))
        inside = bad_mask(ch, t)
        if inside.sum() < 2:
            continue
        x = b.ref_cc_median.values
        o = x[~inside]
        rows.append(dict(chan=ch, n_in=int(inside.sum()), cc_in=np.median(x[inside]), cc_out=np.median(o),
                         z=(np.median(x[inside]) - np.median(o)) / (1.4826 * np.median(np.abs(o - np.median(o))) + 1e-9)))
    T = pd.DataFrame(rows)
    T.to_csv(f'{OUT}/channel_qc_refcc.csv', index=False, float_format='%.4g')
    SUMMARY['channel_qc_refcc'] = T.round(3).to_dict('records')
    return T


def centroid_b933():
    D = pd.read_csv('data/sci/centroids/station_delay_change_M7.csv')

    def wm(g):
        w = 1 / g.se_ms ** 2
        return pd.Series(dict(ddt_ms=np.sum(w * g.ddt_ms) / w.sum(), se_ms=1 / np.sqrt(w.sum()), n_fam=len(g),
                              median_ms=g.ddt_ms.median()))
    a = D.groupby(['sta', 'phase']).apply(wm, include_groups=False)
    c = D[D.ddt_ms.abs() <= 50].groupby(['sta', 'phase']).apply(wm, include_groups=False)
    out = a.join(c, rsuffix='_noskip').reset_index()
    out.to_csv(f'{OUT}/centroid_station_delay_M7_recheck.csv', index=False, float_format='%.4g')
    b = D[(D.sta == 'PB.B933') & (D.phase == 'P')].sort_values('F')
    r = out[(out.sta == 'PB.B933') & (out.phase == 'P')].iloc[0].to_dict()
    r['per_family'] = b[['F', 'ddt_ms', 'se_ms']].round(2).to_dict('records')
    r['window_note'] = ('delay change = post minus pre within +/-0.75 yr of the M7 (2024-03..2025-09); B933 is flagged '
                        'degraded from 2024-08-01, so the whole post window and ~4 of 9 pre months lie in the flagged window')
    # other stations with |z|>3 after removing cycle skips
    out['z_noskip'] = out.ddt_ms_noskip / out.se_ms_noskip
    r['other_P_noskip_absz_gt3'] = out[(out.phase == 'P') & (out.z_noskip.abs() > 3)][['sta', 'ddt_ms_noskip', 'se_ms_noskip', 'n_fam_noskip']].round(2).to_dict('records')
    SUMMARY['centroid_B933_P'] = r
    return out


def noise_summary():
    F = pd.read_csv(f'{OUT}/noise_longterm_fits.csv')
    S = pd.read_csv(f'{OUT}/noise_steps_variants.csv')
    orig_lt = pd.read_csv('data/sci/noise_dvv/noise_dvv_longterm.csv')
    res = {}
    for sta in ['BRIC', 'KMPB', 'B045', 'HAYF']:
        r = {}
        for band in ['1-4Hz', '2-8Hz']:
            for cmb in ['ALL', 'AC', 'SC', 'ZZ']:
                f = F[(F.sta == sta) & (F.band == band) & (F.combo == cmb)]
                k = f'{band}_{cmb}'
                r[k] = {f'{x.variant}_{x.mode}': dict(D_yearly=x.D_yearly_2021_2025_pct, D_steps_sum=x.D_model_steps_sum_pct,
                                                      D_steps_se=x.D_model_se_pct, step_pet=x.step_pet_pct, step_fer=x.step_fer_pct,
                                                      step_m7=x.step_m7_pct, step_m7_se=x.step_m7_se_pct, annual=x.annual_amp_pct)
                        for x in f.itertuples()}
                s = S[(S.sta == sta) & (S.band == band) & (S.combo == cmb)]
                r[k]['steps30d'] = {f'{x.variant}_{x.ref}_{x.event}': dict(step=x.step_pct, z=x.z_placebo) for x in s.itertuples()}
        res[sta] = r
    SUMMARY['noise'] = res
    return F, S


def claims_table(dec, dmod, tim, lev, P, st):
    """compact claim table: original (V0), windows excluded (V1), free offsets (V2, V3)."""
    rows = []

    def add(claim, metric, vals, verdict=''):
        rows.append(dict(claim=claim, metric=metric, **vals, verdict=verdict))
    for g in ['surface', 'borehole', 'surface_long']:
        add(f'decline 2021->2025 {g}', 'median D (%) [z vs pre-2021 4-yr windows]',
            {v: f"{dec[(v, g)]['value']:+.3f} [z {dec[(v, g)]['z']:.1f}, n={dec[(v, g)]['n_chan']}]" if dec[(v, g)].get('z') is not None else 'n/a' for v in VARIANTS})
        add(f'decline model {g} (steps+ramp, free offsets)', 'median D_model (%) [z vs date-shifted placebo]',
            {'V1': f"{dmod[f'V1_connected_steps+ramp_{g}']['value']:+.3f} [z {dmod[f'V1_connected_steps+ramp_{g}']['z']:.1f}]",
             'V2': f"{dmod[f'V2_free_offsets_steps+ramp_{g}']['value']:+.3f} [z {dmod[f'V2_free_offsets_steps+ramp_{g}']['z']:.1f}]",
             'V3': f"{dmod[f'V3_free_offsets_steps+ramp_{g}']['value']:+.3f} [z {dmod[f'V3_free_offsets_steps+ramp_{g}']['z']:.1f}]"})
    for g in ['borehole', 'surface']:
        add(f'decline onset {g}', 'hinge t0 [boot 5-50-95%]; 2021H2 vs prev 2 yr z',
            {v: f"{tim[v][g]['hinge_t0']} [{','.join(tim[v][g]['hinge_boot'])}]; z2021H2 {tim[v][g]['h2021H2']['z']:+.1f}" for v in VARIANTS})
    for q in Q:
        for g in ['surface', 'borehole']:
            add(f'step {q} {g} (group-median curve)', 'level change (%) [z]',
                {v: f"{tim[v][g][f'level_{q}']['value']:+.3f} [z {tim[v][g][f'level_{q}']['z']:.1f}]" for v in VARIANTS})
            add(f'step {q} {g} (per-channel median)', 'level change (%) [z]',
                {v: (f"{lev[f'{v}_{g}_{q}']['value']:+.3f} [z {lev[f'{v}_{g}_{q}']['z']:.1f}, n={lev[f'{v}_{g}_{q}']['n_chan']}]"
                     if lev[f'{v}_{g}_{q}'].get('z') is not None else 'n/a') for v in VARIANTS})
            pp = P[(P.group == g) & (P.quantity == f'step_{q}')].set_index('variant')
            add(f'step {q} {g} (parametric, placebo work)', 'pooled step (%) [z, n]',
                {v: f"{pp.loc[v, 'pooled_pct']:+.3f} [z {pp.loc[v, 'z_placebo']:.1f}, n={int(pp.loc[v, 'n_units'])}]" if v in pp.index else 'n/a' for v in VARIANTS})
    for g in ['surface', 'borehole']:
        pp = P[(P.group == g) & (P.quantity == 'annual')].set_index('variant')
        add(f'annual {g}', 'coherent pooled amp (%) [z; peak doy]',
            {v: f"{pp.loc[v, 'pooled_pct']:.3f} [z {pp.loc[v, 'z_placebo']:.1f}; doy {pp.loc[v, 'pooled_peak_doy']:.0f}]" for v in VARIANTS})
    T = pd.DataFrame(rows)
    T.to_csv(f'{OUT}/claims_table.csv', index=False)
    return T


def figures(V, curves, st, P, B):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    col = {'V0': 'k', 'V1': 'tab:blue', 'V2': 'tab:red', 'V3': 'tab:orange'}
    for i, g in enumerate(['borehole', 'surface']):
        a = ax[0, i]
        for v in VARIANTS:
            med = curves[v][g][0]
            med = med[med.index >= '2017-01-01']
            a.plot(med.index, med.rolling(3, center=True, min_periods=2).mean(), color=col[v], lw=1.4 if v != 'V0' else 2.2,
                   alpha=0.9, label=f'{v} (n={len(curves[v][g][2])})')
        for q in Q.values():
            a.axvline(q, color='0.5', ls='--', lw=0.8)
        a.axhline(0, color='0.7', lw=0.5)
        a.set_title(f'{g}: deseasoned group-median monthly dv/v (3-mo running mean)\nV0 original, V1 windows excluded, V2/V3 baseline segment only', fontsize=9)
        a.set_ylabel('dv/v (%) rel. 2019-2021H1')
        a.legend(fontsize=8)
    a = ax[1, 0]
    s = st.dropna(how='all')
    x = np.arange(len(s))
    for j, v in enumerate(VARIANTS):
        a.plot(x + (j - 1.5) * 0.18, s[v], 'o', color=col[v], label=v)
    a.set_xticks(x)
    a.set_xticklabels([i.split('.')[1] for i in s.index], rotation=60, fontsize=8)
    a.axhline(0, color='0.7', lw=0.5)
    a.set_ylabel('D 2021->2025 (%), median over components')
    a.set_title('per-station decline (deseasoned yearly means; V2/V3 only within one segment)', fontsize=9)
    a.legend(fontsize=8)
    a = ax[1, 1]
    labs = []
    k = 0
    for g in ['surface', 'borehole']:
        for qn in ['step_pet', 'step_fer', 'step_m7', 'annual']:
            pp = P[(P.group == g) & (P.quantity == qn)].set_index('variant')
            for j, v in enumerate(VARIANTS):
                if v in pp.index:
                    a.plot(k + (j - 1.5) * 0.15, pp.loc[v, 'z_placebo'], 'o', color=col[v], label=v if k == 0 else None)
            labs.append(f'{g[:4]} {qn.replace("step_", "")}')
            k += 1
    a.axhline(0, color='0.7', lw=0.5)
    for yy in (-2, 2):
        a.axhline(yy, color='0.7', ls=':', lw=0.8)
    a.set_xticks(range(k))
    a.set_xticklabels(labs, rotation=45, fontsize=8)
    a.set_ylabel('z vs placebo (parametric channel fits)')
    a.set_title('quake steps and annual cycle vs placebo, by variant', fontsize=9)
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_instrument_check.png', dpi=110)
    plt.close(fig)
    # noise figure
    D = pd.read_csv(f'{OUT}/noise_10day_variants.csv.gz', parse_dates=['t'])
    fig, ax = plt.subplots(4, 2, figsize=(15, 12), sharex=True)
    for i, sta in enumerate(['BRIC', 'KMPB', 'B045', 'HAYF']):
        for j, band in enumerate(['1-4Hz', '2-8Hz']):
            a = ax[i, j]
            for (v, cmb), c in [(('orig', 'ALL'), '0.6'), (('masked', 'ALL'), 'tab:blue'), (('masked', 'ZZ'), 'tab:red')]:
                d = D[(D.sta == sta) & (D.band == band) & (D.variant == v) & (D.combo == cmb)]
                a.plot(d.t, d.dvv_pct, '-', color=c, lw=0.8 if v == 'orig' else 1.0, label=f'{cmb} {v}')
            for q in Q.values():
                a.axvline(q, color='0.5', ls='--', lw=0.8)
            for a0, a1 in NOISE_MASK[sta]:
                a.axvspan(pd.Timestamp(a0), pd.Timestamp(a1), color='0.85', zorder=0)
            for b_ in NOISE_BREAK[sta]:
                a.axvline(pd.Timestamp(b_), color='tab:green', lw=1.2)
            a.set_title(f'{sta} {band}: noise dv/v (10-d), all pairs vs vertical-only autocorrelation (ZZ)', fontsize=9)
            a.set_ylabel('dv/v (%)')
            a.set_ylim(-1.2, 0.6)
            if i == 0 and j == 0:
                a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_dvv_instrument_check_noise.png', dpi=100)
    plt.close(fig)


def stage_analyze(args):
    B = get_breaks()
    V = load_variant_bins()
    dec, st = decline_yearly(V)
    print(pd.DataFrame({f'{v}_{g}': (r['value'], r.get('z'), r['n_chan']) for (v, g), r in dec.items()}).T.to_string(), flush=True)
    dmod = decline_model(V, B)
    print(pd.DataFrame({k: (r.get('value'), r.get('z'), r['n_chan']) for k, r in dmod.items()}).T.to_string(), flush=True)
    curves = timing_all(V)
    lev = level_steps_channel(V)
    P = param_pool()
    print(P[['variant', 'group', 'quantity', 'n_units', 'pooled_pct', 'z_placebo']].to_string(), flush=True)
    metadata_steps(V)
    channel_qc()
    centroid_b933()
    if os.path.exists(f'{OUT}/noise_longterm_fits.csv'):
        noise_summary()
    T = claims_table(dec, dmod, SUMMARY['timing'], lev, P, st)
    print(T.to_string(), flush=True)
    figures(V, curves, st, P, B)
    json.dump(SUMMARY, open(f'{OUT}/summary.json', 'w'), indent=1,
              default=lambda o: o.item() if hasattr(o, 'item') else str(o))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['monthly', 'param', 'noise', 'noisefit', 'analyze'])
    ap.add_argument('--nboot', type=int, default=30)
    a = ap.parse_args()
    if a.stage == 'analyze':
        stage_analyze(a)
    else:
        {'monthly': stage_monthly, 'param': stage_param, 'noise': stage_noise, 'noisefit': stage_noisefit}[a.stage](a)
