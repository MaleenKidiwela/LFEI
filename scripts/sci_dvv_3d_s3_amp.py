#!/usr/bin/env python3
"""Stage 3, step 1 (revised after the stage-3 plan review): per-series component amplitudes on a COMMON model,
with a JOINT cross-series covariance.  See data/sci/dvv_3d/stage3/PLAN.md.

Every attribution series (data/sci/dvv_attr; SC10 0.1-0.3 ... 3-5 Hz, LFE coda channels, repeater stations, LP network
bands; the attribution's loaders/design code are imported unchanged from sci_dvv_attr_fit / sci_dvv_attr_lib; the
LFE coda is taken from the UNREGULARISED monthly bins, data/sci/decline/monthly_bins.csv - PRIMARY; the regularised
pool_clean curves only as a variant, LFE_INPUT=pool) is
fitted with the SAME component set (no per-series selection baked in):
   base (const + instrument offsets) + big-3 step + transient (exp(-t/tau_q), ONE network tau per quake, sign-paired)
   + transient PGV dose of other local quakes (nuisance) + trend + ramp21 (2021-07)
   + ENV = representative hydrology regressor (hyd_exp_rean, network tau_h; POWER/MERRA-2 precipitation, available
     at every site) + annual sin/cos (1 and 2 cycles/yr; absorbs thermal / other seasonal drivers)
   + microseism-power nuisance where the band has one.
Network tau_q (per quake) and tau_h are chosen by the summed weighted SSE over all series (coordinate descent on the
attribution grids).  Variant runs: tau_q x 0.5 and x 2.
Amplitudes a = M beta:  drop0_q = step_q + trans_q, level_q = step_q (q = Petrolia, Ferndale, M7), ramp21, trend,
ENV_cos / ENV_sin = annual cos / sin part of the daily ENV component over the series span; the non-annual ENV variance
fraction (multi-year drought / wet signal) is reported per series.
Covariance: JOINT moving-block bootstrap with COMMON calendar blocks (120 d) for all time-series datasets, so the
regional common mode and multiple LFE channels at one station are carried into the cross-series and cross-component
covariance; repeaters (pairwise data) use their own cluster bootstrap (no cross-covariance).  The bootstrap covariance
(rank-deficient) is shrunk toward its per-series block diagonal (lambda = 0.3; variants 0.7, 1.0) by the inversion.
Placebo scaling: the whole model is refitted on pre-2021-06 data with all three quake dates and the ramp hinge shifted
back by whole years (5..12); per amplitude type the network z-RMS (placebo / bootstrap SD) gives one factor >= 1 that
scales that amplitude's whole covariance block (rows and columns, all series).
Stages:  tau | fit [tau_scale] | boot | plac
Outputs: data/sci/dvv_3d/stage3/{tau_net.json, amplitudes*.csv, placebo_scale.json};
         /data/wsd02/maleen_data/dvv_3d/stage3/{fit*.npz, boot*.npz}.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

ROOT = '/home/seismic/LFE'
sys.path.insert(0, f'{ROOT}/scripts')
OUT = f'{ROOT}/data/sci/dvv_3d/stage3'
BIG = '/data/wsd02/maleen_data/dvv_3d/stage3'
NPROC = int(os.environ.get('NPROC', '6'))
AMPS = ['ENV_cos', 'ENV_sin', 'drop0_Petrolia', 'level_Petrolia', 'drop0_Ferndale', 'level_Ferndale', 'drop0_M7',
        'level_M7', 'ramp21', 'trend']
NB = 200
BLOCK = 120
TAUH = (30, 60, 120, 240)
_SER = {}


def load_lfe_monthly():
    """PRIMARY LFE-coda input: the UNREGULARISED monthly bins (data/sci/decline/monthly_bins.csv; every month a free
    level) - the attribution's pool_clean curves are regularised except at the quake dates (quake steps ~8-50x too
    significant vs placebo dates).  Month value dated at mid-month with a 30-d centred window; instrument windows
    and straddled breaks masked and free offsets at the breaks (as pool_clean V2 / attribution EXCL + BREAKS);
    weights 1/se^2 (normalised)."""
    import sci_dvv_attr_fit as AF
    import sci_dvv_attr_lib as L
    m = pd.read_csv(f'{ROOT}/data/sci/decline/monthly_bins.csv', parse_dates=['month'])
    m = m[np.isfinite(m.dvv_pct) & np.isfinite(m.se_pct) & (m.se_pct > 0)]
    out = []
    for ch, g in m.groupby('chan'):
        sta = ch.rsplit('.', 2)[0]
        t = (g.month + pd.Timedelta(days=14)).values
        sm = ('centre', 30)
        brk = L.BREAKS.get(sta, [])
        bad = AF.mask_windows(sta, t, sm, breaks=brk)
        g = g[~bad]
        t = (g.month + pd.Timedelta(days=14)).values
        keep = (pd.DatetimeIndex(t) >= L.DAYS[0]) & (pd.DatetimeIndex(t) <= L.DAYS[-1])
        g, t = g[keep], t[keep]
        if len(g) < 24:
            continue
        w = 1.0 / g.se_pct.values ** 2
        out.append(L.Series('LFE', sta, ch, t, g.dvv_pct.values, sm, w=w / np.median(w), breaks=list(brk)))
    return out


def series_list(only=None):
    import sci_dvv_attr_fit as AF
    if not _SER:
        lfe_pool = os.environ.get('LFE_INPUT', 'monthly') == 'pool'
        ser = AF.all_series(['SC', 'LP', 'REP'] + (['LFE'] if lfe_pool else []))
        if not lfe_pool:
            ser += load_lfe_monthly()
        for S in ser:
            if S.dataset == 'SC10' and S.chan == '0.1-1Hz':
                continue
            _SER['|'.join((S.dataset, S.sta, S.chan))] = S
    keys = sorted(_SER)
    if only:
        keys = [k for k in keys if k.split('|')[1] in only or k.split('|')[0] == 'LP10']
    return keys


def ramp_supported(S, hinge_day):
    ii = S.idx
    return ((ii < hinge_day) & (ii >= hinge_day - 365)).sum() >= 3 and ((ii >= hinge_day + 365)).sum() >= 3 and \
        ii.min() <= hinge_day - 365 and ii.max() >= hinge_day + 365


def blocks_for(F, taus, tau_h, shift=0.0):
    """common component set for one series.  shift (yr): placebo - all quake dates and the hinge moved back
    (the dose block of the other local quakes is kept unshifted).  ramp21 only where the hinge has >= 1 yr of data
    on both sides (real and placebo fits alike)."""
    import sci_dvv_attr_lib as L
    cur = F.core_blocks(taus['M7'], tau_by=taus)
    hinge = int(np.argmax(F.longc['ramp21'][:, 0] > 0)) - int(round(shift * 365.25))
    if shift:
        dose = [b for b in cur if b[0] == 'dose']
        cur = shifted_core(F, taus, shift) + dose
    if not ramp_supported(F.S, hinge):
        cur = [b for b in cur if b[0] != 'ramp21']
    env = []
    if 'hyd_exp_rean' in F.cands:
        labs = [c[0] for c in F.cands['hyd_exp_rean']]
        lab = f'tau_h={tau_h}d (POWER precip)'
        if lab in labs:
            blk, _ = F.env_block('hyd_exp_rean', labs.index(lab))
            env.append(blk)
    env.append(('annual', ['ann_s1', 'ann_c1', 'ann_s2', 'ann_c2'], L.annual_cols()))
    if 'micro' in F.cands:
        blk, _ = F.env_block('micro', 0)
        env.append(blk)
    return cur + env


def shifted_core(F, taus, s):
    import sci_dvv_attr_lib as L
    i = np.arange(L.ND)
    sh = int(round(s * 365.25))
    B = [('base', F.base[1], F.base[0])]
    cols, names = [], []
    for q in L.BIG3:
        d = L.big3_day(q) - sh
        if not F.supported('step', d):
            continue
        cols.append((i >= d).astype(float))
        names.append(f'step_{q}')
        if F.supported('trans', d, taus[q]):
            cols.append(np.where(i >= d, np.exp(-(i - d) / taus[q]), 0.0))
            names.append(f'trans_{q}')
    if cols:
        B.append(('big3', names, np.column_stack(cols)))
    lc = F.longc
    B.append(('trend', ['trend'], lc['trend']))
    r = lc['ramp21'][:, 0]
    hinge = np.argmax(r > 0) - sh
    B.append(('ramp21', ['ramp21'], np.clip((i - hinge) / 365.25, 0, None)[:, None]))
    return B


def mapping(F, names, groups, cur, S):
    import sci_dvv_attr_fit as AF
    import sci_dvv_attr_lib as L
    M = np.zeros((len(AMPS), len(names)))
    avail = np.zeros(len(AMPS), bool)
    for q in L.BIG3:
        if f'step_{q}' in names:
            i = names.index(f'step_{q}')
            M[AMPS.index(f'level_{q}'), i] = 1.0
            M[AMPS.index(f'drop0_{q}'), i] = 1.0
            avail[AMPS.index(f'level_{q}')] = avail[AMPS.index(f'drop0_{q}')] = True
            if f'trans_{q}' in names:
                M[AMPS.index(f'drop0_{q}'), names.index(f'trans_{q}')] = 1.0
    for g in ('ramp21', 'trend'):
        if g in names:
            M[AMPS.index(g), names.index(g)] = 1.0
            avail[AMPS.index(g)] = True
    env = [i for i, g in enumerate(groups) if AF.ENVLIKE(g) and g != 'micro']
    nonann = np.nan
    if env:
        Ad = F.daily_design(cur, names)
        lo, hi = int(S.idx.min()), int(S.idx.max())
        ii = np.arange(lo, hi + 1)
        w = 2 * np.pi * (L.DAYS[ii].dayofyear.values - 1) / 365.25
        X = np.column_stack([np.ones(len(ii)), np.cos(w), np.sin(w)])
        P = np.linalg.pinv(X)
        proj = P @ Ad[ii][:, env]
        M[AMPS.index('ENV_cos'), env] = proj[1]
        M[AMPS.index('ENV_sin'), env] = proj[2]
        avail[AMPS.index('ENV_cos')] = avail[AMPS.index('ENV_sin')] = True
    return M, avail, env


def fit_one(args):
    key, taus, tau_h, shift, want = args
    import sci_dvv_attr_fit as AF
    import sci_dvv_attr_lib as L
    S = _SER[key]
    if shift:
        # placebo: truncate at (series end - shift) so that the post-event support matches the real fit
        cut = int(S.idx.max() - round(shift * 365.25))
        m = S.idx <= cut
        if S.pair_i is not None:
            m &= S.pair_i <= cut
        if m.sum() < 30:
            return key, None
        S = S.subset(m)
    F = AF.Fitter(S)
    F.Lc = 1.0
    cur = blocks_for(F, taus, tau_h, shift)
    ff = F.fit(cur)
    if want == 'sse':
        return key, ff['sse']
    names, groups = ff['names'], ff['groups']
    M, avail, env = mapping(F, names, groups, cur, S)
    beta = ff['beta']
    out = dict(a=M @ beta, avail=avail)
    if want == 'amp':
        if shift:                         # placebo SD from the placebo fit itself (50-draw bootstrap)
            Bt = F.bootstrap(ff, nb=50, seed=int(shift))
            out['sd'] = (Bt @ M.T).std(0)
        return key, out
    ok = ff['ok']
    A = ff['A'][ok]
    fit = A @ beta
    # non-annual ENV variance fraction
    if env:
        Ad = F.daily_design(cur, names)
        lo, hi = int(S.idx.min()), int(S.idx.max())
        e = Ad[lo:hi + 1][:, env] @ beta[env]
        w = 2 * np.pi * (L.DAYS[lo:hi + 1].dayofyear.values - 1) / 365.25
        X = np.column_stack([np.ones(len(e)), np.cos(w), np.sin(w)])
        r = e - X @ np.linalg.lstsq(X, e, rcond=None)[0]
        out['env_nonannual_frac'] = float(np.var(r) / max(np.var(e), 1e-30))
    # time shapes of the stage-3 amplitudes at the samples (y = level*step + (drop0 - level)*trans + ...)
    sh = np.zeros((ok.sum(), len(AMPS)))
    for q in L.BIG3:
        if f'step_{q}' in names:
            st = A[:, names.index(f'step_{q}')]
            tr = A[:, names.index(f'trans_{q}')] if f'trans_{q}' in names else 0 * st
            sh[:, AMPS.index(f'level_{q}')] = st - tr
            sh[:, AMPS.index(f'drop0_{q}')] = tr
    for g in ('ramp21', 'trend'):
        if g in names:
            sh[:, AMPS.index(g)] = A[:, names.index(g)]
    d = S.idx[ok]
    wv = 2 * np.pi * (L.DAYS[d].dayofyear.values - 1) / 365.25
    if S.pair_i is not None:
        w0 = 2 * np.pi * (L.DAYS[S.pair_i[ok]].dayofyear.values - 1) / 365.25
        sh[:, 0], sh[:, 1] = np.cos(wv) - np.cos(w0), np.sin(wv) - np.sin(w0)
    else:
        sh[:, 0], sh[:, 1] = np.cos(wv), np.sin(wv)
    out.update(A=A.astype(np.float64), y=S.y[ok], w=(np.ones(ok.sum()) if S.w is None else S.w[ok]), beta=beta, fit=fit,
               M=M, day=d.astype(np.int32), pairs=np.array(ff['pairs'] or [], int).reshape(-1, 2),
               cluster=(np.asarray(S.cluster)[ok] if S.cluster is not None else np.zeros(0)), shapes=sh,
               pair_day=(S.pair_i[ok].astype(np.int32) if S.pair_i is not None else np.zeros(0, np.int32)))
    return key, out


# ------------------------------------------------------------------------------------------ stages
def stage_tau(only=None):
    import sci_dvv_attr_fit as AF
    keys = series_list(only)
    taus = {'Petrolia': 40, 'Ferndale': 40, 'M7': 40}
    tau_h = 120
    hist = []

    def total(tq, th):
        with Pool(NPROC) as pool:
            r = pool.map(fit_one, [(k, tq, th, 0.0, 'sse') for k in keys], chunksize=4)
        # normalise each series by its own SSE scale so long daily series do not dominate
        return r
    base = dict(total(taus, tau_h))
    scale = {k: max(v, 1e-12) for k, v in base.items()}

    def score(tq, th):
        r = dict(total(tq, th))
        return float(sum(r[k] / scale[k] for k in r))
    best = score(taus, tau_h)
    for it in range(2):
        for q in ('Petrolia', 'Ferndale', 'M7'):
            for t in AF.TAUQ:
                tq = dict(taus, **{q: t})
                s = score(tq, tau_h)
                hist.append(dict(it=it, param=q, value=t, score=s))
                if s < best - 1e-9:
                    best, taus = s, tq
        for th in TAUH:
            s = score(taus, th)
            hist.append(dict(it=it, param='tau_h', value=th, score=s))
            if s < best - 1e-9:
                best, tau_h = s, th
    json.dump(dict(tau_q=taus, tau_h=tau_h, score=best, history=hist), open(f'{OUT}/tau_net{"_smoke" if only else ""}.json', 'w'), indent=1)
    print('network tau', taus, 'tau_h', tau_h, flush=True)


def load_tau(only=None):
    f = f'{OUT}/tau_net{"_smoke" if only else ""}.json'
    J = json.load(open(f))
    return J['tau_q'], J['tau_h']


def stage_fit(tscale=1.0, only=None):
    keys = series_list(only)
    taus, tau_h = load_tau(only)
    taus = {q: max(2.0, t * tscale) for q, t in taus.items()}
    with Pool(NPROC) as pool:
        res = pool.map(fit_one, [(k, taus, tau_h, 0.0, 'full') for k in keys], chunksize=2)
    tag = ('' if tscale == 1 else f'_tau{tscale:g}') + ('_smoke' if only else '')
    rows, arr = [], {}
    for key, o in res:
        if o is None:
            continue
        for j, nm in enumerate(AMPS):
            if o['avail'][j]:
                rows.append(dict(series=key, dataset=key.split('|')[0], sta=key.split('|')[1], chan=key.split('|')[2],
                                 amp=nm, est=o['a'][j], env_nonannual_frac=o.get('env_nonannual_frac', np.nan)))
        for f in ('A', 'y', 'w', 'beta', 'fit', 'M', 'day', 'pairs', 'cluster', 'shapes', 'pair_day', 'avail'):
            arr[f'{key}||{f}'] = np.asarray(o[f])
    pd.DataFrame(rows).to_csv(f'{OUT}/amplitudes{tag}.csv', index=False, float_format='%.6g')
    np.savez_compressed(f'{BIG}/fit{tag}.npz', **arr)
    print('fit', tag, len(set(r['series'] for r in rows)), 'series', flush=True)


def _boot_one(args):
    key, D, draws, seed = args
    import sci_dvv_attr_lib as L
    A, y, w, beta, M, day = D['A'], D['y'], D['w'], D['beta'], D['M'], D['day']
    pairs = [tuple(p) for p in D['pairs']] or None
    fit = A @ beta
    r = y - fit
    out = np.zeros((len(draws), M.shape[0]))
    if len(D['pair_day']):                         # repeaters: cluster bootstrap (independent)
        rng = np.random.default_rng(seed)
        cl = D['cluster']
        uc = np.unique(cl)
        grp = {c: np.where(cl == c)[0] for c in uc}
        for b in range(len(draws)):
            ii = np.concatenate([grp[c] for c in rng.choice(uc, len(uc))])
            bb, _ = L.huber_lsq(A[ii], (fit + r)[ii], w[ii], nit=4, pairs=pairs)
            out[b] = M @ bb
        return key, out
    # COMMON calendar-block draws (perm[k] = source block for target block k, shared by all series).  Per series a
    # target sample takes the weight-standardised residual of the sample nearest to the source day if that sample lies
    # within one sample spacing of it; otherwise (source block outside the series' span or in a gap) the source block
    # is redrawn from the series' OWN occupied blocks with a generator seeded by (draw, block) - never clipped to an edge.
    rs_std = r * np.sqrt(w)
    order = np.argsort(day)
    ds = day[order]
    spacing = max(1.0, float(np.median(np.diff(np.unique(ds))))) if len(ds) > 2 else 1.0
    occ = np.unique(ds // BLOCK)
    kblk = day // BLOCK
    off = day % BLOCK
    for b, perm in enumerate(draws):
        src = perm[np.clip(kblk, 0, len(perm) - 1)] * BLOCK + off
        j = np.clip(np.searchsorted(ds, src), 0, len(ds) - 1)
        j0 = np.clip(j - 1, 0, len(ds) - 1)
        jj = np.where(np.abs(ds[j0] - src) < np.abs(ds[j] - src), j0, j)
        bad = np.abs(ds[jj] - src) > spacing
        if bad.any():
            for k in np.unique(kblk[bad]):
                g = np.random.default_rng([b, int(k), 7]).choice(occ)
                m = bad & (kblk == k)
                src2 = g * BLOCK + off[m]
                j2 = np.clip(np.searchsorted(ds, src2), 0, len(ds) - 1)
                j20 = np.clip(j2 - 1, 0, len(ds) - 1)
                jj[m] = np.where(np.abs(ds[j20] - src2) < np.abs(ds[j2] - src2), j20, j2)
        rsb = rs_std[order[jj]] / np.sqrt(w)
        bb, _ = L.huber_lsq(A, fit + rsb, w, nit=4, pairs=pairs)
        out[b] = M @ bb
    return key, out


def stage_boot(only=None, tag=''):
    import sci_dvv_attr_lib as L
    tag = tag + ('_smoke' if only else '')
    Z = np.load(f'{BIG}/fit{tag}.npz', allow_pickle=False)
    keys = sorted(set(k.split('||')[0] for k in Z.files))
    nblk = L.ND // BLOCK + 1
    rng = np.random.default_rng(20261125)
    draws = [rng.integers(0, nblk, nblk) for _ in range(NB)]          # COMMON calendar-block draws for all series
    jobs = []
    for i, k in enumerate(keys):
        D = {f: Z[f'{k}||{f}'] for f in ('A', 'y', 'w', 'beta', 'M', 'day', 'pairs', 'cluster', 'pair_day')}
        jobs.append((k, D, draws, 1000 + i))
    with Pool(NPROC) as pool:
        res = pool.map(_boot_one, jobs, chunksize=1)
    np.savez_compressed(f'{BIG}/boot{tag}.npz', **{k: v for k, v in res})
    print('boot', tag, len(res), 'series x', NB, 'draws', flush=True)


LOCAL_M6 = ['2010-01-10', '2014-03-10', '2016-12-08', '2021-12-20', '2022-12-20', '2024-12-05']


def stage_plac(only=None):
    """whole-model placebo: all quake dates and the hinge shifted back by 6..12 yr (shifts whose shifted quake dates
    fall within +-60 d of a local M >= 6 event are skipped), data truncated at (end - shift); z = placebo amplitude /
    its own bootstrap SD; factor per dataset x amplitude type = robust z-RMS (raw value reported; covariance scaled
    by max(1, factor))."""
    import sci_dvv_attr_lib as L
    tag = '_smoke' if only else ''
    keys = series_list(only)
    taus, tau_h = load_tau(only)
    big = [pd.Timestamp(v) for v in L.BIG3.values()]
    m6 = [pd.Timestamp(x) for x in LOCAL_M6]
    shifts = []
    for sft in range(6, 13):
        sh = [b - pd.Timedelta(days=round(sft * 365.25)) for b in big]
        if any(abs((a - e).days) <= 60 for a in sh for e in m6):
            continue
        shifts.append(sft)
    rows = []
    for sft in shifts:
        with Pool(NPROC) as pool:
            res = pool.map(fit_one, [(k, taus, tau_h, float(sft), 'amp') for k in keys], chunksize=4)
        for key, o in res:
            if o is None:
                continue
            for j, nm in enumerate(AMPS[2:], start=2):
                if o['avail'][j] and o['sd'][j] > 0:
                    rows.append(dict(shift=sft, series=key, dataset=key.split('|')[0], amp=nm, est=o['a'][j], sd=o['sd'][j],
                                     z=o['a'][j] / o['sd'][j]))
    T = pd.DataFrame(rows)
    T.to_csv(f'{OUT}/placebo_amps{tag}.csv', index=False, float_format='%.5g')
    sc = {}
    for (ds, nm), g in T.groupby(['dataset', 'amp']):
        z = g.z.values
        z = z[np.isfinite(z)]
        raw = float(np.sqrt(np.median(z ** 2) / 0.455)) if len(z) >= 10 else float('nan')
        sc[f'{ds}|{nm}'] = dict(raw=raw, applied=float(max(1.0, raw)) if np.isfinite(raw) else 1.0, n=int(len(z)))
    json.dump(dict(scale=sc, shifts=shifts, note='robust z-RMS (median z^2 / 0.455) of whole-model placebo amplitudes '
                   'divided by their own bootstrap SD; per dataset x amplitude type; ENV not placebo-scaled'),
              open(f'{OUT}/placebo_scale{tag}.json', 'w'), indent=1)
    print('placebo shifts', shifts, flush=True)
    print(json.dumps(sc, indent=0), flush=True)


if __name__ == '__main__':
    st = sys.argv[1]
    only = None
    args = sys.argv[2:]
    if args and args[0].startswith('only='):
        only = args.pop(0)[5:].split(',')
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(BIG, exist_ok=True)
    series_list(only)
    if st == 'tau':
        stage_tau(only)
    elif st == 'fit':
        stage_fit(float(args[0]) if args else 1.0, only)
    elif st == 'boot':
        stage_boot(only, args[0] if args else '')
    elif st == 'plac':
        stage_plac(only)
