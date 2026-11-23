#!/usr/bin/env python
"""Relative LFE moments, completeness, moment budget, healing (slide-hold-slide), slip rate.

Stages:
  extract  : per family, per long-running channel: matched-filter (MF) projection amplitude of the
             S window onto the channel's own stack (per channel-year lag corrected), noise-window
             projections (for per-event uncertainty), S and noise RMS (QC / RMS proxy).
             -> data/sci/moments/raw/fam{F}_chan.parquet, fam{F}_chaninfo.csv
  combine  : per family, joint channel-gain / event-amplitude solution (weighted LS), channel-year
             gain-jump screening, RMS-proxy comparison -> data/sci/moments/events_all.parquet
  analyse  : completeness, cumulative moment / catch-up, healing, distributions, slip -> CSV/JSON/figs
  extras   : healing injection test (log A), count / all-event budgets, 7-day block bootstrap (M7)
  healing_exc : truncation-robust exceedance healing test (+ healing figure)
  rate     : moment-rate ratio 30/60 d after restart vs random-date null
  export   : per-event amplitude table (csv.gz)
Run order: extract, combine, analyse, extras, healing_exc, rate, export  (all write to data/sci/moments/).
Assumption: moment proportional to MF amplitude requires a roughly constant LFE source duration
(Bostock et al. 2015; Thomas et al. 2016) and a fixed path per family/channel.

Why MF and not raw RMS: single-LFE S-window RMS SNR is ~1 (NOTES 2026-11-21), so RMS is dominated by
noise; MF projection is unbiased by (uncorrelated) noise. The RMS proxy is computed too for comparison.
"""
import sys, os, json, argparse, time
import numpy as np, pandas as pd, h5py

ROOT = '/home/seismic/LFE'
OUT = f'{ROOT}/data/sci/moments'
SR = 50.0
PRE = 15.0
LONG = ['PB.B045', 'PB.B046', 'PB.B047', 'PB.B049', 'PB.B932', 'PB.B933', 'PB.B934', 'PB.B935',
        'NC.KMR', 'NC.KMPB', 'NC.KBN', 'NC.KFP', 'NC.KIP', 'NC.KCR', 'NC.KSM']
PAD = 20          # samples of padding (+-0.4 s) for lag search
NS = 175          # S window S-0.5 .. S+3 s
MAXLAG = 15       # +-0.3 s
NOISE_STARTS = [0, 162, 325]   # offsets within the -14..-4 s noise segment (three 3.5-s windows)
MS = {'Petrolia': '2021-12-20T20:10', 'Ferndale': '2022-12-20T10:34', 'M7': '2024-12-05T18:44'}


def s_times(F):
    p = pd.read_csv(f'{ROOT}/data/locations/fam{F}_3d_picks.csv').set_index('sta')
    j = json.load(open(f'{ROOT}/data/locations/fam{F}_3d.json'))
    o = j['origin_rel_catalog_s']
    both = p.dropna(subset=['S_final', 'P'])
    # empirical ts/tp ratio from stations with both picks
    ratio = np.median((both.S_final - o) / (both.P - o)) if len(both) >= 3 else 1.78
    out = {}
    for s in LONG:
        if s in p.index and pd.notna(p.loc[s, 'S_final']):
            out[s] = (float(p.loc[s, 'S_final']), 'S')
        elif s in p.index and pd.notna(p.loc[s, 'P']):
            out[s] = (float(o + ratio * (p.loc[s, 'P'] - o)), 'P*ratio')
    return out, ratio


def extract(F):
    os.makedirs(f'{OUT}/raw', exist_ok=True)
    fn = f'{OUT}/raw/fam{F}_chan.parquet'
    if os.path.exists(fn):
        print('exists', fn); return
    st, ratio = s_times(F)
    f = h5py.File(f'{ROOT}/data/waveforms/fam{F}.h5', 'r')
    tutc = pd.to_datetime(pd.Series(f['events/time_utc'][:].astype(str)))
    year = tutc.dt.year.values
    rows, info = [], []
    for ch in sorted(f['chan'].keys()):
        sta = '.'.join(ch.split('.')[:2])
        if sta not in st:
            continue
        S, src = st[sta]
        i0 = int(round((PRE + S - 0.5) * SR)) - PAD
        i1 = i0 + NS + 2 * PAD
        if i0 < 560 or i1 > 3000:
            continue
        d = f[f'chan/{ch}/data']; ev = f[f'chan/{ch}/ev'][:]
        n = len(ev)
        seg = np.empty((n, i1 - i0), np.float32); noi = np.empty((n, 500), np.float32)
        for k in range(0, n, 4096):
            x = d[k:k + 4096]
            seg[k:k + len(x)] = x[:, i0:i1]; noi[k:k + len(x)] = x[:, 50:550]
        good = (np.abs(noi).max(1) > 0) & (np.abs(seg).max(1) > 0) & np.isfinite(seg).all(1) & np.isfinite(noi).all(1)
        rmsN = np.sqrt((noi ** 2).mean(1)); rmsS = np.sqrt((seg[:, PAD:PAD + NS] ** 2).mean(1))
        # robust stack: exclude events with anomalous noise
        med = np.median(rmsN[good]) if good.any() else 0
        medS = np.median(rmsS[good]) if good.any() else 0
        use = good & (rmsN < 3 * med) & (rmsS < 3 * medS)   # exclude glitches / earthquakes from the stack
        if use.sum() < 50:
            continue
        tmpl = seg[use].mean(0)
        core = tmpl[PAD:PAD + NS]
        nstack = noi[use].mean(0)
        tsnr = np.sqrt((core ** 2).mean()) / np.sqrt((nstack ** 2).mean())
        # per channel-year lag (stack vs template), apply when well correlated
        evyear = year[ev]
        shift = np.zeros(n, int)
        for y in np.unique(evyear):
            m = use & (evyear == y)
            if m.sum() < 30:
                continue
            ys = seg[m].mean(0)
            ccs = []
            for L in range(-MAXLAG, MAXLAG + 1):
                w = ys[PAD + L:PAD + L + NS]
                ccs.append(np.dot(w, core) / np.sqrt(np.dot(w, w) * np.dot(core, core) + 1e-30))
            ccs = np.array(ccs); L = int(np.argmax(ccs)) - MAXLAG
            if ccs.max() >= 0.5 and abs(L) >= 2:
                shift[evyear == y] = L
            info.append(dict(chan=ch, year=int(y), n=int(m.sum()), lag_samp=L, cc_year=float(ccs.max()),
                             cc_zero=float(ccs[MAXLAG])))
        idx = PAD + shift[:, None] + np.arange(NS)[None, :]
        win = np.take_along_axis(seg, idx, 1)
        tt = float(np.dot(core, core))
        a = win @ core / tt
        pn = np.stack([noi[:, s0:s0 + NS] @ core / tt for s0 in NOISE_STARTS], 1)
        df = pd.DataFrame(dict(ev=ev.astype(np.int32), a=a.astype(np.float32),
                               pn1=pn[:, 0].astype(np.float32), pn2=pn[:, 1].astype(np.float32),
                               pn3=pn[:, 2].astype(np.float32), rmsS=rmsS.astype(np.float32),
                               rmsN=rmsN.astype(np.float32), good=good, shift=shift.astype(np.int8)))
        df['chan'] = ch
        rows.append(df)
        info.append(dict(chan=ch, year=-1, n=int(use.sum()), lag_samp=0, cc_year=np.nan, cc_zero=np.nan,
                         tmpl_snr=float(tsnr), s_src=src, S=S, ts_tp_ratio=ratio, tmpl_rms=float(np.sqrt(tt / NS))))
        print(F, ch, n, f'tsnr={tsnr:.1f}', src, flush=True)
    pd.concat(rows).to_parquet(fn)
    pd.DataFrame(info).to_csv(f'{OUT}/raw/fam{F}_chaninfo.csv', index=False)
    pd.DataFrame(dict(time_utc=tutc)).to_parquet(f'{OUT}/raw/fam{F}_events.parquet')


def solve(Y, S2, W0, g):
    """Event amplitudes for fixed channel gains g. Y: E x C projections, S2: noise variance, W0: 0/1 mask.
    A_e = Huber-weighted inverse-variance mean of a_ec/g_c. Returns A, sigA, w (inverse variances), h (Huber factors)."""
    y = Y / g; s2 = S2 / g ** 2
    w = np.where(W0 > 0, 1.0 / s2, 0.0); y0 = np.where(W0 > 0, y, 0.0)
    A = (w * y0).sum(1) / np.maximum(w.sum(1), 1e-30)
    h = np.ones_like(w)
    for _ in range(3):
        r = np.abs(y0 - A[:, None]) / np.sqrt(s2)
        h = np.minimum(1.0, 3.0 / np.maximum(r, 1e-9))
        A = (w * h * y0).sum(1) / np.maximum((w * h).sum(1), 1e-30)
    sigA = 1.0 / np.sqrt(np.maximum(w.sum(1), 1e-30))
    A[w.sum(1) == 0] = np.nan; sigA[w.sum(1) == 0] = np.nan
    return A, sigA, w, h


def block_ratios(Y, S2, W0, g, w, h, blk, minn=20):
    """Leave-one-out relative gain of each channel-block: r = sum(a)/sum(g*Aloo) over events with
    |a - g Aloo| < 5 sigma (outlier-trimmed ratio of sums; not attenuated by noise in Aloo)."""
    y0 = np.where(W0 > 0, Y / g, 0.0); wh = w * h
    num = (wh * y0).sum(1)[:, None] - wh * y0; den = wh.sum(1)[:, None] - wh
    Aloo = np.where(den > 0, num / np.maximum(den, 1e-30), np.nan)
    sAloo = np.where(den > 0, 1 / np.sqrt(np.maximum(w.sum(1)[:, None] - w, 1e-30)), np.nan)
    out = []
    for j in range(Y.shape[1]):
        pres = (W0[:, j] > 0) & np.isfinite(Aloo[:, j])
        for b in np.unique(blk[pres]):
            m = pres & (blk == b)
            if m.sum() < minn:
                continue
            aa = Y[m, j]; al = Aloo[m, j] * g[j]
            sig = np.sqrt(S2[m, j] + (g[j] * sAloo[m, j]) ** 2)
            k = np.abs(aa - al) < 5 * sig
            if k.sum() < minn or al[k].sum() <= 0:
                out.append((j, b, int(m.sum()), np.nan, np.nan)); continue
            r = aa[k].sum() / al[k].sum()
            res = aa[k] - r * al[k]
            se = np.sqrt((res ** 2).sum()) / al[k].sum()
            out.append((j, b, int(m.sum()), r, se))
    return out


def combine_family(F, tsnr_min=2.0, jump=0.1, verbose=True):
    d = pd.read_parquet(f'{OUT}/raw/fam{F}_chan.parquet')
    ci = pd.read_csv(f'{OUT}/raw/fam{F}_chaninfo.csv')
    evt = pd.read_parquet(f'{OUT}/raw/fam{F}_events.parquet')
    tm = evt.time_utc.values
    # gain-screening blocks are half-years (e.g. 2016.0, 2016.5)
    year = evt.time_utc.dt.year.values + 0.5 * (evt.time_utc.dt.month.values > 6)
    tinfo = ci[ci.year == -1].set_index('chan')
    chans = [c for c in tinfo.index if tinfo.loc[c, 'tmpl_snr'] >= tsnr_min]
    d = d[d.chan.isin(chans) & d.good]
    # per-channel noise-projection scale (robust), sigma_ec = k_c * rmsN_ec
    z = np.concatenate([d[c] / d.rmsN for c in ['pn1', 'pn2', 'pn3']])
    zc = np.concatenate([d.chan.values] * 3)
    kk = pd.Series(np.abs(z)).groupby(zc).median() * 1.4826
    d['sig'] = d.rmsN * d.chan.map(kk)
    # QC: noise RMS within factor 4 of the channel-year median (gaps, glitches, earthquakes in noise window)
    d['year'] = year[d.ev.values]
    mN = d.groupby(['chan', 'year']).rmsN.transform('median')
    d = d[(d.rmsN < 4 * mN) & (d.rmsN > mN / 4)]
    E = len(tm); C = len(chans); cidx = {c: i for i, c in enumerate(chans)}
    Y = np.full((E, C), np.nan); S2 = np.full((E, C), np.nan); RS = np.full((E, C), np.nan); RN = np.full((E, C), np.nan)
    ii = d.ev.values; jj = d.chan.map(cidx).values
    Y[ii, jj] = d.a.values; S2[ii, jj] = d.sig.values ** 2; RS[ii, jj] = d.rmsS.values; RN[ii, jj] = d.rmsN.values
    W0 = np.isfinite(Y).astype(float)
    Y = np.nan_to_num(Y); S2 = np.where(W0 > 0, S2, 1.0)
    g = np.array([np.median(Y[W0[:, j] > 0, j]) for j in range(C)])
    g = np.where(g > 0, g, np.abs(g) + 1e-3)
    removed = []
    jall = []
    g = np.ones(C)   # templates are mean stacks, so a_ec ~ 1 for a typical event by construction
    for rnd in range(40):
        dead = (W0 > 0).sum(0) < 50
        W0[:, dead] = 0
        A, sigA, w, h = solve(Y, S2, W0, g)
        br = block_ratios(Y, S2, W0, g, w, h, year)
        # robust gain update: median of block ratios per channel
        upd = np.ones(C)
        for j in range(C):
            rr = [x[3] for x in br if x[0] == j and np.isfinite(x[3]) and x[3] > 0]
            if rr:
                upd[j] = np.median(rr)
        g = g * upd
        sc = np.nanmedian(solve(Y, S2, W0, g)[0])
        g = g * sc
        new = 0
        if rnd >= 2:   # screen only once gains have settled
            # remove at most ONE block (the worst significant one) per half-year per round, because a single
            # broken channel contaminates the leave-one-out reference of all others in that half-year
            worst = {}
            for j, b, n, r, se in br:
                r = r / upd[j] if np.isfinite(r) else r
                jall.append(dict(fam=F, chan=chans[j], year=float(b), n=n, rel_gain=r, rel_gain_se=se, round=rnd))
                bad = (not np.isfinite(r)) or r <= 0 or (abs(np.log10(r)) > jump and abs(r - 1) > 3 * se)
                if bad:
                    sev = 9.0 if ((not np.isfinite(r)) or r <= 0) else abs(np.log10(r))
                    if b not in worst or sev > worst[b][0]:
                        worst[b] = (sev, j, r, se)
            for b, (sev, j, r, se) in worst.items():
                W0[(year == b), j] = 0; new += 1
                removed.append((chans[j], float(b), r, se))
        if verbose:
            print(F, 'round', rnd, 'gain update max |log10|', round(float(np.max(np.abs(np.log10(upd)))), 3),
                  'removed blocks', new, flush=True)
        if rnd >= 2 and new == 0 and np.max(np.abs(np.log10(upd))) < 0.005:
            break
    jr = pd.DataFrame(jall)
    rem = set((x[0], x[1]) for x in removed)
    jr['removed'] = [(c, y) in rem for c, y in zip(jr.chan, jr.year)]
    jr = jr.sort_values('round').groupby(['chan', 'year']).first().reset_index()  # first screening per block
    dead = (W0 > 0).sum(0) < 50
    W0[:, dead] = 0
    # final
    A, sigA, w, h = solve(Y, S2, W0, g)
    wh = w * h
    nch = (W0 > 0).sum(1)
    # RMS proxy (task spec): median over channels of log10(rmsS / channel median rmsS); and noise level
    RSm = np.where(W0 > 0, RS, np.nan); RNm = np.where(W0 > 0, RN, np.nan)
    with np.errstate(all='ignore'):
        rel_rms = np.nanmedian(np.log10(RSm / np.nanmedian(RSm, 0)), 1)
        rel_noise = np.nanmedian(np.log10(RNm / np.nanmedian(RNm, 0)), 1)
        snr_rms = np.nanmedian(RSm / RNm, 1)
        ispb = np.array([c.startswith('PB.') for c in chans])
        def sub(mask):
            ww = np.where((W0 > 0) & mask[None, :], wh, 0.0); yy = np.where(W0 > 0, Y / g, 0.0)
            return np.where(ww.sum(1) > 0, (ww * yy).sum(1) / np.maximum(ww.sum(1), 1e-30), np.nan), \
                np.where(ww.sum(1) > 0, 1 / np.sqrt(np.maximum(np.where((W0 > 0) & mask[None, :], w, 0).sum(1), 1e-30)), np.nan)
        A_pb, s_pb = sub(ispb); A_nc, s_nc = sub(~ispb)
    f = h5py.File(f'{ROOT}/data/waveforms/fam{F}.h5', 'r')
    out = pd.DataFrame(dict(fam=F, ev=np.arange(E), time=evt.time_utc.values, A=A, sigA=sigA, nchan=nch,
                            A_pb=A_pb, sigA_pb=s_pb, A_nc=A_nc, sigA_nc=s_nc,
                            rel_rms=rel_rms, rel_noise=rel_noise, snr_rms=snr_rms,
                            cc_div_thr=f['events/cc_div_thr_v2'][:], meancc=f['events/meancc_v2'][:],
                            nchan_cat=f['events/nchan_v2'][:]))
    gains = pd.DataFrame(dict(fam=F, chan=chans, gain=g, k_noise=[kk[c] for c in chans],
                              tmpl_snr=[tinfo.loc[c, 'tmpl_snr'] for c in chans],
                              n_used=(W0 > 0).sum(0)))
    return out, jr, gains


def combine(fams):
    from multiprocessing import Pool
    with Pool(4) as p:
        res = p.map(combine_family, fams, chunksize=1)
    ev = pd.concat([r[0] for r in res]); jr = pd.concat([r[1] for r in res]); gn = pd.concat([r[2] for r in res])
    ev.to_parquet(f'{OUT}/events_all.parquet')
    jr.to_csv(f'{OUT}/channel_year_gain_check.csv', index=False)
    gn.to_csv(f'{OUT}/channel_gains.csv', index=False)
    print('events', len(ev), 'with A', np.isfinite(ev.A).sum())


# ----------------------------------------------------------------------------------------------------------
# ANALYSIS
# ----------------------------------------------------------------------------------------------------------
T0 = pd.Timestamp('2009-01-01')     # 2008 excluded: sparse network and unstable PB/NC gains (see gain check)
TEND = pd.Timestamp('2026-07-01')
MW_MED = 1.5
MU = 30e9
rng = np.random.default_rng(12345)


def load_events():
    ev = pd.read_parquet(f'{OUT}/events_all.parquet')
    ev['qc_bad'] = (ev.snr_rms > 3) | (ev.A > 15) | (ev.A < -3) | ~np.isfinite(ev.A)
    ev = ev[(ev.time >= T0) & (ev.time < TEND)].copy()
    ev['tday'] = (ev.time - T0).dt.total_seconds() / 86400.0
    ev['year'] = ev.time.dt.year
    return ev.sort_values(['fam', 'time']).reset_index(drop=True)


def kde_mode(x, bw=0.05):
    g = np.arange(-1.0, 1.0, 0.005)
    dens = np.exp(-0.5 * ((g[:, None] - x[None, :]) / bw) ** 2).sum(1)
    return g[np.argmax(dens)]


def restart_time(tf, tms):
    after = tf[tf > tms]
    return after[0] if len(after) else np.nan


def completeness(ev):
    """Per family: epochs = calendar years 2009-2026 + 60 days after the restart following each mainshock.
    Mode of the log10 A distribution per epoch (KDE); threshold T = max over epochs (noisiest epoch) + 0.1."""
    rows, thr = [], {}
    msd = {k: (pd.Timestamp(v) - T0).total_seconds() / 86400 for k, v in MS.items()}
    for F, e in ev[~ev.qc_bad].groupby('fam'):
        tf = e.tday.values
        epochs = [(f'{y}', (e.year == y).values) for y in range(2009, 2027)]
        for k, t in msd.items():
            rs = restart_time(tf, t)
            epochs.append((f'post_{k}', (tf >= rs) & (tf < rs + 60)))
        modes = []
        for name, m in epochs:
            x = np.log10(e.A.values[m & (e.A.values > 0)])
            if len(x) < 80:
                rows.append(dict(fam=F, epoch=name, n=len(x), mode=np.nan)); continue
            md = kde_mode(x)
            modes.append(md)
            rows.append(dict(fam=F, epoch=name, n=len(x), mode=md, mean_noise=float(np.nanmedian(e.rel_noise.values[m]))))
        thr[F] = 10 ** (np.max(modes) + 0.1)
    cp = pd.DataFrame(rows)
    cp['T_fam'] = cp.fam.map(thr)
    # stability check: per epoch, mean log10(A/T) above T and KS p vs all other epochs of the family
    from scipy.stats import ks_2samp
    ks = []
    for F, e in ev[~ev.qc_bad].groupby('fam'):
        T = thr[F]; tf = e.tday.values
        above = e.A.values >= T
        epochs = [(f'{y}', (e.year == y).values) for y in range(2009, 2027)]
        for k, t in msd.items():
            rs = restart_time(tf, t)
            epochs.append((f'post_{k}', (tf >= rs) & (tf < rs + 60)))
        for name, m in epochs:
            x = np.log10(e.A.values[m & above] / T); y = np.log10(e.A.values[~m & above] / T)
            if len(x) < 30:
                continue
            ks.append(dict(fam=F, epoch=name, n_above=len(x), frac_above=float((m & above).sum() / max(m.sum(), 1)),
                           mean_excess=float(x.mean()), se=float(x.std() / np.sqrt(len(x))), ks_p=ks_2samp(x, y).pvalue))
    return thr, cp, pd.DataFrame(ks)


def daily_matrix(ev, thr):
    nd = int((TEND - T0).days)
    fams = sorted(ev.fam.unique())
    M = np.zeros((nd, len(fams))); Nn = np.zeros((nd, len(fams)))
    for i, F in enumerate(fams):
        e = ev[(ev.fam == F) & ~ev.qc_bad & (ev.A >= thr[F])]
        d = np.floor(e.tday.values).astype(int)
        np.add.at(M[:, i], d, e.A.values); np.add.at(Nn[:, i], d, 1)
    return M, Nn, fams


def budget_stat(M, tms, trs, W, base=365):
    """Per-family arrays. tms (day float), trs (restart day per family). Returns r, D, E, N (net offset at trs+W)."""
    nd, nf = M.shape
    cum = np.vstack([np.zeros((1, nf)), np.cumsum(M, 0)])
    def Mint(a, b, i):   # integral over [a, b) days (fractional handled at day resolution)
        a = int(np.clip(np.floor(a), 0, nd)); b = int(np.clip(np.floor(b), 0, nd))
        return cum[b, i] - cum[a, i]
    r = np.array([Mint(tms - base, tms, i) / base for i in range(nf)])
    D = np.array([r[i] * (trs[i] - tms) - Mint(tms, trs[i], i) for i in range(nf)])
    E = np.array([Mint(trs[i], trs[i] + W, i) - r[i] * W for i in range(nf)])
    N = np.array([Mint(tms, trs[i] + W, i) - r[i] * (trs[i] + W - tms) for i in range(nf)])
    return r, D, E, N


def budget(ev, thr, M, fams, nboot=1000, base=365):
    rows, pooled, nullrows = [], [], []
    nd = M.shape[0]
    tfam = {F: ev[(ev.fam == F) & ~ev.qc_bad].tday.values for F in fams}
    for k, v in MS.items():
        tms = (pd.Timestamp(v) - T0).total_seconds() / 86400
        trs = np.array([restart_time(tfam[F], tms) for F in fams])
        for W in [90, 180, 365]:
            if trs.max() + W > nd:
                continue
            r, D, E, N = budget_stat(M, tms, trs, W, base)
            # day bootstrap: resample baseline days, and day offsets within the post-restart window (same offsets
            # for all families => inter-family correlation kept); the silence interval is kept as observed
            bN, bD, bE = [], [], []
            ib = np.arange(int(tms) - base, int(tms))
            sil = trs - tms
            Msil = np.array([M[int(tms):int(np.floor(trs[i])), i].sum() for i in range(len(fams))])
            r0 = np.floor(trs).astype(int)
            for b in range(nboot):
                rr = M[rng.choice(ib, base)].mean(0)
                u = rng.integers(0, W, W)
                Mp = np.array([M[r0[i] + u, i].sum() for i in range(len(fams))])
                DD = rr * sil - Msil; EE = Mp - rr * W; NN = Msil + Mp - rr * (sil + W)
                bN.append(NN.sum() / np.sum(rr * (sil + W))); bD.append(DD.sum() / rr.sum()); bE.append(EE.sum() / max(DD.sum(), 1e-9))
            exp_tot = np.sum(r * (trs + W - tms))
            pooled.append(dict(quake=k, W=W, base_days=base, silence_days_median=float(np.median(trs - tms)),
                               deficit_days=float(D.sum() / r.sum()), deficit_days_ci=list(np.percentile(bD, [2.5, 97.5])),
                               catchup_frac=float(E.sum() / D.sum()) if D.sum() > 0 else np.nan,
                               catchup_frac_ci=list(np.percentile(bE, [2.5, 97.5])),
                               net_offset_frac=float(N.sum() / exp_tot), net_offset_frac_ci=list(np.percentile(bN, [2.5, 97.5])),
                               offset_if_no_catchup_frac=float(-D.sum() / exp_tot)))
            for i, F in enumerate(fams):
                rows.append(dict(quake=k, W=W, fam=F, rate_per_day=r[i], silence_d=trs[i] - tms, deficit=D[i], excess=E[i],
                                 net_offset=N[i], net_offset_frac=N[i] / (r[i] * (trs[i] + W - tms)) if r[i] > 0 else np.nan,
                                 catchup_frac=E[i] / D[i] if D[i] > 0 else np.nan))
    # empirical null: random reference dates 2010-2020 (clear of the three quakes), same statistic
    dates = rng.uniform((pd.Timestamp('2010-01-10') - T0).days, (pd.Timestamp('2020-10-01') - T0).days, 300)
    for tms in dates:
        trs = np.array([restart_time(tfam[F], tms) for F in fams])
        for W in [90, 180, 365]:
            r, D, E, N = budget_stat(M, tms, trs, W, base)
            nullrows.append(dict(t=tms, W=W, net_offset_frac=N.sum() / np.sum(r * (trs + W - tms)),
                                 deficit_days=D.sum() / r.sum()))
    return pd.DataFrame(rows), pd.DataFrame(pooled), pd.DataFrame(nullrows)


def ols_fe(x, y, g):
    """slope of y on x with group fixed effects (within-group demeaning)."""
    df = pd.DataFrame(dict(x=x, y=y, g=g))
    xm = df.x - df.groupby('g').x.transform('mean'); ym = df.y - df.groupby('g').y.transform('mean')
    return float((xm * ym).sum() / (xm ** 2).sum())


def healing(ev, thr, nperm=1000, nboot=1000):
    """First event after a quiet period (1 h - 30 d) vs log10 quiet duration."""
    res, recs = {}, []
    msd = {k: (pd.Timestamp(v) - T0).total_seconds() / 86400 for k, v in MS.items()}
    for variant in ['all', 'above_T', 'above_T_gaps_above_T']:
        for F, e in ev[~ev.qc_bad].groupby('fam'):
            T = thr[F]
            t = e.tday.values; A = e.A.values; yr = e.year.values
            if variant == 'above_T_gaps_above_T':
                k = A >= T; t, A, yr = t[k], A[k], yr[k]
                nz = e.rel_noise.values[k]
            else:
                nz = e.rel_noise.values
            gap = np.r_[np.nan, np.diff(t)]
            for i in range(1, len(t)):
                tau = gap[i]
                m7 = (t[i - 1] < msd['M7'] < t[i])
                if not (m7 or (1 / 24 <= tau <= 30)):
                    continue
                if A[i] <= 0:
                    continue
                if variant != 'all' and A[i] < T:
                    continue
                nxt = A[i + 9:i + 30]
                if variant != 'all':
                    nxt = nxt[nxt >= T]
                nxt = nxt[nxt > 0]
                base = np.median(np.log10(nxt)) if len(nxt) >= 5 else np.nan
                prv = A[max(0, i - 20):i]; prv = prv[(prv > 0) & ((prv >= T) if variant != 'all' else True)]
                recs.append(dict(variant=variant, fam=F, year=yr[i], t=t[i], tau_d=tau, logtau=np.log10(tau * 24),
                                 logA=np.log10(A[i]), base10_30=base, base_prev20=np.median(np.log10(prv)) if len(prv) >= 5 else np.nan,
                                 rel_noise=nz[i], m7=m7, petrolia=(t[i - 1] < msd['Petrolia'] < t[i]),
                                 ferndale=(t[i - 1] < msd['Ferndale'] < t[i])))
    H = pd.DataFrame(recs)
    out = []
    for variant, h in H.groupby('variant'):
        h0 = h[~h.m7 & (h.tau_d <= 30)]
        for yname in ['logA', 'delta10_30']:
            hh = h0.copy()
            hh['delta10_30'] = hh.logA - hh.base10_30
            hh = hh[np.isfinite(hh[yname])]
            s = ols_fe(hh.logtau.values, hh[yname].values, hh.fam.values)
            # permutation null: permute amplitudes among burst-first events within family-year
            grp = (hh.fam.astype(str) + '_' + hh.year.astype(str)).values
            yv = hh[yname].values; xv = hh.logtau.values
            idx_groups = pd.Series(np.arange(len(hh))).groupby(grp).apply(lambda s_: s_.values).values
            perm = []
            for p in range(nperm):
                yp = yv.copy()
                for ig in idx_groups:
                    yp[ig] = yv[rng.permutation(ig)]
                perm.append(ols_fe(xv, yp, hh.fam.values))
            perm = np.array(perm)
            # cluster bootstrap over families
            fams = hh.fam.unique(); bs = []
            gb = {F: hh[hh.fam == F] for F in fams}
            for b in range(nboot):
                pick = rng.choice(fams, len(fams))
                bb = pd.concat([gb[F].assign(fam=j) for j, F in enumerate(pick)])
                bs.append(ols_fe(bb.logtau.values, bb[yname].values, bb.fam.values))
            # noise-covariate check (partial slope controlling rel_noise)
            Xd = hh[['logtau', 'rel_noise']].copy(); Xd['y'] = hh[yname]
            for c in ['logtau', 'rel_noise', 'y']:
                Xd[c] = Xd[c] - Xd.groupby(hh.fam.values)[c].transform('mean')
            Xd = Xd.dropna()
            beta = np.linalg.lstsq(Xd[['logtau', 'rel_noise']].values, Xd.y.values, rcond=None)[0]
            # M7 hold point: mean residual of first events after the M7 silence vs fitted trend
            m7 = h[h.m7].copy()
            if yname == 'delta10_30':
                m7['yy'] = m7.logA - m7.base10_30
            else:
                m7['yy'] = m7.logA
            fm = hh.groupby('fam')[yname].mean(); fx = hh.groupby('fam').logtau.mean()
            m7['pred'] = m7.fam.map(fm) + s * (m7.logtau - m7.fam.map(fx))
            m7r = (m7.yy - m7.pred).dropna()
            # reference: residuals of ordinary first events with tau 5-30 d
            ref = hh[(hh.tau_d >= 5)]
            refr = ref[yname] - (ref.fam.map(fm) + s * (ref.logtau - ref.fam.map(fx)))
            m7boot = [rng.choice(m7r.values, len(m7r)).mean() for _ in range(2000)] if len(m7r) > 2 else [np.nan]
            out.append(dict(variant=variant, y=yname, n=len(hh), slope_per_decade=s,
                            slope_ci95=list(np.percentile(bs, [2.5, 97.5])),
                            perm_null_ci95=list(np.percentile(perm, [2.5, 97.5])),
                            perm_p_two_sided=float((np.abs(perm) >= abs(s)).mean()),
                            slope_ctrl_noise=float(beta[0]), noise_coef=float(beta[1]),
                            m7_n=len(m7r), m7_tau_d_median=float(m7.tau_d.median()) if len(m7) else np.nan,
                            m7_mean_resid=float(m7r.mean()) if len(m7r) else np.nan,
                            m7_resid_ci95=list(np.percentile(m7boot, [2.5, 97.5])),
                            ref_tau5_30_resid_sd=float(refr.std()), ref_n=len(refr),
                            m7_z_vs_ref=float(m7r.mean() / (refr.std() / np.sqrt(max(len(m7r), 1)))) if len(m7r) else np.nan))
    return H, pd.DataFrame(out)


def distributions(ev, thr, M_fams):
    """Above-T amplitude distributions in the 365 d before each mainshock vs 365 d after the restart:
    exponential (A-T ~ Exp(lam)) vs Pareto power law (A/T ~ alpha); characteristic-amplitude ratio."""
    rows = []
    msd = {k: (pd.Timestamp(v) - T0).total_seconds() / 86400 for k, v in MS.items()}
    def fits(x, T):
        ex = x - T; lam = ex.mean(); n = len(x)
        ll_exp = -n * np.log(lam) - ex.sum() / lam
        alpha = 1 + n / np.log(x / T).sum()
        ll_pl = n * np.log(alpha - 1) - n * np.log(T) - alpha * np.log(x / T).sum()
        return lam, alpha, ll_exp, ll_pl
    for k, tms in msd.items():
        for F, e in ev[~ev.qc_bad].groupby('fam'):
            T = thr[F]; t = e.tday.values; A = e.A.values
            rs = restart_time(t, tms)
            pre = A[(t >= tms - 365) & (t < tms) & (A >= T)]
            post = A[(t >= rs) & (t < rs + 365) & (A >= T)]
            if len(pre) < 30 or len(post) < 30:
                continue
            lp, ap, lep, lpp = fits(pre, T); lq, aq, leq, lpq = fits(post, T)
            rows.append(dict(quake=k, fam=F, T=T, n_pre=len(pre), n_post=len(post), lam_pre=lp, lam_post=lq,
                             alpha_pre=ap, alpha_post=aq, dLL_exp_minus_pl_pre=lep - lpp, dLL_exp_minus_pl_post=leq - lpq,
                             lam_ratio=lq / lp))
    D = pd.DataFrame(rows)
    summ = []
    for k, d in D.groupby('quake'):
        lr = np.log(d.lam_ratio.values)
        bs = [np.exp(rng.choice(lr, len(lr)).mean()) for _ in range(2000)]
        summ.append(dict(quake=k, n_fam=len(d), lam_ratio_geomean=float(np.exp(lr.mean())), lam_ratio_ci95=list(np.percentile(bs, [2.5, 97.5])),
                         frac_fam_exp_preferred_pre=float((d.dLL_exp_minus_pl_pre > 0).mean()),
                         frac_fam_exp_preferred_post=float((d.dLL_exp_minus_pl_post > 0).mean()),
                         median_dLL_pre=float(d.dLL_exp_minus_pl_pre.median()), median_dLL_post=float(d.dLL_exp_minus_pl_post.median())))
    # null for lam ratio: random dates 2010-2020
    nl = []
    for tms in rng.uniform((pd.Timestamp('2010-01-10') - T0).days, (pd.Timestamp('2020-10-01') - T0).days, 200):
        lr = []
        for F, e in ev[~ev.qc_bad].groupby('fam'):
            T = thr[F]; t = e.tday.values; A = e.A.values
            pre = A[(t >= tms - 365) & (t < tms) & (A >= T)]; post = A[(t >= tms) & (t < tms + 365) & (A >= T)]
            if len(pre) >= 30 and len(post) >= 30:
                lr.append(np.log((post - T).mean() / (pre - T).mean()))
        nl.append(np.exp(np.mean(lr)))
    return D, pd.DataFrame(summ), np.array(nl)


def first25(ev, thr):
    """Mean A of first 25 events after the restart / mean A in the 365 d before (per family; median over families),
    all events and above-T, vs random-date null."""
    msd = {k: (pd.Timestamp(v) - T0).total_seconds() / 86400 for k, v in MS.items()}
    def stat(tms, aboveT):
        r = []
        for F, e in ev[~ev.qc_bad].groupby('fam'):
            T = thr[F] if aboveT else -np.inf
            t = e.tday.values; A = e.A.values; k = A >= T
            pre = A[k & (t >= tms - 365) & (t < tms)]; post = A[k & (t > tms)][:25]
            if len(pre) >= 25 and len(post) == 25:
                r.append(post.mean() / pre.mean())
        return np.median(r), np.array(r)
    out = {}
    null_dates = rng.uniform((pd.Timestamp('2010-01-10') - T0).days, (pd.Timestamp('2020-10-01') - T0).days, 300)
    for aboveT in [False, True]:
        nul = np.array([stat(t, aboveT)[0] for t in null_dates])
        for k, tms in msd.items():
            m, r = stat(tms, aboveT)
            bs = [np.median(rng.choice(r, len(r))) for _ in range(2000)]
            out[f'{k}_{"aboveT" if aboveT else "all"}'] = dict(median_ratio=float(m), ci95=list(np.percentile(bs, [2.5, 97.5])),
                                                              n_fam=len(r), null_2p5_97p5=list(np.percentile(nul, [2.5, 97.5])),
                                                              null_frac_ge=float((nul >= m).mean()))
    return out


def slip(ev, thr):
    pt = pd.read_csv(f'{ROOT}/data/patches/patch_tensors.csv').set_index('family')
    M0med = 10 ** (1.5 * MW_MED + 9.1)
    t1 = (pd.Timestamp(MS['M7']) - T0).total_seconds() / 86400
    rows = []
    for F, e in ev[~ev.qc_bad].groupby('fam'):
        k = e.tday.values < t1
        sumA = e.A.values[k].sum(); sumA_T = e.A.values[k & (e.A.values >= thr[F])].sum()
        yrs = t1 / 365.25
        p = pt.loc[F]
        res = bool(p.resolved)
        L1 = p.L1_m if res else max(p.L1_m, p.L1_null95)
        L2 = p.L2_m if (res and p.L2_m > 0) else max(p.L2_m, p.L2_null95)
        area = np.pi * L1 / 2 * L2 / 2
        M0 = sumA * M0med
        s = M0 / (MU * area) / yrs * 100
        rows.append(dict(fam=F, n=int(k.sum()), sumA=sumA, frac_moment_above_T=sumA_T / sumA, years=yrs, M0_total_Nm=M0,
                         L1_m=L1, L2_m=L2, resolved=res, area_m2=area, slip_rate_cm_yr=s,
                         slip_rate_lo_cm_yr=s / 10 ** (1.5 * 0.5), slip_rate_hi_cm_yr=s * 10 ** (1.5 * 0.5)))
    return pd.DataFrame(rows)


def network_check(ev):
    """Common-mode network check: yearly log10(sum A_pb / sum A_nc) per family, median over families."""
    rows = []
    for (F, y), e in ev[~ev.qc_bad & np.isfinite(ev.A_pb) & np.isfinite(ev.A_nc)].groupby(['fam', 'year']):
        if len(e) < 30:
            continue
        rows.append(dict(fam=F, year=y, lr=np.log10(e.A_pb.sum() / e.A_nc.sum()) if e.A_nc.sum() > 0 else np.nan))
    d = pd.DataFrame(rows).dropna()
    s = d.groupby('year').lr.agg(['median', 'count', lambda x: 1.2533 * x.std() / np.sqrt(len(x))])
    s.columns = ['median_log10_PB_over_NC', 'n_fam', 'se']
    return s.reset_index()


def analyse():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ev = load_events()
    S = {}
    S['n_events'] = int(len(ev)); S['n_qc_bad'] = int(ev.qc_bad.sum())
    S['median_event_snr'] = float(np.nanmedian(ev.A / ev.sigA))
    S['median_sigma_log10A'] = float(np.nanmedian(ev.sigA / ev.A / np.log(10)))
    ok = ~ev.qc_bad & (ev.A > 0.1)
    S['corr_log10A_vs_rms_proxy'] = float(np.corrcoef(np.log10(ev.A[ok]), ev.rel_rms[ok])[0, 1])
    S['corr_log10A_vs_cc_div_thr'] = float(np.corrcoef(np.log10(ev.A[ok]), ev.cc_div_thr[ok])[0, 1])
    k2 = ok & (ev.A_pb > 0.05) & (ev.A_nc > 0.05)
    S['corr_log10A_PBonly_vs_NConly'] = float(np.corrcoef(np.log10(ev.A_pb[k2]), np.log10(ev.A_nc[k2]))[0, 1])
    jr = pd.read_csv(f'{OUT}/channel_year_gain_check.csv')
    S['gain_check_halfyear_blocks'] = int(len(jr)); S['gain_check_removed_frac'] = float(jr.removed.mean())
    nc = network_check(ev); nc.to_csv(f'{OUT}/network_PB_vs_NC_by_year.csv', index=False)
    S['network_PB_over_NC_log10_range'] = [float(nc.median_log10_PB_over_NC.min()), float(nc.median_log10_PB_over_NC.max())]
    print('network check\n', nc.round(3).to_string(), flush=True)

    thr, cp, ks = completeness(ev)
    cp.to_csv(f'{OUT}/completeness_modes_by_epoch.csv', index=False); ks.to_csv(f'{OUT}/completeness_stability.csv', index=False)
    S['T_by_family'] = {int(k): float(v) for k, v in thr.items()}
    S['T_median'] = float(np.median(list(thr.values())))
    noisiest = cp.loc[cp.groupby('fam')['mode'].idxmax()].epoch.value_counts().to_dict()
    S['noisiest_epoch_counts'] = {str(k): int(v) for k, v in noisiest.items()}
    S['frac_events_above_T'] = float(np.mean([(ev[(ev.fam == F) & ~ev.qc_bad].A >= T).mean() for F, T in thr.items()]))
    S['ks_frac_p_lt_0.01'] = float((ks.ks_p < 0.01).mean())
    yrs = ks[ks.epoch.str.match(r'^\d{4}$')]
    S['mean_excess_by_year_range'] = [float(yrs.groupby('epoch').mean_excess.median().min()), float(yrs.groupby('epoch').mean_excess.median().max())]
    print('T', S['T_median'], S['noisiest_epoch_counts'], flush=True)

    M, Nn, fams = daily_matrix(ev, thr)
    br, bp, bn = budget(ev, thr, M, fams)
    br.to_csv(f'{OUT}/budget_by_family.csv', index=False); bp.to_csv(f'{OUT}/budget_pooled.csv', index=False)
    bn.to_csv(f'{OUT}/budget_null_random_dates.csv', index=False)
    for i, r in bp.iterrows():
        nul = bn[bn.W == r.W].net_offset_frac.values
        bp.loc[i, 'null_net_offset_2p5'] = np.percentile(nul, 2.5); bp.loc[i, 'null_net_offset_97p5'] = np.percentile(nul, 97.5)
        bp.loc[i, 'null_frac_le'] = (nul <= r.net_offset_frac).mean()
    bp.to_csv(f'{OUT}/budget_pooled.csv', index=False)
    S['budget_pooled'] = bp.to_dict('records')
    bp365 = budget(ev, thr, M, fams, nboot=300, base=730)[1]
    bp365.to_csv(f'{OUT}/budget_pooled_base730.csv', index=False)
    S['budget_pooled_base730'] = bp365.to_dict('records')
    print(bp.round(3).to_string(), flush=True)

    H, hs = healing(ev, thr)
    H.to_csv(f'{OUT}/healing_first_events.csv', index=False); hs.to_csv(f'{OUT}/healing_regressions.csv', index=False)
    S['healing'] = hs.to_dict('records')
    print(hs.round(3).to_string(), flush=True)

    D, ds, dnull = distributions(ev, thr, fams)
    D.to_csv(f'{OUT}/distributions_by_family.csv', index=False)
    ds['null_lam_ratio_2p5'] = np.percentile(dnull, 2.5); ds['null_lam_ratio_97p5'] = np.percentile(dnull, 97.5)
    ds.to_csv(f'{OUT}/distributions_summary.csv', index=False)
    S['distributions'] = ds.to_dict('records')
    print(ds.round(3).to_string(), flush=True)

    S['first25'] = first25(ev, thr)
    print(json.dumps(S['first25'], indent=1), flush=True)

    sl = slip(ev, thr); sl.to_csv(f'{OUT}/slip_by_family.csv', index=False)
    S['slip_rate_cm_yr_median'] = float(sl.slip_rate_cm_yr.median())
    S['slip_rate_cm_yr_resolved_median'] = float(sl[sl.resolved].slip_rate_cm_yr.median())
    S['slip_rate_cm_yr_range'] = [float(sl.slip_rate_cm_yr.min()), float(sl.slip_rate_cm_yr.max())]
    S['slip_assumptions'] = 'median family LFE = Mw 1.5 (M0 2.2e11 N m), M0 proportional to MF amplitude, mu = 30 GPa, area = pi*L1/2*L2/2 (unresolved: max(L, null95)), 2009-01-01..M7; +-0.5 Mw => x/÷5.6'
    print(sl.round(3).to_string(), flush=True)

    json.dump(S, open(f'{OUT}/summary.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    figures(ev, thr, M, fams, bp, H, hs, ks, nc, D)
    cumulative_figure()


def figures(ev, thr, M, fams, bp, H, hs, ks, nc, D):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    days = T0 + pd.to_timedelta(np.arange(M.shape[0]), 'D')
    # --- Fig 1: cumulative moment (all families, above-T) + zooms around quakes
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5), gridspec_kw=dict(height_ratios=[1, 1]))
    a0 = plt.subplot2grid((2, 3), (0, 0), colspan=3, fig=fig)
    tot = M.sum(1); cum = np.cumsum(tot)
    a0.plot(days, cum / 1e3, color='#1f4e79', lw=1.2)
    for k, v in MS.items():
        a0.axvline(pd.Timestamp(v), color='#b03a2e', lw=0.8, ls='--'); a0.text(pd.Timestamp(v), cum[-1] / 1e3 * 0.05, k, rotation=90, color='#b03a2e', fontsize=8)
    a0.set_ylabel('cumulative relative moment\n(10^3 median-LFE units, above T)'); a0.set_title('All 27 families: cumulative relative moment (events above completeness T)')
    for j, (k, v) in enumerate(MS.items()):
        ax_ = ax[1, j]
        tms = (pd.Timestamp(v) - T0).total_seconds() / 86400
        i0 = int(tms) - 365; i1 = min(int(tms) + 400, M.shape[0])
        c = np.cumsum(tot[i0:i1]); c = c - c[365 - 1]
        r = tot[int(tms) - 365:int(tms)].mean()
        xx = np.arange(i0, i1) - tms
        ax_.plot(xx, c, color='#1f4e79', lw=1.3, label='observed')
        ax_.plot(xx, r * (xx), color='0.4', ls='--', lw=1, label='pre-quake 365-d trend')
        # day-bootstrap band of the trend
        rb = np.array([tot[rng.choice(np.arange(int(tms) - 365, int(tms)), 365)].mean() for _ in range(300)])
        ax_.fill_between(xx[xx > 0], np.percentile(rb, 2.5) * xx[xx > 0], np.percentile(rb, 97.5) * xx[xx > 0], color='0.6', alpha=0.3, lw=0)
        ax_.axvline(0, color='#b03a2e', lw=0.8)
        row = bp[(bp.quake == k) & (bp.W == 365)]
        if len(row):
            r_ = row.iloc[0]
            ax_.text(0.03, 0.97, f"net offset at restart+365 d:\n{100 * r_.net_offset_frac:.0f}% of expected\n[{100 * r_.net_offset_frac_ci[0]:.0f}, {100 * r_.net_offset_frac_ci[1]:.0f}]% (day bootstrap)\n"
                     f"random-date null 95%: [{100 * r_.null_net_offset_2p5:.0f}, {100 * r_.null_net_offset_97p5:.0f}]%",
                     transform=ax_.transAxes, va='top', fontsize=8)
        ax_.set_title(f'{k} ({v[:10]})'); ax_.set_xlabel('days from mainshock'); ax_.set_ylabel('cum. moment since mainshock')
        ax_.legend(loc='lower right', fontsize=8)
    fig.tight_layout(); fig.savefig(f'{ROOT}/figures/sci_moments_cumulative.png', dpi=130); plt.close(fig)

    # --- Fig 2: healing
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    bins = np.log10(np.array([1, 3, 10, 30, 100, 300, 720]))
    for j, (variant, yname, lab) in enumerate([('all', 'logA', 'all first events: log10 A'),
                                               ('above_T', 'logA', 'first events above T: log10 A'),
                                               ('above_T', 'delta10_30', 'above T: log10 A(first) - median log10 A(events 10-30)')]):
        h = H[(H.variant == variant)].copy(); h['delta10_30'] = h.logA - h.base10_30
        h0 = h[~h.m7 & (h.tau_d <= 30)].dropna(subset=[yname])
        y = h0[yname] - h0.groupby('fam')[yname].transform('mean') + h0[yname].mean()
        cb = pd.cut(h0.logtau, bins)
        g = y.groupby(cb, observed=False)
        xm = h0.logtau.groupby(cb, observed=False).mean()
        ax[j].errorbar(xm, g.mean(), yerr=1.96 * g.std() / np.sqrt(g.count()), fmt='o-', color='#1f4e79', label='ordinary first events (bin mean, 95%)')
        # ordinary 10-30 baseline for comparison
        if yname == 'logA':
            yb = h0.base10_30 - h0.groupby('fam').base10_30.transform('mean') + h0.base10_30.mean()
            gb = yb.groupby(cb, observed=False)
            ax[j].errorbar(xm + 0.03, gb.mean(), yerr=1.96 * gb.std() / np.sqrt(gb.count()), fmt='s--', color='0.5', ms=4, label='median of events 10-30 after')
        m7 = h[h.m7].dropna(subset=[yname])
        if len(m7):
            ym7 = m7[yname] - m7.fam.map(h0.groupby('fam')[yname].mean()) + h0[yname].mean()
            ax[j].scatter(m7.logtau, ym7, s=14, color='#b03a2e', alpha=0.6, label='first event after M7 silence (per family)')
            ax[j].errorbar([m7.logtau.mean()], [ym7.mean()], yerr=[1.96 * ym7.std() / np.sqrt(len(ym7))], fmt='D', color='#b03a2e', ms=8)
        r = hs[(hs.variant == variant) & (hs.y == yname)].iloc[0]
        ax[j].set_title(f"{lab}\nslope {r.slope_per_decade:+.3f}/decade [{r.slope_ci95[0]:+.3f},{r.slope_ci95[1]:+.3f}]; perm null [{r.perm_null_ci95[0]:+.3f},{r.perm_null_ci95[1]:+.3f}]", fontsize=8.5)
        ax[j].set_xlabel('log10 quiet duration (hours)'); ax[j].set_xticks(bins); ax[j].set_xticklabels(['1h', '3h', '10h', '30h', '4d', '12.5d', '30d'])
        ax[j].legend(fontsize=7, loc='upper left')
    fig.tight_layout(); fig.savefig(f'{ROOT}/figures/sci_moments_healing.png', dpi=130); plt.close(fig)

    # --- Fig 3: completeness / gain / distributions
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    e = ev[~ev.qc_bad & (ev.A > 0)].copy(); e['lr'] = np.log10(e.A / e.fam.map(thr))
    cmap = plt.get_cmap('viridis')
    for i, y in enumerate(range(2009, 2027)):
        x = e[e.year == y].lr.values
        h_, b_ = np.histogram(x, bins=np.arange(-1.5, 1.2, 0.05), density=True)
        ax[0].plot(b_[:-1] + 0.025, h_, color=cmap(i / 18), lw=0.9, label=str(y) if y % 3 == 0 else None)
    ax[0].axvline(0, color='k', lw=0.8); ax[0].set_yscale('log'); ax[0].set_xlabel('log10(A / T_family)'); ax[0].set_ylabel('density')
    ax[0].set_title('amplitude distributions by year (all families)\nT = noisiest-epoch mode + 0.1'); ax[0].legend(fontsize=7)
    yrs = ks[ks.epoch.str.match(r'^\d{4}$')].copy(); yrs['y'] = yrs.epoch.astype(int)
    g = yrs.groupby('y').mean_excess
    ax[1].errorbar(g.median().index, g.median(), yerr=1.2533 * g.std() / np.sqrt(g.count()), fmt='o-', color='#1f4e79', label='median over families: mean log10(A/T) above T')
    ax[1].errorbar(nc.year, nc.median_log10_PB_over_NC, yerr=nc.se, fmt='s--', color='#b03a2e', ms=4, label='log10(sum A_PB / sum A_NC) (network check)')
    ax[1].axhline(0, color='0.5', lw=0.6); ax[1].set_xlabel('year'); ax[1].legend(fontsize=7); ax[1].set_title('stability above T and PB-vs-NC gain check')
    for k, c in zip(MS, ['#2e86c1', '#28b463', '#b03a2e']):
        d = D[D.quake == k]
        ax[2].scatter(d.lam_pre, d.lam_post, s=16, color=c, label=k)
    lim = [0, max(D.lam_pre.max(), D.lam_post.max()) * 1.1]
    ax[2].plot(lim, lim, 'k-', lw=0.6); ax[2].set_xlim(lim); ax[2].set_ylim(lim)
    ax[2].set_xlabel('exp. scale lambda, 365 d before'); ax[2].set_ylabel('lambda, 365 d after restart'); ax[2].legend(fontsize=8)
    ax[2].set_title('characteristic amplitude above T, per family')
    fig.tight_layout(); fig.savefig(f'{ROOT}/figures/sci_moments_completeness_distributions.png', dpi=130); plt.close(fig)



def extras():
    """Sensitivity checks: (1) healing injection-recovery through the above-T selection; (2) count-based and
    all-event moment budgets for the M7; (3) 7-day block bootstrap of the M7 net offset."""
    S = json.load(open(f'{OUT}/summary.json'))
    thr = {int(k): v for k, v in S['T_by_family'].items()}
    ev = load_events()
    X = {}
    # (1) injection: multiply first-event amplitudes by 10**(delta*(log10 tau_h - 1.5)) BEFORE the above-T cut
    H = pd.read_csv(f'{OUT}/healing_first_events.csv')
    h = H[(H.variant == 'all') & ~H.m7 & (H.tau_d <= 30)].copy()
    inj = []
    for delta in [0.0, 0.02, 0.05, 0.1]:
        rec = []
        for b in range(50):
            # add measurement-like scatter of the injected trend only once (deterministic), resample events for CI
            hh = h.sample(frac=1, replace=True, random_state=b) if b else h
            la = hh.logA + delta * (hh.logtau - 1.5)
            k = la >= np.log10(hh.fam.map(thr))
            rec.append(ols_fe(hh.logtau.values[k], la.values[k], hh.fam.values[k]))
        inj.append(dict(delta_injected=delta, recovered=rec[0], recovered_sd=float(np.std(rec[1:]))))
    X['healing_injection_aboveT'] = inj
    print(pd.DataFrame(inj).round(4).to_string(), flush=True)
    # (2) budgets with counts above T and with all-event moment (no threshold)
    fams = sorted(ev.fam.unique())
    nd = int((TEND - T0).days)
    Mc = np.zeros((nd, len(fams))); Ma = np.zeros((nd, len(fams)))
    for i, F in enumerate(fams):
        e = ev[(ev.fam == F) & ~ev.qc_bad]
        d = np.floor(e.tday.values).astype(int)
        np.add.at(Ma[:, i], d, e.A.values)
        k = e.A.values >= thr[F]
        np.add.at(Mc[:, i], d[k], 1)
    for name, MM in [('count_aboveT', Mc), ('moment_all_events', Ma)]:
        bp = budget(ev, thr, MM, fams, nboot=500)[1]
        X[f'budget_{name}'] = bp[['quake', 'W', 'deficit_days', 'catchup_frac', 'catchup_frac_ci', 'net_offset_frac',
                                  'net_offset_frac_ci', 'offset_if_no_catchup_frac']].to_dict('records')
        print(name, '\n', bp[['quake', 'W', 'deficit_days', 'catchup_frac', 'catchup_frac_ci', 'net_offset_frac', 'net_offset_frac_ci', 'offset_if_no_catchup_frac']].round(3).to_string(), flush=True)
    # (3) 7-day block bootstrap of M7 net offset (moment above T)
    M, Nn, fams = daily_matrix(ev, thr)
    tfam = {F: ev[(ev.fam == F) & ~ev.qc_bad].tday.values for F in fams}
    tms = (pd.Timestamp(MS['M7']) - T0).total_seconds() / 86400
    trs = np.array([restart_time(tfam[F], tms) for F in fams]); sil = trs - tms
    Msil = np.array([M[int(tms):int(np.floor(trs[i])), i].sum() for i in range(len(fams))])
    r0 = np.floor(trs).astype(int)
    blk = []
    for W in [90, 180, 365]:
        out = []
        for b in range(2000):
            st = rng.integers(int(tms) - 365, int(tms) - 7, 365 // 7 + 1)
            ib = (st[:, None] + np.arange(7)[None, :]).ravel()[:365]
            rr = M[ib].mean(0)
            su = rng.integers(0, W - 7, W // 7 + 1)
            u = (su[:, None] + np.arange(7)[None, :]).ravel()[:W]
            Mp = np.array([M[r0[i] + u, i].sum() for i in range(len(fams))])
            NN = Msil + Mp - rr * (sil + W)
            DD = rr * sil - Msil
            out.append((NN.sum() / np.sum(rr * (sil + W)), (Mp - rr * W).sum() / DD.sum()))
        out = np.array(out)
        blk.append(dict(W=W, net_offset_ci95_block7=list(np.percentile(out[:, 0], [2.5, 97.5])),
                        catchup_frac_ci95_block7=list(np.percentile(out[:, 1], [2.5, 97.5]))))
    X['M7_block7_bootstrap'] = blk
    print(blk, flush=True)
    S['extras'] = X
    json.dump(S, open(f'{OUT}/summary.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))



def healing_exceedance(nperm=1000, nboot=1000):
    """Truncation-robust healing test. For an exponential tail, mean(A - T | A >= T) = lambda scales with any
    multiplicative amplitude change, whereas mean log A above T is strongly attenuated (injection test ~20%).
    z = (A - T_F)/lambda_F (lambda_F = family mean exceedance over all events above T); slope dz/dlog10(tau) =
    ln(10)*delta for an amplitude trend A ~ tau**delta. delta_hat = slope/ln(10)."""
    S = json.load(open(f'{OUT}/summary.json'))
    thr = {int(k): v for k, v in S['T_by_family'].items()}
    ev = load_events()
    lamF = {F: (e.A[e.A >= thr[F]] - thr[F]).mean() for F, e in ev[~ev.qc_bad].groupby('fam')}
    H = pd.read_csv(f'{OUT}/healing_first_events.csv')
    res = []
    for variant in ['above_T', 'above_T_gaps_above_T', 'all_injection']:
        if variant == 'all_injection':
            base = H[(H.variant == 'all') & ~H.m7 & (H.tau_d <= 30)].copy()
        else:
            base = H[(H.variant == variant)].copy()
        for delta in ([0.0, 0.05, 0.1] if variant == 'all_injection' else [None]):
            h = base.copy()
            if delta is not None:
                h['logA'] = h.logA + delta * (h.logtau - 1.5)
                h = h[h.logA >= np.log10(h.fam.map(thr))]
            h['z'] = (10 ** h.logA - h.fam.map(thr)) / h.fam.map(lamF)
            h0 = h[~h.m7 & (h.tau_d <= 30)]
            sl = ols_fe(h0.logtau.values, h0.z.values, h0.fam.values)
            grp = (h0.fam.astype(str) + '_' + h0.year.astype(str)).values
            idx_groups = pd.Series(np.arange(len(h0))).groupby(grp).apply(lambda s_: s_.values).values
            zv = h0.z.values; xv = h0.logtau.values; fv = h0.fam.values
            perm = []
            for p_ in range(nperm if delta is None else 0):
                zp = zv.copy()
                for ig in idx_groups:
                    zp[ig] = zv[rng.permutation(ig)]
                perm.append(ols_fe(xv, zp, fv))
            fams = h0.fam.unique(); gb = {F: h0[h0.fam == F] for F in fams}; bs = []
            for b in range(nboot if delta is None else 200):
                pick = rng.choice(fams, len(fams))
                bb = pd.concat([gb[F].assign(fam=j) for j, F in enumerate(pick)])
                bs.append(ols_fe(bb.logtau.values, bb.z.values, bb.fam.values))
            r = dict(variant=variant, delta_injected=delta, n=len(h0), slope_z=sl, delta_hat=sl / np.log(10),
                     delta_ci95=list(np.percentile(np.array(bs) / np.log(10), [2.5, 97.5])))
            if perm:
                r['perm_null_delta_ci95'] = list(np.percentile(np.array(perm) / np.log(10), [2.5, 97.5]))
                r['perm_p'] = float((np.abs(np.array(perm) - np.mean(perm)) >= abs(sl - np.mean(perm))).mean())
            m7 = h[h.m7]
            if len(m7) and delta is None:
                fm = h0.groupby('fam').z.mean(); fx = h0.groupby('fam').logtau.mean()
                rr = (m7.z - (m7.fam.map(fm) + sl * (m7.logtau - m7.fam.map(fx)))).dropna()
                ref = h0[h0.tau_d >= 5]; refr = ref.z - (ref.fam.map(fm) + sl * (ref.logtau - ref.fam.map(fx)))
                r.update(m7_n=len(rr), m7_mean_resid_z=float(rr.mean()),
                         m7_resid_ci95=list(np.percentile([rng.choice(rr.values, len(rr)).mean() for _ in range(2000)], [2.5, 97.5])),
                         m7_z_score=float(rr.mean() / (refr.std() / np.sqrt(len(rr)))))
            res.append(r); print(r, flush=True)
    S['healing_exceedance'] = res
    json.dump(S, open(f'{OUT}/summary.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    pd.DataFrame(res).to_csv(f'{OUT}/healing_exceedance.csv', index=False)



def healing_figure():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    S = json.load(open(f'{OUT}/summary.json'))
    thr = {int(k): v for k, v in S['T_by_family'].items()}
    ev = load_events()
    lamF = {F: (e.A[e.A >= thr[F]] - thr[F]).mean() for F, e in ev[~ev.qc_bad].groupby('fam')}
    H = pd.read_csv(f'{OUT}/healing_first_events.csv')
    hs = pd.DataFrame(S['healing']); he = pd.DataFrame(S['healing_exceedance'])
    bins = np.log10(np.array([1, 3, 10, 30, 100, 300, 720]))
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.8))
    panels = [('all', 'logA', 'ALL first events (incomplete): log10 A'),
              ('above_T', 'z', 'first events above T (gaps from all events):\nexceedance z=(A-T)/lambda'),
              ('above_T_gaps_above_T', 'z', 'above T, gaps between above-T events:\nexceedance z=(A-T)/lambda')]
    for j, (variant, yname, lab) in enumerate(panels):
        h = H[H.variant == variant].copy()
        h['z'] = (10 ** h.logA - h.fam.map(thr)) / h.fam.map(lamF)
        h0 = h[~h.m7 & (h.tau_d <= 30)].dropna(subset=[yname])
        y = h0[yname] - h0.groupby('fam')[yname].transform('mean') + h0[yname].mean()
        cb = pd.cut(h0.logtau, bins); g = y.groupby(cb, observed=False); xm = h0.logtau.groupby(cb, observed=False).mean()
        ax[j].errorbar(xm, g.mean(), yerr=1.96 * g.std() / np.sqrt(g.count()), fmt='o-', color='#1f4e79', label='ordinary first events (bin mean +-95%)')
        if variant == 'all':
            yb = h0.base10_30 - h0.groupby('fam').base10_30.transform('mean') + h0.base10_30.mean()
            gb = yb.groupby(cb, observed=False)
            ax[j].errorbar(xm + 0.03, gb.mean(), yerr=1.96 * gb.std() / np.sqrt(gb.count()), fmt='s--', color='0.5', ms=4, label='median log10 A of events 10-30 after')
            r = hs[(hs.variant == 'all') & (hs.y == 'logA')].iloc[0]
            txt = f"slope {r.slope_per_decade:+.3f}/decade [{r.slope_ci95[0]:+.3f},{r.slope_ci95[1]:+.3f}]\nperm. null [{r.perm_null_ci95[0]:+.3f},{r.perm_null_ci95[1]:+.3f}] -> detection artefact"
        else:
            r = he[(he.variant == variant)].iloc[0]
            txt = f"delta = {r.delta_hat:+.3f}/decade [{r.delta_ci95[0]:+.3f},{r.delta_ci95[1]:+.3f}]\nperm. null [{r.perm_null_delta_ci95[0]:+.3f},{r.perm_null_delta_ci95[1]:+.3f}]"
        m7 = h[h.m7].dropna(subset=[yname])
        if len(m7):
            ym7 = m7[yname] - m7.fam.map(h0.groupby('fam')[yname].mean()) + h0[yname].mean()
            ax[j].scatter(m7.logtau, ym7, s=14, color='#b03a2e', alpha=0.5, label=f'first event after M7 silence (n={len(m7)})')
            if len(m7) > 2:
                ax[j].errorbar([m7.logtau.mean()], [ym7.mean()], yerr=[1.96 * ym7.std() / np.sqrt(len(ym7))], fmt='D', color='#b03a2e', ms=8)
        ax[j].set_title(lab, fontsize=9); ax[j].text(0.02, 0.03, txt, transform=ax[j].transAxes, fontsize=8)
        ax[j].set_xlabel('quiet duration before the event'); ax[j].set_xticks(bins); ax[j].set_xticklabels(['1h', '3h', '10h', '30h', '4d', '12.5d', '30d'])
        ax[j].legend(fontsize=7, loc='upper left')
    ax[0].set_ylabel('family-demeaned value')
    fig.suptitle('Slide-hold-slide test: amplitude of the first LFE after a quiet period (27 families, 2009-2026)', fontsize=10)
    fig.tight_layout(); fig.savefig(f'{ROOT}/figures/sci_moments_healing.png', dpi=130); plt.close(fig)



def cumulative_figure():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    S = json.load(open(f'{OUT}/summary.json'))
    thr = {int(k): v for k, v in S['T_by_family'].items()}
    bp = pd.DataFrame(S['budget_pooled'])
    ev = load_events()
    M, Nn, fams = daily_matrix(ev, thr)
    tfam = {F: ev[(ev.fam == F) & ~ev.qc_bad].tday.values for F in fams}
    days = T0 + pd.to_timedelta(np.arange(M.shape[0]), 'D')
    tot = M.sum(1)
    fig = plt.figure(figsize=(15, 8.5)); gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15])
    a0 = fig.add_subplot(gs[0, :])
    a0.plot(days, np.cumsum(tot) / 1e3, color='#1f4e79', lw=1.2)
    for k, v in MS.items():
        a0.axvline(pd.Timestamp(v), color='#b03a2e', lw=0.8, ls='--')
        a0.text(pd.Timestamp(v), 5, ' ' + k, rotation=90, color='#b03a2e', fontsize=8, va='bottom')
    a0.set_ylabel('cumulative relative moment\n(10^3 median-LFE units)')
    a0.set_title('All 27 families, events above completeness T (fixed long-running stations): cumulative relative moment')
    for j, (k, v) in enumerate(MS.items()):
        ax_ = fig.add_subplot(gs[1, j])
        tms = (pd.Timestamp(v) - T0).total_seconds() / 86400
        trs = np.array([restart_time(tfam[F], tms) for F in fams])
        ib = np.arange(int(tms) - 365, int(tms))
        r = M[ib].mean(0); rt = r.sum()
        x = np.arange(-120, min(400, M.shape[0] - int(tms)))
        c = np.cumsum(tot[int(tms) + x[0]:int(tms) + x[-1] + 1]); c = c - c[-x[0] - 1]
        resid_days = (c - rt * (x + 1)) / rt    # residual from trend, in days of pre-quake moment rate
        ax_.plot(x, resid_days, color='#1f4e79', lw=1.4, label='observed - pre-quake trend')
        # day-bootstrap band of the trend extrapolation
        rb = np.array([M[rng.choice(ib, 365)].sum(1).mean() for _ in range(500)])
        lo, hi = np.percentile(rb / rt - 1, [2.5, 97.5])
        xp = x[x >= 0]
        ax_.fill_between(xp, -hi * xp, -lo * xp, color='0.6', alpha=0.35, lw=0, label='trend-rate uncertainty (day bootstrap)')
        D = np.sum(r * (trs - tms) - np.array([M[int(tms):int(np.floor(trs[i])), i].sum() for i in range(len(fams))])) / rt
        ax_.axhline(-D, color='#b03a2e', ls=':', lw=1.2, label=f'no catch-up level (deficit {D:.1f} d)')
        ax_.axhline(0, color='0.3', lw=0.7)
        ax_.axvline(0, color='#b03a2e', lw=0.8)
        row = bp[(bp.quake == k) & (bp.W == 90)].iloc[0]; row3 = bp[(bp.quake == k) & (bp.W == 365)].iloc[0]
        ax_.text(0.02, 0.03, f"catch-up fraction at restart+90 d: {row.catchup_frac:.2f} [{row.catchup_frac_ci[0]:.2f}, {row.catchup_frac_ci[1]:.2f}]\n"
                 f"net offset at +365 d: {100 * row3.net_offset_frac:+.0f}% [{100 * row3.net_offset_frac_ci[0]:+.0f}, {100 * row3.net_offset_frac_ci[1]:+.0f}]% of expected\n"
                 f"(random-date null 95%: [{100 * row3.null_net_offset_2p5:+.0f}, {100 * row3.null_net_offset_97p5:+.0f}]%)",
                 transform=ax_.transAxes, fontsize=7.5, va='bottom')
        ax_.set_title(f'{k} ({v[:10]}), median silence {np.median(trs - tms):.1f} d', fontsize=10)
        ax_.set_xlabel('days from mainshock'); ax_.set_ylabel('cum. moment residual (days of pre-quake rate)')
        ax_.legend(fontsize=7, loc='upper left')
    fig.tight_layout(); fig.savefig(f'{ROOT}/figures/sci_moments_cumulative.png', dpi=130); plt.close(fig)



def rate_after_restart():
    """Moment-rate ratio (above T, all families) in the first 30/60 d after each family's restart vs the
    pre-quake 365-d rate, with day bootstrap and a random-date null (same statistic, restart = next event)."""
    S = json.load(open(f'{OUT}/summary.json'))
    thr = {int(k): v for k, v in S['T_by_family'].items()}
    ev = load_events()
    M, Nn, fams = daily_matrix(ev, thr)
    tfam = {F: ev[(ev.fam == F) & ~ev.qc_bad].tday.values for F in fams}
    def stat(tms, W, boot=False):
        trs = np.array([restart_time(tfam[F], tms) for F in fams]); r0 = np.floor(trs).astype(int)
        ib = np.arange(int(tms) - 365, int(tms))
        if boot:
            ib = rng.choice(ib, 365); u = rng.integers(0, W, W)
        else:
            u = np.arange(W)
        base = M[ib].sum(1).mean()
        post = np.mean([sum(M[r0[i] + u, i]) for i in range(len(fams))]) * len(fams) / W
        return post / base
    out = {}
    nd = rng.uniform((pd.Timestamp('2010-01-10') - T0).days, (pd.Timestamp('2020-10-01') - T0).days, 300)
    for W in [30, 60]:
        nul = np.array([stat(t, W) for t in nd])
        for k, v in MS.items():
            tms = (pd.Timestamp(v) - T0).total_seconds() / 86400
            val = stat(tms, W); bs = [stat(tms, W, True) for _ in range(500)]
            out[f'{k}_{W}d'] = dict(rate_ratio=float(val), ci95=list(np.percentile(bs, [2.5, 97.5])),
                                    null95=list(np.percentile(nul, [2.5, 97.5])), null_frac_ge=float((nul >= val).mean()))
    print(json.dumps(out, indent=1))
    S['rate_after_restart'] = out
    json.dump(S, open(f'{OUT}/summary.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))



def export():
    S = json.load(open(f'{OUT}/summary.json'))
    thr = {int(k): v for k, v in S['T_by_family'].items()}
    ev = pd.read_parquet(f'{OUT}/events_all.parquet')
    ev['qc_bad'] = (ev.snr_rms > 3) | (ev.A > 15) | (ev.A < -3) | ~np.isfinite(ev.A)
    ev['T_fam'] = ev.fam.map(thr); ev['above_T'] = ev.A >= ev.T_fam
    cols = ['fam', 'ev', 'time', 'A', 'sigA', 'nchan', 'A_pb', 'A_nc', 'rel_rms', 'rel_noise', 'snr_rms', 'cc_div_thr', 'qc_bad', 'T_fam', 'above_T']
    ev[cols].to_csv(f'{OUT}/event_amplitudes.csv.gz', index=False, float_format='%.4g')
    print('wrote', len(ev))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage')
    ap.add_argument('--fams', default='1-27')
    ap.add_argument('--procs', type=int, default=4)
    a = ap.parse_args()
    if a.fams == '1-27':
        fams = list(range(1, 28))
    else:
        fams = [int(x) for x in a.fams.split(',')]
    if a.stage == 'extract':
        from multiprocessing import Pool
        # largest first for load balance
        sizes = {F: os.path.getsize(f'{ROOT}/data/waveforms/fam{F}.h5') for F in fams}
        fams = sorted(fams, key=lambda F: -sizes[F])
        with Pool(a.procs) as p:
            p.map(extract, fams, chunksize=1)
    elif a.stage == 'combine':
        combine(fams)
    elif a.stage == 'analyse':
        analyse()
    elif a.stage == 'extras':
        extras()
    elif a.stage == 'healing_exc':
        healing_exceedance()
        healing_figure()
    elif a.stage == 'healing_fig':
        healing_figure()
    elif a.stage == 'cum_fig':
        cumulative_figure()
    elif a.stage == 'rate':
        rate_after_restart()
    elif a.stage == 'export':
        export()
