#!/usr/bin/env python
"""Are the LFE detections during teleseismic surface waves real LFEs or false detections?

Stages (run in order):
  metrics  : for EVERY catalog event, per-channel template CC in the S window (S-0.5..S+3 s, 2-8 Hz,
             lag search +-0.3 s) and the same statistic in 3 pre-P noise windows of the same trace
             (event-specific null). Per-channel arrays -> data/sci/remote_check/perchan/fam{F}.npz,
             per-event summary -> data/sci/remote_check/event_metrics.parquet
  raw      : long-period (0.02-0.2 Hz) vs 2-8 Hz energy from raw broadband (BK *HH*) miniSEED for
             candidate (remote-window) events and matched controls
  analyze  : classification, real/false fractions, recomputed triggering excess, figures, JSON

Template/data conventions: templates in data/templates/fam{F}_templates.h5 are 2-8 Hz stacks
(-15..+45 s around catalog time, 50 Hz) aligned at catalog time; data in data/waveforms/fam{F}.h5.
S pick per station from data/locations/fam{F}_3d_picks.csv (S_final, else S), relative to catalog time.
"""
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = 'data/sci/remote_check'
os.makedirs(f'{OUT}/perchan', exist_ok=True)
SR, PRE = 50.0, -15.0
TW = (-0.5, 3.0)            # S window rel. S pick
MAXLAG = 15                 # samples = 0.3 s
NOISE_STARTS = [-14.0, -11.5, -9.0]   # s rel catalog time, start of 3.5-s noise windows
SNR_MIN = 5.0               # template SNR to use a channel
SOS = butter(4, [2.0, 8.0], btype='band', fs=SR, output='sos')
EPOCH = pd.Timestamp('1970-01-01')
NPROC = 4


def idx(t):
    return int(round((t - PRE) * SR))


def load_catalog():
    d = pd.read_csv('data/shelly2026_MTJ_v2/LFEcatV2_MTJ_2008-2026.csv')
    d['t'] = (pd.to_datetime(dict(year=d['%year'], month=d.month, day=d.day)) - EPOCH).dt.total_seconds() + d.s_of_day
    return d


def s_picks(F):
    p = pd.read_csv(f'data/locations/fam{F}_3d_picks.csv')
    s = p.S_final.where(p.S_final.notna(), p.S.where(p.sig_S <= 0.1))
    P = p.P
    return {r.sta: (sv, pv) for r, sv, pv in zip(p.itertuples(), s, P) if np.isfinite(sv)}


def slide_cc(T, D):
    """T (L,), D (n, L+2*MAXLAG). returns cc (n, 2*MAXLAG+1) normalised."""
    L = len(T)
    W = np.lib.stride_tricks.sliding_window_view(D, L, axis=1)       # (n, 2M+1, L)
    T0 = T - T.mean()
    num = np.einsum('nkl,l->nk', W, T0)
    wm = W.mean(-1)
    wss = np.einsum('nkl,nkl->nk', W, W) - L * wm ** 2
    return num / (np.sqrt(np.maximum(wss, 1e-30)) * np.linalg.norm(T0))


def fam_metrics(F):
    t0 = time.time()
    h = h5py.File(f'data/waveforms/fam{F}.h5', 'r')
    tp = h5py.File(f'data/templates/fam{F}_templates.h5', 'r')
    te = pd.to_datetime(h['events/time_utc'][:].astype(str))
    nev = len(te)
    picks = s_picks(F)
    chans = []
    for cid in sorted(tp.keys()):
        sta = '.'.join(cid.split('.')[:2])
        if sta in picks and tp[cid].attrs['snr'] >= SNR_MIN and cid in h['chan']:
            chans.append(cid)
    nc = len(chans)
    A = {k: np.full((nev, nc), np.nan, np.float32) for k in
         ['cc', 'lag', 'ccs', 'cc0', 'snr'] + [f'n{j}_{q}' for j in range(3) for q in ('cc', 'lag', 'ccs', 'cc0')]}
    L = idx(TW[1]) - idx(TW[0])
    for c, cid in enumerate(chans):
        sta = '.'.join(cid.split('.')[:2])
        s, p = picks[sta]
        if s + TW[1] + 0.4 > 44.0 or s + TW[0] - 0.4 < -14.5:
            continue
        T = tp[cid]['stack'][idx(s + TW[0]): idx(s + TW[0]) + L].astype('float64')
        ev = h[f'chan/{cid}/ev'][:]
        lim = (p if np.isfinite(p) else s - 4.0) - 0.5
        lim = min(lim, s - 3.0)
        segs = [('', s + TW[0])] + [(f'n{j}_', ns) for j, ns in enumerate(NOISE_STARTS) if ns + 3.5 + 0.3 <= lim]
        dset = h[f'chan/{cid}/data']
        B = 2048
        for b0 in range(0, len(ev), B):
            X = sosfiltfilt(SOS, dset[b0:b0 + B].astype('float64'), axis=1)
            e = ev[b0:b0 + B]
            for pre, st in segs:
                i0 = idx(st) - MAXLAG
                cc = slide_cc(T, X[:, i0: i0 + L + 2 * MAXLAG])
                kp = np.argmax(cc, 1)
                # polarity: extremum of |cc| within +-3 samples (< quarter period), avoids cycle skips
                ka = np.argmax(np.abs(cc[:, MAXLAG - 3: MAXLAG + 4]), 1) + MAXLAG - 3
                r = np.arange(len(e))
                A[pre + 'cc0'][e, c] = cc[:, MAXLAG]
                A[pre + 'cc'][e, c] = cc[r, kp]
                A[pre + 'lag'][e, c] = (kp - MAXLAG) / SR
                A[pre + 'ccs'][e, c] = cc[r, ka]
            sig = X[:, idx(s + TW[0]): idx(s + TW[1])]
            nz = X[:, idx(-14.5): idx(min(lim, -5.0))]
            A['snr'][e, c] = np.sqrt((sig ** 2).mean(1) / ((nz ** 2).mean(1) + 1e-30))
    np.savez_compressed(f'{OUT}/perchan/fam{F}.npz', chans=np.array(chans), time_utc=np.array(te.astype(str)).astype('U32'), **A)
    print(f'fam {F}: {nev} ev, {nc} ch, {time.time() - t0:.0f} s', flush=True)
    return F


def summarize_fam(F):
    z = np.load(f'{OUT}/perchan/fam{F}.npz')
    chans = z['chans']
    sta = np.array(['.'.join(c.split('.')[:2]) for c in chans])
    usta = np.unique(sta)

    def sta_avg(X):
        return np.stack([np.nanmean(X[:, sta == s_], 1) for s_ in usta], 1)   # (n, nsta)

    def agg(pfx):
        cc, lag, ccs, cc0 = z[pfx + 'cc'], z[pfx + 'lag'], z[pfx + 'ccs'], z[pfx + 'cc0']
        ok = np.isfinite(cc)
        n = ok.sum(1)
        r = {}
        r['nch'] = n
        r['med_cc'] = np.nanmedian(cc, 1)
        r['frac02'] = np.where(n > 0, np.nansum(cc > 0.2, 1) / np.maximum(n, 1), np.nan)
        g = cc > 0.2
        r['nsta02'] = np.stack([(g[:, sta == s_]).any(1) for s_ in usta], 1).sum(1)
        r['n_good'] = g.sum(1)
        r['lag_in01'] = np.where(g.sum(1) > 0, np.sum(g & (np.abs(lag) <= 0.1), 1) / np.maximum(g.sum(1), 1), np.nan)
        L = np.where(g, lag, np.nan)
        r['lag_med'] = np.nanmedian(L, 1)
        r['lag_mad'] = np.nanmedian(np.abs(L - r['lag_med'][:, None]), 1)
        ga = np.abs(ccs) > 0.15
        r['n_abs015'] = ga.sum(1)
        r['frac_pos'] = np.where(ga.sum(1) > 0, np.sum(ga & (ccs > 0), 1) / np.maximum(ga.sum(1), 1), np.nan)
        r['mean_cc0'] = np.nanmean(cc0, 1)
        S0 = sta_avg(cc0)
        r['nsta'] = np.isfinite(S0).sum(1)
        r['sta_cc0'] = np.nanmean(S0, 1)
        return r, S0
    with np.errstate(all='ignore'):
        import warnings
        warnings.simplefilter('ignore')
        r, _ = agg('')
        out = pd.DataFrame(r)
        Sn = {}
        for j in range(3):
            if f'n{j}_cc' in z:
                rj, Sn[j] = agg(f'n{j}_')
                for k in ['med_cc', 'frac02', 'nsta02', 'lag_in01', 'frac_pos', 'n_abs015', 'n_good', 'mean_cc0', 'sta_cc0', 'nsta']:
                    out[f'n{j}_{k}'] = rj[k]
        # event-specific noise sd of station-averaged lag-0 CC (all noise windows), and Z scores
        alln = np.concatenate([Sn[j] for j in Sn], 1)
        sd = np.nanstd(alln, 1)
        out['noise_sd_sta'] = sd
        out['Z'] = out.sta_cc0 / (sd / np.sqrt(out.nsta))
        # null Z: noise window 0 scored against sd from windows 1,2 (independent of window 0)
        if 1 in Sn and 2 in Sn:
            sd12 = np.nanstd(np.concatenate([Sn[1], Sn[2]], 1), 1)
            out['Z_null'] = out.n0_sta_cc0 / (sd12 / np.sqrt(out.n0_nsta))
            out['Z_sig12'] = out.sta_cc0 / (sd12 / np.sqrt(out.nsta))
        out['med_snr'] = np.nanmedian(z['snr'], 1)
    out['fam'] = F
    out['row'] = np.arange(len(out))
    out['time_utc'] = z['time_utc']
    return out


def stage_metrics():
    fams = [int(a) for a in sys.argv[2:]] or list(range(1, 28))
    todo = [F for F in fams if not os.path.exists(f'{OUT}/perchan/fam{F}.npz')]
    # biggest first
    todo.sort(key=lambda F: -os.path.getsize(f'data/waveforms/fam{F}.h5'))
    with Pool(NPROC) as pool:
        for F in pool.imap_unordered(fam_metrics, todo):
            pass
    E = pd.concat([summarize_fam(F) for F in range(1, 28) if os.path.exists(f'{OUT}/perchan/fam{F}.npz')])
    E.to_parquet(f'{OUT}/event_metrics.parquet')
    print(E.describe().T.to_string())


# ---------------------------------------------------------------- shared: catalog <-> metrics, windows
def catalog_with_metrics():
    d = load_catalog()
    E = pd.read_parquet(f'{OUT}/event_metrics.parquet')
    E['t'] = (pd.to_datetime(E.time_utc) - EPOCH).dt.total_seconds()
    d['key'] = d.idnum.astype(int).astype(str) + '_' + np.round(d.t, 1).astype(str)
    E['key'] = E.fam.astype(int).astype(str) + '_' + np.round(E.t, 1).astype(str)
    E = E.drop(columns=['t']).drop_duplicates('key')
    m = d.merge(E, on='key', how='left')
    print('catalog rows', len(d), 'matched to metrics', m.Z.notna().sum(), flush=True)
    return m.sort_values('t').reset_index(drop=True)


def load_quakes():
    Q = pd.read_csv('data/sci/cascades2/remote_quakes.csv')
    Q['pgv_rank'] = Q.pgv_proxy_um_s.rank(ascending=False)
    return Q


def window_membership(t, Q, H=1.0, lag0=0.0):
    """index of quake whose [t_surf+lag0, t_surf+H h) window contains t, else -1"""
    qi = np.full(len(t), -1)
    # overlapping windows (aftershocks/doublets): assign to the quake with the largest PGV proxy
    for i in np.argsort(-Q.pgv_proxy_um_s.values):
        ts_ = Q.t_surf.values[i]
        m = (t >= ts_ + lag0 * 3600) & (t < ts_ + H * 3600)
        qi[m & (qi < 0)] = i
    return qi


def select_candidates_controls(m, Q, rng, ncon=5, pgv_min=100.0):
    """candidates: events in [t_surf, t_surf+1h) of quakes with PGV proxy > 100 um/s.
    controls: per candidate, ncon events of the same family, same year +-1, nearest cc/thr, not within
    -1 h..+12 h of any M>=6.5 (>300 km) surface-wave arrival, drawn without replacement."""
    t = m.t.values
    qi = window_membership(t, Q, 1.0)
    big = Q.pgv_proxy_um_s.values > pgv_min
    cand = m[(qi >= 0) & big[np.maximum(qi, 0)]].copy()
    cand['quake'] = qi[cand.index]
    cand['pgv'] = Q.pgv_proxy_um_s.values[cand.quake]
    cand['pgv_rank'] = Q.pgv_rank.values[cand.quake]
    cand['dt_surf_h'] = (cand.t - Q.t_surf.values[cand.quake]) / 3600
    excl = np.zeros(len(t), bool)
    for ts_ in Q.t_surf.values:
        excl |= (t > ts_ - 3600) & (t < ts_ + 12 * 3600)
    pool = m[~excl & m.Z.notna()]
    used = set()
    rows = []
    for _, c in cand.iterrows():
        g = pool[(pool.idnum == c.idnum) & (np.abs(pool['%year'] - c['%year']) <= 1)]
        g = g[~g.index.isin(used)]
        dd = np.abs(g['cc/thr'] - c['cc/thr']) + rng.random(len(g)) * 1e-3
        pick = g.loc[dd.nsmallest(ncon).index].copy()
        used.update(pick.index)
        pick['match_of'] = c.name
        rows.append(pick)
    ctrl = pd.concat(rows)
    return cand, ctrl


# ---------------------------------------------------------------- stage raw
def raw_one(args):
    from obspy import read, UTCDateTime
    key, fam, t = args
    ut = UTCDateTime(float(t))
    day = ut.strftime('%Y-%m-%d')
    dirs = [f'data/waveforms/allfam_raw/{ut.year}/{day}']
    if fam == 23:
        dirs.append(f'data/waveforms/fam23_raw/{ut.year}/{day}')
    files = []
    for d_ in dirs:
        files += glob.glob(f'{d_}/*.mseed')
    seen, out = set(), []
    for f in sorted(files):
        sta = os.path.basename(f)[:-6]
        if sta in seen:
            continue
        try:
            st = read(f, starttime=ut - 15, endtime=ut + 45)
        except Exception:
            continue
        st = st.select(channel='HH?')
        if not len(st):
            continue
        seen.add(sta)
        for tr in st:
            n_exp = int(60 * tr.stats.sampling_rate)
            if tr.stats.npts < 0.98 * n_exp:
                continue
            x = tr.data.astype('float64')
            x -= x.mean()
            n = len(x)
            w = np.ones(n)
            k = int(0.05 * n)
            w[:k] = 0.5 * (1 - np.cos(np.pi * np.arange(k) / k))
            w[-k:] = w[:k][::-1]
            X = np.abs(np.fft.rfft(x * w)) ** 2
            fr = np.fft.rfftfreq(n, tr.stats.delta)
            lp = X[(fr >= 0.02) & (fr <= 0.2)].sum()
            hf = X[(fr >= 2) & (fr <= 8)].sum()
            out.append(dict(key=key, chan=tr.id, log_lp_hf=np.log10(lp / hf), log_lp=0.5 * np.log10(lp / n),
                            log_hf=0.5 * np.log10(hf / n)))
    return out


def stage_raw():
    rng = np.random.default_rng(7)
    m = catalog_with_metrics()
    Q = load_quakes()
    cand, ctrl = select_candidates_controls(m, Q, rng)
    cand.to_csv(f'{OUT}/candidates_raw_sel.csv', index=False)
    ctrl.to_csv(f'{OUT}/controls_raw_sel.csv', index=False)
    allk = pd.concat([cand[['key', 'idnum', 't']], ctrl[['key', 'idnum', 't']]]).drop_duplicates('key')
    print('raw: n events', len(allk), flush=True)
    with Pool(NPROC) as pool:
        res = pool.map(raw_one, list(zip(allk.key, allk.idnum.astype(int), allk.t)), chunksize=4)
    R = pd.DataFrame([r for rr in res for r in rr])
    R.to_csv(f'{OUT}/raw_spectral_perchan.csv', index=False)
    print(R.describe().to_string())


# ---------------------------------------------------------------- stage analyze
CLS = dict(Zc=None, frac_pos_min=0.5, lag_min=None)


def classify(df, Zc, lag_min, pfx=''):
    Z = df[pfx + 'Z'] if pfx == '' else df['Z_null']
    return (Z >= Zc) & (df[pfx + 'frac_pos'] > CLS['frac_pos_min']) & (df[pfx + 'lag_in01'] >= lag_min)


def cluster_boot(cand, ctrl, fn, nb=2000, rng=None):
    """paired cluster bootstrap over quakes: resample quakes, take their candidates and matched controls"""
    qs = cand.quake.unique()
    cq = {q: cand.index[cand.quake == q] for q in qs}
    kq = {q: ctrl.index[ctrl.match_of.isin(cq[q])] for q in qs}
    out = []
    for _ in range(nb):
        s_ = rng.choice(qs, len(qs))
        a = cand.loc[np.concatenate([cq[q] for q in s_])]
        b = ctrl.loc[np.concatenate([kq[q] for q in s_])]
        out.append(fn(a, b))
    return np.array(out)


def excess_tests(dsub, Q, rng, label):
    import sci_cascades2 as C
    te = np.sort(dsub.t.values)
    Bo = np.sort(C.make_bursts(dsub).onset.values)
    t_lo, t_hi = C.ts(C.T_START), dsub.t.max()
    ctrl = Q.t_surf.values[:, None] + rng.choice(C.SHIFTS, (len(Q), 300))
    ctrl = np.where((ctrl > t_lo) & (ctrl < t_hi - 86400), ctrl, Q.t_surf.values[:, None] - rng.choice(np.abs(C.SHIFTS), (len(Q), 300)))

    def cnt(arr, a, b):
        return np.searchsorted(arr, b) - np.searchsorted(arr, a)
    res = {}
    for sub, sel in [('all_tele', np.ones(len(Q), bool)), ('top30_pgv', Q.pgv_rank.values <= 30),
                     ('pgv_gt_100um_s', Q.pgv_proxy_um_s.values > 100)]:
        for H in [0.5, 1]:
            for kind, arr in [('events', te), ('onsets', Bo)]:
                o = cnt(arr, Q.t_surf.values[sel], Q.t_surf.values[sel] + H * 3600).sum()
                cs = ctrl[sel]
                cc = cnt(arr, cs, cs + H * 3600)
                jj = rng.integers(0, cs.shape[1], (2000, len(cs)))
                nul = cc[np.arange(len(cs))[None, :], jj].sum(1)
                res[f'{label}|{sub}|{kind}|{H}h'] = dict(obs=int(o), expected=round(float(nul.mean()), 2),
                                                        null_ci95=[float(np.quantile(nul, .025)), float(np.quantile(nul, .975))],
                                                        ratio=round(float(o / nul.mean()), 3) if nul.mean() > 0 else None,
                                                        p_upper=round(float((np.sum(nul >= o) + 1) / (len(nul) + 1)), 4))
    return res


def stage_analyze():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import warnings
    warnings.simplefilter('ignore')
    rng = np.random.default_rng(2026)
    m = catalog_with_metrics()
    Q = load_quakes()
    summ = {}
    ok = m.Z.notna() & m.Z_null.notna()
    # ---- calibrate classification thresholds on the event-specific noise-window null
    Zc = float(np.nanpercentile(m.loc[ok, 'Z_null'], 99))
    lag_min = float(np.nanmedian(m.loc[ok, 'n0_lag_in01']))
    CLS.update(Zc=Zc, lag_min=lag_min)
    m['real'] = classify(m, Zc, lag_min).astype(float)
    m.loc[~ok, 'real'] = np.nan
    null_real = classify(m[ok], Zc, lag_min, pfx='n0_')
    summ['classifier'] = dict(rule='real = Z>=Zc & frac_pos>0.5 & lag_in01>=lag_min; Z = station-averaged lag-0 template CC in S window / '
                              '(event-specific noise sd/sqrt(nsta)); thresholds from pre-P noise windows of the same traces',
                              Zc_99pct_null=Zc, lag_min_median_null=lag_min,
                              false_positive_rate_on_noise_windows=float(null_real.mean()),
                              frac_real_whole_catalog=float(m.loc[ok, 'real'].mean()),
                              n_classified=int(ok.sum()), n_catalog=len(m))
    # recall vs cc/thr in whole catalog
    bins = [1, 1.1, 1.2, 1.35, 1.5, 1.75, 2, 2.5, 3, 10]
    m['cbin'] = pd.cut(m['cc/thr'], bins)
    summ['frac_real_by_ccthr'] = {str(k): round(float(v), 3) for k, v in m[ok].groupby('cbin', observed=True).real.mean().items()}
    print(json.dumps(summ, indent=1), flush=True)

    # another family detected within +-5 s (possible cross-family duplicate of one source)
    tt, ff = m.t.values, m.idnum.values
    dup = np.zeros(len(m), bool)
    for sh in range(1, 40):
        dt_ = np.abs(tt[sh:] - tt[:-sh])
        hit = (dt_ <= 5) & (ff[sh:] != ff[:-sh])
        dup[sh:] |= hit
        dup[:-sh] |= hit
    m['dup5'] = dup
    # ---- candidates and cc/thr-matched controls (all quakes; subsets below)
    cand, ctrl = select_candidates_controls(m[ok], Q, rng, ncon=5, pgv_min=0.0)
    ctrl['quake'] = cand.quake.reindex(ctrl.match_of).values
    ctrl['pgv_rank'] = cand.pgv_rank.reindex(ctrl.match_of).values
    # raw spectral metrics (computed for PGV>100 quakes and their own controls)
    R = pd.read_csv(f'{OUT}/raw_spectral_perchan.csv')
    R = R[np.isfinite(R[['log_lp_hf', 'log_lp', 'log_hf']]).all(1)]
    Rc = pd.read_csv(f'{OUT}/controls_raw_sel.csv')
    # station-matched LP ratio: per channel, subtract the median log ratio of the raw-control events on that channel
    rk = set(Rc.key)
    for k in ['log_lp_hf', 'log_lp', 'log_hf']:
        base = R[R.key.isin(rk)].groupby('chan')[k].median()
        R['d' + k] = R[k] - R.chan.map(base)
    Re = R.groupby('key').agg(log_lp_hf=('log_lp_hf', 'median'), dlog_lp_hf=('dlog_lp_hf', 'median'),
                              dlog_lp=('dlog_lp', 'median'), dlog_hf=('dlog_hf', 'median'), n_hh=('chan', 'size'))
    for D_ in (cand, ctrl, m):
        for c in Re.columns:
            D_[c] = D_.key.map(Re[c])

    metrics = ['Z', 'med_cc', 'frac02', 'nsta02', 'lag_in01', 'lag_mad', 'frac_pos', 'med_snr', 'dlog_lp_hf', 'log_lp_hf', 'dlog_lp', 'dlog_hf']
    comp = {}
    for lab, sel in [('top30_1h', cand.pgv_rank <= 30), ('top30_0.5h', (cand.pgv_rank <= 30) & (cand.dt_surf_h < 0.5)),
                     ('pgv100_1h', cand.pgv > 100), ('all_tele_1h', cand.pgv > 0)]:
        a = cand[sel]
        b = ctrl[ctrl.match_of.isin(a.index)]
        r = dict(n_cand=len(a), n_ctrl=len(b), n_quakes=int(a.quake.nunique()),
                 median_ccthr_cand=float(a['cc/thr'].median()), median_ccthr_ctrl=float(b['cc/thr'].median()))
        r['frac_real_cand'] = float(a.real.mean())
        r['frac_real_ctrl'] = float(b.real.mean())
        bt = cluster_boot(a, b, lambda x, y: x.real.mean() - y.real.mean(), rng=rng)
        r['diff_real_ci95'] = [float(np.nanquantile(bt, .025)), float(np.nanquantile(bt, .975))]
        # permutation within matched sets: swap candidate with one of its controls
        r['fisher_p'] = float(stats.fisher_exact([[a.real.sum(), (~a.real.astype(bool)).sum()],
                                                  [b.real.sum(), (~b.real.astype(bool)).sum()]])[1])
        # upper bound on the admixture of pure-noise (false) detections: Z_cand = (1-f) Z_ctrl + f * 0
        fz = cluster_boot(a, b, lambda u, v: 1 - u.Z.mean() / v.Z.mean(), rng=rng)
        r['false_admixture_upper_from_meanZ'] = dict(est=float(1 - a.Z.mean() / b.Z.mean()),
                                                     ci95=[float(np.nanquantile(fz, .025)), float(np.nanquantile(fz, .975))])
        r['frac_with_otherfam_within5s_cand'] = float(a.dup5.mean())
        r['frac_with_otherfam_within5s_ctrl'] = float(b.dup5.mean())
        for k in metrics:
            x, y = a[k].dropna(), b[k].dropna()
            if len(x) < 3 or len(y) < 3:
                continue
            bt = cluster_boot(a, b, lambda u, v, k=k: np.nanmedian(u[k]) - np.nanmedian(v[k]), nb=1000, rng=rng)
            r[k] = dict(med_cand=round(float(x.median()), 3), med_ctrl=round(float(y.median()), 3),
                        diff_ci95=[round(float(np.nanquantile(bt, .025)), 3), round(float(np.nanquantile(bt, .975)), 3)],
                        mwu_p=float(stats.mannwhitneyu(x, y).pvalue), n=[len(x), len(y)])
        comp[lab] = r
    summ['cand_vs_ctrl'] = comp
    # null reference values of the per-event metrics (noise windows)
    summ['noise_window_medians'] = {k: float(np.nanmedian(m.loc[ok, 'n0_' + k])) for k in
                                    ['med_cc', 'frac02', 'lag_in01', 'frac_pos', 'nsta02']}
    summ['noise_window_medians']['Z_null'] = float(np.nanmedian(m.loc[ok, 'Z_null']))
    # within candidates: does long-period energy predict low Z?
    c100 = cand[(cand.pgv > 100) & cand.dlog_lp_hf.notna()]
    summ['cand_spearman_Z_vs_dlog_lp_hf'] = [float(v) for v in stats.spearmanr(c100.Z, c100.dlog_lp_hf)] + [len(c100)]
    # per-quake table (top 30)
    cols = ['quake', 'idnum', 'time_utc', 'dt_surf_h', 'cc/thr', 'meancc', 'NSta', 'Z', 'med_cc', 'frac02', 'nsta02', 'lag_in01',
            'frac_pos', 'med_snr', 'log_lp_hf', 'dlog_lp_hf', 'dlog_lp', 'dlog_hf', 'n_hh', 'real', 'pgv', 'pgv_rank']
    out = cand[cols].copy()
    out['quake_time'] = Q.time.values[out.quake]
    out['quake_place'] = Q.place.values[out.quake]
    out['mag'] = Q.mag.values[out.quake]
    out.index.name = 'cand_id'
    out.sort_values(['pgv_rank', 'time_utc']).to_csv(f'{OUT}/candidates.csv', index=True)
    ctrl[['match_of', 'quake', 'idnum', 'time_utc', 'cc/thr', 'meancc', 'Z', 'med_cc', 'frac02', 'nsta02', 'lag_in01', 'frac_pos',
          'med_snr', 'log_lp_hf', 'dlog_lp_hf', 'real']].to_csv(f'{OUT}/controls.csv', index=False)
    pq = cand[cand.pgv_rank <= 30].groupby('quake').agg(n=('Z', 'size'), n_real=('real', 'sum'), med_Z=('Z', 'median'),
                                                        med_ccthr=('cc/thr', 'median'), n_fam=('idnum', 'nunique'),
                                                        med_dlog_lp_hf=('dlog_lp_hf', 'median'))
    pq['place'] = Q.place.values[pq.index]
    pq['mag'] = Q.mag.values[pq.index]
    pq['ctrl_frac_real'] = [ctrl[ctrl.quake == q].real.mean() for q in pq.index]
    print(pq.round(2).to_string())
    summ['per_quake_top30'] = pq.reset_index().round(3).to_dict('records')

    # ---- recompute triggering excess with real-only / false-only catalogs (cascades2 null)
    import sci_cascades2 as C
    C.T_END = m.t.max()
    base_cat = m[m.t >= C.ts(C.T_START)]
    ex = {}
    for lab, dsub in [('all', base_cat), ('classified', base_cat[base_cat.real.notna()]),
                      ('real', base_cat[base_cat.real == 1]), ('false', base_cat[base_cat.real == 0]),
                      ('thin_meancc013', base_cat[base_cat.meancc >= 0.13]),
                      ('real_Zge3', base_cat[base_cat.Z >= 3])]:
        ex.update(excess_tests(dsub[['t', 'idnum']], Q, np.random.default_rng(44), lab))
    summ['excess'] = ex
    for k, v in ex.items():
        if '|1h' in k or 'top30' in k:
            print(k, v)
    json.dump(summ, open(f'{OUT}/summary.json', 'w'), indent=1, default=float)

    # ---- figure 1: metric distributions
    a = cand[cand.pgv_rank <= 30]
    b = ctrl[ctrl.match_of.isin(a.index)]
    a_all = cand
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    def hist(axx, k, null=None, bins=30, xl=None):
        rngx = np.nanpercentile(np.r_[a_all[k].dropna(), b[k].dropna()], [0.5, 99.5])
        bb = np.linspace(*rngx, bins)
        axx.hist(b[k].dropna(), bb, density=True, color='0.6', alpha=.7, label=f'cc/thr-matched controls (n={b[k].notna().sum()})')
        axx.hist(a_all[k].dropna(), bb, density=True, histtype='step', color='C0', lw=1.5, label=f'remote windows, all quakes (n={a_all[k].notna().sum()})')
        axx.hist(a[k].dropna(), bb, density=True, histtype='step', color='C3', lw=2, label=f'remote windows, top-30 PGV (n={a[k].notna().sum()})')
        if null is not None:
            axx.hist(m.loc[ok, null].dropna().sample(20000, random_state=1), bb, density=True, histtype='step', color='k', ls='--',
                     label='noise-window null')
        axx.set_xlabel(xl or k)
    hist(ax[0, 0], 'Z', 'Z_null', xl='Z: station-avg lag-0 S-window CC / noise sd')
    ax[0, 0].axvline(Zc, color='k', lw=.8)
    ax[0, 0].legend(fontsize=7)
    hist(ax[0, 1], 'med_cc', 'n0_med_cc', xl='median channel CC (max over +-0.3 s)')
    hist(ax[0, 2], 'frac02', 'n0_frac02', xl='fraction of channels CC>0.2')
    hist(ax[1, 0], 'lag_in01', 'n0_lag_in01', xl='fraction of CC>0.2 channels with |lag|<=0.1 s')
    hist(ax[1, 1], 'frac_pos', 'n0_frac_pos', xl='fraction positive polarity (|CC|>0.15)')
    x = ax[1, 2]
    rc = pd.read_csv(f'{OUT}/controls_raw_sel.csv').key
    cr = m[m.key.isin(set(rc))]
    x.scatter(cr.dlog_lp_hf, cr.Z, s=6, c='0.6', label='matched controls')
    c1 = cand[cand.pgv > 100]
    sc = x.scatter(c1.dlog_lp_hf, c1.Z, s=18, c=np.log10(c1.pgv), cmap='plasma', label='remote windows (PGV>100)')
    plt.colorbar(sc, ax=x, label='log10 PGV proxy (um/s)')
    x.axhline(Zc, color='k', lw=.8)
    x.set_xlabel('log10 (0.02-0.2 Hz / 2-8 Hz energy), rel. channel median of controls')
    x.set_ylabel('Z')
    x.legend(fontsize=7)
    fig.suptitle('Template checks of detections during teleseismic surface waves vs cc/thr-matched controls')
    fig.tight_layout()
    fig.savefig('figures/sci_remote_check_metrics.png', dpi=120)
    plt.close(fig)
    m.drop(columns=['cbin']).to_parquet(f'{OUT}/catalog_classified.parquet')


# ---------------------------------------------------------------- figure 2: examples
def event_traces(F, time_utc, nch=8):
    h = h5py.File(f'data/waveforms/fam{F}.h5', 'r')
    tp = h5py.File(f'data/templates/fam{F}_templates.h5', 'r')
    te = h['events/time_utc'][:].astype(str)
    row = int(np.where(te == pd.Timestamp(time_utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:22])[0][0]) if False else None
    tt = pd.to_datetime(te)
    row = int(np.argmin(np.abs((tt - pd.Timestamp(time_utc)).total_seconds())))
    z = np.load(f'{OUT}/perchan/fam{F}.npz')
    chans = list(z['chans'])
    picks = s_picks(F)
    have = [c for j, c in enumerate(chans) if np.isfinite(z['cc0'][row, j])]
    have.sort(key=lambda c: -tp[c].attrs['snr'])
    out = []
    for cid in have[:nch]:
        ev = h[f'chan/{cid}/ev'][:]
        k = int(np.where(ev == row)[0][0])
        x = sosfiltfilt(SOS, h[f'chan/{cid}/data'][k].astype('float64'))
        s_, _ = picks['.'.join(cid.split('.')[:2])]
        out.append(dict(cid=cid, s=s_, x=x, T=tp[cid]['stack'][:], cc0=float(z['cc0'][row, chans.index(cid)])))
    return out


def stage_fig2():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from obspy import read, UTCDateTime
    C_ = pd.read_csv(f'{OUT}/candidates.csv')
    K = pd.read_csv(f'{OUT}/controls.csv')
    top = C_[C_.pgv_rank <= 30]
    ex_false = top[top.real == 0].sort_values('cc/thr', ascending=False).iloc[0]
    ex_real = top[top.real == 1].sort_values('Z', ascending=False).iloc[0] if (top.real == 1).any() else None
    ctl = K[(K.real == 1)].iloc[(K[K.real == 1]['cc/thr'] - ex_false['cc/thr']).abs().argsort().iloc[0]]
    exs = [('remote, classified FALSE', ex_false), ('remote, classified REAL', ex_real), ('control, classified REAL', ctl)]
    exs = [e for e in exs if e[1] is not None]
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, len(exs), height_ratios=[2.2, 1])
    t_ax = np.arange(3000) / SR + PRE
    for j, (lab, e) in enumerate(exs):
        ax = fig.add_subplot(gs[0, j])
        tr = event_traces(int(e.idnum), e.time_utc)
        for i, d_ in enumerate(tr):
            w = (t_ax >= d_['s'] - 2) & (t_ax <= d_['s'] + 5)
            x = d_['x'][w] / (np.abs(d_['x'][w]).max() + 1e-20)
            T = d_['T'][w] / (np.abs(d_['T'][w]).max() + 1e-20)
            tt = t_ax[w] - d_['s']
            ax.plot(tt, x * 0.45 - i, 'k', lw=.6)
            ax.plot(tt, T * 0.45 - i, 'C3', lw=.8, alpha=.8)
            ax.text(5.1, -i, f"{d_['cid']}\nCC0={d_['cc0']:.2f}", fontsize=6, va='center')
        ax.axvspan(TW[0], TW[1], color='y', alpha=.15)
        ax.set_xlim(-2, 7)
        ax.set_yticks([])
        ax.set_xlabel('time rel. S pick (s); black data, red template (2-8 Hz)')
        ax.set_title(f"{lab}\nfam {int(e.idnum)} {str(e.time_utc)[:19]} cc/thr={e['cc/thr']:.2f}\nZ={e.Z:.1f} frac_pos={e.frac_pos:.2f} "
                     f"lag_in01={e.lag_in01:.2f}", fontsize=8)
    # bottom: raw broadband around Chignik with detection times
    ax = fig.add_subplot(gs[1, :])
    Q = load_quakes()
    qi = int(top.loc[top.quake_place.str.contains('Chignik')].quake.iloc[0]) if top.quake_place.str.contains('Chignik').any() else int(top.quake.iloc[0])
    q = Q.iloc[qi]
    t0 = UTCDateTime(float(q.t_surf)) - 600
    day = t0.strftime('%Y-%m-%d')
    st = None
    for sta in ['NC.KCPB', 'NC.KMPB', 'BK.HAYF', 'BK.RBOW']:
        f = f'data/waveforms/allfam_raw/{t0.year}/{day}/{sta}.mseed'
        if os.path.exists(f):
            st = read(f).select(channel='HHZ')
            if len(st):
                break
    if st is not None and len(st):
        segs = []
        for tr_ in st:
            tr2 = tr_.copy()
            tr2.detrend('demean')
            tr2.taper(0.05)
            tlo = tr2.copy().filter('bandpass', freqmin=0.02, freqmax=0.2, corners=2, zerophase=True)
            thi = tr2.copy().filter('bandpass', freqmin=2, freqmax=8, corners=4, zerophase=True)
            segs.append(((tr_.times('timestamp') - q.t_surf) / 60, tlo.data, thi.data))
        slo = np.percentile(np.abs(np.concatenate([g[1] for g in segs])), 99.5)
        shi = np.percentile(np.abs(np.concatenate([g[2] for g in segs])), 99.5)
        for tm, lo, hi in segs:
            ax.plot(tm, lo / slo + 1.2, 'C0', lw=.6)
            ax.plot(tm, hi / shi - 1.2, 'k', lw=.3)
        ax.set_ylabel(f'{st[0].id} (normalised)\nblue 0.02-0.2 Hz\nblack 2-8 Hz', fontsize=7)
        ax.set_yticks([])
    ev = C_[C_.quake == qi]
    for _, e in ev.iterrows():
        ax.axvline(e.dt_surf_h * 60, color='C2' if e.real == 1 else 'C3', lw=.8, alpha=.8)
    ax.set_xlim(-10, 60)
    ax.set_xlabel('minutes after surface-wave arrival (4 km/s); green/red lines = detections classified real/false')
    ax.set_title(f"{q.place} M{q.mag} {str(q.time)[:16]}: raw data are only available in +-15/45 s windows around detections", fontsize=9)
    fig.tight_layout()
    fig.savefig('figures/sci_remote_check_examples.png', dpi=120)
    plt.close(fig)


if __name__ == '__main__':
    st = sys.argv[1]
    if st == 'metrics':
        stage_metrics()
    elif st == 'raw':
        stage_raw()
    elif st == 'analyze':
        stage_analyze()
    elif st == 'fig2':
        stage_fig2()
