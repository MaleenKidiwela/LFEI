#!/usr/bin/env python3
"""In-patch LFE dynamics from split-station (A/B) relocations.

Tasks
 1. Burst-front direction inside patches (early->late centroid, cross-estimated A/B), vs network sweep direction.
 2. Core vs halo: isolated (no same-family catalog event within +-1 h) vs in-burst events, matched on mean_cc terciles.
 3. Tidal (M2) centroid shift, along-strike / down-dip, vs burst-wise phase-randomised null.
 4. D^2(tau) structure function <(xA_i-xA_j).(xB_i-xB_j)> vs time lag, with time-permutation nulls.

Outputs: data/sci/patch_dynamics/*.csv, summary.json; figures/sci_patch_dynamics_*.png
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT = 'data/sci/patch_dynamics'
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)
SDH = float(os.environ.get('SDH', 0.5))          # quality cut sd_x,sd_y < SDH, sd_z < 2*SDH in BOTH halves
TAG = '' if SDH == 0.5 else f'_sd{SDH}'
M2_S = 12.4206012 * 3600
TREF = pd.Timestamp('2000-01-01T12:00:00')
GAP = 2 * 3600                                    # burst gap (same as burst_cascades.py)
EP_GAP, MINFAM = 6 * 3600, 6
RESOLVED = [1, 6, 8, 12, 15, 16, 18, 20, 21, 23, 24]
NPERM = int(os.environ.get('NPERM', 1000))
ALLT = ALLF = XS = None   # sorted catalog times / families, family along-strike coords (set in main)

# ---------------- geometry ----------------
AZ = np.radians(102.0)
U_S = np.array([np.sin(AZ), np.cos(AZ), 0.0])                     # along-strike (E,N,down), toward ESE
DIP, DIPDIR = np.radians(46.0), np.radians(24.4)                 # Shelly centroid plane (fault_geometry.json)
d0 = np.array([np.cos(DIP) * np.sin(DIPDIR), np.cos(DIP) * np.cos(DIPDIR), np.sin(DIP)])
U_D = d0 - (d0 @ U_S) * U_S
U_D /= np.linalg.norm(U_D)                                        # down-dip (in plane, orthogonal to U_S)
U_N = np.cross(U_S, U_D)                                          # plane normal
R = np.vstack([U_S, U_D, U_N])                                    # rotates (E,N,down) -> (s,d,n)


def tsec(x):
    return (pd.to_datetime(x, format='mixed') - pd.Timestamp('1970-01-01')).dt.total_seconds().values


def load_catalog():
    d = pd.read_csv('data/shelly2026_MTJ_v2/LFEcatV2_MTJ_2008-2026.csv')
    d['t'] = (pd.to_datetime(dict(year=d['%year'], month=d.month, day=d.day)) - pd.Timestamp('1970-01-01')).dt.total_seconds() + d.s_of_day
    return d[['idnum', 't']].rename(columns={'idnum': 'fam'})


def family_positions():
    lat0, lon0 = 40.06, -123.625
    pos = {}
    for F in range(1, 28):
        j = json.load(open(f'data/locations/fam{F}_3d.json'))
        pos[F] = np.array([(j['lon'] - lon0) * 111.19 * np.cos(np.radians(lat0)), (j['lat'] - lat0) * 111.19])
    return pos


def catalog_bursts(cat):
    """Catalog family bursts (2 h gap) and network episode ids exactly as in burst_cascades.py."""
    on = []
    for F, g in cat.groupby('fam'):
        t = np.sort(g.t.values)
        st = np.r_[0, np.where(np.diff(t) > GAP)[0] + 1]
        en = np.r_[st[1:], len(t)]
        for a, b in zip(st, en):
            on.append((F, t[a], t[b - 1], b - a))
    on = pd.DataFrame(on, columns=['fam', 'onset', 'end', 'n']).sort_values('onset').reset_index(drop=True)
    on['ep'] = np.r_[0, np.cumsum(np.diff(on.onset.values) > EP_GAP)]
    on['bid'] = np.arange(len(on))
    return on


def load_family(F, cat_t, cb, sdh=None):
    SDH_ = SDH if sdh is None else sdh
    A = pd.read_csv(f'data/relocations/fam{F}_events_3d_halfA.csv')
    B = pd.read_csv(f'data/relocations/fam{F}_events_3d_halfB.csv')
    m = A.merge(B.drop(columns='time_utc'), on='ev', suffixes=('_a', '_b'))
    m = m[(m.sd_x_a < SDH_) & (m.sd_y_a < SDH_) & (m.sd_z_a < 2 * SDH_) & (m.sd_x_b < SDH_) & (m.sd_y_b < SDH_) & (m.sd_z_b < 2 * SDH_)].copy()
    if len(m) == 0:
        return None
    m['fam'] = F
    m['t'] = tsec(m.time_utc)
    m = m.sort_values('t').reset_index(drop=True)
    xa = m[['dx_a', 'dy_a', 'dz_a']].values * 1000 @ R.T          # metres, (s,d,n)
    xb = m[['dx_b', 'dy_b', 'dz_b']].values * 1000 @ R.T
    for k, c in enumerate('sdn'):
        m[f'{c}_a'], m[f'{c}_b'] = xa[:, k], xb[:, k]
    m['cc'] = 0.5 * (m.mean_cc_a + m.mean_cc_b)
    # catalog neighbours within +-1 h (catalog is a superset; relocated time ~ catalog time - O(10 s))
    ct = np.sort(cat_t)
    lo, hi = np.searchsorted(ct, m.t.values - 3600 - 30), np.searchsorted(ct, m.t.values + 3600 + 30)
    near = np.abs(ct[np.clip(np.searchsorted(ct, m.t.values), 1, len(ct) - 1)[:, None] + np.array([-1, 0])] - m.t.values[:, None]).min(1)
    m['dt_cat'] = near
    jn = np.clip(np.searchsorted(ct, m.t.values), 1, len(ct) - 1)
    jn = np.where(np.abs(ct[jn - 1] - m.t.values) < np.abs(ct[jn] - m.t.values), jn - 1, jn)
    m['t_cat'] = ct[jn]                                            # matched catalog time
    m['n_any10s'] = (np.searchsorted(ALLT, m.t_cat.values + 10, side='right') - np.searchsorted(ALLT, m.t_cat.values - 10)) - 1
    i0, i1 = np.searchsorted(ALLT, m.t_cat.values - 10), np.searchsorted(ALLT, m.t_cat.values + 10, side='right')
    m['ov_east'] = [np.sum(np.sign(XS[ALLF[a:b]][ALLF[a:b] != F] - XS[F])) for a, b in zip(i0, i1)]   # net east-ness of overlapping other-family LFEs
    m['n_nb1h'] = hi - lo - 1                                      # minus the event itself
    # catalog burst membership (2 h gap bursts of the same family); +-120 s slack for origin-time shifts
    g = cb[cb.fam == F].sort_values('onset')
    i = np.searchsorted(g.onset.values - 120, m.t.values, side='right') - 1
    ok = (i >= 0) & (m.t.values <= g.end.values[np.clip(i, 0, None)] + 120)
    m['bid'] = np.where(ok, g.bid.values[np.clip(i, 0, None)], -1)
    m['burst_n_cat'] = np.where(ok, g.n.values[np.clip(i, 0, None)], 0)
    m['ep'] = np.where(ok, g.ep.values[np.clip(i, 0, None)], -1)
    return m


# =================================================================== Task 1
def task1(D, E, on, pos, label='base', nmin=8):
    Xs = {F: pos[F] @ U_S[:2] for F in pos}                       # along-strike coordinate of families (km)
    Xmid = np.median(list(Xs.values()))
    Ep = E.set_index('ep')
    rows, orders = [], []
    for (F, b), g in D[D.bid >= 0].groupby(['fam', 'bid']):
        if len(g) < nmin:
            continue
        g = g.sort_values('t')
        rows.append(dict(fam=F, bid=b, ep=int(g.ep.iloc[0]), n=len(g), t0=g.t.iloc[0], dur_h=(g.t.iloc[-1] - g.t.iloc[0]) / 3600))
        orders.append((g[['s_a', 'd_a', 'n_a']].values, g[['s_b', 'd_b', 'n_b']].values))
    T = pd.DataFrame(rows)
    if len(T) < 5:
        return T, dict(variant=label, nmin=nmin, n_bursts=len(T), n_events_in=len(D), note='too few bursts'), None

    def disp(a, b, perm=None):
        n = len(a)
        if perm is not None:
            a, b = a[perm], b[perm]
        h = n // 2
        e, l = slice(0, h), slice(n - h, n)                        # early half / late half (odd n: middle event dropped)
        cross = 0.5 * ((b[l].mean(0) - a[e].mean(0)) + (a[l].mean(0) - b[e].mean(0)))
        dA = a[l].mean(0) - a[e].mean(0)
        dB = b[l].mean(0) - b[e].mean(0)
        return cross, dA, dB

    res = np.array([disp(a, b) for a, b in orders])               # (nb, 3, 3)
    T['ds'], T['dd'], T['dn'] = res[:, 0, 0], res[:, 0, 1], res[:, 0, 2]
    T['dAs'], T['dBs'] = res[:, 1, 0], res[:, 2, 0]
    T['dAB_dot'] = np.sum(res[:, 1, :2] * res[:, 2, :2], 1)        # (s,d) in-plane true |d|^2 estimator
    # sweep directions
    gsign, lsign, sig, lead = [], [], [], []
    for _, r in T.iterrows():
        if r.ep in Ep.index:
            e = Ep.loc[r.ep]
            lead.append(1.0 if Xs[int(e.first_fam)] < Xmid else -1.0)   # +1 west(NW)-led -> expected ESE (+s) front
            gsign.append(np.sign(np.array([e.sx_h_per_km, e.sy_h_per_km]) @ U_S[:2]))
            sig.append(e.p_perm < 0.05)
            g = on[on.ep == r.ep].sort_values('onset').drop_duplicates('fam')
            x = np.array([Xs[f] for f in g.fam]); t = (g.onset.values - g.onset.min()) / 3600
            sel = np.abs(x - Xs[r.fam]) <= 4.0
            if sel.sum() >= 3 and np.ptp(x[sel]) > 1.0:
                lsign.append(np.sign(np.polyfit(x[sel], t[sel], 1)[0]))
            else:
                lsign.append(np.nan)
        else:
            gsign.append(np.nan); lsign.append(np.nan); sig.append(False); lead.append(np.nan)
    T['sweep_global'], T['sweep_local'], T['ep_sig'], T['leader_side'] = gsign, lsign, sig, lead
    T['Xs_km'] = [Xs[f] for f in T.fam]
    T.to_csv(f'{OUT}/task1_bursts_{label}.csv', index=False)

    def stats_of(res_cross, Tsub_idx=None):
        ds = res_cross[:, 0]
        out = dict(mean_ds=ds.mean(), frac_pos=(ds > 0).mean())
        ang = np.arctan2(res_cross[:, 0], res_cross[:, 1])         # direction in (s,d) plane
        out['rayleigh_R'] = np.hypot(np.cos(ang).mean(), np.sin(ang).mean())
        L = T.leader_side.values
        out['flip_west_minus_east'] = np.mean(ds[L == 1]) - np.mean(ds[L == -1])
        dsm = ds - pd.Series(ds).groupby(T.fam.values).transform('mean').values   # removes per-family fixed biases
        out['flip_famdemeaned'] = np.mean(dsm[L == 1]) - np.mean(dsm[L == -1])
        out['proj_local_famdemeaned'] = np.nanmean(dsm * T.sweep_local.values)
        for k in ['sweep_global', 'sweep_local', 'leader_side']:
            sw = T[k].values
            ok = np.isfinite(sw)
            out[f'agree_{k}'] = np.mean(np.sign(ds[ok]) == sw[ok])
            out[f'proj_{k}'] = np.mean(ds[ok] * sw[ok])
        return out

    obs = stats_of(res[:, 0])
    null = []
    for _ in range(NPERM):
        rp = np.array([disp(a, b, rng.permutation(len(a)))[0] for a, b in orders])
        null.append(stats_of(rp))
    null = pd.DataFrame(null)
    S = dict(variant=label, nmin=nmin, n_bursts=len(T), n_matched_episode=int(np.isfinite(T.leader_side).sum()),
             n_west_led=int((T.leader_side == 1).sum()), n_east_led=int((T.leader_side == -1).sum()), n_events=int(T.n.sum()), n_fams=int(T.fam.nunique()),
             median_n=float(T.n.median()), median_dur_h=float(T.dur_h.median()))
    for k, v in obs.items():
        nv = null[k].values
        if k.startswith('rayleigh'):
            p = (np.sum(nv >= v) + 1) / (len(nv) + 1)
        else:
            p = (np.sum(np.abs(nv - nv.mean()) >= abs(v - nv.mean())) + 1) / (len(nv) + 1)
        S[k] = dict(obs=float(v), null_mean=float(nv.mean()), null_sd=float(nv.std()), p_shuffle=float(p))
    ds = T.ds.values
    S['mean_ds_m_se'] = float(ds.std(ddof=1) / np.sqrt(len(ds)))
    S['sign_test_p'] = float(stats.binomtest(int((ds > 0).sum()), len(ds)).pvalue)
    ang = np.arctan2(T.ds, T.dd)
    Rb = np.hypot(np.cos(ang).mean(), np.sin(ang).mean())
    S['rayleigh_analytic_p'] = float(np.exp(-len(ang) * Rb ** 2))
    # true-migration amplitude: <dA.dB> over bursts (independent errors -> unbiased |d_true|^2)
    ab = T.dAB_dot.values
    S['true_disp_sq_m2'] = dict(mean=float(ab.mean()), se=float(ab.std(ddof=1) / np.sqrt(len(ab))),
                                rms_m=float(np.sqrt(max(ab.mean(), 0))))
    nab = []
    for _ in range(min(NPERM, 500)):
        rp = np.array([disp(a, b, rng.permutation(len(a))) for a, b in orders])
        nab.append(np.mean(np.sum(rp[:, 1, :2] * rp[:, 2, :2], 1)))
    nab = np.array(nab)
    S['true_disp_sq_m2'].update(null_mean=float(nab.mean()), null_sd=float(nab.std()),
                                p_shuffle=float((np.sum(nab >= ab.mean()) + 1) / (len(nab) + 1)))
    # sweep agreement with simple binomial on matched bursts, sig episodes only, and correlation with family position
    w, e_ = T.ds[T.leader_side == 1].values, T.ds[T.leader_side == -1].values
    if len(w) > 1 and len(e_) > 1:
        S['flip_welch'] = dict(mean_west_led=float(w.mean()), se_west=float(w.std(ddof=1) / np.sqrt(len(w))),
                               mean_east_led=float(e_.mean()), se_east=float(e_.std(ddof=1) / np.sqrt(len(e_))),
                               p=float(stats.ttest_ind(w, e_, equal_var=False).pvalue))
    for k in ['sweep_global', 'sweep_local', 'leader_side']:
        for lab, sub in [('all', T), ('sig_ep', T[T.ep_sig])]:
            ok = np.isfinite(sub[k].values) & (sub.ds.values != 0)
            n = int(ok.sum())
            if n:
                a = int((np.sign(sub.ds.values[ok]) == sub[k].values[ok]).sum())
                S[f'{k}_{lab}'] = dict(n=n, n_agree=a, frac=a / n, binom_p=float(stats.binomtest(a, n).pvalue),
                                       mean_proj_m=float(np.mean(sub.ds.values[ok] * sub[k].values[ok])),
                                       se_proj_m=float(np.std(sub.ds.values[ok] * sub[k].values[ok], ddof=1) / np.sqrt(n)) if n > 1 else None)
    rho, p = stats.spearmanr(T.Xs_km, T.ds)
    S['ds_vs_family_alongstrike_spearman'] = dict(rho=float(rho), p=float(p))
    return T, S, null


def overlap_pull(D):
    """Do overlapping other-family LFEs (+-10 s) pull locations toward them? Family-demeaned along-strike (A+B)/2
    position for events whose overlaps lie net-east vs net-west; null: permute ov_east sign within family."""
    X = D.copy()
    X['s'] = 0.5 * (X.s_a + X.s_b)
    X['s'] = X.s - X.groupby('fam').s.transform('mean')
    e, w = X.ov_east.values > 0, X.ov_east.values < 0
    obs = X.s.values[e].mean() - X.s.values[w].mean()
    se = np.sqrt(X.s.values[e].var() / e.sum() + X.s.values[w].var() / w.sum())
    sub = X[e | w]
    null = []
    for _ in range(NPERM):
        lab = sub.groupby('fam').ov_east.transform(lambda v: v.sample(frac=1, random_state=int(rng.integers(1e9))).values).values
        null.append(sub.s.values[lab > 0].mean() - sub.s.values[lab < 0].mean())
    null = np.array(null)
    return dict(n_east=int(e.sum()), n_west=int(w.sum()), n_none=int((X.ov_east == 0).sum()), pull_east_minus_west_m=float(obs), se_m=float(se),
                p_perm=float((np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1)))


# =================================================================== Task 2
def trC(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    return np.sum(a * b) / len(a)


def task2(D, fams, clean_burst=False):
    X = D[D.fam.isin(fams)].copy()
    X['grp'] = np.where(X.n_nb1h == 0, 'iso', np.where(X.burst_n_cat >= 10, 'burst', 'mid'))
    if clean_burst:                                               # in-burst events without any catalog LFE within +-10 s
        X.loc[(X.grp == 'burst') & (X.n_any10s > 0), 'grp'] = 'mid'
    X['terc'] = X.groupby('fam').cc.transform(lambda c: pd.qcut(c.rank(method='first'), 3, labels=False))
    X = X[X.grp != 'mid']
    cols_a, cols_b = ['s_a', 'd_a', 'n_a'], ['s_b', 'd_b', 'n_b']

    def stat(X):
        dl, wl, tri, trb, wt = [], [], [], [], []
        for (F, k), g in X.groupby(['fam', 'terc']):
            gi, gb = g[g.grp == 'iso'], g[g.grp == 'burst']
            if len(gi) < 3 or len(gb) < 3:
                continue
            ci = 0.5 * (gi[cols_a].values + gi[cols_b].values).mean(0)
            cb = 0.5 * (gb[cols_a].values + gb[cols_b].values).mean(0)
            w = 1 / (1 / len(gi) + 1 / len(gb))
            dl.append(ci - cb); wl.append(w)
            tri.append(trC(gi[cols_a].values, gi[cols_b].values)); trb.append(trC(gb[cols_a].values, gb[cols_b].values)); wt.append(w)
        dl, wl = np.array(dl), np.array(wl)
        dmean = (dl * wl[:, None]).sum(0) / wl.sum()
        dsq = np.sum(wl * np.sum(dl ** 2, 1)) / wl.sum()          # mean |dcentroid|^2 (noise-inflated)
        return dict(d_s=dmean[0], d_d=dmean[1], d_n=dmean[2], dsq=dsq,
                    tr_iso=np.average(tri, weights=wt), tr_burst=np.average(trb, weights=wt))

    obs = stat(X)
    # null 1: permute iso/burst labels within family x tercile
    null = []
    for _ in range(NPERM // 2):
        Y = X.copy()
        Y['grp'] = Y.groupby(['fam', 'terc']).grp.transform(lambda s: s.sample(frac=1, random_state=int(rng.integers(1e9))).values)
        null.append(stat(Y))
    null = pd.DataFrame(null)
    # cluster bootstrap: units = isolated events and whole bursts, resampled within family
    X['unit'] = np.where(X.grp == 'iso', -(np.arange(len(X)) + 1), X.bid)
    boots = []
    units = {F: g.unit.unique() for F, g in X.groupby('fam')}
    gx = {u: g for u, g in X.groupby('unit')}
    for _ in range(300):
        parts = []
        for F, u in units.items():
            parts += [gx[v] for v in rng.choice(u, len(u))]
        boots.append(stat(pd.concat(parts, ignore_index=True)))
    boots = pd.DataFrame(boots)
    S = dict(n_iso=int((X.grp == 'iso').sum()), n_burst=int((X.grp == 'burst').sum()), families=fams)
    for k, v in obs.items():
        nv = null[k].values
        p = (np.sum(np.abs(nv - nv.mean()) >= abs(v - nv.mean())) + 1) / (len(nv) + 1)
        S[k] = dict(obs=float(v), boot_se=float(boots[k].std()), null_mean=float(nv.mean()), null_sd=float(nv.std()), p_perm=float(p))
    rat = boots.tr_iso / boots.tr_burst
    S['tr_ratio_iso_over_burst'] = dict(obs=float(obs['tr_iso'] / obs['tr_burst']), boot_lo=float(rat.quantile(.025)),
                                        boot_hi=float(rat.quantile(.975)),
                                        null_lo=float((null.tr_iso / null.tr_burst).quantile(.025)),
                                        null_hi=float((null.tr_iso / null.tr_burst).quantile(.975)))
    # per family table
    rows = []
    for F, g in X.groupby('fam'):
        gi, gb = g[g.grp == 'iso'], g[g.grp == 'burst']
        if len(gi) < 3 or len(gb) < 3:
            continue
        ci = 0.5 * (gi[cols_a].values + gi[cols_b].values)
        cb = 0.5 * (gb[cols_a].values + gb[cols_b].values)
        rows.append(dict(fam=F, n_iso=len(gi), n_burst=len(gb), cc_iso=gi.cc.mean(), cc_burst=gb.cc.mean(),
                         d_s=ci[:, 0].mean() - cb[:, 0].mean(), d_d=ci[:, 1].mean() - cb[:, 1].mean(),
                         se_s=np.sqrt(ci[:, 0].var() / len(gi) + cb[:, 0].var() / len(gb)),
                         se_d=np.sqrt(ci[:, 1].var() / len(gi) + cb[:, 1].var() / len(gb)),
                         tr_iso=trC(gi[cols_a].values, gi[cols_b].values), tr_burst=trC(gb[cols_a].values, gb[cols_b].values)))
    P = pd.DataFrame(rows)
    P.to_csv(f'{OUT}/task2_core_halo_by_family{"_cleanburst" if clean_burst else ""}{TAG}.csv', index=False)
    S['cc_mean_iso'] = float(X[X.grp == 'iso'].cc.mean()); S['cc_mean_burst'] = float(X[X.grp == 'burst'].cc.mean())
    return S, P


# =================================================================== Task 3
def task3(D, fams):
    X = D[D.fam.isin(fams)].copy()
    X['ph'] = 2 * np.pi * (((X.t.values - (TREF - pd.Timestamp('1970-01-01')).total_seconds()) / M2_S) % 1.0)
    for c in 'sdn':
        X[c] = 0.5 * (X[f'{c}_a'] + X[f'{c}_b'])
        X[c] = X[c] - X.groupby('fam')[c].transform('mean')
    # phase-randomisation units: catalog bursts (isolated events -> own unit)
    X['unit'] = np.where(X.bid >= 0, X.bid, -(np.arange(len(X)) + 1))
    uid, uinv = np.unique(X.unit.values, return_inverse=True)
    fam_codes = pd.factorize(X.fam)[0]
    nF = fam_codes.max() + 1

    def fit(ph):
        c, s = np.cos(ph), np.sin(ph)
        out = {}
        for comp in 'sd':
            y = X[comp].values
            # common-phase pooled amplitude (families demeaned)
            M = np.c_[c, s]
            coef = np.linalg.lstsq(M, y, rcond=None)[0]
            out[f'{comp}_amp'] = np.hypot(*coef); out[f'{comp}_phase_deg'] = np.degrees(np.arctan2(coef[1], coef[0])) % 360
            # family-specific: sum_F n_F * A_F^2 / n
            sc = np.bincount(fam_codes, y * c, nF); ss = np.bincount(fam_codes, y * s, nF)
            cc = np.bincount(fam_codes, c * c, nF); s2 = np.bincount(fam_codes, s * s, nF)
            nn = np.bincount(fam_codes, minlength=nF)
            aF = np.hypot(sc / cc, ss / s2)
            out[f'{comp}_famamp_rms'] = np.sqrt(np.sum(nn * aF ** 2) / nn.sum())
        return out

    obs = fit(X.ph.values)
    null = pd.DataFrame([fit((X.ph.values + rng.uniform(0, 2 * np.pi, len(uid))[uinv]) % (2 * np.pi)) for _ in range(NPERM)])
    S = dict(n_events=len(X), n_units=len(uid), families=fams)
    for k in obs:
        if 'phase' in k:
            S[k] = float(obs[k]); continue
        nv = null[k].values
        S[k] = dict(obs=float(obs[k]), null_mean=float(nv.mean()), null_95=float(np.quantile(nv, .95)),
                    p=float((np.sum(nv >= obs[k]) + 1) / (len(nv) + 1)))
    # binned centroids with burst-cluster SE
    X['bin'] = (X.ph / (2 * np.pi) * 8).astype(int).clip(0, 7)
    rows = []
    for b, g in X.groupby('bin'):
        r = dict(bin=b, phase_center_deg=(b + 0.5) * 45, n=len(g), n_units=g.unit.nunique())
        for comp in 'sdn':
            um = g.groupby('unit')[comp].agg(['sum', 'count'])
            mu = um['sum'].sum() / um['count'].sum()
            # cluster-robust SE of the ratio estimator
            resid = um['sum'] - mu * um['count']
            se = np.sqrt(np.sum(resid ** 2)) / um['count'].sum()
            r[f'{comp}_mean_m'], r[f'{comp}_se_m'] = mu, se
        rows.append(r)
    Bn = pd.DataFrame(rows)
    Bn.to_csv(f'{OUT}/task3_m2_bins{TAG}.csv', index=False)
    for comp in 'sd':
        chi2 = np.sum((Bn[f'{comp}_mean_m'] / Bn[f'{comp}_se_m']) ** 2)
        S[f'{comp}_bin_chi2_dof8'] = dict(chi2=float(chi2), p=float(stats.chi2.sf(chi2, 8)))
    return S, Bn, null


# =================================================================== Task 4
EDGES = np.logspace(1, np.log10(10 * 365.25 * 86400), 31)          # 10 s .. 10 yr, 30 bins


def d2_accumulate(fam_arrays, perm_strata=None):
    """Return per-bin (sum of normalised D2, count) and unnormalised sum, pooled over families.
    fam_arrays: list of (t, xa, xb, trc, year, stratum)."""
    nb = len(EDGES) - 1
    S, Su, N = np.zeros(nb), np.zeros(nb), np.zeros(nb)   # S: sum of 2trC_F per pair (normaliser)
    for t, xa, xb, trc, yr, st in fam_arrays:
        if perm_strata is not None:
            idx = np.arange(len(t))
            for u in np.unique(st):
                w = np.where(st == u)[0]
                idx[w] = w[rng.permutation(len(w))]
            xa, xb = xa[idx], xb[idx]
        n = len(t)
        for i0 in range(0, n, 400):
            i1 = min(n, i0 + 400)
            ii = np.arange(i0, i1)[:, None]
            jj = np.arange(n)[None, :]
            mask = jj > ii
            tau = np.abs(t[None, :] - t[i0:i1, None])
            dd = np.einsum('ijk,ijk->ij', xa[None, :, :] - xa[i0:i1, None, :], xb[None, :, :] - xb[i0:i1, None, :])
            k = np.searchsorted(EDGES, tau[mask]) - 1
            ok = (k >= 0) & (k < nb)
            v = dd[mask][ok]
            S += 2 * trc * np.bincount(k[ok], minlength=nb)
            Su += np.bincount(k[ok], v, nb)
            N += np.bincount(k[ok], minlength=nb)
    return S, Su, N


def task4(D, fams):
    arrs = []
    for F in fams:
        g = D[D.fam == F].sort_values('t')
        if len(g) < 30:
            continue
        xa = g[['s_a', 'd_a', 'n_a']].values; xb = g[['s_b', 'd_b', 'n_b']].values
        trc = trC(xa, xb)
        if trc <= 0:
            continue
        yr = pd.to_datetime(g.t, unit='s').dt.year.values
        arrs.append((g.t.values, xa, xb, trc, yr, yr))
    S, Su, N = d2_accumulate(arrs)
    obs = Su / np.where(S > 0, S, np.nan)                         # ratio of sums: sum D2 / sum 2trC_F
    # leave-one-family-out jackknife
    jk = []
    for i in range(len(arrs)):
        s, su, n = d2_accumulate([arrs[i]])
        jk.append((Su - su) / np.where(S - s > 0, S - s, np.nan))
    jk = np.array(jk)
    nj = len(arrs)
    se = np.sqrt((nj - 1) / nj * np.sum((jk - jk.mean(0)) ** 2, 0))
    # nulls: permute times within family x year (keeps >1 yr drift) and within family (full)
    nperm = int(os.environ.get('NPERM4', 100))
    nyr = []
    for _ in range(nperm):
        s, su, n = d2_accumulate(arrs, perm_strata=True)
        nyr.append(su / np.where(s > 0, s, np.nan))
    nyr = np.array(nyr)
    full = [(a[0], a[1], a[2], a[3], a[4], np.zeros(len(a[0]), int)) for a in arrs]
    nfull = []
    for _ in range(nperm):
        s, su, n = d2_accumulate(full, perm_strata=True)
        nfull.append(su / np.where(s > 0, s, np.nan))
    nfull = np.array(nfull)
    tc = np.sqrt(EDGES[1:] * EDGES[:-1])
    T = pd.DataFrame(dict(tau_lo_s=EDGES[:-1], tau_hi_s=EDGES[1:], tau_c_s=tc, n_pairs=N, D2_over_2trC=obs, se_jk=se,
                          D2_m2=Su / np.maximum(N, 1),
                          null_year_mean=nyr.mean(0), null_year_lo=np.quantile(nyr, .025, 0), null_year_hi=np.quantile(nyr, .975, 0),
                          null_full_mean=nfull.mean(0), null_full_lo=np.quantile(nfull, .025, 0), null_full_hi=np.quantile(nfull, .975, 0)))
    T.to_csv(f'{OUT}/task4_D2_tau{TAG}.csv', index=False)
    # summary: grouped tau ranges
    out = dict(n_families=len(arrs), n_events=int(sum(len(a[0]) for a in arrs)))
    ranges = {'<1h': (10, 3600), '1h-1d': (3600, 86400), '1d-30d': (86400, 30 * 86400), '30d-1yr': (30 * 86400, 365.25 * 86400), '>1yr': (365.25 * 86400, 1e10)}
    for lab, (lo, hi) in ranges.items():
        sel = (tc >= lo) & (tc < hi) & (N > 0)
        w = N[sel]
        def agg(v):
            return float(np.sum(v[sel] * w) / w.sum())
        val = agg(obs)
        # jackknife SE of the grouped value
        jv = np.array([np.sum(j[sel] * w) / w.sum() for j in jk])
        sej = float(np.sqrt((nj - 1) / nj * np.sum((jv - jv.mean()) ** 2)))
        ny = np.array([np.sum(v[sel] * w) / w.sum() for v in nyr]); nf = np.array([np.sum(v[sel] * w) / w.sum() for v in nfull])
        out[lab] = dict(n_pairs=int(w.sum()), D2_over_2trC=val, se_jk=sej,
                        null_year_mean=float(ny.mean()), p_lt_null_year=float((np.sum(ny <= val) + 1) / (len(ny) + 1)),
                        null_full_mean=float(nf.mean()), p_lt_null_full=float((np.sum(nf <= val) + 1) / (len(nf) + 1)))
    # power-law slope of D2 vs tau over the rising range (10 s - 1 d), weighted by 1/se^2 where finite
    sel = (tc < 86400) & (N > 50) & (obs > 0)
    if sel.sum() >= 3:
        w = 1 / np.maximum(se[sel], 1e-3) ** 2
        A = np.c_[np.ones(sel.sum()), np.log10(tc[sel])]
        coef, *_ = np.linalg.lstsq(A * np.sqrt(w)[:, None], np.log10(obs[sel]) * np.sqrt(w), rcond=None)
        # jackknife the slope
        sl = []
        for j in jk:
            ss = sel & (j > 0)
            if ss.sum() >= 3:
                sl.append(np.polyfit(np.log10(tc[ss]), np.log10(j[ss]), 1, w=1 / np.maximum(se[ss], 1e-3))[0])
        sl = np.array(sl)
        out['loglog_slope_10s_1d'] = dict(slope=float(coef[1]), se_jk=float(np.sqrt((len(sl) - 1) / len(sl) * np.sum((sl - sl.mean()) ** 2))) if len(sl) > 1 else None)
    return T, out


# =================================================================== figures
def fig1(T, null, S1):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    a = ax[0]
    a.hist(T.ds, bins=30, color='0.6', edgecolor='k')
    a.axvline(0, color='k', lw=.8); a.axvline(T.ds.mean(), color='C3', lw=2, label=f'mean {T.ds.mean():.0f}±{S1["mean_ds_m_se"]:.0f} m')
    a.set_xlabel('along-strike early→late displacement (m, +ESE)'); a.set_ylabel('bursts'); a.legend(fontsize=8)
    a.set_title(f'{len(T)} bursts (≥8 good events)')
    a = ax[1]
    for sgn, col, lab in [(1, 'C0', 'west(NW)-led episodes'), (-1, 'C1', 'east(SE)-led episodes')]:
        v = T.ds[T.leader_side == sgn]
        a.hist(v, bins=20, alpha=.6, color=col, label=f'{lab} (n={len(v)}, mean {v.mean():.0f}±{v.std(ddof=1)/np.sqrt(max(len(v),1)):.0f} m)')
    a.axvline(0, color='k', lw=.8); a.legend(fontsize=8); a.set_xlabel('along-strike displacement (m, +ESE)')
    a.set_title('sign-flip test: expected + for west-led, − for east-led', fontsize=9)
    a = ax[2]
    for k, col, lab in [('flip_west_minus_east', 'C3', 'west−east led'), ('proj_sweep_local', 'C2', 'along local sweep')]:
        a.hist(null[k], bins=40, color=col, alpha=.35, label=f'null ({lab})')
        a.axvline(S1[k]['obs'], color=col, lw=2, label=f'obs {lab}: {S1[k]["obs"]:.0f} m (p={S1[k]["p_shuffle"]:.2f})')
    a.set_xlabel('statistic (m)'); a.legend(fontsize=7); a.set_title('order-shuffle null')
    fig.tight_layout(); fig.savefig(f'figures/sci_patch_dynamics_burstfront{TAG}.png', dpi=120); plt.close(fig)


def fig2(T4, Bn, S3):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    a = ax[0]
    tc = T4.tau_c_s / 3600
    ok = T4.n_pairs > 20
    a.fill_between(tc[ok], T4.null_year_lo[ok], T4.null_year_hi[ok], color='C1', alpha=.25, label='null: times permuted within family×year')
    a.fill_between(tc[ok], T4.null_full_lo[ok], T4.null_full_hi[ok], color='0.6', alpha=.3, label='null: times permuted within family')
    a.errorbar(tc[ok], T4.D2_over_2trC[ok], 2 * T4.se_jk[ok], fmt='o-', color='k', ms=4, label='observed (±2 jackknife SE)')
    a.set_xscale('log'); a.axhline(1, color='k', ls=':', lw=.8)
    for x, l in [(1, '1 h'), (24, '1 d'), (24 * 30, '1 mo'), (24 * 365, '1 yr')]:
        a.axvline(x, color='0.8', lw=.6); a.text(x, 4.2, l, fontsize=7, ha='center')
    a.set_ylim(-0.5, 4.5)
    a.set_xlabel('time lag τ (h)'); a.set_ylabel('D²(τ) / 2 tr C_true')
    a.set_title('Spatial structure function (A×B, resolved families)'); a.legend(fontsize=7, loc='lower right')
    a = ax[1]
    ph = Bn.phase_center_deg
    for comp, col, lab in [('s', 'C0', 'along-strike'), ('d', 'C2', 'down-dip')]:
        a.errorbar(ph + (-3 if comp == 's' else 3), Bn[f'{comp}_mean_m'], Bn[f'{comp}_se_m'], fmt='o-', color=col,
                   label=f'{lab}: amp {S3[comp + "_amp"]["obs"]:.1f} m (null95 {S3[comp + "_amp"]["null_95"]:.1f}, p={S3[comp + "_amp"]["p"]:.2f})')
    a.axhline(0, color='k', lw=.8); a.set_xlabel('M2 phase (deg, ref 2000-01-01T12:00)'); a.set_ylabel('centroid offset (m)')
    a.set_title('Tidal (M2) centroid shift'); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f'figures/sci_patch_dynamics_D2_tide{TAG}.png', dpi=120); plt.close(fig)


def fig3(P2, S2):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    a = ax[0]
    y = np.arange(len(P2))
    a.errorbar(P2.d_s, y - .1, xerr=2 * P2.se_s, fmt='o', label='along-strike')
    a.errorbar(P2.d_d, y + .1, xerr=2 * P2.se_d, fmt='s', label='down-dip')
    a.set_yticks(y); a.set_yticklabels([f'fam {f}' for f in P2.fam]); a.axvline(0, color='k', lw=.8)
    a.set_xlabel('isolated − in-burst centroid (m, ±2 SE)'); a.legend(fontsize=8)
    a = ax[1]
    a.scatter(P2.tr_burst / 1e4, P2.tr_iso / 1e4)
    for _, r in P2.iterrows():
        a.annotate(int(r.fam), (r.tr_burst / 1e4, r.tr_iso / 1e4), fontsize=7)
    m = max(P2.tr_burst.max(), P2.tr_iso.max()) / 1e4
    a.plot([min(0, P2.tr_iso.min() / 1e4), m], [min(0, P2.tr_iso.min() / 1e4), m], 'k:')
    r = S2['tr_ratio_iso_over_burst']
    a.set_title(f'tr C_true: pooled iso/burst = {r["obs"]:.2f} [{r["boot_lo"]:.2f}, {r["boot_hi"]:.2f}]', fontsize=9)
    a.set_xlabel('in-burst tr C (10⁴ m²)'); a.set_ylabel('isolated tr C (10⁴ m²)')
    fig.tight_layout(); fig.savefig(f'figures/sci_patch_dynamics_corehalo{TAG}.png', dpi=120); plt.close(fig)


def main():
    global ALLT, ALLF, XS
    cat = load_catalog()
    o = np.argsort(cat.t.values)
    ALLT, ALLF = cat.t.values[o], cat.fam.values[o]
    pos = family_positions()
    XS = np.zeros(28)
    for F in pos:
        XS[F] = pos[F] @ U_S[:2]
    pos = family_positions()
    on = catalog_bursts(cat)
    E = pd.read_csv('data/cascades/episodes.csv')
    # verify episode reconstruction
    ept = on.groupby('ep').onset.min()
    chk = np.abs(ept.reindex(E.ep).values - tsec(E.t0)) < 1
    print(f'episode reconstruction: {chk.mean():.3f} of {len(E)} episode t0 match')
    D = pd.concat([load_family(F, cat[cat.fam == F].t.values, on) for F in range(1, 28)], ignore_index=True)
    print(f'{len(D)} events good in both halves (SDH={SDH}); median |t_reloc - t_cat| = {D.dt_cat.median():.1f} s; '
          f'frac > 60 s: {(D.dt_cat > 60).mean():.3f}; unassigned to catalog burst: {(D.bid < 0).mean():.3f}')
    summ = dict(SDH=SDH, n_good_events=len(D), episode_reconstruction_match=float(chk.mean()),
                axes=dict(along_strike=U_S.tolist(), down_dip=U_D.tolist(), normal=U_N.tolist()))

    # Task 1: very few bursts have >=8 events good at sd<0.5 km, so the primary uses sd_xy<1.0, sd_z<2.0 km
    # (A/B cross-estimates are unbiased for any error size; a looser cut only adds noise). Sensitivity variants below.
    def ccdemean(X):
        X = X.copy()
        X['terc'] = X.groupby('fam').cc.transform(lambda c: pd.qcut(c.rank(method='first'), 3, labels=False))
        for c in ['s_a', 'd_a', 'n_a', 's_b', 'd_b', 'n_b']:
            X[c] = X[c] - X.groupby(['fam', 'terc'])[c].transform('mean')
        return X
    if os.environ.get('SKIP1') != '1':
        T1_SDH = float(os.environ.get('T1_SDH', 1.0))
        D1 = pd.concat([load_family(F, cat[cat.fam == F].t.values, on, T1_SDH) for F in range(1, 28)], ignore_index=True)
        T1, S1, null1 = task1(D1, E, on, pos, f'sd{T1_SDH}_n8')
        summ['task1_primary'] = S1
        T1r = T1[T1.fam.isin(RESOLVED)]
        summ['task1_primary_resolved_only'] = dict(n=len(T1r), mean_ds=float(T1r.ds.mean()), se=float(T1r.ds.std(ddof=1) / np.sqrt(len(T1r))),
                                                   flip=float(T1r.ds[T1r.leader_side == 1].mean() - T1r.ds[T1r.leader_side == -1].mean()))
        print('task1 primary', json.dumps(S1, indent=1, default=float))
        fig1(T1, null1, S1)
        variants = [(f'sd{T1_SDH}_n8_clean10s', D1[D1.n_any10s == 0], 8), (f'sd{T1_SDH}_n8_ccdemean', ccdemean(D1), 8)]
        D2_ = pd.concat([load_family(F, cat[cat.fam == F].t.values, on, 2.0) for F in range(1, 28)], ignore_index=True)
        variants += [('sd2.0_n8', D2_, 8), ('sd2.0_n8_clean10s', D2_[D2_.n_any10s == 0], 8), ('sd2.0_n8_ccdemean', ccdemean(D2_), 8),
                     ('sd0.5_n5', D, 5)]
        summ['task1_overlap_pull'] = overlap_pull(D1)
        print('overlap pull', summ['task1_overlap_pull'])
        def ovcorr(X):
            X = X.copy()
            X['ocls'] = np.sign(X.ov_east)
            for c in ['s_a', 'd_a', 'n_a', 's_b', 'd_b', 'n_b']:
                X[c] = X[c] - (X.groupby(['fam', 'ocls'])[c].transform('mean') - X.groupby('fam')[c].transform('mean'))
            return X
        variants += [(f'sd{T1_SDH}_n8_overlapcorr', ovcorr(D1), 8), ('sd2.0_n8_overlapcorr', ovcorr(D2_), 8)]
        variants += [(f'sd{T1_SDH}_n4_clean10s', D1[D1.n_any10s == 0], 4), ('sd2.0_n4_clean10s', D2_[D2_.n_any10s == 0], 4)]
        summ['task1_variants'] = {}
        for lab, X, nm in variants:
            _, Sv, _ = task1(X, E, on, pos, lab, nm)
            summ['task1_variants'][lab] = Sv
            print(lab, len(X), {k: Sv.get(k) for k in ['n_bursts', 'mean_ds', 'flip_west_minus_east', 'flip_famdemeaned', 'proj_sweep_local', 'proj_local_famdemeaned', 'flip_welch']})

    S2, P2 = task2(D, RESOLVED)
    summ['task2_core_halo_resolved'] = S2
    S2c, _ = task2(D, RESOLVED, clean_burst=True)
    summ['task2_core_halo_resolved_cleanburst'] = S2c
    print('task2 clean', {k: S2c[k] for k in ['n_iso', 'n_burst', 'd_s', 'd_d']})
    print('task2', json.dumps(S2, indent=1, default=float))
    fig3(P2, S2)

    S3, Bn, _ = task3(D, RESOLVED)
    S3all, _, _ = task3(D, list(range(1, 28)))
    summ['task3_m2_resolved'] = S3
    summ['task3_m2_all_families'] = S3all
    print('task3', json.dumps(S3, indent=1, default=float))

    T4, S4 = task4(D, RESOLVED)
    summ['task4_D2_resolved'] = S4
    print('task4', json.dumps(S4, indent=1, default=float))
    fig2(T4, Bn, S3)

    with open(f'{OUT}/summary{TAG}.json', 'w') as f:
        json.dump(summ, f, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))


if __name__ == '__main__':
    main()
