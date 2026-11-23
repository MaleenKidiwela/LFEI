#!/usr/bin/env python3
"""Analysis stage of sci_transient_source.py (tasks 2-4, figures, summary). See that file's docstring."""
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dvv_cwire import epoch_s, param_G  # noqa: E402,F401
from sci_dvv_depth import GroupFit  # noqa: E402
from sci_transient_source_common import AX, DAY, FAMS, M7, OUT, make_stacks, periods  # noqa: E402

rng = np.random.default_rng(7)
NB = 200                     # block bootstrap draws (stack resampling) for the group fits
NMC = 300                    # Monte Carlo draws for K = Sigma_true Sigma_meas^-1
OBS = {('surface', 'b28_early'): (-0.1766, 0.0349), ('surface', 'b28_late'): (-0.0166, 0.0293),
       ('borehole', 'b28_early'): (-0.0995, 0.0535), ('borehole', 'b28_late'): (-0.0517, 0.0409)}
GROUPS = {'surface': lambda c: ~c.str.startswith('PB.'), 'borehole': lambda c: c.str.startswith('PB.')}
TAU_MID = {'w3_5': 4.0, 'w5_8': 6.5, 'w8_11': 9.5, 'w11_16': 13.5, 'early': 5.5, 'late': 11.0}


def load(F):
    T = pd.read_parquet(f'{OUT}/fam{F}_events.parquet')
    # own intermediate file written by sci_transient_source.py --prep (trusted)
    D = pickle.load(open(f'{OUT}/fam{F}_defs.pkl', 'rb'))
    T.attrs['restart'] = D['restart']
    return T, D


def keep_channels(F):
    m = pd.read_csv(f'data/sci/dvv_depth/fam{F}_channels.csv')
    ok = (m.b28_early_ccref >= 0.25) & (m.b28_late_ccref >= 0.25)
    return set(m.chan[ok])


# ------------------------------------------------------------------ K = Sigma_t Sigma_F^-1
def k_matrix(T, pool, ab, boot=False):
    ia = pool if not boot else pool[rng.integers(0, len(pool), len(pool))]
    ib = ab if not boot else ab[rng.integers(0, len(ab), len(ab))]
    SF = np.cov(T[AX].values[ia].T)
    xa, xb = T[[c + 'A' for c in AX]].values[ib], T[[c + 'B' for c in AX]].values[ib]
    a, b = xa - xa.mean(0), xb - xb.mean(0)
    C = (a.T @ b) / len(a)
    St = 0.5 * (C + C.T)
    return St @ np.linalg.inv(SF), St, SF


def split_separations(F, T, D):
    """Measured and TRUE (s, d, n) separations Hi - Lo of every location split, with MC draws."""
    mem = D['members']
    pre = T.iso.values & ~T.ovl10.values & (T.day.values < 0)
    pool = np.where(pre & T['loc'].values)[0]
    ab = np.where(~T.ovl10.values & (T.day.values < 0) & T.locAB.values)[0]
    K, St, SF = k_matrix(T, pool, ab)
    Kmc = np.array([k_matrix(T, pool, ab, boot=True)[0] for _ in range(NMC)])
    rows, mc = [], {}
    X = T[AX].values
    XA, XB = T[[c + 'A' for c in AX]].values, T[[c + 'B' for c in AX]].values
    for name, (hi, lo) in mem.items():
        if name.startswith('F'):
            dF = np.nanmean(X[hi], 0) - np.nanmean(X[lo], 0)
            dt = K @ dF
            mc[name] = np.einsum('kij,j->ki', Kmc, dF)
        else:   # selected on A; true separation from the independent B half (events with both)
            dF = np.nanmean(XA[hi], 0) - np.nanmean(XA[lo], 0)
            hb, lb = hi[np.isfinite(XB[hi, 0])], lo[np.isfinite(XB[lo, 0])]
            dt = XB[hb].mean(0) - XB[lb].mean(0)
            mc[name] = np.array([XB[rng.choice(hb, len(hb))].mean(0) - XB[rng.choice(lb, len(lb))].mean(0)
                                 for _ in range(NMC)])
        r = dict(fam=F, split=name, n_hi=len(hi), n_lo=len(lo))
        for k, c in enumerate(AX):
            r[f'meas_d{c}'], r[f'true_d{c}'], r[f'true_d{c}_se'] = dF[k], dt[k], mc[name][:, k].std()
        rows.append(r)
    info = dict(fam=F, n_pool=len(pool), n_AB=len(ab),
                **{f'std_true_{c}': float(np.sqrt(max(St[k, k], 0))) for k, c in enumerate(AX)},
                **{f'std_meas_{c}': float(np.sqrt(SF[k, k])) for k, c in enumerate(AX)},
                **{f'K_{c}{c}': float(K[k, k]) for k, c in enumerate(AX)})
    return pd.DataFrame(rows), mc, info


# ------------------------------------------------------------------ task 2: sensitivity
def tau_eff_inv(w0, w1, tau_end):
    """1/tau_eff = sum(tau)/sum(tau^2) over the window: LSQ stretch that a constant coda-vs-S delay maps into."""
    hi = tau_end if w1 is None else min(w1, tau_end)
    if hi - w0 < 1.9:
        return np.nan
    t = np.linspace(w0, hi, 200)
    return t.sum() / (t ** 2).sum()


WIN = {'early': (3.0, 8.0), 'late': (8.0, None), 'w3_5': (3.0, 5.0), 'w5_8': (5.0, 8.0), 'w8_11': (8.0, 11.0),
       'w11_16': (11.0, 16.0)}


def wls1(x, y, w, intercept_groups):
    """y = a_group + c x (weighted); returns c."""
    ug, gi = np.unique(intercept_groups, return_inverse=True)
    X = np.c_[np.eye(len(ug))[gi], x]
    sw = np.sqrt(w)
    return np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0][-1]


def sensitivity(F, grp, E, sep, mc, grads, meta, keep):
    """Per window: group-mean eps of each real split (+ null sd), empirical along-strike g from the s-split,
    S-shift model calibration (c = eps per s of differential direct-S delay, kappa = measured / predicted
    differential S delay), model g vectors, and model-predicted split eps (validation)."""
    E = E[GROUPS[grp](E.chan) & E.chan.isin(keep) & (E.cc > 0.2)].copy()
    E['sta'] = E.chan.str.split('.').str[:2].str.join('.')
    E = E[E.sta.isin(grads)]
    tend = meta.set_index('chan').tau_end
    out = []
    for win, g in E.groupby('window'):
        sd = g[g.split.str.startswith('Fnull')].groupby('chan').eps.std().rename('sd')
        g = g.join(sd, on='chan')
        g = g[np.isfinite(g.sd) & (g.sd > 0)]
        g['w'] = 1 / g.sd ** 2
        if g.chan.nunique() < 3:
            continue
        wm = lambda x, c='eps': np.sum(x[c] * x.w) / np.sum(x.w)  # noqa: E731
        real = g[~g.split.str.contains('null')]
        nul = np.array([wm(x) for _, x in g[g.split.str.startswith('Fnull')].groupby('split')])
        nulA = np.array([wm(x) for _, x in g[g.split.str.startswith('Anull')].groupby('split')])
        r = dict(fam=F, group=grp, window=win, n_chan=int(g.chan.nunique()), null_sd_F_pct=nul.std() * 100,
                 null_sd_A_pct=nulA.std() * 100 if len(nulA) else np.nan, tau_mid=TAU_MID[win])
        for sp, x in real.groupby('split'):
            r[f'eps_{sp}_pct'] = wm(x) * 100
            r[f'dS_{sp}_ms'] = wm(x, 'dS') * 1e3
        # empirical along-strike g from the s-splits (true separation, MC)
        for pool, ns in [('F', nul.std()), ('A', nulA.std() if len(nulA) else np.nan)]:
            sp = pool + 's'
            if f'eps_{sp}_pct' not in r:
                continue
            e = r[f'eps_{sp}_pct'] / 100
            dt = sep.loc[sp, 'true_ds']
            draws = (e + rng.normal(0, ns, NMC)) / mc[sp][:, 0]
            r[f'gemp_s_{pool}'] = e / dt
            r[f'gemp_s_{pool}_se'] = 0.5 * np.diff(np.percentile(draws, [16, 84]))[0]
        # S-shift model calibration on the F splits (s, d, n)
        fr = real[real.split.isin(['Fs', 'Fd', 'Fn'])].copy()
        G3 = np.array([grads[s_] for s_ in fr.sta]) / 1000.0                      # s per m
        fr['dS_pred'] = [G3[i] @ sep.loc[sp, ['true_ds', 'true_dd', 'true_dn']].values.astype(float)
                         for i, sp in enumerate(fr.split)]
        c = wls1(fr.dS.values, fr.eps.values, fr.w.values, fr.split.values)
        kap = wls1(fr.dS_pred.values, fr.dS.values, fr.w.values, fr.split.values)
        chans = fr.chan.unique()
        cb, kb = [], []
        for _ in range(NMC):
            pick = rng.choice(chans, len(chans))
            sub = pd.concat([fr[fr.chan == ch] for ch in pick])
            cb.append(wls1(sub.dS.values, sub.eps.values, sub.w.values, sub.split.values))
            kb.append(wls1(sub.dS_pred.values, sub.dS.values, sub.w.values, sub.split.values))
        r.update(c_per_s=c, c_se=np.std(cb), kappa=kap, kappa_se=np.std(kb))
        # model g vectors (group weighted mean over channels): calibrated c, and the 1/tau_eff upper bound
        ch = g.drop_duplicates('chan').set_index('chan')
        Gc = np.array([grads[s_] for s_ in ch.sta]) / 1000.0
        wc = ch.w.values / ch.w.sum()
        ti = np.array([tau_eff_inv(*WIN[win], tend[cn]) for cn in ch.index])
        gm = wc @ Gc
        gb = np.nansum((wc * ti)[:, None] * Gc, 0) / np.sum(wc[np.isfinite(ti)])
        r['tau_eff_inv'] = float(np.nansum(wc * ti) / np.sum(wc[np.isfinite(ti)]))
        for k, a in enumerate(AX):
            r[f'gmod_{a}'] = c * gm[k]
            r[f'gmod_{a}_se'] = np.std(cb) * abs(gm[k])
            r[f'gbound_{a}'] = gb[k]
            r[f'grad_mean_{a}_s_per_km'] = gm[k] * 1000
        r['_c_draws'] = np.array(cb)
        r['_gm'] = gm
        # validation: model-predicted group eps of each F split vs observed
        for sp in ['Fs', 'Fd', 'Fn']:
            dt = sep.loc[sp, ['true_ds', 'true_dd', 'true_dn']].values.astype(float)
            r[f'pred_mod_{sp}_pct'] = c * (gm @ dt) * 100
            r[f'pred_bound_{sp}_pct'] = (gb @ dt) * 100
        out.append(r)
    return out


# ------------------------------------------------------------------ group fits
def make_rows(P, stack_t, F):
    P = P.copy()
    P['fam'] = F
    P['tb'] = stack_t[P.b.values]
    P['ta'] = np.where(P.a >= 0, stack_t[np.clip(P.a.values, 0, None)], np.nan)
    return P


def synth_transient(fit, P, cent, draws=None):
    """Transient amplitude returned by the SAME group fit (rows, IRLS weights) when the data are a pure
    stack-attribute signal d = c(b) - c(a) [ref rows: c(b); the constant is absorbed by the per-channel offsets].
    cent: dict fam -> (n_stacks,) attribute (e.g. centroid s in m). Returns transient (and boot draws if given)."""
    d = np.zeros(len(P))
    for F, c in cent.items():
        m = (P.fam.values == F)
        cb = c[P.b.values[m]]
        ca = np.where(P.a.values[m] >= 0, c[np.clip(P.a.values[m], 0, None)], 0.0)
        d[m] = cb - ca
    d0 = fit.d
    fit.d = d
    x = fit.solve(fit.w)[6]
    if draws is not None:
        xb = np.array([fit.solve(fit.w * mult)[6] for mult in draws])
    fit.d = d0
    return x if draws is None else (x, xb)


def boot_counts(fits, nb=NB):
    """Block bootstrap over (family, stack) ids shared by all given fits (paired). Returns per fit a list of row
    multipliers."""
    ids = np.unique(np.concatenate([np.r_[f.sid_b, f.sid_a[f.sid_a >= 0]] for f in fits]))
    out = [[] for _ in fits]
    for _ in range(nb):
        cnt = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids)).astype(float)
        for j, f in enumerate(fits):
            ib = np.searchsorted(ids, f.sid_b)
            ia = np.searchsorted(ids, np.where(f.sid_a >= 0, f.sid_a, ids[0]))
            out[j].append(cnt[ib] * np.where(f.sid_a >= 0, cnt[ia], 1.0))
    return out


def stack_centroids(T, stacks, weights=None, true_from=None):
    """(n_stacks, 3) measured centroid of located non-overlapping members (NaN-safe, filled by the family mean)."""
    X = T[AX].values
    ok = T['loc'].values & ~T.ovl10.values
    C = np.full((len(stacks), 3), np.nan)
    N = np.zeros(len(stacks))
    for b, m in enumerate(stacks):
        m = np.asarray(m)
        k = ok[m]
        if k.sum() == 0:
            continue
        w = np.ones(k.sum()) if weights is None else np.asarray(weights[b])[k]
        C[b] = (w[:, None] * X[m[k]]).sum(0) / w.sum()
        N[b] = k.sum()
    fill = np.nanmean(C, 0)
    C = np.where(np.isfinite(C), C, fill)
    return C, N


# ------------------------------------------------------------------ main
def analyze():
    from sci_transient_source_grad import s_gradients
    SEP, INFO, SENS = [], [], []
    ORIGP, LOCP, CENT, CENT_LOC, CENT_MATCH_T, STK, SD = {}, {}, {}, {}, {}, {}, {}
    for F in FAMS:
        T, D = load(F)
        sep, mc, info = split_separations(F, T, D)
        SEP.append(sep)
        INFO.append(info)
        E = pd.read_parquet(f'{OUT}/fam{F}_split_eps.parquet')
        meta = pd.read_csv(f'{OUT}/fam{F}_channels_meas.csv')
        keep = keep_channels(F)
        grads = s_gradients(F, sorted({'.'.join(c.split('.')[:2]) for c in E.chan.unique()}))
        for grp in GROUPS:
            for r in sensitivity(F, grp, E, sep.set_index('split'), mc, grads, meta, keep):
                SD[(F, grp, r['window'])] = r
                SENS.append({k: v for k, v in r.items() if not k.startswith('_')})
        # original stacks (dvv_cwire definition) and their location centroids
        iso_idx = np.where(T.iso.values)[0]
        iso_idx = iso_idx[np.argsort(T.t.values[iso_idx])]
        st = make_stacks(T.t.values, iso_idx, 50, 25)
        st_t = np.array([T.t.values[m].mean() for m in st])
        old = pd.read_csv(f'data/dvv/fam{F}_cwire_stacks.csv')
        assert len(old) == len(st)
        C, N = stack_centroids(T, st)
        CENT[F] = C
        STK[F] = pd.DataFrame(dict(fam=F, stack=np.arange(len(st)), t=pd.to_datetime(st_t, unit='s'), n=[len(m) for m in st],
                                   n_loc=N, cs=C[:, 0], cd=C[:, 1], cn=C[:, 2]))
        CL, _ = stack_centroids(T, D['stacks'])
        CW, _ = stack_centroids(T, D['stacks'], weights=[D['w_ev'][np.asarray(m)] for m in D['stacks']])
        pool = np.where(T.iso.values & ~T.ovl10.values & (T.day.values < 0) & T['loc'].values)[0]
        ab = np.where(~T.ovl10.values & (T.day.values < 0) & T.locAB.values)[0]
        K = k_matrix(T, pool, ab)[0]
        CENT_LOC[F] = CL
        CENT_MATCH_T[F] = CL + (CW - CL) @ K.T          # expected TRUE centroid change from the re-weighting
        p = pd.read_parquet(f'data/sci/dvv_depth/fam{F}_pairs.parquet')
        p = p[p.cfg.isin(['b28_early', 'b28_late']) & p.chan.isin(keep) & (p.cc >= 0.2)]
        ORIGP[F] = make_rows(p, epoch_s(pd.to_datetime(old.t, format='mixed')), F)
        for v in ['loc', 'match']:
            q = pd.read_parquet(f'{OUT}/fam{F}_pairs_{v}.parquet')
            q = q[q.chan.isin(keep) & (q.cc >= 0.2)]
            LOCP[(F, v)] = make_rows(q, D['stack_t'], F)
    SEP = pd.concat(SEP, ignore_index=True)
    SEP.to_csv(f'{OUT}/task2_split_separations.csv', index=False)
    INFO = pd.DataFrame(INFO)
    INFO.to_csv(f'{OUT}/task2_true_vs_measured_spread.csv', index=False)
    SENS = pd.DataFrame(SENS)
    SENS.to_csv(f'{OUT}/task2_sensitivity.csv', index=False)
    pd.concat(STK.values()).to_csv(f'{OUT}/stack_centroids_orig.csv', index=False)
    print(INFO.round(3).to_string())
    print(SEP[~SEP.split.str.contains('null')].round(1).to_string())
    cols = ['fam', 'group', 'window', 'n_chan', 'null_sd_F_pct', 'eps_Fs_pct', 'eps_Fd_pct', 'eps_Fn_pct', 'pred_mod_Fs_pct',
            'pred_mod_Fd_pct', 'pred_mod_Fn_pct', 'gemp_s_F', 'gemp_s_F_se', 'gemp_s_A', 'gemp_s_A_se', 'c_per_s', 'c_se',
            'kappa', 'kappa_se', 'tau_eff_inv', 'gmod_s', 'gmod_d', 'gmod_n', 'gbound_s', 'gbound_d', 'gbound_n']
    print(SENS[cols].to_string(float_format=lambda x: f'{x:.3g}'))

    # ---------------- tasks 3 + 4: group fits
    res, synth, locsig = [], [], []
    for grp, gsel in GROUPS.items():
        for cfg in ['b28_early', 'b28_late']:
            win = 'early' if cfg == 'b28_early' else 'late'
            sel = lambda P: pd.concat([P[F][(P[F].cfg == cfg) & gsel(P[F].chan)] for F in FAMS], ignore_index=True)  # noqa: E731
            po = sel(ORIGP)
            pl = sel({F: LOCP[(F, 'loc')] for F in FAMS})
            pm = sel({F: LOCP[(F, 'match')] for F in FAMS})
            fits = {'orig': GroupFit(po), 'loc': GroupFit(pl), 'match': GroupFit(pm)}
            (dr_o,) = boot_counts([fits['orig']])
            dr_l, dr_m = boot_counts([fits['loc'], fits['match']])
            drs = {'orig': dr_o, 'loc': dr_l, 'match': dr_m}
            B = {v: np.array([fits[v].solve(fits[v].w * mult) for mult in drs[v]]) for v in fits}
            for v, f in fits.items():
                res.append(dict(group=grp, cfg=cfg, variant=v, transient60_pct=f.m[6] * 100, se=B[v][:, 6].std() * 100,
                                step_m7_pct=f.m[5] * 100, step_se=B[v][:, 5].std() * 100, n_rows=len(f.d)))
            dlt = (B['match'][:, 6] - B['loc'][:, 6]) * 100
            res.append(dict(group=grp, cfg=cfg, variant='match-loc', transient60_pct=(fits['match'].m[6] - fits['loc'].m[6]) * 100,
                            se=dlt.std()))
            # ---- task 4: transient-shaped LOCATION signal through the same fits, and the predicted apparent transient
            for vname, P, cents in [('orig', po, CENT), ('loc', pl, CENT_LOC)]:
                f, dr = fits[vname], drs[vname]
                A = {(F, a): synth_transient(f, P, {F: cents[F][:, k]}, dr) for F in FAMS for k, a in enumerate(AX)}
                for k, a in enumerate(AX):
                    x, xb = synth_transient(f, P, {F: cents[F][:, k] for F in FAMS}, dr)
                    locsig.append(dict(group=grp, cfg=cfg, stacks=vname, axis=a, loc_transient_m=x, se_m=xb.std(),
                                       **{f'fam{F}_contrib_m': A[(F, a)][0] for F in FAMS}))
                for gtype in ['emp_s_F', 'emp_s_A', 'model', 'bound', 'hyb_model', 'hyb_bound']:
                    tot, tot_d, per_f = 0.0, np.zeros(len(dr)), {}
                    for F in FAMS:
                        r = SD.get((F, grp, win))
                        if r is None:
                            continue
                        if gtype.startswith('emp'):
                            pool = gtype[-1]
                            if f'gemp_s_{pool}' not in r:
                                continue
                            gv = np.array([r[f'gemp_s_{pool}'], 0.0, 0.0])
                            gd = gv[None, :] + np.c_[rng.normal(0, r[f'gemp_s_{pool}_se'], len(dr)), np.zeros((len(dr), 2))]
                        elif gtype == 'model':
                            gv = np.array([r[f'gmod_{a}'] for a in AX])
                            gd = r['_c_draws'][rng.integers(0, len(r['_c_draws']), len(dr))][:, None] * r['_gm'][None, :]
                        elif gtype == 'bound':
                            gv = np.array([r[f'gbound_{a}'] for a in AX])
                            gd = np.repeat(gv[None, :], len(dr), 0)
                        else:   # hybrid: empirical along strike (where the patch has true spread), model / bound for d, n
                            if 'gemp_s_F' not in r:
                                continue
                            src = 'gmod' if gtype == 'hyb_model' else 'gbound'
                            gv = np.array([r['gemp_s_F'], r[f'{src}_d'], r[f'{src}_n']])
                            if src == 'gmod':
                                dn = r['_c_draws'][rng.integers(0, len(r['_c_draws']), len(dr))][:, None] * r['_gm'][None, 1:]
                            else:
                                dn = np.repeat(gv[None, 1:], len(dr), 0)
                            gd = np.c_[gv[0] + rng.normal(0, r['gemp_s_F_se'], len(dr)), dn]
                        contrib = sum(gv[k] * A[(F, a)][0] for k, a in enumerate(AX))
                        tot += contrib
                        tot_d += sum(gd[:, k] * A[(F, a)][1] for k, a in enumerate(AX))
                        per_f[F] = contrib * 100
                    synth.append(dict(group=grp, cfg=cfg, stacks=vname, g_type=gtype, pred_transient_pct=tot * 100,
                                      pred_se=tot_d.std() * 100, pred_lo95=np.percentile(tot_d, 2.5) * 100,
                                      pred_hi95=np.percentile(tot_d, 97.5) * 100,
                                      obs_transient_pct=f.m[6] * 100, obs_se=B[vname][:, 6].std() * 100,
                                      pred_over_obs=tot / f.m[6], **{f'pred_fam{F}_pct': v for F, v in per_f.items()}))
            # expected change of the transient from the re-weighting IF the location hypothesis were true (bound g)
            for gtype in ['model', 'bound']:
                dd = 0.0
                for F in FAMS:
                    r = SD.get((F, grp, win))
                    if r is None:
                        continue
                    gv = np.array([r[f'gmod_{a}' if gtype == 'model' else f'gbound_{a}'] for a in AX])
                    dC = CENT_MATCH_T[F] - CENT_LOC[F]
                    dd += sum(gv[k] * synth_transient(fits['loc'], pl, {F: dC[:, k]}) for k in range(3))
                res.append(dict(group=grp, cfg=cfg, variant=f'match-loc_expected_if_location_{gtype}', transient60_pct=dd * 100))
            print(grp, cfg, 'done', flush=True)
    R = pd.DataFrame(res)
    R.to_csv(f'{OUT}/task3_matched_transient.csv', index=False)
    SY = pd.DataFrame(synth)
    SY.to_csv(f'{OUT}/task4_predicted_transient.csv', index=False)
    LS = pd.DataFrame(locsig)
    LS.to_csv(f'{OUT}/task1_location_transient.csv', index=False)
    print(R.round(4).to_string())
    print(LS.round(2).to_string())
    print(SY.round(4).to_string())
    pickle.dump(dict(SD={k: {kk: vv for kk, vv in v.items()} for k, v in SD.items()}),
                open(f'{OUT}/_sens_draws.pkl', 'wb'))
    return dict(SEP=SEP, INFO=INFO, SENS=SENS, R=R, SY=SY, LS=LS, STK=pd.concat(STK.values()))
