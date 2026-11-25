#!/usr/bin/env python3
"""sci_extras_trigger: has remote dynamic triggering of MTJ LFEs changed through time?
(closeness-to-failure proxy: triggered excess per unit dynamic stress before 2021-12-20, between
Petrolia and the M7, and after the M7.)

Inputs (all on disk, no downloads):
  data/sci/dyn_trigger/per_quake_pgv_stress_triggering.csv  measured PGV / dynamic stress sigma_d (BK.JCC/HOPS)
  data/sci/remote_check/catalog_classified.parquet          'real' flag + Z per catalog event
  scripts/sci_cascades2.py                                  burst onsets (2-h gap), tidal-phase control shifts

Counts: LFE burst onsets (real only; 2-h family gap) in [t_surf, t_surf + 1 h) (Love 4 km/s), as in
sci_dyn_trigger. Expected = mean over 300 tide-phase/time-of-day matched control windows (+-3..60 d)
drawn ONLY from the same epoch, outside the M7 silence (M7..M7+30 d) and outside 3 d after any local
M>=5.5 (<200 km). Teleseisms in the silence / 3 d after local quakes / overlapped by a larger
teleseism (2 h) / without measured PGV are dropped.

Variants: real onsets (baseline), real Z>=3 onsets, real meancc>=0.137 onsets (constant completeness),
real events (not onsets), 0.5-h window.

Tests (strong = sigma_d >= S0 = 0.63 kPa, the sci_dyn_trigger threshold):
  * per-epoch R = O/E (strong, weak), excess X = (O-E)/E; Poisson GLM log mu = log E + a + b log10 sigma_d
    per epoch; binned ratio vs stress per epoch; per-epoch step threshold S0 profile.
  * change statistic D = log((O_ep+0.5)/E_ep) - log((O_pre+0.5)/E_pre) on strong quakes:
      - permutation null: epoch labels shuffled among strong quakes within sigma_d quartile strata;
      - placebo: same statistic for every 3-yr (and 1.5-yr) window inside 2009-2021 vs the rest of the pre epoch;
      - quake bootstrap CI.
  * injection / power: synthetic catalogues: background count of each quake = one random control-window
    count (empirical, bursty); triggered count ~ NegBin(mean = E*(R_pre-1)*f, dispersion fitted to pre
    strong quakes) with f = 1 before and f in a grid for the test epoch. Detection limit = f at which
    the two-sided 5% test (vs f=1 sims) rejects 80% of the time; Neyman 95% CI on f for the observed D.
Outputs: data/sci/extras/trigger/*.csv, summary.json; figures/sci_extras_trigger.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats

sys.path.insert(0, 'scripts')
import sci_cascades2 as C  # noqa: E402

OUT = 'data/sci/extras/trigger'
os.makedirs(OUT, exist_ok=True)
EPOCH = pd.Timestamp('1970-01-01')
T_PET = C.ts('2021-12-20 20:10:00')
T_M7 = C.ts('2024-12-05 18:44:00')
SIL = 30 * 86400            # M7 silence (restart bulk at 17-26 d)
LOCAL_EXCL = 3 * 86400
S0 = 0.63                   # kPa, sci_dyn_trigger step threshold
EPN = ['pre2021', 'pet_m7', 'post_m7']
rng0 = np.random.default_rng(2026)


def epoch_of(t):
    return np.where(t < T_PET, 0, np.where(t < T_M7, 1, 2))


def bad_time(t, local):
    b = (t >= T_M7) & (t < T_M7 + SIL)
    for tl in local:
        b |= (t >= tl) & (t < tl + LOCAL_EXCL)
    return b


def load():
    T = pd.read_csv('data/sci/dyn_trigger/per_quake_pgv_stress_triggering.csv', low_memory=False)
    L = pd.read_csv('data/catalogs/comcat_M5.5_200km_2008_2026.csv')
    local = ((pd.to_datetime(L.time, utc=True).dt.tz_localize(None) - EPOCH).dt.total_seconds()).values
    m = pd.read_parquet('data/sci/remote_check/catalog_classified.parquet', columns=['t', 'idnum', 'real', 'meancc', 'Z'])
    C.T_END = m.t.max()
    m = m[m.t >= C.ts(C.T_START)]
    r = m[m.real == 1]
    cats = {'real_onsets': np.sort(C.make_bursts(r[['t', 'idnum']]).onset.values),
            'realZ3_onsets': np.sort(C.make_bursts(r[r.Z >= 3][['t', 'idnum']]).onset.values),
            'realcc137_onsets': np.sort(C.make_bursts(r[r.meancc >= 0.137][['t', 'idnum']]).onset.values),
            'real_events': np.sort(r.t.values)}
    return T, local, cats, m.t.max()


def controls(ts_, local, t_end, n=300, rng=None):
    """per quake n control start times: same-epoch tidal shifts, outside silence / local-aftershock windows"""
    ep = epoch_of(ts_)
    lo = np.array([C.ts(C.T_START), T_PET, T_M7 + SIL])
    hi = np.array([T_PET, T_M7, t_end - 86400])
    S = C.SHIFTS
    out = np.full((len(ts_), n), np.nan)
    for i, t in enumerate(ts_):
        c = t + S
        ok = (c >= lo[ep[i]]) & (c + 3600 <= hi[ep[i]]) & ~bad_time(c, local) & ~bad_time(c + 3600, local)
        cand = c[ok]
        if len(cand) >= 10:
            out[i] = rng.choice(cand, n, replace=True)
    return out


def cnt(arr, a, b):
    return np.searchsorted(arr, b) - np.searchsorted(arr, a)


def counts(T, cat, ctrl, H=3600.0, off=None):
    off = np.zeros(len(T)) if off is None else off
    t0 = T.t_surf.values + off
    o = cnt(cat, t0, t0 + H)
    c0 = np.nan_to_num(ctrl) + off[:, None]
    cc = cnt(cat, c0, c0 + H).astype(float)
    cc[np.isnan(ctrl)] = np.nan
    return o, cc


def pois_ll(o, mu):
    mu = np.maximum(mu, 1e-12)
    return np.sum(o * np.log(mu) - mu)


def glm_fit(o, e, S):
    f = lambda p: -pois_ll(o, e * np.exp(p[0] + p[1] * S))
    r = optimize.minimize(f, [0.0, 0.3], method='Nelder-Mead', options=dict(xatol=1e-6, fatol=1e-8, maxiter=4000))
    return r.x, -r.fun


def logR(o, e):
    return np.log((o.sum() + 0.5) / max(e.sum(), 1e-9))


def dstat(o, e, ep, k):
    a, b = ep == 0, ep == k
    return logR(o[b], e[b]) - logR(o[a], e[a])


def step_S0_profile(o, e, S, grid):
    """mu = e*r_lo below S0, e*r_hi above; profile LL over S0 (log10 kPa)"""
    ll = []
    for g in grid:
        hi = S >= g
        L = 0.0
        for msk in (hi, ~hi):
            if msk.sum() == 0 or e[msk].sum() == 0:
                continue
            r = max(o[msk].sum(), 1e-9) / e[msk].sum()
            L += pois_ll(o[msk], e[msk] * r)
        ll.append(L)
    return np.array(ll)


def nb_disp(o, mu):
    """method-of-moments NB dispersion k (var = mu + mu^2/k) for counts o with means mu"""
    num = np.sum(mu ** 2)
    den = np.sum((o - mu) ** 2 - o)
    return np.inf if den <= 0 else num / den


def main():
    T, local, cats, t_end = load()
    T['t_surf'] = T.t_surf.astype(float)
    T['epoch'] = epoch_of(T.t_surf.values)
    T['S'] = np.log10(T.sigma_d_kPa)
    keep = T.sigma_d_kPa.notna() & ~T.overlap_larger.astype(bool) & ~bad_time(T.t_surf.values, local) & \
        ~bad_time(T.t_surf.values + 3600, local) & (T.t_surf.values + 3600 < t_end)
    T['dropped'] = np.where(T.sigma_d_kPa.isna(), 'no_pgv', np.where(T.overlap_larger.astype(bool), 'overlap',
                            np.where(~keep, 'silence_local_or_end', '')))
    U = T[keep].reset_index(drop=True).copy()
    ctrl = controls(U.t_surf.values, local, t_end, rng=rng0)
    okc = ~np.isnan(ctrl).all(1)
    U = U[okc].reset_index(drop=True)
    ctrl = ctrl[okc]
    ep = U.epoch.values
    S = U.S.values
    strong = U.sigma_d_kPa.values >= S0
    summ = dict(n_teleseisms_total=int(len(T)), dropped=T.dropped.value_counts().to_dict(), n_used=int(len(U)),
                epochs=dict(pre2021='2009-01-01..2021-12-20', pet_m7='2021-12-20..2024-12-05',
                            post_m7='2025-01-04..2026-06-29 (after 30-d silence)'),
                S0_kPa=S0, catalog_end=str(EPOCH + pd.Timedelta(seconds=float(t_end))))
    rows = []
    res_var = {}
    for vname, cat in cats.items():
        for H, wtag in [(3600.0, ''), (1800.0, '_0.5h'), (1800.0, '_peak-10to+20min')]:
            if wtag and vname != 'real_onsets':
                continue
            tag = vname + wtag
            off = (U.bb_t_pk.fillna(U.t_surf).values - U.t_surf.values - 600.0) if 'peak' in wtag else None
            o, cc = counts(U, cat, ctrl, H, off)
            e = np.nanmean(cc, 1)
            if tag == 'real_onsets':
                U['obs'], U['exp'] = o, e
                U['pctl'] = np.nanmean(cc < o[:, None], 1) + 0.5 * np.nanmean(cc == o[:, None], 1)
                CC0 = cc
            R = {}
            for k, en in enumerate(EPN):
                for sn, sm in [('strong', strong), ('weak', ~strong)]:
                    s = (ep == k) & sm
                    # control-draw null for the epoch ratio
                    jj = rng0.integers(0, cc.shape[1], (2000, s.sum()))
                    ccs = cc[s]
                    nul = np.nansum(ccs[np.arange(s.sum())[None, :], jj], 1) if s.sum() else np.zeros(2000)
                    R[f'{en}|{sn}'] = dict(n_quakes=int(s.sum()), obs=int(o[s].sum()), exp=round(float(e[s].sum()), 2),
                                           ratio=round(float(o[s].sum() / e[s].sum()), 3) if e[s].sum() > 0 else None,
                                           p_upper=round(float((np.sum(nul >= o[s].sum()) + 1) / 2001), 4),
                                           p_lower=round(float((np.sum(nul <= o[s].sum()) + 1) / 2001), 4))
                    rows.append(dict(variant=tag, epoch=en, subset=sn, **R[f'{en}|{sn}']))
            # GLM per epoch, all quakes with e>0
            g = {}
            for k, en in enumerate(EPN):
                s = (ep == k) & (e > 0)
                p, _ = glm_fit(o[s], e[s], S[s])
                # bootstrap slope
                bs = []
                idx = np.where(s)[0]
                for _ in range(300):
                    b = rng0.choice(idx, len(idx))
                    bs.append(glm_fit(o[b], e[b], S[b])[0])
                bs = np.array(bs)
                g[en] = dict(a=round(float(p[0]), 3), slope_per_log10kPa=round(float(p[1]), 3),
                             slope_ci95=[round(float(x), 3) for x in np.quantile(bs[:, 1], [.025, .975])],
                             ratio_at_1kPa=round(float(np.exp(p[0])), 3),
                             ratio_at_1kPa_ci95=[round(float(x), 3) for x in np.exp(np.quantile(bs[:, 0], [.025, .975]))])
            res_var[tag] = dict(ratios=R, glm=g)
    pd.DataFrame(rows).to_csv(f'{OUT}/epoch_ratios.csv', index=False)
    summ['variants'] = res_var

    # ------------------------------------------------ main tests on baseline (real onsets, 1 h)
    o, e, cc = U.obs.values, U.exp.values, CC0
    sidx = np.where(strong & (e > 0))[0]
    os_, es_, eps_, Ss_ = o[sidx], e[sidx], ep[sidx], S[sidx]
    qbins = np.digitize(Ss_, np.quantile(Ss_, [.25, .5, .75]))
    tests = {}
    for k in (1, 2):
        D = dstat(os_, es_, eps_, k)
        # permutation within stress strata
        perm = []
        for _ in range(5000):
            lab = eps_.copy()
            for q in range(4):
                m_ = qbins == q
                lab[m_] = rng0.permutation(lab[m_])
            perm.append(dstat(os_, es_, lab, k))
        perm = np.array(perm)
        # quake bootstrap within epochs
        bt = []
        ia, ib = np.where(eps_ == 0)[0], np.where(eps_ == k)[0]
        for _ in range(5000):
            a, b = rng0.choice(ia, len(ia)), rng0.choice(ib, len(ib))
            bt.append(logR(os_[b], es_[b]) - logR(os_[a], es_[a]))
        bt = np.array(bt)
        tests[EPN[k]] = dict(D_logratio=round(float(D), 3), ratio_change_factor=round(float(np.exp(D)), 3),
                             boot_ci95_factor=[round(float(x), 3) for x in np.exp(np.quantile(bt, [.025, .975]))],
                             perm_p_two_sided=round(float((np.sum(np.abs(perm - perm.mean()) >= abs(D - perm.mean())) + 1) / 5001), 4),
                             perm_pctl=round(float((perm < D).mean() * 100), 1),
                             n_strong=int((eps_ == k).sum()), n_strong_pre=int((eps_ == 0).sum()))
    # combined post-2021 (epochs 1+2) vs pre
    D12 = logR(os_[eps_ > 0], es_[eps_ > 0]) - logR(os_[eps_ == 0], es_[eps_ == 0])
    perm = []
    for _ in range(5000):
        lab = eps_.copy()
        for q in range(4):
            m_ = qbins == q
            lab[m_] = rng0.permutation(lab[m_])
        perm.append(logR(os_[lab > 0], es_[lab > 0]) - logR(os_[lab == 0], es_[lab == 0]))
    perm = np.array(perm)
    tests['post2021_all'] = dict(D_logratio=round(float(D12), 3), ratio_change_factor=round(float(np.exp(D12)), 3),
                                 perm_pctl=round(float((perm < D12).mean() * 100), 1),
                                 perm_p_two_sided=round(float((np.sum(np.abs(perm - perm.mean()) >= abs(D12 - perm.mean())) + 1) / 5001), 4))
    # leave-one-quake-out (strong) for epoch D: most influential quakes
    for k in (1, 2):
        lo = []
        for j in np.where((eps_ == k) | (eps_ == 0))[0]:
            mm = np.ones(len(os_), bool)
            mm[j] = False
            lo.append(dstat(os_[mm], es_[mm], eps_[mm], k))
        tests[EPN[k]]['leave_one_out_factor_range'] = [round(float(np.exp(min(lo))), 3), round(float(np.exp(max(lo))), 3)]
    # placebo windows within the pre epoch: W-yr windows vs rest of pre epoch
    pl_rows = []
    tpre = U.t_surf.values[sidx]
    for W in (3.0, 1.5):
        starts = np.arange(C.ts('2009-01-01'), T_PET - W * C.YR + 1, 0.25 * C.YR)
        for t0 in starts:
            inw = (eps_ == 0) & (tpre >= t0) & (tpre < t0 + W * C.YR)
            out = (eps_ == 0) & ~inw
            pl_rows.append(dict(W_yr=W, start=str(EPOCH + pd.Timedelta(seconds=float(t0)))[:10], n=int(inw.sum()),
                                D=logR(os_[inw], es_[inw]) - logR(os_[out], es_[out]),
                                O=int(os_[inw].sum()), E=float(es_[inw].sum())))
    PL = pd.DataFrame(pl_rows)
    PL.to_csv(f'{OUT}/placebo_windows.csv', index=False)
    for k, W in [(1, 3.0), (2, 1.5)]:
        pw = PL[PL.W_yr == W].D.values
        tests[EPN[k]][f'placebo_{W}yr_pctl'] = round(float((pw < tests[EPN[k]]['D_logratio']).mean() * 100), 1)
        tests[EPN[k]][f'placebo_{W}yr_D_range'] = [round(float(pw.min()), 3), round(float(pw.max()), 3)]
        tests[EPN[k]][f'placebo_{W}yr_n_windows'] = int(len(pw))
    summ['tests_strong_real_onsets_1h'] = tests

    # ------------------------------------------------ robust statistics (heavy-tailed counts: one coincident
    # spontaneous network episode adds 10-20 onsets). Per quake mid-p PIT of obs among its own controls and
    # the binary 'any onset' indicator vs control P(any).
    ccs_all = cc[sidx]
    srt = [np.sort(r_[~np.isnan(r_)]) for r_ in ccs_all]
    pany = np.array([np.mean(r_ > 0) for r_ in srt])

    def pit(ovec):
        return np.array([(np.searchsorted(srt[i], ovec[i], 'left') + np.searchsorted(srt[i], ovec[i], 'right')) / 2 / len(srt[i])
                         for i in range(len(ovec))])

    def robust_stats(ovec, lab, k):
        P_ = pit(ovec)
        A_ = (ovec > 0).astype(float)
        a, b = lab == 0, lab == k
        return P_[b].mean() - P_[a].mean(), (A_[b] - pany[b]).mean() - (A_[a] - pany[a]).mean()
    PIT = pit(os_)
    rob = {}
    for k in (0, 1, 2):
        b = eps_ == k
        rob[EPN[k]] = dict(n=int(b.sum()), mean_PIT=round(float(PIT[b].mean()), 3),
                           mean_PIT_null_sd=round(float(np.sqrt(1 / 12 / b.sum())), 3),
                           n_any=int((os_[b] > 0).sum()), exp_any=round(float(pany[b].sum()), 2),
                           excess_any_frac=round(float(((os_[b] > 0) - pany[b]).mean()), 3))
    for k in (1, 2):
        dP, dA = robust_stats(os_, eps_, k)
        pp = []
        for _ in range(3000):
            lab = eps_.copy()
            for q in range(4):
                m_ = qbins == q
                lab[m_] = rng0.permutation(lab[m_])
            pp.append(robust_stats(os_, lab, k))
        pp = np.array(pp)
        rob[EPN[k]].update(dPIT_vs_pre=round(float(dP), 3), dPIT_perm_pctl=round(float((pp[:, 0] < dP).mean() * 100), 1),
                           dAny_vs_pre=round(float(dA), 3), dAny_perm_pctl=round(float((pp[:, 1] < dA).mean() * 100), 1))
    summ['robust_strong_real_onsets_1h'] = rob

    # ------------------------------------------------ threshold per epoch
    grid = np.linspace(-1.0, 1.0, 41)
    th = {}
    for k, en in enumerate(EPN):
        s = (ep == k) & (e > 0)
        ll = step_S0_profile(o[s], e[s], S[s], grid)
        best = grid[np.argmax(ll)]
        ok_ = grid[ll >= ll.max() - 1.92]
        # ratio above/below fixed S0
        hi, lo_ = s & strong, s & ~strong
        th[en] = dict(S0_best_kPa=round(float(10 ** best), 3), S0_LR95_kPa=[round(float(10 ** ok_.min()), 3), round(float(10 ** ok_.max()), 3)],
                      ratio_above_0p63=round(float(o[hi].sum() / e[hi].sum()), 3), ratio_below_0p63=round(float(o[lo_].sum() / e[lo_].sum()), 3))
    summ['threshold_per_epoch'] = th
    # binned ratio vs stress per epoch
    edges = [0, 0.2, 0.5, 1, 2, 5, 100]
    br = []
    for k, en in enumerate(EPN):
        for a, b in zip(edges[:-1], edges[1:]):
            s = (ep == k) & (U.sigma_d_kPa.values >= a) & (U.sigma_d_kPa.values < b)
            br.append(dict(epoch=en, lo_kPa=a, hi_kPa=b, n=int(s.sum()), obs=int(o[s].sum()), exp=round(float(e[s].sum()), 2),
                           ratio=round(float(o[s].sum() / e[s].sum()), 3) if e[s].sum() > 0 else np.nan))
    pd.DataFrame(br).to_csv(f'{OUT}/binned_ratio_vs_stress_by_epoch.csv', index=False)

    # ------------------------------------------------ injection / power (strong quakes)
    pre = eps_ == 0
    Rpre = os_[pre].sum() / es_[pre].sum()
    trig_mu_pre = es_[pre] * (Rpre - 1)
    kdisp = nb_disp(os_[pre], es_[pre] * Rpre)
    ccs = cc[sidx]

    def simulate(f, k, nsim=2000, rng=None):
        out = np.empty(nsim)
        nq = len(os_)
        valid = [np.where(~np.isnan(ccs[i]))[0] for i in range(nq)]
        bg = np.array([ccs[i, rng.choice(valid[i], nsim)] for i in range(nq)])     # (nq, nsim)
        fac = np.where(eps_ == k, f, 1.0)
        mu = (es_ * (Rpre - 1) * fac)[:, None] * np.ones((1, nsim))
        if np.isfinite(kdisp):
            p = kdisp / (kdisp + np.maximum(mu, 1e-12))
            tr = rng.negative_binomial(kdisp, p)
        else:
            tr = rng.poisson(mu)
        osim = bg + tr
        out2 = np.empty((nsim, 2))
        for j in range(nsim):
            out[j] = dstat(osim[:, j], es_, eps_, k)
            out2[j] = robust_stats(osim[:, j], eps_, k)
        return out, out2
    fgrid = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0]
    inj = []
    power = {}
    for k in (1, 2):
        sims2 = {f: simulate(f, k, rng=np.random.default_rng(100 + k)) for f in fgrid}
        sims = {f: v[0] for f, v in sims2.items()}
        rsim = {f: v[1][:, 0] for f, v in sims2.items()}
        n0 = sims[1.0]
        r0 = rsim[1.0]
        rl, rh = np.quantile(r0, [.025, .975])
        dPobs = rob[EPN[k]]['dPIT_vs_pre']
        lo_q, hi_q = np.quantile(n0, [.025, .975])
        Dobs = tests[EPN[k]]['D_logratio']
        inside = []
        for f in fgrid:
            sm = sims[f]
            pw = float(np.mean((sm < lo_q) | (sm > hi_q)))
            ql, qh = np.quantile(sm, [.025, .975])
            inside.append(ql <= Dobs <= qh)
            rp = rsim[f]
            inj.append(dict(epoch=EPN[k], f_excess_factor=f, median_D=float(np.median(sm)), D_q025=float(ql), D_q975=float(qh),
                            power_vs_f1=pw, ratio_factor_median=float(np.exp(np.median(sm))),
                            median_dPIT=float(np.median(rp)), dPIT_q025=float(np.quantile(rp, .025)), dPIT_q975=float(np.quantile(rp, .975)),
                            power_dPIT_vs_f1=float(np.mean((rp < rl) | (rp > rh))),
                            dPIT_obs_inside95=bool(np.quantile(rp, .025) <= dPobs <= np.quantile(rp, .975))))
        I = pd.DataFrame([r for r in inj if r['epoch'] == EPN[k]])
        dec = I[I.f_excess_factor < 1]
        inc = I[I.f_excess_factor > 1]
        lim_dec = dec[dec.power_vs_f1 >= 0.8].f_excess_factor.max() if (dec.power_vs_f1 >= 0.8).any() else None
        lim_inc = inc[inc.power_vs_f1 >= 0.8].f_excess_factor.min() if (inc.power_vs_f1 >= 0.8).any() else None
        cons = [f for f, ins in zip(fgrid, inside) if ins]
        consP = I[I.dPIT_obs_inside95].f_excess_factor.tolist()
        decP = dec[dec.power_dPIT_vs_f1 >= 0.8].f_excess_factor
        incP = inc[inc.power_dPIT_vs_f1 >= 0.8].f_excess_factor
        power[EPN[k] + '_PIT'] = dict(observed_dPIT=dPobs, sim_null_f1_pctl=round(float((r0 < dPobs).mean() * 100), 1),
                                      detectable_decrease_f_le=float(decP.max()) if len(decP) else None,
                                      detectable_increase_f_ge=float(incP.min()) if len(incP) else None,
                                      neyman95_f_consistent=[min(consP), max(consP)] if consP else None)
        power[EPN[k]] = dict(observed_D=Dobs, sim_null_f1_pctl_of_observed=round(float((n0 < Dobs).mean() * 100), 1),
                             detectable_decrease_f_le=lim_dec, detectable_increase_f_ge=lim_inc,
                             neyman95_f_consistent=[min(cons), max(cons)] if cons else None,
                             f_grid=fgrid)
    pd.DataFrame(inj).to_csv(f'{OUT}/injection_power.csv', index=False)
    summ['injection'] = dict(R_pre_strong=round(float(Rpre), 3), nb_dispersion_k=None if not np.isfinite(kdisp) else round(float(kdisp), 3),
                             note='f = multiplier on triggered EXCESS (R-1) in the test epoch; background = random matched control count',
                             **power)
    # per-quake table
    cols = ['id', 'time', 'mag', 'place', 'name', 'dist_km', 'pgv_um_s', 'sigma_d_kPa', 'epoch', 'obs', 'exp', 'pctl']
    U['epoch'] = [EPN[k] for k in U.epoch]
    U[cols].to_csv(f'{OUT}/per_quake.csv', index=False)
    json.dump(summ, open(f'{OUT}/summary.json', 'w'), indent=1, default=lambda x: x if not isinstance(x, (np.integer, np.floating)) else x.item())
    print(json.dumps(summ, indent=1, default=str)[:6000])

    # ------------------------------------------------ figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    col = ['k', 'C0', 'C3']
    B = pd.DataFrame(br)
    for k, en in enumerate(EPN):
        b = B[(B.epoch == en) & (B.n > 0)]
        xm = np.sqrt(np.maximum(b.lo_kPa, 0.1) * np.minimum(b.hi_kPa, 50))
        ax[0].errorbar(xm * (1 + 0.08 * (k - 1)), b.ratio, yerr=np.sqrt(b.obs + 1) / b.exp.clip(1e-3), fmt='o-', color=col[k],
                       label=f'{en} ({(U.epoch == en).sum()} q)')
    ax[0].axhline(1, color='grey', lw=.8)
    ax[0].axvline(S0, color='grey', ls=':')
    ax[0].set_xscale('log')
    ax[0].set_xlabel('dynamic stress sigma_d (kPa)')
    ax[0].set_ylabel('obs / expected (real onsets, 1 h)')
    ax[0].set_ylim(0, 12)
    ax[0].legend(fontsize=8)
    ax[0].set_title('triggering ratio vs stress by epoch')
    tq = pd.to_datetime(U.time)
    sel = U.sigma_d_kPa >= S0
    ax[1].scatter(tq[sel], U.obs[sel] - U.exp[sel], s=10 + 20 * np.log10(U.sigma_d_kPa[sel] / S0 + 1) * 5, c='C1', alpha=.7)
    ax[1].axhline(0, color='grey', lw=.8)
    for t_ in ['2021-12-20', '2024-12-05']:
        ax[1].axvline(pd.Timestamp(t_), color='C3', ls='--')
    ax[1].set_ylabel('obs - expected onsets (strong quakes)')
    ax[1].set_title(f'per-quake excess, sigma_d >= {S0} kPa')
    I = pd.DataFrame(inj)
    for k, c in [(1, 'C0'), (2, 'C3')]:
        ii = I[I.epoch == EPN[k]]
        ax[2].plot(ii.f_excess_factor, ii.power_vs_f1, 'o-', color=c, label=f'{EPN[k]} power')
        ax[2].axvline(np.exp(0) if False else 1, color='grey', lw=.5)
    ax[2].axhline(0.8, color='grey', ls=':')
    ax[2].set_xlim(-0.2, 10.5)
    ax[2].set_xlabel('injected factor on triggered excess f')
    ax[2].set_ylabel('P(reject f=1 at 5%)')
    ax[2].legend(fontsize=8)
    ax[2].set_title('injection power (strong quakes)')
    fig.tight_layout()
    fig.savefig('figures/sci_extras_trigger.png', dpi=120)


if __name__ == '__main__':
    main()
