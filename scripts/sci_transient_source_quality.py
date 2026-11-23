#!/usr/bin/env python3
"""Event-mix (quality / size) confound for the early-coda 60-d M7 transient (families 23, 8, 27).

Quality variables per event: log10 A (relative amplitude, data/sci/moments/event_amplitudes.csv.gz;
qc_bad or A<=0 -> NaN), catalog meancc_v2 and cc/thr (cc_div_thr_v2).
  --prep     1) distributions: isolated events of the first 60 d after restart vs reference (pre-M7 year +
                >182 d after the M7), bootstrap SE, KS test, placebo (random pre-M7 60-d windows vs their
                preceding year); split / stack definitions for sci_transient_source_measure.py --prefix quality_:
                time-matched pre-M7 splits by each variable (+20 random-split nulls), and the ORIGINAL dv/v stacks
                unweighted ('qbase') and with post-M7 (first 182 d) events weighted to the reference
                (log10A x meancc) tercile histogram ('qmatch').
  --analyze  2) qbase vs qmatch transient (same group fit, paired block bootstrap over stacks)
             3) sensitivity h (apparent dv/v per unit of each variable, joint 3x3 solve of the three splits)
                x the transient-shaped quality signal of the stacks (same parametric fit) -> predicted transient.
Outputs: data/sci/transient_source/quality_*.csv, quality_summary.json, figures/sci_transient_source_quality.png
"""
import json
import os
import pickle
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sci_transient_source import BLOCK, blocks_split  # noqa: E402
from sci_transient_source_common import DAY, FAMS, M7, OUT, make_stacks, periods  # noqa: E402

QV = ['log10A', 'meancc', 'ccthr']
NNULL = 20
rng = np.random.default_rng(99)
P = f'{OUT}/quality_'


def table(F):
    T = pd.read_parquet(f'{OUT}/fam{F}_events.parquet')
    a = pd.read_csv('data/sci/moments/event_amplitudes.csv.gz')
    a = a[a.fam == F].set_index('ev')
    A = a.A.reindex(T.ev).values
    bad = a.qc_bad.reindex(T.ev).fillna(True).values.astype(bool)
    T['log10A'] = np.where((A > 0) & ~bad, np.log10(np.where(A > 0, A, 1)), np.nan)
    with h5py.File(f'data/waveforms/fam{F}.h5', 'r') as h:
        T['meancc'] = h['events/meancc_v2'][:]
        T['ccthr'] = h['events/cc_div_thr_v2'][:]
    with h5py.File(f'data/waveforms/fam{F}.h5', 'r') as h:
        pass
    restart = np.sort(T.t.values[T.t.values >= M7])[9]
    T.attrs['restart'] = restart
    T['dr'] = (T.t - restart) / DAY
    return T


def distributions(F, T):
    Pm = periods(T)
    iso = T.iso.values
    ref = (Pm['pre1y'] | Pm['post_late']) & iso
    rows = []
    for pn in ['post_early60', 'post_60_182', 'post_late', 'pre1y']:
        m = Pm[pn] & iso
        for v in QV:
            x, y = T.loc[m, v].dropna().values, T.loc[ref & ~m, v].dropna().values
            if pn in ('post_late', 'pre1y'):
                y = T.loc[(Pm['pre1y'] if pn == 'post_late' else Pm['post_late']) & iso, v].dropna().values
            bs = [x[rng.integers(0, len(x), len(x))].mean() - y[rng.integers(0, len(y), len(y))].mean() for _ in range(500)]
            rows.append(dict(fam=F, period=pn, vs='ref' if pn not in ('post_late', 'pre1y') else ('pre1y' if pn == 'post_late' else 'post_late'),
                             var=v, n=len(x), n_ref=len(y), mean=x.mean(), mean_ref=y.mean(), diff=x.mean() - y.mean(),
                             diff_se=np.std(bs), median_diff=np.median(x) - np.median(y), ks_p=stats.ks_2samp(x, y).pvalue))
    # placebo: random pre-M7 60-d windows vs the preceding year
    t = T.t.values
    pl = []
    for t0 in rng.uniform(M7 - 13 * 365 * DAY, M7 - 120 * DAY, 400):
        m1 = iso & (t >= t0) & (t < t0 + 60 * DAY)
        m2 = iso & (t >= t0 - 365 * DAY) & (t < t0)
        if m1.sum() < 30:
            continue
        pl.append(dict(fam=F, t0=t0, **{v: np.nanmean(T[v].values[m1]) - np.nanmean(T[v].values[m2]) for v in QV}))
    return pd.DataFrame(rows), pd.DataFrame(pl)


def match_weights(T, stacks):
    Pm = periods(T)
    iso = T.iso.values
    R = iso & (Pm['pre1y'] | Pm['post_late']) & np.isfinite(T.log10A.values)
    ea = np.quantile(T.log10A[R], [1 / 3, 2 / 3])
    em = np.quantile(T.meancc[R], [1 / 3, 2 / 3])
    ok = np.isfinite(T.log10A.values)
    binid = np.digitize(np.nan_to_num(T.log10A.values), ea) * 3 + np.digitize(T.meancc.values, em)
    pR = np.bincount(binid[R], minlength=9) / R.sum()
    w = np.ones(len(T))
    d, dr = T.day.values, T.dr.values
    info = {}
    for k, m in {'P1': (d >= 0) & (dr < 60), 'P2': (dr >= 60) & (d < 182)}.items():
        mm = iso & m & ok
        pP = np.bincount(binid[mm], minlength=9) / max(mm.sum(), 1)
        ratio = np.where(pP > 0, pR / np.where(pP > 0, pP, 1), 1.0)
        w[mm] = np.clip(ratio[binid[mm]], 0.1, 10.0)
        ww = w[mm]
        info[k] = dict(n=int(mm.sum()), n_eff=float(ww.sum() ** 2 / (ww ** 2).sum()),
                       **{f'{v}_diff_unweighted': float(np.nanmean(T[v].values[mm]) - np.nanmean(T[v].values[R])) for v in QV},
                       **{f'{v}_diff_weighted': float(np.nansum(ww * T[v].values[mm]) / np.sum(ww * np.isfinite(T[v].values[mm]))
                                                      - np.nanmean(T[v].values[R])) for v in QV})
    return [w[np.asarray(s)] for s in stacks], w, info


def prep():
    D, PL, info = [], [], {}
    for F in FAMS:
        T = table(F)
        T[['ev', 't', 'iso', 'ovl10', 'day', 'dr'] + QV].to_parquet(f'{P}fam{F}_events.parquet')
        d, pl = distributions(F, T)
        D.append(d)
        PL.append(pl)
        te = T.t.values
        iso_idx = np.where(T.iso.values)[0]
        pre = T.iso.values & ~T.ovl10.values & (T.day.values < 0) & np.isfinite(T.log10A.values)
        pool = np.where(pre)[0]
        pool = pool[np.argsort(te[pool])]
        mem = {f'Q{v}': blocks_split(pool, T[v].values[pool]) for v in QV}
        r2 = np.random.default_rng(100 + F)
        for i in range(NNULL):
            mem[f'Qnull{i}'] = blocks_split(pool, None, r2)
        splits = [(k, v[0], v[1]) for k, v in mem.items()]
        idx = iso_idx[np.argsort(te[iso_idx])]
        stacks = make_stacks(te, idx, 50, 25)
        stack_t = np.array([te[m].mean() for m in stacks])
        weights, w_ev, minfo = match_weights(T, stacks)
        variants = {'qbase': (stacks, [None] * len(stacks), stack_t), 'qmatch': (stacks, weights, stack_t)}
        # own intermediate files (trusted), read by sci_transient_source_measure.py --prefix quality_
        pickle.dump(dict(iso_idx=iso_idx, splits=splits, variants=variants), open(f'{P}fam{F}_jobdef.pkl', 'wb'))
        pickle.dump(dict(members=mem, stacks=stacks, stack_t=stack_t, w_ev=w_ev), open(f'{P}fam{F}_defs.pkl', 'wb'))
        info[F] = dict(n_pool=len(pool), restart_day=(T.attrs['restart'] - M7) / DAY, match=minfo)
    D, PL = pd.concat(D), pd.concat(PL)
    D.to_csv(f'{P}task1_distributions.csv', index=False)
    PL.to_csv(f'{P}task1_placebo.csv', index=False)
    json.dump(info, open(f'{P}prep_info.json', 'w'), indent=1, default=float)
    sd = PL.groupby('fam')[QV].std()
    D = D.join(sd.stack().rename('placebo_sd'), on=['fam', 'var'])
    D['z_placebo'] = D['diff'] / D.placebo_sd
    D.to_csv(f'{P}task1_distributions.csv', index=False)
    print(D[D.period == 'post_early60'].round(4).to_string())
    print(json.dumps(info, indent=1, default=float))


# =================================================================== analysis
def analyze():
    from sci_dvv_depth import GroupFit
    from sci_transient_source_analysis import GROUPS, boot_counts, keep_channels, make_rows, synth_transient
    from dvv_cwire import epoch_s
    SENS, HD, CQ, ORIGP, QP, STQ = [], {}, {}, {}, {}, []
    for F in FAMS:
        T = pd.read_parquet(f'{P}fam{F}_events.parquet')
        # own intermediate file (trusted)
        Dd = pickle.load(open(f'{P}fam{F}_defs.pkl', 'rb'))
        mem = Dd['members']
        dq = {k: np.array([np.nanmean(T[v].values[hi]) - np.nanmean(T[v].values[lo]) for v in QV])
              for k, (hi, lo) in mem.items()}
        E = pd.read_parquet(f'{P}fam{F}_split_eps.parquet')
        keep = keep_channels(F)
        for grp, gsel in GROUPS.items():
            g0 = E[gsel(E.chan) & E.chan.isin(keep) & (E.cc > 0.2)]
            for win, g in g0.groupby('window'):
                sd = g[g.split.str.startswith('Qnull')].groupby('chan').eps.std().rename('sd')
                g = g.join(sd, on='chan')
                g = g[np.isfinite(g.sd) & (g.sd > 0)]
                if g.chan.nunique() < 3:
                    continue
                g = g.assign(w=1 / g.sd ** 2)
                wm = lambda x: np.sum(x.eps * x.w) / np.sum(x.w)  # noqa: E731
                eps = {s: wm(x) for s, x in g.groupby('split')}
                nul = np.array([eps[s] for s in eps if s.startswith('Qnull')])
                names = [f'Q{v}' for v in QV]
                e = np.array([eps[n] for n in names])
                M = np.array([dq[n] for n in names])
                h = np.linalg.solve(M, e)
                hd = np.array([np.linalg.solve(M, e + rng.normal(0, nul.std(), 3)) for _ in range(500)])
                HD[(F, grp, win)] = (h, hd)
                r = dict(fam=F, group=grp, window=win, n_chan=g.chan.nunique(), null_sd_pct=nul.std() * 100)
                for j, v in enumerate(QV):
                    r[f'eps_Q{v}_pct'] = e[j] * 100
                    r[f'dq_Q{v}_{v}'] = M[j, j]
                    r[f'h1_{v}_pct_per_unit'] = e[j] / M[j, j] * 100          # single-variable (confounded)
                    r[f'h_{v}_pct_per_unit'] = h[j] * 100                    # joint
                    r[f'h_{v}_se'] = hd[:, j].std() * 100
                SENS.append(r)
        # stack-level quality (original stacks)
        st, stack_t = Dd['stacks'], Dd['stack_t']
        C = np.array([[np.nanmean(T[v].values[np.asarray(m)]) for v in QV] for m in st])
        C = np.where(np.isfinite(C), C, np.nanmean(C, 0))
        CQ[F] = C
        STQ.append(pd.DataFrame(dict(fam=F, stack=np.arange(len(st)), t=pd.to_datetime(stack_t, unit='s'),
                                     **{v: C[:, j] for j, v in enumerate(QV)})))
        old = pd.read_csv(f'data/dvv/fam{F}_cwire_stacks.csv')
        p = pd.read_parquet(f'data/sci/dvv_depth/fam{F}_pairs.parquet')
        p = p[p.cfg.isin(['b28_early', 'b28_late']) & p.chan.isin(keep) & (p.cc >= 0.2)]
        ORIGP[F] = make_rows(p, epoch_s(pd.to_datetime(old.t, format='mixed')), F)
        for v in ['qbase', 'qmatch']:
            q = pd.read_parquet(f'{P}fam{F}_pairs_{v}.parquet')
            q = q[q.chan.isin(keep) & (q.cc >= 0.2)]
            QP[(F, v)] = make_rows(q, stack_t, F)
    SENS = pd.DataFrame(SENS)
    SENS.to_csv(f'{P}task3_sensitivity.csv', index=False)
    pd.concat(STQ).to_csv(f'{P}stack_quality.csv', index=False)
    print(SENS[SENS.window.isin(['early', 'late'])].round(4).T.to_string())

    res, pred, qsig = [], [], []
    for grp, gsel in GROUPS.items():
        for cfg in ['b28_early', 'b28_late']:
            win = 'early' if cfg == 'b28_early' else 'late'
            sel = lambda Pd: pd.concat([Pd[F][(Pd[F].cfg == cfg) & gsel(Pd[F].chan)] for F in FAMS], ignore_index=True)  # noqa: E731
            po = sel(ORIGP)
            pb = sel({F: QP[(F, 'qbase')] for F in FAMS})
            pm = sel({F: QP[(F, 'qmatch')] for F in FAMS})
            fits = {'orig': GroupFit(po), 'qbase': GroupFit(pb), 'qmatch': GroupFit(pm)}
            (dro,) = boot_counts([fits['orig']])
            drb, drm = boot_counts([fits['qbase'], fits['qmatch']])
            drs = {'orig': dro, 'qbase': drb, 'qmatch': drm}
            B = {v: np.array([fits[v].solve(fits[v].w * m) for m in drs[v]]) for v in fits}
            for v, f in fits.items():
                res.append(dict(group=grp, cfg=cfg, variant=v, transient60_pct=f.m[6] * 100, se=B[v][:, 6].std() * 100,
                                step_m7_pct=f.m[5] * 100, step_se=B[v][:, 5].std() * 100))
            d = (B['qmatch'][:, 6] - B['qbase'][:, 6]) * 100
            res.append(dict(group=grp, cfg=cfg, variant='qmatch-qbase', transient60_pct=(fits['qmatch'].m[6] - fits['qbase'].m[6]) * 100, se=d.std()))
            # transient-shaped quality signal of the original stacks and the predicted apparent transient
            f, dr = fits['orig'], dro
            A = {(F, v): synth_transient(f, po, {F: CQ[F][:, j]}, dr) for F in FAMS for j, v in enumerate(QV)}
            for j, v in enumerate(QV):
                x, xb = synth_transient(f, po, {F: CQ[F][:, j] for F in FAMS}, dr)
                qsig.append(dict(group=grp, cfg=cfg, var=v, quality_transient=x, se=xb.std()))
            for mode in ['joint', 'single_log10A', 'single_meancc']:
                tot, tot_d, perf = 0.0, np.zeros(len(dr)), {}
                for F in FAMS:
                    if (F, grp, win) not in HD:
                        continue
                    h, hd = HD[(F, grp, win)]
                    r = SENS[(SENS.fam == F) & (SENS.group == grp) & (SENS.window == win)].iloc[0]
                    if mode == 'joint':
                        hv, hdv = h, hd[rng.integers(0, len(hd), len(dr))]
                    else:
                        v0 = mode.split('_', 1)[1]
                        hv = np.array([r[f'h1_{v}_pct_per_unit'] / 100 if v == v0 else 0.0 for v in QV])
                        se1 = r['null_sd_pct'] / 100 / abs(r[f'dq_Q{v0}_{v0}'])
                        hdv = np.repeat(hv[None, :], len(dr), 0)
                        hdv[:, QV.index(v0)] += rng.normal(0, se1, len(dr))
                    c = sum(hv[j] * A[(F, v)][0] for j, v in enumerate(QV))
                    tot += c
                    tot_d += sum(hdv[:, j] * A[(F, v)][1] for j, v in enumerate(QV))
                    perf[F] = c * 100
                pred.append(dict(group=grp, cfg=cfg, h_mode=mode, pred_transient_pct=tot * 100, pred_se=tot_d.std() * 100,
                                 pred_lo95=np.percentile(tot_d, 2.5) * 100, pred_hi95=np.percentile(tot_d, 97.5) * 100,
                                 obs_transient_pct=f.m[6] * 100, obs_se=B['orig'][:, 6].std() * 100,
                                 **{f'pred_fam{F}_pct': c for F, c in perf.items()}))
            print(grp, cfg, 'done', flush=True)
    R, PR, QS = pd.DataFrame(res), pd.DataFrame(pred), pd.DataFrame(qsig)
    R.to_csv(f'{P}task2_matched_transient.csv', index=False)
    PR.to_csv(f'{P}task3_predicted_transient.csv', index=False)
    QS.to_csv(f'{P}task1_quality_transient.csv', index=False)
    print(R.round(4).to_string())
    print(QS.round(4).to_string())
    print(PR.round(4).to_string())
    figure(SENS, R, PR, QS)
    D = pd.read_csv(f'{P}task1_distributions.csv')
    json.dump(dict(task1_distributions=D[D.period == 'post_early60'].to_dict('records'),
                   task1_quality_transient=QS.to_dict('records'), task2=R.to_dict('records'),
                   task3_sensitivity=SENS.to_dict('records'), task3_prediction=PR.to_dict('records'),
                   match_info=json.load(open(f'{P}prep_info.json'))),
              open(f'{P}summary.json', 'w'), indent=1, default=float)


def figure(SENS, R, PR, QS):
    D = pd.read_csv(f'{P}task1_distributions.csv')
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    a = ax[0]
    for i, F in enumerate(FAMS):
        T = pd.read_parquet(f'{P}fam{F}_events.parquet')
        Pm = periods(T)
        ref = (Pm['pre1y'] | Pm['post_late']) & T.iso
        post = Pm['post_early60'] & T.iso
        bins = np.linspace(-1, 1, 41)
        a.hist(T.log10A[ref].dropna(), bins, density=True, histtype='step', color=f'C{i}', ls='--')
        a.hist(T.log10A[post].dropna(), bins, density=True, histtype='step', color=f'C{i}', lw=1.8,
               label=f'fam {F}: post {T.log10A[post].mean() - T.log10A[ref].mean():+.3f} dex')
    a.set_xlabel('log10 relative amplitude (isolated events)')
    a.set_title('first 60 d after restart (solid) vs reference (dashed)', fontsize=9)
    a.legend(fontsize=8)
    a = ax[1]
    for i, (grp, mk) in enumerate([('surface', 'o'), ('borehole', 's')]):
        for j, v in enumerate(QV):
            d = SENS[(SENS.group == grp) & (SENS.window == 'early')]
            a.errorbar(np.arange(3) + 0.1 * j + 0.4 * i, d[f'eps_Q{v}_pct'], d.null_sd_pct, fmt=mk, color=f'C{j}',
                       mfc='none' if grp == 'borehole' else f'C{j}', label=f'{v} split ({grp})')
    a.set_xticks(np.arange(3) + 0.3)
    a.set_xticklabels([f'fam {F}' for F in FAMS])
    a.axhline(0, color='0.5', lw=0.5)
    a.set_ylabel('apparent dv/v, high - low half, early coda (%)')
    a.set_title('Quality splits (time-matched, pre-M7)', fontsize=9)
    a.legend(fontsize=6)
    a = ax[2]
    x, ticks = 0, []
    for grp in ['surface', 'borehole']:
        for cfg in ['b28_early', 'b28_late']:
            for v, col in [('orig', 'k'), ('qbase', '0.4'), ('qmatch', 'C3')]:
                r = R[(R.group == grp) & (R.cfg == cfg) & (R.variant == v)].iloc[0]
                a.errorbar(x, r.transient60_pct, r.se, fmt='o', color=col, capsize=2, label=f'observed: {v}' if not ticks else None)
                x += 0.25
            for m, col, mk in [('joint', 'C0', 'D'), ('single_log10A', 'C1', 'v'), ('single_meancc', 'C2', '^')]:
                r = PR[(PR.group == grp) & (PR.cfg == cfg) & (PR.h_mode == m)].iloc[0]
                a.errorbar(x, r.pred_transient_pct, r.pred_se, fmt=mk, color=col, capsize=2,
                           label=f'predicted from event mix ({m})' if not ticks else None)
                x += 0.25
            ticks.append((x - 0.75, f'{grp}\n{cfg.replace("b28_", "")}'))
            x += 0.5
    a.set_xticks([t[0] for t in ticks])
    a.set_xticklabels([t[1] for t in ticks], fontsize=8)
    a.axhline(0, color='0.5', lw=0.5)
    a.set_ylabel('60-d M7 transient amplitude (%)')
    a.set_title('Observed (orig / re-measured / quality-matched) vs predicted from event mix', fontsize=9)
    a.legend(fontsize=6, loc='lower right')
    fig.tight_layout()
    fig.savefig('figures/sci_transient_source_quality.png', dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    if '--prep' in sys.argv:
        prep()
    if '--analyze' in sys.argv:
        analyze()
