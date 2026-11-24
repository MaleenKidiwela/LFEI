#!/usr/bin/env python3
"""Source-region dv/v: COMPOSITION CONTROL (select on the S pattern only, test on S-P).

build   : per family, every isolated LFE gets an S-ONLY relative location from its horizontal S fingerprint
          (meas/fam{F}_fingerprint.npz: 50-Hz CC lags vs the family reference on high-SNR horizontals):
          lag_c = tau + g_S,c . dx  (weighted by cc^2/(1-cc^2), cc >= 0.3, >= 5 channels).  P is never used.
          Windows: M7 pre = [M7-365 d, M7), restart = [M7, M7+60 d), post = [M7+60 d, M7+365 d);
                   Ferndale pre = [Fer-365 d, Fer), post = [Fer+60 d, Fer+365 d).
          Matched pre stacks: 1:1 nearest-neighbour (without replacement) in S-only (dx, dy, dz) to each post /
          restart event; random pre stacks of equal size as the unmatched control.
          -> comp/fam{F}_stacks_in.npz (members, bounds, names), comp/event_sloc.parquet
measure : python scripts/sci_dvv_source_region_measure.py --stacks data/sci/dvv_source_region/comp/fam{F}_stacks_in.npz
          --out meas_comp
analyze : joint inversion (sci_dvv_source_region.invert_family, sph5/sph10/loc) of the matched stacks per family;
          eps(post) - eps(pre_matched) vs eps(post) - eps(pre_random); network inverse-variance mean,
          bootstrap over families.  -> comp/comp_family.csv, comp/comp_summary.csv
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/seismic/LFE'
sys.path.insert(0, f'{ROOT}/scripts')
import sci_dvv_source_region as M  # noqa: E402

OUT = f'{ROOT}/data/sci/dvv_source_region/comp'
DAY = 86400.0


def sloc(F, G):
    z = np.load(f'{M.MEAS}/fam{F}_fingerprint.npz')
    ev, chans, L, C = z['ev'], z['chans'], z['lag'].astype(float), z['cc'].astype(float)
    g = G[G.F == F].set_index('sta')
    stas = np.array([M.sta_of(c) for c in chans])
    ok_ch = np.isin(stas, g.index)
    gS = np.array([[g.loc[s, f'gS{c}'] for c in 'xyz'] if s in g.index else [np.nan] * 3 for s in stas])
    out = np.full((len(ev), 5), np.nan)
    for i in range(len(ev)):
        m = ok_ch & np.isfinite(L[i]) & (C[i] >= 0.3) & (np.abs(L[i]) < 0.15)
        if m.sum() < 5 or len(np.unique(stas[m])) < 4:
            continue
        cc = np.clip(C[i, m], 0.3, 0.98)
        w = cc ** 2 / (1 - cc ** 2)
        A = np.c_[np.ones(m.sum()), gS[m]]
        W = np.sqrt(w)[:, None]
        x, *_ = np.linalg.lstsq(A * W, L[i, m] * W[:, 0], rcond=None)
        out[i, :4] = x
        out[i, 4] = m.sum()
    return pd.DataFrame(dict(F=F, ev=ev, tau=out[:, 0], dx=out[:, 1], dy=out[:, 2], dz=out[:, 3], nch=out[:, 4]))


def nn_match(P, Q, rng):
    """for each row of Q pick the nearest unused row of P (greedy, random order) -> indices into P."""
    used = np.zeros(len(P), bool)
    out = []
    sc = np.nanstd(P, 0) + 1e-9
    for i in rng.permutation(len(Q)):
        d = np.sqrt((((P - Q[i]) / sc) ** 2).sum(1))
        d[used] = np.inf
        j = int(np.argmin(d))
        if not np.isfinite(d[j]):
            break
        used[j] = True
        out.append(j)
    return np.array(out, int)


def stage_build(a):
    os.makedirs(OUT, exist_ok=True)
    G = pd.read_csv(f'{M.OUT}/geometry_3d.csv')
    rng = np.random.default_rng(7)
    allr = []
    for F in range(1, 28):
        if not os.path.exists(f'{M.MEAS}/fam{F}_fingerprint.npz'):
            continue
        S = sloc(F, G)
        mem = np.load(f'{M.MEAS}/fam{F}_members.npz')
        te = mem['te']
        S['t'] = te[S.ev.values]
        S = S[np.isfinite(S.dx)].reset_index(drop=True)
        allr.append(S)
        X = S[['dx', 'dy', 'dz']].values
        names, members = [], []
        for q, tq in (('m7', M.T_Q['m7']), ('fer', M.T_Q['fer'])):
            pre = np.where((S.t >= tq - 365 * DAY) & (S.t < tq))[0]
            posts = {'post': np.where((S.t >= tq + 60 * DAY) & (S.t < tq + 365 * DAY))[0]}
            if q == 'm7':
                posts['restart'] = np.where((S.t >= tq) & (S.t < tq + 60 * DAY))[0]
            for pn, post in posts.items():
                if len(post) < 15 or len(pre) < len(post) + 10:
                    continue
                k = nn_match(X[pre], X[post], rng)
                rnd = rng.choice(pre, len(k), replace=False)
                for nm, idx in ((f'{q}_{pn}', post), (f'{q}_{pn}_prematch', pre[k]), (f'{q}_{pn}_prerand', rnd),
                                (f'{q}_{pn}_preall', pre)):
                    names.append(nm)
                    members.append(S.ev.values[idx])
        if not names:
            continue
        bounds = np.cumsum([0] + [len(m_) for m_ in members])
        np.savez(f'{OUT}/fam{F}_stacks_in.npz', members=np.concatenate(members), bounds=bounds, names=np.array(names))
        print(F, len(S), {n: len(m_) for n, m_ in zip(names, members)}, flush=True)
    E = pd.concat(allr, ignore_index=True)
    E.to_parquet(f'{OUT}/event_sloc.parquet')
    # S-only centroid shifts (mean of event locations) post - pre per family
    rows = []
    for F, S in E.groupby('F'):
        for q, tq in (('m7', M.T_Q['m7']), ('fer', M.T_Q['fer'])):
            pre = S[(S.t >= tq - 365 * DAY) & (S.t < tq)]
            for pn, (t0, t1) in (('restart', (0, 60)), ('post', (60, 365))):
                if q == 'fer' and pn == 'restart':
                    continue
                po = S[(S.t >= tq + t0 * DAY) & (S.t < tq + t1 * DAY)]
                if len(po) < 10 or len(pre) < 10:
                    continue
                r = dict(F=F, event=q, window=pn, n_pre=len(pre), n_post=len(po))
                for c in ('dx', 'dy', 'dz'):
                    r[c + '_m'] = (po[c].mean() - pre[c].mean()) * 1e3
                    r[c + '_se_m'] = np.hypot(po[c].std() / np.sqrt(len(po)), pre[c].std() / np.sqrt(len(pre))) * 1e3
                rows.append(r)
    C = pd.DataFrame(rows)
    C.to_csv(f'{OUT}/sonly_centroid_shifts_family.csv', index=False)
    for (q, pn), g in C.groupby(['event', 'window']):
        s = {}
        for c in ('dx', 'dy', 'dz'):
            w = 1 / g[c + '_se_m'] ** 2
            s[c] = (np.sum(w * g[c + '_m']) / w.sum(), 1 / np.sqrt(w.sum()))
        print(q, pn, {k: (round(v[0], 1), round(v[1], 1)) for k, v in s.items()})


def stage_analyze(a):
    M.load_geo('3d')
    rows = []
    for F in range(1, 28):
        f = f'{M.OUT}/meas_comp/fam{F}.parquet'
        if not os.path.exists(f):
            continue
        names = np.load(f'{OUT}/fam{F}_stacks_in.npz')['names']
        R, stk = M.load_rows(F, f'{M.OUT}/meas_comp')
        for m in ('sph5', 'sph10', 'loc'):
            try:
                o, _ = M.invert_family(R, M.GEO[F], m)
            except np.linalg.LinAlgError:
                continue
            o['name'] = names[o['stack'].values]
            o = o.set_index('name')
            for base in [n for n in names if not ('_pre' in n)]:
                for ctrl in ('prematch', 'prerand', 'preall'):
                    c = f'{base}_{ctrl}'
                    if base not in o.index or c not in o.index:
                        continue
                    r = dict(F=F, model=m, comp=base, control=ctrl, n_post=int(stk.n[o.loc[base, 'stack']]))
                    for p in ('eps0', 'dx', 'dy', 'dz', 'tau'):
                        if p in o.columns:
                            r[p] = o.loc[base, p] - o.loc[c, p]
                            r[p + '_se'] = np.hypot(o.loc[base, p + '_se'], o.loc[c, p + '_se'])
                    rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(f'{OUT}/comp_family.csv', index=False)
    rng = np.random.default_rng(3)
    S = []
    for (m, comp, ctrl), g in T.groupby(['model', 'comp', 'control']):
        for p in ('eps0', 'dz', 'dx', 'dy'):
            if p not in g or not np.isfinite(g[p]).any():
                continue
            gg = g[np.isfinite(g[p]) & (g[p + '_se'] > 0)]
            w = 1 / gg[p + '_se'] ** 2
            mu = np.sum(w * gg[p]) / w.sum()
            bs = []
            for b in range(500):
                i = rng.integers(0, len(gg), len(gg))
                wb = w.values[i]
                bs.append(np.sum(wb * gg[p].values[i]) / wb.sum())
            sc = 100 if p == 'eps0' else 1e3
            chi = np.sum(w * (gg[p] - mu) ** 2) / max(len(gg) - 1, 1)
            S.append(dict(model=m, comp=comp, control=ctrl, param=p, n_fam=len(gg), val=mu * sc,
                          se_formal=sc / np.sqrt(w.sum()), se_boot=np.std(bs) * sc, chi2r=chi))
    S = pd.DataFrame(S)
    S['se'] = S[['se_formal', 'se_boot']].max(1)
    S.to_csv(f'{OUT}/comp_summary.csv', index=False)
    pd.set_option('display.width', 200)
    print(S.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['build', 'analyze'])
    a = ap.parse_args()
    {'build': stage_build, 'analyze': stage_analyze}[a.stage](a)


if __name__ == '__main__':
    main()
