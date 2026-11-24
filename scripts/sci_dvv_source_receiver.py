#!/usr/bin/env python3
"""Joint SOURCE-RECEIVER separation of LFE-coda dv/v (all families x all station-channels).

Model for every instrument-cleaned (V2) stack-pair row of family F on channel(-segment) C:
    pair:      d = [R_C(t_b) - R_C(t_a)] + [S_F(t_b) - S_F(t_a)] + noise
    reference: d =  R_C(t_b) + S_F(t_b) - c_{F,C}                     (free offset per family x channel)
R_C = receiver curve (near-surface), S_F = source/deep-path curve of family F.

IDENTIFIABILITY.  Adding x(t) to every S_F and subtracting it from every R_C leaves all rows unchanged, so a
common source term is in the null space.  The gauge is fixed EXPLICITLY:  sum_F S_F(t) = 0 at every epoch
(hard penalty), i.e. the fitted S_F are the identifiable family deviations dS_F = S_F - mean_F S_F, and every
R_C carries R_C + Sbar.  Damping never chooses the split (same node grid and the same lambda for R and S).
The common term Sbar is NOT identified; it is only BOUNDED (stage `common`) from the receiver curves under
stated assumptions (cross-station common mode; the smallest-signal station; PGV intercept; direct S).
Stretching about the S arrival also cancels any change on the common ballistic leg.

Solver: exact weighted LSQ by Schur complement.  Rows of one channel touch only that channel's R nodes and
offsets (block A_C, ~160 unknowns, dense Cholesky) plus S columns; the reduced S system (<= NF x nodes) is
dense.  CC weights cc^2/(1-cc^2), Huber IRLS (per-channel robust scale, k=2), 3 iterations.
S designs:  'np' piecewise-linear on the same nodes as R (61 d; 15 d for 120 d after each quake; breaks at
Petrolia, Ferndale, M7, i.e. the step at a quake is free);  'param' trend + annual + steps (Pet, Fer, M7) +
60-d M7 transient + 4-yr decline ramp from 2021-07 (+ optional placebo columns).  With 'param' the receiver
side stays fully non-parametric (R_C absorbs any step / log-healing / damage shape common to all families).

Stages (python scripts/sci_dvv_source_receiver.py STAGE [--panel balanced|all]):
  prep      load + V2 cleaning (EXCL dropped, free offsets at BREAKS2 via channel segments) -> rows cache
  np        non-parametric R_C and dS_F curves + stack-block bootstrap
  param     parametric dS_F (+ bootstrap) ; primary pre-registered test inputs
  placebo   >=50 pre-2021 shifted 'M7' dates (step+60d transient) and pre-2021 4-yr decline windows,
            identical estimator, R untouched; also random pseudo-family splits and family-label permutation
            within station-epoch
  chanfit   per-channel parametric receiver(+common) fits after removing dS_F (steps + log-healing at
            quakes, transient, decline ramp, annual, trend) + placebo dates -> common-mode bounds
  synth     synthetic recovery with the exact design: observed R_C injected (+ block-wild-bootstrap residual
            noise, correlated within station-epoch) -> leakage floor; + common and family-specific steps
  coherence inter-family coda coherence vs lapse time at shared stations -> empirical source-side weight w_S
  analyze   tests, regressions, bounds, stress sensitivity -> summary.json
  figs      figures/sci_dvv_source_receiver_*.png
Outputs: data/sci/dvv_source_receiver/
"""
import os
import sys
_NT = os.environ.get('SR_THREADS', '8')
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = _NT
import argparse
import json
import pickle          # only for this script's own caches in OUT (trusted)
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve

ROOT = '/home/seismic/LFE'
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from dvv_cwire import QUAKES, epoch_s, nodes_for, reg_matrix  # noqa: E402

OUT = 'data/sci/dvv_source_receiver'
FIG = 'figures'
os.makedirs(OUT, exist_ok=True)
DAY = 86400.0
YR = 365.25 * DAY
TQ = {k: epoch_s([v])[0] for k, v in QUAKES.items()}
T0, T1 = epoch_s(['2008-01-01'])[0], epoch_s(['2026-07-01'])[0]
T_DECL = epoch_s(['2021-07-01'])[0]
TAU_TR = 60.0
HUBER = 2.0
LAM = 300.0
PANEL_START, PANEL_END = epoch_s(['2011-02-01'])[0], epoch_s(['2025-06-01'])[0]
MIN_FAM_CH = 5
PNAMES = ['trend', 'ann_s', 'ann_c', 'step_pet', 'step_fer', 'step_m7', 'tr60_m7', 'decl4yr']
RNG = np.random.default_rng(20261123)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ============================================================================================ data
def prep():
    from dvv_cwire_pool import load_all
    from sci_dvv_instrument_check import BREAKS2, bad_mask, sta_of
    P, T = load_all(0.4)
    P = P[P.cc >= 0.2].reset_index(drop=True)
    tb = np.empty(len(P))
    ta = np.full(len(P), np.nan)
    for F, idx in P.groupby('fam').indices.items():
        a, b = P.a.values[idx], P.b.values[idx]
        tb[idx] = T[F][b]
        ta[idx] = np.where(a >= 0, T[F][np.clip(a, 0, None)], np.nan)
    keep = np.ones(len(P), bool)
    seg = np.zeros(len(P), int)
    for c, idx in P.groupby('chan').indices.items():
        has = P.a.values[idx] >= 0
        t_b, t_a = tb[idx], np.nan_to_num(ta[idx], nan=-1e18)
        bad = bad_mask(c, t_b) | (has & bad_mask(c, t_a))
        br = sorted(epoch_s([pd.Timestamp(x) for x in BREAKS2.get(sta_of(c), [])]))
        sb = np.searchsorted(br, t_b, side='right')
        sa = np.where(has, np.searchsorted(br, t_a, side='right'), sb)
        keep[idx] = ~bad & (sa == sb)
        seg[idx] = sb
    brk = P.chan.map(lambda c: sta_of(c) in BREAKS2).values
    chseg = np.where(brk, P.chan.values.astype(str) + '__seg' + seg.astype(str), P.chan.values.astype(str))
    R = pd.DataFrame(dict(chseg=chseg, chan=P.chan.values.astype(str), fam=P.fam.values.astype(np.int16),
                          a=P.a.values.astype(np.int32), b=P.b.values.astype(np.int32), eps=P.eps.values.astype(np.float32),
                          cc=P.cc.values.astype(np.float32), tb=tb, ta=ta))[keep]
    n = R.groupby('chseg').size()
    R = R[R.chseg.isin(n[n >= 1500].index)].reset_index(drop=True)
    R.to_parquet(f'{OUT}/rows_v2.parquet', index=False)
    span = R.groupby('chan').agg(t_first=('tb', 'min'), t_last=('tb', 'max'), n_rows=('tb', 'size'), n_fam=('fam', 'nunique'))
    span['panel'] = (span.t_first < PANEL_START) & (span.t_last > PANEL_END)
    span['borehole'] = span.index.str.startswith('PB.')
    span['first'] = pd.to_datetime(span.t_first, unit='s').dt.date
    span['last'] = pd.to_datetime(span.t_last, unit='s').dt.date
    span.to_csv(f'{OUT}/channels.csv')
    stk = {int(F): pd.read_csv(f'data/dvv/fam{F}_cwire_stacks.csv').shape[0] for F in sorted(R.fam.unique())}
    json.dump(dict(n_rows=int(len(R)), n_chseg=int(R.chseg.nunique()), n_fam=int(R.fam.nunique()),
                   stacks=stk), open(f'{OUT}/prep_info.json', 'w'), indent=1)
    log('prep', len(R), 'rows', R.chseg.nunique(), 'channel-segments', R.fam.nunique(), 'families')
    log(span[['first', 'last', 'n_rows', 'n_fam', 'panel']].to_string())


def load_rows(panel):
    R = pd.read_parquet(f'{OUT}/rows_v2.parquet')
    ch = pd.read_csv(f'{OUT}/channels.csv', index_col=0)
    if panel == 'balanced':
        R = R[R.chan.isin(ch.index[ch.panel])]
    elif panel == 'bk':                      # sensitivity: everything that is NOT in the balanced panel + panel
        pass
    # families measurable on this channel set
    nch = R.groupby('fam').chseg.nunique()
    fams = sorted(nch[nch >= MIN_FAM_CH].index)
    R = R[R.fam.isin(fams)].reset_index(drop=True)
    return R, fams


class Problem:
    """Pre-grouped rows per channel-segment and the node grid."""

    def __init__(self, R, fams, lam=LAM, fam_s=None, fam_c=None):
        self.segs = nodes_for(T0, T1, post_days=120.0, d_norm=61.0, d_post=15.0)
        self.nodes = np.concatenate(self.segs)
        self.NN = len(self.nodes)
        L = reg_matrix(self.segs, self.NN, 0).toarray()
        self.LtL = L.T @ L
        self.lam = lam
        self.fams = list(fams)
        self.NF = len(self.fams)
        fidx = {F: i for i, F in enumerate(self.fams)}
        self.chans = sorted(R.chseg.unique())
        R = R.copy()
        R['fi'] = R.fam.map(fidx).values
        # S labels (fam_sa, fam_sb) may differ from the offset label (pseudo-families / permutations)
        R['fsa'] = R.fi.values if fam_s is None else fam_s[0]
        R['fsb'] = R.fi.values if fam_s is None else fam_s[1]
        R['fc'] = R.fi.values if fam_c is None else fam_c
        if fam_s is not None:
            self.NF = int(max(R.fsa.max(), R.fsb.max()) + 1)
        self.R = R
        self.blocks = []
        for c, idx in R.groupby('chseg').indices.items():
            q = R.iloc[idx]
            has = (q.a.values >= 0)
            tb, ta = q.tb.values, np.where(has, q.ta.values, q.tb.values)
            cc = np.clip(q.cc.values.astype(float), 0, 0.99)
            w0 = cc ** 2 / (1 - cc ** 2)
            fl = np.unique(q.fc.values)
            lf = np.searchsorted(fl, q.fc.values)
            n = len(q)
            jb, wb = interp_w(tb, self.segs)
            ja, wa = interp_w(ta, self.segs)
            hv = has.astype(float)
            rows = np.r_[np.arange(n), np.arange(n), np.arange(n), np.arange(n), np.where(~has)[0]]
            cols = np.r_[jb, jb + 1, ja, ja + 1, self.NN + lf[~has]]
            vals = np.r_[1 - wb, wb, -(1 - wa) * hv, -wa * hv, -np.ones((~has).sum())]
            X = sp.csr_matrix((vals, (rows, cols)), shape=(n, self.NN + len(fl)))
            self.blocks.append(dict(name=c, idx=idx, X=X, d=q.eps.values.astype(float), w0=w0, has=has, tb=tb, ta=ta,
                                    fsa=q.fsa.values, fsb=q.fsb.values, fc=q.fc.values, nf=len(fl),
                                    unit_b=q.fam.values.astype(np.int64) * 100000 + q.b.values,
                                    unit_a=np.where(has, q.fam.values.astype(np.int64) * 100000 + q.a.values, -1),
                                    sta=c.split('__')[0].rsplit('.', 2)[0]))
        self.w0mean = np.mean(np.concatenate([b['w0'] for b in self.blocks]))
        for b in self.blocks:
            b['w0'] = b['w0'] / self.w0mean
            b['w'] = b['w0'].copy()


def interp_w(t, segs):
    allq = [s[0] for s in segs[1:]]
    offs = np.cumsum([0] + [len(s) for s in segs])
    sg = np.searchsorted(allq, t, side='right')
    j0 = np.empty(len(t), int)
    w = np.empty(len(t))
    for s, nd in enumerate(segs):
        m = sg == s
        if not m.any():
            continue
        j = np.clip(np.searchsorted(nd, t[m]) - 1, 0, len(nd) - 2)
        j0[m] = offs[s] + j
        w[m] = np.clip((t[m] - nd[j]) / (nd[j + 1] - nd[j]), 0, 1)
    return j0, w


# ------------------------------------------------------------------ S designs
def phi_param(t, extra=()):
    """parametric source basis (columns PNAMES + extra).  extra: list of ('step'|'tr'|'ramp', t0)."""
    yr = (t - T0) / YR
    cols = [yr - 9.0, np.sin(2 * np.pi * yr), np.cos(2 * np.pi * yr)]
    for k in ('pet', 'fer', 'm7'):
        cols.append((t >= TQ[k]).astype(float))
    dt = (t - TQ['m7']) / DAY
    cols.append(np.where(dt >= 0, np.exp(-np.clip(dt, 0, None) / TAU_TR), 0.0))
    cols.append(np.clip((t - T_DECL) / (4 * YR), 0, 1))
    for kind, t0 in extra:
        if kind == 'step':
            cols.append((t >= t0).astype(float))
        elif kind == 'tr':
            d = (t - t0) / DAY
            cols.append(np.where(d >= 0, np.exp(-np.clip(d, 0, None) / TAU_TR), 0.0))
        elif kind == 'ramp':
            cols.append(np.clip((t - t0) / (4 * YR), 0, 1))
    return np.array(cols).T


def s_design(P, blk, kind, extra=()):
    """cached wrapper: base design (no extra columns) cached per block; extra placebo columns appended
    (column layout family-major: col = f*NP + j, the extra columns after the base ones of each family)."""
    key = '_sd_' + kind
    if key not in blk:
        G, used, NP = _s_design(P, blk, kind)
        blk[key] = (G.tocoo(), used, NP)
    Gc, used0, NP0 = blk[key]
    if not extra:
        return blk[key][0].tocsr(), used0, NP0
    Ge, usede, NE = _s_design(P, blk, 'extra', extra)
    Ge = Ge.tocoo()
    NP = NP0 + NE
    c0 = used0[Gc.col]
    c0 = (c0 // NP0) * NP + c0 % NP0
    ce = usede[Ge.col]
    ce = (ce // NE) * NP + NP0 + ce % NE
    rows = np.r_[Gc.row, Ge.row]
    cols = np.r_[c0, ce]
    vals = np.r_[Gc.data, Ge.data]
    used, inv = np.unique(cols, return_inverse=True)
    G = sp.csr_matrix((vals, (rows, inv)), shape=(len(blk['d']), len(used)))
    return G, used, NP


def _s_design(P, blk, kind, extra=()):
    """sparse (n x NF*NP) source design for one block; returns matrix restricted to used columns + col ids."""
    n = len(blk['d'])
    has = blk['has']
    if kind == 'np':
        NP = P.NN
        jb, wb = interp_w(blk['tb'], P.segs)
        ja, wa = interp_w(blk['ta'], P.segs)
        hv = has.astype(float)
        fb, fa = blk['fsb'] * NP, blk['fsa'] * NP
        rows = np.r_[np.arange(n), np.arange(n), np.arange(n), np.arange(n)]
        cols = np.r_[fb + jb, fb + jb + 1, fa + ja, fa + ja + 1]
        vals = np.r_[1 - wb, wb, -(1 - wa) * hv, -wa * hv]
    else:
        if kind == 'extra':
            Fb = phi_param(blk['tb'], extra)[:, len(PNAMES):]
            Fa = phi_param(blk['ta'], extra)[:, len(PNAMES):] * has[:, None]
        else:
            Fb = phi_param(blk['tb'])
            Fa = phi_param(blk['ta']) * has[:, None]
        NP = Fb.shape[1]
        rr = np.repeat(np.arange(n), NP)
        cc_ = np.tile(np.arange(NP), n)
        rows = np.r_[rr, rr]
        cols = np.r_[blk['fsb'].repeat(NP) * NP + cc_, blk['fsa'].repeat(NP) * NP + cc_]
        vals = np.r_[Fb.ravel(), -Fa.ravel()]
    G = sp.csr_matrix((vals, (rows, cols)), shape=(n, P.NF * NP))
    used = np.unique(cols[vals != 0])
    return G[:, used], used, NP


def a_matrix(P, blk, w):
    X = blk['X']
    A = (X.T @ sp.diags(w) @ X).toarray()
    NN = P.NN
    A[:NN, :NN] += P.lam ** 2 * P.LtL
    s = np.mean(np.diag(A)[:NN]) + 1e-12
    A[:NN, :NN] += 10 * s / NN                     # gauge: sum of R nodes = 0 (exact null direction R+c, c-c)
    A[np.arange(NN), np.arange(NN)] += 1e-5 * s    # tiny ridge: segments without data on this channel -> 0
    A[np.arange(NN, A.shape[0]), np.arange(NN, A.shape[0])] += 1e-9 * s
    return A


def solve_joint(P, kind='param', extra=(), iters=3, mult=None, store=False, fixed_w=False, data=None):
    """Exact Schur-complement solve.  Returns dict with s (NF x NP), per-block r (R nodes + offsets)."""
    out_blocks = []
    for it in range(iters):
        M = None
        v = None
        cache = []
        for bi, blk in enumerate(P.blocks):
            d = blk['d'] if data is None else data[bi]
            w = blk['w'] if mult is None else blk['w'] * mult[bi]
            Gs, used, NP = s_design(P, blk, kind, extra)
            if M is None:
                NS = P.NF * NP
                M = np.zeros((NS, NS))
                v = np.zeros(NS)
            A = a_matrix(P, blk, w)
            cf = cho_factor(A, lower=False, check_finite=False)
            X = blk['X']
            WX = sp.diags(w) @ X
            B = (WX.T @ Gs).toarray() if sp.issparse(Gs) else WX.T @ Gs
            g = X.T @ (w * d)
            D = (Gs.T @ sp.diags(w) @ Gs)
            D = D.toarray() if sp.issparse(D) else D
            h = Gs.T @ (w * d)
            Q = cho_solve(cf, B, check_finite=False)
            M[np.ix_(used, used)] += D - B.T @ Q
            v[used] += h - Q.T @ g
            cache.append((cf, B, g, used, Gs))
        NS = M.shape[0]
        sc = np.mean(np.diag(M)) + 1e-12
        # gauge 1: sum_F S_F = 0 for every basis element / node
        C = np.kron(np.ones((1, P.NF)), np.eye(NP))
        M2 = M + 1e3 * sc * C.T @ C
        if kind == 'np':
            M2 += P.lam ** 2 * np.kron(np.eye(P.NF), P.LtL)
            Cf = np.kron(np.eye(P.NF), np.ones((1, NP)) / NP)      # gauge 2: mean level per family = 0
            M2 += 1e3 * sc * Cf.T @ Cf
        M2 += 1e-9 * sc * np.eye(NS)
        s = np.linalg.solve(M2, v)
        out_blocks = []
        for bi, (blk, (cf, B, g, used, Gs)) in enumerate(zip(P.blocks, cache)):
            r = cho_solve(cf, g - B @ s[used], check_finite=False)
            d = blk['d'] if data is None else data[bi]
            res = d - blk['X'] @ r - Gs @ s[used]
            out_blocks.append(dict(r=r, res=res))
            if not fixed_w and mult is None:                 # Huber IRLS update (per-channel robust scale)
                w0 = blk['w0']
                z = np.abs(res) * np.sqrt(w0) / (1.4826 * np.median(np.abs(res) * np.sqrt(w0)) + 1e-12)
                blk['w'] = w0 * np.where(z <= HUBER, 1.0, HUBER / np.maximum(z, 1e-12))
        if fixed_w or mult is not None:
            break
    res = dict(s=s.reshape(P.NF, NP), NP=NP, M=M2 if store else None)
    res['blocks'] = out_blocks
    return res


def boot_mult(P, rng):
    units = np.unique(np.concatenate([b['unit_b'] for b in P.blocks]))
    cnt = rng.multinomial(len(units), np.ones(len(units)) / len(units)).astype(float)
    out = []
    for b in P.blocks:
        mb = cnt[np.searchsorted(units, b['unit_b'])]
        ma = np.where(b['unit_a'] >= 0, cnt[np.searchsorted(units, np.maximum(b['unit_a'], 0))], 1.0)
        ma = np.where(b['unit_a'] >= 0, np.where(np.isin(b['unit_a'], units), ma, 0.0), 1.0)
        out.append(mb * ma)
    return out


# ============================================================================================ stages
def stage_param(panel, nboot):
    R, fams = load_rows(panel)
    P = Problem(R, fams)
    log(f'param [{panel}]: {len(R)} rows, {len(P.blocks)} channel-segments, {P.NF} families')
    sol = solve_joint(P, 'param')
    S0 = sol['s']
    B = []
    rng = np.random.default_rng(11)
    for i in range(nboot):
        B.append(solve_joint(P, 'param', mult=boot_mult(P, rng), fixed_w=True)['s'])
        if i % 20 == 0:
            log('boot', i)
    B = np.array(B)
    pickle.dump(dict(fams=fams, s=S0, boot=B, pnames=PNAMES, weights=[b['w'] for b in P.blocks],
                     chans=[b['name'] for b in P.blocks], r=[b['r'] for b in sol['blocks']],
                     res=[b['res'] for b in sol['blocks']]),
                open(f'{OUT}/param_{panel}.pkl', 'wb'))
    df = pd.DataFrame(S0 * 100, columns=PNAMES, index=fams)
    se = pd.DataFrame(B.std(0) * 100, columns=[p + '_se_boot' for p in PNAMES], index=fams)
    pd.concat([df, se], axis=1).rename_axis('family').to_csv(f'{OUT}/param_dS_{panel}.csv')
    log(pd.concat([df, se], axis=1)[['step_m7', 'step_m7_se_boot', 'tr60_m7', 'tr60_m7_se_boot', 'decl4yr',
                                     'decl4yr_se_boot']].round(4).to_string())


def stage_np(panel, nboot, lam):
    R, fams = load_rows(panel)
    P = Problem(R, fams, lam=lam)
    log(f'np [{panel}] lam {lam}: {len(R)} rows, {len(P.blocks)} blocks, {P.NF} families, {P.NN} nodes')
    sol = solve_joint(P, 'np')
    log('main solve done')
    rng = np.random.default_rng(12)
    SB, RB = [], []
    for i in range(nboot):
        sb = solve_joint(P, 'np', mult=boot_mult(P, rng), fixed_w=True)
        SB.append(sb['s'])
        RB.append(np.array([b['r'][:P.NN] for b in sb['blocks']]))
        log('np boot', i)
    tag = f'{panel}_lam{int(lam)}'
    pickle.dump(dict(fams=fams, nodes=P.nodes, segs=P.segs, s=sol['s'], sboot=np.array(SB),
                     chans=[b['name'] for b in P.blocks], R=np.array([b['r'][:P.NN] for b in sol['blocks']]),
                     Rboot=np.array(RB) if RB else None, weights=[b['w'] for b in P.blocks],
                     res=[b['res'] for b in sol['blocks']]), open(f'{OUT}/np_{tag}.pkl', 'wb'))
    # family x station interaction residual: weighted mean residual per (family, station) pre/post M7
    rows = []
    for blk, ob in zip(P.blocks, sol['blocks']):
        for k, F in enumerate(fams):
            m = blk['fsb'] == k
            if m.sum() < 30:
                continue
            post = m & (blk['tb'] >= TQ['m7'])
            pre = m & (blk['tb'] < TQ['m7']) & (blk['tb'] > TQ['m7'] - 2 * YR)
            w = blk['w']
            mu = lambda q: (np.sum(w[q] * ob['res'][q]) / np.sum(w[q])) if q.sum() > 5 else np.nan  # noqa: E731
            rows.append(dict(chseg=blk['name'], sta=blk['sta'], fam=F, n=int(m.sum()), res_all=mu(m) * 100,
                             res_post_minus_pre=(mu(post) - mu(pre)) * 100,
                             rms=np.sqrt(np.sum(w[m] * ob['res'][m] ** 2) / np.sum(w[m])) * 100))
    pd.DataFrame(rows).to_csv(f'{OUT}/interaction_residual_{tag}.csv', index=False)
    log('np saved', tag)


def placebo_dates_m7():
    from sci_transient_all import placebo_dates
    d = placebo_dates()
    return d[d < epoch_s(['2021-01-01'])[0] - 120 * DAY]


def decline_placebo_starts():
    return epoch_s(pd.date_range('2009-07-01', '2016-07-01', freq='3MS'))


def stage_placebo(panel, nsplit, nperm):
    R, fams = load_rows(panel)
    P = Problem(R, fams)
    sol = solve_joint(P, 'param')                 # sets IRLS weights
    W = [b['w'].copy() for b in P.blocks]
    out = dict(fams=fams)
    # (a) shifted M7 dates: extra step + 60 d transient at the date (identical estimator)
    dates = placebo_dates_m7()
    log(f'placebo: {len(dates)} M7-like dates, {len(decline_placebo_starts())} decline windows')
    PS = []
    for t0 in dates:
        s = solve_joint(P, 'param', extra=[('step', t0), ('tr', t0)], fixed_w=True)['s']
        PS.append(s[:, len(PNAMES):])
    out['m7_dates'] = dates
    out['m7_placebo'] = np.array(PS)              # (ndates, NF, 2): step, tr
    log('m7 placebo done')
    PD = []
    for t0 in decline_placebo_starts():
        s = solve_joint(P, 'param', extra=[('ramp', t0)], fixed_w=True)['s']
        PD.append(s[:, len(PNAMES)])
    out['decl_starts'] = decline_placebo_starts()
    out['decl_placebo'] = np.array(PD)
    log('decline placebo done')
    # (b) pseudo-family splits: stacks of each family randomly halved; family F -> 2i, 2i+1
    rng = np.random.default_rng(21)
    SPL = []
    fidx = {F: i for i, F in enumerate(fams)}
    for k in range(nsplit):
        half = {}
        for F in fams:
            nst = int(max(R.b.max(), R.a.max()) + 2)
            half[F] = rng.integers(0, 2, nst)
        fi = R.fam.map(fidx).values
        hb = np.array([half[F][b] for F, b in zip(R.fam.values, R.b.values)])
        ha = np.array([half[F][a] if a >= 0 else 0 for F, a in zip(R.fam.values, R.a.values)])
        ha = np.where(R.a.values >= 0, ha, hb)
        Pk = Problem(R, fams, fam_s=(2 * fi + ha, 2 * fi + hb), fam_c=2 * fi + hb)
        s = solve_joint(Pk, 'param')['s']
        SPL.append(s[0::2] - s[1::2])
        log('split', k)
    out['split_diff'] = np.array(SPL)            # (nsplit, NF, NP): dS(half A) - dS(half B)
    # (c) family-label permutation within station-epoch (epoch = calendar year of t_b); offsets keep true family
    PERM = []
    for k in range(nperm):
        fi = R.fam.map(fidx).values.copy()
        new = fi.copy()
        yr = pd.to_datetime(R.tb.values, unit='s').year.values
        key = pd.Series(R.chan.str.rsplit('.', n=2).str[0].values + '_' + yr.astype(str))
        for _, idx in key.groupby(key).indices.items():
            u = np.unique(fi[idx])
            pm = dict(zip(u, rng.permutation(u)))
            new[idx] = [pm[x] for x in fi[idx]]
        Pk = Problem(R, fams, fam_s=(new, new), fam_c=fi)
        s = solve_joint(Pk, 'param')['s']
        PERM.append(s)
        log('perm', k)
    out['perm'] = np.array(PERM)
    pickle.dump(out, open(f'{OUT}/placebo_{panel}.pkl', 'wb'))
    log('placebo saved')


def stage_jack(panel):
    """delete-one-station jackknife of the parametric dS_F (balanced panel)"""
    R, fams = load_rows(panel)
    sta = R.chan.str.rsplit('.', n=2).str[0]
    out = {}
    for st in sorted(sta.unique()):
        Rk = R[sta != st].reset_index(drop=True)
        P = Problem(Rk, fams)
        out[st] = solve_joint(P, 'param')['s']
        log('jack', st)
    pickle.dump(dict(fams=fams, jack=out), open(f'{OUT}/jack_{panel}.pkl', 'wb'))


# ------------------------------------------------------------------ per-channel receiver(+common) fits
def chan_basis(t, extra=()):
    yr = (t - T0) / YR
    cols = [yr - 9.0, np.sin(2 * np.pi * yr), np.cos(2 * np.pi * yr)]
    for k in ('pet', 'fer', 'm7'):
        cols.append((t >= TQ[k]).astype(float))
    for k in ('pet', 'fer', 'm7'):                                     # log-time healing after each quake
        dt = (t - TQ[k]) / DAY
        cols.append(np.where(dt >= 0, np.log1p(np.clip(dt, 0, None) / 10.0), 0.0))
    cols.append(np.clip((t - T_DECL) / (4 * YR), 0, 1))
    for kind, t0 in extra:
        if kind == 'step':
            cols.append((t >= t0).astype(float))
        elif kind == 'ramp':
            cols.append(np.clip((t - t0) / (4 * YR), 0, 1))
    return np.array(cols).T


CNAMES = ['trend', 'ann_s', 'ann_c', 'step_pet', 'step_fer', 'step_m7', 'heal_pet', 'heal_fer', 'heal_m7', 'decl4yr']


def chan_fit_one(blk, d, w, extra=()):
    has = blk['has']
    Fb = chan_basis(blk['tb'], extra)
    Fa = chan_basis(blk['ta'], extra) * has[:, None]
    nf = blk['nf']
    fl = np.unique(blk['fc'])
    lf = np.searchsorted(fl, blk['fc'])
    C = np.zeros((len(d), nf))
    C[np.where(~has)[0], lf[~has]] = -1.0
    A = np.c_[Fb - Fa, C]
    sw = np.sqrt(w)
    m, *_ = np.linalg.lstsq(A * sw[:, None], d * sw, rcond=None)
    return m[:Fb.shape[1]]


def stage_chanfit(panel_all='all'):
    """per channel-segment parametric fit of R_C + Sbar after removing the fitted family deviations dS_F
    (from the param joint solution on ALL channels), + the same placebo dates."""
    R, fams = load_rows('all')
    P = Problem(R, fams)
    sol = solve_joint(P, 'param')
    s = sol['s']
    dates = placebo_dates_m7()
    starts = decline_placebo_starts()
    rows = []
    PK = {}
    rng = np.random.default_rng(5)
    for blk in P.blocks:
        Gs, used, NP = s_design(P, blk, 'param')
        d = blk['d'] - Gs @ s.ravel()[used]
        w = blk['w']
        m = chan_fit_one(blk, d, w)
        pm = np.array([chan_fit_one(blk, d, w, extra=[('step', t0)])[-1] for t0 in dates])
        pdl = np.array([chan_fit_one(blk, d, w, extra=[('ramp', t0)])[-1] for t0 in starts])
        # stack bootstrap within channel
        units = np.unique(blk['unit_b'])
        bs = []
        for _ in range(60):
            cnt = rng.multinomial(len(units), np.ones(len(units)) / len(units)).astype(float)
            mb = cnt[np.searchsorted(units, blk['unit_b'])]
            ma = np.where(blk['unit_a'] >= 0, cnt[np.searchsorted(units, np.maximum(blk['unit_a'], 0)).clip(0, len(units) - 1)]
                          * np.isin(blk['unit_a'], units), 1.0)
            bs.append(chan_fit_one(blk, d, w * mb * ma))
        bs = np.array(bs)
        t_first, t_last = blk['tb'].min(), blk['tb'].max()
        r = dict(chseg=blk['name'], sta=blk['sta'], n=len(d), t_first=t_first, t_last=t_last)
        for k, nm in enumerate(CNAMES):
            r[nm] = m[k] * 100
            r[nm + '_se_boot'] = bs[:, k].std() * 100
        r['step_m7_placebo_sd'] = np.std(pm) * 100
        r['step_m7_placebo_mean'] = np.mean(pm) * 100
        r['decl_placebo_sd'] = np.std(pdl) * 100
        r['decl_placebo_mean'] = np.mean(pdl) * 100
        rows.append(r)
        PK[blk['name']] = dict(m7=pm, decl=pdl, boot=bs)
        log(f"{blk['name']:24s} M7 {r['step_m7']:+.3f} (plc sd {r['step_m7_placebo_sd']:.3f}) decl {r['decl4yr']:+.3f} "
            f"(plc sd {r['decl_placebo_sd']:.3f})")
    pd.DataFrame(rows).to_csv(f'{OUT}/chanfit.csv', index=False)
    pickle.dump(dict(dates=dates, starts=starts, arr=PK), open(f'{OUT}/chanfit_placebo.pkl', 'wb'))


# ------------------------------------------------------------------ synthetics
def stage_synth(panel, nrep):
    """Exact real design.  Truth: observed (non-parametric) R_C curves + offsets from the np solution on the
    same panel; noise: real residuals with a block-wild bootstrap (random sign per station x quarter block,
    i.e. correlated within station-epoch).  Cases:
      S0 receiver only -> leakage floor of the parametric dS_F
      S1 + common M7 step -3e-4 (all families)  (must NOT leak into dS_F)
      S2 + family-specific M7 steps: 1e-4 .. 5e-4 on 5 families (true apparent = w_true x local)"""
    R, fams = load_rows(panel)
    P = Problem(R, fams)
    npk = pickle.load(open(f'{OUT}/np_{panel}_lam{int(LAM)}.pkl', 'rb'))
    assert npk['chans'] == [b['name'] for b in P.blocks]
    # truth receiver signal: full np solution R (nodes) + offsets; S part dropped (=observed R_C only)
    solnp = solve_joint(P, 'np', iters=3)
    W = [b['w'].copy() for b in P.blocks]
    truthR = [blk['X'] @ ob['r'] for blk, ob in zip(P.blocks, solnp['blocks'])]
    resid = [ob['res'] for ob in solnp['blocks']]
    fam_inj = {fams[i]: v for i, v in zip(range(0, len(fams), max(1, len(fams) // 5)), [-5e-4, -3e-4, -1e-4, 2e-4, 4e-4])}
    rng = np.random.default_rng(33)
    out = dict(fams=fams, fam_inj=fam_inj, cases={})
    for case in ('S0', 'S1', 'S2'):
        reps = []
        for k in range(nrep):
            data = []
            for blk, tr, rs in zip(P.blocks, truthR, resid):
                q = pd.to_datetime(blk['tb'], unit='s')
                key = (q.year * 4 + (q.month - 1) // 3).values
                uk, inv = np.unique(key, return_inverse=True)
                sgn = rng.choice([-1.0, 1.0], len(uk))[inv]
                d = tr + sgn * rs
                if case in ('S1', 'S2'):
                    stp = lambda t: (t >= TQ['m7']).astype(float)  # noqa: E731
                    dS = stp(blk['tb']) - stp(blk['ta']) * blk['has']
                    if case == 'S1':
                        d = d - 3e-4 * dS
                    else:
                        famv = np.array([fam_inj.get(fams[i], 0.0) for i in blk['fsb']])
                        d = d + famv * dS
                data.append(d)
            for b, w in zip(P.blocks, W):
                b['w'] = w.copy()
            s = solve_joint(P, 'param', fixed_w=True, data=data)['s']
            reps.append(s)
            log('synth', case, k)
        out['cases'][case] = np.array(reps)
    pickle.dump(out, open(f'{OUT}/synth_{panel}.pkl', 'wb'))


# ------------------------------------------------------------------ coherence -> w_S
def stage_coherence(fam_list, max_ch):
    """Inter-family coda coherence vs lapse time at shared channels (reference stacks of all isolated events,
    2-8 Hz, aligned on each family's own S by cross-correlation, normalised by split-half reliability).
    Receiver-side scattering is coherent between sources km apart; source-side scattering decorrelates, so
    1 - gamma_AB(tau) estimates the source-side weight of the coda at lapse tau (tau measured from S)."""
    import h5py
    from scipy.signal import butter, sosfiltfilt
    cat = pd.read_csv('data/merged/LFEcat_MTJ_merged_v1v2.csv')
    cat = cat[cat.source != 'v1_only']
    allt = np.sort(epoch_s(pd.to_datetime(cat.time_utc)))
    ch = pd.read_csv(f'{OUT}/channels.csv', index_col=0)
    chans = list(ch.index[ch.panel])[:max_ch]
    REF = {}
    for F in fam_list:
        h5f = f'data/waveforms/fam{F}.h5'
        picks = pd.read_csv(f'data/locations/fam{F}_3d_picks.csv').set_index('sta')
        with h5py.File(h5f, 'r') as h:
            sr, pre = h.attrs['sr'], h.attrs['pre']
            te = epoch_s(pd.to_datetime(h['events/time_utc'][:].astype(str)))
            iso = (np.searchsorted(allt, te + 5.0, side='right') - np.searchsorted(allt, te - 5.0)) == 1
            for c in chans:
                if f'chan/{c}' not in h or c.rsplit('.', 2)[0] not in picks.index:
                    continue
                ev = h[f'chan/{c}/ev'][:]
                sel = np.where(iso[ev])[0]
                if len(sel) < 200:
                    continue
                X = h[f'chan/{c}/data'][np.sort(sel)].astype('float64')
                sos = butter(4, [2, 8], btype='band', fs=sr, output='sos')
                X = sosfiltfilt(sos, X, axis=1)
                X /= (np.std(X, axis=1) + 1e-20)[:, None]
                S = float(picks.loc[c.rsplit('.', 2)[0], 'S_final'] if 'S_final' in picks.columns else picks.loc[c.rsplit('.', 2)[0], 'S'])
                odd = X[0::2].mean(0)
                even = X[1::2].mean(0)
                REF[(F, c)] = dict(full=X.mean(0), odd=odd, even=even, S=S, sr=sr, pre=pre, n=len(sel))
        log('coherence refs', F, sum(1 for k in REF if k[0] == F))
    win, maxlag = 2.0, 0.24
    taus = np.arange(-0.5, 22.0, 1.0)
    rows = []

    def seg(x, S, t0, sr, pre):
        i = int(round((S + t0 - pre) * sr))
        n = int(win * sr)
        if i < 0 or i + n > len(x):
            return None
        return x[i:i + n]

    def mcc(x, y, sr):
        L = int(maxlag * sr)
        best = -1
        for l in range(-L, L + 1):
            a = x[max(0, l):len(x) + min(0, l)]
            b = y[max(0, -l):len(y) + min(0, -l)]
            a = a - a.mean()
            b = b - b.mean()
            best = max(best, a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20))
        return best

    for c in chans:
        fs = [F for F in fam_list if (F, c) in REF]
        for i, A in enumerate(fs):
            for Bf in fs[i + 1:]:
                ra, rb = REF[(A, c)], REF[(Bf, c)]
                sr, pre = ra['sr'], ra['pre']
                # align B on A by the S-window cross-correlation (direct S)
                for t0 in taus:
                    xa, xb = seg(ra['full'], ra['S'], t0, sr, pre), seg(rb['full'], rb['S'], t0, sr, pre)
                    if xa is None or xb is None:
                        continue
                    cab = mcc(xa, xb, sr)
                    rel = []
                    for r_ in (ra, rb):
                        o, e = seg(r_['odd'], r_['S'], t0, sr, pre), seg(r_['even'], r_['S'], t0, sr, pre)
                        rh = np.corrcoef(o, e)[0, 1]
                        rel.append(max(2 * rh / (1 + rh), 1e-3) if rh > 0 else 1e-3)
                    # chance level: A at tau vs B at tau+5 s (different lapse, same spectra)
                    xb2 = seg(rb['full'], rb['S'], t0 + 5.0, sr, pre)
                    cnull = mcc(xa, xb2, sr) if xb2 is not None else np.nan
                    rows.append(dict(chan=c, famA=A, famB=Bf, tau=t0 + win / 2, cc=cab, cc_null=cnull,
                                     relA=rel[0], relB=rel[1], gamma=cab / np.sqrt(rel[0] * rel[1])))
    pd.DataFrame(rows).to_csv(f'{OUT}/coherence.csv', index=False)
    log('coherence saved', len(rows))


# ============================================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage')
    ap.add_argument('--panel', default='balanced')
    ap.add_argument('--nboot', type=int, default=100)
    ap.add_argument('--lam', type=float, default=LAM)
    ap.add_argument('--nsplit', type=int, default=6)
    ap.add_argument('--nperm', type=int, default=10)
    ap.add_argument('--nrep', type=int, default=8)
    ap.add_argument('--fams', default='23,27,8,6,19,22,1,13')
    ap.add_argument('--max-ch', type=int, default=40)
    a = ap.parse_args()
    t = time.time()
    if a.stage == 'prep':
        prep()
    elif a.stage == 'param':
        stage_param(a.panel, a.nboot)
    elif a.stage == 'np':
        stage_np(a.panel, a.nboot, a.lam)
    elif a.stage == 'placebo':
        stage_placebo(a.panel, a.nsplit, a.nperm)
    elif a.stage == 'jack':
        stage_jack(a.panel)
    elif a.stage == 'chanfit':
        stage_chanfit()
    elif a.stage == 'synth':
        stage_synth(a.panel, a.nrep)
    elif a.stage == 'coherence':
        stage_coherence([int(x) for x in a.fams.split(',')], a.max_ch)
    elif a.stage == 'analyze':
        import sci_dvv_source_receiver_analyze as AN
        AN.main()
    log('stage', a.stage, 'done in %.0f s' % (time.time() - t))


if __name__ == '__main__':
    main()
