#!/usr/bin/env python
"""dv/v attribution fits (per station x band / channel / network band / repeater station).

Stages (python scripts/sci_dvv_attr_fit.py <stage> [--only STA] [--chan C] [--workers N] [--tag T]):
  prep      convert the big SC CSV to parquet once (bulk dir)
  fit       selection + full-core final fit + bootstrap + model averaging + placebos -> results pickles (bulk dir)
  collect   tidy tables: data/sci/dvv_attr/components.csv, model_scores.csv, candidate_tests.csv, model_average.csv
  oos       out-of-sample: environmental selection AND fit on data before 2021-07, predict 2021-07..2026
  afterslip GNSS M7 postseismic time constant vs dv/v recovery time constants
Model:  dv/v = const + offsets(instrument breaks)
              + big-3 [drop0 = perm + trans, transient exp(-t/tau) with the same sign (recovered fraction in [0, 1])]
              + transient-only PGV dose of all other local events (outside the big-3 aftershock windows)
              + trend + ramp(2021-07)                                  [the four 'core' groups are always kept]
              + environmental groups selected by water-year CV AND BIC (forward, then the same test backward)
Core-group uncertainty: all 16 subsets of {big3, dose, trend, ramp21} are fitted; BIC-weighted values and the
min-max over models within dBIC < 6 are reported next to the full-model value and its bootstrap SD (headline).
Revision 2026-11-25 after code review (M1-M6 + should-fixes); see sci_dvv_attr_lib docstring.
"""
import os, sys, json, pickle, argparse, time, warnings, zlib, itertools
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import pandas as pd

ROOT = '/home/seismic/LFE'
sys.path.insert(0, f'{ROOT}/scripts')
import sci_dvv_attr_lib as L

warnings.filterwarnings('ignore')
OUT = f'{ROOT}/data/sci/dvv_attr'
RES = f'{L.BULK}/results'
os.makedirs(RES, exist_ok=True)
TAUQ = (5, 15, 40, 120)          # shared quake-recovery grid (<= 120 d: Petrolia-Ferndale spacing 365 d), capped at 1/3 of the shortest post-quake span
TAU_PROFILE = (5, 15, 50, 150, 500, 1500, 5000)
NBOOT = 200
NPERM = 200                      # water-year permutations (common to all series: fixed seed)
CV_GAIN = 0.005                  # environmental group: CV RMS must drop >= 0.5 % (relative) ...
BIC_GAIN = -2.0                  # ... AND BIC by >= 2 (same rule forward and backward)
COV_MIN = 0.8
DBIC_SET = 6.0                   # models within this dBIC define the min-max range
DECLINE_WIN = ('2021-01-01', '2025-12-31')
PERMS = None


def perms():
    global PERMS
    if PERMS is None:
        rng = np.random.default_rng(20261125)
        PERMS = [rng.permutation(19) for _ in range(NPERM)]          # WY2008..WY2026
    return PERMS


def seed_of(*a):
    return zlib.crc32('|'.join(map(str, a)).encode()) & 0xffffffff


# ================================================================================================ data
def sc_parquet():
    fn = f'{L.BULK}/sc_10day_back.parquet'
    if not os.path.exists(fn):
        x = pd.read_csv(f'{ROOT}/data/sci/noise_dvv_sc_full/sc_10day_back.csv', usecols=['t', 'band', 'combo', 'dvv_pct', 'err_pct', 'sta'])
        x = x[x.combo == 'SC'].drop(columns='combo')
        x['t'] = pd.to_datetime(x.t)
        x.to_parquet(fn)
    return pd.read_parquet(fn)


def mask_windows(sta, t, sm, breaks=None):
    """True where the sample window overlaps an instrument-problem window or straddles a break."""
    kind, n = sm
    t = pd.DatetimeIndex(t)
    pre, post = (0, n - 1) if kind == 'back' else ((n // 2, n // 2) if kind == 'centre' else (0, 0))
    bad = np.zeros(len(t), bool)
    for a, b in L.EXCL.get(sta, []):
        bad |= (t >= pd.Timestamp(a) - pd.Timedelta(days=pre)) & (t <= pd.Timestamp(b) + pd.Timedelta(days=post))
    for b in (L.BREAKS.get(sta, []) if breaks is None else breaks):
        bb = pd.Timestamp(b)
        bad |= (t >= bb - pd.Timedelta(days=pre)) & (t < bb + pd.Timedelta(days=post))
    return bad


def load_sc():
    x = sc_parquet()
    out = []
    for (sta, band), g in x.groupby(['sta', 'band']):
        g = g.dropna(subset=['dvv_pct']).sort_values('t')
        sm = ('back', 10)
        bad = mask_windows(sta, g.t, sm)
        g = g[~bad & (g.t >= L.DAYS[0]) & (g.t <= L.DAYS[-1])]
        if len(g) < 300:
            continue
        out.append(L.Series('SC10', sta, band, g.t.values, g.dvv_pct.values, sm))
    return out


def load_lfe():
    import glob
    out = []
    files = sorted(glob.glob(f'{ROOT}/data/dvv/pool_clean/*_curve.csv'))
    chans = sorted(set(os.path.basename(f).split('__')[0].replace('_curve.csv', '') for f in files))
    for ch in chans:
        parts = [pd.read_csv(f) for f in files if os.path.basename(f).split('__')[0].replace('_curve.csv', '') == ch]
        g = pd.concat(parts).dropna(subset=['dvv_pct'])
        g['t'] = pd.to_datetime(g.t).dt.normalize()
        sta = ch.rsplit('.', 2)[0]
        sm = ('centre', 30)
        # offsets from the segment column (each instrument segment was inverted with its own reference)
        brk = [str(g[g.segment == k].t.min().date()) for k in sorted(g.segment.unique())[1:]]
        bad = mask_windows(sta, g.t, sm, breaks=[])
        g = g[~bad].sort_values('t')
        if len(g) < 24:
            continue
        w = 1.0 / np.maximum(g.err_pct.values, 1e-3) ** 2
        out.append(L.Series('LFE', sta, ch, g.t.values, g.dvv_pct.values, sm, w=w / np.median(w), breaks=brk))
    return out


def load_lp():
    x = pd.read_csv(f'{ROOT}/data/sci/noise_lp/network_series_10d.csv', parse_dates=['t'])
    out = []
    for band, kind, group in (('0.05-0.1', 'BB', 'all'), ('0.1-0.2', 'BB', 'all'), ('0.2-0.3', 'BB', 'all'),
                              ('0.1-0.2', 'all', 'all'), ('0.2-0.3', 'all', 'all')):
        g = x[(x.band == band) & (x.kind == kind) & (x.group == group) & (x.ref == 'full') & (x.side == 'both')].dropna(subset=['dvv'])
        if len(g) < 100:
            continue
        w = 1.0 / np.maximum(g.err.values, 1e-4) ** 2
        y = g.dvv.values * 100.0                    # stored as a fraction -> %
        out.append(L.Series('LP10', 'REGION', f'{band}Hz_{kind}', g.t.values, y, ('centre', 10), w=w / np.median(w), breaks=[]))
    return out


def load_rep():
    R = pd.read_csv(f'{ROOT}/data/sci/dvv_deep_shallow/rep/rep_pairs_dvv.csv', parse_dates=['ti', 'tj'])
    R = R[(R.cc >= 0.5) & (R.snr >= 2) & np.isfinite(R.eps) & (np.abs(R.eps) < 0.009) & (R.ti >= L.DAYS[0])]
    out = []
    for sta, g in R.groupby('sta'):
        bi = mask_windows(sta, g.ti, ('point', 1)) | mask_windows(sta, g.tj, ('point', 1))
        g = g[~bi]
        cc = np.clip(g.cc.values, 0, 0.99)
        w = cc ** 2 / (1 - cc ** 2)
        s = L.Series('REP', sta, 'all_comp', g.tj.values, g.eps.values * 100.0, ('point', 1), w=w / np.median(w),
                     pair_i=L.dayidx(g.ti.values))
        s.cluster = g.cl.values
        s.n_events = len(set(g.i) | set(g.j))
        out.append(s)
    return out


ENVLIKE = lambda g: g not in ('base', 'big3', 'dose', 'trend', 'ramp21', 'postdil', 'fake')
COMPOSITE = {'QUAKES': ('big3', 'dose'), 'LONGTERM': ('trend', 'ramp21', 'postdil'), 'POST21': ('big3', 'dose', 'ramp21')}


# ================================================================================================ fitting one series
class Fitter:
    def __init__(self, S):
        self.S = S
        band = S.chan if S.dataset == 'SC10' else None
        self.cands, self.D = L.candidate_groups(S.sta, band, L.drivers(S.sta))
        self.base = L.base_cols(S.breaks)
        self.longc = L.long_cols()
        self.pairs = S.pair_i is not None
        u = np.unique(S.idx)
        self.spacing = int(np.median(np.diff(u))) if len(u) > 2 else 1
        self._final_blocks = None

    # ---------------------------------------------------------------- support / blocks
    def supported(self, kind, d, tau=50, nmin=10):
        """Step needs >= nmin samples on each side of t_q (between instrument breaks); transient needs >= nmin
        samples within min(3 tau, 365) d after t_q."""
        ii = np.sort(self.S.idx if self.S.pair_i is None else np.concatenate([self.S.idx, self.S.pair_i]))
        nmin = nmin if self.spacing < 5 else 3
        bks = sorted(int(L.dayidx([pd.Timestamp(b)])[0]) for b in self.S.breaks)
        nxt = min([b for b in bks if b > d], default=L.ND)
        prv = max([b for b in bks if b <= d], default=0)
        if kind == 'step':
            return ((ii >= d) & (ii < nxt)).sum() >= nmin and ((ii < d) & (ii >= prv)).sum() >= nmin
        return ((ii >= d) & (ii <= d + min(3 * tau, 365))).sum() >= nmin

    def tau_grid(self):
        spans = [self.S.idx.max() - L.big3_day(n) for n in L.BIG3 if self.supported('step', L.big3_day(n))]
        cap = max(min(spans) / 3.0, 5) if spans else 150
        lo = 2 * self.spacing if self.spacing >= 5 else 0          # sparse series: tau must span >= 2 samples
        return tuple(t for t in TAUQ if lo <= t <= cap) or (max([t for t in TAUQ if t <= cap] or [5]),)

    def core_blocks(self, tau, qmode='free3', groups=L.CORE, tau_by=None):
        Xq, nq = L.quake_cols(self.S.sta, tau, qmode, tau_by=tau_by)
        B = [('base', self.base[1], self.base[0])]
        keep = []
        for i, n in enumerate(nq):
            if n.startswith('step_') or n.startswith('trans_'):
                d = L.big3_day(n.split('_', 1)[1])
                tq = (tau_by or {}).get(n.split('_', 1)[1], tau)
                if n.startswith('step_') and not self.supported('step', d):
                    continue
                if n.startswith('trans_') and not (self.supported('trans', d, tq) and self.supported('step', d)):
                    continue
            keep.append(i)
        big = [i for i in keep if nq[i].startswith(('step_', 'trans_'))]
        dose = [i for i in keep if nq[i].startswith('dose')]
        if big and 'big3' in groups:
            B.append(('big3', [nq[i] for i in big], Xq[:, big]))
        if dose and 'dose' in groups:
            B.append(('dose', [nq[i] for i in dose], Xq[:, dose]))
        for k in ('trend', 'ramp21'):
            if k in groups:
                B.append((k, [k], self.longc[k]))
        return B

    def env_block(self, g, j, cands=None):
        lab, X = (cands or self.cands)[g][j]
        nm = [f'{g}[{k}]' for k in range(X.shape[1])] if X.shape[1] > 1 else [g]
        return (g, nm, X), lab

    def matrix(self, blocks):
        A, names, groups = L.assemble(self.S, blocks)
        y = self.S.y
        env = np.array([ENVLIKE(g) and g != 'micro' for g in groups])       # gaps -> 0 (mean anomaly); micro: rows dropped
        A[:, env] = np.where(np.isfinite(A[:, env]), A[:, env], 0.0)
        ok = L.finite_rows(A, y)
        sup = np.abs(A[ok]).sum(0) > 1e-12
        if self.pairs and 'const' in names:
            sup[names.index('const')] = False
        return A[:, sup], [n for n, s in zip(names, sup) if s], [g for g, s in zip(groups, sup) if s], ok

    def fit(self, blocks, rows=None, cv=False, constrain=True, nit=4):
        A, names, groups, ok = self.matrix(blocks)
        if rows is not None:
            ok = ok & rows
        y = self.S.y[ok]; w = None if self.S.w is None else self.S.w[ok]
        pr = L.sign_pairs(names) if constrain else None
        beta, ww = L.huber_lsq(A[ok], y, w, nit=nit, pairs=pr)
        r = y - A[ok] @ beta
        sse = float(np.sum(r ** 2 * (1 if w is None else w)))
        res = dict(beta=beta, names=names, groups=groups, ok=ok, A=A, r=r, sse=sse, n=int(ok.sum()), k=len(beta), pairs=pr)
        if cv:
            res['cv'], res['cv_mad'] = L.cv_rms(A[ok], y, self.S.idx[ok], w, names=names if constrain else None)
        return res

    def B(self, f):
        return L.bic(f['sse'], f['n'], f['k'], self.Lc)

    def coverage(self, g, j):
        Xs = self.S.design(self.cands[g][j][1])
        return float(np.isfinite(Xs).all(1)[np.isfinite(self.S.y)].mean())

    def daily_design(self, blocks, names):
        S = self.S
        idx_save, pair_save = S.idx, S.pair_i
        S.idx = np.arange(L.ND); S.pair_i = None
        try:
            A, nm, gr = L.assemble(S, blocks)
        finally:
            S.idx, S.pair_i = idx_save, pair_save
        env = np.array([ENVLIKE(g) for g in gr])
        A[:, env] = np.where(np.isfinite(A[:, env]), A[:, env], 0.0)
        return A[:, [nm.index(n) for n in names]]

    # ---------------------------------------------------------------- environmental selection
    def select_env(self, core):
        S = self.S
        tests, best_param = [], {}
        for g in self.cands:
            ss = []
            for j in range(len(self.cands[g])):
                blk, lab = self.env_block(g, j)
                cov = self.coverage(g, j)
                rows = None
                if cov < COV_MIN:
                    rows = np.isfinite(S.design(self.cands[g][j][1])).all(1)
                ss.append((self.fit(core + [blk], rows=rows, nit=2)['sse'] if cov > 0.2 else np.inf, j, lab, cov, rows))
            sse_b, j, lab, cov, rows = min(ss, key=lambda z: z[0])
            if not np.isfinite(sse_b):
                continue
            best_param[g] = j
            blk, _ = self.env_block(g, j)
            fa = self.fit(core + [blk], rows=rows, cv=True)
            fb = self.fit(core, rows=fa['ok'], cv=True)
            ib = [i for i, gg in enumerate(fa['groups']) if gg == g]
            tests.append(dict(group=g, param=lab, coverage=cov, n=fa['n'], cv_core=fb['cv'], cv_with=fa['cv'],
                              dcv_rel=(fa['cv'] - fb['cv']) / fb['cv'], dbic=self.B(fa) - self.B(fb),
                              coef=[float(fa['beta'][i]) for i in ib], dsse_rel=(fa['sse'] - fb['sse']) / fb['sse'],
                              coef_sd=self.bootstrap(fa, nb=100, seed=11)[:, ib].std(0).tolist()))
        T = {t['group']: t for t in tests}
        pool = [g for g in best_param if g != 'annual' and T[g]['coverage'] >= COV_MIN]

        def passes(fa, fb):
            return (fa['cv'] - fb['cv']) / fb['cv'] <= -CV_GAIN and self.B(fa) - self.B(fb) <= BIC_GAIN

        sel, cur = [], list(core)
        while True:                                   # forward
            best = None
            for g in pool:
                if g in sel or (g in L.HYDRO and any(s in L.HYDRO for s in sel)):
                    continue
                blk, _ = self.env_block(g, best_param[g])
                fa = self.fit(cur + [blk], cv=True)
                fb = self.fit(cur, rows=fa['ok'], cv=True)
                dcv = (fa['cv'] - fb['cv']) / fb['cv']
                if passes(fa, fb) and (best is None or dcv < best[1]):
                    best = (g, dcv, blk)
            if best is None:
                break
            sel.append(best[0]); cur.append(best[2])
        while True:                                   # backward, same criteria
            drop = None
            for blk in [b for b in cur if b[0] in sel]:
                fa = self.fit(cur, cv=True)
                fb = self.fit([b for b in cur if b is not blk], rows=fa['ok'], cv=True)
                if not passes(fa, fb):
                    drop = blk; break
            if drop is None:
                break
            sel.remove(drop[0]); cur = [b for b in cur if b is not drop]
        return sel, best_param, tests, cur

    # ---------------------------------------------------------------- main procedure
    def run(self):
        S = self.S
        out = dict(dataset=S.dataset, sta=S.sta, chan=S.chan, n=len(S.y), t0=str(L.DAYS[S.idx.min()].date()),
                   t1=str(L.DAYS[S.idx.max()].date()), breaks=list(S.breaks))
        out['chg_identifiable'] = not any(DECLINE_WIN[0] <= str(b)[:10] <= DECLINE_WIN[1] for b in S.breaks)
        # 1. shared quake time constant (core + provisional annual so seasonality does not bias it)
        ann = ('annual', ['ann_s1', 'ann_c1', 'ann_s2', 'ann_c2'], L.annual_cols())
        taus = self.tau_grid()
        self.Lc = 1.0
        sse_tau = {t: self.fit(self.core_blocks(t) + [ann], nit=2)['sse'] for t in taus}
        tau = min(sse_tau, key=sse_tau.get)
        out['tau_q'] = tau; out['tau_grid'] = list(taus); out['sse_tau'] = sse_tau
        core = self.core_blocks(tau)
        f0 = self.fit(core, cv=True)
        if self.pairs:
            ne = max(getattr(S, 'n_events', f0['n']) - len(np.unique(S.cluster)), 10)
            self.Lc = max(1.0, f0['n'] / ne)
        else:
            self.Lc = L.decorr_length(f0['r'], self.spacing)
        out['L_decorr'] = self.Lc
        out['core'] = dict(cv=f0['cv'], sse=f0['sse'], n=f0['n'], k=f0['k'], bic=self.B(f0))
        # 2-3. environmental selection
        sel, best_param, tests, cur = self.select_env(core)
        out['tests'] = tests; out['selected'] = sel
        out['selected_params'] = {g: self.cands[g][best_param[g]][0] for g in sel}
        # collinearity flag: selected env regressor vs const + trend + ramp21
        A0 = S.design(np.column_stack([np.ones(L.ND), self.longc['trend'], self.longc['ramp21']]))
        out['env_r2_vs_longterm'] = {}
        for g in sel:
            X = S.design(self.cands[g][best_param[g]][1])
            ok = np.isfinite(X).all(1) & np.isfinite(A0).all(1)
            r2 = []
            for c in range(X.shape[1]):
                b, *_ = np.linalg.lstsq(A0[ok], X[ok, c], rcond=None)
                r2.append(1 - np.var(X[ok, c] - A0[ok] @ b) / max(np.var(X[ok, c]), 1e-30))
            out['env_r2_vs_longterm'][g] = float(max(r2))
        # 4. final = full core + selected environment
        self._final_blocks = cur
        ff = self.fit(cur, cv=True)
        out['final'] = dict(cv=ff['cv'], sse=ff['sse'], n=ff['n'], k=ff['k'], bic=self.B(ff), names=ff['names'],
                            groups=ff['groups'], beta=ff['beta'].tolist(), r2=1 - np.var(ff['r']) / np.var(S.y[ff['ok']]))
        drop = {}
        for blk in cur:
            g = blk[0]
            if g == 'base':
                continue
            fd = self.fit([b for b in cur if b is not blk], rows=ff['ok'], cv=True)
            sst = np.sum((S.y[ff['ok']] - S.y[ff['ok']].mean()) ** 2 * (1 if S.w is None else S.w[ff['ok']]))
            drop[g] = dict(dcv_rel=(ff['cv'] - fd['cv']) / fd['cv'], dbic=out['final']['bic'] - self.B(fd), partial_r2=(fd['sse'] - ff['sse']) / sst)
        out['drop_one'] = drop
        # 5. bootstrap of the full model (headline SDs)
        Bt = self.bootstrap(ff)
        out['boot_sd'] = Bt.std(0).tolist()
        # 6. component series and derived quantities
        out['components'] = self.derived(ff, Bt, tau, cur)
        # 7. model averaging over the 16 core subsets (environment fixed)
        out['model_avg'] = self.model_average(cur, tau)
        # 8. unconstrained fit: delayed post-quake change (opposite-sign step/transient)
        fu = self.fit(cur, constrain=False)
        dl = {}
        for q in L.BIG3:
            if f'step_{q}' in fu['names'] and f'trans_{q}' in fu['names']:
                s_, t_ = fu['beta'][fu['names'].index(f'step_{q}')], fu['beta'][fu['names'].index(f'trans_{q}')]
                dl[q] = dict(step_u=float(s_), trans_u=float(t_), delayed=float(-t_) if s_ * t_ < 0 else 0.0)
        out['delayed'] = dict(by_quake=dl, dbic_constrained_minus_unconstrained=self.B(ff) - self.B(fu))
        # 9. quake-model tests and long-term alternatives
        out['quake_tests'] = self.quake_tests(cur, tau)
        out['long_tests'] = self.long_tests(cur)
        # 10. placebos
        out['placebo_quake'] = self.placebo_quake(cur, tau)
        out['placebo_env'] = self.placebo_env(cur, sel)
        # 11. per-quake recovery-time profiles with likelihood intervals
        out['tau_profiles'] = self.tau_profiles(cur, tau)
        return out

    # ---------------------------------------------------------------- pieces
    def bootstrap(self, ff, nb=NBOOT, seed=0):
        rng = np.random.default_rng(seed_of(seed, self.S.dataset, self.S.sta, self.S.chan))
        A = ff['A'][ff['ok']]; r = ff['r']; fit = A @ ff['beta']
        w = None if self.S.w is None else self.S.w[ff['ok']]
        n = len(r)
        out = np.zeros((nb, len(ff['beta'])))
        pr = ff.get('pairs')
        if self.pairs:
            cl = self.S.cluster[ff['ok']]; uc = np.unique(cl)
            groups = {c: np.where(cl == c)[0] for c in uc}
            for b in range(nb):
                ii = np.concatenate([groups[c] for c in rng.choice(uc, len(uc))])
                out[b], _ = L.huber_lsq(A[ii], (fit + r)[ii], None if w is None else w[ii], nit=2, pairs=pr)
            return out
        bl = max(3, 180 // self.spacing)
        for b in range(nb):
            starts = rng.integers(0, n, int(np.ceil(n / bl)))
            ii = (starts[:, None] + np.arange(bl)[None, :]).ravel()[:n] % n
            out[b], _ = L.huber_lsq(A, fit + r[ii], w, nit=2, pairs=pr)
        return out

    def comp_series(self, Ad, groups, beta, B=None):
        """Daily component series per group and composite: dict name -> (series, bootstrap matrix or None)."""
        groups = np.array(groups)
        ug = [g for g in dict.fromkeys(groups) if g != 'base']
        comp = {'ENV': [g for g in ug if ENVLIKE(g)], 'TOTAL': ug}
        for k, v in COMPOSITE.items():
            comp[k] = [g for g in ug if g in v]
        out = {}
        for g in ug + [k for k, v in comp.items() if v]:
            ii = np.where(np.isin(groups, comp.get(g, [g])))[0]
            out[g] = (Ad[:, ii] @ beta[ii], None if B is None else Ad[:, ii] @ B[:, ii].T, ii)
        return out

    def chg(self, cs):
        S = self.S
        span = np.zeros(L.ND, bool); span[S.idx.min():S.idx.max() + 1] = True
        i21 = span & (L.DAYS >= '2021-01-01') & (L.DAYS < '2021-12-19')
        i25 = span & (L.DAYS >= '2025-01-01') & (L.DAYS < '2026-01-01')
        if i21.sum() < 200 or i25.sum() < 200:
            return None
        return np.nanmean(cs[i25], 0) - np.nanmean(cs[i21], 0)

    def derived(self, ff, Bt, tau, blocks):
        S = self.S
        names, beta = ff['names'], ff['beta']
        Ad = self.daily_design(blocks, names)
        ok = ff['ok']
        yv = S.y[ok]
        vy = np.var(yv)
        span = np.zeros(L.ND, bool); span[S.idx.min():S.idx.max() + 1] = True
        H = np.column_stack([np.ones(L.ND), np.sin(2 * np.pi * L.TY), np.cos(2 * np.pi * L.TY)])
        comp = {}
        for g, (cs, csb, ii) in self.comp_series(Ad, ff['groups'], beta, Bt).items():
            samp = ff['A'][ok][:, ii] @ beta[ii]
            d = dict(group=g, cols=[names[i] for i in ii], coef=beta[ii].tolist(), coef_sd=Bt[:, ii].std(0).tolist(),
                     var_frac=float(np.var(samp) / vy))
            sp = span & np.isfinite(cs)
            if sp.sum() > 730:
                c, *_ = np.linalg.lstsq(H[sp], cs[sp], rcond=None)
                cb, *_ = np.linalg.lstsq(H[sp], np.nan_to_num(csb[sp]), rcond=None)
                d['annual_amp'] = float(np.hypot(c[1], c[2])); d['annual_amp_sd'] = float(np.std(np.hypot(cb[1], cb[2])))
                d['annual_peak_doy'] = float((np.degrees(np.arctan2(c[1], c[2])) % 360) / 360 * 365.25)
            v = self.chg(cs)
            if v is not None:
                d['chg_2125'] = float(v); d['chg_2125_sd'] = float(np.std(self.chg(csb)))
            comp[g] = (d, cs.astype(np.float32))
        tt = L.DAYS[S.idx]
        extra = {}
        o21 = ok & (tt >= '2021-01-01') & (tt < '2021-12-19'); o25 = ok & (tt >= '2025-01-01') & (tt < '2026-01-01')
        if o21.sum() > 20 and o25.sum() > 20 and S.pair_i is None:       # pairwise (repeater) data have no level
            io = [i for i, n in enumerate(names) if n.startswith('offset_')]
            ycor = S.y - (ff['A'][:, io] @ beta[io] if io else 0.0)
            extra['obs_chg_2125'] = float(ycor[o25].mean() - ycor[o21].mean())
            extra['obs_chg_2125_raw'] = float(S.y[o25].mean() - S.y[o21].mean())
        Hs = S.design(H)[ok]
        if S.pair_i is None:
            c, *_ = np.linalg.lstsq(Hs, yv, rcond=None)
            extra['obs_annual_amp'] = float(np.hypot(c[1], c[2]))
            extra['obs_annual_peak_doy'] = float((np.degrees(np.arctan2(c[1], c[2])) % 360) / 360 * 365.25)
        c, *_ = np.linalg.lstsq(Hs, ff['r'], rcond=None)
        extra['resid_annual_amp'] = float(np.hypot(c[1], c[2]))
        ry = pd.Series(ff['r'], index=tt[ok].year).groupby(level=0).agg(['mean', 'size'])
        ry = ry[ry['size'] >= (100 if self.spacing < 5 and S.pair_i is None else 6)]['mean']
        d4 = [ry[y + 4] - ry[y] for y in ry.index if y + 4 in ry.index]
        extra['resid_4yr_rms'] = float(np.sqrt(np.mean(np.square(d4)))) if len(d4) >= 3 else np.nan
        extra['resid_4yr_n'] = len(d4)
        for q in L.BIG3:
            if f'step_{q}' in names:
                i1 = names.index(f'step_{q}'); i2 = names.index(f'trans_{q}') if f'trans_{q}' in names else None
                tr_, trb = (beta[i2], Bt[:, i2]) if i2 is not None else (0.0, 0.0)
                v0 = beta[i1] + tr_; vb = Bt[:, i1] + trb
                f1 = np.exp(-365 / tau)
                extra[f'drop0_{q}'] = float(v0); extra[f'drop0_{q}_sd'] = float(np.std(vb))
                extra[f'level1yr_{q}'] = float(beta[i1] + tr_ * f1); extra[f'level1yr_{q}_sd'] = float(np.std(Bt[:, i1] + trb * f1))
                extra[f'recovered_frac_{q}'] = float(tr_ / v0) if v0 != 0 else np.nan
                rfb = np.where(vb != 0, trb / np.where(vb != 0, vb, 1), np.nan)
                extra[f'recovered_frac_{q}_sd'] = float(np.nanstd(rfb))
                extra[f'recovered_frac_{q}_q05'] = float(np.nanpercentile(rfb, 5)) if np.isfinite(rfb).any() else np.nan
                extra[f'recovered_frac_{q}_q95'] = float(np.nanpercentile(rfb, 95)) if np.isfinite(rfb).any() else np.nan
        extra['fitted_daily'] = (Ad @ beta).astype(np.float32)
        return dict(groups=comp, extra=extra)

    def model_average(self, cur, tau):
        env = [b for b in cur if ENVLIKE(b[0])]
        rows = []
        for k in range(len(L.CORE) + 1):
            for sub in itertools.combinations(L.CORE, k):
                blocks = self.core_blocks(tau, groups=sub) + env
                f = self.fit(blocks, nit=3)
                Ad = self.daily_design(blocks, f['names'])
                cs = self.comp_series(Ad, f['groups'], f['beta'])
                row = dict(subset='+'.join(sub) or 'none', bic=self.B(f), k=f['k'])
                for g in ('QUAKES', 'LONGTERM', 'ENV', 'TOTAL', 'POST21'):
                    v = self.chg(cs[g][0]) if g in cs else (0.0 if self.chg(np.zeros(L.ND)) is not None else None)
                    row[f'chg_{g}'] = np.nan if v is None else float(v)
                for q in L.BIG3:
                    if f'step_{q}' in f['names']:
                        row[f'drop0_{q}'] = float(f['beta'][f['names'].index(f'step_{q}')] +
                                                  (f['beta'][f['names'].index(f'trans_{q}')] if f'trans_{q}' in f['names'] else 0))
                rows.append(row)
        M = pd.DataFrame(rows)
        M['dbic'] = M.bic - M.bic.min()
        M['w'] = np.exp(-0.5 * M.dbic); M['w'] /= M.w.sum()
        return M.to_dict('list')

    def quake_tests(self, cur, tau):
        env = [b for b in cur if ENVLIKE(b[0])]
        res = {}
        for mode in ('free3', 'free3_log', 'free3_only', 'pgv_all', 'strain_all'):
            fa = self.fit(self.core_blocks(tau, mode) + env, cv=True)
            res[mode] = dict(cv=fa['cv'], bic=self.B(fa), sse=fa['sse'], n=fa['n'], k=fa['k'],
                             coef={n: float(b) for n, b in zip(fa['names'], fa['beta']) if n.startswith(('dose', 'step', 'trans'))})
        return res

    def long_tests(self, cur):
        env = [b for b in cur if b[0] not in ('trend', 'ramp21')]
        lc = self.longc
        variants = {'trend+ramp21': [('trend', ['trend'], lc['trend']), ('ramp21', ['ramp21'], lc['ramp21'])],
                    'trend_only': [('trend', ['trend'], lc['trend'])], 'none': []}
        if 'gDilPost' in self.D and np.isfinite(self.D['gDilPost']).all():
            variants['trend+gnss_postdil'] = [('trend', ['trend'], lc['trend']), ('postdil', ['gnss_postdil'], L.anom(self.D['gDilPost'])[:, None])]
        res = {}
        for k, v in variants.items():
            fa = self.fit(env + v, cv=True)
            res[k] = dict(cv=fa['cv'], bic=self.B(fa),
                          coef={n: float(b) for n, b in zip(fa['names'], fa['beta']) if n in ('trend', 'ramp21', 'gnss_postdil')})
        return res

    def placebo_quake(self, cur, tau, nrand=30):
        S = self.S
        i = np.arange(L.ND)
        t0, t1 = S.idx.min() + 60, S.idx.max() - 60
        big = [L.big3_day(n) for n in L.BIG3]
        dates = []
        for y in range(2008, 2027):
            d = int(L.dayidx([pd.Timestamp(f'{y}-12-06')])[0])
            if t0 < d < t1 and all(abs(d - b) > 90 for b in big):
                dates.append(('season', d))
        rng = np.random.default_rng(seed_of('placebo', S.dataset, S.sta, S.chan))
        cand = [int(d) for d in np.unique(S.idx) if t0 < d < t1 and all(abs(d - b) > 90 for b in big)]
        for d in (rng.choice(cand, min(nrand, len(cand)), replace=False) if cand else []):
            dates.append(('random', int(d)))
        out = []
        for kind, d in dates:
            if not (self.supported('step', d) and self.supported('trans', d, tau)):
                continue
            X = np.column_stack([(i >= d).astype(float), np.where(i >= d, np.exp(-(i - d) / tau), 0.0)])
            fa = self.fit(cur + [('fake', ['fake_step', 'fake_trans'], X)], nit=2)
            if 'fake_step' in fa['names']:
                b = fa['beta']; nm = fa['names']
                v1 = b[nm.index('fake_step')]
                v = v1 + (b[nm.index('fake_trans')] if 'fake_trans' in nm else 0)
                out.append(dict(kind=kind, day=str(L.DAYS[d].date()), drop0=float(v), step=float(v1)))
        return out

    def placebo_env(self, cur, sel):
        """Water-year permutation placebo (NPERM common permutations): each selected group is rebuilt from its raw
        driver(s) with complete water years re-ordered (seasonality kept, interannual timing scrambled); its grid
        parameter is re-optimised for every permutation, as it is for the real driver."""
        res = {}
        for g in sel:
            keys = [k for k in L.RAWKEY.get(g, []) if k in self.D]
            if not keys:
                continue

            def best_sse(D2):
                v = L.group_variants(g, D2)
                if not v:
                    return np.nan
                best = np.inf
                for lab, X in v:
                    blk = (g, [f'{g}[{k}]' for k in range(X.shape[1])] if X.shape[1] > 1 else [g], X)
                    best = min(best, self.fit([b if b[0] != g else blk for b in cur], nit=2)['sse'])
                return best

            real = best_sse(self.D)
            null = []
            for p in perms():
                D2 = dict(self.D)
                for k in keys:
                    D2[k] = L.wy_permute(self.D[k], p)
                null.append(best_sse(D2))
            null = np.array(null)
            res[g] = dict(sse_real=float(real), sse_perm=null.astype(np.float32),
                          p=float((1 + np.sum(null <= real)) / (1 + np.isfinite(null).sum())))
        return res

    def tau_profiles(self, cur, tau):
        env = [b for b in cur if ENVLIKE(b[0])]
        out = {}
        for q in L.BIG3:
            prof = {}
            for tq in TAU_PROFILE:
                fa = self.fit(self.core_blocks(tau, tau_by={q: tq}) + env, nit=3)
                if f'trans_{q}' in fa['names']:
                    b = fa['beta']; nm = fa['names']
                    prof[tq] = dict(sse=fa['sse'], n=fa['n'], trans=float(b[nm.index(f'trans_{q}')]),
                                    step=float(b[nm.index(f'step_{q}')]) if f'step_{q}' in nm else np.nan)
            if len(prof) < 3:
                continue
            ts = np.array(sorted(prof)); sse = np.array([prof[t]['sse'] for t in ts]); n = prof[ts[0]]['n']
            dev = (n / self.Lc) * np.log(sse / sse.min())
            inside = ts[dev < 3.84]
            lo_edge, hi_edge = inside.min() == ts.min(), inside.max() == ts.max()
            if lo_edge and hi_edge:
                summ = 'unconstrained'
            elif hi_edge:
                summ = f'>= {inside.min()} d (no upper bound within {ts.max()} d)'
            elif lo_edge:
                summ = f'<= {inside.max()} d'
            else:
                summ = f'{ts[np.argmin(sse)]} d (95% {inside.min()}-{inside.max()} d)'
            out[q] = dict(profile={int(t): prof[t] for t in ts}, dev=dev.tolist(), interval=[int(inside.min()), int(inside.max())], summary=summ,
                          best=int(ts[np.argmin(sse)]) if not (lo_edge or hi_edge) else None)
        return out


def run_one(S):
    t0 = time.time()
    try:
        res = Fitter(S).run()
        res['runtime_s'] = time.time() - t0
    except Exception:
        import traceback
        res = dict(dataset=S.dataset, sta=S.sta, chan=S.chan, error=traceback.format_exc())
    return res


def all_series(which):
    out = []
    if 'SC' in which:
        out += load_sc()
    if 'LFE' in which:
        out += load_lfe()
    if 'LP' in which:
        out += load_lp()
    if 'REP' in which:
        out += load_rep()
    return out


def select_series(a):
    SS = all_series(a.datasets.split(','))
    if a.only:
        SS = [s for s in SS if s.sta in a.only.split(',') and (not a.chan or s.chan in a.chan.split(','))]
    SS.sort(key=lambda s: (s.sta, s.dataset, s.chan))
    return SS


def stage_fit(a):
    SS = select_series(a)
    print('series', len(SS), flush=True)
    results, done = [], 0
    if a.workers <= 1:
        it = map(run_one, SS)
    else:
        import multiprocessing as mp
        pool = mp.get_context('fork').Pool(a.workers)
        it = pool.imap_unordered(run_one, SS, chunksize=1)
    for r in it:
        results.append(r); done += 1
        print(done, r['dataset'], r['sta'], r['chan'], r.get('selected'), 'ERR' if 'error' in r else f"{r.get('runtime_s', 0):.0f}s", flush=True)
        if 'error' in r:
            print(r['error'], flush=True)
    if a.workers > 1:
        pool.close(); pool.join()
    with open(f'{RES}/results_{a.tag}.pkl', 'wb') as f:
        pickle.dump(results, f)
    print('saved', f'{RES}/results_{a.tag}.pkl', flush=True)


# ================================================================================================ collect
UNITS = {'const': '%', 'offset': '%', 'step': '% (permanent part)', 'trans': '% (recovering part at t_q, decays exp(-t/tau))',
         'dose_pgv_trans': '% per cm/s PGV, transient (recovers with tau_q)', 'dose_logpgv_trans': '% per log10(PGV/0.05 cm/s), transient',
         'dose_pgvmod': '% per cm/s PGV (modelled events)', 'dose_strain': '% per nanostrain static vol. strain (modelled events)',
         'trend': '%/yr', 'ramp21': '%/yr after 2021-07-01', 'hyd_exp_rean': '% per mm (POWER precip, exp-drained)',
         'hyd_exp': '% per mm (exp-drained rain storage)', 'hyd_diff': '% per mm (diffused rain anomaly)',
         'rain_lags': '% per mm (rain in lag window)', 'soilm': '% per unit soil moisture/wetness', 'streamflow': '% per log10(cfs)',
         'thermal': '% per degC', 'pressure': '% per hPa', 'sealevel': '% per cm', 'gnss_up': '% per mm uplift',
         'gnss_dil': '% per nanostrain (GNSS areal)', 'bstrain': '% per nanostrain (borehole areal, uncalibrated)',
         'micro': '% per decade of power', 'annual': '%', 'postdil': '% per nanostrain'}
FAMILY = {**{g: 'hydrology' for g in L.HYDRO}, 'thermal': 'thermal', 'gnss_up': 'geodesy', 'gnss_dil': 'geodesy',
          'bstrain': 'geodesy', 'micro': 'nuisance', 'pressure': 'pressure', 'sealevel': 'sea level',
          'big3': 'quakes', 'dose': 'quakes', 'trend': 'long-term', 'ramp21': 'long-term'}


def unit_of(name):
    for k in sorted(UNITS, key=len, reverse=True):
        if name.startswith(k):
            return UNITS[k]
    return ''


def load_results(tag):
    import glob
    R = []
    for fn in sorted(glob.glob(f'{RES}/results_{tag}*.pkl')):
        R += pickle.load(open(fn, 'rb'))
    return R


def stage_collect(a):
    cc = L.station_coords()
    R = load_results(a.tag)
    rows, scores, tests, ts, mavg = [], [], [], [], []
    perm_tab = {}
    for r in R:
        if 'error' in r:
            print('ERROR', r['dataset'], r['sta'], r['chan']); continue
        key = dict(dataset=r['dataset'], sta=r['sta'], chan_band=r['chan'])
        loc = dict(lat=cc.lat.get(r['sta'], np.nan), lon=cc.lon.get(r['sta'], np.nan), sensor_depth_m=cc.depth_m.get(r['sta'], np.nan),
                   t0=r['t0'], t1=r['t1'], n_samples=r['n'], chg_identifiable=r['chg_identifiable'])
        f = r['final']; comps = r['components']['groups']; ex = r['components']['extra']; drop = r['drop_one']
        pq = pd.DataFrame(r['placebo_quake'])
        sd_pl = float(np.sqrt(np.mean(pq.drop0 ** 2))) if len(pq) >= 5 else np.nan
        sd_pl_step = float(np.sqrt(np.mean(pq.step ** 2))) if len(pq) >= 5 else np.nan
        sd_pl_season = float(np.sqrt(np.mean(pq[pq.kind == 'season'].drop0 ** 2))) if len(pq) and (pq.kind == 'season').sum() >= 3 else np.nan
        pe = r['placebo_env']
        for g, v in pe.items():
            perm_tab.setdefault(g, []).append((r['dataset'], r['sta'], r['chan'], v['sse_real'], v['sse_perm']))
        for nm, b, sd, g in zip(f['names'], f['beta'], r['boot_sd'], f['groups']):
            d = comps.get(g, ({}, None))[0]
            par = r['selected_params'].get(g, f"tau_q={r['tau_q']}d" if g in ('big3', 'dose') else '')
            sdh = max(sd, sd_pl_step) if nm.startswith('step_') and np.isfinite(sd_pl_step) else sd
            pv = pe.get(g, {}).get('p', np.nan)
            rows.append(dict(**key, component=nm, kind='coef', group=g, family=FAMILY.get(g, g), amp=b, sd=sdh, sd_boot=sd, units=unit_of(nm),
                             param=par, var_frac=d.get('var_frac', np.nan), partial_r2=drop.get(g, {}).get('partial_r2', np.nan),
                             dbic_drop=drop.get(g, {}).get('dbic', np.nan), dcv_drop=drop.get(g, {}).get('dcv_rel', np.nan),
                             p_wy_perm=pv, attribution=('seasonal, cause unresolved' if (ENVLIKE(g) and np.isfinite(pv) and pv > 0.05) else ''),
                             r2_vs_longterm=r['env_r2_vs_longterm'].get(g, np.nan),
                             note=('PGV includes a station site factor (measured/BSSA14, event terms removed; class median where unmeasured): '
                                   'compare dose coefficients across stations with caution' if g == 'dose' else ''), **loc))
        M = pd.DataFrame(r['model_avg'])
        for g, (d, cs) in comps.items():
            for k in ('annual_amp', 'chg_2125'):
                if k not in d:
                    continue
                sdk = d.get(f'{k}_sd', np.nan)
                p4 = ex.get('resid_4yr_rms', np.nan) if k == 'chg_2125' else np.nan
                extra = {}
                if k == 'chg_2125' and g in ('QUAKES', 'LONGTERM', 'ENV', 'TOTAL', 'POST21') and f'chg_{g}' in M:
                    mm = M[np.isfinite(M[f'chg_{g}'])]
                    inset = mm[mm.dbic < DBIC_SET]
                    extra = dict(bma=float(np.sum(mm.w * mm[f'chg_{g}']) / mm.w.sum()) if len(mm) else np.nan,
                                 range_lo=float(inset[f'chg_{g}'].min()) if len(inset) else np.nan,
                                 range_hi=float(inset[f'chg_{g}'].max()) if len(inset) else np.nan)
                note = ''
                if k == 'chg_2125' and g in ('QUAKES', 'LONGTERM'):
                    note = 'HEADLINE = range_lo..range_hi over core subsets (dBIC<6); full-model amp/sd not separable from the ramp'
                if k == 'chg_2125' and g == 'POST21':
                    note = 'post-2021 change = quake terms + ramp21 (identifiable sum; a positive ramp21 with negative quake terms may be slow healing)'
                depth_row = ((k == 'chg_2125' and g in ('POST21', 'TOTAL', 'ENV') and r['chg_identifiable']) or
                             (k == 'annual_amp' and g in ('ENV', 'TOTAL')))
                rows.append(dict(**key, component=f'{k}[{g}]', kind='derived', group=g, family=FAMILY.get(g, g), amp=d[k],
                                 sd=np.nanmax([sdk, p4]) if np.isfinite(p4) else sdk, sd_boot=sdk, sd_placebo_4yr=p4, units='%',
                                 param=f"peak_doy={d.get('annual_peak_doy', np.nan):.0f}" if k == 'annual_amp' else '2025 mean - 2021 mean (pre-Petrolia)',
                                 note=note, for_depth_mapping=bool(depth_row),
                                 var_frac=d['var_frac'], partial_r2=drop.get(g, {}).get('partial_r2', np.nan), **extra, **loc))
            ts.append((r['dataset'], r['sta'], r['chan'], g, cs))
        ts.append((r['dataset'], r['sta'], r['chan'], 'FIT', ex['fitted_daily']))
        for q in L.BIG3:
            if f'drop0_{q}' in ex:
                sdb = ex[f'drop0_{q}_sd']; sdh = max(sdb, sd_pl) if np.isfinite(sd_pl) else sdb
                tp = r['tau_profiles'].get(q, {})
                mq = M[f'drop0_{q}'] if f'drop0_{q}' in M else pd.Series(dtype=float)
                ok = np.isfinite(mq)
                rows.append(dict(**key, component=f'drop0_{q}', kind='derived', group='big3', family='quakes', amp=ex[f'drop0_{q}'], sd=sdh, sd_boot=sdb,
                                 for_depth_mapping=True,
                                 units='% (permanent + recovering part at t_q)', param=f"tau_q={r['tau_q']}d; tau_{q}: {tp.get('summary', 'n/a')}",
                                 z_placebo=ex[f'drop0_{q}'] / sd_pl if np.isfinite(sd_pl) else np.nan, sd_placebo=sd_pl, sd_placebo_season=sd_pl_season,
                                 bma=float(np.sum(M.w[ok] * mq[ok]) / M.w[ok].sum()) if ok.any() else np.nan, **loc))
                rows.append(dict(**key, component=f'level1yr_{q}', kind='derived', group='big3', family='quakes', amp=ex[f'level1yr_{q}'],
                                 sd=ex[f'level1yr_{q}_sd'], sd_boot=ex[f'level1yr_{q}_sd'], units='% one year after t_q', for_depth_mapping=True, **loc))
                rf, lo5, hi95 = ex[f'recovered_frac_{q}'], ex.get(f'recovered_frac_{q}_q05', np.nan), ex.get(f'recovered_frac_{q}_q95', np.nan)
                if np.isfinite(sd_pl) and abs(ex[f'drop0_{q}']) < 3 * sd_pl:
                    rf_amp, rf_note = np.nan, f'suppressed: |drop0| < 3 placebo SD ({3 * sd_pl:.3f}%)'
                elif not np.isfinite(lo5) or lo5 <= 0.02:
                    rf_amp, rf_note = np.nan, f'not distinguishable from 0: upper limit (95th pct) {hi95:.2f}; point value {rf:.2f}'
                else:
                    rf_amp, rf_note = rf, f'90% interval {lo5:.2f}-{hi95:.2f}'
                rows.append(dict(**key, component=f'recovered_frac_{q}', kind='derived', group='big3', family='quakes', amp=rf_amp,
                                 sd=ex[f'recovered_frac_{q}_sd'], sd_boot=ex[f'recovered_frac_{q}_sd'], range_lo=lo5, range_hi=hi95,
                                 units='fraction of drop0 that recovers (0-1)', note=rf_note, **loc))
            dq = r['delayed']['by_quake'].get(q)
            if dq:
                rows.append(dict(**key, component=f'delayed_{q}', kind='derived', group='delayed', family='quakes (delayed post-quake change)',
                                 amp=dq['delayed'], sd=np.nan, units='% (unconstrained fit: post-quake change opposite to recovery; 0 if none)',
                                 param=f"unconstrained step {dq['step_u']:.4f}, trans {dq['trans_u']:.4f}; dBIC(constr-unconstr)={r['delayed']['dbic_constrained_minus_unconstrained']:.1f}", **loc))
        for k in ('obs_chg_2125', 'obs_chg_2125_raw', 'obs_annual_amp', 'resid_annual_amp', 'resid_4yr_rms'):
            if k in ex:
                rows.append(dict(**key, component=k, kind='observed', group='data', family='data', amp=ex[k], sd=np.nan, units='%', **loc))
        for t in r['tests']:
            tests.append(dict(**key, **{k: (v if not isinstance(v, list) else json.dumps([round(x, 8) for x in v])) for k, v in t.items()}, **loc))
            if t['group'] in ('gnss_dil', 'bstrain') and len(t['coef']) == 1:
                rows.append(dict(**key, component=f'eta_{t["group"]}', kind='test', group=t['group'], family='geodesy',
                                 amp=t['coef'][0] * 1e7, sd=t['coef_sd'][0] * 1e7,
                                 units='dimensionless (dv/v per unit areal strain); single-driver model, seasonal/short-term only (high-passed / non-tectonic strain)',
                                 param=t['param'], dbic_drop=-t['dbic'], dcv_drop=-t['dcv_rel'], **loc))
        M['dataset'], M['sta'], M['chan_band'] = r['dataset'], r['sta'], r['chan']
        mavg.append(M)
        sc = dict(**key, **loc, tau_q=r['tau_q'], tau_grid=json.dumps(r['tau_grid']), L_decorr=r['L_decorr'], core_cv=r['core']['cv'],
                  core_bic=r['core']['bic'], final_cv=f['cv'], final_bic=f['bic'], final_r2=f['r2'], selected=';'.join(r['selected']),
                  selected_families=';'.join(sorted(set(FAMILY.get(g, g) for g in r['selected']))),
                  selected_params=json.dumps(r['selected_params']), env_r2_vs_longterm=json.dumps(r['env_r2_vs_longterm']),
                  sd_placebo_quake=sd_pl, sd_placebo_season=sd_pl_season,
                  dbic_constr_minus_unconstr=r['delayed']['dbic_constrained_minus_unconstrained'])
        for m, v in r['quake_tests'].items():
            sc[f'q_{m}_cv'] = v['cv']; sc[f'q_{m}_bic'] = v['bic']
            for kk, vv in v['coef'].items():
                if kk.startswith('dose'):
                    sc[f'q_{m}_{kk}'] = vv
        for m, v in r['long_tests'].items():
            sc[f'lt_{m}_cv'] = v['cv']; sc[f'lt_{m}_bic'] = v['bic']
        for g, v in pe.items():
            sc[f'p_wy_perm_{g}'] = v['p']
        for q, v in r['tau_profiles'].items():
            sc[f'tau_{q}'] = v['summary']
        scores.append(sc)
    C = pd.DataFrame(rows)
    # network-level permutation p per environmental group and per family (common permutations across series)
    net = []
    fam_tab = {}
    for g, lst in perm_tab.items():
        fam_tab.setdefault(FAMILY.get(g, g), []).extend(lst)
    ntot = pd.Series([r['dataset'] for r in R if 'error' not in r]).value_counts().to_dict()
    for lab, lst in list(perm_tab.items()) + [(f'FAMILY:{k}', v) for k, v in fam_tab.items()]:
        for ds in sorted(set(x[0] for x in lst)):
            sub = [x for x in lst if x[0] == ds and np.isfinite(x[3]) and np.isfinite(x[4]).all()]
            if not sub:
                continue
            nsel_ = len(sub)
            Treal = np.sum([np.log(x[3]) for x in sub])
            Tnull = np.sum([np.log(np.asarray(x[4], float)) for x in sub], axis=0)
            net.append(dict(group=lab, dataset=ds, n_selected=nsel_, n_total=ntot.get(ds, np.nan),
                            mean_dlogSSE_real_vs_null=float((Treal - np.median(Tnull)) / len(sub)),
                            p_network=float((1 + np.sum(Tnull <= Treal)) / (1 + len(Tnull))),
                            note='conditional on selection: only series where the group was selected enter'))
    N = pd.DataFrame(net)
    N.to_csv(f'{OUT}/permutation_network.csv', index=False, float_format='%.6g')
    C.to_csv(f'{OUT}/components.csv', index=False, float_format='%.6g')
    pd.DataFrame(scores).to_csv(f'{OUT}/model_scores.csv', index=False, float_format='%.6g')
    pd.DataFrame(tests).to_csv(f'{OUT}/candidate_tests.csv', index=False, float_format='%.6g')
    pd.concat(mavg).to_csv(f'{OUT}/model_average.csv', index=False, float_format='%.6g')
    T = pd.DataFrame({f'{d}|{s}|{c}|{g}': v for d, s, c, g, v in ts}, index=L.DAYS)
    T.to_parquet(f'{RES}/component_series_{a.tag}.parquet')
    print('components', C.shape, 'scores', len(scores), 'tests', len(tests))
    if len(N):
        print(N.to_string())


# ================================================================================================ out-of-sample
def stage_oos(a):
    R = {(r['dataset'], r['sta'], r['chan']): r for r in load_results(a.tag) if 'error' not in r}
    SS = select_series(a)
    cut = int(L.dayidx([pd.Timestamp(L.HINGE)])[0])
    rows = []
    for S in SS:
        r = R.get((S.dataset, S.sta, S.chan))
        if r is None or S.pair_i is not None:
            continue
        pre = S.idx < cut
        if pre.sum() < (4 * 365 if S.sm[1] <= 10 else 40) or (S.idx >= cut).sum() < 50:
            continue
        if any(pd.Timestamp(b) >= pd.Timestamp('2021-01-01') for b in S.breaks):
            continue
        # environmental selection redone on the pre-2021-07 data only
        Fp = Fitter(S.subset(pre))
        Fp.Lc = 1.0
        taus = Fp.tau_grid()
        tau = r['tau_q']
        corep = Fp.core_blocks(min(taus, key=lambda t: abs(t - tau)), groups=('dose', 'trend'))
        f0 = Fp.fit(corep)
        Fp.Lc = L.decorr_length(f0['r'], Fp.spacing)
        sel, best_param, _, _ = Fp.select_env(corep)
        Fi = Fitter(S)
        env = [Fi.env_block(g, best_param[g])[0] for g in sel]
        variants = (('pgv_dose+env', Fi.core_blocks(tau, 'pgv_all', groups=('dose', 'trend')) + env),
                    ('env_only', [Fi.core_blocks(tau, groups=())[0], ('trend', ['trend'], Fi.longc['trend'])] + env),
                    ('pgv_dose_only', Fi.core_blocks(tau, 'pgv_all', groups=('dose', 'trend'))))
        for variant, blocks in variants:
            A, names, groups, ok = Fi.matrix(blocks)
            tr = ok & pre
            beta, _ = L.huber_lsq(A[tr], S.y[tr], None if S.w is None else S.w[tr], pairs=L.sign_pairs(names))
            pred = A @ beta
            po = ok & ~pre
            tt = L.DAYS[S.idx]
            o21 = ok & (tt >= '2021-01-01') & (tt < '2021-07-01'); o25 = ok & (tt >= '2025-01-01') & (tt < '2026-01-01')
            m7 = L.big3_day('M7')
            p7 = ok & (S.idx > m7) & (S.idx <= m7 + 30); b7 = ok & (S.idx < m7) & (S.idx >= m7 - 30)
            res = S.y[po] - pred[po]
            row = dict(dataset=S.dataset, sta=S.sta, chan_band=S.chan, variant=variant, selected_pre=';'.join(sel),
                       n_pre=int(tr.sum()), n_post=int(po.sum()),
                       rms_post=float(np.sqrt(np.mean(res ** 2))),
                       rms_post_const=float(np.sqrt(np.mean((S.y[po] - np.median(S.y[tr])) ** 2))),
                       rms_post_offset=float(np.sqrt(np.mean((res - res.mean()) ** 2))),
                       rms_post_const_offset=float(np.std(S.y[po])),
                       r_post_detrended=float(np.corrcoef(S.y[po] - S.y[po].mean(), pred[po] - pred[po].mean())[0, 1]) if np.std(pred[po]) > 0 else np.nan,
                       dose_coef=json.dumps({n: round(float(b), 5) for n, b in zip(names, beta) if n.startswith('dose')}),
                       note=f'tau_q={tau} d taken from the full-record fit (not re-estimated pre-2021)')
            if o21.sum() > 10 and o25.sum() > 10:
                row['obs_chg'] = float(S.y[o25].mean() - S.y[o21].mean()); row['pred_chg'] = float(pred[o25].mean() - pred[o21].mean())
            if p7.sum() > 3 and b7.sum() > 3:
                row['obs_m7_30d'] = float(S.y[p7].mean() - S.y[b7].mean()); row['pred_m7_30d'] = float(pred[p7].mean() - pred[b7].mean())
            rows.append(row)
        print(S.dataset, S.sta, S.chan, 'pre-selected:', sel, flush=True)
    O = pd.DataFrame(rows)
    O.to_csv(f'{OUT}/out_of_sample.csv', index=False, float_format='%.6g')
    if len(O):
        print(O.groupby('variant')[['rms_post', 'rms_post_const', 'rms_post_offset', 'rms_post_const_offset', 'r_post_detrended']].median())


# ================================================================================================ afterslip time constant
def stage_afterslip(a):
    """GNSS M7 postseismic: per near station, pre-M7 trajectory (2022-12-25..2024-12-04: offset, velocity,
    annual+semiannual, equipment steps) extrapolated; post-M7 residual (CME removed) fitted with
    c + A*log(1+t/tau) and c + A*(1-exp(-t/tau)) for a tau grid (A per station/component, tau shared);
    station bootstrap; likelihood interval on the pooled profile."""
    import sci_gnss as gn
    M = pd.read_csv(f'{ROOT}/data/sci/gnss/stations.csv'); M = M[M.set == 'near']
    CME = pd.read_csv(f'{ROOT}/data/sci/gnss/cme.csv.gz').set_index('mjd')
    m7 = gn.dt2mjd('2024-12-05'); fe = gn.dt2mjd('2022-12-25')
    post = {}
    for s in M.sta:
        fn = f'{gn.TEN}/{s}.tenv3'
        if not os.path.exists(fn):
            continue
        df = gn.load_tenv3(s)
        df = df[(df.mjd >= fe) & (df.mjd <= m7 + 700)]
        if (df.mjd < m7).sum() < 400 or (df.mjd > m7 + 300).sum() < 200:
            continue
        mjd = df.mjd.values.astype(float)
        st = gn.steps_for(s); st = st[(st.typ == 1)]
        sm = [m for m in gn.usable_steps(mjd, st.mjd.values.astype(float)) if m < m7]
        pre = mjd < m7
        G, names = gn.design(mjd, sm, (), (), seasonal=True)
        rr = {}
        for c in 'en':
            y = df[c].values * 1e3 - CME[f'cme_{c}'].reindex(df.mjd.values).values
            okk = np.isfinite(y)
            x, _, ok = gn.robust_fit(G[pre & okk], y[pre & okk])
            r = y - G @ x
            rr[c] = (mjd[~pre & okk] - m7, r[~pre & okk])
        post[s] = rr
    taus = np.array([2, 5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560])
    stas = list(post)

    def prof(sel, form):
        out = []
        for tau in taus:
            sse = 0.0
            for s in sel:
                for c in 'en':
                    t, r = post[s][c]
                    f = np.log1p(t / tau) if form == 'log' else 1 - np.exp(-t / tau)
                    A = np.column_stack([np.ones_like(t), f])
                    b, *_ = np.linalg.lstsq(A, r, rcond=None)
                    sse += np.sum((r - A @ b) ** 2)
            out.append(sse)
        return np.array(out)

    rng = np.random.default_rng(3)
    res = {}
    for form in ('log', 'exp'):
        p = prof(stas, form)
        bs = [taus[np.argmin(prof(list(rng.choice(stas, len(stas))), form))] for _ in range(100)]
        edge = np.argmin(p) in (0, len(taus) - 1)
        res[form] = dict(tau_best=None if edge else float(taus[np.argmin(p)]), at_grid_edge=bool(edge), sse=p.tolist(),
                         tau_boot_q=np.percentile(bs, [5, 50, 95]).tolist())
    res['n_stations'] = len(stas); res['taus'] = taus.tolist()
    json.dump(res, open(f'{OUT}/afterslip_tau.json', 'w'), indent=1)
    print(json.dumps({k: (v if k == 'n_stations' else {kk: vv for kk, vv in v.items() if kk != 'sse'}) for k, v in res.items() if k != 'taus'}, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage')
    ap.add_argument('--datasets', default='SC,LFE,LP,REP')
    ap.add_argument('--only', default='')
    ap.add_argument('--chan', default='')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--tag', default='main')
    a = ap.parse_args()
    if a.stage == 'prep':
        print(sc_parquet().shape)
    else:
        globals()[f'stage_{a.stage}'](a)
