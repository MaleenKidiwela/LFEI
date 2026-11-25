#!/usr/bin/env python
"""dv/v attribution fits (per station x band / channel / network band / repeater station).

Stages (python scripts/sci_dvv_attr_fit.py <stage> [--only STA] [--workers N]):
  prep      convert the big SC CSV to parquet once (bulk dir)
  fit       model selection + final fit + bootstrap + placebos for every series -> results pickles (bulk dir)
  collect   tidy tables: data/sci/dvv_attr/components.csv, model_scores.csv, candidate_tests.csv, quake_model_tests.csv
  oos       out-of-sample: fit pre-2021-07, predict 2021-07..2026 (long-record series)
  afterslip GNSS M7 postseismic time constant vs dv/v recovery time constants
Model:  dv/v = const + offsets(instrument breaks) + big-3 [perm step + exp transient, shared tau]
              + shared PGV dose for all other quakes (perm + transient) + trend + ramp(2021-07)
              + selected environmental groups (hydrology / thermal / geodesy / microseism / pressure / sea level)
Every regressor is averaged over the same window as the dv/v sample (sci_dvv_attr_lib.smooth).
"""
import os, sys, json, pickle, argparse, time, warnings
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
SC_BANDS = ['0.1-0.3Hz', '0.3-0.6Hz', '0.5-1Hz', '1-3Hz', '3-5Hz', '0.1-1Hz']
TAUQ = (5, 15, 50, 150, 500)
NBOOT = 200
CV_GAIN = 0.005          # a group must lower CV RMS by >= 0.5 % (relative) ...
BIC_GAIN = -2.0          # ... and BIC by >= 2 to be selected
COV_MIN = 0.8            # environmental regressor must cover >= 80 % of samples to enter the greedy selection


# ================================================================================================ data
def sc_parquet():
    fn = f'{L.BULK}/sc_10day_back.parquet'
    if not os.path.exists(fn):
        x = pd.read_csv(f'{ROOT}/data/sci/noise_dvv_sc_full/sc_10day_back.csv', usecols=['t', 'band', 'combo', 'dvv_pct', 'err_pct', 'sta'])
        x = x[x.combo == 'SC'].drop(columns='combo')
        x['t'] = pd.to_datetime(x.t)
        x.to_parquet(fn)
    return pd.read_parquet(fn)


def mask_windows(sta, t, sm):
    """True where the sample window overlaps an instrument-problem window or straddles a break."""
    kind, n = sm
    t = pd.DatetimeIndex(t)
    pre, post = (0, n - 1) if kind == 'back' else ((n // 2, n // 2) if kind == 'centre' else (0, 0))
    bad = np.zeros(len(t), bool)
    for a, b in L.EXCL.get(sta, []):
        bad |= (t >= pd.Timestamp(a) - pd.Timedelta(days=pre)) & (t <= pd.Timestamp(b) + pd.Timedelta(days=post))
    for b in L.BREAKS.get(sta, []):
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
        bad = mask_windows(sta, g.t, ('point', 1))
        g = g[~bad].sort_values('t')
        if len(g) < 24:
            continue
        w = 1.0 / np.maximum(g.err_pct.values, 1e-3) ** 2
        out.append(L.Series('LFE', sta, ch, g.t.values, g.dvv_pct.values, sm, w=w / np.median(w)))
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
        y = g.dvv.values
        y = y * 100.0                    # stored as a fraction -> %
        out.append(L.Series('LP10', 'REGION', f'{band}Hz_{kind}', g.t.values, y, ('centre', 10), w=w / np.median(w), breaks=[]))
    return out


def load_rep():
    R = pd.read_csv(f'{ROOT}/data/sci/dvv_deep_shallow/rep/rep_pairs_dvv.csv', parse_dates=['ti', 'tj'])
    R = R[(R.cc >= 0.5) & (R.snr >= 2) & np.isfinite(R.eps) & (np.abs(R.eps) < 0.009)]
    out = []
    for sta, g in R.groupby('sta'):
        bi = mask_windows(sta, g.ti, ('point', 1)) | mask_windows(sta, g.tj, ('point', 1))
        g = g[~bi]
        cc = np.clip(g.cc.values, 0, 0.99)
        w = cc ** 2 / (1 - cc ** 2)
        s = L.Series('REP', sta, 'all_comp', g.tj.values, g.eps.values * 100.0, ('point', 1), w=w / np.median(w),
                     pair_i=L.dayidx(g.ti.values))
        s.cluster = g.cl.values
        out.append(s)
    return out


# ================================================================================================ fitting one series
class Fitter:
    def __init__(self, S):
        self.S = S
        sta = S.sta
        self.D = L.drivers(sta)
        band = S.chan if S.dataset == 'SC10' else None
        self.cands = L.candidate_groups(sta, band, self.D)
        self.base = L.base_cols(sta, S.breaks)
        self.longc = L.long_cols()
        self.pairs = S.pair_i is not None

    # ---------------------------------------------------------------- blocks
    def core_blocks(self, tau, qmode='free3', long=('trend', 'ramp21'), tau_by=None):
        Xq, nq = L.quake_cols(self.S.sta, tau, qmode, tau_by=tau_by)
        B = [('base', self.base[1], self.base[0])]
        big = [i for i, n in enumerate(nq) if not n.startswith('dose') and self.supported(n, tau)]
        dose = [i for i, n in enumerate(nq) if n.startswith('dose')]
        if big:
            B.append(('big3', [nq[i] for i in big], Xq[:, big]))
        if dose:
            B.append(('dose', [nq[i] for i in dose], Xq[:, dose]))
        for k in long:
            B.append((k, [k], self.longc[k]))
        return B

    def supported(self, name, tau, d=None, nmin=10):
        """A quake step needs >= nmin samples before t_q and between t_q and the next instrument break;
        a transient needs >= nmin samples within min(3 tau, 365) d after t_q."""
        if d is None:
            d = L.big3_day(name.split('_', 1)[1])
        ii = np.sort(self.S.idx if self.S.pair_i is None else np.concatenate([self.S.idx, self.S.pair_i]))
        bks = sorted(int(L.dayidx([pd.Timestamp(b)])[0]) for b in self.S.breaks)
        nxt = min([b for b in bks if b > d], default=L.ND)
        prv = max([b for b in bks if b <= d], default=0)
        if name.startswith('step') or name.startswith('fake_step'):
            return ((ii >= d) & (ii < nxt)).sum() >= nmin and ((ii < d) & (ii >= prv)).sum() >= nmin
        return ((ii >= d) & (ii <= d + min(3 * tau, 365))).sum() >= nmin

    def env_block(self, g, j):
        lab, X = self.cands[g][j]
        nm = [f'{g}[{k}]' for k in range(X.shape[1])] if X.shape[1] > 1 else [g]
        return (g, nm, X), lab

    def matrix(self, blocks, fill_env=True):
        A, names, groups = L.assemble(self.S, blocks)
        y, w = self.S.y, self.S.w
        if fill_env:          # environmental regressors: NaN (gaps) -> 0 (= mean anomaly)
            env = np.array([g not in ('base', 'big3', 'dose', 'trend', 'ramp21', 'postdil') for g in groups])
            A[:, env] = np.where(np.isfinite(A[:, env]), A[:, env], 0.0)
        ok = L.finite_rows(A, y)
        sup = np.abs(A[ok]).sum(0) > 1e-12
        if self.pairs and 'const' in names:
            sup[names.index('const')] = False
        return A[:, sup], [n for n, s in zip(names, sup) if s], [g for g, s in zip(groups, sup) if s], ok

    def fit(self, blocks, rows=None, cv=False):
        A, names, groups, ok = self.matrix(blocks)
        if rows is not None:
            ok = ok & rows
        y = self.S.y[ok]; w = None if self.S.w is None else self.S.w[ok]
        beta, ww = L.huber_lsq(A[ok], y, w)
        r = y - A[ok] @ beta
        sse = float(np.sum(r ** 2 * (1 if w is None else w)))
        res = dict(beta=beta, names=names, groups=groups, ok=ok, A=A, r=r, sse=sse, n=int(ok.sum()), k=len(beta))
        if cv:
            idx = self.S.idx[ok]
            res['cv'], res['cv_mad'] = L.cv_rms(A[ok], y, idx, w)
        return res

    def coverage(self, g, j):
        X = self.cands[g][j][1]
        Xs = self.S.design(X)
        return float(np.isfinite(Xs).all(1)[np.isfinite(self.S.y)].mean())

    # ---------------------------------------------------------------- main procedure
    def run(self):
        S = self.S
        out = dict(dataset=S.dataset, sta=S.sta, chan=S.chan, n=len(S.y), t0=str(L.DAYS[S.idx.min()].date()),
                   t1=str(L.DAYS[S.idx.max()].date()))
        # 1. quake time constant (core + provisional annual so seasonality does not bias tau)
        ann = ('annual', ['ann_s1', 'ann_c1', 'ann_s2', 'ann_c2'], self.cands['annual'][0][1])
        sse_tau = {}
        for tau in TAUQ:
            sse_tau[tau] = self.fit(self.core_blocks(tau) + [ann])['sse']
        tau = min(sse_tau, key=sse_tau.get)
        out['tau_q'] = tau; out['sse_tau'] = sse_tau
        core = self.core_blocks(tau)
        f0 = self.fit(core, cv=True)
        Lc = L.neff_factor(f0['r']) if not self.pairs else 1
        out['L_decorr'] = Lc
        out['core'] = dict(cv=f0['cv'], sse=f0['sse'], n=f0['n'], k=f0['k'], bic=L.bic(f0['sse'], f0['n'], f0['k'], Lc))
        # 2. single-candidate tests (on each candidate's own sample subset, vs core on the same subset)
        tests = []
        best_param = {}
        for g in self.cands:
            ss = []
            for j in range(len(self.cands[g])):
                blk, lab = self.env_block(g, j)
                cov = self.coverage(g, j)
                rows = None
                if cov < COV_MIN:
                    Xs = S.design(self.cands[g][j][1]); rows = np.isfinite(Xs).all(1)
                ss.append((self.fit(core + [blk], rows=rows)['sse'] if cov > 0.2 else np.inf, j, lab, cov, rows))
            sse_b, j, lab, cov, rows = min(ss, key=lambda z: z[0])
            if not np.isfinite(sse_b):
                continue
            best_param[g] = j
            blk, _ = self.env_block(g, j)
            fa = self.fit(core + [blk], rows=rows, cv=True)
            fb = self.fit(core, rows=rows if rows is not None else fa['ok'], cv=True)
            ib = [i for i, gg in enumerate(fa['groups']) if gg == g]
            tests.append(dict(group=g, param=lab, coverage=cov, n=fa['n'], cv_core=fb['cv'], cv_with=fa['cv'],
                              dcv_rel=(fa['cv'] - fb['cv']) / fb['cv'],
                              dbic=L.bic(fa['sse'], fa['n'], fa['k'], Lc) - L.bic(fb['sse'], fb['n'], fb['k'], Lc),
                              coef=[float(fa['beta'][i]) for i in ib], dsse_rel=(fa['sse'] - fb['sse']) / fb['sse'],
                              coef_sd=self.bootstrap(fa, nb=100, seed=11)[:, ib].std(0).tolist()))
        out['tests'] = tests
        # 3. greedy forward selection over physical groups
        T = {t['group']: t for t in tests}
        pool = [g for g in best_param if g != 'annual' and T[g]['coverage'] >= COV_MIN]
        sel, cur = [], core
        fcur = f0
        while True:
            best = None
            for g in pool:
                if g in sel or (g in L.HYDRO and any(s in L.HYDRO for s in sel)):
                    continue
                blk, _ = self.env_block(g, best_param[g])
                fa = self.fit(cur + [blk], cv=True)
                fb = self.fit(cur, rows=fa['ok'], cv=True)
                dcv = (fa['cv'] - fb['cv']) / fb['cv']
                db = L.bic(fa['sse'], fa['n'], fa['k'], Lc) - L.bic(fb['sse'], fb['n'], fb['k'], Lc)
                if dcv <= -CV_GAIN and db <= BIC_GAIN and (best is None or dcv < best[1]):
                    best = (g, dcv, db, blk, fa)
            if best is None:
                break
            sel.append(best[0]); cur = cur + [best[3]]; fcur = best[4]
        out['selected'] = sel
        out['selected_params'] = {g: self.cands[g][best_param[g]][0] for g in sel}
        # 3b. backward elimination by BIC (n_eff) over the core groups (big3, dose, trend, ramp21): removes
        #     collinear unneeded terms; the coefficient of each pruned group is still reported from an add-back fit
        cur_full = list(cur)
        removed = []
        while True:
            fc = self.fit(cur)
            bc = L.bic(fc['sse'], fc['n'], fc['k'], Lc)
            best = None
            for blk in cur:
                if blk[0] in ('base',) or blk[0] in sel:
                    continue
                fd = self.fit([b for b in cur if b is not blk], rows=fc['ok'])
                bd = L.bic(fd['sse'], fd['n'], fd['k'], Lc)
                if bd < bc and (best is None or bd < best[1]):
                    best = (blk, bd)
            if best is None:
                break
            removed.append(best[0]); cur = [b for b in cur if b is not best[0]]
        out['pruned'] = []
        for blk in removed:
            fa = self.fit(cur + [blk])
            ib = [i for i, g in enumerate(fa['groups']) if g == blk[0]]
            sdb = self.bootstrap(fa, nb=100, seed=5)[:, ib].std(0)
            out['pruned'].append(dict(group=blk[0], names=[fa['names'][i] for i in ib], coef=[float(fa['beta'][i]) for i in ib],
                                      coef_sd=sdb.tolist(), dbic_addback=L.bic(fa['sse'], fa['n'], fa['k'], Lc) - L.bic(self.fit(cur, rows=fa['ok'])['sse'], fa['n'], fa['k'] - len(ib), Lc)))
        out['full_core_names'] = [b[0] for b in cur_full]
        # 4. final fit, drop-one diagnostics, components
        self._final_blocks = cur
        ff = self.fit(cur, cv=True)
        out['final'] = dict(cv=ff['cv'], sse=ff['sse'], n=ff['n'], k=ff['k'], bic=L.bic(ff['sse'], ff['n'], ff['k'], Lc),
                            names=ff['names'], groups=ff['groups'], beta=ff['beta'].tolist(),
                            r2=1 - np.var(ff['r']) / np.var(S.y[ff['ok']]))
        drop = {}
        for blk in cur:
            g = blk[0]
            if g == 'base':
                continue
            fd = self.fit([b for b in cur if b[0] != g], rows=ff['ok'], cv=True)
            drop[g] = dict(dcv_rel=(ff['cv'] - fd['cv']) / fd['cv'], dbic=out['final']['bic'] - L.bic(fd['sse'], fd['n'], fd['k'], Lc),
                           partial_r2=(fd['sse'] - ff['sse']) / np.sum((S.y[ff['ok']] - S.y[ff['ok']].mean()) ** 2 * (1 if S.w is None else S.w[ff['ok']])))
        out['drop_one'] = drop
        # 5. bootstrap (moving blocks of residuals; clusters for repeaters)
        B = self.bootstrap(ff)
        out['boot_sd'] = B.std(0).tolist()
        # 6. component series and derived quantities (with bootstrap SDs)
        out['components'] = self.derived(ff, B, tau)
        # 7. quake-model tests (static strain vs PGV vs free big three), long-term alternatives
        out['quake_tests'] = self.quake_tests(cur, tau, Lc)
        out['long_tests'] = self.long_tests(cur, Lc)
        # 8. placebos: fake big quake (season-matched Dec 5 and random dates), year-shifted drivers
        out['placebo_quake'] = self.placebo_quake(cur, tau)
        out['placebo_env'] = self.placebo_env(cur, sel, best_param)
        # 9. quake-specific recovery time-constant profiles (others kept at the shared tau)
        out['tau_profiles'] = {}
        env = [b for b in cur if b[0] not in ('base', 'big3', 'dose', 'trend', 'ramp21')]
        for q in L.BIG3:
            prof = {}
            for tq in (5, 15, 50, 150, 500, 1500, 5000):
                blks = self.core_blocks(tau, tau_by={q: tq}) + env
                fa = self.fit(blks)
                if f'trans_{q}' in fa['names']:
                    prof[tq] = dict(sse=fa['sse'], trans=float(fa['beta'][fa['names'].index(f'trans_{q}')]),
                                    step=float(fa['beta'][fa['names'].index(f'step_{q}')]) if f'step_{q}' in fa['names'] else np.nan)
            out['tau_profiles'][q] = prof
        return out

    # ---------------------------------------------------------------- pieces
    def bootstrap(self, ff, nb=NBOOT, seed=0):
        rng = np.random.default_rng(seed + abs(hash((self.S.sta, self.S.chan))) % 1000)
        A = ff['A'][ff['ok']]; r = ff['r']; fit = A @ ff['beta']
        w = None if self.S.w is None else self.S.w[ff['ok']]
        n = len(r)
        out = np.zeros((nb, len(ff['beta'])))
        if self.pairs:
            cl = self.S.cluster[ff['ok']]; uc = np.unique(cl)
            for b in range(nb):
                pick = rng.choice(uc, len(uc))
                ii = np.concatenate([np.where(cl == c)[0] for c in pick])
                out[b], _ = L.huber_lsq(A[ii], (fit + r)[ii], None if w is None else w[ii], nit=2)
            return out
        blk = {'back': 180, 'centre': 180, 'point': 180}[self.S.sm[0]]
        step = max(1, int(np.median(np.diff(np.sort(self.S.idx)))))
        bl = max(3, blk // step)
        for b in range(nb):
            starts = rng.integers(0, n, int(np.ceil(n / bl)))
            ii = (starts[:, None] + np.arange(bl)[None, :]).ravel()[:n] % n
            out[b], _ = L.huber_lsq(A, fit + r[ii], w, nit=2)
        return out

    def derived(self, ff, B, tau):
        """Per group: component daily series (smoothed), var fraction, annual amplitude/phase, 2021->2025 change."""
        S = self.S
        names, groups, beta = ff['names'], np.array(ff['groups']), ff['beta']
        # rebuild the daily (smoothed) design for the selected columns
        Ad = self._daily_design(ff)
        ok = ff['ok']
        yv = S.y[ok]
        vy = np.var(yv)
        i21 = (L.DAYS >= '2021-01-01') & (L.DAYS < '2021-12-19')
        i25 = (L.DAYS >= '2025-01-01') & (L.DAYS < '2026-01-01')
        span = np.zeros(L.ND, bool); span[S.idx.min():S.idx.max() + 1] = True
        w = 2 * np.pi * L.TY
        H = np.column_stack([np.ones(L.ND), np.sin(w), np.cos(w)])
        comp = {}
        ug = [g for g in dict.fromkeys(groups) if g != 'base']
        composite = {'QUAKES': [g for g in ug if g in ('big3', 'dose')],
                     'LONGTERM': [g for g in ug if g in ('trend', 'ramp21', 'postdil')],
                     'ENV': [g for g in ug if g not in ('big3', 'dose', 'trend', 'ramp21', 'postdil')],
                     'TOTAL': ug}
        for g in ug + [k for k, v in composite.items() if v]:
            ii = np.where(np.isin(groups, composite.get(g, [g])))[0]
            cs = Ad[:, ii] @ beta[ii]
            csb = Ad[:, ii] @ B[:, ii].T                  # (ND, nb)
            samp = ff['A'][ok][:, ii] @ beta[ii]
            d = dict(group=g, cols=[names[i] for i in ii], coef=beta[ii].tolist(),
                     coef_sd=B[:, ii].std(0).tolist(), var_frac=float(np.var(samp) / vy))
            sp = span & np.isfinite(cs)
            if sp.sum() > 730:
                c, *_ = np.linalg.lstsq(H[sp], cs[sp], rcond=None)
                cb, *_ = np.linalg.lstsq(H[sp], np.nan_to_num(csb[sp]), rcond=None)
                d['annual_amp'] = float(np.hypot(c[1], c[2])); d['annual_amp_sd'] = float(np.std(np.hypot(cb[1], cb[2])))
                d['annual_peak_doy'] = float((np.degrees(np.arctan2(c[1], c[2])) % 360) / 360 * 365.25)
            if (span & i21).sum() > 200 and (span & i25).sum() > 200:
                d['chg_2125'] = float(np.nanmean(cs[span & i25]) - np.nanmean(cs[span & i21]))
                d['chg_2125_sd'] = float(np.std(np.nanmean(csb[span & i25], 0) - np.nanmean(csb[span & i21], 0)))
            comp[g] = (d, cs.astype(np.float32))
        # observed change and residual annual cycle
        tt = L.DAYS[S.idx]
        o21 = ok & (tt >= '2021-01-01') & (tt < '2021-12-19'); o25 = ok & (tt >= '2025-01-01') & (tt < '2026-01-01')
        extra = {}
        if o21.sum() > 20 and o25.sum() > 20:
            # data corrected for the fitted instrument-break offsets (offsets are not part of any cause)
            io = [i for i, n in enumerate(names) if n.startswith('offset_')]
            ycor = S.y - (ff['A'][:, io] @ beta[io] if io else 0.0)
            extra['obs_chg_2125'] = float(ycor[o25].mean() - ycor[o21].mean())
            extra['obs_chg_2125_raw'] = float(S.y[o25].mean() - S.y[o21].mean())
        Hs = H[S.idx[ok]]
        c, *_ = np.linalg.lstsq(Hs, yv, rcond=None)
        extra['obs_annual_amp'] = float(np.hypot(c[1], c[2]))
        extra['obs_annual_peak_doy'] = float((np.degrees(np.arctan2(c[1], c[2])) % 360) / 360 * 365.25)
        c, *_ = np.linalg.lstsq(Hs, ff['r'], rcond=None)
        extra['resid_annual_amp'] = float(np.hypot(c[1], c[2]))
        # placebo for multi-year changes: (mean of year Y+4) - (mean of year Y) of the final residual, all Y with data
        ry = pd.Series(ff['r'], index=tt[ok].year).groupby(level=0).agg(['mean', 'size'])
        ry = ry[ry['size'] >= (100 if S.sm[1] <= 10 and S.pair_i is None else 6)]['mean']
        d4 = [ry[y + 4] - ry[y] for y in ry.index if y + 4 in ry.index]
        extra['resid_4yr_rms'] = float(np.sqrt(np.mean(np.square(d4)))) if len(d4) >= 3 else np.nan
        extra['resid_4yr_n'] = len(d4)
        # big-three coseismic drop (perm + trans at t=0+) and 1-yr level, with bootstrap SD
        for q in L.BIG3:
            if f'step_{q}' in names:
                i1 = names.index(f'step_{q}'); i2 = names.index(f'trans_{q}') if f'trans_{q}' in names else None
                v0 = beta[i1] + (beta[i2] if i2 is not None else 0); vb = B[:, i1] + (B[:, i2] if i2 is not None else 0)
                v1 = beta[i1] + (beta[i2] * np.exp(-365 / tau) if i2 is not None else 0)
                vb1 = B[:, i1] + (B[:, i2] * np.exp(-365 / tau) if i2 is not None else 0)
                extra[f'drop0_{q}'] = float(v0); extra[f'drop0_{q}_sd'] = float(vb.std())
                extra[f'level1yr_{q}'] = float(v1); extra[f'level1yr_{q}_sd'] = float(vb1.std())
        extra['fitted_daily'] = (Ad @ beta).astype(np.float32)
        return dict(groups=comp, extra=extra)

    def _daily_design(self, ff):
        """Daily smoothed design with the same columns as ff (rebuilt from blocks of the final model)."""
        # Recompute from blocks stored on the instance during run(): easiest is to re-assemble with a dummy series
        S = self.S
        idx_save, pair_save = S.idx, S.pair_i
        S.idx = np.arange(L.ND); S.pair_i = None
        try:
            A, names, groups = L.assemble(S, self._final_blocks)
        finally:
            S.idx, S.pair_i = idx_save, pair_save
        env = np.array([g not in ('base', 'big3', 'dose', 'trend', 'ramp21', 'postdil') for g in groups])
        A[:, env] = np.where(np.isfinite(A[:, env]), A[:, env], 0.0)
        cols = [names.index(n) for n in ff['names']]
        return A[:, cols]

    def quake_tests(self, cur, tau, Lc):
        env = [b for b in cur if b[0] not in ('base', 'big3', 'dose', 'trend', 'ramp21')]
        res = {}
        for mode in ('free3', 'free3_only', 'pgv_all', 'strain_all'):
            fa = self.fit(self.core_blocks(tau, mode) + env, cv=True)
            res[mode] = dict(cv=fa['cv'], bic=L.bic(fa['sse'], fa['n'], fa['k'], Lc), sse=fa['sse'], n=fa['n'], k=fa['k'],
                             coef={n: float(b) for n, b in zip(fa['names'], fa['beta']) if n.startswith(('dose', 'step', 'trans'))})
        return res

    def long_tests(self, cur, Lc):
        env = [b for b in cur if b[0] not in ('trend', 'ramp21')]
        res = {}
        lc = self.longc
        variants = {'trend+ramp21': [('trend', ['trend'], lc['trend']), ('ramp21', ['ramp21'], lc['ramp21'])],
                    'trend_only': [('trend', ['trend'], lc['trend'])], 'none': []}
        if 'gDilPost' in self.D and np.isfinite(self.D['gDilPost']).all():
            variants['trend+gnss_postdil'] = [('trend', ['trend'], lc['trend']), ('postdil', ['gnss_postdil'], L.anom(self.D['gDilPost'])[:, None])]
        for k, v in variants.items():
            fa = self.fit(env + v, cv=True)
            res[k] = dict(cv=fa['cv'], bic=L.bic(fa['sse'], fa['n'], fa['k'], Lc),
                          coef={n: float(b) for n, b in zip(fa['names'], fa['beta']) if n in ('trend', 'ramp21', 'gnss_postdil')})
        return res

    def placebo_quake(self, cur, tau, nrand=30, seed=1):
        """Fake quake (perm step + transient) added to the final model; season-matched (Dec 5 of other years)
        and random dates >= 90 d from the big three.  Returns fake drop0 (= perm + trans) values."""
        S = self.S
        i = np.arange(L.ND)
        t0, t1 = S.idx.min() + 60, S.idx.max() - 60
        big = [L.big3_day(n) for n in L.BIG3]
        dates = []
        for y in range(2008, 2027):
            d = int(L.dayidx([pd.Timestamp(f'{y}-12-05')])[0])
            if t0 < d < t1 and all(abs(d - b) > 90 for b in big):
                dates.append(('season', d))
        rng = np.random.default_rng(7)
        cand = [int(d) for d in np.unique(S.idx) if t0 < d < t1 and all(abs(d - b) > 90 for b in big)]
        for d in rng.choice(cand, min(nrand, len(cand)), replace=False) if cand else []:
            dates.append(('random', int(d)))
        out = []
        for kind, d in dates:
            if not (self.supported('fake_step', tau, d) and self.supported('fake_trans', tau, d)):
                continue
            X = np.column_stack([(i >= d).astype(float), np.where(i >= d, np.exp(-(i - d) / tau), 0.0)])
            fa = self.fit(cur + [('fake', ['fake_step', 'fake_trans'], X)])
            if 'fake_step' in fa['names']:
                v = fa['beta'][fa['names'].index('fake_step')] + (fa['beta'][fa['names'].index('fake_trans')] if 'fake_trans' in fa['names'] else 0)
                v1 = fa['beta'][fa['names'].index('fake_step')]
                out.append(dict(kind=kind, day=str(L.DAYS[d].date()), drop0=float(v), step=float(v1)))
        return out

    def placebo_env(self, cur, sel, best_param, shifts=(-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6)):
        """For each selected environmental group with a raw driver: rebuild it from the driver circularly shifted
        by whole years (keeps the seasonal shape, scrambles interannual timing); SSE of the final model with the
        shifted driver vs the real one."""
        res = {}
        f_real = self.fit(cur)
        raw = {'hyd_exp': 'P', 'hyd_diff': 'P', 'rain_lags': 'P', 'hyd_exp_era5': 'P_era5', 'thermal': 'T',
               'soilm': None, 'streamflow': 'logQ', 'gnss_up': 'gU', 'gnss_dil': 'gDil', 'bstrain': 'bStr', 'pressure': 'Ps', 'sealevel': 'SL'}
        for g in sel:
            if g not in raw:
                continue
            ss = []
            for k in shifts:
                n = int(round(k * 365.25))
                D2 = dict(self.D)
                keys = [raw[g]] if raw[g] else [kk for kk in ('SM0_7', 'SM28_100', 'SM100_255', 'SM0_100') if kk in self.D]
                for kk in keys:
                    D2[kk] = np.roll(self.D[kk], n)
                C2 = L.candidate_groups(self.S.sta, None, D2) if g != 'micro' else None
                if C2 is None or g not in C2:
                    continue
                lab, X = C2[g][best_param[g]]
                blk = (g, [f'{g}[{k}]' for k in range(X.shape[1])] if X.shape[1] > 1 else [g], X)
                cur2 = [b if b[0] != g else blk for b in cur]
                ss.append(self.fit(cur2)['sse'])
            if ss:
                res[g] = dict(sse_real=f_real['sse'], sse_shift=ss, p=float((np.sum(np.array(ss) <= f_real['sse']) + 1) / (len(ss) + 1)))
        return res


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


def stage_fit(a):
    which = a.datasets.split(',')
    SS = all_series(which)
    if a.only:
        SS = [s for s in SS if s.sta in a.only.split(',') and (not a.chan or s.chan in a.chan.split(','))]
    print('series', len(SS), flush=True)
    # group by station so drivers are cached per worker
    SS.sort(key=lambda s: (s.sta, s.dataset, s.chan))
    tag = a.tag
    done = 0
    if a.workers <= 1:
        results = []
        for S in SS:
            r = run_one(S); results.append(r); done += 1
            print(done, S.dataset, S.sta, S.chan, r.get('selected'), 'ERR' if 'error' in r else f"{r.get('runtime_s', 0):.0f}s", flush=True)
            if 'error' in r:
                print(r['error'], flush=True)
    else:
        import multiprocessing as mp
        ctx = mp.get_context('fork')
        results = []
        with ctx.Pool(a.workers) as pool:
            for r in pool.imap_unordered(run_one, SS, chunksize=1):
                results.append(r); done += 1
                print(done, r['dataset'], r['sta'], r['chan'], r.get('selected'), 'ERR' if 'error' in r else f"{r.get('runtime_s', 0):.0f}s", flush=True)
                if 'error' in r:
                    print(r['error'], flush=True)
    with open(f'{RES}/results_{tag}.pkl', 'wb') as f:
        pickle.dump(results, f)
    print('saved', f'{RES}/results_{tag}.pkl', flush=True)


# ================================================================================================ collect
UNITS = {'const': '%', 'offset': '%', 'step': '%', 'trans': '% (at t_q, decays exp(-t/tau))',
         'dose_pgv_perm': '% per cm/s PGV (BSSA14)', 'dose_pgv_trans': '% per cm/s PGV (BSSA14), transient',
         'dose_strain_perm': '% per nanostrain static vol. strain', 'dose_strain_trans': '% per nanostrain, transient',
         'trend': '%/yr', 'ramp21': '%/yr after 2021-07-01', 'hyd_exp': '% per mm (exp-drained rain storage)',
         'hyd_exp_era5': '% per mm (ERA5 rain, exp-drained)', 'hyd_diff': '% per mm (diffused rain anomaly)',
         'rain_lags': '% per mm (rain in lag window)', 'soilm': '% per m3/m3', 'streamflow': '% per log10(cfs)',
         'thermal': '% per degC', 'pressure': '% per hPa', 'sealevel': '% per cm', 'gnss_up': '% per mm uplift',
         'gnss_dil': '% per nanostrain (GNSS areal)', 'bstrain': '% per nanostrain (borehole areal, uncalibrated)',
         'micro': '% per decade of power', 'annual': '%', 'postdil': '% per nanostrain'}


def unit_of(name):
    for k in sorted(UNITS, key=len, reverse=True):
        if name.startswith(k):
            return UNITS[k]
    return ''


def stage_collect(a):
    import glob
    cc = L.station_coords()
    R = []
    for fn in sorted(glob.glob(f'{RES}/results_{a.tag}*.pkl')):
        R += pickle.load(open(fn, 'rb'))
    rows, scores, tests, qt, ts = [], [], [], [], []
    for r in R:
        if 'error' in r:
            print('ERROR', r['dataset'], r['sta'], r['chan']); continue
        key = dict(dataset=r['dataset'], sta=r['sta'], chan_band=r['chan'])
        loc = dict(lat=cc.lat.get(r['sta'], np.nan), lon=cc.lon.get(r['sta'], np.nan), sensor_depth_m=cc.depth_m.get(r['sta'], np.nan),
                   t0=r['t0'], t1=r['t1'], n_samples=r['n'])
        f = r['final']; comps = r['components']['groups']; ex = r['components']['extra']; drop = r['drop_one']
        # placebo spread of a fake quake (season-matched + random) -> honest SD for coseismic items
        pq = pd.DataFrame(r['placebo_quake'])
        sd_pl = float(np.sqrt(np.mean(pq.drop0 ** 2))) if len(pq) >= 5 else np.nan
        sd_pl_step = float(np.sqrt(np.mean(pq.step ** 2))) if len(pq) >= 5 else np.nan
        sd_pl_season = float(np.sqrt(np.mean(pq[pq.kind == 'season'].drop0 ** 2))) if (pq.kind == 'season').sum() >= 3 else np.nan
        for nm, b, sd, g in zip(f['names'], f['beta'], r['boot_sd'], f['groups']):
            d = comps.get(g, ({}, None))[0]
            par = r['selected_params'].get(g, f"tau_q={r['tau_q']}d" if g in ('big3', 'dose') else '')
            sdh = sd
            if nm.startswith('step_'):
                sdh = max(sd, sd_pl_step) if np.isfinite(sd_pl_step) else sd
            rows.append(dict(**key, component=nm, kind='coef', group=g, amp=b, sd=sdh, sd_boot=sd, units=unit_of(nm),
                             param=par, var_frac=d.get('var_frac', np.nan), partial_r2=drop.get(g, {}).get('partial_r2', np.nan),
                             dbic_drop=drop.get(g, {}).get('dbic', np.nan), dcv_drop=drop.get(g, {}).get('dcv_rel', np.nan),
                             p_yearshift=r['placebo_env'].get(g, {}).get('p', np.nan), **loc))
        for pr in r.get('pruned', []):
            for nm, b, sd in zip(pr['names'], pr['coef'], pr['coef_sd']):
                rows.append(dict(**key, component=nm, kind='coef_pruned', group=pr['group'], amp=b, sd=sd, sd_boot=sd, units=unit_of(nm),
                                 param='not selected (BIC); amplitude from add-back fit', dbic_drop=-pr['dbic_addback'], **loc))
        for g, (d, cs) in comps.items():
            for k in ('annual_amp', 'chg_2125'):
                if k in d:
                    sdk = d.get(f'{k}_sd', np.nan)
                    p4 = ex.get('resid_4yr_rms', np.nan) if k == 'chg_2125' else np.nan
                    rows.append(dict(**key, component=f'{k}[{g}]', kind='derived', group=g, amp=d[k],
                                     sd=np.nanmax([sdk, p4]) if np.isfinite(p4) else sdk, sd_placebo_4yr=p4,
                                     sd_boot=d.get(f'{k}_sd', np.nan), units='%', param=f"peak_doy={d.get('annual_peak_doy', np.nan):.0f}" if k == 'annual_amp' else '2025 mean - 2021 mean (pre-Petrolia)',
                                     var_frac=d['var_frac'], partial_r2=drop.get(g, {}).get('partial_r2', np.nan), **loc))
            ts.append((r['dataset'], r['sta'], r['chan'], g, cs))
        ts.append((r['dataset'], r['sta'], r['chan'], 'FIT', ex['fitted_daily']))
        for q in L.BIG3:
            if f'drop0_{q}' in ex:
                sdb = ex[f'drop0_{q}_sd']; sdh = max(sdb, sd_pl) if np.isfinite(sd_pl) else sdb
                prof = r['tau_profiles'].get(q, {})
                tbest = min(prof, key=lambda t: prof[t]['sse']) if prof else np.nan
                rows.append(dict(**key, component=f'drop0_{q}', kind='derived', group='big3', amp=ex[f'drop0_{q}'], sd=sdh, sd_boot=sdb,
                                 units='% (step+transient at t_q)', param=f"tau_q={r['tau_q']}d; tau_{q}_best={tbest}d",
                                 z_placebo=ex[f'drop0_{q}'] / sd_pl if np.isfinite(sd_pl) else np.nan, sd_placebo=sd_pl,
                                 sd_placebo_season=sd_pl_season, **loc))
                rows.append(dict(**key, component=f'level1yr_{q}', kind='derived', group='big3', amp=ex[f'level1yr_{q}'],
                                 sd=ex[f'level1yr_{q}_sd'], sd_boot=ex[f'level1yr_{q}_sd'], units='% one year after t_q', **loc))
        for k in ('obs_chg_2125', 'obs_annual_amp', 'resid_annual_amp', 'resid_4yr_rms'):
            if k in ex:
                rows.append(dict(**key, component=k, kind='observed', group='data', amp=ex[k], sd=np.nan, units='%', **loc))
        # single-driver tests (gain of each candidate on its own, added to the core model)
        for t in r['tests']:
            tests.append(dict(**key, **{k: (v if not isinstance(v, list) else json.dumps([round(x, 8) for x in v])) for k, v in t.items()}, **loc))
            if t['group'] in ('gnss_dil', 'bstrain') and len(t['coef']) == 1:
                eta, etsd = t['coef'][0] * 1e7, t['coef_sd'][0] * 1e7      # (% per nanostrain) -> d(v)/v per unit strain
                rows.append(dict(**key, component=f'eta_{t["group"]}', kind='test', group=t['group'], amp=eta, sd=etsd,
                                 units='dimensionless (dv/v per unit areal strain); single-driver model', param=t['param'],
                                 dbic_drop=-t['dbic'], dcv_drop=-t['dcv_rel'], **loc))
        sc = dict(**key, **loc, tau_q=r['tau_q'], L_decorr=r['L_decorr'], core_cv=r['core']['cv'], core_bic=r['core']['bic'],
                  final_cv=f['cv'], final_bic=f['bic'], final_r2=f['r2'], selected=';'.join(r['selected']),
                  pruned_core=';'.join(p['group'] for p in r.get('pruned', [])),
                  selected_params=json.dumps(r['selected_params']), sd_placebo_quake=sd_pl, sd_placebo_season=sd_pl_season)
        for m, v in r['quake_tests'].items():
            sc[f'q_{m}_cv'] = v['cv']; sc[f'q_{m}_bic'] = v['bic']
            for kk, vv in v['coef'].items():
                if kk.startswith('dose'):
                    sc[f'q_{m}_{kk}'] = vv
        for m, v in r['long_tests'].items():
            sc[f'lt_{m}_cv'] = v['cv']; sc[f'lt_{m}_bic'] = v['bic']
        for g, v in r['placebo_env'].items():
            sc[f'p_yearshift_{g}'] = v['p']
        scores.append(sc)
    C = pd.DataFrame(rows)
    C.to_csv(f'{OUT}/components.csv', index=False, float_format='%.6g')
    pd.DataFrame(scores).to_csv(f'{OUT}/model_scores.csv', index=False, float_format='%.6g')
    pd.DataFrame(tests).to_csv(f'{OUT}/candidate_tests.csv', index=False, float_format='%.6g')
    # component time series (daily, smoothed like the data) -> bulk parquet
    T = pd.DataFrame({f'{d}|{s}|{c}|{g}': v for d, s, c, g, v in ts}, index=L.DAYS)
    T.to_parquet(f'{RES}/component_series_{a.tag}.parquet')
    print('components', C.shape, 'scores', len(scores), 'tests', len(tests))


# ================================================================================================ out-of-sample
def stage_oos(a):
    import glob
    R = {}
    for fn in sorted(glob.glob(f'{RES}/results_{a.tag}*.pkl')):
        for r in pickle.load(open(fn, 'rb')):
            if 'error' not in r:
                R[(r['dataset'], r['sta'], r['chan'])] = r
    SS = all_series(a.datasets.split(','))
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
        Fi = Fitter(S)
        tau = r['tau_q']
        env = []
        for g in r['selected']:
            j = [k for k, (lab, X) in enumerate(Fi.cands[g]) if lab == r['selected_params'][g]][0]
            env.append(Fi.env_block(g, j)[0])
        for variant, blocks in (('pgv_dose+env', Fi.core_blocks(tau, 'pgv_all', long=('trend',)) + env),
                                ('env_only', [Fi.core_blocks(tau)[0], ('trend', ['trend'], Fi.longc['trend'])] + env),
                                ('pgv_dose_only', Fi.core_blocks(tau, 'pgv_all', long=('trend',)))):
            A, names, groups, ok = Fi.matrix(blocks)
            tr = ok & pre
            beta, _ = L.huber_lsq(A[tr], S.y[tr], None if S.w is None else S.w[tr])
            pred = A @ beta
            po = ok & ~pre
            tt = L.DAYS[S.idx]
            o21 = ok & (tt >= '2021-01-01') & (tt < '2021-07-01') ; o25 = ok & (tt >= '2025-01-01') & (tt < '2026-01-01')
            m7 = L.big3_day('M7')
            p7 = ok & (S.idx > m7) & (S.idx <= m7 + 30); b7 = ok & (S.idx < m7) & (S.idx >= m7 - 30)
            res = S.y[po] - pred[po]
            row = dict(dataset=S.dataset, sta=S.sta, chan_band=S.chan, variant=variant, n_pre=int(tr.sum()), n_post=int(po.sum()),
                       rms_post=float(np.sqrt(np.mean(res ** 2))),
                       rms_post_const=float(np.sqrt(np.mean((S.y[po] - np.median(S.y[tr])) ** 2))),
                       rms_post_offset=float(np.sqrt(np.mean((res - res.mean()) ** 2))),
                       r_post_detrended=float(np.corrcoef(S.y[po] - S.y[po].mean(), pred[po] - pred[po].mean())[0, 1]),
                       dose_coef=json.dumps({n: round(float(b), 5) for n, b in zip(names, beta) if n.startswith('dose')}))
            if o21.sum() > 10 and o25.sum() > 10:
                row['obs_chg'] = float(S.y[o25].mean() - S.y[o21].mean()); row['pred_chg'] = float(pred[o25].mean() - pred[o21].mean())
            if p7.sum() > 3 and b7.sum() > 3:
                row['obs_m7_30d'] = float(S.y[p7].mean() - S.y[b7].mean()); row['pred_m7_30d'] = float(pred[p7].mean() - pred[b7].mean())
            rows.append(row)
        print(S.dataset, S.sta, S.chan, flush=True)
    O = pd.DataFrame(rows)
    O.to_csv(f'{OUT}/out_of_sample.csv', index=False, float_format='%.6g')
    print(O.groupby('variant')[['rms_post', 'rms_post_const', 'rms_post_offset', 'r_post_detrended']].median())


# ================================================================================================ afterslip time constant
def stage_afterslip(a):
    """GNSS M7 postseismic: per near station, pre-M7 trajectory (2022-12-25..2024-12-04: offset, velocity,
    annual+semiannual, equipment steps) extrapolated; post-M7 residual (CME removed) fitted with
    c + A*log(1+t/tau) and c + A*(1-exp(-t/tau)) for a tau grid (A per station/component, tau shared);
    station bootstrap for the tau interval."""
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
    taus = np.array([2, 5, 10, 20, 40, 80, 160, 320, 640, 1280])
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
        res[form] = dict(tau_best=float(taus[np.argmin(p)]), sse=p.tolist(), tau_boot_q=np.percentile(bs, [5, 50, 95]).tolist())
    res['n_stations'] = len(stas); res['taus'] = taus.tolist()
    json.dump(res, open(f'{OUT}/afterslip_tau.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'taus'}, indent=1)[:1500])


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
