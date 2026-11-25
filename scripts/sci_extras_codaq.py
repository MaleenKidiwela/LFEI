#!/usr/bin/env python3
"""CODA Q (coda decay rate) THROUGH TIME from LFE stacks -- analysis.

Two observables (both from data already on disk, extracted by sci_extras_codaq_extract{,_coh}.py into
/data/wsd02/maleen_data/extras/codaq/):
  coh : COHERENT coda of linear LFE stacks (isolated +-5 s, RMS-normalised events). Coherent energy
        E = |full stack|^2 - |even-odd half difference|^2 * 4 N0 N1 / (N0+N1)^2 (removes noise + incoherent coda
        with the right 1/N scaling), averaged in 1-s blocks.  High SNR; decay = attenuation x coherence loss.
  env : INCOHERENT energy average of single isolated LFEs (no other detection in [-20, +40] s), minus the
        pre-origin noise floor. Physically the classic Qc, but single-LFE coda SNR is only ~0.05-0.5.
Decay model: E(t) ~ t^-2 exp(-2 pi fc t / Qc), t = lapse from the LFE origin; Q^-1 from a two-window (early/late half) ratio of E t^2 (robust at low SNR),
over a FIXED per-channel lapse window S+3 s .. last contiguous block whose all-time SNR is high enough that a typical
60-d window still has per-block SNR >~ 0.5 (coh) / reference SNR >= 0.1 (env), capped at S+20 s, >= 2 s long.
Per channel-band: change d = (Q^-1(post) - Q^-1(pre)) / Q^-1(all-time reference) (stable denominator). Channel variance k/N from
year-to-year differences (includes real variability; used only for weights).  Group means (all / PB borehole /
surface; per band and both bands) are inverse-variance weighted with a cap (10x median weight).
Tests: steps (pre | post) at 2010-01-10 M6.5, 2014-03-10 M6.8, Petrolia, Ferndale, the M7 restart (post = M7+18 d ..
+78 d, i.e. the first 60 d of activity), M7 1 yr (pre 365 d | M7+78 .. +365 d); placebo = half-month bin edges whose
windows are clear of all 5 quakes by >= 30 d (same window lengths); injection of dQ^-1/Q^-1 = +10 % into the post
window -> recovery; 2-sigma detection limit = 2 x placebo SD / recovery.  Long term: yearly series, OLS trend with
scatter-based SE, post-2021-12-20 vs the 7 yr before, with split-date placebos 2012-2016.
Outputs: data/sci/extras/codaq/{mode}_{channels,yearly,quarterly,steps,longterm}.csv, {mode}_placebo_steps.csv.gz,
         summary_{mode}.json
Run: nice -n 10 python scripts/sci_extras_codaq.py --mode coh   (and --mode env)
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse
import glob
import json

import h5py
import numpy as np
import pandas as pd

ROOT = '/home/seismic/LFE'
BIG = '/data/wsd02/maleen_data/extras/codaq'
OUT = f'{ROOT}/data/sci/extras/codaq'
FC = np.array([np.sqrt(2 * 4), np.sqrt(4 * 8)])
BANDN = ['2-4Hz', '4-8Hz']
C0, C1 = 3.0, 20.0
MINWIN = 2.0
MINN = 20
WCAP = 10.0
DAY = 86400.0
QUAKES = {'eureka2010': '2010-01-10T00:27:39', 'mendo2014': '2014-03-10T05:18:13',
          'petrolia': '2021-12-20T20:10', 'ferndale': '2022-12-20T10:34', 'm7': '2024-12-05T18:44'}
EXCL = {'PB.B045..EH1': [('2000-01-01', '2011-01-01')],
        'PB.B933': [('2017-01-01', '2021-11-01'), ('2024-08-01', '2030-01-01')],
        'PB.B935': [('2024-05-01', '2030-01-01')],
        'PB.B047': [('2000-01-01', '2009-03-01')],
        'NC.KMPB': [('2000-01-01', '2010-11-03')],
        'NC.KMR': [('2000-01-01', '2009-09-17'), ('2025-09-17', '2030-01-01')],
        'NC.KMR..HHE': [('2024-09-01', '2025-09-17')], 'NC.KMR..HHN': [('2024-09-01', '2025-09-17')],
        'BK.BRIC': [('2022-12-01', '2024-05-22')],
        'BK.RVIT': [('2022-12-20T10:34', '2024-06-21')],
        'BK.BJES': [('2022-12-20T10:34', '2024-05-23')]}


def ep(s):
    return (pd.Timestamp(s) - pd.Timestamp('1970-01-01')).total_seconds()


QT = {k: ep(v) for k, v in QUAKES.items()}


def excl_mask(cid, t):
    m = np.zeros(len(t), bool)
    for k, wins in EXCL.items():
        if cid == k or cid.startswith(k + '.'):
            for a, b in wins:
                m |= (t >= ep(a)) & (t < ep(b))
    return m


def loglin_q(E, lap, fc, mask):
    """two-window decay estimator (robust at low SNR): the contiguous fit window (mask) is split into an early and a
    late half; E1, E2 = mean energies, t1, t2 = mean lapse times;  Q^-1 = -ln(E2 t2^2 / (E1 t1^2)) / (2 pi fc (t2-t1)).
    E (I, nw), lap (I, nw), fc (I,), mask (I, nw) -> Q^-1 (I,); NaN if a half-window energy <= 0."""
    nm = mask.sum(-1)
    k = np.arange(mask.shape[-1])[None]
    h1 = mask & (k < np.ceil(nm / 2)[:, None])
    h2 = mask & ~h1
    with np.errstate(invalid='ignore', divide='ignore'):
        E1 = np.where(h1, E, 0).sum(-1) / h1.sum(-1)
        E2 = np.where(h2, E, 0).sum(-1) / h2.sum(-1)
        t1 = np.where(h1, lap, 0).sum(-1) / h1.sum(-1)
        t2 = np.where(h2, lap, 0).sum(-1) / h2.sum(-1)
        q = -np.log((E2 * t2 ** 2) / (E1 * t1 ** 2)) / (2 * np.pi * fc * (t2 - t1))
    return np.where((E1 > 0) & (E2 > 0) & (nm >= 2) & np.isfinite(q), q, np.nan)


# ================================================================================================ backends
class Coh:
    BLK = 5                      # 5 samples at 5 Hz = 1 s

    def __init__(self, prefix='coh'):
        C, NH, meta = [], [], []
        self.edges = None
        for f in sorted(glob.glob(f'{BIG}/{prefix}_fam*.h5')):
            F = int(os.path.basename(f)[len(prefix) + 4:-3])
            with h5py.File(f, 'r') as h:
                E = h['edges'][:]
                self.edges = E if self.edges is None else self.edges
                assert np.allclose(E, self.edges)
                tS = h['tS'][:]
                o = float(h.attrs['origin'])
                S = h['S'][:]
                chans = h['chan'][:].astype(str)
                tb = 0.5 * (E[:-1] + E[1:])
                nblk = len(tS) // self.BLK
                tcen = tS[:nblk * self.BLK].reshape(nblk, self.BLK).mean(1)
                kb = np.where((tcen >= C0 + 0.4 - 1e-6) & (tcen <= C1 - 0.4 + 1e-6))[0]
                s0 = kb[0] * self.BLK
                s1 = (kb[-1] + 1) * self.BLK
                self.nblk = len(kb)
                CS = h['CS'][:, :, :, :, s0:s1]
                nh = h['NH'][:]
                for k, cid in enumerate(chans):
                    ex = excl_mask(cid, tb)
                    if nh[k][:, ~ex].sum() < 200:
                        continue
                    cs = CS[k].copy()
                    cs[:, :, ex] = 0
                    n_ = nh[k].copy()
                    n_[:, ex] = 0
                    C.append(cs)
                    NH.append(n_)
                    meta.append(dict(F=F, cid=cid, S=S[k], origin=o, lap=S[k] + tcen[kb] - o,
                                     t_active_d=float(((n_.sum(0) > 0).sum()) * 15.2)))
        self.C = np.stack(C)            # (nch, band, half, nb, ns)
        self.NH = np.stack(NH).astype(float)  # (nch, half, nb)
        self.meta = meta
        print('coh: channels', len(meta), 'array GB', self.C.nbytes / 1e9, flush=True)

    def energy(self, a, b, inject=None, ch=None, band=None):
        """coherent energy blocks (nch, band, nblk) and N (nch,) for bins [a, b)."""
        W = self.C[:, :, :, a:b].sum(3)                 # (nch, band, half, ns)
        N = self.NH[:, :, a:b].sum(2)                   # (nch, half)
        N0, N1 = N[:, 0][:, None, None], N[:, 1][:, None, None]
        with np.errstate(invalid='ignore', divide='ignore'):
            full = (W[:, :, 0] + W[:, :, 1]) / (N0 + N1)
            diff = (W[:, :, 0] / N0 - W[:, :, 1] / N1) / 2
            Ec = np.abs(full) ** 2 - np.abs(diff) ** 2 * 4 * N0 * N1 / (N0 + N1) ** 2
            Ei = np.abs(diff) ** 2 * 4 * N0 * N1 / (N0 + N1) ** 2
        nb_ = Ec.shape[-1] // self.BLK
        Ec = Ec[..., :nb_ * self.BLK].reshape(*Ec.shape[:2], nb_, self.BLK).mean(-1)
        Ei = Ei[..., :nb_ * self.BLK].reshape(*Ei.shape[:2], nb_, self.BLK).mean(-1)
        return Ec, Ei, N.sum(1)

    def prepare(self):
        nb = len(self.edges) - 1
        Ec, Ei, N = self.energy(0, nb)
        items = []
        for i, m in enumerate(self.meta):
            for j in range(2):
                snr = Ec[i, j] / Ei[i, j]
                thr = 0.5 * m["t_active_d"] / 60.0   # a typical 60-d window keeps per-block SNR >~ 0.5
                good = snr >= thr
                if not good[0]:
                    continue
                last = np.argmin(good) if not good.all() else len(good)
                mask = np.zeros(len(good), bool)
                mask[:last] = True
                if mask.sum() < MINWIN:
                    continue
                q = loglin_q(Ec[i, j][None], m['lap'][None], np.array([FC[j]]), mask[None])[0]
                if not np.isfinite(q) or q <= 0:
                    continue
                items.append(dict(i=i, band=j, mask=mask, qref=q, snr0=float(snr[0]), wlen=float(mask.sum()),
                                  F=m['F'], cid=m['cid'], n_total=float(N[i])))
        self.items = items
        self.ii = np.array([it['i'] for it in items])
        self.jj = np.array([it['band'] for it in items])
        self.mask = np.stack([it['mask'] for it in items])
        self.lap = np.stack([self.meta[it['i']]['lap'] for it in items])
        self.fc = FC[self.jj]
        self.qref = np.array([it['qref'] for it in items])
        return items

    def measure(self, a, b, inject=None):
        Ec, Ei, N = self.energy(a, b)
        E = Ec[self.ii, self.jj]
        if inject is not None:
            E = E * np.exp(-2 * np.pi * self.fc[:, None] * inject * self.qref[:, None] * (self.lap - self.lap[:, :1]))
        q = loglin_q(E, self.lap, self.fc, self.mask)
        n = N[self.ii]
        q[n < MINN] = np.nan
        return q, n


class Env:
    def __init__(self):
        recs = []
        self.edges = None
        for f in sorted(glob.glob(f'{BIG}/env_fam*.h5')):
            F = int(os.path.basename(f)[7:-3])
            with h5py.File(f, 'r') as h:
                E = h['edges'][:]
                self.edges = E if self.edges is None else self.edges
                assert np.allclose(E, self.edges)
                o = float(h.attrs['origin'])
                t = -15 + 0.5 * np.arange(120) + 0.25
                nz = t < o - 1.0
                if nz.sum() < 4:
                    continue
                ES, N = h['ESUM'][:], h['N'][:]
                chans = h['chan'][:].astype(str)
                S = h['S'][:]
                tb = 0.5 * (E[:-1] + E[1:])
                # 1-s blocks over S+3 .. S+20
                for k, cid in enumerate(chans):
                    i0 = int(np.ceil((S[k] + C0 + 15) / 0.5 - 1e-6))
                    nbk = int((min(S[k] + C1, 45.0) - (-15 + 0.5 * i0)) // 1.0)
                    if nbk < MINWIN:
                        continue
                    ex = excl_mask(cid, tb)
                    n_ = N[k].copy()
                    n_[ex] = 0
                    if n_.sum() < 200:
                        continue
                    es = ES[k][:, :, i0:i0 + 2 * nbk].copy()
                    es[:, ex] = 0
                    es = es.reshape(2, len(tb), nbk, 2).mean(-1)
                    wn = ES[k][:, :, nz].mean(-1)
                    wn[:, ex] = 0
                    lap = (-15 + 0.5 * i0) + 0.5 + np.arange(nbk) - o
                    recs.append(dict(F=F, cid=cid, W=es, Wn=wn, N=n_, lap=lap))
        nw = max(len(r['lap']) for r in recs)
        R = len(recs)
        nb = len(self.edges) - 1
        self.CW = np.zeros((R, 2, nb + 1, nw))
        self.CWn = np.zeros((R, 2, nb + 1))
        self.CN = np.zeros((R, nb + 1))
        self.lapr = np.ones((R, nw))
        self.valid = np.zeros((R, nw), bool)
        for i, r in enumerate(recs):
            m = len(r['lap'])
            self.CW[i, :, 1:, :m] = np.cumsum(r['W'], 1)
            self.CWn[i, :, 1:] = np.cumsum(r['Wn'], 1)
            self.CN[i, 1:] = np.cumsum(r['N'])
            self.lapr[i, :m] = r['lap']
            self.lapr[i, m:] = r['lap'][-1] + np.arange(1, nw - m + 1)
            self.valid[i, :m] = True
        self.meta = [dict(F=r['F'], cid=r['cid']) for r in recs]
        print('env: channels', R, flush=True)

    def energy(self, a, b):
        N = self.CN[:, b] - self.CN[:, a]
        with np.errstate(invalid='ignore', divide='ignore'):
            e = (self.CW[:, :, b] - self.CW[:, :, a]) / N[:, None, None]
            n = (self.CWn[:, :, b] - self.CWn[:, :, a]) / N[:, None]
        return e - n[..., None], n, N

    def prepare(self):
        nb = len(self.edges) - 1
        s, n, N = self.energy(0, nb)
        items = []
        for i, m in enumerate(self.meta):
            for j in range(2):
                snr = s[i, j] / n[i, j]
                good = (snr >= 0.1) & self.valid[i]
                if not good[0]:
                    continue
                last = np.argmin(good) if not good.all() else len(good)
                mask = np.zeros(len(good), bool)
                mask[:last] = True
                if mask.sum() < MINWIN:
                    continue
                q = loglin_q(s[i, j][None], self.lapr[i][None], np.array([FC[j]]), mask[None])[0]
                if not np.isfinite(q) or q <= 0:
                    continue
                items.append(dict(i=i, band=j, mask=mask, qref=q, snr0=float(snr[0]), wlen=float(mask.sum()),
                                  F=m['F'], cid=m['cid'], n_total=float(N[i])))
        self.items = items
        self.ii = np.array([it['i'] for it in items])
        self.jj = np.array([it['band'] for it in items])
        self.mask = np.stack([it['mask'] for it in items])
        self.lap = self.lapr[self.ii]
        self.fc = FC[self.jj]
        self.qref = np.array([it['qref'] for it in items])
        return items

    def measure(self, a, b, inject=None):
        s, n, N = self.energy(a, b)
        E = s[self.ii, self.jj]
        if inject is not None:
            E = E * np.exp(-2 * np.pi * self.fc[:, None] * inject * self.qref[:, None] * (self.lap - self.lap[:, :1]))
        q = loglin_q(E, self.lap, self.fc, self.mask)
        nn = N[self.ii]
        q[nn < MINN] = np.nan
        return q, nn


# ================================================================================================ analysis
def combine(d, var, groups, cap=WCAP):
    res = {}
    for g, m in groups.items():
        ok = m & np.isfinite(d) & np.isfinite(var) & (var > 0)
        if ok.sum() < 3:
            res[g] = (np.nan, np.nan, int(ok.sum()))
            continue
        dd, vv = d[ok], var[ok]
        # robust: winsorise each item at +-4 of its own sd around the group median
        sdv = np.sqrt(vv)
        med = np.median(dd)
        dd = np.clip(dd, med - 4 * sdv, med + 4 * sdv)
        w = 1 / vv
        w = np.minimum(w, cap * np.median(w))
        mu = np.sum(w * dd) / w.sum()
        se = np.sqrt(np.sum(w ** 2 * vv)) / w.sum()
        res[g] = (float(mu), float(se), int(ok.sum()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['coh', 'env', 'coh137'], default='coh')
    A = ap.parse_args()
    mode = A.mode
    os.makedirs(OUT, exist_ok=True)
    B = Env() if mode == 'env' else Coh(mode)
    edges = B.edges
    nb = len(edges) - 1
    items = B.prepare()
    I = len(items)
    band = B.jj
    bh = np.array([it['cid'].startswith('PB.') for it in items])
    qref = B.qref
    print(mode, 'items', I, flush=True)

    def rng(t0, t1):
        a = int(np.searchsorted(edges, t0 - 1, side='left'))
        b = int(np.searchsorted(edges, t1 + 1, side='right')) - 1
        return a, max(a, b)

    # ---- noise constant from year-to-year differences: var(q) ~ k / N
    years = np.arange(2009, 2027)
    yq = np.full((I, len(years)), np.nan)
    yn = np.zeros((I, len(years)))
    for a_, y in enumerate(years):
        q, n = B.measure(*rng(ep(f'{y}-01-01'), ep(f'{y + 1}-01-01')))
        yq[:, a_], yn[:, a_] = q, n
    dq = np.diff(yq, axis=1) ** 2
    dn = 1 / np.maximum(yn[:, 1:], 1) + 1 / np.maximum(yn[:, :-1], 1)
    kq = np.nanmedian(dq / dn, 1) / 0.455
    sd1000 = np.sqrt(kq / 1000) / qref
    good = np.isfinite(kq) & (kq > 0) & (sd1000 < 1.0)
    groups = {}
    for j in range(2):
        groups[f'all_{BANDN[j]}'] = (band == j) & good
        groups[f'borehole_{BANDN[j]}'] = (band == j) & bh & good
        groups[f'surface_{BANDN[j]}'] = (band == j) & ~bh & good
    groups['all_both'] = good.copy()
    groups['borehole_both'] = bh & good
    groups['surface_both'] = ~bh & good
    CH = pd.DataFrame(dict(F=[it['F'] for it in items], cid=[it['cid'] for it in items],
                           band=[BANDN[b] for b in band], borehole=bh, Qinv_ref=qref, Qc_ref=1 / qref,
                           snr_first_block=[it['snr0'] for it in items], fit_window_s=[it['wlen'] for it in items],
                           n_total=[it['n_total'] for it in items], k_noise=kq, sd_rel_per_1000ev=sd1000, used=good))
    CH.to_csv(f'{OUT}/{mode}_channels.csv', index=False, float_format='%.6g')
    print('usable items', good.sum(), flush=True)

    def rel(q, n):
        return q / qref - 1, kq / np.maximum(n, 1) / qref ** 2

    rows = []
    for a_, y in enumerate(years):
        d, var = rel(yq[:, a_], yn[:, a_])
        for g, (mu, se, nc) in combine(d, var, groups).items():
            rows.append(dict(year=int(y), group=g, dq_rel=mu, se_formal=se, nchan=nc))
    Y = pd.DataFrame(rows)
    Y.to_csv(f'{OUT}/{mode}_yearly.csv', index=False, float_format='%.6g')
    # per-station yearly series (both bands) -> is a long-term change common to all stations?
    stas = np.array(['.'.join(it['cid'].split('.')[:2]) for it in items])
    sg = {s_: (stas == s_) & good for s_ in sorted(set(stas))}
    rows = []
    for a_, y in enumerate(years):
        d, var = rel(yq[:, a_], yn[:, a_])
        for g, (mu, se, nc) in combine(d, var, sg).items():
            rows.append(dict(year=int(y), station=g, dq_rel=mu, se_formal=se, nchan=nc))
    pd.DataFrame(rows).to_csv(f'{OUT}/{mode}_station_yearly.csv', index=False, float_format='%.6g')

    rows = []
    qe = pd.date_range('2008-07-01', '2027-01-01', freq='QS')
    for a_ in range(len(qe) - 1):
        q, n = B.measure(*rng(ep(qe[a_]), ep(qe[a_ + 1])))
        d, var = rel(q, n)
        for g, (mu, se, nc) in combine(d, var, groups).items():
            rows.append(dict(t0=str(qe[a_].date()), group=g, dq_rel=mu, se_formal=se, nchan=nc))
    pd.DataFrame(rows).to_csv(f'{OUT}/{mode}_quarterly.csv', index=False, float_format='%.6g')

    def step(t, L, d0, Lp, inject=None):
        q0, n0 = B.measure(*rng(t - L * DAY, t))
        q1, n1 = B.measure(*rng(t + d0 * DAY, t + (d0 + Lp) * DAY), inject=inject)
        d = (q1 - q0) / qref
        var = kq / np.maximum(n0, 1) / qref ** 2 + kq / np.maximum(n1, 1) / qref ** 2
        return combine(d, var, groups)

    defs = {'eureka2010': (QT['eureka2010'], 60, 0, 60), 'mendo2014': (QT['mendo2014'], 60, 0, 60),
            'petrolia': (QT['petrolia'], 60, 0, 60), 'ferndale': (QT['ferndale'], 60, 0, 60),
            'm7_restart60': (QT['m7'], 60, 18, 60), 'm7_1yr': (QT['m7'], 365, 78, 287)}
    real = {k: step(*v) for k, v in defs.items()}
    inj = {k: step(*v, inject=0.10) for k, v in defs.items()}
    cand = edges[(edges > ep('2008-11-01')) & (edges < ep('2026-06-01'))]
    qts = np.array(list(QT.values()))
    rows, prow = [], []
    for k, (t, L, d0, Lp) in defs.items():
        lo, hi = L + 30, d0 + Lp + 30
        ok = np.array([np.all((c - qts > lo * DAY) | (qts - c > hi * DAY)) for c in cand])
        dates = cand[ok]
        P = [(c, step(c, L, d0, Lp)) for c in dates]
        Pi = [(c, step(c, L, d0, Lp, inject=0.10)) for c in dates[::8]]
        print(k, 'placebos', len(dates), flush=True)
        for g in groups:
            pv = np.array([p[g][0] for _, p in P])
            fin = np.isfinite(pv)
            for (c, p) in P:
                prow.append(dict(test=k, group=g, date=str(pd.Timestamp(c, unit='s').date()), dq_rel=p[g][0],
                                 se_formal=p[g][1]))
            mu, se, nc = real[k][g]
            pv = pv[fin]
            if len(pv) < 10 or not np.isfinite(mu):
                rows.append(dict(test=k, group=g, dq_rel=mu, se_formal=se, nchan=nc, n_placebo=len(pv)))
                continue
            med = np.median(pv)
            psd = 1.4826 * np.median(np.abs(pv - med))
            rec = (inj[k][g][0] - mu) / 0.10
            pr = np.array([(pi[g][0] - p[g][0]) / 0.10 for (c, pi), (c2, p) in
                           zip(Pi, [x for x in P if x[0] in set(dates[::8])])])
            pr = pr[np.isfinite(pr)]
            rec_pl = float(np.median(pr)) if len(pr) else np.nan
            rr = rec_pl if np.isfinite(rec_pl) and rec_pl > 0.1 else 1.0
            rows.append(dict(test=k, group=g, dq_rel=mu, se_formal=se, nchan=nc, n_placebo=len(pv),
                             placebo_median=med, placebo_sd=psd, z_placebo=(mu - med) / psd,
                             pct_placebo=100 * np.mean(pv < mu), placebo_sd_over_formal=psd / np.nanmedian(
                                 [p[g][1] for _, p in P]),
                             inject_recovery_event=rec, inject_recovery_placebo=rec_pl,
                             detect_2sigma_rel=2 * psd / rr,
                             detect_2sigma_dQinv=2 * psd / rr * float(np.median(qref[groups[g]]))))
    ST = pd.DataFrame(rows)
    ST.to_csv(f'{OUT}/{mode}_steps.csv', index=False, float_format='%.6g')
    pd.DataFrame(prow).to_csv(f'{OUT}/{mode}_placebo_steps.csv.gz', index=False, float_format='%.6g')

    # ---- long term
    lt = []
    for g in groups:
        y = Y[(Y.group == g)].dropna(subset=['dq_rel'])
        y = y[y.year >= 2009]
        if len(y) < 8:
            continue
        x = y.year.values + 0.5
        Am = np.c_[np.ones(len(x)), x - 2017]
        cf = np.linalg.lstsq(Am, y.dq_rel.values, rcond=None)[0]
        r = y.dq_rel.values - Am @ cf
        s2 = np.sum(r ** 2) / (len(x) - 2)
        se_tr = np.sqrt(s2 * np.linalg.inv(Am.T @ Am)[1, 1])
        lt.append(dict(group=g, kind='trend_per_yr_2009_2026', value=cf[1], se_scatter=se_tr, z=cf[1] / se_tr,
                       yearly_rms=float(np.sqrt(s2))))

    def contrast(tc, Lb, La):
        q0, n0 = B.measure(*rng(tc - Lb * 365.25 * DAY, tc))
        q1, n1 = B.measure(*rng(tc, tc + La * 365.25 * DAY))
        d = (q1 - q0) / qref
        var = kq / np.maximum(n0, 1) / qref ** 2 + kq / np.maximum(n1, 1) / qref ** 2
        return combine(d, var, groups)
    La = (ep('2026-11-01') - QT['petrolia']) / (365.25 * DAY)
    creal = contrast(QT['petrolia'], 7.0, La)
    cpl = []
    for c in pd.date_range('2012-01-01', '2016-12-01', freq='2MS'):
        tc = ep(c)
        Lb = min(7.0, (tc - ep('2009-01-01')) / (365.25 * DAY))
        cpl.append(contrast(tc, Lb, La))
    for g in groups:
        mu, se, nc = creal[g]
        pv = np.array([p[g][0] for p in cpl])
        pv = pv[np.isfinite(pv)]
        if len(pv) < 5:
            continue
        sd = max(1.4826 * np.median(np.abs(pv - np.median(pv))), np.std(pv, ddof=1))
        lt.append(dict(group=g, kind='post2021_vs_7yr_before', value=mu, se_formal=se, n_placebo=len(pv),
                       placebo_median=float(np.median(pv)), placebo_sd=sd, pct_placebo=100 * np.mean(pv < mu),
                       z=(mu - np.median(pv)) / sd, detect_2sigma_rel=2 * sd))
    LT = pd.DataFrame(lt)
    LT.to_csv(f'{OUT}/{mode}_longterm.csv', index=False, float_format='%.6g')

    gsel = ['all_both', 'borehole_both', 'surface_both', 'all_2-4Hz', 'all_4-8Hz']
    summ = dict(mode=mode, n_items=I, n_used=int(good.sum()),
                n_used_band={BANDN[j]: int(groups[f'all_{BANDN[j]}'].sum()) for j in range(2)},
                n_used_borehole=int(groups['borehole_both'].sum()),
                n_stations=int(len(set('.'.join(items[i]['cid'].split('.')[:2]) for i in np.where(good)[0]))),
                n_families=int(len(set(items[i]['F'] for i in np.where(good)[0]))),
                Qc_ref_median={BANDN[j]: float(np.median(1 / qref[groups[f'all_{BANDN[j]}']])) for j in range(2)},
                Qc_ref_iqr={BANDN[j]: [float(x) for x in np.percentile(1 / qref[groups[f'all_{BANDN[j]}']], [25, 75])]
                            if groups[f'all_{BANDN[j]}'].sum() else None for j in range(2)},
                Qc_ref_median_bh_surf={f'{BANDN[j]}_{c}': float(np.median(1 / qref[groups[f'{c}_{BANDN[j]}']]))
                                       for j in range(2) for c in ('borehole', 'surface')},
                fit_window_s_median=float(np.median(CH.fit_window_s[good])),
                steps=ST[ST.group.isin(gsel)].replace({np.nan: None}).to_dict('records'),
                longterm=LT[LT.group.isin(gsel)].replace({np.nan: None}).to_dict('records'))
    json.dump(summ, open(f'{OUT}/summary_{mode}.json', 'w'), indent=1, default=float)
    pd.set_option('display.width', 250)
    print(ST[ST.group.isin(gsel)][['test', 'group', 'dq_rel', 'se_formal', 'nchan', 'placebo_sd', 'z_placebo',
                                    'pct_placebo', 'inject_recovery_placebo', 'detect_2sigma_rel']].to_string())
    print(LT[LT.group.isin(gsel)].to_string())


if __name__ == '__main__':
    main()
