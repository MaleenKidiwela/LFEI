#!/usr/bin/env python3
"""LFE secondary phases -- SEARCH stage: are there source-side secondary phases inside the long-term LFE stacks?

Input: OUT/stacks/fam{F}.h5 (sci_lfe_phases_extract.py; long-term mean of isolated LFEs, 1-15 Hz), geometry and
1D delay tables (sci_lfe_phases_common.py; Shelly family locations, Bloch 2023 t and m).

Per family-station (fs): band-pass (2-8 Hz default), envelope (Hilbert, 0.1 s smoothing), horizontal channels
combined in energy (H), vertical (Z).  Direct S refined on the H envelope (max within +-0.5 s of the pick), direct P
on the Z envelope (+-0.4 s).  Observable: detrended log envelope  e(t) = log10 env(t) - running median (2 s) of it
(scale-free, removes the coda decay), and the plain normalised envelope env/env(S) (as the 2026-11-24 scan).

Phase hypotheses (delays from exact layered ray tracing, sci_lfe_phases_common):
  below-source reflector at absolute depth Z_F = m_F + dz (Bloch slab Moho + common offset dz):
     SmS (H, after S), PmS (H, after S), PmP (Z, after P), SmP (Z, after P)
     -> delay depends on each family's height H_F = m_F + dz - z_F  (1.5-7.4 km for dz = 0): the discriminator
  above-source converter at Z_F = t_F + dz (Bloch top of the LVL): Sp (Z, between P and S), Ps (H, between P and S)
For each phase and dz: A(dz) = mean over fs of e at the predicted time (fs excluded where the predicted time is within
0.3 s of a direct phase or outside the window).  Nulls (1000 draws): (i) PERMUTE family heights among families
(keeps the station geometry and the delay distribution, breaks the height scaling); (ii) for receiver-side
phases: a CONSTANT-delay stack.  z = (A - mean_null)/sd_null; global significance from the max over dz of the
permuted maxima (look-elsewhere).  Required for a robust phase: z_global > 3 in all fs, and the same sign/peak
in boreholes (PB) and surface stations, in the two event halves, and odd/even families.
Also: per-station record sections (families sorted by H_m) with the predicted SmS/PmP curves.
Outputs OUT/search_scan.csv, OUT/search_summary.json, figures/sci_lfe_phases_{scan,sections}.png
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse
import json
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import butter, hilbert, sosfiltfilt

ROOT = '/home/seismic/LFE'
sys.path.insert(0, f'{ROOT}/scripts')
from sci_lfe_phases_common import OUT, FIG, HGRID, geometry  # noqa: E402

SR = 50.0
TAU = np.arange(-15.0, 20.0, 0.02)     # lag grid relative to direct S / P
DZ_BELOW = np.round(np.arange(-6.0, 10.01, 0.1), 2)
DZ_ABOVE = np.round(np.arange(-8.0, 8.01, 0.1), 2)
PHASES = {   # name: (component, reference phase, table key, side)
    'SmS': ('H', 'S', 'SmS', 'below'), 'PmS': ('H', 'S', 'PmS', 'below'),
    'PmP': ('Z', 'P', 'PmP', 'below'), 'SmP': ('Z', 'P', 'SmP', 'below'),
    'Sp_above': ('Z', 'P', 'Sp_above', 'above'), 'Ps_above': ('H', 'P', 'Ps_above', 'above')}
NPERM = 1000
TMPL = {'SmS': 'S', 'PmS': 'S', 'PmP': 'P', 'SmP': 'P', 'Sp_above': 'S', 'Ps_above': 'S'}
EXCL = 0.3


def envelope(x, sos):
    y = sosfiltfilt(sos, x)
    e = np.abs(hilbert(y))
    return uniform_filter1d(e, 5)


def inject_trace(x, t, tref, tau, amp):
    """add amp x (copy of the direct-phase window [tref-0.5, tref+1.5] s, cosine-tapered) delayed by tau"""
    w = np.clip(np.minimum((t - (tref - 0.5)) / 0.2, ((tref + 1.5) - t) / 0.2), 0, 1)
    w = 0.5 - 0.5 * np.cos(np.pi * w)
    return x + amp * np.interp(t - tau, t, x * w, left=0.0, right=0.0)


def load_fs(band=(2.0, 8.0), stack='all', snr_min=15.0, fams=range(1, 28), inject=None):
    """-> list of dict per family-station.  inject = (phase, amp, dz): waveform-level synthetic secondary phase
    (copy of the direct reference phase, amplitude ratio amp, delay from the delay table for reflector offset dz)
    added to every channel of the component that phase is observed on."""
    sos = butter(4, band, btype='band', fs=SR, output='sos')
    G, tabs = geometry()
    gi = {(r.F, r.sta): i for i, r in enumerate(G.itertuples())}
    out = []
    for F in fams:
        fn = f'{OUT}/stacks/fam{F}.h5'
        if not os.path.exists(fn):
            continue
        with h5py.File(fn, 'r') as h:
            names = [n.decode() for n in h['names'][:]]
            chan = [c.decode() for c in h['chan'][:]]
            si = names.index(stack)
            X = h['stack'][si]
            cnt = h['count'][si]
            snr, Sp, Pp = h['snr'][:], h['S'][:], h['P'][:]
            t0 = h.attrs['t0']
        t = t0 + np.arange(X.shape[1]) / SR
        bysta = {}
        for k, c in enumerate(chan):
            if cnt[k] < 50 or snr[k] < snr_min:
                continue
            sta = '.'.join(c.split('.')[:2])
            comp = 'Z' if c[-1] == 'Z' else 'H'
            bysta.setdefault(sta, {}).setdefault(comp, []).append(k)
        for sta, d in bysta.items():
            if (F, sta) not in gi:
                continue
            r = dict(F=F, sta=sta, bh=sta.startswith('PB.'), gi=gi[(F, sta)])
            k0 = (d.get('H') or d.get('Z'))[0]
            S, P = Sp[k0], Pp[k0]
            noise_m = (t > -5) & (t < -1.5)
            inj_tau = None
            if inject is not None:
                ph, amp, dzi = inject
                c_, ref_, key_, side_ = PHASES[ph]
                g_ = G.iloc[gi[(F, sta)]]
                Hh = (g_.z_m + dzi - g_.zs) if side_ == 'below' else (g_.zs - (g_.z_t + dzi))
                tr_ = S if ref_ == 'S' else P
                if 0.2 < Hh < HGRID[-1] and np.isfinite(tr_):
                    inj_tau = (c_, tr_, float(np.interp(Hh, HGRID, tabs[key_][gi[(F, sta)]])), amp)
            for comp, ks in d.items():
                if comp == 'H' and len(ks) < 2:
                    continue
                xs = [X[k].astype(float) for k in ks]
                if inj_tau is not None and inj_tau[0] == comp and np.isfinite(inj_tau[2]):
                    xs = [inject_trace(x_, t, inj_tau[1], inj_tau[2], inj_tau[3]) for x_ in xs]
                env = np.sqrt(np.sum([envelope(x_, sos) ** 2 for x_ in xs], 0))
                r[f'x{comp}'] = [sosfiltfilt(sos, x_) for x_ in xs]
                r[f'env{comp}'] = env
                r[f'noise{comp}'] = float(np.median(env[noise_m])) if noise_m.any() else np.nan
            if 'envH' not in r and 'envZ' not in r:
                continue
            eS = r.get('envH', r.get('envZ'))
            if np.isfinite(S) and (np.abs(t - S) <= 0.5).any():
                m = np.abs(t - S) <= 0.5
                S = t[m][np.argmax(eS[m])]
            else:
                S = np.nan
            if np.isfinite(P) and 'envZ' in r and (np.abs(t - P) <= 0.4).any():
                m = np.abs(t - P) <= 0.4
                P = t[m][np.argmax(r['envZ'][m])]
            else:
                P = np.nan
            r.update(S=S, P=P, t=t)
            # detrended log-envelope on the TAU grids (rel. S and rel. P)
            for comp in ('H', 'Z'):
                if f'env{comp}' not in r:
                    continue
                L = np.log10(r[f'env{comp}'] + 1e-12)
                bg = median_filter(L, size=int(2.0 * SR) + 1, mode='nearest')
                e = L - bg
                ok = r[f'env{comp}'] > 3 * r[f'noise{comp}']      # only above-noise samples count
                e = np.where(ok, e, np.nan)
                a = r[f'env{comp}'] / r[f'env{comp}'][np.argmin(np.abs(t - S))] if np.isfinite(S) else None
                for ref, tr in (('S', S), ('P', P)):
                    if not np.isfinite(tr):
                        continue
                    r[f'e{comp}_{ref}'] = np.interp(TAU, t - tr, e, left=np.nan, right=np.nan)
                    if a is not None:
                        r[f'a{comp}_{ref}'] = np.interp(TAU, t - tr, a, left=np.nan, right=np.nan)
            # matched-filter (self-correlation) detection function: correlate each channel with its own tapered
            # direct-phase window (template), sum over channels, envelope, normalise by the zero-lag value.
            # Value at time t = relative amplitude of a copy of the template arriving at t (phase-shift tolerant).
            for comp in ('H', 'Z'):
                if f'x{comp}' not in r:
                    continue
                for tm, ttm in (('S', S), ('P', P)):
                    if not np.isfinite(ttm) or (comp == 'H' and tm == 'P'):
                        continue
                    w = np.clip(np.minimum((t - (ttm - 0.3)) / 0.1, ((ttm + 0.9) - t) / 0.1), 0, 1)
                    w = 0.5 - 0.5 * np.cos(np.pi * w)
                    C = 0.0
                    nrm = 0.0
                    for x_ in r[f'x{comp}']:
                        tp = x_ * w
                        n_ = len(x_)
                        cc_ = np.fft.irfft(np.fft.rfft(x_, 2 * n_) * np.conj(np.fft.rfft(tp, 2 * n_)), 2 * n_)
                        C = C + np.r_[cc_[n_:], cc_[:n_]]            # lags -n .. n-1 samples
                        nrm += np.sum(tp * tp)
                    # C[k] = sum x(t_k + u) tp(u): lag k samples; template starts at its own position -> arrival time
                    # of a copy = ttm + k/SR
                    ce = np.abs(hilbert(C)) / nrm
                    tarr = ttm + (np.arange(len(C)) - len(C) // 2) / SR
                    for ref, tr in (('S', S), ('P', P)):
                        if np.isfinite(tr):
                            r[f'c{comp}{tm}_{ref}'] = np.interp(TAU, tarr - tr, ce, left=np.nan, right=np.nan)
                del r[f'x{comp}']
            out.append(r)
    return out, G


class Prep:
    """vectorised per-fs arrays for one phase."""

    def __init__(self, recs, G, tabs, phase, kind='e'):
        comp, ref, key, side = PHASES[phase]
        self.side = side
        gi = np.array([r['gi'] for r in recs])
        g = G.iloc[gi]
        self.F = g.F.values
        self.zm, self.zt, self.zs = g.z_m.values, g.z_t.values, g.zs.values
        self.T = tabs[key][gi]
        n = len(recs)
        self.E = np.full((n, len(TAU)), np.nan)
        self.ref_t = np.full(n, np.nan)
        self.S = np.array([r['S'] for r in recs])
        self.P = np.array([r['P'] for r in recs])
        for i, r in enumerate(recs):
            arr = r.get(f'{kind}{comp}_{ref}') if kind != 'c' else r.get(f'c{comp}{TMPL[phase]}_{ref}')
            if arr is not None:
                self.E[i] = arr
                self.ref_t[i] = r['S'] if ref == 'S' else r['P']

    def delays(self, dz, zs=None):
        zs = self.zs if zs is None else zs
        H = (self.zm + dz - zs) if self.side == 'below' else (zs - (self.zt + dz))
        x = H / (HGRID[1] - HGRID[0])
        i = np.clip(np.floor(x).astype(int), 0, len(HGRID) - 2)
        w = x - i
        r = np.arange(len(H))
        d = self.T[r, i] * (1 - w) + self.T[r, i + 1] * w
        d[(H < 0.2) | (H > HGRID[-1])] = np.nan
        return d

    def sample(self, d):
        ta = self.ref_t + d
        bad = ~np.isfinite(d) | (np.abs(d) < EXCL) | (np.abs(ta - self.S) < EXCL) | (np.abs(ta - self.P) < EXCL)
        x = (d - TAU[0]) / (TAU[1] - TAU[0])
        i = np.clip(np.floor(np.nan_to_num(x)).astype(int), 0, len(TAU) - 2)
        w = np.nan_to_num(x) - i
        r = np.arange(len(d))
        v = self.E[r, i] * (1 - w) + self.E[r, i + 1] * w
        v[bad] = np.nan
        return v


def scan(recs, G, tabs, phase, subset_mask, rng, kind='e', nperm=NPERM, prep=None, null='fam', nmin=60):
    """null 'fam': permute source depths among families (breaks the height scaling);
    'sta': within each family give every fs the delay table of a random other station of that family (keeps the
    height scaling, breaks the station moveout)."""
    comp, ref, key, side = PHASES[phase]
    DZ = DZ_BELOW if side == 'below' else DZ_ABOVE
    P = prep or Prep(recs, G, tabs, phase, kind)
    fams = np.unique(P.F)
    zs_f = {F: P.zs[P.F == F][0] for F in fams}

    def stat(v):
        v = v[subset_mask]
        return np.nanmean(v) if np.isfinite(v).sum() >= nmin else np.nan
    real = np.array([stat(P.sample(P.delays(dz))) for dz in DZ])
    nn = np.array([np.isfinite(P.sample(P.delays(dz))[subset_mask]).sum() for dz in DZ])
    perm = np.full((nperm, len(DZ)), np.nan)
    T0 = P.T
    for b in range(nperm):
        if null in ('fam', 'famclass'):
            if null == 'fam':
                pf = dict(zip(fams, rng.permutation([zs_f[F] for F in fams])))
            else:   # permute only within depth classes (shallow < 26.5 km, deep >= 26.5 km): removes any
                    # shallow-vs-deep difference in coda level, keeps the within-class height scaling
                pf = {}
                for cls in (True, False):
                    ff = [F for F in fams if (zs_f[F] < 26.5) == cls]
                    pf.update(dict(zip(ff, rng.permutation([zs_f[F] for F in ff]))))
            zs = np.array([pf[F] for F in P.F])
            perm[b] = [stat(P.sample(P.delays(dz, zs))) for dz in DZ]
        else:
            idx = np.arange(len(P.F))
            for F in fams:
                ii = np.where(P.F == F)[0]
                idx[ii] = rng.permutation(ii)
            P.T = T0[idx]
            perm[b] = [stat(P.sample(P.delays(dz))) for dz in DZ]
    P.T = T0
    return DZ, real, nn, perm


def summarise(DZ, real, perm):
    mu, sd = np.nanmean(perm, 0), np.nanstd(perm, 0) + 1e-12
    z = (real - mu) / sd
    zp = (perm - mu) / sd
    gmax = np.nanmax(zp, 1)
    k = int(np.nanargmax(z)) if np.isfinite(z).any() else 0
    pg = float((np.sum(gmax >= z[k]) + 1) / (len(gmax) + 1))
    return mu, sd, z, k, pg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--band', type=float, nargs=2, default=[2.0, 8.0])
    ap.add_argument('--nperm', type=int, default=300)
    ap.add_argument('--tag', default='')
    ap.add_argument('--phases', nargs='*', default=list(PHASES))
    ap.add_argument('--kinds', nargs='*', default=['e', 'a'])
    ap.add_argument('--nulls', nargs='*', default=['fam', 'sta'])
    ap.add_argument('--subsets', nargs='*', default=['all', 'boreholes', 'surface', 'fam_odd', 'fam_even',
                                                     'half0', 'half1'])
    ap.add_argument('--inject', nargs=3, default=None, help='phase amp dz')
    a = ap.parse_args()
    G, tabs = geometry()
    rng = np.random.default_rng(7)
    rows, summ = [], {}
    inj = (a.inject[0], float(a.inject[1]), float(a.inject[2])) if a.inject else None
    stacks = {'all': load_fs(tuple(a.band), 'all', inject=inj)[0]}
    for h in ('half0', 'half1'):
        if h in a.subsets:
            stacks[h] = load_fs(tuple(a.band), h, inject=inj)[0]
    recs = stacks['all']
    print(len(recs), 'family-stations', flush=True)
    bh = np.array([r['bh'] for r in recs])
    odd = np.array([r['F'] % 2 == 1 for r in recs])
    subsets = {'all': np.ones(len(recs), bool), 'boreholes': bh, 'surface': ~bh, 'fam_odd': odd, 'fam_even': ~odd}
    for phase in a.phases:
        for kind in a.kinds:
            for sname in a.subsets:
                msk = subsets.get(sname)
                rr = recs if msk is not None else stacks[sname]
                mm = msk if msk is not None else np.ones(len(rr), bool)
                prep = Prep(rr, G, tabs, phase, kind)
                for null in a.nulls:
                    DZ, real, nn, perm = scan(rr, G, tabs, phase, mm, rng, kind, a.nperm, prep, null)
                    mu, sd, z, k, pg = summarise(DZ, real, perm)
                    for j in range(len(DZ)):
                        rows.append(dict(phase=phase, kind=kind, subset=sname, null=null, dz=DZ[j], A=real[j], n=nn[j],
                                         null_mean=mu[j], null_sd=sd[j], z=z[j]))
                    j0 = int(np.argmin(np.abs(DZ - (inj[2] if inj else 0.0))))
                    summ[f'{phase}|{kind}|{sname}|{null}'] = dict(dz_best=float(DZ[k]), z_best=float(z[k]),
                                                                  p_global=pg, n=int(nn[k]), z_at_ref=float(z[j0]))
                    print(phase, kind, sname, null, f'best dz {DZ[k]:+.1f} km z {z[k]:.2f} p_global {pg:.3f} '
                          f'n {nn[k]} | z at dz={DZ[j0]:+.1f}: {z[j0]:.2f}', flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/search_scan{a.tag}.csv', index=False)
    json.dump(summ, open(f'{OUT}/search_summary{a.tag}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
