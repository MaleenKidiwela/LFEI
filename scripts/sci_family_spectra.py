#!/usr/bin/env python
"""Relative source spectra BETWEEN LFE families (path/site cancelled by same-channel ratios of
neighbouring families).

Inputs (read only): data/sci/source_spectra/raw/fam{F}.h5 (per-event S-window and pre-event noise
multitaper band power, stored as log10(P / sc) with sc = per-(family, channel, band) median noise power,
12 log bands 1-15 Hz), data/waveforms/fam{F}.h5 (only the first 64-event chunk per channel, to recover
the absolute scale sc, which the raw file does not store), data/locations/fam{F}_3d*.{json,csv}
(S travel times), data/families/shelly_family_locations.csv, catalogs, patch/moment tables.

Stages
  sc     : recover sc(F, c, band) exactly: sc = pn / 10**NP for the first-chunk events (pn recomputed with
           the identical multitaper code) -> data/sci/family_spectra/stage1/sc_fam{F}.npz
  accum  : per family, channel, calendar year: sums of absolute S and noise power over CLEAN events
           (no other catalog event in -25..+20 s), after an earthquake/glitch screen, for replicate
           weight vectors: 0 = all, 1/2 = even/odd halves, 3.. = Poisson event bootstrap
           -> stage1/acc_fam{F}.npz
  solve  : pair x channel log amplitude ratios (year-matched weights, noise corrected), joint LSQ for
           per-family relative band spectra x_F(b) and indices (LO, HFI, slope), path terms, distance
           thresholds, event + station bootstrap, validation splits, Brune fits, scaling tests
           -> data/sci/family_spectra/*.csv, summary.json
  figs   : figures/sci_family_spectra_*.png
Run: nice python scripts/sci_family_spectra.py sc --procs 8 ; ... accum --procs 8 ; ... solve ; ... figs
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse
import json
import sys
import time
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import h5py
import numpy as np
import pandas as pd
from scipy.signal.windows import dpss
from scipy import stats

ROOT = '/home/seismic/LFE'
sys.path.insert(0, f'{ROOT}/scripts')
import sci_source_spectra as SS          # noqa: E402  (read-only reuse of identical code)

OUT = f'{ROOT}/data/sci/family_spectra'
ST1 = f'{OUT}/stage1'
FIG = f'{ROOT}/figures'
RAW = f'{ROOT}/data/sci/source_spectra/raw'
FAMS = list(range(1, 28))
EDGES = SS.EDGES
FCB = np.sqrt(EDGES[:-1] * EDGES[1:])
BSEL = np.arange(2, 10)                  # stored bands: 1.57-9.55 Hz (8 bands)
FB = FCB[BSEL]
# indices into BSEL
U = np.array([1, 2, 3, 4, 5])            # usable 1.97-6.08 Hz
I_LO = [1, 2]                            # 1.97-3.09 Hz
I_MID = [3, 4]                           # 3.09-4.85 Hz
I_HI = [5]                               # 4.85-6.08 Hz
I_HF2 = [6]                              # 6.08-7.62 Hz (beyond usable band; check only)
NB = len(BSEL)
Y0 = 2008
NY = 19
NBOOT = 100
R = 3 + NBOOT
NMINY = 5                                # min events of each family in a year for that year to count
SNRMIN = 2.0                             # noise-corrected mean power / bootstrap sd, both families, all U bands
STRIKE = 293.0


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ------------------------------------------------------------------------------------------- sc
def sc_one(F):
    fn = f'{ST1}/sc_fam{F}.npz'
    if os.path.exists(fn):
        return
    with h5py.File(f'{RAW}/fam{F}.h5', 'r') as h:
        NP = h['NP'][:]
        chans = h['chan'][:].astype(str)
        nstarts = list(h.attrs['noise_starts'])
    tapers = dpss(SS.NWIN, 2.5, 4)
    M = SS.band_matrix()
    sc = np.full((len(chans), len(FCB)), np.nan)
    spread = np.full(len(chans), np.nan)
    with h5py.File(f'{ROOT}/data/waveforms/fam{F}.h5', 'r') as h:
        for k, cid in enumerate(chans):
            ev = h[f'chan/{cid}/ev'][:64]
            X = h[f'chan/{cid}/data'][:64].astype(np.float64)
            pn = np.mean([SS.mt_band_power(X[:, j:j + SS.NWIN], tapers, M) for j in nstarts], 0)
            npk = NP[ev, k].astype(np.float64)
            ok = np.isfinite(npk).all(1) & (pn > 0).all(1) & (npk > -20).all(1)
            if ok.sum() < 5:
                continue
            lr = np.log10(pn[ok]) - npk[ok]
            sc[k] = 10 ** np.median(lr, 0)
            spread[k] = np.max(np.std(lr, 0))
    np.savez(fn, sc=sc, spread=spread, chan=chans)
    log(f'sc fam {F}: {np.isfinite(sc[:, 0]).sum()}/{len(chans)} channels, max log-spread {np.nanmax(spread):.4f}')


def run_sc(a):
    from multiprocessing import Pool
    os.makedirs(ST1, exist_ok=True)
    with Pool(a.procs) as p:
        list(p.imap_unordered(sc_one, a.fams or FAMS))


# ------------------------------------------------------------------------------------------ accum
def tt_of(F, chans):
    origin = json.load(open(f'{ROOT}/data/locations/fam{F}_3d.json'))['origin_rel_catalog_s']
    with h5py.File(f'{RAW}/fam{F}.h5', 'r') as h:
        S = h['S'][:]
    return S - origin


KW = 2.0            # event weight w = A^2 / (A^4 + KW^2) (A = per-event MF amplitude rel. to own stack, median 1)
SCLEAN = (-7.0, 12.0)   # no other catalog event (any family) in this window -> S window uncontaminated
NCLEAN = (-35.0, -9.0)  # no other catalog event in this window -> pre-event noise windows uncontaminated


def accum_one(F):
    """Per family/channel/year sums for two estimators of the family's mean S power spectrum:
    'u' plain mean over S-clean events; 'w' size-normalised: sum w (S - N) / sum w A^2 with
    w = A^2/(A^4+KW^2) (removes the heavy-tailed event-size scatter; ~ spectrum of an A=1 event).
    Noise power from events whose noise windows are clean (mean per year)."""
    fn = f'{ST1}/acc_fam{F}.npz'
    if os.path.exists(fn):
        return
    cat = SS.load_catalog()
    z = np.load(f'{ST1}/sc_fam{F}.npz')
    sc = z['sc'][:, BSEL]
    with h5py.File(f'{RAW}/fam{F}.h5', 'r') as h:
        SP = h['SP'][:, :, BSEL].astype(np.float32)
        NPw = h['NP'][:, :, BSEL].astype(np.float32)
        chans = h['chan'][:].astype(str)
        t = SS.epoch_s(h['time_utc'][:].astype(str))
    n, nch, _ = SP.shape
    allt = cat.t.values
    nS = np.searchsorted(allt, t + SCLEAN[1], 'right') - np.searchsorted(allt, t + SCLEAN[0], 'left')
    nN = np.searchsorted(allt, t + NCLEAN[1], 'right') - np.searchsorted(allt, t + NCLEAN[0], 'left')
    sclean, nclean_ = nS == 1, nN == 0
    Am = pd.read_csv(f'{ROOT}/data/sci/moments/event_amplitudes.csv.gz', usecols=['fam', 'ev', 'A', 'qc_bad'])
    Am = Am[Am.fam == F].set_index('ev').reindex(range(n))
    Aev = Am.A.values
    aok = np.isfinite(Aev) & ~Am.qc_bad.fillna(True).values.astype(bool)
    Aev = np.where(aok, Aev, 0.0)
    wev = Aev ** 2 / (Aev ** 4 + KW ** 2)
    yr = pd.to_datetime(t, unit='s').year.values - Y0
    base = np.isfinite(SP).all(2) & np.isfinite(NPw).all(2) & (SP > -20).all(2) & (NPw > -20).all(2) \
        & np.isfinite(sc[:, 0])[None, :] & (yr >= 0)[:, None] & (yr < NY)[:, None]
    # screen (gross outliers only, so that means are not truncated): S-window power at 2-3 Hz > 10^2.5 x the
    # channel median (earthquakes, glitches) or band-averaged noise power > 10^1.5 x median (glitches)
    lo = SP[:, :, I_LO].mean(2)
    nz = NPw[:, :, U].mean(2)
    nscr = np.zeros(nch, int)
    for k in range(nch):
        v = base[:, k]
        if v.sum() < 20:
            base[:, k] = False
            continue
        bad = v & ((lo[:, k] > np.median(lo[v, k]) + 2.5) | (nz[:, k] > np.median(nz[v, k]) + 1.5))
        nscr[k] = bad.sum()
        base[bad, k] = False
    vS = base & sclean[:, None] & aok[:, None]
    vN = base & nclean_[:, None]
    PS = np.where(vS[:, :, None], 10.0 ** SP * sc[None], 0.0)
    PN = np.where(vN[:, :, None], 10.0 ** NPw * sc[None], 0.0)
    del SP, NPw
    rng = np.random.default_rng(1000 + F)
    W = np.zeros((R, n), np.float32)
    W[0] = 1
    ordr = np.argsort(t)
    rank = np.empty(n, int)
    rank[ordr] = np.arange(n)
    W[1] = (rank % 2 == 0)
    W[2] = (rank % 2 == 1)
    W[3:] = rng.poisson(1.0, (NBOOT, n))
    out = {k: np.zeros((R, NY, nch, NB), np.float32) for k in ['SWw', 'SWu', 'NS']}
    out.update({k: np.zeros((R, NY, nch), np.float32) for k in ['WA', 'W1', 'CU', 'NC']})
    for y in range(NY):
        iy = np.where(yr == y)[0]
        if len(iy) == 0:
            continue
        Wy = W[:, iy].astype(np.float64)
        Wyw = Wy * wev[iy][None]
        out['SWw'][:, y] = (Wyw @ PS[iy].reshape(len(iy), -1)).reshape(R, nch, NB)
        out['SWu'][:, y] = (Wy @ PS[iy].reshape(len(iy), -1)).reshape(R, nch, NB)
        out['NS'][:, y] = (Wy @ PN[iy].reshape(len(iy), -1)).reshape(R, nch, NB)
        out['WA'][:, y] = (Wyw * (Aev[iy] ** 2)[None]) @ vS[iy].astype(np.float64)
        out['W1'][:, y] = Wyw @ vS[iy].astype(np.float64)
        out['CU'][:, y] = Wy @ vS[iy].astype(np.float64)
        out['NC'][:, y] = Wy @ vN[iy].astype(np.float64)
    np.savez_compressed(fn, chan=chans, tt=tt_of(F, chans), nsclean=sclean.sum(), nnclean=nclean_.sum(),
                        n=n, nscreen=nscr, **out)
    log(f'accum fam {F}: {n} events, S-clean {sclean.sum()}, N-clean {nclean_.sum()}, {nch} channels, '
        f'median S-valid/channel {np.median(vS.sum(0)):.0f}, screened {nscr.sum()}')


def run_accum(a):
    from multiprocessing import Pool
    with Pool(a.procs) as p:
        list(p.imap_unordered(accum_one, a.fams or FAMS))


# ---------------------------------------------------------------------------------------- helpers
def family_table():
    L = pd.read_csv(f'{ROOT}/data/families/shelly_family_locations.csv').set_index('idnum')
    lat0, lon0 = L.lat.mean(), L.lon.mean()
    x = (L.lon - lon0) * 111.195 * np.cos(np.radians(lat0))
    y = (L.lat - lat0) * 111.195
    s = np.radians(STRIKE)
    L['E_km'], L['N_km'] = x, y
    L['along_km'] = x * np.sin(s) + y * np.cos(s)            # + WNW
    L['across_km'] = x * np.sin(s + np.pi / 2) + y * np.cos(s + np.pi / 2)   # + NNE (down-dip side)
    our = {F: json.load(open(f'{ROOT}/data/locations/fam{F}_3d.json')) for F in FAMS}
    L['lat3d'] = [our[F]['lat'] for F in L.index]
    L['lon3d'] = [our[F]['lon'] for F in L.index]
    L['dep3d'] = [our[F]['depth_km'] for F in L.index]
    return L


def dist_matrix(L, which='shelly'):
    if which == 'shelly':
        la, lo, d = L.lat.values, L.lon.values, L.depth_km.values
    else:
        la, lo, d = L.lat3d.values, L.lon3d.values, L.dep3d.values
    x = lo * 111.195 * np.cos(np.radians(40.06))
    y = la * 111.195
    P = np.c_[x, y, d]
    return np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))


def load_acc():
    A = {}
    for F in FAMS:
        z = np.load(f'{ST1}/acc_fam{F}.npz', allow_pickle=False)
        A[F] = {k: z[k] for k in ['SWw', 'SWu', 'NS', 'WA', 'W1', 'CU', 'NC', 'tt']}
        A[F].update(chan=list(z['chan'].astype(str)), nclean=int(z['nsclean']), n=int(z['n']))
    return A


def fam_year_terms(f, idx, mode):
    """-> num (R, NY, nc, NB), den (R, NY, nc), ok-count (NY, nc) for a family on channels idx."""
    nbar = f['NS'][:, :, idx, :] / np.maximum(f['NC'][:, :, idx], 1e-6)[..., None]
    if mode == 'w':
        num = f['SWw'][:, :, idx, :] - f['W1'][:, :, idx][..., None] * nbar
        den = f['WA'][:, :, idx]
    else:
        num = f['SWu'][:, :, idx, :] - f['CU'][:, :, idx][..., None] * nbar
        den = f['CU'][:, :, idx]
    okc = np.minimum(f['CU'][0][:, idx], f['NC'][0][:, idx])
    return num, den, okc


def pair_obs(A, L, D, mode='w', group='sta'):
    """All pair x channel observations (no distance cut; D stored). Returns DataFrame + arrays
    y[obs, R, NB] (0.5*log10 power ratio = log10 amplitude ratio). Year-matched: both families'
    per-year estimates are combined with the same year weights (cancels gain/instrument epochs)."""
    rows, Ys = [], []
    for i, Fa in enumerate(FAMS):
        for Fb in FAMS[i + 1:]:
            a, b = A[Fa], A[Fb]
            com = [c for c in a['chan'] if c in b['chan']]
            if not com:
                continue
            ia = np.array([a['chan'].index(c) for c in com])
            ib = np.array([b['chan'].index(c) for c in com])
            numa, dena, oka = fam_year_terms(a, ia, mode)
            numb, denb, okb = fam_year_terms(b, ib, mode)
            da, db = dena[0], denb[0]
            w = np.where((oka >= NMINY) & (okb >= NMINY) & (da > 0) & (db > 0),
                         da * db / np.maximum(da + db, 1e-12), 0.0)                # (NY, nc)
            sw = w.sum(0)
            k = sw > 0
            if not k.any():
                continue

            def comb(num, den):
                m = num / np.maximum(den, 1e-12)[..., None]
                return np.einsum('rycb,yc->rcb', m, w) / np.maximum(sw, 1e-12)[None, :, None]
            Ma, Mb = comb(numa, dena), comb(numb, denb)
            cc = [c for c, kk in zip(com, k) if kk]
            ia, ib, Ma, Mb = ia[k], ib[k], Ma[:, k], Mb[:, k]
            w = w[:, k]
            if group == 'sta':                  # sum noise-corrected power over a station's components
                stl = np.array(['.'.join(c.split('.')[:2]) for c in cc])
                us = list(dict.fromkeys(stl))
                G = (stl[None, :] == np.array(us)[:, None]).astype(float)        # (ns, nc)
                Ma = np.einsum('sc,rcb->rsb', G, Ma)
                Mb = np.einsum('sc,rcb->rsb', G, Mb)
                ia = np.array([ia[stl == u_].min() for u_ in us])
                ib = np.array([ib[stl == u_].min() for u_ in us])
                tta = np.array([a['tt'][ia_[stl == u_]].mean() for u_ in us for ia_ in [np.array([a['chan'].index(c) for c in cc])]])
                ttb = np.array([b['tt'][ib_[stl == u_]].mean() for u_ in us for ib_ in [np.array([b['chan'].index(c) for c in cc])]])
                ncomp = G.sum(1)
                com = [u_ + '.sum' for u_ in us]
                w = np.ones((NY, len(us)))
                sw = np.ones(len(us))
            else:
                tta, ttb = a['tt'][ia], b['tt'][ib]
                ncomp = np.ones(len(cc))
                com = cc
                sw = sw[k]
            k = np.ones(len(com), bool)
            sda, sdb = Ma[3:].std(0), Mb[3:].std(0)
            snra, snrb = Ma[0] / sda, Mb[0] / sdb
            good = k & (snra[:, U] > SNRMIN).all(1) & (snrb[:, U] > SNRMIN).all(1)
            if not good.any():
                continue
            tiny = 1e-30
            Y = 0.5 * (np.log10(np.maximum(Ma, tiny)) - np.log10(np.maximum(Mb, tiny)))   # (R, nc, NB)
            for j in np.where(good)[0]:
                c = com[j]
                rows.append(dict(A=Fa, B=Fb, chan=c, sta='.'.join(c.split('.')[:2]), comp=c[-1] if group == 'chan' else 'S',
                                 is_pb=c.startswith('PB.'), D=D[Fa - 1, Fb - 1],
                                 ttA=tta[j], ttB=ttb[j], ncomp=ncomp[j],
                                 nyears=float((w[:, j] > 0).sum()), wsum=float(sw[j]),
                                 snr_min=float(min(snra[j, U].min(), snrb[j, U].min())),
                                 snr_hf2=float(min(snra[j, I_HF2[0]], snrb[j, I_HF2[0]]))))
                Ys.append(Y[:, j, :].astype(np.float32))
    O = pd.DataFrame(rows)
    Y = np.stack(Ys)                                                        # (nobs, R, NB)
    O['dtt'] = O.ttA - O.ttB
    O['dlogtt'] = np.log10(O.ttA / O.ttB)
    return O, Y


def stat_cols(Y):
    """Y (..., NB) -> dict of index arrays (...,): per-band and LO/MID/HI/HFI/slope."""
    lo = Y[..., I_LO].mean(-1)
    out = {f'b{j}': Y[..., j] for j in range(NB)}
    out['LO'] = lo
    out['MID'] = Y[..., I_MID].mean(-1) - lo
    out['HFI'] = Y[..., I_HI].mean(-1) - lo
    out['HF2'] = Y[..., I_HF2].mean(-1) - lo
    fu = FB[U] - FB[U].mean()
    out['slope'] = (Y[..., U] * fu).sum(-1) / (fu ** 2).sum()             # log10 amp per Hz
    return out


def solve_ls(A_idx, B_idx, y, se, X=None, nF=27, tau2=None, huber=2.0, iters=6, fix=None):
    """Weighted LSQ y = x_A - x_B + X g; sum(x over families with data) = 0. Returns x, g, tau2, info.
    tau2 (extra path scatter) is estimated so that the robust chi2/dof = 1 unless given."""
    n = len(y)
    k = 0 if X is None else X.shape[1]
    G = np.zeros((n, nF + k))
    G[np.arange(n), A_idx] = 1.0
    G[np.arange(n), B_idx] -= 1.0
    if k:
        G[:, nF:] = X
    have = np.zeros(nF, bool)
    have[A_idx] = True
    have[B_idx] = True
    C = np.zeros((1, nF + k))
    C[0, :nF] = have
    est_tau = tau2 is None
    t2 = np.median(se ** 2) if est_tau else tau2
    hw = np.ones(n)
    for it in range(iters):
        w = hw / (se ** 2 + t2)
        sw = np.sqrt(w)
        Gw = np.vstack([G * sw[:, None], 1e3 * C])
        yw = np.r_[y * sw, 0.0]
        keep = np.r_[have, np.ones(k, bool)]
        sol = np.full(nF + k, np.nan)
        sol[keep] = np.linalg.lstsq(Gw[:, keep], yw, rcond=None)[0]
        res = y - G[:, keep] @ sol[keep]
        if est_tau:
            dof = max(n - keep.sum(), 1)
            lo_, hi_ = 0.0, max(np.var(res) * 4, 1e-8)
            for _ in range(50):
                mid = 0.5 * (lo_ + hi_)
                if (hw * res ** 2 / (se ** 2 + mid)).sum() > dof:
                    lo_ = mid
                else:
                    hi_ = mid
            t2 = 0.5 * (lo_ + hi_)
        z = np.abs(res) / np.sqrt(se ** 2 + t2)
        hw = np.where(z > huber, huber / np.maximum(z, 1e-9), 1.0)
    return sol[:nF], sol[nF:], t2, dict(res=res, w=hw / (se ** 2 + t2), have=have)


def solve_stat(O, ys, ses, mask, path, tau2=None, nboot_ev=None, Yboot=None, sta_boot=0, rng=None):
    """Solve for one statistic. ys (nobs,) replicate-0 values, ses (nobs,), Yboot (nobs, nb) replicates.
    path: list of regressor column names in O. Returns dict with x, g, sd_ev, sd_sta."""
    m = mask & np.isfinite(ys) & np.isfinite(ses)
    Ai = O.A.values[m] - 1
    Bi = O.B.values[m] - 1
    X = O.loc[m, path].values if path else None
    x, g, t2, info = solve_ls(Ai, Bi, ys[m], ses[m], X, tau2=tau2)
    out = dict(x=x, g=g, tau2=t2, nobs=int(m.sum()), have=info['have'])
    if Yboot is not None:
        xb = []
        for r in range(Yboot.shape[1]):
            xr, _, _, _ = solve_ls(Ai, Bi, Yboot[m, r], ses[m], X, tau2=t2, iters=2)
            xb.append(xr)
        out['sd_ev'] = np.nanstd(np.array(xb), 0)
        out['xb_ev'] = np.array(xb)
    if sta_boot:
        stas = O.sta.values[m]
        us = np.unique(stas)
        xs = []
        for r in range(sta_boot):
            pick = rng.choice(us, len(us), replace=True)
            cnts = pd.Series(pick).value_counts()
            mult = pd.Series(stas).map(cnts).fillna(0).values
            kk = mult > 0
            if kk.sum() < 10:
                continue
            # multiplicity via scaled variance
            xr, _, _, _ = solve_ls(Ai[kk], Bi[kk], ys[m][kk], ses[m][kk] / np.sqrt(mult[kk]),
                                   None if X is None else X[kk], tau2=t2 / 1.0, iters=2)
            xs.append(xr)
        xs = np.array(xs)
        out['sd_sta'] = np.nanstd(xs, 0)
        out['xb_sta'] = xs
    return out


def brune_band(fc, edges=EDGES[np.r_[BSEL, BSEL[-1] + 1]]):
    """0.5*log10 band-mean POWER of a velocity omega^-2 source (f^2/(1+(f/fc)^2)^2), per stored band,
    minus the same for fc -> infinity (so it is the corner-only effect)."""
    out = []
    for b in range(NB):
        f = np.linspace(edges[b], edges[b + 1], 60)
        out.append(0.5 * np.log10(np.mean(f ** 2 / (1 + (f / fc) ** 2) ** 2) / np.mean(f ** 2)))
    return np.array(out)


def brune_sens(fc0s=(1.5, 2, 3, 4, 6, 8, 12)):
    rows = []
    for fc0 in fc0s:
        d = 0.01
        g = (brune_band(fc0 * np.exp(d)) - brune_band(fc0 * np.exp(-d))) / (2 * d)
        s = stat_cols(g[None])
        rows.append(dict(fc0=fc0, **{f'd{k}_dlnfc': float(s[k][0]) for k in ['LO', 'MID', 'HFI', 'HF2', 'slope']}))
    return pd.DataFrame(rows)


def brune_fit(xb, sb, fc0, lgrid=np.linspace(-2.5, 2.5, 251)):
    """xb (NB,) relative log10 amplitude spectrum of a family (bands U used), sb (NB,) its sd.
    Fit x = m + B(fc0*e^l) - B(fc0); returns l (ln fc ratio), m, chi2 profile."""
    w = 1 / sb[U] ** 2
    B0 = brune_band(fc0)[U]
    best = (np.inf, np.nan, np.nan)
    prof = []
    for l in lgrid:
        d = brune_band(fc0 * np.exp(l))[U] - B0
        m = np.sum(w * (xb[U] - d)) / w.sum()
        c2 = np.sum(w * (xb[U] - d - m) ** 2)
        prof.append(c2)
        if c2 < best[0]:
            best = (c2, l, m)
    return best[1], best[2], np.array(prof)


# ------------------------------------------------------------------------------------------ solve
def covariates(L, xinfo):
    P = pd.read_csv(f'{ROOT}/data/patches/patch_tensors.csv').set_index('family')
    Rs = pd.read_csv(f'{ROOT}/data/sci/patch_images/ovl_incl/real_summary.csv').set_index('F')
    cat = pd.read_csv(f'{ROOT}/data/shelly2026_MTJ_v2/LFEcatV2_MTJ_2008-2026.csv')
    cat['t'] = pd.to_datetime(dict(year=cat['%year'], month=cat.month, day=cat.day, hour=cat.hr,
                                   minute=cat['min'])).values.astype('datetime64[s]').astype(float) + cat.sec
    span = (cat.t.max() - cat.t.min()) / (365.25 * 86400)
    C = pd.DataFrame(index=FAMS)
    C.index.name = 'family'
    for F in FAMS:
        tf = np.sort(cat.t.values[cat.idnum.values == F])
        iet = np.diff(tf)
        ep = np.r_[True, iet > 12 * 3600]
        C.loc[F, 'n_cat'] = len(tf)
        C.loc[F, 'rate_per_yr'] = len(tf) / span
        C.loc[F, 'median_iet_h'] = np.median(iet) / 3600
        C.loc[F, 'episodes'] = ep.sum()
        C.loc[F, 'episode_recur_d'] = span * 365.25 / ep.sum()
        C.loc[F, 'events_per_episode'] = len(tf) / ep.sum()
    C['L1_m'] = P.L1_m.where(P.resolved == True)
    C['L2_m'] = P.L2_m.where(P.resolved == True)
    C['area_res_m2'] = np.pi / 4 * C.L1_m * C.L2_m
    C['Bstd_s_m'] = Rs.B_full_std_s
    C['Bstd_d_m'] = Rs.B_full_std_d
    C['Barea_m2'] = 2 * np.pi * C.Bstd_s_m * C.Bstd_d_m          # ~ ellipse of 1-sigma... used as a proxy
    C['logL_B'] = 0.5 * np.log10(C.Bstd_s_m * C.Bstd_d_m)
    C['depth_km'] = L.depth_km
    C['along_km'] = L.along_km
    C['across_km'] = L.across_km
    C['dep3d_km'] = L.dep3d
    return C


def rank_partial(x, y, Z):
    """Spearman partial correlation of x and y given columns Z (rank-transform, linear residuals)."""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    if Z is None or Z.shape[1] == 0:
        return stats.pearsonr(rx, ry)[0]
    Zr = np.column_stack([np.ones(len(x))] + [stats.rankdata(z) for z in Z.T])
    ex = rx - Zr @ np.linalg.lstsq(Zr, rx, rcond=None)[0]
    ey = ry - Zr @ np.linalg.lstsq(Zr, ry, rcond=None)[0]
    return stats.pearsonr(ex, ey)[0]


def local_perm(n_idx, close_pairs, rng, nswap=400):
    """Random permutation built from swaps along close pairs (labels move only between neighbours)."""
    p = np.arange(n_idx)
    for _ in range(nswap):
        i, j = close_pairs[rng.integers(len(close_pairs))]
        p[[i, j]] = p[[j, i]]
    return p


def run_solve(a):
    t0 = time.time()
    OUTS = OUT if a.mode == 'w' else f'{OUT}/plain'
    os.makedirs(OUTS, exist_ok=True)
    rng = np.random.default_rng(7)
    L = family_table()
    Dm = dist_matrix(L, 'shelly')
    D3 = dist_matrix(L, '3d')
    A = load_acc()
    log('loaded accumulators')
    O, Y = pair_obs(A, L, Dm, a.mode, 'sta')
    O['D3'] = [D3[i - 1, j - 1] for i, j in zip(O.A, O.B)]
    Oc, Yc = pair_obs(A, L, Dm, a.mode, 'chan')
    Oc['D3'] = [D3[i - 1, j - 1] for i, j in zip(Oc.A, Oc.B)]
    Sc_ = stat_cols(Yc)
    sec_ = {k: v[:, 3:].std(1) for k, v in Sc_.items()}
    log(f'per-channel observations: {len(Oc)}')
    log(f'pair x channel observations: {len(O)} ({O.groupby(["A", "B"]).ngroups} pairs, '
        f'{O.sta.nunique()} stations)')
    S = stat_cols(Y)                         # each (nobs, R)
    se = {k: v[:, 3:].std(1) for k, v in S.items()}
    sens = brune_sens()
    sens.to_csv(f'{OUTS}/brune_sensitivity_sp.csv', index=False)
    O.assign(**{f'y_{k}': S[k][:, 0] for k in ['LO', 'MID', 'HFI', 'HF2', 'slope']},
             **{f'se_{k}': se[k] for k in ['LO', 'MID', 'HFI', 'HF2', 'slope']}) \
        .round(5).to_csv(f'{OUTS}/pair_channel_obs.csv.gz', index=False)

    path_sets = {'none': [], 'dtt': ['dtt'], 'dtt_dlog': ['dtt', 'dlogtt']}
    QK = ['LO', 'MID', 'HFI', 'slope'] + [f'b{j}' for j in range(NB)]
    variants = []
    # main variant definition
    for Dcut in [2.0, 3.0, 5.0, 99.0]:
        for pth in ['none', 'dtt']:
            variants.append(dict(name=f'D{Dcut:g}_{pth}', D=Dcut, path=pth, sel='all'))
    for sel in ['pb', 'surf', 'Z', 'H', 'chan', 'half1', 'half2', 'D3d']:
        variants.append(dict(name=f'D3_dtt_{sel}', D=3.0, path='dtt', sel=sel))
    variants.append(dict(name='D3_dtt_dlog', D=3.0, path='dtt_dlog', sel='all'))
    MAIN = 'D3_dtt'
    res_rows, boot_store = [], {}
    for V in variants:
        if V['sel'] in ['Z', 'H', 'chan']:
            O_, S_, se_ = Oc, Sc_, sec_
        else:
            O_, S_, se_ = O, S, se
        if V['sel'] == 'D3d':
            m = np.array(O_.D3.values < V['D'])
        else:
            m = np.array(O_.D.values < V['D'])
        if V['sel'] == 'pb':
            m &= O_.is_pb.values
        elif V['sel'] == 'surf':
            m &= ~O_.is_pb.values
        elif V['sel'] == 'Z':
            m &= (O_.comp == 'Z').values
        elif V['sel'] == 'H':
            m &= (O_.comp != 'Z').values
        rep = 1 if V['sel'] == 'half1' else 2 if V['sel'] == 'half2' else 0
        for q in QK:
            pth = path_sets[V['path']]
            if q == 'LO' and V['path'] == 'dtt_dlog':
                pth = ['dtt', 'dlogtt']
            elif V['path'] == 'dtt_dlog':
                pth = ['dtt']
            full = (V['name'] == MAIN) or (q in ['LO', 'HFI', 'slope', 'MID'] and V['sel'] in ['all', 'pb', 'surf'])
            r = solve_stat(O_, S_[q][:, rep], se_[q], m, pth,
                           Yboot=S_[q][:, 3:] if full else None,
                           sta_boot=200 if full else 0, rng=rng)
            for F in FAMS:
                i = F - 1
                row = dict(variant=V['name'], D=V['D'], path=V['path'], sel=V['sel'], stat=q, family=F,
                           x=r['x'][i], nobs=r['nobs'], tau=np.sqrt(r['tau2']),
                           g=r['g'][0] if len(r['g']) else np.nan)
                if 'sd_ev' in r:
                    row.update(sd_ev=r['sd_ev'][i], sd_sta=r['sd_sta'][i],
                               se=np.sqrt(r['sd_ev'][i] ** 2 + r['sd_sta'][i] ** 2))
                res_rows.append(row)
            if V['name'] == MAIN:
                boot_store[q] = (r['x'], r['xb_ev'], r['xb_sta'])
        log(f"variant {V['name']}: nobs {m.sum()}")
    Rdf = pd.DataFrame(res_rows)
    Rdf.to_csv(f'{OUTS}/family_solutions_all_variants.csv', index=False)

    # ---------------- main per-family table
    M = Rdf[Rdf.variant == MAIN].pivot(index='family', columns='stat', values='x')
    Mse = Rdf[Rdf.variant == MAIN].pivot(index='family', columns='stat', values='se')
    T = pd.DataFrame(index=FAMS)
    T.index.name = 'family'
    for q in ['LO', 'MID', 'HFI', 'slope']:
        T[q] = M[q]
        T[q + '_se'] = Mse[q]
    nob = O[O.D < 3].groupby('A').size().add(O[O.D < 3].groupby('B').size(), fill_value=0)
    T['n_pairchan_D3'] = nob.reindex(FAMS).fillna(0).astype(int)
    T['n_partners_D3'] = [int(((Dm[F - 1] < 3) & (np.arange(27) != F - 1)).sum()) for F in FAMS]
    T['nclean'] = [A[F]['nclean'] for F in FAMS]
    # band spectra with bootstrap
    xbands = np.array([boot_store[f'b{j}'][0] for j in range(NB)]).T              # (27, NB)
    xb_ev = np.stack([boot_store[f'b{j}'][1] for j in range(NB)], -1)            # (nboot, 27, NB)
    xb_st = np.stack([boot_store[f'b{j}'][2] for j in range(NB)], -1)
    sdb = np.sqrt(np.nanstd(xb_ev, 0) ** 2 + np.nanstd(xb_st, 0) ** 2)
    for j in range(NB):
        T[f'x_{FB[j]:.2f}Hz'] = xbands[:, j]
        T[f'x_{FB[j]:.2f}Hz_se'] = sdb[:, j]
    # Brune fits: per fc0, point + bootstrap (event & station replicates combined)
    nb_ev, nb_st = xb_ev.shape[0], xb_st.shape[0]
    for fc0 in [2.0, 3.0, 4.0, 6.0]:
        lf, lfb = np.full(27, np.nan), []
        for i in range(27):
            if not np.isfinite(xbands[i, U]).all():
                continue
            lf[i] = brune_fit(xbands[i], sdb[i], fc0)[0]
        # bootstrap: combine an event replicate deviation and a station replicate deviation
        for r in range(100):
            xr = xbands + (xb_ev[r % nb_ev] - xbands) + (xb_st[r % nb_st] - xbands)
            lfb.append([brune_fit(xr[i], sdb[i], fc0)[0] if np.isfinite(xr[i, U]).all() else np.nan
                        for i in range(27)])
        lfb = np.array(lfb)
        T[f'lnfc_fc0_{fc0:g}'] = lf
        T[f'lnfc_fc0_{fc0:g}_lo'] = np.nanpercentile(lfb, 16, 0)
        T[f'lnfc_fc0_{fc0:g}_hi'] = np.nanpercentile(lfb, 84, 0)
        log(f'brune fc0 {fc0}')
    # linearised HFI -> dln fc using the SP-band sensitivity
    for fc0 in [2.0, 3.0, 4.0, 6.0]:
        sH = float(sens.loc[sens.fc0 == fc0, 'dHFI_dlnfc'].iloc[0])
        T[f'dlnfc_lin_HFI_fc0_{fc0:g}'] = T.HFI / sH
        T[f'dlnfc_lin_HFI_fc0_{fc0:g}_se'] = T.HFI_se / sH
    # chi2 of family differences vs noise
    chi = {}
    for q in ['LO', 'HFI', 'slope', 'MID']:
        ok = T[q + '_se'].notna() & T[q].notna()
        z = (T.loc[ok, q] - np.average(T.loc[ok, q], weights=1 / T.loc[ok, q + '_se'] ** 2)) / T.loc[ok, q + '_se']
        c2 = float((z ** 2).sum())
        chi[q] = dict(chi2=c2, dof=int(ok.sum() - 1), p=float(stats.chi2.sf(c2, ok.sum() - 1)),
                      sd_obs=float(T.loc[ok, q].std()), median_se=float(T.loc[ok, q + '_se'].median()),
                      sd_intrinsic=float(np.sqrt(max(T.loc[ok, q].var() - (T.loc[ok, q + '_se'] ** 2).mean(), 0))))
    C = covariates(L, T)
    mean_tt = pd.Series({F: np.nanmean(A[F]['tt']) for F in FAMS})
    C['mean_tt_s'] = mean_tt
    T = T.join(C)
    T.round(5).to_csv(f'{OUTS}/family_spectra.csv')

    # ---------------- validation
    val = {}

    def get(vn, q):
        d = Rdf[(Rdf.variant == vn) & (Rdf.stat == q)].set_index('family')
        return d

    for q in ['LO', 'HFI', 'slope', 'MID']:
        for va, vb in [('D3_dtt_pb', 'D3_dtt_surf'), ('D3_dtt_half1', 'D3_dtt_half2'), ('D3_dtt_Z', 'D3_dtt_H'), ('D3_dtt', 'D3_dtt_chan'),
                       ('D3_dtt', 'D3_none'), ('D3_dtt', 'D2_dtt'), ('D3_dtt', 'D5_dtt'), ('D3_dtt', 'D99_dtt'),
                       ('D3_dtt', 'D3_dtt_D3d'), ('D3_dtt', 'D99_none')]:
            da, db = get(va, q), get(vb, q)
            ok = da.x.notna() & db.x.notna()
            xa, xb = da.x[ok], db.x[ok]
            dd = dict(n=int(ok.sum()), pearson=float(stats.pearsonr(xa, xb)[0]),
                      spearman=float(stats.spearmanr(xa, xb)[0]), p_spearman=float(stats.spearmanr(xa, xb)[1]),
                      rms_diff=float(np.sqrt(np.mean((xa - xb) ** 2))), slope_tls=float(np.std(xb) / np.std(xa)))
            if 'se' in da and 'se' in db and da.se.notna().any() and db.se.notna().any():
                okk = ok & da.se.notna() & db.se.notna()
                z = (da.x[okk] - db.x[okk]) / np.sqrt(da.se[okk] ** 2 + db.se[okk] ** 2)
                dd.update(chi2_diff=float((z ** 2).sum()), p_diff=float(stats.chi2.sf((z ** 2).sum(), okk.sum())))
            val[f'{q}:{va}~{vb}'] = dd
    # path coefficients
    gq = Rdf[(Rdf.family == 1) & Rdf.path.isin(['dtt'])][['variant', 'stat', 'g', 'tau', 'nobs']]
    gq.to_csv(f'{OUTS}/path_terms.csv', index=False)

    # ---------------- azimuthal consistency: residuals of the main HFI solution by station
    m3 = (O.D < 3).values
    xm = M['HFI'].values
    gm = float(Rdf[(Rdf.variant == MAIN) & (Rdf.stat == 'HFI')].g.iloc[0])
    resid = S['HFI'][:, 0] - (xm[O.A.values - 1] - xm[O.B.values - 1]) - gm * O.dtt.values
    sres = pd.DataFrame(dict(sta=O.sta, r=resid, se=se['HFI']))[m3].groupby('sta').apply(
        lambda g: pd.Series(dict(n=len(g), mean_res=np.average(g.r, weights=1 / (g.se ** 2 + 0.02 ** 2)),
                                 med_res=g.r.median())))
    sres.to_csv(f'{OUTS}/station_residuals_HFI.csv')

    # ---------------- scaling tests
    close = np.array([(i, j) for i in range(27) for j in range(i + 1, 27) if Dm[i, j] < 3])
    cov_list = ['L1_m', 'area_res_m2', 'Bstd_s_m', 'Bstd_d_m', 'logL_B', 'LO', 'rate_per_yr', 'median_iet_h',
                'episode_recur_d', 'events_per_episode', 'slip_rel', 'depth_km', 'along_km', 'across_km',
                'mean_tt_s', 'dep3d_km']
    T['slip_rel'] = T.LO - np.log10(T.Barea_m2)       # log10 (M0 proxy / area) relative
    T['logM0_rel'] = T.LO
    sc_rows = []
    targets = ['HFI', 'slope', 'MID', 'lnfc_fc0_3', 'LO']
    for tq in targets:
        for cv in cov_list:
            if cv == tq:
                continue
            d = T[[tq, cv, 'depth_km', 'along_km', 'mean_tt_s'] + ([tq + '_se'] if tq + '_se' in T else [])].dropna()
            if len(d) < 6:
                continue
            x, y = d[tq].values, d[cv].values
            rho = stats.spearmanr(x, y)[0]
            idx = np.array(d.index) - 1
            # global permutation null
            nul = np.array([stats.spearmanr(x, rng.permutation(y))[0] for _ in range(2000)])
            p_perm = float((np.abs(nul) >= abs(rho)).mean())
            # local (close-pair swap) permutation null: covariates move only between families < 3 km apart
            pos = {f: k for k, f in enumerate(idx)}
            cp = [(pos[i], pos[j]) for i, j in close if i in pos and j in pos]
            if len(cp) >= 3:
                nl = []
                for _ in range(1000):
                    pp = local_perm(len(idx), cp, rng)
                    nl.append(stats.spearmanr(x, y[pp])[0])
                nl = np.array(nl)
                p_local = float((np.abs(nl) >= abs(rho)).mean())
                frac_moved = float(np.mean([np.mean(local_perm(len(idx), cp, rng) != np.arange(len(idx))) for _ in range(50)]))
            else:
                p_local, frac_moved = np.nan, np.nan
            # measurement-error MC for the rho CI
            if tq + '_se' in d:
                rmc = [stats.spearmanr(x + rng.normal(0, d[tq + '_se'].values), y)[0] for _ in range(500)]
                ci = np.percentile(rmc, [16, 84])
            else:
                ci = [np.nan, np.nan]
            Z = d[['depth_km', 'along_km', 'mean_tt_s']].values
            pz = [c for c in ['depth_km', 'along_km', 'mean_tt_s'] if c != cv]
            rp = rank_partial(x, y, d[pz].values)
            # partial-correlation null by global permutation of the covariate
            npn = np.array([rank_partial(x, rng.permutation(y), d[pz].values) for _ in range(1000)])
            sc_rows.append(dict(target=tq, covariate=cv, n=len(d), spearman=rho, rho_lo=ci[0], rho_hi=ci[1],
                                p_perm=p_perm, p_local=p_local, local_frac_moved=frac_moved,
                                partial_rho=rp, p_partial=float((np.abs(npn) >= abs(rp)).mean()),
                                partial_on='+'.join(pz)))
    SC = pd.DataFrame(sc_rows)
    SC.round(4).to_csv(f'{OUTS}/scaling_tests.csv', index=False)

    # ---------------- pair-level (model-free) scaling: close pairs, channel-averaged contrasts
    pr = []
    Om = O[O.D < 3].copy()
    for q in ['HFI', 'LO', 'slope']:
        Om[q] = S[q][(O.D < 3).values, 0]
        Om[q + '_se'] = se[q][(O.D < 3).values]
    gp = Om.groupby(['A', 'B'])
    PP = gp.apply(lambda g: pd.Series({**{q: np.average(g[q], weights=1 / (g[q + '_se'] ** 2 + 0.03 ** 2))
                                           for q in ['HFI', 'LO', 'slope']},
                                        'nchan': len(g), 'D': g.D.iloc[0]})).reset_index()
    for cv in ['L1_m', 'logL_B', 'rate_per_yr', 'episode_recur_d', 'depth_km', 'along_km']:
        dc = T[cv].reindex(PP.A).values - T[cv].reindex(PP.B).values
        for q in ['HFI', 'slope']:
            ok = np.isfinite(dc) & np.isfinite(PP[q].values)
            if ok.sum() < 6:
                continue
            dx, dcv = PP[q].values[ok], dc[ok]
            r0 = stats.spearmanr(dx, dcv)[0]
            fl = []
            for _ in range(2000):
                sgn = rng.choice([-1, 1], ok.sum())
                fl.append(stats.spearmanr(dx, dcv * sgn)[0])
            pr.append(dict(target=q, covariate=cv, npairs=int(ok.sum()), spearman_pairs=r0,
                           p_signflip=float((np.abs(fl) >= abs(r0)).mean())))
    PRdf = pd.DataFrame(pr)
    PRdf.round(4).to_csv(f'{OUTS}/pair_scaling_tests.csv', index=False)
    PP.round(5).to_csv(f'{OUTS}/close_pair_contrasts.csv', index=False)

    # ---------------- moment vs corner (within-population scaling): regress dln fc on LO (ln M0 proxy)
    scal = {}
    for fc0 in [3.0, 4.0]:
        col = f'lnfc_fc0_{fc0:g}'
        d = T[[col, 'LO', 'HFI', 'HFI_se', 'LO_se']].dropna()
        # Brune-fit moment proxy: m from the fit is ~LO + corner correction; use LO (2-3 Hz) and also m_fit
        b, a0 = np.polyfit(d.LO * np.log(10), d[col], 1)
        scal[f'dlnfc_dlnLO_fc0_{fc0:g}'] = float(b)
    # Brune-consistent joint view: predicted HFI-LO relation for self-similar (fc ~ M0^-1/3) and constant
    # duration, using the SP sensitivity table at fc0=3,4 Hz
    for fc0 in [3.0, 4.0]:
        s_ = sens[sens.fc0 == fc0].iloc[0]
        # if ln M0 changes by dm (natural), self-similar: dlnfc = -dm/3; LO changes by dm/ln10 + dLO_dlnfc*dlnfc
        dm = 1.0
        dln = -dm / 3
        scal[f'pred_selfsim_dHFI_dLO_fc0_{fc0:g}'] = float((s_.dHFI_dlnfc * dln) /
                                                           (dm / np.log(10) + s_.dLO_dlnfc * dln))
    d = T[['HFI', 'LO', 'HFI_se', 'LO_se']].dropna()
    # errors-in-variables: Deming-like via bootstrap over families with measurement noise
    bb = []
    for _ in range(2000):
        k = rng.integers(0, len(d), len(d))
        bb.append(np.polyfit(d.LO.values[k], d.HFI.values[k], 1)[0])
    scal['obs_dHFI_dLO_ols'] = float(np.polyfit(d.LO, d.HFI, 1)[0])
    scal['obs_dHFI_dLO_ci68'] = [float(v) for v in np.percentile(bb, [16, 84])]
    # attenuation correction of the OLS slope for noise in LO
    rel = 1 - (d.LO_se ** 2).mean() / d.LO.var()
    scal['LO_reliability'] = float(rel)
    scal['obs_dHFI_dLO_disattenuated'] = float(scal['obs_dHFI_dLO_ols'] / max(rel, 0.05))
    # correlated-noise check: LO and HFI share the LO band, so noise in LO enters HFI with slope -1
    summ = dict(n_obs=int(len(O)), n_pairs=int(O.groupby(['A', 'B']).ngroups), n_stations=int(O.sta.nunique()),
                chi2_family_differences=chi, validation=val, scaling_moment_corner=scal,
                band_centres_hz=[float(v) for v in FB], usable_bands=[float(v) for v in FB[U]],
                main_variant=MAIN, runtime_s=time.time() - t0)
    json.dump(summ, open(f'{OUTS}/summary.json', 'w'), indent=1, default=float)
    log('solve done', f'{time.time() - t0:.0f} s')


# ------------------------------------------------------------------------------------------- figs
def run_figs(a):
    OUTS = OUT if a.mode == 'w' else f'{OUT}/plain'
    sfx = '' if a.mode == 'w' else '_plain'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    T = pd.read_csv(f'{OUTS}/family_spectra.csv').set_index('family')
    Rdf = pd.read_csv(f'{OUTS}/family_solutions_all_variants.csv')
    SC = pd.read_csv(f'{OUTS}/scaling_tests.csv')
    summ = json.load(open(f'{OUTS}/summary.json'))
    # --- 1: spectra + map
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
    cm = plt.get_cmap('coolwarm')
    v = T.HFI
    nv = (v - v.min()) / (v.max() - v.min())
    for F in T.index:
        xs = np.array([T.loc[F, f'x_{f:.2f}Hz'] for f in FB])
        es = np.array([T.loc[F, f'x_{f:.2f}Hz_se'] for f in FB])
        ax[0].errorbar(FB, xs - T.loc[F, 'LO'], yerr=es, color=cm(nv[F]), lw=1, alpha=0.8, capsize=0)
    ax[0].axvspan(FB[U[0]] / 1.12, FB[U[-1]] * 1.12, color='0.9', zorder=-1)
    for fc0, ls in [(3.0, '--'), (4.0, ':')]:
        for l in [-0.5, 0.5]:
            d = brune_band(fc0 * np.exp(l)) - brune_band(fc0)
            ax[0].plot(FB, d - d[I_LO].mean(), 'k', ls=ls, lw=0.8)
    ax[0].set_xscale('log')
    ax[0].set_xlabel('Hz')
    ax[0].set_ylabel('relative log10 amplitude (normalised at 2-3 Hz)')
    ax[0].set_title('Per-family relative S spectra (D<3 km pairs, dtt path term)\n'
                    'black: Brune fc x e^{+-0.5} around 3 Hz (--) / 4 Hz (:)', fontsize=9)
    ax[0].set_ylim(-0.6, 0.6)
    o = T.sort_values('HFI')
    ax[1].errorbar(np.arange(len(o)), o.HFI, yerr=o.HFI_se, fmt='o', color='k', ms=4)
    ax[1].set_xticks(np.arange(len(o)))
    ax[1].set_xticklabels(o.index, fontsize=7)
    ax[1].axhline(0, color='0.6')
    ax2 = ax[1].twinx()
    sH = summ_s = pd.read_csv(f'{OUTS}/brune_sensitivity_sp.csv').set_index('fc0').loc[3.0, 'dHFI_dlnfc']
    yl = ax[1].get_ylim()
    ax2.set_ylim(np.array(yl) / sH)
    ax2.set_ylabel('implied dln fc (Brune, fc0=3 Hz)')
    ax[1].set_xlabel('family (sorted)')
    ax[1].set_ylabel('HFI = log10 A(4.9-6.1 Hz) - log10 A(2-3 Hz), relative')
    c = chi = summ['chi2_family_differences']['HFI']
    ax[1].set_title(f"HFI chi2 {c['chi2']:.0f}/{c['dof']} (p={c['p']:.1g}); intrinsic sd {c['sd_intrinsic']:.3f}", fontsize=9)
    sc = ax[2].scatter(T.along_km, T.depth_km, c=T.HFI, cmap='coolwarm', s=60 + 300 * (T.LO - T.LO.min()),
                       edgecolor='k', vmin=-max(abs(T.HFI)), vmax=max(abs(T.HFI)))
    for F in T.index:
        ax[2].annotate(str(F), (T.along_km[F], T.depth_km[F]), fontsize=7, xytext=(3, 3), textcoords='offset points')
    ax[2].invert_yaxis()
    ax[2].set_xlabel('along strike (km, + WNW, az 293)')
    ax[2].set_ylabel('Shelly depth (km)')
    ax[2].set_title('HFI (colour) and LO size (marker size)', fontsize=9)
    plt.colorbar(sc, ax=ax[2], label='HFI')
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_family_spectra_spectra{sfx}.png', dpi=130)
    plt.close(fig)
    # --- 2: validation
    pairs = [('D3_dtt_pb', 'D3_dtt_surf', 'borehole (PB)', 'surface'),
             ('D3_dtt_half1', 'D3_dtt_half2', 'even events', 'odd events'),
             ('D3_dtt_Z', 'D3_dtt_H', 'vertical', 'horizontal'),
             ('D3_dtt', 'D3_none', 'with dtt term', 'no path term'),
             ('D2_dtt', 'D99_dtt', 'D<2 km', 'all pairs'),
             ('D3_dtt', 'D3_dtt_D3d', 'D<3 km (Shelly)', 'D<3 km (ours 3D)')]
    fig, axs = plt.subplots(2, 6, figsize=(22, 7.5))
    for row, q in enumerate(['HFI', 'LO']):
        for k, (va, vb, la, lb) in enumerate(pairs):
            ax = axs[row, k]
            da = Rdf[(Rdf.variant == va) & (Rdf.stat == q)].set_index('family')
            db = Rdf[(Rdf.variant == vb) & (Rdf.stat == q)].set_index('family')
            xe = da.se if 'se' in da and da.se.notna().any() else None
            ye = db.se if 'se' in db and db.se.notna().any() else None
            ax.errorbar(da.x, db.x, xerr=xe, yerr=ye, fmt='o', ms=3, color='k', elinewidth=0.6)
            lim = np.nanmax(np.abs(np.r_[da.x, db.x])) * 1.15
            ax.plot([-lim, lim], [-lim, lim], 'r-', lw=0.7)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            vv = summ['validation'].get(f'{q}:{va}~{vb}', {})
            ax.set_title(f"{q}: r={vv.get('pearson', np.nan):.2f}, rho={vv.get('spearman', np.nan):.2f}", fontsize=9)
            ax.set_xlabel(la)
            ax.set_ylabel(lb)
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_family_spectra_validation{sfx}.png', dpi=120)
    plt.close(fig)
    # --- 3: scaling
    cvs = [('L1_m', 'patch L1 (m, resolved)'), ('logL_B', 'log10 sqrt(B_full std_s*std_d)'),
           ('LO', 'LO size (log10 amp, rel.)'), ('episode_recur_d', 'episode recurrence (d)'),
           ('slip_rel', 'slip proxy log10(A/area)'), ('depth_km', 'depth (km, Shelly)'),
           ('along_km', 'along strike (km)'), ('mean_tt_s', 'mean S travel time (s)')]
    fig, axs = plt.subplots(2, 4, figsize=(18, 8))
    for ax, (cv, lab) in zip(axs.flat, cvs):
        d = T[['HFI', 'HFI_se', cv]].dropna()
        ax.errorbar(d[cv], d.HFI, yerr=d.HFI_se, fmt='o', ms=4, color='k', elinewidth=0.6)
        for F in d.index:
            ax.annotate(str(F), (d.loc[F, cv], d.loc[F, 'HFI']), fontsize=6, xytext=(2, 2), textcoords='offset points')
        s = SC[(SC.target == 'HFI') & (SC.covariate == cv)]
        if len(s):
            s = s.iloc[0]
            ax.set_title(f"rho={s.spearman:.2f} p={s.p_perm:.3f} p_loc={s.p_local:.3f}\n"
                         f"partial={s.partial_rho:.2f} (p={s.p_partial:.3f})", fontsize=9)
        ax.set_xlabel(lab)
        ax.set_ylabel('HFI (rel. falloff)')
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_family_spectra_scaling{sfx}.png', dpi=120)
    plt.close(fig)
    log('figs done')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['sc', 'accum', 'solve', 'figs'])
    ap.add_argument('--procs', type=int, default=6)
    ap.add_argument('--fams', type=int, nargs='*')
    ap.add_argument('--mode', default='w', choices=['w', 'u'])
    a = ap.parse_args()
    {'sc': run_sc, 'accum': run_accum, 'solve': run_solve, 'figs': run_figs}[a.stage](a)
