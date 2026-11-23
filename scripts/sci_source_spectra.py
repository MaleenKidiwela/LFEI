#!/usr/bin/env python
"""Did the LFE sources change? Source spectra / S durations per family through time and across
the Petrolia (2021-12-20), Ferndale (2022-12-20) and M7 (2024-12-05) quakes.

Method (empirical-Green's-function style): every comparison is a ratio of two EVENT SETS of the SAME
family on the SAME channel (same path, site and instrument epoch), so path/site/instrument cancel.
Single LFEs are noisy (single-event S RMS SNR ~1), so each set is a spectral AVERAGE over many events
with the average noise power (pre-event windows of the same events) subtracted.

Stages
  extract : per family, per channel (template SNR >= 8, S pick): multitaper power in 12 log bands
            (1-15 Hz) of the S window (S-0.4 .. S+2.8 s) and of pre-event noise windows; 2-8 Hz energy
            envelopes (S-1 .. S+3 s, 0.08 s) for channels with template SNR >= 30.
            -> data/sci/source_spectra/raw/fam{F}.h5
  analyse : all tests (quake steps, restart, time series, bursts vs isolated, amplitude scaling),
            event-bootstrap errors, random-date placebos, permutation nulls, station-group splits.
            -> data/sci/source_spectra/*.csv, summary.json
  figs    : figures/sci_source_spectra_*.png
Quantities per comparison A/B, per channel c: r_c(f) = log10 P_A(f) - log10 P_B(f) (P = noise-corrected
mean S power); LO = mean r over 1.5-3 Hz (= 2 x log10 amplitude ratio), HFI = mean r over 6-12 Hz - LO
(spectral falloff change; >0 = relatively richer in high frequencies = higher corner / shorter source);
dD = change of the RMS width of the noise-corrected 2-8 Hz S energy envelope (s).
Channels combined by capped inverse-variance weights; families pooled by inverse placebo variance.
Instrument epochs (station metadata) are never mixed: every comparison on a channel uses one epoch.
Run: nice python scripts/sci_source_spectra.py extract --procs 4 ; ... analyse ; ... figs
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
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.signal.windows import dpss
from scipy.ndimage import uniform_filter1d

ROOT = '/home/seismic/LFE'
OUT = f'{ROOT}/data/sci/source_spectra'
FIG = f'{ROOT}/figures'
SR = 50.0
PRE = -15.0
NWIN = 160                      # 3.2 s
S_OFF = -0.4                    # S window start rel. S pick
NFFT = 256
EDGES = np.geomspace(1.0, 15.0, 13)
FC = np.sqrt(EDGES[:-1] * EDGES[1:])
NB = len(FC)
LO_B = np.where((FC >= 1.5) & (FC <= 3.0))[0]
MID_B = np.where((FC > 3.0) & (FC < 6.0))[0]
HI_B = np.where((FC >= 6.0) & (FC <= 12.0))[0]
ENV_SNR = 30.0
ENV_T0, ENV_DT, ENV_N = -1.0, 0.08, 50     # envelope samples rel. S
DUR_WIN = (-0.5, 2.0)                       # window for RMS duration rel. S
QUAKES = {'Petrolia': '2021-12-20T20:10', 'Ferndale': '2022-12-20T10:34', 'M7': '2024-12-05T18:44'}
FAMS = list(range(1, 28))
CLEAN_WIN = (-25.0, 20.0)       # no other catalog event (any family) in this window -> 'clean'


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def epoch_s(x):
    x = pd.DatetimeIndex(pd.to_datetime(x))
    return np.asarray((x - pd.Timestamp('1970-01-01')) / pd.Timedelta(seconds=1), dtype=float)


# ----------------------------------------------------------------------------------------- extract
def band_matrix():
    f = np.fft.rfftfreq(NFFT, 1 / SR)
    M = np.zeros((len(f), NB))
    for b in range(NB):
        k = (f >= EDGES[b]) & (f < EDGES[b + 1])
        M[k, b] = 1.0 / k.sum()
    return M


def mt_band_power(W, tapers, M):
    """W (n, NWIN) -> band-averaged multitaper power (n, NB)."""
    W = W - W.mean(1, keepdims=True)
    P = 0.0
    for tp in tapers:
        P = P + np.abs(np.fft.rfft(W * tp[None, :], NFFT, axis=1)) ** 2
    return (P / len(tapers)) @ M


def extract(F):
    fn = f'{OUT}/raw/fam{F}.h5'
    if os.path.exists(fn):
        log('exists', fn)
        return
    os.makedirs(f'{OUT}/raw', exist_ok=True)
    h5f = f'{ROOT}/data/waveforms/fam{F}.h5'
    T = h5py.File(f'{ROOT}/data/templates/fam{F}_templates.h5', 'r')
    picks = pd.read_csv(f'{ROOT}/data/locations/fam{F}_3d_picks.csv').set_index('sta')
    origin = json.load(open(f'{ROOT}/data/locations/fam{F}_3d.json'))['origin_rel_catalog_s']
    tapers = dpss(NWIN, 2.5, 4)
    M = band_matrix()
    sos = butter(4, [2.0, 8.0], btype='band', fs=SR, output='sos')
    noise_end = min(origin + 2.0, -6.0)
    nstarts = []
    t0 = -14.8
    while t0 + NWIN / SR <= noise_end + 1e-9:
        nstarts.append(int(round((t0 - PRE) * SR)))
        t0 += NWIN / SR
    with h5py.File(h5f, 'r') as h:
        n_ev = len(h['events/time_utc'])
        times = h['events/time_utc'][:]
        chans = []
        for cid in sorted(h['chan']):
            if cid not in T:
                continue
            sta = '.'.join(cid.split('.')[:2])
            snr = float(T[cid].attrs['snr'])
            if snr < 8 or sta not in picks.index or not np.isfinite(picks.loc[sta, 'S_final']):
                continue
            S = float(picks.loc[sta, 'S_final'])
            if S + S_OFF + NWIN / SR > 44.0 or S + ENV_T0 < -6:
                continue
            chans.append((cid, snr, S))
        nch = len(chans)
        env_ch = [k for k, c in enumerate(chans) if c[1] >= ENV_SNR]
        SP = np.full((n_ev, nch, NB), np.nan, np.float16)
        NP = np.full((n_ev, nch, NB), np.nan, np.float16)
        ENV = np.full((n_ev, len(env_ch), ENV_N), np.nan, np.float16)
        ENVN = np.full((n_ev, len(env_ch)), np.nan, np.float32)
        envscale = np.full(len(env_ch), np.nan)
        for k, (cid, snr, S) in enumerate(chans):
            ev = h[f'chan/{cid}/ev'][:]
            X = h[f'chan/{cid}/data'][:]
            i0 = int(round((S + S_OFF - PRE) * SR))
            ps = mt_band_power(X[:, i0:i0 + NWIN].astype(np.float64), tapers, M)
            pn = np.mean([mt_band_power(X[:, j:j + NWIN].astype(np.float64), tapers, M) for j in nstarts], 0)
            # store log10 power, float16 (quantisation ~0.1-0.5 %, random)
            sc = np.nanmedian(pn, 0)
            SP[ev, k] = np.log10(np.maximum(ps / sc, 1e-30)).astype(np.float16)
            NP[ev, k] = np.log10(np.maximum(pn / sc, 1e-30)).astype(np.float16)
            if k in env_ch:
                j = env_ch.index(k)
                for a in range(0, len(ev), 2000):
                    Y = sosfiltfilt(sos, X[a:a + 2000].astype(np.float64), axis=1)
                    E = np.abs(hilbert(Y, axis=1)) ** 2
                    E = uniform_filter1d(E, 5, axis=1)
                    ts = S + ENV_T0 + ENV_DT * np.arange(ENV_N)
                    idx = np.round((ts - PRE) * SR).astype(int)
                    nm = np.zeros(len(Y))
                    for jj in nstarts:
                        nm += E[:, jj:jj + NWIN].mean(1)
                    nm /= len(nstarts)
                    if np.isnan(envscale[j]):
                        envscale[j] = np.median(nm)
                    ENV[ev[a:a + 2000], j] = np.log10(np.maximum(E[:, idx] / envscale[j], 1e-30)).astype(np.float16)
                    ENVN[ev[a:a + 2000], j] = nm / envscale[j]
            del X
    T.close()
    with h5py.File(fn + '.tmp', 'w') as o:
        o['SP'] = SP
        o['NP'] = NP
        o['ENV'] = ENV
        o['ENVN'] = ENVN
        o['time_utc'] = times
        o['chan'] = np.array([c[0] for c in chans], dtype='S20')
        o['snr'] = np.array([c[1] for c in chans])
        o['S'] = np.array([c[2] for c in chans])
        o['env_ch'] = np.array(env_ch, dtype=int)
        o.attrs.update(dict(edges=EDGES, noise_starts=nstarts, s_off=S_OFF, nwin=NWIN, env_t0=ENV_T0,
                            env_dt=ENV_DT, origin=origin))
    os.replace(fn + '.tmp', fn)
    log(f'fam {F}: {n_ev} events, {nch} channels ({len(env_ch)} with envelopes), noise windows {len(nstarts)}')


def run_extract(a):
    from multiprocessing import Pool
    fams = a.fams or FAMS
    # biggest first for load balance
    sizes = {F: os.path.getsize(f'{ROOT}/data/waveforms/fam{F}.h5') for F in fams}
    fams = sorted(fams, key=lambda F: -sizes[F])
    with Pool(a.procs) as p:
        list(p.imap_unordered(extract, fams))




# ------------------------------------------------------------------------------- extract_mf
MF_EDGES = np.array([1.25, 2.0, 3.0, 4.5, 6.5, 9.0, 13.0])
MF_FC = np.sqrt(MF_EDGES[:-1] * MF_EDGES[1:])
MF_WIN = (-0.5, 2.5)      # rel. S


def extract_mf(F):
    """Band-limited matched-filter amplitudes: per channel and band, project each event's band-passed
    S window onto the channel's band-passed stack of clean events (traces RMS-normalised for the stack
    only). Noise-unbiased (zero-mean noise), far higher SNR than incoherent power. Same for the two
    pre-event noise windows (error estimate)."""
    fn = f'{OUT}/raw/fam{F}_mf.h5'
    if os.path.exists(fn):
        log('exists', fn)
        return
    h5f = f'{ROOT}/data/waveforms/fam{F}.h5'
    T = h5py.File(f'{ROOT}/data/templates/fam{F}_templates.h5', 'r')
    picks = pd.read_csv(f'{ROOT}/data/locations/fam{F}_3d_picks.csv').set_index('sta')
    origin = json.load(open(f'{ROOT}/data/locations/fam{F}_3d.json'))['origin_rel_catalog_s']
    nw = int(round((MF_WIN[1] - MF_WIN[0]) * SR))
    noise_end = min(origin + 2.0, -6.0)
    nstarts, t0 = [], -14.8
    while t0 + nw / SR <= noise_end + 1e-9:
        nstarts.append(int(round((t0 - PRE) * SR)))
        t0 += nw / SR
    nstarts = nstarts[:2]
    cat = load_catalog()
    soss = [butter(4, [MF_EDGES[b], MF_EDGES[b + 1]], btype='band', fs=SR, output='sos') for b in range(len(MF_FC))]
    nbm = len(MF_FC)
    with h5py.File(h5f, 'r') as h:
        times = h['events/time_utc'][:]
        t = epoch_s(times.astype(str))
        clean = event_flags(t, F, cat)[0]
        n_ev = len(t)
        chans = []
        for cid in sorted(h['chan']):
            if cid not in T:
                continue
            sta = '.'.join(cid.split('.')[:2])
            snr = float(T[cid].attrs['snr'])
            if snr < 8 or sta not in picks.index or not np.isfinite(picks.loc[sta, 'S_final']):
                continue
            S = float(picks.loc[sta, 'S_final'])
            if S + MF_WIN[1] > 44.0:
                continue
            chans.append((cid, snr, S))
        nch = len(chans)
        AM = np.full((n_ev, nch, nbm), np.nan, np.float32)
        AN = np.full((n_ev, nch, nbm, len(nstarts)), np.nan, np.float32)
        for k, (cid, snr, S) in enumerate(chans):
            ev = h[f'chan/{cid}/ev'][:]
            X = h[f'chan/{cid}/data'][:]
            i0 = int(round((S + MF_WIN[0] - PRE) * SR))
            for b, sos in enumerate(soss):
                W = np.empty((len(ev), nw))
                Nw = np.empty((len(ev), len(nstarts), nw))
                for a in range(0, len(ev), 2000):
                    Y = sosfiltfilt(sos, X[a:a + 2000].astype(np.float64), axis=1)
                    W[a:a + 2000] = Y[:, i0:i0 + nw]
                    for j, js in enumerate(nstarts):
                        Nw[a:a + 2000, j] = Y[:, js:js + nw]
                nr = np.sqrt((Nw ** 2).mean((1, 2))) + 1e-20
                cl = clean[ev]
                if cl.sum() < 30:
                    cl = np.ones(len(ev), bool)
                tmpl = (W[cl] / nr[cl, None]).mean(0)
                tt = tmpl @ tmpl + 1e-30
                AM[ev, k, b] = (W @ tmpl) / tt
                AN[ev, k, b] = np.einsum('ejt,t->ej', Nw, tmpl) / tt
            del X
    T.close()
    with h5py.File(fn + '.tmp', 'w') as o:
        o['AM'] = AM
        o['AN'] = AN
        o['chan'] = np.array([c[0] for c in chans], dtype='S20')
        o['snr'] = np.array([c[1] for c in chans])
        o.attrs.update(dict(edges=MF_EDGES, win=MF_WIN, noise_starts=nstarts))
    os.replace(fn + '.tmp', fn)
    log(f'fam {F} mf: {n_ev} events, {nch} channels')


def run_extract_mf(a):
    from multiprocessing import Pool
    fams = a.fams or FAMS
    sizes = {F: os.path.getsize(f'{ROOT}/data/waveforms/fam{F}.h5') for F in fams}
    fams = sorted(fams, key=lambda F: -sizes[F])
    with Pool(a.procs) as p:
        list(p.imap_unordered(extract_mf, fams))


# ----------------------------------------------------------------------------------------- analysis
DAY = 86400.0
QT = {k: epoch_s([v])[0] for k, v in QUAKES.items()}
NMIN = 8            # min effective events per set per unit
ZMIN = 3.0          # min expected significance (mean MF amplitude / its expected SE) per set and band
WCAP = 5.0          # cap unit weights at WCAP x median


def meta_epochs():
    c = pd.read_csv(f'{ROOT}/data/stations/candidate_channels_65km_2008_2026.csv', dtype={'loc': str})
    c['loc'] = c['loc'].fillna('')
    c['cid'] = c.net + '.' + c.sta + '.' + c['loc'] + '.' + c.cha
    out = {}
    for cid, g in c.groupby('cid'):
        st = np.sort(epoch_s(g.start.str.slice(0, 19)))
        out[cid] = st[1:]          # boundaries (interior epoch starts)
    return out


def load_catalog():
    cat = pd.read_csv(f'{ROOT}/data/merged/LFEcat_MTJ_merged_v1v2.csv', usecols=['time_utc', 'idnum', 'source'])
    cat = cat[cat.source != 'v1_only'].copy()
    cat['t'] = epoch_s(cat.time_utc)
    return cat.sort_values('t').reset_index(drop=True)


def event_flags(t, F, cat):
    """clean: no other catalog event (any family) within CLEAN_WIN; burst / isolated (same family)."""
    allt = cat.t.values
    n_win = np.searchsorted(allt, t + CLEAN_WIN[1], 'right') - np.searchsorted(allt, t + CLEAN_WIN[0], 'left')
    clean = n_win == 1
    ft = np.sort(cat.t.values[cat.idnum.values == F])
    def nbr(lo, hi):     # same-family events with |dt| in (lo, hi]
        return (np.searchsorted(ft, t + hi, 'right') - np.searchsorted(ft, t + lo, 'right')
                + np.searchsorted(ft, t - lo, 'left') - np.searchsorted(ft, t - hi, 'left'))
    burst = clean & (nbr(15.0, 1800.0) > 0)
    isolated = clean & (nbr(0.0, 3 * 3600.0) == 0)
    return clean, burst, isolated


def rolling_med(x, w=201):
    return pd.DataFrame(x).rolling(w, center=True, min_periods=20).median().values


def load_family(F, cat, epochs):
    """Units = (channel, instrument/noise segment). Spectral data = MF band amplitudes (raw/fam{F}_mf.h5);
    envelopes from raw/fam{F}.h5."""
    with h5py.File(f'{OUT}/raw/fam{F}_mf.h5', 'r') as h:
        AM = h['AM'][:]
        AN = h['AN'][:]
        chans = h['chan'][:].astype(str)
        snr = h['snr'][:]
    with h5py.File(f'{OUT}/raw/fam{F}.h5', 'r') as h:
        ENVl = h['ENV'][:].astype(np.float32)
        ENVN = h['ENVN'][:]
        pchans = list(h['chan'][:].astype(str))
        env_ch = list(h['env_ch'][:])
        t = epoch_s(h['time_utc'][:].astype(str))
    n, nc, nb = AM.shape
    clean, burst, isolated = event_flags(t, F, cat)
    units, U_ev, seg_info = [], [], []
    for c in range(nc):
        ii = np.where(np.isfinite(AM[:, c, 0]) & np.isfinite(AN[:, c, 0, 0]))[0]
        if len(ii) < 50:
            continue
        ii = ii[np.argsort(t[ii])]
        # noise level per event (log rms of noise projections over bands 1-3)
        lvl = 0.5 * np.log10(np.mean(AN[ii, c, 1:4, :] ** 2, axis=(1, 2)) + 1e-30)
        # segments: metadata epochs + data-driven noise-level change points (>3x in noise amplitude)
        bnds = list(epochs.get(chans[c], []))
        mon = pd.Series(lvl, index=pd.to_datetime(t[ii], unit='s')).resample('MS').median().dropna()
        if len(mon) > 10:
            v, tm = mon.values, mon.index
            dif = np.array([abs(np.median(v[max(0, k - 4):k]) - np.median(v[k:k + 4])) if 2 <= k <= len(v) - 2
                            else 0 for k in range(len(v))])
            for k in range(len(v)):
                if dif[k] > 0.5 and dif[k] == dif[max(0, k - 3):k + 4].max():
                    bnds.append(epoch_s([tm[k]])[0])
        bnds = np.sort(np.array(bnds))
        sid = np.searchsorted(bnds, t[ii])
        for sg in np.unique(sid):
            jj = ii[sid == sg]
            if len(jj) < 30:
                continue
            # event outliers (per unit): noise glitch (any |noise proj| > 6 robust sd) or S-window
            # amplitude beyond 12 MAD (earthquakes, glitches)
            nz = AN[jj, c]                               # (m, nb, 2)
            sdn = 1.4826 * np.median(np.abs(nz), axis=(0, 2))
            am = AM[jj, c]
            med = np.median(am, 0)
            mad = 1.4826 * np.median(np.abs(am - med), 0)
            bad = (np.abs(nz) > 6 * sdn[None, :, None]).any((1, 2)) | (np.abs(am - med) > 12 * mad).any(1)
            jj = jj[~bad]
            if len(jj) < 30:
                continue
            units.append(c)
            U_ev.append(jj)
            seg_info.append((chans[c], int(sg), pd.to_datetime(t[jj].min(), unit='s').date(),
                             pd.to_datetime(t[jj].max(), unit='s').date(), len(jj)))
    nu = len(units)
    S0 = np.zeros((n, nu, nb), np.float32)
    N2 = np.zeros((n, nu, nb), np.float32)
    V = np.zeros((n, nu), np.float32)
    for u, (c, jj) in enumerate(zip(units, U_ev)):
        S0[jj, u] = AM[jj, c]
        N2[jj, u] = np.mean(AN[jj, c] ** 2, -1)
        V[jj, u] = 1
    cn = chans[np.array(units, dtype=int)]
    eu, ej = [], []
    for u, c in enumerate(cn):
        if c in pchans and pchans.index(c) in env_ch:
            eu.append(u)
            ej.append(env_ch.index(pchans.index(c)))
    E0 = np.zeros((n, len(eu), ENV_N), np.float32)
    EN = np.zeros((n, len(eu)), np.float32)
    VE = np.zeros((n, len(eu)), np.float32)
    for k, (u, j) in enumerate(zip(eu, ej)):
        jj = U_ev[u]
        okk = jj[np.isfinite(ENVN[jj, j]) & np.isfinite(ENVl[jj, j]).all(1)]
        E0[okk, k] = 10 ** ENVl[okk, j]
        EN[okk, k] = ENVN[okk, j]
        VE[okk, k] = 1
    fd = dict(F=F, t=t, n=n, S0=S0, N2=N2, V=V, E0=E0, EN=EN, VE=VE, unit_chan=cn,
              is_pb=np.array([c.startswith('PB.') for c in cn]), env_units=np.array(eu, dtype=int),
              env_snr=snr[np.array(units, dtype=int)[eu]] if eu else np.zeros(0))
    ic = np.where(clean)[0]
    st0 = set_stats(fd, np.ones(n), ic, env=False)
    fd['Pall'] = st0['S'][0]
    fd['v1'] = st0['var'] * st0['neff'][:, None]          # per-event variance of the MF amplitude
    if not hasattr(load_family, 'A'):
        load_family.A = pd.read_csv(f'{ROOT}/data/sci/moments/event_amplitudes.csv.gz')
    A = load_family.A
    am = A[A.fam == F].set_index('ev').reindex(range(n))
    fd.update(clean=clean, burst=burst, isolated=isolated,
              A=am.A.values, A_pb=am.A_pb.values, A_nc=am.A_nc.values, above_T=am.above_T.fillna(False).values.astype(bool),
              qc_bad=am.qc_bad.fillna(True).values.astype(bool), seg_info=seg_info,
              ft=np.sort(cat.t.values[cat.idnum.values == F]))
    return fd


TE = ENV_T0 + ENV_DT * np.arange(ENV_N)
DW = (TE >= DUR_WIN[0]) & (TE <= DUR_WIN[1])
# MF bands: 0 1.25-2, 1 2-3, 2 3-4.5, 3 4.5-6.5, 4 6.5-9, 5 9-13 Hz.  Indices in log10 AMPLITUDE ratio.
B_LO, B_MID, B_HI, B_HF2 = 1, 2, 3, 4


def _means(Cw, X, Vm):
    """Cw (k, ni) weights; X (ni, nu, nb) zero-filled; Vm (ni, nu) -> mean (k, nu, nb), sw (k, nu)."""
    k, ni = Cw.shape
    sw = Cw @ Vm
    num = (Cw @ X.reshape(ni, -1)).reshape(k, X.shape[1], X.shape[2])
    return num / np.maximum(sw, 1e-12)[:, :, None], sw


def set_stats(fd, w, idx, counts=None, env=True, key='S0'):
    ww = w[idx].astype(np.float32)
    Cw = ww[None, :] if counts is None else counts.astype(np.float32) * ww[None, :]
    Vm = fd['V'][idx]
    X = fd[key][idx]
    S, sw = _means(Cw, X, Vm)
    out = dict(S=S, sw=sw)
    if counts is None:
        W = ww[:, None] * Vm
        out['neff'] = W.sum(0) ** 2 / np.maximum((W ** 2).sum(0), 1e-12)
        D = (X - S[0][None]) * Vm[:, :, None]
        out['var'] = np.einsum('iu,iub->ub', W ** 2, D ** 2) / np.maximum(W.sum(0), 1e-12)[:, None] ** 2
    if env and len(fd['env_units']):
        VE = fd['VE'][idx]
        E, swe = _means(Cw, fd['E0'][idx], VE)
        EN = (Cw @ (fd['EN'][idx] * VE)) / np.maximum(swe, 1e-12)
        out.update(E=E - EN[:, :, None], swe=swe)
    return out


def rms_dur(Es):
    """Es (..., ENV_N) noise-corrected envelopes -> RMS width (s) in DUR_WIN."""
    x = np.clip(Es[..., DW], 0, None)
    tt = TE[DW]
    m0 = x.sum(-1)
    tc = (x * tt).sum(-1) / np.maximum(m0, 1e-12)
    return np.sqrt((x * (tt - tc[..., None]) ** 2).sum(-1) / np.maximum(m0, 1e-12))


def idx_stats(rr):
    lo = rr[..., B_LO]
    return dict(LO=lo, MID=rr[..., B_MID] - lo, HFI=rr[..., B_HI] - lo, HF2=rr[..., B_HF2] - lo)


QS = ['LO', 'MID', 'HFI', 'HF2']


def compare(fd, wA, wB, nboot=0, rng=None, noise=False, bcache=None):
    """Compare event sets A and B (weights over all events). Spectral ratio per unit and MF band:
    r = log10(mean amp_A / mean amp_B) (noise: log10 rms of noise projections). Returns pooled indices
    for unit groups all / pb / surf, bootstrap SEs, unit-level values, and envelope RMS-duration change."""
    iA, iB = np.where(wA > 0)[0], np.where(wB > 0)[0]
    if len(iA) < NMIN or len(iB) < NMIN:
        return None
    key = 'N2' if noise else 'S0'
    a = set_stats(fd, wA, iA, key=key, env=not noise)
    if bcache is not None and not noise:
        if 'b' not in bcache:
            bcache['b'] = set_stats(fd, wB, iB)
        b = bcache['b']
    else:
        b = set_stats(fd, wB, iB, key=key, env=not noise)
    ok = (a['neff'] >= NMIN) & (b['neff'] >= NMIN)
    if noise:
        PA, PB = np.sqrt(np.maximum(a['S'][0], 1e-30)), np.sqrt(np.maximum(b['S'][0], 1e-30))
        okb = np.broadcast_to(ok[:, None], PA.shape).copy()
        varr = np.full(PA.shape, 1e-3)
    else:
        # validity from the unit's all-clean-event amplitude and the expected error for each set size
        # (independent of the set's own fluctuation -> no selection bias)
        Pall, v1 = fd['Pall'], fd['v1']
        okb = ((Pall / np.sqrt(v1 / np.maximum(a['neff'], 1)[:, None]) >= ZMIN)
               & (Pall / np.sqrt(v1 / np.maximum(b['neff'], 1)[:, None]) >= ZMIN) & ok[:, None])
        PA = np.maximum(a['S'][0], 0.1 * np.abs(Pall))
        PB = np.maximum(b['S'][0], 0.1 * np.abs(Pall))
        varr = (a['var'] / PA ** 2 + b['var'] / PB ** 2) / np.log(10) ** 2
    r = np.where(okb, np.log10(PA) - np.log10(PB), np.nan)
    st = idx_stats(r)
    vq = dict(LO=varr[:, B_LO], MID=varr[:, B_MID] + varr[:, B_LO], HFI=varr[:, B_HI] + varr[:, B_LO],
              HF2=varr[:, B_HF2] + varr[:, B_LO])
    groups = dict(all=np.ones(len(ok), bool), pb=fd['is_pb'], surf=~fd['is_pb'])
    res = dict(n_A=len(iA), n_B=len(iB), neffA=float(wA.sum() ** 2 / (wA ** 2).sum()),
               neffB=float(wB.sum() ** 2 / (wB ** 2).sum()))
    gw = {}
    for q in QS:
        fin = np.isfinite(st[q])
        w = np.where(fin, 1 / np.maximum(vq[q], 1e-8), 0.0)
        for g, m in groups.items():
            sel = fin & m
            if sel.sum() == 0:
                continue
            wg = np.where(sel, w, 0.0)
            wg = np.minimum(wg, WCAP * np.median(wg[sel]))
            wg = wg / wg.sum()
            gw[(q, g)] = wg
            res[f'{q}_{g}'] = float(np.sum(wg[sel] * st[q][sel]))
            res[f'nunit_{q}_{g}'] = int(sel.sum())
            res[f'fse_{q}_{g}'] = float(np.sqrt(np.sum(wg[sel] ** 2 * vq[q][sel])))
            res[f'chi2_{q}_{g}'] = float(np.sum((st[q][sel] - res[f'{q}_{g}']) ** 2 / vq[q][sel]) / max(sel.sum() - 1, 1))
    for g, m in groups.items():       # pooled band curve (weights = inverse band variance)
        wb = np.where(np.isfinite(r) & m[:, None], 1 / np.maximum(varr, 1e-8), 0)
        wb = np.minimum(wb, WCAP * np.nanmedian(np.where(wb > 0, wb, np.nan), 0)[None])
        res[f'curve_{g}'] = (np.nansum(np.nan_to_num(r) * wb, 0) / np.maximum(wb.sum(0), 1e-12)).tolist()
    res['unit_HFI'], res['unit_MID'], res['unit_LO'] = st['HFI'], st['MID'], st['LO']
    res['unit_vHFI'], res['unit_vMID'], res['unit_vLO'] = vq['HFI'], vq['MID'], vq['LO']
    okd = None
    if not noise and len(fd['env_units']):
        DA, DB = rms_dur(a['E'][0]), rms_dur(b['E'][0])
        eA = np.clip(a['E'][0][:, DW], 0, None).sum(1)
        okd = (a['swe'][0] >= NMIN) & (b['swe'][0] >= NMIN) & (eA > 0)
        wd = np.minimum(fd['env_snr'], 100.0) ** 2 / (1 / np.maximum(a['swe'][0], 1) + 1 / np.maximum(b['swe'][0], 1))
        isp = fd['is_pb'][fd['env_units']]
        for g, m in dict(all=np.ones(len(okd), bool), pb=isp, surf=~isp).items():
            sl = okd & m
            if sl.sum():
                res[f'dD_{g}'] = float(np.sum(wd[sl] * (DA - DB)[sl]) / wd[sl].sum())
                res[f'D_B_{g}'] = float(np.sum(wd[sl] * DB[sl]) / wd[sl].sum())
        res['unit_dD'] = np.where(okd, DA - DB, np.nan)
    if nboot and rng is not None:
        bt = {}
        cA = rng.multinomial(len(iA), np.ones(len(iA)) / len(iA), size=nboot)
        cB = rng.multinomial(len(iB), np.ones(len(iB)) / len(iB), size=nboot)
        for k0 in range(0, nboot, 50):
            ab = set_stats(fd, wA, iA, cA[k0:k0 + 50], key=key, env=not noise)
            bb = set_stats(fd, wB, iB, cB[k0:k0 + 50], key=key, env=not noise)
            if noise:
                PAb, PBb = np.sqrt(np.maximum(ab['S'], 1e-30)), np.sqrt(np.maximum(bb['S'], 1e-30))
            else:
                PAb = np.maximum(ab['S'], 0.1 * np.abs(fd['Pall'])[None])
                PBb = np.maximum(bb['S'], 0.1 * np.abs(fd['Pall'])[None])
            rb_ = np.where(okb[None], np.log10(PAb) - np.log10(PBb), np.nan)
            sb = idx_stats(rb_)
            for (q, g), wg in gw.items():
                fin = np.isfinite(sb[q])
                bt.setdefault(f'{q}_{g}', []).extend(
                    np.sum(np.where(fin, wg[None] * sb[q], 0), 1) / np.maximum((wg[None] * fin).sum(1), 1e-12))
            if okd is not None:
                DAb, DBb = rms_dur(ab['E']), rms_dur(bb['E'])
                isp = fd['is_pb'][fd['env_units']]
                for g, m in dict(all=np.ones(len(okd), bool), pb=isp, surf=~isp).items():
                    sl = okd & m
                    if sl.sum():
                        bt.setdefault(f'dD_{g}', []).extend(((DAb - DBb)[:, sl] * wd[sl]).sum(1) / wd[sl].sum())
        for k, v in bt.items():
            res[f'se_{k}'] = float(np.std(v))
    return res


def amp_match_weights(fd, selA, selB, extra=None, nbins=8):
    """Weights for B so its (log amplitude [x extra stratum]) distribution matches A's."""
    la = np.log10(np.clip(np.nan_to_num(fd['A'], nan=1.0), 0.05, None))
    both = selA | selB
    q = np.quantile(la[both], np.linspace(0, 1, nbins + 1)[1:-1])
    k = np.searchsorted(q, la)
    if extra is not None:
        k = k * 100 + extra
    wB = np.zeros(fd['n'])
    for v in np.unique(k[selB]):
        nA, nB = (selA & (k == v)).sum(), (selB & (k == v)).sum()
        wB[selB & (k == v)] = nA / nB
    return wB


SCALARS = ['LO', 'HFI', 'MID', 'dD']
GROUPS = ['all', 'pb', 'surf']


def flat(res, prefix=''):
    if res is None:
        return {}
    out = {}
    for k, v in res.items():
        if isinstance(v, (int, float, np.floating, np.integer)):
            out[prefix + k] = float(v)
    return out


def placebo_dates(n=200, seed=11):
    rng = np.random.default_rng(seed)
    lo, hi = epoch_s(['2009-07-01'])[0], epoch_s(['2021-06-01'])[0]
    return np.sort(rng.uniform(lo, hi, n))


def first_after(ft, T0):
    k = np.searchsorted(ft, T0, 'right')
    return ft[k] if k < len(ft) else np.nan


def step_sets(fd, Q, W=180.0, kind='step', win=30.0, base=365.0, sel=None):
    t = fd['t']
    sel = fd['clean'] if sel is None else sel
    if kind == 'step':
        A = sel & (t >= Q) & (t < Q + W * DAY)
        B = sel & (t >= Q - W * DAY) & (t < Q)
    else:   # restart: first `win` days after the first family event after Q vs `base` days before Q
        r0 = first_after(fd['ft'], Q)
        A = sel & (t >= r0) & (t < r0 + win * DAY)
        B = sel & (t >= Q - base * DAY) & (t < Q)
    return A, B


def analyse_family(F):
    t0 = time.time()
    cat = load_catalog()
    fd = load_family(F, cat, meta_epochs())
    rng = np.random.default_rng(100 + F)
    NBOOT = 100
    out = dict(F=F, units=len(fd['unit_chan']), n_clean=int(fd['clean'].sum()), n=fd['n'],
               seg_info=fd['seg_info'])
    t = fd['t']
    rows = []

    def add(test, variant, res, extra=None):
        if res is None:
            return
        d = dict(F=F, test=test, variant=variant, **flat(res))
        for g in GROUPS:
            if f'curve_{g}' in res:
                d[f'curve_{g}'] = json.dumps(np.round(res[f'curve_{g}'], 4).tolist())
        if extra:
            d.update(extra)
        rows.append(d)

    # 1) quake steps (+-180 d) and restart windows
    for qn, Q in QT.items():
        A, B = step_sets(fd, Q)
        add('step', 'raw', compare(fd, A * 1.0, B * 1.0, NBOOT, rng), dict(quake=qn))
        add('step', 'amp', compare(fd, A * 1.0, amp_match_weights(fd, A, B), NBOOT, rng), dict(quake=qn))
        add('step', 'noise', compare(fd, A * 1.0, B * 1.0, NBOOT, rng, noise=True), dict(quake=qn))
        AT, BT = A & fd['above_T'], B & fd['above_T']
        add('step', 'aboveT', compare(fd, AT * 1.0, BT * 1.0, NBOOT, rng), dict(quake=qn))
        for win in (14.0, 30.0, 60.0):
            A, B = step_sets(fd, Q, kind='restart', win=win)
            add(f'restart{int(win)}', 'raw', compare(fd, A * 1.0, B * 1.0, NBOOT, rng), dict(quake=qn))
            add(f'restart{int(win)}', 'amp', compare(fd, A * 1.0, amp_match_weights(fd, A, B), NBOOT, rng),
                dict(quake=qn))
            if win == 30.0:
                add('restart30', 'noise', compare(fd, A * 1.0, B * 1.0, NBOOT, rng, noise=True), dict(quake=qn))
                A2, B2 = step_sets(fd, Q, kind='restart', win=win, sel=np.ones(fd['n'], bool))
                add('restart30', 'allev', compare(fd, A2 * 1.0, B2 * 1.0, NBOOT, rng), dict(quake=qn))
                # odd/even independent halves
                ev_ = np.arange(fd['n']) % 2 == 0
                add('restart30', 'even', compare(fd, (A & ev_) * 1.0, (B & ev_) * 1.0, 0, rng), dict(quake=qn))
                add('restart30', 'odd', compare(fd, (A & ~ev_) * 1.0, (B & ~ev_) * 1.0, 0, rng), dict(quake=qn))
        # later post-quake period (60-240 d after the first event)
        r0 = first_after(fd['ft'], Q)
        A = fd['clean'] & (t >= r0 + 60 * DAY) & (t < r0 + 240 * DAY)
        B = fd['clean'] & (t >= Q - 365 * DAY) & (t < Q)
        add('late60_240', 'raw', compare(fd, A * 1.0, B * 1.0, NBOOT, rng), dict(quake=qn))
    log(f'fam {F}: quake tests done ({time.time() - t0:.0f} s)')

    # 2) placebos (point estimates)
    prow = []
    for k, D in enumerate(placebo_dates()):
        for test, kw in (('step', dict(kind='step')), ('restart14', dict(kind='restart', win=14.0)),
                         ('restart30', dict(kind='restart', win=30.0)), ('restart60', dict(kind='restart', win=60.0))):
            A, B = step_sets(fd, D, **kw)
            for variant in ('raw', 'amp'):
                wB = B * 1.0 if variant == 'raw' else amp_match_weights(fd, A, B)
                r = compare(fd, A * 1.0, wB)
                if r is not None:
                    prow.append(dict(F=F, k=k, date=D, test=test, variant=variant, **flat(r)))
            if test in ('step', 'restart30'):
                r = compare(fd, A * 1.0, B * 1.0, noise=True)
                if r is not None:
                    prow.append(dict(F=F, k=k, date=D, test=test, variant='noise', **flat(r)))
    log(f'fam {F}: placebos done ({time.time() - t0:.0f} s)')

    # 3) time series: quarterly bins vs the unit's own (segment) all-clean reference
    trow = []
    bc = {}
    qb = pd.date_range('2008-01-01', '2026-07-01', freq='QS')
    qbs = epoch_s(qb)
    for k in range(len(qbs) - 1):
        A = fd['clean'] & (t >= qbs[k]) & (t < qbs[k + 1])
        r = compare(fd, A * 1.0, fd['clean'] * 1.0, bcache=bc)
        if r is None:
            continue
        trow.append(dict(F=F, bin=k, t0=str(qb[k].date()), tmid=(qbs[k] + qbs[k + 1]) / 2, **flat(r),
                         unit_HFI=json.dumps(np.round(r['unit_HFI'], 4).tolist()),
                         unit_vHFI=json.dumps(np.round(r['unit_vHFI'], 6).tolist()),
                         unit_MID=json.dumps(np.round(r['unit_MID'], 4).tolist()),
                         unit_vMID=json.dumps(np.round(r['unit_vMID'], 6).tolist()),
                         unit_LO=json.dumps(np.round(r['unit_LO'], 4).tolist()),
                         unit_vLO=json.dumps(np.round(r['unit_vLO'], 6).tolist()),
                         unit_dD=json.dumps(np.round(r.get('unit_dD', np.array([])), 4).tolist()),
                         curve_all=json.dumps(np.round(r['curve_all'], 4).tolist()) if 'curve_all' in r else ''))
    out['unit_chan'] = list(fd['unit_chan'])
    out['unit_is_pb'] = fd['is_pb'].tolist()
    out['env_units'] = fd['env_units'].tolist()
    out['env_wd'] = (np.minimum(fd['env_snr'], 100.0) ** 2).tolist()
    log(f'fam {F}: time series done ({time.time() - t0:.0f} s)')

    # 4) bursts vs isolated (amplitude x year matched), permutation null
    yr = pd.to_datetime(t, unit='s').year.values - 2000
    A, B = fd['burst'], fd['isolated']
    add('burst', 'amp_year', compare(fd, A * 1.0, amp_match_weights(fd, A, B, extra=yr), NBOOT, rng))
    add('burst', 'raw', compare(fd, A * 1.0, B * 1.0, NBOOT, rng))
    U = A | B
    for k in range(100):
        lab = np.zeros(fd['n'], bool)
        for y in np.unique(yr[U]):        # permute labels within year
            ii = np.where(U & (yr == y))[0]
            nA = (A & (yr == y)).sum()
            lab[rng.choice(ii, nA, replace=False)] = True
        Ap, Bp = lab, U & ~lab
        r = compare(fd, Ap * 1.0, amp_match_weights(fd, Ap, Bp, extra=yr))
        if r is not None:
            prow.append(dict(F=F, k=k, date=np.nan, test='burst', variant='perm', **flat(r)))
    log(f'fam {F}: bursts done ({time.time() - t0:.0f} s)')

    # 5) amplitude scaling: classes from PB amplitudes measured on surface units, and vice versa
    base = fd['clean'] & ~fd['qc_bad']
    # own-band per-event amplitudes from one station group (selection) -> measured on the other group
    for gname, gm in (('pb', fd['is_pb']), ('surf', ~fd['is_pb'])):
        for bname, bb in (('LO', B_LO), ('HI', B_HI)):
            P_ = fd['Pall'][gm, bb]
            wu = np.where(P_ > 0, P_ ** 2 / np.maximum(fd['v1'][gm, bb], 1e-30), 0)
            W_ = fd['V'][:, gm] * wu[None]
            z_ = fd['S0'][:, gm, bb] / np.where(P_ > 0, P_, 1)[None]
            ea = (z_ * W_).sum(1) / np.maximum(W_.sum(1), 1e-12)
            ea[W_.sum(1) <= 0] = np.nan
            fd[f'{bname}_{gname}'] = ea
    for src, col in (('A_pb', 'A_pb'), ('A_nc', 'A_nc'), ('A', 'A'), ('LO_pb', 'LO_pb'), ('LO_surf', 'LO_surf'),
                     ('HI_pb', 'HI_pb'), ('HI_surf', 'HI_surf')):
        x = fd[col]
        ok = base & np.isfinite(x)
        if ok.sum() < 200:
            continue
        q = np.quantile(x[ok], [0.2, 0.4, 0.6, 0.8])
        cls = np.searchsorted(q, x)
        for c in range(5):
            Ac = ok & (cls == c)
            add('scaling', src, compare(fd, Ac * 1.0, ok * 1.0, NBOOT, rng),
                dict(cls=c, amp_med=float(np.median(x[Ac])), amp_ref=float(np.median(x[ok]))))
    log(f'fam {F}: scaling done ({time.time() - t0:.0f} s)')
    out['rows'], out['prow'], out['trow'] = rows, prow, trow
    return out


def run_analyse(a):
    from multiprocessing import Pool
    os.makedirs(f'{OUT}/fam', exist_ok=True)
    fams = a.fams or FAMS
    todo = [F for F in fams if a.force or not os.path.exists(f'{OUT}/fam/fam{F}_rows.csv')]
    sizes = {F: os.path.getsize(f'{OUT}/raw/fam{F}.h5') for F in todo}
    todo = sorted(todo, key=lambda F: -sizes[F])
    import multiprocessing as mp
    if a.procs <= 1:      # in-process (no pool: a dead worker cannot hang the parent)
        it = map(analyse_family, todo)
    else:                 # spawn context (no fork of HDF5/thread state); dead workers -> timeout
        p = mp.get_context('spawn').Pool(a.procs, maxtasksperchild=1)
        it = (r.get(timeout=3600) for r in [p.apply_async(analyse_family, (F,)) for F in todo])
    for out in it:
            F = out['F']
            pd.DataFrame(out['rows']).to_csv(f'{OUT}/fam/fam{F}_rows.csv', index=False)
            pd.DataFrame(out['prow']).to_csv(f'{OUT}/fam/fam{F}_placebo.csv', index=False)
            pd.DataFrame(out['trow']).to_csv(f'{OUT}/fam/fam{F}_timeseries.csv', index=False)
            json.dump({k: out[k] for k in ('F', 'units', 'n_clean', 'n', 'unit_chan', 'unit_is_pb', 'env_units',
                                            'env_wd')} | {'seg_info': [list(map(str, s)) for s in out['seg_info']]},
                      open(f'{OUT}/fam/fam{F}_info.json', 'w'))
            log(f'saved family {F}')


# ----------------------------------------------------------------------------------------- pooling
STATS = [f'{q}_{g}' for q in QS + ['dD'] for g in GROUPS]
# ('step', 'aboveT') has no matching placebo (much smaller sets) -> not pooled/z-scored
PMAP = {('step', 'raw'): ('step', 'raw'), ('step', 'amp'): ('step', 'amp'),
        ('step', 'noise'): ('step', 'noise'), ('late60_240', 'raw'): ('step', 'raw'),
        ('burst', 'amp_year'): ('burst', 'perm'), ('burst', 'raw'): ('burst', 'perm')}
for _w in (14, 30, 60):
    for _v in ('raw', 'amp', 'allev', 'even', 'odd', 'noise'):
        PMAP[(f'restart{_w}', _v)] = (f'restart{_w}', 'amp' if _v == 'amp' else ('noise' if _v == 'noise' else 'raw'))
PMAP[('restart14', 'noise')] = ('restart30', 'noise')
PMAP[('restart60', 'noise')] = ('restart30', 'noise')


def pool_one(R, P, stat, minp=20):
    """R: per-family rows (one per family) with stat, se_stat; P: placebo rows (F, k, stat).
    Families weighted by inverse placebo variance; pooled placebo distribution over shared k."""
    if stat not in R or stat not in P:
        return None
    R = R[np.isfinite(R[stat])]
    pv = P.pivot_table(index='k', columns='F', values=stat)
    w = {}
    for F in R.F:
        if F in pv and pv[F].count() >= minp:
            w[F] = 1 / max(np.nanvar(pv[F]), 1e-8)
    if not w:
        return None
    R = R[R.F.isin(list(w))]
    ws = np.array([w[F] for F in R.F])
    X = float(np.sum(ws * R[stat]) / ws.sum())
    se = R['se_' + stat].values if 'se_' + stat in R else np.full(len(R), np.nan)
    bse = float(np.sqrt(np.nansum((ws / ws.sum()) ** 2 * se ** 2)))
    M = pv[list(w)].values
    W = np.array([w[F] for F in w])[None] * np.isfinite(M)
    Pk = np.nansum(np.nan_to_num(M) * W, 1) / np.maximum(W.sum(1), 1e-12)
    cov = W.sum(1) / sum(w.values())
    Pk = Pk[cov > 0.6]
    if len(Pk) < 20:
        return None
    mu, sd = float(np.mean(Pk)), float(np.std(Pk))
    return dict(stat=stat, value=X, boot_se=bse, placebo_mean=mu, placebo_sd=sd, z=(X - mu) / sd,
                p=float(np.mean(np.abs(Pk - mu) >= abs(X - mu))), n_fam=len(R), n_placebo=len(Pk),
                q025=float(np.quantile(Pk, 0.025)), q975=float(np.quantile(Pk, 0.975)),
                n_A=int(R.n_A.sum()), n_B=int(R.n_B.sum()))


def fixed_effects(X, Wt, iters=40):
    """X, Wt (units x bins), NaN/0 = missing -> common s_b + unit offsets; returns s, se."""
    m = np.isfinite(X) & (Wt > 0)
    X = np.where(m, X, 0)
    Wt = np.where(m, Wt, 0)
    o = np.zeros(X.shape[0])
    s = np.zeros(X.shape[1])
    for _ in range(iters):
        s = ((X - o[:, None]) * Wt).sum(0) / np.maximum(Wt.sum(0), 1e-12)
        o = ((X - s[None]) * Wt).sum(1) / np.maximum(Wt.sum(1), 1e-12)
        c = np.sum(s * Wt.sum(0)) / Wt.sum()
        s -= c
        o += c
    se = 1 / np.sqrt(np.maximum(Wt.sum(0), 1e-12))
    s[Wt.sum(0) == 0] = np.nan
    # inflate by unit-residual scatter (reduced chi2)
    res = (X - s[None] - o[:, None]) ** 2 * Wt
    dof = max(m.sum() - X.shape[0] - X.shape[1], 1)
    se *= np.sqrt(max(np.nansum(res) / dof, 1.0))
    return s, se


def brune_band(fc):
    out = []
    for b in range(len(MF_FC)):
        f = np.linspace(MF_EDGES[b], MF_EDGES[b + 1], 50)
        out.append(np.log10(np.mean(1 / (1 + (f / fc) ** 2))))      # log10 mean amplitude (omega^2)
    return np.array(out)


def brune_pred(mu, beta, fc0):
    """log10 amplitude ratio per MF band for a moment ratio 10^mu with fc ~ M0^-beta."""
    return mu + brune_band(fc0 * 10 ** (-beta * mu)) - brune_band(fc0)


def run_pool(a):
    rows = pd.concat([pd.read_csv(f) for f in sorted(glob_(f'{OUT}/fam/fam*_rows.csv'))], ignore_index=True)
    plc = pd.concat([pd.read_csv(f) for f in sorted(glob_(f'{OUT}/fam/fam*_placebo.csv'))], ignore_index=True)
    ts = pd.concat([pd.read_csv(f) for f in sorted(glob_(f'{OUT}/fam/fam*_timeseries.csv'))], ignore_index=True)
    info = {json.load(open(f))['F']: json.load(open(f)) for f in glob_(f'{OUT}/fam/fam*_info.json')}
    summ = dict(n_families=int(rows.F.nunique()))
    # ---- 1. quake steps, restart windows, bursts (placebo / permutation calibrated)
    out = []
    keys = rows[['test', 'variant', 'quake']].fillna('-').drop_duplicates().values.tolist()
    for test, variant, quake in keys:
        if test == 'scaling' or (test, variant) not in PMAP:
            continue
        R = rows[(rows.test == test) & (rows.variant == variant) & (rows.quake.fillna('-') == quake)]
        pt, pvv = PMAP[(test, variant)]
        P = plc[(plc.test == pt) & (plc.variant == pvv)]
        for st in STATS:
            r = pool_one(R, P, st)
            if r:
                out.append(dict(test=test, variant=variant, quake=quake, **r))
    T = pd.DataFrame(out)
    T.to_csv(f'{OUT}/pooled_tests.csv', index=False)
    # pooled band curves (inverse-SE weights per family)
    cur = []
    for (test, variant, quake), R in rows[rows.test != 'scaling'].groupby(['test', 'variant', rows.quake.fillna('-')]):
        for g in GROUPS:
            c = f'curve_{g}'
            if c not in R:
                continue
            C = np.array([json.loads(x) if isinstance(x, str) else [np.nan] * len(MF_FC) for x in R[c]])
            se = R.get(f'se_LO_{g}', pd.Series(np.nan, index=R.index)).values
            w = np.where(np.isfinite(se) & (se > 0), 1 / np.maximum(se, 0.01) ** 2, 0)[:, None] * np.isfinite(C)
            if w.sum() == 0:
                continue
            cur.append(dict(test=test, variant=variant, quake=quake, group=g,
                            **{f'b{b}': float(np.nansum(np.nan_to_num(C[:, b]) * w[:, b]) / max(w[:, b].sum(), 1e-12))
                               for b in range(len(MF_FC))}))
    pd.DataFrame(cur).to_csv(f'{OUT}/pooled_curves.csv', index=False)

    # ---- 2. scaling
    sc = rows[rows.test == 'scaling'].copy()
    srows = []
    grp_for = {'A_pb': 'surf', 'A_nc': 'pb', 'A': 'all', 'LO_pb': 'surf', 'LO_surf': 'pb', 'HI_pb': 'surf',
               'HI_surf': 'pb'}
    for src, g in grp_for.items():
        for c in range(5):
            R = sc[(sc.variant == src) & (sc.cls == c)]
            d = dict(src=src, group=g, cls=c, n_fam=len(R), amp_med_over_ref=float(np.nanmedian(R.amp_med / R.amp_ref)))
            for q in QS + ['dD']:
                v, e = R.get(f'{q}_{g}'), R.get(f'se_{q}_{g}')
                if v is None or e is None:
                    continue
                m = np.isfinite(v) & np.isfinite(e) & (e > 0)
                if m.sum() == 0:
                    continue
                w = 1 / np.maximum(e[m], 0.005) ** 2
                d[q] = float(np.sum(w * v[m]) / w.sum())
                d[f'se_{q}'] = float(1 / np.sqrt(w.sum()))
                d[f'nf_{q}'] = int(m.sum())
            C = np.array([json.loads(x) if isinstance(x, str) else [np.nan] * len(MF_FC) for x in R[f'curve_{g}']])
            e = R[f'se_LO_{g}'].values
            w = np.where(np.isfinite(e) & (e > 0), 1 / np.maximum(e, 0.005) ** 2, 0)[:, None] * np.isfinite(C)
            for b in range(len(MF_FC)):
                d[f'b{b}'] = float(np.nansum(np.nan_to_num(C[:, b]) * w[:, b]) / max(w[:, b].sum(), 1e-12))
            srows.append(d)
    S = pd.DataFrame(srows)
    S.to_csv(f'{OUT}/scaling_classes.csv', index=False)
    # beta fit on (LO, MID, HFI) per class, family bootstrap
    from scipy.optimize import least_squares

    def fit_beta(LO, MID, HFI, sLO, sMID, sHFI, fc0):
        def resid(p):
            beta, mus = p[0], p[1:]
            pr = np.array([brune_pred(m, beta, fc0) for m in mus])
            return np.concatenate([(pr[:, B_LO] - LO) / sLO, (pr[:, B_MID] - pr[:, B_LO] - MID) / sMID,
                                   (pr[:, B_HI] - pr[:, B_LO] - HFI) / sHFI])
        r = least_squares(resid, np.concatenate([[0.1], LO]), bounds=([-2] + [-3] * len(LO), [3] + [3] * len(LO)))
        return r.x[0], float(np.sum(r.fun ** 2))
    bfit = []
    rng = np.random.default_rng(5)
    fams = sorted(sc.F.unique())
    for src, g in grp_for.items():
        Ss = S[S.src == src].sort_values('cls')
        if Ss[['LO', 'MID', 'HFI']].isna().any().any():
            continue
        for fc0 in (3.0, 5.0, 8.0):
            beta, chi = fit_beta(Ss.LO.values, Ss.MID.values, Ss.HFI.values, Ss.se_LO.values, Ss.se_MID.values,
                                 Ss.se_HFI.values, fc0)
            bb = []
            for k in range(40):           # family bootstrap
                fs = rng.choice(fams, len(fams))
                sub = pd.concat([sc[(sc.F == F) & (sc.variant == src)] for F in fs])
                vals = []
                for c in range(5):
                    R = sub[sub.cls == c]
                    rr = []
                    for q in ('LO', 'MID', 'HFI'):
                        v, e = R[f'{q}_{g}'], R[f'se_{q}_{g}']
                        m = np.isfinite(v) & np.isfinite(e) & (e > 0)
                        w = 1 / np.maximum(e[m], 0.005) ** 2
                        rr.append(np.sum(w * v[m]) / w.sum() if m.sum() else np.nan)
                    vals.append(rr)
                vals = np.array(vals)
                if np.isfinite(vals).all():
                    bb.append(fit_beta(vals[:, 0], vals[:, 1], vals[:, 2], Ss.se_LO.values, Ss.se_MID.values,
                                       Ss.se_HFI.values, fc0)[0])
            bfit.append(dict(src=src, group=g, fc0=fc0, beta=beta, chi2=chi, beta_boot_sd=float(np.std(bb)),
                             beta_q16=float(np.quantile(bb, 0.16)), beta_q84=float(np.quantile(bb, 0.84))))
    BF = pd.DataFrame(bfit)
    BF.to_csv(f'{OUT}/scaling_beta_fit.csv', index=False)
    # simple slope: dHFI/dLO across classes
    for src in grp_for:
        Ss = S[S.src == src]
        if len(Ss) == 5 and Ss[['LO', 'HFI']].notna().all().all():
            summ[f'scaling_slope_HFI_vs_LO_{src}'] = float(np.polyfit(Ss.LO, Ss.HFI, 1)[0])
            summ[f'scaling_slope_MID_vs_LO_{src}'] = float(np.polyfit(Ss.LO, Ss.MID, 1)[0])
            if 'dD' in Ss and Ss.dD.notna().all():
                summ[f'scaling_slope_dD_vs_LO_{src}'] = float(np.polyfit(Ss.LO, Ss.dD, 1)[0])

    # ---- 3. time series (fixed effects over units per family, then network inverse-variance pooling)
    tsr = []
    bins = sorted(ts.bin.unique())
    tmid = ts.groupby('bin').tmid.mean()
    for F, g in ts.groupby('F'):
        inf = info[F]
        ispb = np.array(inf['unit_is_pb'])
        envu = np.array(inf['env_units'], dtype=int)
        wd = np.array(inf['env_wd'])
        g = g.sort_values('bin')
        for q in ('HFI', 'MID', 'LO'):
            X = np.array([json.loads(x) for x in g[f'unit_{q}']]).T          # units x bins
            Vv = np.array([json.loads(x) for x in g[f'unit_v{q}']]).T
            Wt = 1 / np.maximum(Vv, 1e-6)
            Wt = np.minimum(Wt, WCAP * np.nanmedian(np.where(np.isfinite(X), Wt, np.nan)))
            for gn, m in dict(all=np.ones(len(ispb), bool), pb=ispb, surf=~ispb).items():
                if m.sum() == 0:
                    continue
                s_, se_ = fixed_effects(X[m], np.where(np.isfinite(X[m]), Wt[m], 0))
                for k, b in enumerate(g.bin.values):
                    if np.isfinite(s_[k]):
                        tsr.append(dict(F=F, bin=b, q=q, group=gn, s=s_[k], se=se_[k]))
        if len(envu):
            X = np.array([json.loads(x) if isinstance(x, str) and x != '[]' else [np.nan] * len(envu)
                          for x in g['unit_dD']]).T
            for gn, m in dict(all=np.ones(len(envu), bool), pb=ispb[envu], surf=~ispb[envu]).items():
                if m.sum() == 0:
                    continue
                Wt = np.where(np.isfinite(X[m]), wd[m][:, None], 0)
                s_, se_ = fixed_effects(X[m], Wt)
                for k, b in enumerate(g.bin.values):
                    if np.isfinite(s_[k]):
                        tsr.append(dict(F=F, bin=b, q='dD', group=gn, s=s_[k], se=np.nan))
    TS = pd.DataFrame(tsr)
    TS.to_csv(f'{OUT}/timeseries_family.csv', index=False)
    net = []
    for (q, gn, b), g in TS.groupby(['q', 'group', 'bin']):
        if q == 'dD':
            w = np.ones(len(g))
        else:
            w = 1 / np.maximum(g.se.values, 1e-3) ** 2
        v = float(np.sum(w * g.s) / w.sum())
        if len(g) > 2:
            se = float(np.sqrt(np.sum(w ** 2 * (g.s - v) ** 2) / w.sum() ** 2 * len(g) / (len(g) - 1)))
        else:
            se = np.nan
        net.append(dict(q=q, group=gn, bin=b, tmid=tmid[b], date=pd.to_datetime(tmid[b], unit='s').date(),
                        value=v, se=se, n_fam=len(g)))
    NET = pd.DataFrame(net)
    NET.to_csv(f'{OUT}/timeseries_network.csv', index=False)
    # long-term tests: mean after 2021-12-20 minus mean 2009-2021.9 ; null from pre-2021 split points
    lt = []
    t_pet = QT['Petrolia']
    t09 = epoch_s(['2009-01-01'])[0]
    for (q, gn), g in NET.groupby(['q', 'group']):
        g = g[np.isfinite(g.value) & (g.tmid > t09)]
        w = 1 / np.maximum(g.se.fillna(g.se.median()), 1e-3) ** 2
        post, pre = g.tmid >= t_pet, g.tmid < t_pet
        if post.sum() < 4 or pre.sum() < 8:
            continue
        D = np.average(g.value[post], weights=w[post]) - np.average(g.value[pre], weights=w[pre])
        nul = []
        L = post.sum()
        gp = g[pre].reset_index(drop=True)
        wp = w[pre].reset_index(drop=True)
        for k in range(8, len(gp) - L + 1):
            blk = slice(k, k + L)
            nul.append(np.average(gp.value[blk], weights=wp[blk]) - np.average(gp.value[:k], weights=wp[:k]))
        nul = np.array(nul)
        sl = np.polyfit((g.tmid[pre] - t09) / (365.25 * DAY), g.value[pre], 1, w=np.sqrt(w[pre]))[0]
        lt.append(dict(q=q, group=gn, post_minus_pre=D, null_sd=float(np.std(nul)) if len(nul) > 3 else np.nan,
                       null_absmax=float(np.max(np.abs(nul))) if len(nul) else np.nan,
                       z=D / np.std(nul) if len(nul) > 3 else np.nan, n_null=len(nul), pre_slope_per_yr=sl,
                       quarter_scatter_pre=float(np.std(g.value[pre])), median_se=float(g.se.median())))
    LT = pd.DataFrame(lt)
    LT.to_csv(f'{OUT}/longterm_tests.csv', index=False)

    # ---- 4. Brune sensitivity table
    sens = []
    for fc0 in (2.0, 3.0, 4.0, 6.0, 8.0):
        d = brune_band(fc0 * 1.1) - brune_band(fc0)                 # +10% corner at fixed moment (dln fc=0.0953)
        sens.append(dict(fc0=fc0, dHFI_dlnfc=float((d[B_HI] - d[B_LO]) / np.log(1.1)),
                         dMID_dlnfc=float((d[B_MID] - d[B_LO]) / np.log(1.1)),
                         dLO_dlnfc=float(d[B_LO] / np.log(1.1))))
    SE = pd.DataFrame(sens)
    SE.to_csv(f'{OUT}/brune_sensitivity.csv', index=False)

    # ---- summary
    def pick(test, variant, quake, stat):
        r = T[(T.test == test) & (T.variant == variant) & (T.quake == quake) & (T.stat == stat)]
        return r.iloc[0].to_dict() if len(r) else None
    key = {}
    for test, variant, quake in [('step', 'raw', q) for q in QUAKES] + [('step', 'amp', q) for q in QUAKES] + \
            [('restart30', 'raw', q) for q in QUAKES] + [('restart30', 'amp', q) for q in QUAKES] + \
            [('restart14', 'amp', 'M7'), ('restart60', 'amp', 'M7'), ('late60_240', 'raw', 'M7'),
             ('burst', 'amp_year', '-'), ('burst', 'raw', '-')]:
        for st in ('HFI_all', 'MID_all', 'LO_all', 'dD_all', 'HFI_pb', 'HFI_surf', 'MID_pb', 'MID_surf', 'dD_pb', 'dD_surf'):
            r = pick(test, variant, quake, st)
            if r:
                key[f'{test}|{variant}|{quake}|{st}'] = {k: (round(v, 4) if isinstance(v, float) else v)
                                                         for k, v in r.items() if k in ('value', 'boot_se', 'placebo_sd', 'z', 'p', 'n_fam', 'n_A')}
    summ['key_tests'] = key
    summ['beta_fit'] = BF.round(3).to_dict('records')
    summ['longterm'] = LT.round(4).to_dict('records')
    summ['brune_sensitivity'] = SE.round(3).to_dict('records')
    summ['n_tests_pooled'] = int(len(T))
    summ['n_abs_z_gt_2'] = int((T.z.abs() > 2).sum())
    json.dump(summ, open(f'{OUT}/summary.json', 'w'), indent=1, default=float)
    log('pool done', len(T), 'pooled tests;', int((T.z.abs() > 2).sum()), 'with |z|>2')


def glob_(p):
    import glob
    return glob.glob(p)


# ----------------------------------------------------------------------------------------- figures
def run_figs(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    T = pd.read_csv(f'{OUT}/pooled_tests.csv')
    C = pd.read_csv(f'{OUT}/pooled_curves.csv')
    S = pd.read_csv(f'{OUT}/scaling_classes.csv')
    NET = pd.read_csv(f'{OUT}/timeseries_network.csv')
    qcol = {'Petrolia': '#2a9d8f', 'Ferndale': '#e9c46a', 'M7': '#e76f51'}

    # ---- Fig 1: quake steps / restart: indices vs placebo
    tests = [('step', 'amp', 'Petrolia'), ('step', 'amp', 'Ferndale'), ('step', 'amp', 'M7'),
             ('restart14', 'amp', 'M7'), ('restart30', 'amp', 'Petrolia'), ('restart30', 'amp', 'Ferndale'),
             ('restart30', 'amp', 'M7'), ('restart60', 'amp', 'M7'), ('late60_240', 'raw', 'M7'),
             ('burst', 'amp_year', '-')]
    labels = ['Petrolia step', 'Ferndale step', 'M7 step', 'M7 restart 14 d', 'Petrolia first 30 d',
              'Ferndale first 30 d', 'M7 restart 30 d', 'M7 restart 60 d', 'M7 +60-240 d', 'burst vs isolated']
    fig, axs = plt.subplots(1, 4, figsize=(15, 5.5), sharey=True)
    for ax, (q, name) in zip(axs, [('MID', 'MID: log10 A(3-4.5)/A(2-3) change'),
                                   ('HFI', 'HFI: log10 A(4.5-6.5)/A(2-3) change'),
                                   ('LO', 'LO: log10 amplitude (2-3 Hz) change'),
                                   ('dD', 'dD: RMS S-envelope width change (s)')]):
        for k, (te, va, qu) in enumerate(tests):
            for j, (g, col, off) in enumerate([('all', 'k', 0), ('pb', '#1f77b4', 0.22), ('surf', '#d62728', -0.22)]):
                r = T[(T.test == te) & (T.variant == va) & (T.quake == qu) & (T.stat == f'{q}_{g}')]
                if not len(r):
                    continue
                r = r.iloc[0]
                y = k + off
                if g == 'all':
                    ax.fill_betweenx([y - 0.35, y + 0.35], r.q025, r.q975, color='0.85', zorder=0)
                ax.errorbar(r.value, y, xerr=r.boot_se, fmt='o' if g == 'all' else '.', color=col, ms=6 if g == 'all' else 4,
                            label={'all': 'all stations', 'pb': 'PB boreholes', 'surf': 'surface'}[g] if k == 0 else None)
                if g == 'all' and abs(r.z) >= 2:
                    ax.text(r.value, y + 0.38, f'z={r.z:.1f}', ha='center', fontsize=7)
        ax.axvline(0, color='0.5', lw=0.8)
        ax.set_title(name, fontsize=9)
    axs[0].set_yticks(range(len(tests)))
    axs[0].set_yticklabels(labels)
    axs[0].invert_yaxis()
    axs[0].legend(fontsize=7, loc='lower left')
    fig.suptitle('LFE source-spectral changes, 27 families pooled (amplitude-matched; grey = 95% placebo range)')
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_source_spectra_quakes.png', dpi=130)
    plt.close(fig)

    # ---- Fig 2: pooled spectral-ratio curves
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, qu in zip(axs, ['Petrolia', 'Ferndale', 'M7']):
        for te, va, ls in (('step', 'amp', '-'), ('restart30', 'amp', '--')):
            for g, col in (('pb', '#1f77b4'), ('surf', '#d62728'), ('all', 'k')):
                r = C[(C.test == te) & (C.variant == va) & (C.quake == qu) & (C.group == g)]
                if len(r):
                    y = r[[f'b{b}' for b in range(len(MF_FC))]].values[0]
                    ax.plot(MF_FC[:5], y[:5], ls, color=col, marker='o', ms=3, label=f'{te} {g}')
        ax.axhline(0, color='0.5', lw=0.8)
        ax.set_xscale('log')
        ax.set_xlabel('Hz')
        ax.set_title(f'{qu}: after/before (log10 amplitude)')
    axs[0].set_ylabel('log10 amplitude ratio')
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_source_spectra_curves.png', dpi=130)
    plt.close(fig)

    # ---- Fig 3: time series
    fig, axs = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for ax, q, yl in zip(axs, ['MID', 'HFI', 'dD'], ['MID (log10)', 'HFI (log10)', 'dD (s)']):
        for g, col in (('pb', '#1f77b4'), ('surf', '#d62728')):
            d = NET[(NET.q == q) & (NET.group == g)].sort_values('tmid')
            tt = pd.to_datetime(d.tmid, unit='s')
            ax.errorbar(tt, d.value, yerr=d.se, fmt='.-', color=col, lw=0.8, ms=4, capsize=0,
                        label='PB boreholes' if g == 'pb' else 'surface')
        for qn, qt in QUAKES.items():
            ax.axvline(pd.Timestamp(qt), color=qcol[qn], lw=1.2)
        ax.axhline(0, color='0.5', lw=0.6)
        ax.set_ylabel(yl)
    axs[0].legend(fontsize=8)
    axs[0].set_title('Network-pooled quarterly source-spectral indices (each family vs its own long-term mean)')
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_source_spectra_timeseries.png', dpi=130)
    plt.close(fig)

    # ---- Fig 4: scaling
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    cols = plt.cm.viridis(np.linspace(0, 0.9, 5))
    Ss = S[S.src == 'LO_pb']
    for c in range(5):
        r = Ss[Ss.cls == c]
        if len(r):
            y = r[[f'b{b}' for b in range(len(MF_FC))]].values[0]
            axs[0].plot(MF_FC[:5], y[:5] - y[B_LO], 'o-', color=cols[c], label=f'quintile {c + 1}')
    for fc0, ls in ((3.0, ':'), (5.0, '--')):
        for mu in (-0.4, 0.4):
            pr = brune_pred(mu, 1 / 3, fc0)
            axs[0].plot(MF_FC[:5], pr[:5] - pr[B_LO], ls, color='0.5', lw=0.8)
    axs[0].set_xscale('log')
    axs[0].set_title('Amplitude classes (PB-selected, surface-measured)\nnormalised at 2-3 Hz; grey: self-similar, fc0 3/5 Hz, +-0.4')
    axs[0].legend(fontsize=7)
    axs[0].set_xlabel('Hz')
    for src, mk, g in (('LO_pb', 'v', 'surf'), ('LO_surf', 'v', 'pb'), ('A_pb', 'o', 'surf'), ('A_nc', 's', 'pb'),
                       ('HI_pb', '^', 'surf'), ('HI_surf', '^', 'pb')):
        r = S[S.src == src]
        axs[1].errorbar(r.LO, r.HFI, xerr=r.se_LO, yerr=r.se_HFI, fmt=mk, label=f'{src} classes ({g})')
        if 'dD' in r:
            axs[2].errorbar(r.LO, r.dD, xerr=r.se_LO, yerr=r.get('se_dD'), fmt=mk, label=src)
    mus = np.linspace(-0.5, 0.5, 21)
    for beta, ls in ((0, '-'), (1 / 3, '--')):
        for fc0 in (3.0, 5.0):
            pr = np.array([brune_pred(m, beta, fc0) for m in mus])
            axs[1].plot(pr[:, B_LO], pr[:, B_HI] - pr[:, B_LO], ls, color='0.4', lw=0.8,
                        label=f'beta={beta:.2f}, fc0={fc0:g}')
    axs[1].set_xlabel('LO = log10 amplitude ratio (2-3 Hz)')
    axs[1].set_ylabel('HFI')
    axs[1].legend(fontsize=6)
    axs[2].set_xlabel('LO')
    axs[2].set_ylabel('dD (s)')
    axs[2].axhline(0, color='0.5', lw=0.6)
    axs[2].legend(fontsize=7)
    axs[2].set_title('S-envelope RMS width vs size')
    fig.tight_layout()
    fig.savefig(f'{FIG}/sci_source_spectra_scaling.png', dpi=130)
    plt.close(fig)
    log('figures written')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['extract', 'extract_mf', 'analyse', 'pool', 'figs'])
    ap.add_argument('--fams', type=int, nargs='*')
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    if args.stage == 'extract':
        run_extract(args)
    elif args.stage == 'extract_mf':
        run_extract_mf(args)
    elif args.stage == 'analyse':
        run_analyse(args)
    elif args.stage == 'pool':
        run_pool(args)
    elif args.stage == 'figs':
        run_figs(args)
