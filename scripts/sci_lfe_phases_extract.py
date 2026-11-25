#!/usr/bin/env python3
"""LFE secondary phases / attenuation -- EXTRACTION (one pass over data/waveforms/fam{F}.h5).

Per family, channels with template SNR >= 8 and a P or S pick (data/locations/fam{F}_3d_picks.csv):
  stacks : isolated LFEs (no other merged-catalog detection within +-5 s), 1-15 Hz as stored, each event divided by
           its RMS over the window, plain mean (catalog-time aligned, as sci_dvv_source_region_measure.py).
           Window -5..+35 s rel. catalog time.  Stacks: all, half0/half1 (alternate events), yearly 2008..2026,
           eras pre-Petrolia | Petrolia-Ferndale | Ferndale-M7 | M7..M7+60 d | after M7+60 d.
           -> OUT/stacks/fam{F}.h5 (stack (nstack, nch, nt) float32, count (nstack, nch), names, chan)
           monthly stacks (calendar months; the months holding Petrolia / Ferndale / M7 are split at the quake time)
           -> mstack (nmonth, nch, nt) float16, mcount, medges (epoch s)
  spectra: every event (not only isolated): multitaper power (4 dpss, NW 2.5, 3.2 s) in the 12 log bands of
           sci_source_spectra.py (1-15 Hz) for the P window (P-0.4..P+2.8 s) [P pick, sig_P<=0.25 s, and the P
           window ending >= 0.3 s before S], the S window (S-0.4..S+2.8 s) and the pre-event noise windows,
           log10(power / channel median noise power), float16.  -> OUT/raw/fam{F}.h5 (PP, SP, NP, time_utc, chan)
Run: nice -n 10 python scripts/sci_lfe_phases_extract.py --procs 4
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse
import json
import sys
import time

import h5py
import numpy as np
import pandas as pd
from scipy.signal.windows import dpss

ROOT = '/home/seismic/LFE'
sys.path.insert(0, f'{ROOT}/scripts')
from sci_lfe_phases_common import OUT, QUAKES, epoch_s  # noqa: E402
from sci_source_spectra import band_matrix, mt_band_power, NWIN, NB  # noqa: E402

SR, PRE = 50.0, -15.0
W0, W1 = -5.0, 35.0
I0, I1 = int((W0 - PRE) * SR), int((W1 - PRE) * SR)
YEARS = list(range(2008, 2027))
QT = dict(zip(QUAKES, epoch_s(list(QUAKES.values()))))


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def stack_defs(te, iso):
    """-> names, list of boolean masks over events"""
    names, masks = ['all', 'half0', 'half1'], []
    ii = np.where(iso)[0]
    masks.append(iso.copy())
    h0 = np.zeros_like(iso)
    h0[ii[0::2]] = True
    masks += [h0, iso & ~h0]
    yr = pd.to_datetime(te, unit='s').year.values
    for y in YEARS:
        names.append(f'y{y}')
        masks.append(iso & (yr == y))
    p, f, m = QT['petrolia'], QT['ferndale'], QT['m7']
    for nm, a, b in (('era_prePet', -1e18, p), ('era_PetFer', p, f), ('era_FerM7', f, m),
                     ('era_M7_60d', m, m + 60 * 86400), ('era_postM7', m + 60 * 86400, 1e18)):
        names.append(nm)
        masks.append(iso & (te >= a) & (te < b))
    return names, masks


def month_edges():
    e = epoch_s(pd.date_range('2008-01-01', '2027-01-01', freq='MS'))
    return np.sort(np.r_[e, list(QT.values())])


def extract(F):
    fs, fr = f'{OUT}/stacks/fam{F}.h5', f'{OUT}/raw/fam{F}.h5'
    if os.path.exists(fs) and os.path.exists(fr):
        log('exists', F)
        return
    T = h5py.File(f'{ROOT}/data/templates/fam{F}_templates.h5', 'r')
    picks = pd.read_csv(f'{ROOT}/data/locations/fam{F}_3d_picks.csv').set_index('sta')
    origin = json.load(open(f'{ROOT}/data/locations/fam{F}_3d.json'))['origin_rel_catalog_s']
    tapers = dpss(NWIN, 2.5, 4)
    M = band_matrix()
    noise_end = min(origin + 2.0, -6.0)
    nstarts, t0 = [], -14.8
    while t0 + NWIN / SR <= noise_end + 1e-9:
        nstarts.append(int(round((t0 - PRE) * SR)))
        t0 += NWIN / SR
    cat = pd.read_csv(f'{ROOT}/data/merged/LFEcat_MTJ_merged_v1v2.csv', usecols=['time_utc', 'source'])
    allt = np.sort(epoch_s(pd.to_datetime(cat[cat.source != 'v1_only'].time_utc)))
    with h5py.File(f'{ROOT}/data/waveforms/fam{F}.h5', 'r') as h:
        times = h['events/time_utc'][:]
        te = epoch_s(pd.to_datetime(times.astype(str)))
        n_ev = len(te)
        iso = (np.searchsorted(allt, te + 5.0, side='right') - np.searchsorted(allt, te - 5.0)) == 1
        names, masks = stack_defs(te, iso)
        chans = []
        for cid in sorted(h['chan']):
            if cid not in T or T[cid].attrs['snr'] < 8:
                continue
            sta = '.'.join(cid.split('.')[:2])
            if sta not in picks.index:
                continue
            pk = picks.loc[sta]
            S = float(pk.S_final) if np.isfinite(pk.S_final) else (float(pk.S) if np.isfinite(pk.S) and pk.sig_S <= 0.1 else np.nan)
            P = float(pk.P) if np.isfinite(pk.P) and pk.sig_P <= 0.25 else np.nan
            if np.isfinite(P) and np.isfinite(S) and P - 0.4 + NWIN / SR > S - 0.3:
                P = np.nan
            if not (np.isfinite(S) or np.isfinite(P)):
                continue
            chans.append((cid, float(T[cid].attrs['snr']), S, P))
        nch = len(chans)
        STK = np.zeros((len(names), nch, I1 - I0), np.float32)
        CNT = np.zeros((len(names), nch), np.int32)
        ME = month_edges()
        mb = np.searchsorted(ME, te, side='right') - 1
        MST = np.zeros((len(ME) - 1, nch, I1 - I0), np.float32)
        MCN = np.zeros((len(ME) - 1, nch), np.int32)
        PP = np.full((n_ev, nch, NB), np.nan, np.float16)
        SP = np.full((n_ev, nch, NB), np.nan, np.float16)
        NP = np.full((n_ev, nch, NB), np.nan, np.float16)
        for k, (cid, snr, S, P) in enumerate(chans):
            ev = h[f'chan/{cid}/ev'][:]
            X = h[f'chan/{cid}/data'][:].astype(np.float64)
            pn = np.mean([mt_band_power(X[:, j:j + NWIN], tapers, M) for j in nstarts], 0)
            sc = np.nanmedian(pn, 0)
            NP[ev, k] = np.log10(np.maximum(pn / sc, 1e-30)).astype(np.float16)
            for t_, A in ((S, SP), (P, PP)):
                if np.isfinite(t_):
                    i0 = int(round((t_ - 0.4 - PRE) * SR))
                    if i0 + NWIN <= X.shape[1]:
                        pw = mt_band_power(X[:, i0:i0 + NWIN], tapers, M)
                        A[ev, k] = np.log10(np.maximum(pw / sc, 1e-30)).astype(np.float16)
            Y = X[:, I0:I1]
            Y = Y / (np.sqrt(np.mean(Y ** 2, 1)) + 1e-20)[:, None]
            for s, m in enumerate(masks):
                sel = m[ev]
                if sel.any():
                    STK[s, k] = Y[sel].mean(0)
                    CNT[s, k] = sel.sum()
            e_iso = iso[ev]
            bk = mb[ev][e_iso]
            np.add.at(MST[:, k], bk, Y[e_iso])
            MCN[:, k] = np.bincount(bk, minlength=len(ME) - 1)
            nz = MCN[:, k] > 0
            MST[nz, k] /= MCN[nz, k][:, None]
            del X, Y
    T.close()
    os.makedirs(f'{OUT}/stacks', exist_ok=True)
    os.makedirs(f'{OUT}/raw', exist_ok=True)
    with h5py.File(fs + '.tmp', 'w') as o:
        o['stack'] = STK
        o['count'] = CNT
        o.create_dataset('mstack', data=MST.astype(np.float16), compression='gzip', compression_opts=1)
        o['mcount'] = MCN
        o['medges'] = ME
        o['names'] = np.array(names, dtype='S16')
        o['chan'] = np.array([c[0] for c in chans], dtype='S20')
        o['snr'] = np.array([c[1] for c in chans])
        o['S'] = np.array([c[2] for c in chans])
        o['P'] = np.array([c[3] for c in chans])
        o.attrs.update(dict(sr=SR, t0=W0, origin=origin, n_iso=int(iso.sum())))
    os.replace(fs + '.tmp', fs)
    with h5py.File(fr + '.tmp', 'w') as o:
        o.create_dataset('PP', data=PP, compression='gzip', compression_opts=1)
        o.create_dataset('SP', data=SP, compression='gzip', compression_opts=1)
        o.create_dataset('NP', data=NP, compression='gzip', compression_opts=1)
        o['time_utc'] = times
        o['iso'] = iso
        o['chan'] = np.array([c[0] for c in chans], dtype='S20')
        o['snr'] = np.array([c[1] for c in chans])
        o['S'] = np.array([c[2] for c in chans])
        o['P'] = np.array([c[3] for c in chans])
        o.attrs.update(dict(noise_starts=nstarts, origin=origin))
    os.replace(fr + '.tmp', fr)
    log(f'fam {F}: {n_ev} events ({iso.sum()} isolated), {nch} channels')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fams', type=int, nargs='*')
    ap.add_argument('--procs', type=int, default=4)
    a = ap.parse_args()
    fams = a.fams or list(range(1, 28))
    sizes = {F: os.path.getsize(f'{ROOT}/data/waveforms/fam{F}.h5') for F in fams}
    fams = sorted(fams, key=lambda F: -sizes[F])
    from multiprocessing import Pool
    with Pool(a.procs) as p:
        list(p.imap_unordered(extract, fams))
