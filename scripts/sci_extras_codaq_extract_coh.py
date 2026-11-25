#!/usr/bin/env python3
"""CODA Q THROUGH TIME -- extraction of COHERENT (linear) stacks with even/odd halves.

Why: single isolated LFEs have coda energy SNR << 1 (energy averaging, sci_extras_codaq_extract.py, leaves the coda
at 5-50 % of the noise floor); linear stacks raise the coherent coda by N, and the even/odd half-difference gives the
incoherent (noise + incoherent coda) energy with the same N scaling, so coherent energy = |full|^2 - |half-diff|^2.
Per family and channel (template SNR >= 8, S pick), isolated LFEs (+-5 s, as sci_lfe_phases_extract.py), each event
RMS-normalised over the whole -15..+45 s trace; bands 2-4 / 4-8 Hz (Butterworth 4, zero phase); analytic signal
demodulated by exp(-i 2 pi fc t) and sampled at 5 Hz over S-1 .. S+21 s (110 samples; linear, so stacks of the
baseband signals = baseband of the stack); summed per half-month bin (split at the 5 quakes) and per even/odd half.
Output (large) -> /data/wsd02/maleen_data/extras/codaq/coh_fam{F}.h5  (CS complex64 (nch, 2 band, 2 half, nbin, 110))
Run: nice -n 10 python scripts/sci_extras_codaq_extract_coh.py --procs 2
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
from scipy.signal import butter, sosfiltfilt, hilbert

ROOT = '/home/seismic/LFE'
sys.path.insert(0, f'{ROOT}/scripts')
from sci_extras_codaq_extract import BIG, SR, PRE, BANDS, bin_edges, epoch_s, log  # noqa: E402

ISO = (-5.0, 5.0)
DEC = 10                       # 50 Hz -> 5 Hz
T0, T1 = -1.0, 21.0            # rel S
NS = int((T1 - T0) * SR / DEC)  # 110


MINMCC = 0.0
PREFIX = 'coh'


def extract(F):
    fo = f'{BIG}/{PREFIX}_fam{F}.h5'
    if os.path.exists(fo):
        log('exists', F)
        return
    T = h5py.File(f'{ROOT}/data/templates/fam{F}_templates.h5', 'r')
    picks = pd.read_csv(f'{ROOT}/data/locations/fam{F}_3d_picks.csv').set_index('sta')
    origin = json.load(open(f'{ROOT}/data/locations/fam{F}_3d.json'))['origin_rel_catalog_s']
    cat = pd.read_csv(f'{ROOT}/data/merged/LFEcat_MTJ_merged_v1v2.csv', usecols=['time_utc', 'source'])
    allt = np.sort(epoch_s(pd.to_datetime(cat[cat.source != 'v1_only'].time_utc)))
    E = bin_edges()
    nb = len(E) - 1
    sos = [butter(4, [a, b], btype='band', fs=SR, output='sos') for a, b in BANDS]
    fcs = [np.sqrt(a * b) for a, b in BANDS]
    tt = PRE + np.arange(int(60 * SR)) / SR
    with h5py.File(f'{ROOT}/data/waveforms/fam{F}.h5', 'r') as h:
        te = epoch_s(pd.to_datetime(h['events/time_utc'][:].astype(str)))
        iso = (np.searchsorted(allt, te + ISO[1], side='right') - np.searchsorted(allt, te + ISO[0])) == 1
        if MINMCC > 0:
            iso &= h['events/meancc_v2'][:] >= MINMCC
        bi = np.searchsorted(E, te, side='right') - 1
        chans = []
        for cid in sorted(h['chan']):
            if cid not in T or T[cid].attrs['snr'] < 8:
                continue
            sta = '.'.join(cid.split('.')[:2])
            if sta not in picks.index:
                continue
            pk = picks.loc[sta]
            S = float(pk.S_final) if np.isfinite(pk.S_final) else (
                float(pk.S) if np.isfinite(pk.S) and pk.sig_S <= 0.1 else np.nan)
            if not np.isfinite(S):
                continue
            i0 = int(round((S + T0 - PRE) * SR))
            if i0 < 0 or i0 + NS * DEC > len(tt):
                continue
            chans.append((cid, float(T[cid].attrs['snr']), S, i0))
        nch = len(chans)
        CS = np.zeros((nch, 2, 2, nb, NS), np.complex64)
        NH = np.zeros((nch, 2, nb), np.int32)
        for k, (cid, snr, S, i0) in enumerate(chans):
            ev = h[f'chan/{cid}/ev'][:]
            keep = iso[ev] & (bi[ev] >= 0) & (bi[ev] < nb)
            rows = np.where(keep)[0]
            if len(rows) == 0:
                continue
            b_ev = bi[ev[rows]]
            half = (np.arange(len(rows)) % 2)
            for hh in (0, 1):
                NH[k, hh] = np.bincount(b_ev[half == hh], minlength=nb)
            dset = h[f'chan/{cid}/data'][:]
            sl = slice(i0, i0 + NS * DEC, DEC)
            for c0 in range(0, len(rows), 3000):
                rr = rows[c0:c0 + 3000]
                X = dset[rr].astype(np.float64)
                X /= (np.sqrt(np.mean(X ** 2, 1)) + 1e-20)[:, None]
                bb = b_ev[c0:c0 + 3000]
                hb = half[c0:c0 + 3000]
                for j, s in enumerate(sos):
                    Z = hilbert(sosfiltfilt(s, X, axis=1), axis=1)[:, sl]
                    Z *= np.exp(-2j * np.pi * fcs[j] * tt[sl])[None]
                    for hh in (0, 1):
                        m = hb == hh
                        np.add.at(CS[k, j, hh], bb[m], Z[m].astype(np.complex64))
                del X
            del dset
    T.close()
    with h5py.File(fo + '.tmp', 'w') as o:
        o.create_dataset('CS', data=CS, compression='gzip', compression_opts=1)
        o['NH'] = NH
        o['edges'] = E
        o['chan'] = np.array([c[0] for c in chans], dtype='S20')
        o['snr'] = np.array([c[1] for c in chans])
        o['S'] = np.array([c[2] for c in chans])
        o['tS'] = T0 + np.arange(NS) * DEC / SR          # sample times rel S
        o.attrs.update(dict(origin=origin, n_iso=int(iso.sum()), bands=json.dumps(BANDS), iso=json.dumps(ISO),
                            fc=json.dumps(fcs)))
    os.replace(fo + '.tmp', fo)
    log(f'fam {F}: {len(te)} events ({iso.sum()} isolated +-5 s), {nch} channels')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fams', type=int, nargs='*')
    ap.add_argument('--procs', type=int, default=2)
    ap.add_argument('--minmcc', type=float, default=0.0, help='event-mix control: keep meancc_v2 >= this (constant-completeness thin catalog 0.137) -> coh137_fam*.h5')
    a = ap.parse_args()
    if a.minmcc > 0:
        MINMCC = a.minmcc
        PREFIX = 'coh%d' % round(a.minmcc * 1000)
    fams = a.fams or list(range(1, 28))
    sizes = {F: os.path.getsize(f'{ROOT}/data/waveforms/fam{F}.h5') for F in fams}
    fams = sorted(fams, key=lambda F: -sizes[F])
    if a.procs == 1:
        for F in fams:
            extract(F)
    else:
        from multiprocessing import Pool
        with Pool(a.procs) as p:
            list(p.imap_unordered(extract, fams))
