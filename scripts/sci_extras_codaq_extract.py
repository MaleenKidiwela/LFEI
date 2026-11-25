#!/usr/bin/env python3
"""CODA Q THROUGH TIME -- extraction (one pass over data/waveforms/fam{F}.h5).

Per family and channel (station has an S pick in data/locations/fam{F}_3d_picks.csv, template SNR >= 5):
  events: catalog LFEs isolated from every other merged-catalog detection (any family) in [t-20 s, t+40 s]
          (coda to S+20 s must not contain another LFE);
  bands : 2-4 and 4-8 Hz (Butterworth 4, zero phase) of the stored 1-15 Hz, 50 Hz traces (-15..+45 s rel catalog);
  energy: |hilbert|^2, averaged in 0.5 s blocks (120 blocks, block k centre = -15 + 0.5 k + 0.25 s),
          each event divided by its mean band energy over the whole trace (equal event weights, as the
          RMS-normalised stacks of sci_lfe_phases_extract.py);
  bins  : half-month bins (1st, 16th) 2008-2027, split at Petrolia / Ferndale / M7 (and 2010-01-10, 2014-03-10)
          -> sum of normalised energy envelopes per bin (ESUM), event count (N), and the same for even/odd events
          (ESUM_H: halves for a noise-based SE).
Output (large) -> /data/wsd02/maleen_data/extras/codaq/env_fam{F}.h5
Run: nice -n 10 python scripts/sci_extras_codaq_extract.py --procs 2
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
BIG = '/data/wsd02/maleen_data/extras/codaq'
SR, PRE = 50.0, -15.0
BLK = 25                      # 0.5 s blocks
BANDS = [(2.0, 4.0), (4.0, 8.0)]
ISO = (-20.0, 40.0)
QUAKES = {'eureka2010': '2010-01-10T00:27:39', 'mendo2014': '2014-03-10T05:18:13',
          'petrolia': '2021-12-20T20:10', 'ferndale': '2022-12-20T10:34', 'm7': '2024-12-05T18:44'}


def epoch_s(x):
    x = pd.DatetimeIndex(x)
    return np.asarray((x - pd.Timestamp('1970-01-01')) / pd.Timedelta(seconds=1), dtype=float)


def bin_edges():
    a = pd.date_range('2008-01-01', '2027-01-01', freq='MS')
    b = a + pd.Timedelta(days=15)
    e = np.sort(np.r_[epoch_s(a), epoch_s(b[:-1]), epoch_s(list(QUAKES.values()))])
    return e


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def extract(F):
    fo = f'{BIG}/env_fam{F}.h5'
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
    with h5py.File(f'{ROOT}/data/waveforms/fam{F}.h5', 'r') as h:
        te = epoch_s(pd.to_datetime(h['events/time_utc'][:].astype(str)))
        iso = (np.searchsorted(allt, te + ISO[1], side='right') - np.searchsorted(allt, te + ISO[0])) == 1
        bi = np.searchsorted(E, te, side='right') - 1
        chans = []
        for cid in sorted(h['chan']):
            if cid not in T or T[cid].attrs['snr'] < 5:
                continue
            sta = '.'.join(cid.split('.')[:2])
            if sta not in picks.index:
                continue
            pk = picks.loc[sta]
            S = float(pk.S_final) if np.isfinite(pk.S_final) else (
                float(pk.S) if np.isfinite(pk.S) and pk.sig_S <= 0.1 else np.nan)
            if not np.isfinite(S):
                continue
            chans.append((cid, float(T[cid].attrs['snr']), S))
        nch = len(chans)
        nblk = int(60 * SR) // BLK
        ES = np.zeros((nch, 2, nb, nblk), np.float32)
        EH = np.zeros((nch, 2, 2, nb, nblk), np.float32)
        N = np.zeros((nch, nb), np.int32)
        NH = np.zeros((nch, 2, nb), np.int32)
        for k, (cid, snr, S) in enumerate(chans):
            ev = h[f'chan/{cid}/ev'][:]
            keep = iso[ev] & (bi[ev] >= 0) & (bi[ev] < nb)
            rows = np.where(keep)[0]
            if len(rows) == 0:
                continue
            b_ev = bi[ev[rows]]
            half = (np.arange(len(rows)) % 2)
            N[k] = np.bincount(b_ev, minlength=nb)
            for hh in (0, 1):
                NH[k, hh] = np.bincount(b_ev[half == hh], minlength=nb)
            dset = h[f'chan/{cid}/data'][:]
            for c0 in range(0, len(rows), 3000):
                rr = rows[c0:c0 + 3000]
                X = dset[rr].astype(np.float64)
                bb = b_ev[c0:c0 + 3000]
                hb = half[c0:c0 + 3000]
                for j, s in enumerate(sos):
                    Y = sosfiltfilt(s, X, axis=1)
                    En = np.abs(hilbert(Y, axis=1)) ** 2
                    En = En[:, :nblk * BLK].reshape(len(rr), nblk, BLK).mean(2)
                    En /= (En.mean(1, keepdims=True) + 1e-30)
                    np.add.at(ES[k, j], bb, En)
                    for hh in (0, 1):
                        m = hb == hh
                        np.add.at(EH[k, j, hh], bb[m], En[m])
                del X
            del dset
    T.close()
    os.makedirs(BIG, exist_ok=True)
    with h5py.File(fo + '.tmp', 'w') as o:
        o.create_dataset('ESUM', data=ES, compression='gzip', compression_opts=1)
        o.create_dataset('ESUM_H', data=EH, compression='gzip', compression_opts=1)
        o['N'] = N
        o['NH'] = NH
        o['edges'] = E
        o['chan'] = np.array([c[0] for c in chans], dtype='S20')
        o['snr'] = np.array([c[1] for c in chans])
        o['S'] = np.array([c[2] for c in chans])
        o.attrs.update(dict(sr=SR, pre=PRE, blk_s=BLK / SR, origin=origin, n_iso=int(iso.sum()),
                            bands=json.dumps(BANDS), iso=json.dumps(ISO)))
    os.replace(fo + '.tmp', fo)
    log(f'fam {F}: {len(te)} events ({iso.sum()} isolated [-20,+40] s), {nch} channels')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fams', type=int, nargs='*')
    ap.add_argument('--procs', type=int, default=2)
    a = ap.parse_args()
    fams = a.fams or list(range(1, 28))
    sizes = {F: os.path.getsize(f'{ROOT}/data/waveforms/fam{F}.h5') for F in fams}
    fams = sorted(fams, key=lambda F: -sizes[F])
    os.makedirs(BIG, exist_ok=True)
    if a.procs == 1:
        for F in fams:
            extract(F)
    else:
        from multiprocessing import Pool
        with Pool(a.procs) as p:
            list(p.imap_unordered(extract, fams))
