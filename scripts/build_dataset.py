#!/usr/bin/env python3
"""Cut, filter and pack event waveforms of one family into an HDF5 file.

For each v2 event of the family and each channel with data:
  window [t+PRE, t+POST] (catalog time t), demean/detrend, taper, bandpass FMIN-FMAX,
  resample to SR, float32. Traces with gaps inside the window are skipped.

HDF5 layout (data/waveforms/fam{F}.h5):
  /events            table (time_utc, ccsum_v2, cc_div_thr_v2, meancc_v2, nchan_v2, source)
                     stored as separate datasets under /events/
  /chan/{NET.STA.LOC.CHA}/data      (n, npts) float32
  /chan/{NET.STA.LOC.CHA}/ev        (n,) int32 row index into /events
  attrs: pre, post, sr, fmin, fmax
Response is NOT removed (relative measurements only; instruments kept separate per channel).
"""
import argparse
import glob
import os
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
from obspy import UTCDateTime, read

PRE, POST, SR, FMIN, FMAX = -15.0, 45.0, 50.0, 1.0, 15.0
NPTS = int(round((POST - PRE) * SR))


def process_day(args):
    daydir, evrows = args
    out = {}
    files = glob.glob(os.path.join(daydir, '*.mseed'))
    if not files:
        return out
    for f in files:
        try:
            st = read(f)
        except Exception:
            continue
        st.merge(method=1, fill_value=None)
        st = st.split()
        for tr in st:
            if tr.stats.npts < 10 * tr.stats.sampling_rate:
                continue
            tr.detrend('demean')
            tr.detrend('linear')
            tr.taper(max_percentage=None, max_length=1.0)
            tr.filter('bandpass', freqmin=FMIN, freqmax=FMAX, corners=4, zerophase=True)
            if tr.stats.sampling_rate != SR:
                tr.resample(SR, no_filter=True) if tr.stats.sampling_rate % SR else \
                    tr.decimate(int(tr.stats.sampling_rate // SR), no_filter=True)
        for idx, t in evrows:
            t0 = UTCDateTime(t) + PRE
            for tr in st:
                s = tr.stats
                if s.starttime > t0 + 0.5 / SR or s.endtime < t0 + (NPTS - 1) / SR - 0.5 / SR:
                    continue
                i0 = int(round((t0 - s.starttime) * SR))
                x = tr.data[i0:i0 + NPTS]
                if len(x) != NPTS or not np.isfinite(x).all() or np.std(x) == 0:
                    continue
                out.setdefault(tr.id, []).append((idx, x.astype('float32')))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', type=int, default=23)
    ap.add_argument('--catalog', default='data/merged/LFEcat_MTJ_merged_v1v2.csv')
    ap.add_argument('--wfdir', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--procs', type=int, default=24)
    a = ap.parse_args()
    F = a.family
    wfdir = a.wfdir or (f'data/waveforms/fam{F}_raw' if os.path.isdir(f'data/waveforms/fam{F}_raw') else 'data/waveforms/allfam_raw')
    out = a.out or f'data/waveforms/fam{F}.h5'

    cat = pd.read_csv(a.catalog)
    cat = cat[(cat.idnum == F) & (cat.source != 'v1_only')]          # v2 catalog only
    cat['t'] = pd.to_datetime(cat.time_utc)
    ev = cat.sort_values('t').reset_index(drop=True)
    ev['daydir'] = [os.path.join(wfdir, f'{t:%Y}', f'{t:%Y-%m-%d}') for t in ev.t]
    jobs = [(d, [(i, t.isoformat()) for i, t in zip(g.index, g.t)])
            for d, g in ev.groupby('daydir') if os.path.isdir(d)]
    print(f'family {F}: {len(ev)} v2 events, {len(jobs)} day dirs present', flush=True)

    chans = {}
    with Pool(a.procs) as p:
        for k, res in enumerate(p.imap_unordered(process_day, jobs, chunksize=4)):
            for cid, lst in res.items():
                chans.setdefault(cid, []).extend(lst)
            if (k + 1) % 500 == 0:
                print(f'  {k + 1}/{len(jobs)} days', flush=True)

    with h5py.File(out, 'w') as h:
        h.attrs.update(pre=PRE, post=POST, sr=SR, fmin=FMIN, fmax=FMAX, family=F,
                       catalog='v2 (merged file, source != v1_only)')
        g = h.create_group('events')
        g['time_utc'] = ev.time_utc.astype('S26').values
        for c in ['ccsum_v2', 'cc_div_thr_v2', 'meancc_v2', 'nchan_v2']:
            g[c] = ev[c].values.astype('float32')
        g['source'] = ev.source.astype('S8').values
        gc = h.create_group('chan')
        for cid in sorted(chans):
            lst = sorted(chans[cid], key=lambda x: x[0])
            idx = np.array([x[0] for x in lst], dtype='int32')
            # drop duplicate event rows (overlapping traces) keeping the first
            _, keep = np.unique(idx, return_index=True)
            data = np.stack([lst[i][1] for i in keep])
            c = gc.create_group(cid)
            c.create_dataset('data', data=data, chunks=(min(256, len(data)), NPTS), compression='lzf')
            c['ev'] = idx[keep]
    tot = sum(len(v) for v in chans.values())
    print(f'wrote {out}: {len(chans)} channels, {tot} traces', flush=True)


if __name__ == '__main__':
    main()
