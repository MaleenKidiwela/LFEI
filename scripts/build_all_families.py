#!/usr/bin/env python3
"""One-pass version of build_dataset.py for many families at once.

build_dataset.py reads and filters every day file of the shared all-family archive for EACH
family (each day file holds all families' windows), i.e. ~N_families x the necessary work.
Here each day is read + filtered once and windows are cut for every requested family.
Output files/format are identical to build_dataset.py (data/waveforms/fam{F}.h5): events
group (v2 rows, sorted by time) + chan/{id}/data (n x NPTS float32, sorted by event index,
unique) + chan/{id}/ev. Days are processed in time-ordered blocks and appended to resizable
datasets, so memory stays bounded. A marker fam{F}.h5.done is written when a file is complete.

Usage: python build_all_families.py --families 1 2 3 ... --procs 24
"""
import argparse
import glob
import os
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
from obspy import UTCDateTime, read

from build_dataset import PRE, POST, SR, FMIN, FMAX, NPTS


def process_day(args):
    daydir, fam_rows = args          # fam_rows: {F: [(idx, iso_time), ...]}
    out = {}
    for f in glob.glob(os.path.join(daydir, '*.mseed')):
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
        st = [tr for tr in st if tr.stats.sampling_rate == SR and tr.stats.npts >= NPTS]
        for F, rows in fam_rows.items():
            for idx, t in rows:
                t0 = UTCDateTime(t) + PRE
                for tr in st:
                    s = tr.stats
                    if s.starttime > t0 + 0.5 / SR or s.endtime < t0 + (NPTS - 1) / SR - 0.5 / SR:
                        continue
                    i0 = int(round((t0 - s.starttime) * SR))
                    x = tr.data[i0:i0 + NPTS]
                    if len(x) != NPTS or not np.isfinite(x).all() or np.std(x) == 0:
                        continue
                    out.setdefault(F, {}).setdefault(tr.id, []).append((idx, x.astype('float32')))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--families', type=int, nargs='+', required=True)
    ap.add_argument('--catalog', default='data/merged/LFEcat_MTJ_merged_v1v2.csv')
    ap.add_argument('--wfdir', default='data/waveforms/allfam_raw')
    ap.add_argument('--procs', type=int, default=24)
    ap.add_argument('--block-days', type=int, default=200)
    ap.add_argument('--outdir', default='data/waveforms')
    a = ap.parse_args()

    cat = pd.read_csv(a.catalog)
    cat = cat[cat.source != 'v1_only']
    cat['t'] = pd.to_datetime(cat.time_utc)
    evs, h5 = {}, {}
    for F in a.families:
        ev = cat[cat.idnum == F].sort_values('t').reset_index(drop=True)   # same rows/order as build_dataset
        evs[F] = ev
        out = os.path.join(a.outdir, f'fam{F}.h5')
        if os.path.exists(out + '.done'):
            os.remove(out + '.done')
        h = h5py.File(out, 'w')
        h.attrs.update(pre=PRE, post=POST, sr=SR, fmin=FMIN, fmax=FMAX, family=F,
                       catalog='v2 (merged file, source != v1_only)', builder='build_all_families.py')
        g = h.create_group('events')
        g['time_utc'] = ev.time_utc.astype('S26').values
        for c in ['ccsum_v2', 'cc_div_thr_v2', 'meancc_v2', 'nchan_v2']:
            g[c] = ev[c].values.astype('float32')
        g['source'] = ev.source.astype('S8').values
        h.create_group('chan')
        h5[F] = h

    # day -> {F: rows}
    days = {}
    for F, ev in evs.items():
        dd = [os.path.join(a.wfdir, f'{t:%Y}', f'{t:%Y-%m-%d}') for t in ev.t]
        for d, g in ev.groupby(pd.Series(dd, index=ev.index)):
            days.setdefault(d, {})[F] = [(i, t.isoformat()) for i, t in zip(g.index, g.t)]
    jobs = [(d, days[d]) for d in sorted(days) if os.path.isdir(d)]
    print(f'{len(a.families)} families, {sum(len(e) for e in evs.values())} events, {len(jobs)} day dirs', flush=True)

    ntr = {F: 0 for F in a.families}
    with Pool(a.procs) as p:
        for b0 in range(0, len(jobs), a.block_days):
            block = {}
            for res in p.imap_unordered(process_day, jobs[b0:b0 + a.block_days], chunksize=2):
                for F, chans in res.items():
                    for cid, lst in chans.items():
                        block.setdefault(F, {}).setdefault(cid, []).extend(lst)
            # append this time block (later blocks are later days -> larger event indices)
            for F, chans in block.items():
                gc = h5[F]['chan']
                for cid, lst in chans.items():
                    lst.sort(key=lambda x: x[0])
                    idx = np.array([x[0] for x in lst], dtype='int32')
                    _, keep = np.unique(idx, return_index=True)
                    data = np.stack([lst[i][1] for i in keep])
                    idx = idx[keep]
                    if cid not in gc:
                        c = gc.create_group(cid)
                        c.create_dataset('data', data=data, maxshape=(None, NPTS), chunks=(64, NPTS), compression='lzf')
                        c.create_dataset('ev', data=idx, maxshape=(None,), chunks=(4096,))
                    else:
                        c = gc[cid]
                        if len(c['ev']) and idx[0] <= c['ev'][-1]:     # cannot happen with time-ordered days; guard anyway
                            m = idx > c['ev'][-1]
                            data, idx = data[m], idx[m]
                            if not len(idx):
                                continue
                        n0 = len(c['ev'])
                        c['data'].resize(n0 + len(idx), axis=0)
                        c['data'][n0:] = data
                        c['ev'].resize(n0 + len(idx), axis=0)
                        c['ev'][n0:] = idx
                    ntr[F] += len(idx)
                h5[F].flush()
            print(f'  {min(b0 + a.block_days, len(jobs))}/{len(jobs)} days', flush=True)

    for F, h in h5.items():
        n = len(h['chan'])
        h.close()
        open(os.path.join(a.outdir, f'fam{F}.h5.done'), 'w').write(f'{n} channels {ntr[F]} traces\n')
        print(f'wrote {a.outdir}/fam{F}.h5: {n} channels, {ntr[F]} traces', flush=True)


if __name__ == '__main__':
    main()
