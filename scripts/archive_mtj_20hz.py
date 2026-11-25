#!/usr/bin/env python3
"""20-Hz continuous archive of the 27 LFE-area stations (PB boreholes EH, NC/BK HH with BH fallback)
over their full records, for later re-analysis without re-downloading.

Per station-day: raw day (local data/continuous/<day>/<NET.STA>.mseed if it holds >= 23.9 h on 3
components, else PNWstore for PB 2008-2022, else NCEDC FDSN; same sources as
sci_noise_dvv_sc_full.py) -> M._decimate (merge, split at gaps, demean, short taper, 8-Hz 6-pole
zero-phase low-pass, 100 -> 20 Hz, rounded int32 counts; = the processing of
sci_noise_dvv_sc_full.py / sci_noise_dvv_jewt.py) -> dominant location / instrument, 3 components ->
trimmed to the UTC day -> int32 STEIM2 miniSEED:
  <root>/<NET.STA>/<YYYY>/<NET.STA>.<YYYY>.<DDD>.mseed
root = /data/wsd02/maleen_data/MTJ_20Hz while /data/wsd02 has > 40 GB free, else
/data/wsd03/maleen_MTJ_20Hz (same layout).  StationXML (NCEDC, level=response) in <root>/StationXML/.
manifest.csv in both roots (identical; written by the main process): station, day, path, bytes, disk,
source, status.  Resumable: days in the manifest (or whose file exists) are skipped.

Usage: nice -n 10 python scripts/archive_mtj_20hz.py [--workers 4] [--stations B049 ...]
"""
import argparse
import os
import sys
import time

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/seismic/LFE/scripts')
import sci_noise_dvv_sc_full as S      # noqa: E402  (station list, data access)
M = S.M

ROOTS = ['/data/wsd02/maleen_data/MTJ_20Hz', '/data/wsd03/maleen_MTJ_20Hz']
MIN_FREE = {'/data/wsd02/maleen_data/MTJ_20Hz': 40.0, '/data/wsd03/maleen_MTJ_20Hz': 40.0}
LOCAL = '/home/seismic/LFE/data/continuous'
MANIFEST_COLS = ['station', 'day', 'path', 'bytes', 'disk', 'source', 'status']


def free_gb(path):
    s = os.statvfs(path)
    return s.f_bavail * s.f_frsize / 1e9


def pick_root():
    for r in ROOTS:
        if free_gb(r) > MIN_FREE[r]:
            return r
    return None


def rel_path(net, sta, day):
    return f'{net}.{sta}/{day:%Y}/{net}.{sta}.{day:%Y}.{day:%j}.mseed'


def _local(net, sta, day):
    from obspy import read
    fn = os.path.join(LOCAL, f'{day:%Y-%m-%d}', f'{net}.{sta}.mseed')
    if not os.path.exists(fn):
        return None
    try:
        st = read(fn)
    except Exception:
        return None
    pref = 'EH' if net == 'PB' else 'HH'
    st = st.select(channel=pref + '?')
    comps = {tr.stats.channel for tr in st}
    secs = sum(tr.stats.npts / tr.stats.sampling_rate for tr in st)
    if len(comps) >= 3 and secs >= 3 * 23.9 * 3600:
        return st
    return None


def do_day(job):
    net, sta, dstr = job
    day = pd.Timestamp(dstr)
    rp = rel_path(net, sta, day)
    for r in ROOTS:
        if os.path.exists(os.path.join(r, rp)):
            return dict(station=f'{net}.{sta}', day=dstr, path=os.path.join(r, rp),
                        bytes=os.path.getsize(os.path.join(r, rp)), disk=r, source='existing', status='ok'), 0.0
    t0 = time.time()
    root = pick_root()
    if root is None:
        return dict(station=f'{net}.{sta}', day=dstr, status='nodisk'), 0.0
    st = _local(net, sta, day)
    src = 'local'
    if st is None:
        try:
            st, src = S._fetch(net, sta, day)
        except Exception as e:
            return dict(station=f'{net}.{sta}', day=dstr, status=f'FAILED {str(e)[:40]}'), time.time() - t0
    if st is None or len(st) == 0:
        return dict(station=f'{net}.{sta}', day=dstr, path='', bytes=0, disk='', source=src, status='nodata'), time.time() - t0
    try:
        from obspy import UTCDateTime
        st20 = M._decimate(st)
        del st
        if len(st20) == 0:
            return dict(station=f'{net}.{sta}', day=dstr, path='', bytes=0, disk='', source=src, status='nodata'), time.time() - t0
        loc, cm = S._cmap(st20)
        st20 = st20.select(location=loc)
        st20 = st20.select(channel=cm['Z'][:2] + '?')
        a = UTCDateTime(day)
        st20.trim(a, a + 86400 - 1e-6, nearest_sample=False)       # samples in [00:00, 24:00)
        st20 = st20.__class__([tr for tr in st20 if tr.stats.npts > 0])
        for tr in st20:
            tr.data = np.ascontiguousarray(tr.data, dtype='int32')
        if len(st20) == 0:
            return dict(station=f'{net}.{sta}', day=dstr, path='', bytes=0, disk='', source=src, status='nodata'), time.time() - t0
        fn = os.path.join(root, rp)
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        tmp = fn + '.part'
        st20.write(tmp, format='MSEED', encoding='STEIM2', reclen=4096)
        os.replace(tmp, fn)
        return dict(station=f'{net}.{sta}', day=dstr, path=fn, bytes=os.path.getsize(fn), disk=root, source=src,
                    status='ok'), time.time() - t0
    except Exception as e:
        return dict(station=f'{net}.{sta}', day=dstr, status=f'PROCFAIL {str(e)[:50]}'), time.time() - t0


def do_chunk(jobs):
    return [do_day(j) for j in jobs]


def stationxml(stations):
    from obspy.clients.fdsn import Client
    c = Client('NCEDC', timeout=120)
    for r in ROOTS:
        os.makedirs(os.path.join(r, 'StationXML'), exist_ok=True)
    for net, sta, _, _ in stations:
        fns = [os.path.join(r, 'StationXML', f'{net}.{sta}.xml') for r in ROOTS]
        if all(os.path.exists(f) for f in fns):
            continue
        try:
            inv = c.get_stations(network=net, station=sta, channel='EH?' if net == 'PB' else 'HH?,BH?', level='response')
            for f in fns:
                inv.write(f, format='STATIONXML')
            print('StationXML', net, sta, flush=True)
        except Exception as e:
            print('StationXML FAILED', net, sta, str(e)[:80], flush=True)


def main():
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--stations', nargs='*')
    ap.add_argument('--start')
    ap.add_argument('--end')
    ap.add_argument('--chunk', type=int, default=4)
    ap.add_argument('--report', type=int, default=200)
    a = ap.parse_args()
    for r in ROOTS:
        os.makedirs(r, exist_ok=True)
    stations = [s for s in S.STATIONS if not a.stations or s[1] in a.stations]     # 27 stations incl. BK.JEWT
    stationxml(stations)
    mans = [os.path.join(r, 'manifest.csv') for r in ROOTS]
    done = set()
    for m in mans:
        if os.path.exists(m):
            d = pd.read_csv(m)
            d = d[d.status.isin(['ok', 'nodata'])]
            done |= set(zip(d.station, d.day))
    jobs = []
    for net, sta, d0, d1 in stations:
        jobs += [j for j in S._station_days(net, sta, d0, d1, a.start, a.end) if (f'{net}.{sta}', j[2]) not in done]
    print(f'{len(jobs)} station-days to archive, {a.workers} workers; free wsd02 {free_gb(ROOTS[0]):.0f} GB, '
          f'wsd03 {free_gb(ROOTS[1]):.0f} GB', flush=True)
    mf = []
    for m in mans:
        new = not os.path.exists(m)
        f = open(m, 'a')
        if new:
            f.write(','.join(MANIFEST_COLS) + '\n')
        mf.append(f)
    chunks = [jobs[i:i + a.chunk] for i in range(0, len(jobs), a.chunk)]
    it = iter(chunks)
    t0 = time.time()
    n = nb = nok = 0
    stop = False
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        pend = set()
        while True:
            while not stop and len(pend) < 2 * a.workers:
                c = next(it, None)
                if c is None:
                    break
                pend.add(ex.submit(do_chunk, c))
            if not pend:
                break
            fin, pend = wait(pend, return_when=FIRST_COMPLETED)
            for fu in fin:
                for r, dt in fu.result():
                    if r['status'] == 'nodisk':
                        if not stop:
                            print('STOP: both disks below their free-space limits', flush=True)
                        stop = True
                        continue
                    if r['status'].startswith(('FAILED', 'PROCFAIL')):
                        print(r['station'], r['day'], r['status'], flush=True)
                        continue
                    line = ','.join(str(r.get(c, '')) for c in MANIFEST_COLS) + '\n'
                    for f in mf:
                        f.write(line)
                        f.flush()
                    n += 1
                    if r['status'] == 'ok':
                        nok += 1
                        nb += r['bytes']
                    if n % a.report == 0:
                        el = time.time() - t0
                        print(f"{r['station']} {r['day']} {r['status']} {r.get('source', '')} {dt:.1f}s | done {n} ok {nok} "
                              f"{el / n:.2f} s/day wall, {nb / max(nok, 1) / 1e6:.2f} MB/day, total {nb / 1e9:.1f} GB, "
                              f"disk {r.get('disk', '')}, free wsd02 {free_gb(ROOTS[0]):.0f} GB wsd03 {free_gb(ROOTS[1]):.0f} GB",
                              flush=True)
            if stop:
                for fu in pend:
                    fu.cancel()
    el = time.time() - t0
    print(f'FINISHED {n} station-days ({nok} ok, {nb / 1e9:.1f} GB) in {el / 3600:.1f} h', flush=True)


if __name__ == '__main__':
    main()
