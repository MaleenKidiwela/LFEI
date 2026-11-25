#!/usr/bin/env python3
"""Long-period inter-station ambient-noise dv/v: download 1-sps LH? data (NCEDC FDSN, no key).

Per station x calendar month: one dataselect request (all LH? channels, all location codes), then the
stream is merged and re-packed into fully packed Steim2 day files (NCEDC/BK LH records are sparsely filled:
~4 MB/day as served -> ~0.6 MB/day re-packed, lossless).

Raw:  /data/wsd02/maleen_data/MTJ_LH/raw/NET.STA/YYYY/NET.STA.YYYY-MM-DD.mseed
Resp: /data/wsd02/maleen_data/MTJ_LH/resp/NET.STA.xml   (level=response, LH?)
Log:  /data/wsd02/maleen_data/MTJ_LH/download_log.csv ; month markers raw/NET.STA/.done/YYYY-MM
<= 4 concurrent requests (shared machine / shared server).
"""
import os
import sys
import time
import threading
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd

BASE = '/data/wsd02/maleen_data/MTJ_LH'
RAW = os.path.join(BASE, 'raw')
RESP = os.path.join(BASE, 'resp')
ROOT = '/home/seismic/LFE'
CHLIST = os.path.join(ROOT, 'data/sci/noise_lp/lhbh_channels_ncedc_r1.3.txt')
T0, T1 = pd.Timestamp('2008-01-01'), pd.Timestamp('2026-11-24')
DS = 'https://service.ncedc.org/fdsnws/dataselect/1/query'
STQ = 'https://service.ncedc.org/fdsnws/station/1/query'

STATIONS = [  # (net, sta)  broadband within ~80 km of 40.06N -123.62 + long-record JCC/GASB; PB boreholes (LH, 2 Hz)
    ('BK', 'JEWT'), ('BK', 'SPRL'), ('BK', 'BRIC'), ('BK', 'USAL'), ('BK', 'RVIT'), ('BK', 'HALS'), ('BK', 'BLAS'),
    ('BK', 'PRDS'), ('BK', 'GHOP'), ('BK', 'HUNT'), ('BK', 'RBOW'), ('BK', 'BJES'), ('BK', 'INGL'), ('BK', 'HAYF'),
    ('BK', 'ETSL'), ('BK', 'SHWD'), ('BK', 'PETL'), ('BK', 'WLKR'), ('BK', 'DCMP'), ('BK', 'PETY'), ('BK', 'DMOR'),
    ('BK', 'DLIK'), ('BK', 'KNEE'), ('BK', 'JCC'), ('BK', 'GASB'),
    ('NC', 'KMR'), ('NC', 'KZE'), ('NC', 'KCPB'), ('NC', 'KCPC'), ('NC', 'KHBB'), ('NC', 'KCT'), ('NC', 'KMPB'),
    ('TA', 'O02D'),
    ('PB', 'B934'), ('PB', 'B049'), ('PB', 'B933'), ('PB', 'B935'), ('PB', 'B047'), ('PB', 'B045'), ('PB', 'B932'),
    ('PB', 'B046'),
]
_lock = threading.Lock()


def epochs():
    d = pd.read_csv(CHLIST, sep='|', dtype=str)
    d.columns = [c.strip('#').strip() for c in d.columns]
    d = d[d.Channel.str.startswith('LH')]
    d['t0'] = pd.to_datetime(d.StartTime.str[:19])
    d['t1'] = pd.to_datetime(d.EndTime.str[:19].str.replace('3000-01-01T00:00:00', '2100-01-01T00:00:00'))
    return d


def tasks():
    d = epochs()
    out = []
    for n, s in STATIONS:
        e = d[(d.Network == n) & (d.Station == s)]
        if e.empty:
            print('no LH epochs', n, s)
            continue
        months = pd.period_range(T0, T1, freq='M')
        for m in months:
            a, b = m.start_time, m.end_time
            if ((e.t0 <= b) & (e.t1 >= a)).any():
                prio = 0 if pd.Timestamp('2024-11-01') <= a <= pd.Timestamp('2025-01-31') else 1
                out.append((prio, n, s, m))
    out.sort(key=lambda x: (x[0], x[1] == 'PB', x[3], x[1], x[2]))   # PB last: NCEDC PB LH requests hang since 2026-11-25
    return out


def fetch(url, tries=5):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.getcode(), r.read()
        except urllib.error.HTTPError as e:
            if e.code in (204, 404):
                return e.code, b''
            time.sleep(10 * (k + 1))
        except Exception:
            time.sleep(10 * (k + 1))
    return -1, b''


def get_resp(n, s):
    os.makedirs(RESP, exist_ok=True)
    fn = os.path.join(RESP, f'{n}.{s}.xml')
    if os.path.exists(fn) and os.path.getsize(fn) > 1000:
        return
    code, b = fetch(f'{STQ}?net={n}&sta={s}&cha=LH?&level=response')
    if code == 200:
        open(fn, 'wb').write(b)


def do_month(t):
    from obspy import read, UTCDateTime
    import io
    _, n, s, m = t
    sd = os.path.join(RAW, f'{n}.{s}')
    mark = os.path.join(sd, '.done', f'{m}')
    if os.path.exists(mark):
        return
    a = m.start_time
    b = m.end_time.normalize() + pd.Timedelta(days=1)
    url = f'{DS}?net={n}&sta={s}&loc=*&cha=LH?&start={a:%Y-%m-%dT%H:%M:%S}&end={b:%Y-%m-%dT%H:%M:%S}'
    t0 = time.time()
    code, buf = fetch(url)
    nday = 0
    if code == 200 and len(buf) > 0:
        try:
            st = read(io.BytesIO(buf))
        except Exception as e:
            print('read fail', n, s, m, e, '-> per-day requests', flush=True)
            from obspy import Stream
            st = Stream()
            for day in pd.date_range(a, b - pd.Timedelta(days=1), freq='D'):
                u = (f'{DS}?net={n}&sta={s}&loc=*&cha=LH?&start={day:%Y-%m-%dT%H:%M:%S}'
                     f'&end={day + pd.Timedelta(days=1):%Y-%m-%dT%H:%M:%S}')
                c2, b2 = fetch(u)
                if c2 == 200 and b2:
                    try:
                        st += read(io.BytesIO(b2))
                    except Exception as e2:
                        print('  day read fail', n, s, day.date(), e2, flush=True)
        if st is not None:
            for tr in st:
                tr.data = tr.data.astype('int32')
            st.merge(method=1, fill_value=None)
            for day in pd.date_range(a, b - pd.Timedelta(days=1), freq='D'):
                u0 = UTCDateTime(day.to_pydatetime())
                d = st.slice(u0, u0 + 86400 - 0.001).copy()
                d = d.split()
                d.traces = [tr for tr in d if tr.stats.npts > 600]
                if len(d) == 0:
                    continue
                fd = os.path.join(sd, f'{day:%Y}')
                os.makedirs(fd, exist_ok=True)
                for tr in d:
                    tr.data = np.ascontiguousarray(tr.data, dtype='int32')
                d.write(os.path.join(fd, f'{n}.{s}.{day:%Y-%m-%d}.mseed'), format='MSEED', encoding='STEIM2',
                        reclen=4096)
                nday += 1
    if code in (200, 204, 404):
        os.makedirs(os.path.dirname(mark), exist_ok=True)
        open(mark, 'w').write(f'{code} {nday}\n')
    with _lock:
        with open(os.path.join(BASE, 'download_log.csv'), 'a') as f:
            f.write(f'{n},{s},{m},{code},{len(buf)},{nday},{time.time() - t0:.1f}\n')


def main():
    os.makedirs(RAW, exist_ok=True)
    for n, s in STATIONS:
        get_resp(n, s)
    T = tasks()
    T = [t for t in T if not os.path.exists(os.path.join(RAW, f'{t[1]}.{t[2]}', '.done', f'{t[3]}'))]
    print('tasks', len(T), flush=True)
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    with ProcessPoolExecutor(min(nw, 4)) as ex:
        for i, _ in enumerate(ex.map(do_month, T, chunksize=1)):
            if i % 50 == 0:
                print(time.strftime('%H:%M:%S'), i, len(T), flush=True)


if __name__ == '__main__':
    main()
