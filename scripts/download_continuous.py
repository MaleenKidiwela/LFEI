#!/usr/bin/env python3
"""Download continuous day-long waveforms for the candidate stations (one instrument per
station, as in download_windows.py) for a date range. One request per station-day.
PB 2008-2022 from PNWstore; everything else from NCEDC.
Output: data/continuous/YYYY-MM-DD/NET.STA.mseed
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from obspy import Stream, UTCDateTime
from obspy.clients.fdsn import Client

import sys
sys.path.insert(0, os.path.dirname(__file__))
from download_windows import channels_for_day, load_channels, read_pnwstore, PNWSTORE_YEARS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--channels', default='data/stations/candidate_channels_65km_2008_2026.csv')
    ap.add_argument('--out', default='data/continuous')
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    chans = load_channels(a.channels)
    jobs = []
    for day in pd.date_range(a.start, a.end, freq='D'):
        nslc = channels_for_day(chans, day)
        for sta in sorted({(n, s) for n, s, _, _ in nslc}):
            jobs.append((day, sta, [x for x in nslc if (x[0], x[1]) == sta]))
    print(f'{len(jobs)} station-days', flush=True)

    def work(job):
        day, (n, s), nslc = job
        od = os.path.join(a.out, f'{day:%Y-%m-%d}')
        fn = os.path.join(od, f'{n}.{s}.mseed')
        if os.path.exists(fn) or os.path.exists(fn + '.empty'):
            return 'skip'
        os.makedirs(od, exist_ok=True)
        t0 = UTCDateTime(day.to_pydatetime())
        st = Stream()
        try:
            if n == 'PB' and day.year in PNWSTORE_YEARS:
                st = read_pnwstore(nslc, [(day, day + pd.Timedelta(days=1))], day.year)
            else:
                cl = Client('NCEDC', timeout=600)
                bulk = [(n, s, l or '--', c, t0, t0 + 86400) for _, _, l, c in nslc]
                for attempt in range(5):
                    try:
                        st = cl.get_waveforms_bulk(bulk)
                        break
                    except Exception as e:
                        if 'No data' in str(e) or '204' in str(e):
                            break
                        time.sleep(5 * 2 ** attempt)
        except Exception as e:
            return f'ERROR {n}.{s} {day:%Y-%m-%d} {e!r}'
        if len(st):
            for tr in st:
                if tr.data.dtype.kind == 'f':
                    tr.data = tr.data.astype('float32')
            st.write(fn, format='MSEED')
            return f'{day:%Y-%m-%d} {n}.{s} {len(st)} tr'
        open(fn + '.empty', 'w').close()
        return f'{day:%Y-%m-%d} {n}.{s} no data'

    with ThreadPoolExecutor(a.workers) as ex:
        for r in ex.map(work, jobs):
            if r != 'skip':
                print(r, flush=True)


if __name__ == '__main__':
    main()
