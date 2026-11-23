#!/usr/bin/env python3
"""Download waveform windows around LFE detections for all candidate stations.

For each day with detections:
  * build windows [t + PRE, t + POST] around each catalog time, merge overlapping ones
  * pick one instrument per station active that day (HH > BH > EH > SH > EP > DP)
  * PB stations for 2008-2022: read from local PNWstore (ROVER archive on pnwstore1)
  * everything else: NCEDC FDSN dataselect (bulk, anonymous); EarthScope as fallback for PB
  * write raw miniSEED: OUT/YYYY/YYYY-MM-DD/NET.STA.mseed  (+ a per-day JSON log)

Usage:
  python download_windows.py --start 2022-10-01 --end 2022-10-07 [--catalog ...] [--workers 4]
"""
import argparse
import json
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from obspy import Stream, UTCDateTime, read
from obspy.clients.fdsn import Client

PREF = ['HH', 'BH', 'EH', 'SH', 'EP', 'DP']
PNWSTORE_IDX = '/data/wsd01/PNWstore_sqlite/{year}.sqlite'
PNWSTORE_YEARS = range(2008, 2023)   # years where PB is in PNWstore


def pnw_path(p):
    """Index stores /data/wdXX/...; this machine sees it as /auto/pnwstore1-wdXX/..."""
    parts = p.split('/')
    return '/auto/pnwstore1-' + parts[2] + '/' + '/'.join(parts[3:])


def load_catalog(path):
    d = pd.read_csv(path)
    if 'time_utc' in d.columns:
        d['t'] = pd.to_datetime(d.time_utc)
    else:
        d.columns = ['year', 'month', 'day', 'hour', 'minute', 'second', 's_of_day', 'ccsum',
                     'thresh', 'cc_div_thr', 'meancc', 'nchan', 'ID', 'idnum']
        d['t'] = pd.to_datetime(dict(year=d.year, month=d.month, day=d.day)) + \
            pd.to_timedelta(d.s_of_day, unit='s')
    return d


def load_channels(path):
    c = pd.read_csv(path, dtype={'loc': str}, keep_default_na=False)
    c['start'] = pd.to_datetime(c.start)
    c['end'] = pd.to_datetime(c.end.replace('', '2100-01-01').str.replace('^(2599|3000)', '2100', regex=True))
    return c


def channels_for_day(c, day):
    """One instrument (band code + loc) per station, all its components active that day."""
    t0, t1 = day, day + pd.Timedelta(days=1)
    act = c[(c.start < t1) & (c.end > t0)]
    out = []
    for (n, s), g in act.groupby(['net', 'sta']):
        for band in PREF:
            gg = g[g.cha.str[:2] == band]
            if gg.empty:
                continue
            loc = sorted(gg['loc'].unique())[0]
            gg = gg[gg['loc'] == loc].drop_duplicates('cha')
            out += [(n, s, loc, ch) for ch in sorted(gg.cha)]
            break
    return out


def merge_windows(times, pre, post, gap=0.0):
    """Merge event windows; windows closer than `gap` s are joined (fewer requests)."""
    w = sorted((t + pd.Timedelta(seconds=pre), t + pd.Timedelta(seconds=post)) for t in times)
    merged = [list(w[0])]
    for a, b in w[1:]:
        if a <= merged[-1][1] + pd.Timedelta(seconds=gap):
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def read_pnwstore(nslc, windows, year):
    con = sqlite3.connect(f'file:{PNWSTORE_IDX.format(year=year)}?mode=ro&immutable=1', uri=True)
    st = Stream()
    q = ("select filename from tsindex where network=? and station=? and location=? "
         "and channel=? and starttime<=? and endtime>=?")
    for n, s, l, ch in nslc:
        # each day file is read ONCE and all windows are cut from it (was: one read per window)
        files = {}
        for a, b in windows:
            for r in con.execute(q, (n, s, l, ch, b.isoformat(), a.isoformat())):
                files.setdefault(r[0], []).append((a, b))
        for f, wins in files.items():
            try:
                full = read(pnw_path(f), sourcename=f'{n}.{s}.{l}.{ch}')
            except Exception:
                continue
            for a, b in wins:
                st += full.slice(UTCDateTime(a), UTCDateTime(b)).copy()
    con.close()
    return st


def fetch_fdsn(client, nslc, windows):
    """One bulk request per time window (all channels). NCEDC's bulk dataselect returns
    only ONE time window per channel per request, so windows must not share a request."""
    st = Stream()
    if not nslc:
        return st
    for a, b in windows:
        bulk = [(n, s, l or '--', ch, UTCDateTime(a), UTCDateTime(b)) for n, s, l, ch in nslc]
        for attempt in range(5):
            try:
                st += client.get_waveforms_bulk(bulk)
                break
            except Exception as e:
                if 'No data' in str(e) or '204' in str(e):
                    break
                time.sleep(5 * 2 ** attempt)
        else:
            raise RuntimeError(f'failed after retries: window {a}')
    return st


def do_day(day, ev_times, chans, args, clients):
    outdir = os.path.join(args.out, f'{day:%Y}', f'{day:%Y-%m-%d}')
    logf = os.path.join(outdir, '_log.json')
    if os.path.exists(logf) and not args.overwrite:
        return json.load(open(logf))
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    windows = merge_windows(ev_times, args.pre, args.post, args.gap)
    nslc = channels_for_day(chans, day)
    local = [x for x in nslc if x[0] == 'PB' and day.year in PNWSTORE_YEARS]
    remote = [x for x in nslc if x not in local]
    st = Stream()
    if local:
        st += read_pnwstore(local, windows, day.year)
    st += fetch_fdsn(clients['NCEDC'], remote, windows)
    # PB fallback to EarthScope for anything missing
    have = {(tr.stats.network, tr.stats.station, tr.stats.channel) for tr in st}
    missing_pb = [x for x in nslc if x[0] == 'PB' and (x[0], x[1], x[3]) not in have]
    if missing_pb and clients.get('EARTHSCOPE'):
        st += fetch_fdsn(clients['EARTHSCOPE'], missing_pb, windows)
    for tr in st:   # PB headers sometimes say 99.99999237 Hz -> merge refuses; snap to nominal
        sr = tr.stats.sampling_rate
        if abs(sr - round(sr)) < 1e-4 * sr:
            tr.stats.sampling_rate = float(round(sr))
    st.merge(method=1, fill_value=None)
    st = st.split()
    nbytes = 0
    for n, s in sorted({(tr.stats.network, tr.stats.station) for tr in st}):
        sub = st.select(network=n, station=s)
        fn = os.path.join(outdir, f'{n}.{s}.mseed')
        for tr in sub:
            if tr.data.dtype.kind == 'f':
                tr.data = tr.data.astype('float32')
        sub.write(fn, format='MSEED')
        nbytes += os.path.getsize(fn)
    got = sorted({f'{tr.stats.network}.{tr.stats.station}.{tr.stats.location}.{tr.stats.channel}' for tr in st})
    log = dict(day=f'{day:%Y-%m-%d}', n_events=len(ev_times), n_windows=len(windows),
               window_s=float(sum((b - a).total_seconds() for a, b in windows)),
               requested=[f'{n}.{s}.{l}.{c}' for n, s, l, c in nslc], received=got,
               n_requested=len(nslc), n_received=len(got), bytes=nbytes,
               seconds=round(time.time() - t0, 1))
    json.dump(log, open(logf, 'w'), indent=1)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--catalog', default='data/merged/LFEcat_MTJ_merged_v1v2.csv')
    ap.add_argument('--channels', default='data/stations/candidate_channels_65km_2008_2026.csv')
    ap.add_argument('--out', default='data/waveforms/raw')
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--pre', type=float, default=-15.0)
    ap.add_argument('--post', type=float, default=45.0)
    ap.add_argument('--family', type=int, default=None, help='only this idnum')
    ap.add_argument('--exclude-family', type=int, nargs='*', default=[], help='skip these idnums')
    ap.add_argument('--gap', type=float, default=0.0, help='join windows closer than this (s)')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()

    cat = load_catalog(a.catalog)
    chans = load_channels(a.channels)
    days = pd.date_range(a.start, a.end, freq='D')
    cat = cat[(cat.t >= days[0]) & (cat.t < days[-1] + pd.Timedelta(days=1))]
    if a.family is not None:
        cat = cat[cat.idnum == a.family]
    if a.exclude_family:
        cat = cat[~cat.idnum.isin(a.exclude_family)]
    groups = {d: g.t.tolist() for d, g in cat.groupby(cat.t.dt.normalize())}

    def make_clients():
        # NC, BK and PB are all served anonymously by NCEDC; EarthScope (public PB data,
        # no token needed as of 2026-11) is only a fallback for missing PB channels.
        return {'NCEDC': Client('NCEDC', timeout=300),
                'EARTHSCOPE': Client('https://service.earthscope.org', timeout=300)}

    import threading
    local = threading.local()

    def worker(d):
        if not hasattr(local, 'clients'):
            local.clients = make_clients()
        try:
            return do_day(d, groups[d], chans, a, local.clients)
        except Exception as e:
            return dict(day=f'{d:%Y-%m-%d}', error=repr(e))

    with ThreadPoolExecutor(a.workers) as ex:
        for log in ex.map(worker, sorted(groups)):
            if 'error' in log:
                print(f"{log['day']}  ERROR {log['error']}", flush=True)
            else:
                print(f"{log['day']}  ev={log['n_events']:4d} win={log['n_windows']:4d} "
                      f"({log['window_s']/60:5.1f} min)  ch {log['n_received']}/{log['n_requested']}  "
                      f"{log['bytes']/1e6:6.1f} MB  {log['seconds']:6.1f} s", flush=True)


if __name__ == '__main__':
    main()
