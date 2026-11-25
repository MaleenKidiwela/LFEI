#!/usr/bin/env python3
"""Stage 2: event-window waveforms for receiver functions (resumable).
For every event in data/sci/rf_timelapse/events.csv and every station active at that time:
  window = station P time (iasp91 at the reference point + p * d(distance)) -35 .. +95 s
  NCEDC FDSN dataselect bulk request, one window per channel (NCEDC returns only one
  window per channel per bulk request, so one request per event); PB 2008-2022 fall-back: PNWstore (local archive; byte ranges from the
  tsindex hourly time index) when NCEDC is missing/incomplete.
-> /data/wsd02/maleen_data/MTJ_RF/wf/{evid}/{pnw,ncedc}.mseed + status.json ({sta: source|none|error})
Usage: sci_rf_download.py [--stations A,B] [--workers 4] [--mag 5.5]
"""
import argparse
import io
import json
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sci_rf_common import OUT, STATIONS, PRE, POST, station_active, coords, ev_dir, CENTER  # noqa: E402

PAD = 5.0
_tl = threading.local()


def pnw_path(p):
    parts = p.split('/')
    return '/auto/pnwstore1-' + parts[2] + '/' + '/'.join(parts[3:])


def read_pnw(sta, t1, t2):
    """PB EH? channels from PNWstore for [t1, t2] (UTCDateTime). Reads only the needed hourly byte range."""
    from obspy import read, Stream
    st = Stream()
    years = sorted({t1.year, t2.year})
    a, b = t1.strftime('%Y-%m-%dT%H:%M:%S'), t2.strftime('%Y-%m-%dT%H:%M:%S')
    for y in years:
        con = sqlite3.connect(f'file:/data/wsd01/PNWstore_sqlite/{y}.sqlite?mode=ro&immutable=1', uri=True)
        rows = con.execute("select channel, filename, byteoffset, bytes, timeindex from tsindex where network='PB' "
                           "and station=? and channel like 'EH_' and starttime<=? and endtime>=?",
                           (sta, b, a)).fetchall()
        con.close()
        for ch, fn, off, nb, tix in rows:
            o0, o1 = off, off + nb
            if tix:
                pairs = [(float(x.split('=>')[0]), int(x.split('=>')[1])) for x in tix.split(',')
                         if '=>' in x and not x.startswith('latest')]
                pairs.sort()
                before = [o for tt, o in pairs if tt <= t1.timestamp - 10]
                after = [o for tt, o in pairs if tt > t2.timestamp + 10]
                if before:
                    o0 = max(before)
                if after:
                    o1 = min(after)
            try:
                with open(pnw_path(fn), 'rb') as f:
                    f.seek(o0)
                    buf = f.read(o1 - o0)
                s = read(io.BytesIO(buf), format='MSEED', starttime=t1, endtime=t2)
                st += s.select(channel=ch)
            except Exception:
                pass
    if len(st):
        st.merge(method=1, fill_value=None)
        st.trim(t1, t2)
    return st


def ncedc_client():
    if not hasattr(_tl, 'c'):
        from obspy.clients.fdsn import Client
        _tl.c = Client('NCEDC', timeout=180)
    return _tl.c


def fetch_ncedc(reqs):
    from obspy import Stream
    if not reqs:
        return Stream(), None
    c = ncedc_client()
    for att in range(5):
        try:
            return c.get_waveforms_bulk(reqs), None
        except Exception as e:
            msg = str(e)
            if 'No data' in msg or '204' in msg:
                return Stream(), None
            time.sleep(5 * 2 ** att)
    return Stream(), 'error'


def station_windows(ev, C):
    from obspy import UTCDateTime
    from obspy.geodetics import locations2degrees
    t0 = UTCDateTime(pd.Timestamp(ev.time).to_pydatetime())
    out = {}
    for code, (la, lo, _, _) in C.items():
        if not station_active(code, ev.time):
            continue
        d = locations2degrees(la, lo, ev.lat, ev.lon)
        tp = t0 + ev.tP + ev.p_sdeg * (d - ev.dist)
        out[code] = (tp - PRE - PAD, tp + POST + PAD)
    return out


def do_event(ev, C, stations):
    from obspy import read, Stream
    d = ev_dir(ev.evid)
    os.makedirs(d, exist_ok=True)
    sfn = os.path.join(d, 'status.json')
    status = json.load(open(sfn)) if os.path.exists(sfn) else {}
    win = {k: v for k, v in station_windows(ev, C).items() if k in stations and status.get(k) in (None, 'error')}
    if not win:
        return ev.evid, 0, 'skip'
    # NCEDC first (one bulk request per event; PB samples identical to PNWstore, NOTES 2026-11), PNWstore as the
    # fall-back for PB 2008-2022 windows missing / incomplete at NCEDC (e.g. B933 2022)
    pnw = Stream()
    nc = [(*code.split('.'), '*', STATIONS[code][2], t1, t2) for code, (t1, t2) in win.items()]
    st, err = fetch_ncedc(nc)
    for code, (t1, t2) in win.items():
        net, sta = code.split('.')
        s = st.select(network=net, station=sta)
        ok = len({tr.stats.channel for tr in s}) >= 3 and sum(tr.stats.npts for tr in s) > 3 * 0.9 * (t2 - t1) * 100
        if not ok and net == 'PB' and t1.year <= 2022:
            try:
                sp = read_pnw(sta, t1, t2)
            except Exception:
                sp = Stream()
            if len({tr.stats.channel for tr in sp}) == 3 and sum(tr.stats.npts for tr in sp) > sum(tr.stats.npts for tr in s):
                for tr in s:
                    st.remove(tr)
                pnw += sp; status[code] = 'pnw'; continue
        status[code] = 'ncedc' if len(s) else ('error' if err else 'none')
    for name, s in [('pnw', pnw), ('ncedc', st)]:
        if len(s):
            fn = os.path.join(d, f'{name}.mseed')
            if os.path.exists(fn):
                old = read(fn)
                have = {(tr.stats.network, tr.stats.station) for tr in s}
                s = s + Stream([tr for tr in old if (tr.stats.network, tr.stats.station) not in have])
            s = s.split()       # gaps (masked arrays) -> separate traces
            for tr in s:
                if tr.data.dtype.kind == 'f':
                    tr.data = tr.data.astype(np.float32)
            s.write(fn + '.tmp', format='MSEED')
            os.replace(fn + '.tmp', fn)
    json.dump(status, open(sfn + '.tmp', 'w')); os.replace(sfn + '.tmp', sfn)
    return ev.evid, len(win), f"pnw{sum(v == 'pnw' for v in status.values())} nc{sum(v == 'ncedc' for v in status.values())}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stations', default='all')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--mag', type=float, default=5.5)
    ap.add_argument('--order', default='mag')     # download larger events first (most useful)
    a = ap.parse_args()
    stations = list(STATIONS) if a.stations == 'all' else a.stations.split(',')
    E = pd.read_csv(os.path.join(OUT, 'events.csv'), parse_dates=['time'])
    E = E[E.mag >= a.mag]
    if a.order == 'mag':
        E = E.sort_values('mag', ascending=False)
    C = {k: v for k, v in coords().items() if k in stations}
    print(len(E), 'events', len(C), 'stations', flush=True)
    t0 = time.time(); n = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(do_event, ev, C, stations) for ev in E.itertuples()]
        for f in as_completed(futs):
            n += 1
            try:
                evid, nw, msg = f.result()
            except Exception as e:
                evid, nw, msg = '?', 0, f'EXC {e}'
            if n % 50 == 0 or msg.startswith('EXC'):
                print(f'{n}/{len(E)} {evid} {nw} {msg} {time.time() - t0:.0f}s', flush=True)
    print('done', time.time() - t0)


if __name__ == '__main__':
    main()
