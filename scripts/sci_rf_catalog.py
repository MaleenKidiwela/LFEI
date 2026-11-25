#!/usr/bin/env python3
"""Stage 1: teleseismic event catalogue for receiver functions.
USGS FDSN event service (CSV): M>=5.5, epicentral distance 30-95 deg from the MTJ reference point, 2008-01-01..
2026-11-23.  Adds iasp91 (TauP) P travel time and ray parameter at the reference point, back-azimuth, and the
per-station P time correction (p * d(distance)) used to cut station windows.
-> data/sci/rf_timelapse/events.csv
"""
import io
import os
import sys
import time
import urllib.request
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sci_rf_common import OUT, CENTER, T0, T1  # noqa: E402


def fetch():
    rows = []
    years = list(range(int(T0[:4]), int(T1[:4]) + 1))
    for y in years:
        a = max(pd.Timestamp(f'{y}-01-01'), pd.Timestamp(T0)); b = min(pd.Timestamp(f'{y + 1}-01-01'), pd.Timestamp(T1))
        url = ('https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&orderby=time-asc'
               f'&starttime={a:%Y-%m-%dT%H:%M:%S}&endtime={b:%Y-%m-%dT%H:%M:%S}&minmagnitude=5.5'
               f'&latitude={CENTER[0]}&longitude={CENTER[1]}&minradius=30&maxradius=95&eventtype=earthquake')
        for att in range(5):
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    txt = r.read().decode()
                break
            except Exception as e:
                print('retry', y, e, flush=True); time.sleep(5 * 2 ** att)
        d = pd.read_csv(io.StringIO(txt))
        print(y, len(d), flush=True)
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    raw = os.path.join(OUT, 'usgs_raw.csv')
    if not os.path.exists(raw):
        fetch().to_csv(raw, index=False)
    d = pd.read_csv(raw)
    from obspy.geodetics import gps2dist_azimuth, locations2degrees
    from obspy.taup import TauPyModel
    m = TauPyModel('iasp91')
    out = []
    for i, r in d.iterrows():
        dist = locations2degrees(CENTER[0], CENTER[1], r.latitude, r.longitude)
        _, baz, _ = gps2dist_azimuth(CENTER[0], CENTER[1], r.latitude, r.longitude)  # az station->event = back-azimuth
        dep = max(float(r.depth), 0.0)
        arr = m.get_travel_times(dep, dist, phase_list=['P'])
        if not arr:
            continue
        a = arr[0]
        out.append(dict(evid=r.id, time=r.time, lat=r.latitude, lon=r.longitude, depth=dep, mag=r.mag,
                        magtype=r.magType, dist=dist, baz=baz, tP=a.time, p_sdeg=a.ray_param_sec_degree,
                        p_skm=a.ray_param_sec_degree / 111.195, inc=a.incident_angle))
        if i % 500 == 0:
            print(i, len(d), flush=True)
    e = pd.DataFrame(out)
    e['time'] = pd.to_datetime(e.time).dt.tz_localize(None)
    e = e.drop_duplicates('evid').sort_values('time').reset_index(drop=True)
    e.to_csv(os.path.join(OUT, 'events.csv'), index=False)
    print('events', len(e), 'M>=6', (e.mag >= 6).sum())


if __name__ == '__main__':
    main()
