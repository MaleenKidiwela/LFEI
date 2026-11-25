#!/usr/bin/env python3
"""EQ-above step 1: download event waveforms (NCEDC FDSN, anonymous) for ordinary earthquakes near the LFEs.

Catalog: data/sci/eq_above/raw/ncedc_cat_all.txt (NCEDC fdsnws/event, all magnitudes, 2008-2026,
39.70-40.45 N, 124.10-123.10 W, depth 5-50 km).  Selection: epicentre within R_MAX km of 40.06 N 123.62 W,
depth 14-35 km (+ optional extra event table, e.g. matched-filter detections, via --extra csv with EventID,Time).
One bulk request per event (NCEDC returns one window per channel per request).  Window origin -5 .. +45 s.
Output data/sci/eq_above/wf/{evid}.mseed.  Stops if /home free < 25 GB.
Usage: nice -n 10 python scripts/sci_eq_above_dl.py [threads]
"""
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

ROOT = '/home/seismic/LFE'
OUT = f'{ROOT}/data/sci/eq_above'
BIG = '/data/wsd02/maleen_data/eq_above'          # bulk waveforms (kept off /home)
R_MAX = 30.0
LAT0, LON0 = 40.06, -123.62
CH = ([('NC', s, '*', 'HH?') for s in ('KMR', 'KMPB', 'KCPB', 'KCPC', 'KZE', 'KNR')] +
      [('NC', s, '*', 'EHZ') for s in ('KBS', 'KBSB', 'KIP', 'KBN', 'KCR', 'KSM', 'KFP', 'KKP', 'KPP', 'KPPB', 'KCS',
                                         'KCSB', 'KBU', 'KPR', 'KLR', 'KTJ', 'KCO')] +
      [('PB', s, '*', 'EH?') for s in ('B045', 'B046', 'B047', 'B049', 'B932', 'B933', 'B934', 'B935')] +
      [('BK', s, '*', 'HH?') for s in ('JEWT', 'SPRL', 'BRIC', 'USAL', 'RVIT', 'HALS', 'BLAS', 'PRDS', 'GHOP', 'HUNT',
                                         'RBOW', 'HAYF', 'BJES')])


def read_cat():
    c = pd.read_csv(f'{OUT}/raw/ncedc_cat_all.txt', sep='|')
    c.columns = [x.strip().lstrip('#').strip() for x in c.columns]
    c['r'] = np.hypot((c.Latitude - LAT0) * 111.2, (c.Longitude - LON0) * 111.2 * np.cos(np.radians(LAT0)))
    c['depth'] = c['Depth/km']
    return c


def select():
    c = read_cat()
    return c[(c.r < R_MAX) & (c.depth >= 14) & (c.depth <= 35)].reset_index(drop=True)


def fetch(evid, t, client, wdir):
    f = f'{wdir}/{evid}.mseed'
    if os.path.exists(f) or os.path.exists(f + '.none'):
        return 'skip'
    t0 = UTCDateTime(str(t))
    bulk = [(n, s, l, c, t0 - 5, t0 + 45) for n, s, l, c in CH]
    for k in range(3):
        try:
            st = client.get_waveforms_bulk(bulk)
            st.write(f, format='MSEED')
            return 'ok'
        except Exception as e:                      # noqa: BLE001
            if 'No data' in str(e) or '204' in str(e):
                open(f + '.none', 'w').close()
                return 'none'
            time.sleep(5 * (k + 1))
    return 'fail'


def main():
    nthr = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    extra = sys.argv[2] if len(sys.argv) > 2 else None
    wdir = f'{BIG}/wf'
    if extra:
        e = pd.read_csv(extra)
        rows = list(zip(e.EventID.astype(str), e.Time))
        wdir = f'{BIG}/wf_mf'
    else:
        c = select()
        c.to_csv(f'{OUT}/catalog_selected.csv', index=False)
        rows = list(zip(c.EventID.astype(str), c.Time))
    os.makedirs(wdir, exist_ok=True)
    print(len(rows), 'events', flush=True)
    client = Client('NCEDC', timeout=180)
    t0 = time.time()
    n = 0
    with ThreadPoolExecutor(nthr) as ex:
        for k in range(0, len(rows), 40):
            if shutil.disk_usage('/home').free / 1e9 < 25:
                print('STOP: /home free < 25 GB', flush=True)
                break
            res = list(ex.map(lambda r: fetch(r[0], r[1], client, wdir), rows[k:k + 40]))
            n += len(res)
            print(f'{n}/{len(rows)} {pd.Series(res).value_counts().to_dict()} {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
