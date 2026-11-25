#!/usr/bin/env python3
"""EQ-above step 3 input: NCSN focal mechanisms (FPFIT, via USGS ComCat 'focal-mechanism' products) for
15-35 km earthquakes within 40 km of the LFEs (40.06 N 123.62 W), 2008-2026.
In : data/sci/eq_above/raw/usgs_fm.json (ComCat geojson summary, producttype=focal-mechanism)
Out: data/sci/eq_above/focal_mechanisms.csv (one row per event: preferred focal-mechanism product)
"""
import json
import time
import urllib.request

import numpy as np
import pandas as pd

ROOT = '/home/seismic/LFE'
OUT = f'{ROOT}/data/sci/eq_above'


def get(url):
    for k in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception:                         # noqa: BLE001
            time.sleep(3 * (k + 1))
    return None


def main():
    d = json.load(open(f'{OUT}/raw/usgs_fm.json'))
    rows = []
    for f in d['features']:
        lo, la, z = f['geometry']['coordinates']
        r = np.hypot((la - 40.06) * 111.2, (lo + 123.62) * 111.2 * np.cos(np.radians(40.06)))
        if not (r < 40 and 14 <= z <= 35):
            continue
        det = get(f['properties']['detail'])
        if det is None:
            continue
        fm = det['properties']['products'].get('focal-mechanism', [])
        if not fm:
            continue
        p = fm[0]['properties']
        g = lambda k: float(p[k]) if k in p and p[k] not in (None, '') else np.nan   # noqa: E731
        rows.append(dict(id=f['id'], time=pd.to_datetime(f['properties']['time'], unit='ms'), lat=la, lon=lo, depth=z,
                         mag=f['properties']['mag'], r_km=r, source=fm[0].get('source'),
                         strike1=g('nodal-plane-1-strike'), dip1=g('nodal-plane-1-dip'), rake1=g('nodal-plane-1-rake'),
                         strike2=g('nodal-plane-2-strike'), dip2=g('nodal-plane-2-dip'), rake2=g('nodal-plane-2-rake'),
                         p_az=g('p-axis-azimuth'), p_pl=g('p-axis-plunge'), t_az=g('t-axis-azimuth'),
                         t_pl=g('t-axis-plunge'), nsta=g('num-stations-used'),
                         misfit=g('fault-plane-solution-misfit'), stdr=g('station-distribution-ratio'),
                         quality=p.get('quality'), errstrike=g('fault-plane-solution-error-strike'),
                         errdip=g('fault-plane-solution-error-dip'), errrake=g('fault-plane-solution-error-rake')))
        time.sleep(0.2)
    df = pd.DataFrame(rows).sort_values('time')
    df.to_csv(f'{OUT}/focal_mechanisms.csv', index=False)
    print(len(df), df.quality.value_counts().to_dict() if 'quality' in df else '')


if __name__ == '__main__':
    main()
