"""Download the public PNSN tremor catalog month by month (2009-08 .. 2026-11), keep raw JSON on
/data/wsd02, and write southern Cascadia tremor (lat < 42.2) to data/sci/ducellier/pnsn_tremor_south.csv.
<=4 concurrent requests. Run with python -I."""
import json, os, time, requests, pandas as pd
from concurrent.futures import ThreadPoolExecutor
RAW = '/data/wsd02/maleen_data/ducellier/pnsn_tremor'
OUT = '/home/seismic/LFE/data/sci/ducellier/pnsn_tremor_south.csv'
months = pd.date_range('2009-08-01', '2026-12-01', freq='MS')
def get(m):
    f = f'{RAW}/{m:%Y-%m}.json'
    if not os.path.exists(f) or os.path.getsize(f) < 20:
        n = m + pd.offsets.MonthBegin(1)
        url = f'https://tremorapi.pnsn.org/api/v3.0/events?starttime={m:%Y-%m-%d}T00:00:00&endtime={n:%Y-%m-%d}T00:00:00'
        for a in range(4):
            try:
                r = requests.get(url, timeout=300); r.raise_for_status()
                open(f, 'wb').write(r.content); break
            except Exception:
                time.sleep(15)
    rows = []
    try:
        for ft in json.load(open(f))['features']:
            lon, lat = ft['geometry']['coordinates'][:2]
            if lat < 42.2:
                p = ft['properties']
                rows.append((p['time'], lat, lon, p.get('depth'), p.get('num_stas'), p.get('energy')))
    except Exception as e:
        print('fail', m, e)
    return rows
with ThreadPoolExecutor(4) as ex:
    allr = [r for rr in ex.map(get, months[:-1]) for r in rr]
df = pd.DataFrame(allr, columns=['time', 'lat', 'lon', 'depth', 'num_stas', 'energy'])
df['time'] = pd.to_datetime(df.time, format='%a, %d %b %Y %H:%M:%S GMT')
df.sort_values('time').to_csv(OUT, index=False)
print(len(df), df.time.min(), df.time.max())
print(df.groupby(df.time.dt.year).size())
