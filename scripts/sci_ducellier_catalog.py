"""Phase 1: convert Ducellier & Creager (2022) catalogs (untrusted pickles, loaded with a
restricted unpickler) to CSV and summarise families. Run with python -I."""
import sys, os, glob
sys.path.insert(0, '/home/seismic/LFE/scripts')
import numpy as np, pandas as pd
from sci_ducellier_safe_pickle import load
REPO = '/data/wsd02/maleen_data/ducellier/gh_catalog/catalog-master'
OUT = '/data/wsd02/maleen_data/ducellier/catalog_csv'
SMALL = '/home/seismic/LFE/data/sci/ducellier'
os.makedirs(OUT, exist_ok=True)
loc = pd.read_csv(f'{REPO}/data/Plourde_2015/templates_list.txt', sep=r'\s+', skiprows=1, header=None,
                  names=['template', 'fam', 'lat', 'lon', 'depth', 'eH', 'eZ', 'ntempl'], dtype={'template': str, 'fam': str})
fp = pd.read_csv(f'{REPO}/data/Ducellier/families_permanent_timeinterval.txt', sep=r'\s+', header=None,
                 names=['template', 'stations', 't0', 't1'], dtype={'template': str})
thr = pd.read_csv(f'{REPO}/data/Ducellier/threshold_cc.txt', sep=r'\s+', header=None, names=['template', 'thr_a', 'thr_b'], dtype={'template': str})
rows = []; allcat = []
for d in sorted(glob.glob(f'{REPO}/data/Ducellier/catalogs/*')):
    t = os.path.basename(d)
    for name in ['catalog_2004_2011', 'catalog_2007_2009']:
        p = f'{d}/{name}.pkl'
        if not os.path.exists(p):
            continue
        df = load(p)
        df = pd.DataFrame(df)
        sec = df['second'].astype(float)
        tt = pd.to_datetime(dict(year=df.year, month=df.month, day=df.day, hour=df.hour, minute=df.minute)) + pd.to_timedelta(sec, unit='s')
        df.insert(0, 'time', tt)
        df.insert(0, 'template', t)
        df.to_csv(f'{OUT}/{t}_{name}.csv', index=False)
        if name == 'catalog_2004_2011':
            allcat.append(df[['template', 'time', 'cc', 'nchannel']])
        rows.append(dict(template=t, cat=name, n=len(df), tmin=tt.min(), tmax=tt.max(),
                         cc_med=df.cc.median(), nch_med=df.nchannel.median()))
summ = pd.DataFrame(rows)
summ.to_csv(f'{SMALL}/catalog_file_summary.csv', index=False)
cat = pd.concat(allcat, ignore_index=True)
cat.to_parquet(f'{OUT}/ducellier_2004_2011_all_raw.parquet')
s = summ[summ.cat == 'catalog_2004_2011'].merge(loc, on='template', how='left').merge(fp, on='template', how='left').merge(thr, on='template', how='left')
s.to_csv(f'{SMALL}/families_summary.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200); pd.set_option('display.max_colwidth', 60)
print(s[['template', 'fam', 'lat', 'lon', 'depth', 'n', 'tmin', 'tmax', 'cc_med', 'nch_med', 'thr_a', 'thr_b', 'stations']].to_string())
print(summ.groupby('cat').n.sum())
