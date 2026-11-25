"""Phase 1: probe actual NCEDC data availability (Z channels) for candidate permanent
stations, quarterly 2004-2026 (5-min snippets at 12:00 UTC on the 15th of Feb/May/Aug/Nov).
<=4 concurrent requests. Output: data/sci/ducellier/availability_probe.csv"""
import io, time, itertools, requests, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from obspy import read
STA = {'BK': 'WDC,YBH,JCC,GASB,HATC', 'PB': 'B039,B040,B935,B933,B045,B049',
       'NC': 'KHBB,KHMB,KRMB,KRP,KSXB,KCPB,KCT,KMPB,KMR,KCR,KCS,KKP,KOM,KPP,KBN,KIP,KTR,LBK,LGP,LSF,LBP,LPG,LRB,LVR,LSH,LSR,LAM,LBF,LTC,LWH,LDB,LRR,LGB,GBB,GRO'}
URL = 'https://service.ncedc.org/fdsnws/dataselect/1/query'
dates = [pd.Timestamp(y, m, 15, 12) for y in range(2004, 2027) for m in (2, 5, 8, 11) if pd.Timestamp(y, m, 15) < pd.Timestamp('2026-11-20')]
def probe(t):
    t0 = t.strftime('%Y-%m-%dT%H:%M:%S'); t1 = (t + pd.Timedelta(minutes=5)).strftime('%Y-%m-%dT%H:%M:%S')
    lines = []
    for net, s in STA.items():
        for sta in s.split(','):
            lines.append(f'{net} {sta} * ?HZ {t0} {t1}')
    for att in range(3):
        try:
            r = requests.post(URL, data='\n'.join(lines), timeout=300)
            if r.status_code == 204:
                return t, []
            st = read(io.BytesIO(r.content))
            return t, [(tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel, tr.stats.npts / tr.stats.sampling_rate) for tr in st]
        except Exception as e:
            time.sleep(10)
    return t, None
rows = []
with ThreadPoolExecutor(4) as ex:
    for t, res in ex.map(probe, dates):
        if res is None:
            rows.append(dict(date=t, net='ERR', sta='ERR', loc='', cha='', dur=0)); continue
        for n, s, l, c, d in res:
            rows.append(dict(date=t, net=n, sta=s, loc=l, cha=c, dur=d))
df = pd.DataFrame(rows)
df.to_csv('/home/seismic/LFE/data/sci/ducellier/availability_probe.csv', index=False)
g = df[df.net != 'ERR'].groupby(['net', 'sta', 'date']).dur.sum().unstack('date').fillna(0)
yrs = sorted(set(d.year for d in dates))
summ = pd.DataFrame({y: (g[[d for d in g.columns if d.year == y]] > 200).sum(axis=1) for y in yrs})
pd.set_option('display.width', 300)
print(summ.to_string())
print('errors:', (df.net == 'ERR').sum())
