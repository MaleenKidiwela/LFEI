#!/usr/bin/env python3
"""Long-term dv/v: yearly means of the pooled per-station CWIRE curves (a full-year mean removes the
seasonal cycle), each channel demeaned over 2009-2025; median and IQR over borehole (PB) and surface channels.
Output data/dvv/pool/longterm_yearly.csv, figures/dvv_longterm.png"""
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rows = []
for f in glob.glob('data/dvv/pool/*_curve.csv'):
    ch = f.split('/')[-1].replace('_curve.csv', '')
    c = pd.read_csv(f)
    c['y'] = pd.to_datetime(c.t).dt.year
    ym = c[c.y.between(2009, 2025)].groupby('y').dvv_pct.mean()
    if len(ym) < 8:
        continue
    for y, v in (ym - ym.mean()).items():
        rows.append(dict(chan=ch, type='borehole PB' if ch.startswith('PB') else 'surface BK/NC', y=y, v=v))
d = pd.DataFrame(rows)
s = d.groupby(['type', 'y']).v.describe()[['count', '25%', '50%', '75%']].reset_index()
s.to_csv('data/dvv/pool/longterm_yearly.csv', index=False)
fig, ax = plt.subplots(figsize=(10, 4.5))
for t, c in (('borehole PB', 'C0'), ('surface BK/NC', 'C3')):
    g = s[s.type == t]
    ax.plot(g.y, g['50%'], 'o-', color=c, label=f'{t} (median of {int(g["count"].max())} channels)')
    ax.fill_between(g.y, g['25%'], g['75%'], color=c, alpha=0.2)
for q, lab in ((2021.97, 'Petrolia'), (2022.97, 'Ferndale'), (2024.93, 'M7')):
    ax.axvline(q, color='k', ls='--', lw=0.7)
    ax.text(q, ax.get_ylim()[1] * 0.9, ' ' + lab, fontsize=8)
ax.axhline(0, color='0.5', lw=0.5)
ax.set_ylabel('yearly mean dv/v (%), channel-demeaned')
ax.set_title('Long-term dv/v from LFE coda (all families pooled per station): flat 2009-2021, decline since')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig('figures/dvv_longterm.png', dpi=130)
print(s.round(3).to_string(index=False))
