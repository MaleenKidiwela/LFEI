#!/usr/bin/env python3
"""All relocated families together: each family's well-constrained relative relocations
(sd < 0.5 km horiz, < 1 km depth) placed at its absolute 3D location. Map + E-W and N-S sections.
Output figures/all_families_clouds.png"""
import glob, json, re
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fams = sorted(int(re.search(r'fam(\d+)_events_3d.csv', f).group(1)) for f in glob.glob('data/relocations/fam*_events_3d.csv'))
sh = pd.read_csv('data/families/shelly_family_locations.csv').set_index('idnum')
lat0, lon0 = 40.06, -123.625
kx, ky = 111.19 * np.cos(np.radians(lat0)), 111.19
fig = plt.figure(figsize=(15, 11))
axm = fig.add_axes([0.06, 0.38, 0.55, 0.58]); axe = fig.add_axes([0.06, 0.06, 0.55, 0.26]); axn = fig.add_axes([0.66, 0.38, 0.3, 0.58])
cmap = plt.get_cmap('tab20')
for i, F in enumerate(fams):
    try:
        loc = json.load(open(f'data/locations/fam{F}_3d.json'))
        e = pd.read_csv(f'data/relocations/fam{F}_events_3d.csv')
    except Exception:
        continue
    e = e[(e.sd_x < 0.5) & (e.sd_y < 0.5) & (e.sd_z < 1.0)]
    if len(e) > 3000:
        e = e.sample(3000, random_state=0)
    X = (loc['lon'] - lon0) * kx + e.dx
    Y = (loc['lat'] - lat0) * ky + e.dy
    Z = loc['depth_km'] + e.dz
    c = cmap(i % 20)
    kw = dict(s=1, color=c, alpha=0.35, rasterized=True)
    axm.scatter(X, Y, **kw); axe.scatter(X, Z, **kw); axn.scatter(Z, Y, **kw)
    cx, cy = (loc['lon'] - lon0) * kx, (loc['lat'] - lat0) * ky
    axm.annotate(str(F), (cx, cy), fontsize=9, weight='bold', ha='center', va='center')
    axe.annotate(str(F), (cx, loc['depth_km']), fontsize=8, weight='bold', ha='center')
    s = sh.loc[F]
    axm.plot((s.lon - lon0) * kx, (s.lat - lat0) * ky, 'k+', ms=6)
axm.set_xlabel('E (km from -123.625)'); axm.set_ylabel('N (km from 40.06)'); axm.set_aspect('equal')
axm.set_title(f'{len(fams)} families: relative relocations placed at our 3D family locations (+ = Shelly)')
axe.set_xlabel('E (km)'); axe.set_ylabel('depth (km)'); axe.invert_yaxis(); axe.set_xlim(axm.get_xlim())
axn.set_xlabel('depth (km)'); axn.set_ylabel('N (km)'); axn.set_ylim(axm.get_ylim())
axe.text(0.01, 0.05, 'absolute depths ~2.5 km too deep (earthquake test); relative scatter is meaningful',
         transform=axe.transAxes, fontsize=8)
fig.savefig('figures/all_families_clouds.png', dpi=140)
print('families:', fams)
