#!/usr/bin/env python3
"""Compare fixed-station (core) vs all-station relative relocations of individual events.

Well constrained: sd_x, sd_y < 0.5 km and sd_z < 1 km (as for the all-station run).
Reports: counts, per-event correlation of dx/dy/dz between runs, median offsets of
well-constrained events before vs after each quake (core run, +-90 days and +-1 year windows,
bootstrap CI), and the yearly median position for both runs. Figure: figures/fam23_events_core.png
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

QUAKES = {'M6.1 Petrolia': '2021-12-20T20:10', 'M6.4 Ferndale': '2022-12-20T10:34',
          'M7.0': '2024-12-05T18:44'}


def load(f):
    d = pd.read_csv(f)
    d['t'] = pd.to_datetime(d.time_utc, format='mixed')
    d['good'] = (d.sd_x < 0.5) & (d.sd_y < 0.5) & (d.sd_z < 1.0)
    return d


def step(d, q, win, rng):
    b = d[(d.t >= q - win) & (d.t < q)]
    a = d[(d.t >= q) & (d.t < q + win)]
    if len(a) < 10 or len(b) < 10:
        return None
    res = {}
    for c in ('dx', 'dy', 'dz'):
        boot = [np.median(rng.choice(a[c].values, len(a))) - np.median(rng.choice(b[c].values, len(b)))
                for _ in range(1000)]
        res[c] = (1000 * (a[c].median() - b[c].median()), 1000 * np.std(boot))
    return len(b), len(a), res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', type=int, default=23)
    a = ap.parse_args()
    F = a.family
    al = load(f'data/relocations/fam{F}_events_3d.csv')
    co = load(f'data/relocations/fam{F}_events_3d_core.csv')
    print(f'all-station: {len(al)} relocated, {al.good.sum()} well constrained')
    print(f'core       : {len(co)} relocated, {co.good.sum()} well constrained')
    m = al.merge(co, on='ev', suffixes=('_all', '_core'))
    mg = m[m.good_all & m.good_core]
    print(f'common: {len(m)}, both well constrained: {len(mg)}')
    for c in ('dx', 'dy', 'dz'):
        r = np.corrcoef(mg[c + '_all'], mg[c + '_core'])[0, 1]
        diff = 1000 * (mg[c + '_core'] - mg[c + '_all'])
        print(f'  {c}: r={r:.2f}  median diff {diff.median():+.0f} m  MAD {np.median(np.abs(diff - diff.median())):.0f} m')
    rng = np.random.default_rng(0)
    g = co[co.good]
    print('\ncore well-constrained events: median shift after - before quake (m, bootstrap 1 sd)')
    for k, q in QUAKES.items():
        q = pd.Timestamp(q)
        for win in (pd.Timedelta(days=90), pd.Timedelta(days=365)):
            r = step(g, q, win, rng)
            if r is None:
                print(f'  {k:14s} +-{win.days}d: too few events')
                continue
            nb, na, res = r
            s = '  '.join(f'{c} {v:+5.0f}+-{e:3.0f}' for c, (v, e) in res.items())
            print(f'  {k:14s} +-{win.days:3d}d (n={nb}/{na}): {s}')
    print('\nyearly medians of well-constrained events (m): all-station | core')
    ya = al[al.good].groupby(al.t.dt.year)[['dx', 'dy', 'dz']].median() * 1000
    yc = g.groupby(g.t.dt.year)[['dx', 'dy', 'dz']].median() * 1000
    nc = g.groupby(g.t.dt.year).size().rename('n_core')
    print(ya.round(0).add_suffix('_all').join(yc.round(0).add_suffix('_core')).join(nc).to_string())

    fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for i, (c, lab) in enumerate([('dx', 'East (m)'), ('dy', 'North (m)'), ('dz', 'Depth (m, +down)')]):
        ax[i].plot(g.t, 1000 * g[c], '.', ms=1, color='0.6', alpha=0.4, rasterized=True)
        for d, col, lb in ((al[al.good], 'C0', 'all stations'), (g, 'C3', 'fixed 15 stations')):
            r = d.set_index('t')[c].sort_index().rolling(301, center=True, min_periods=100).median() * 1000
            ax[i].plot(r.index, r.values, color=col, lw=1.5, label=f'{lb} (running median, 301 events)')
        ax[i].set_ylabel(lab)
        ax[i].set_ylim(-400, 400)
        for k, q in QUAKES.items():
            ax[i].axvline(pd.Timestamp(q), color='r', ls='--', lw=0.8)
    ax[0].legend(fontsize=8, loc='upper left')
    ax[0].set_title(f'Family {F}: individual-event relative locations, fixed 15-station network '
                    f'({len(g)} well-constrained events; grey dots)')
    fig.tight_layout()
    fig.savefig(f'figures/fam{F}_events_core.png', dpi=130)


if __name__ == '__main__':
    main()
