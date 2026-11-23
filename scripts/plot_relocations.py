#!/usr/bin/env python3
"""Figures for relative relocations of one family.
(a-c) bin-stack centroid offsets (E, N, depth) through time with bootstrap errors,
(d) map and (e) along-strike/depth view of individually relocated events coloured by time.
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

QUAKES = {'M6.1 Petrolia': '2021-12-20', 'M6.4 Ferndale': '2022-12-20', 'M7.0 Mendocino': '2024-12-05'}
STRIKE = 293.0   # Shelly et al. 2026 LFE fault strike


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', type=int, default=23)
    ap.add_argument('--model', default='3d')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    F = a.family
    b = pd.read_csv(f'data/relocations/fam{F}_bins_{a.model}{a.tag}.csv')
    e = pd.read_csv(f'data/relocations/fam{F}_events_{a.model}{a.tag}.csv')
    b['t_mid'] = pd.to_datetime(b.t_mid, format='mixed')       # times with/without fractional s
    e['time_utc'] = pd.to_datetime(e.time_utc, format='mixed')
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 2.2])
    for k, (col, sd, lab) in enumerate([('dx', 'sd_x', 'E offset (m)'), ('dy', 'sd_y', 'N offset (m)'),
                                         ('dz', 'sd_z', 'depth offset (m, +down)')]):
        ax = fig.add_subplot(gs[k, :])
        ax.errorbar(b.t_mid, b[col] * 1000, yerr=b[sd] * 1000, fmt='o', ms=3, lw=0.6, color='k')
        for q, d in QUAKES.items():
            ax.axvline(pd.Timestamp(d), color='r', ls='--', lw=0.8)
        ax.set_ylabel(lab)
        ax.axhline(0, color='0.7', lw=0.5)
        if k == 0:
            ax.set_title(f'Family {F}: centroid of {int(b.n_ev.median())}-event stacks vs master '
                         f'({len(b)} bins, {a.model})')
    sel = e[(e[['sd_x', 'sd_y']].max(axis=1) < 0.5) & (e.sd_z < 1.0)]
    tnum = (sel.time_utc - sel.time_utc.min()).dt.days / 365.25 + sel.time_utc.min().year
    ax = fig.add_subplot(gs[3, 0])
    sc = ax.scatter(sel.dx * 1000, sel.dy * 1000, c=tnum, s=3, cmap='viridis', alpha=0.6)
    ax.set_aspect('equal')
    ax.set_xlabel('E (m)')
    ax.set_ylabel('N (m)')
    ax.set_title(f'{len(sel)} events with sd<0.5 km horiz, <1 km depth (of {len(e)} relocated)')
    plt.colorbar(sc, ax=ax, label='year')
    ax = fig.add_subplot(gs[3, 1])
    s = np.radians(STRIKE)
    along = sel.dx * np.sin(s) + sel.dy * np.cos(s)
    ax.scatter(along * 1000, sel.dz * 1000, c=tnum, s=3, cmap='viridis', alpha=0.6)
    ax.invert_yaxis()
    ax.set_xlabel(f'along strike {STRIKE:.0f}° (m)')
    ax.set_ylabel('depth offset (m)')
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(f'figures/fam{F}_relocations_{a.model}{a.tag}.png', dpi=130)
    print('wrote', f'figures/fam{F}_relocations_{a.model}{a.tag}.png')


if __name__ == '__main__':
    main()
