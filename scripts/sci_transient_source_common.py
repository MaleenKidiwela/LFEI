#!/usr/bin/env python3
"""Shared event/location tables for the early-coda transient vs source-location test
(sci_transient_source.py, sci_transient_source_measure.py).

Per family: all h5 events, the dvv_cwire isolation flag (+-5 s, merged catalog without v1_only),
an overlap flag (other-family catalog LFE within +-10 s; such events are located ~50 m toward
the other family, NOTES 2026-11-22), full-network relocations and the A/B split-station halves,
rotated into Shelly's fault frame (strike 290, dip 45 NNE):
  s = along strike (+ toward az 290, WNW), d = down-dip (+ toward NNE and deeper), n = plane normal.
"""
import os
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dvv_cwire import QUAKES, epoch_s, make_stacks  # noqa: E402,F401

OUT = 'data/sci/transient_source'
FAMS = [23, 8, 27]
M7 = epoch_s([QUAKES['m7']])[0]
DAY = 86400.0
SDH = 0.5            # location quality cut: sd_x, sd_y < SDH km, sd_z < 2 SDH (as sci_patch_dynamics)

STRIKE, DIP = np.radians(290.0), np.radians(45.0)
DIPDIR = STRIKE + np.pi / 2
U_S = np.array([np.sin(STRIKE), np.cos(STRIKE), 0.0])
U_D = np.array([np.cos(DIP) * np.sin(DIPDIR), np.cos(DIP) * np.cos(DIPDIR), np.sin(DIP)])
U_N = np.cross(U_S, U_D)
ROT = np.vstack([U_S, U_D, U_N])          # (E, N, down) -> (s, d, n)
AX = ['s', 'd', 'n']

_CAT = None


def catalog():
    global _CAT
    if _CAT is None:
        c = pd.read_csv('data/merged/LFEcat_MTJ_merged_v1v2.csv', usecols=['time_utc', 'idnum', 'source'])
        c = c[c.source != 'v1_only']
        t = epoch_s(pd.to_datetime(c.time_utc, format='mixed'))
        o = np.argsort(t)
        _CAT = (t[o], c.idnum.values[o])
    return _CAT


def frame(df, sfx=''):
    x = df[[f'dx{sfx}', f'dy{sfx}', f'dz{sfx}']].values * 1000.0 @ ROT.T     # metres
    return x


def family_table(F, sdh=SDH):
    """One row per h5 event: t, iso, ovl10, located (full, quality cut), s/d/n (full), sA.., sB.."""
    with h5py.File(f'data/waveforms/fam{F}.h5', 'r') as h:
        te = epoch_s(pd.to_datetime(h['events/time_utc'][:].astype(str)))
    allt, allf = catalog()
    iso = (np.searchsorted(allt, te + 5.0, side='right') - np.searchsorted(allt, te - 5.0)) == 1
    lo, hi = np.searchsorted(allt, te - 10.0), np.searchsorted(allt, te + 10.0, side='right')
    ovl = np.array([np.any(allf[a:b] != F) for a, b in zip(lo, hi)])
    T = pd.DataFrame(dict(ev=np.arange(len(te)), t=te, iso=iso, ovl10=ovl))
    T['day'] = (T.t - M7) / DAY

    def add(path, sfx):
        d = pd.read_csv(path)
        good = (d.sd_x < sdh) & (d.sd_y < sdh) & (d.sd_z < 2 * sdh)
        d = d[good]
        x = frame(d)
        for k, c in enumerate(AX):
            T[c + sfx] = np.nan
            T.loc[d.ev.values, c + sfx] = x[:, k]
    add(f'data/relocations/fam{F}_events_3d.csv', '')
    add(f'data/relocations/fam{F}_events_3d_halfA.csv', 'A')
    add(f'data/relocations/fam{F}_events_3d_halfB.csv', 'B')
    T['loc'] = np.isfinite(T.s)
    T['locAB'] = np.isfinite(T.sA) & np.isfinite(T.sB)
    # restart: time of the 10th event (any, in the h5 catalog) after the M7
    post = np.sort(te[te >= M7])
    restart = post[9] if len(post) >= 10 else np.nan
    T.attrs['restart'] = restart
    T['dr'] = (T.t - restart) / DAY          # days after restart
    return T


def periods(T):
    """Named boolean masks over events (time periods used throughout)."""
    d, dr = T.day.values, T.dr.values
    return {'pre1y': (d >= -365) & (d < 0),
            'post_early60': (dr >= 0) & (dr < 60),
            'post_60_182': (dr >= 60) & (d < 182),
            'post_late': d >= 182}
