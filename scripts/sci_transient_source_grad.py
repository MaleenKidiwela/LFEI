#!/usr/bin/env python3
"""Source-side S travel-time gradients (s/km) per station at a family location, rotated into the fault frame
(s, d, n), from the same 3D (Furlong) travel-time grids and s_scale as relocate_relative.py."""
import json
import os
import sys

import numpy as np
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sci_transient_source_common import ROT  # noqa: E402


def s_gradients(F, stations, model='3d'):
    loc = json.load(open(f'data/locations/fam{F}_3d.json'))
    g = json.load(open(f'data/velocity/tt_{model}/grid.json'))
    gx, gy, gz = map(np.array, (g['sub_x'], g['sub_y'], g['sub_z']))
    tr = Transformer.from_crs('EPSG:4326', g['proj'], always_xy=True)
    x0, y0 = tr.transform(loc['lon'], loc['lat'])
    i0, j0, k0 = np.argmin(abs(gx - x0)), np.argmin(abs(gy - y0)), np.argmin(abs(gz - loc['depth_km']))
    d = g['d']
    out = {}
    for sta in stations:
        f = f'data/velocity/tt_{model}/{sta}.npz'
        if not os.path.exists(f):
            continue
        V = np.load(f)['ts'] * loc.get('s_scale', 1.0)
        grad = np.array([(V[i0 + 1, j0, k0] - V[i0 - 1, j0, k0]) / (2 * d),
                         (V[i0, j0 + 1, k0] - V[i0, j0 - 1, k0]) / (2 * d),
                         (V[i0, j0, k0 + 1] - V[i0, j0, k0 - 1]) / (2 * d)])      # (E, N, down) s/km
        out[sta] = ROT @ grad                                                     # (s, d, n) s/km
    return out
