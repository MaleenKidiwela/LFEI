#!/usr/bin/env python3
"""Stage 8a (lfe env): inputs for the PyRaysum dipping-layer modelling (sci_rf_model_raysum.py, 'rf' env).
Per station with power results:
  dipping layered model from Bloch, Bostock & Audet (2023): grid depths of t (top of the slab low-velocity
  layer, LVL), c (LVL base) and m (slab Moho) at the station; interface dip / strike from the local gradient of
  each grid (+-3 km finite differences); layer Vs, Vp/Vs from the nearest control point (<10 km) or regional
  medians; density from Vp (Brocher 2005). Layers: overriding plate (flat surface) to z_t - 2 km; 2-km layer
  above the LVL (same velocities as the overriding plate, its top parallel to the slab); LVL; oceanic crust;
  mantle half-space (Vp 8.0, Vs 4.5). Raysum thicknesses are vertical below the station.
  observed: power-set events (time, back-azimuth, slowness, geometry bin), 1.5-yr power series (slab R/T, null
  envelopes), non-overlapping slab contrasts, signal fraction of the slab-window band power
  (1 - noise-window power / slab-window power, for the noise dilution of predicted dB changes).
-> data/sci/rf_timelapse/raysum/input_{STA}.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sci_rf_common import OUT, STATIONS, coords  # noqa: E402
import sci_rf_power as PW  # noqa: E402
from sci_rf_static import bloch_at  # noqa: E402

RDIR = os.path.join(OUT, 'raysum')


def rho(vp):
    return 1.6612 * vp - 0.4721 * vp ** 2 + 0.0671 * vp ** 3 - 0.0043 * vp ** 4 + 0.000106 * vp ** 5


def grid_dip(k, la, lo, d=3.0):
    import h5py
    from scipy.interpolate import RegularGridInterpolator
    f = h5py.File(f'/home/seismic/LFE/data/slab/bloch2023/grids_masked/{k}_depth_masked.nc', 'r')
    I = RegularGridInterpolator((f['lat'][:], f['lon'][:]), f['z'][:], bounds_error=False)
    dla = d / 111.2; dlo = d / (111.2 * np.cos(np.radians(la)))
    zn, zs = I((la + dla, lo)), I((la - dla, lo))
    ze, zw = I((la, lo + dlo)), I((la, lo - dlo))
    gn, ge = (zn - zs) / (2 * d), (ze - zw) / (2 * d)          # km depth per km (depth positive down)
    if not np.all(np.isfinite([gn, ge])):
        return np.nan, np.nan
    dip = np.degrees(np.arctan(np.hypot(gn, ge)))
    dipdir = np.degrees(np.arctan2(ge, gn)) % 360            # direction of increasing depth
    return float(dip), float((dipdir - 90) % 360)            # right-hand-rule strike


def main():
    os.makedirs(RDIR, exist_ok=True)
    C = coords()
    for code in STATIONS:
        if not os.path.exists(os.path.join(OUT, f'power_series_{code}.csv')):
            continue
        B = bloch_at(code, C)
        la, lo = C[code][:2]
        ok = all(np.isfinite(B.get(k, np.nan)) for k in ('z_t', 'z_c', 'z_m'))
        src = 'Bloch et al. 2023 grids'
        if not ok:                                              # outside the masked model: regional default
            B.update(z_t=17.0, z_c=19.0, z_m=28.0); src = 'regional default depths (outside model mask)'
        dips = {}
        for k in 'tcm':
            dd, ss = grid_dip(k, la, lo)
            if not np.isfinite(dd):
                dd, ss = 10.0, 290.0
            dips[k] = (dd, ss)
        elev = C[code][2] / 1000.0
        zt, zc, zm = B['z_t'] + elev, max(B['z_c'] + elev, B['z_t'] + elev + 0.3), max(B['z_m'] + elev, B['z_c'] + elev + 1.0)
        vp0, vp1, vp2 = B['vs0'] * B['vpvs0'], B['vs1'] * B['vpvs1'], B['vs2'] * B['vpvs2']
        L = [dict(name='overriding plate', thick_km=zt - 2.0, vp=vp0, vs=B['vs0'], strike=0.0, dip=0.0),
             dict(name='2-km layer above LVL', thick_km=2.0, vp=vp0, vs=B['vs0'], strike=dips['t'][1], dip=dips['t'][0]),
             dict(name='slab low-velocity layer', thick_km=zc - zt, vp=vp1, vs=B['vs1'], strike=dips['t'][1], dip=dips['t'][0]),
             dict(name='oceanic crust', thick_km=zm - zc, vp=vp2, vs=B['vs2'], strike=dips['c'][1], dip=dips['c'][0]),
             dict(name='mantle', thick_km=0.0, vp=8.0, vs=4.5, strike=dips['m'][1], dip=dips['m'][0])]
        for l in L:
            l['rho'] = rho(l['vp']) * 1000
        S = PW.PStation(code)
        tot = S.total(S.psd.mean(0))                       # (window, comp)
        sigfrac = [max(tot[0, c] - tot[2, c], 0) / tot[0, c] for c in range(2)]
        ser = pd.read_csv(os.path.join(OUT, f'power_series_{code}.csv'))
        con = pd.read_csv(os.path.join(OUT, f'power_{code}.csv'))
        con = con[con.window == 'slab']
        out = dict(sta=code, model_src=src, vel_src=B['vel_src'], layers=L, windows=S.win, win_src=S.win_src,
                   band=list(PW.BAND), signal_fraction=sigfrac,
                   events=dict(time=S.time.tolist(), baz=S.m.baz.tolist(), p=S.m.p.tolist(), geo=S.geo.tolist()),
                   series=dict(tmid=(pd.to_datetime(ser.tmid).values.astype('datetime64[s]').astype(np.int64) / 86400.0).tolist(),
                               **{c: ser[c].tolist() for c in ser.columns if c.startswith('slab_')}),
                   contrasts=con[['contrast', 'comp', 'est_dB', 'null_sd', 'nA', 'nB']].to_dict(orient='records'))
        json.dump(out, open(os.path.join(RDIR, f'input_{code}.json'), 'w'), default=float)
        print(code, src, 'dips', {k: tuple(round(x, 1) for x in v) for k, v in dips.items()}, 'n_ev', len(S.m),
              'signal fraction', np.round(sigfrac, 2), flush=True)


if __name__ == '__main__':
    main()
