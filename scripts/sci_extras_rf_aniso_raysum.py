#!/usr/bin/env python3
"""RUN WITH THE 'rf' CONDA ENV (PyRaysum 1.0.0).  Converts RF harmonic amplitudes to anisotropy (test 3 extras).
For stations with a raysum input model (data/sci/rf_timelapse/raysum/input_{STA}.json; Bloch 2023 dipping layers):
synthetic RFs (PyRaysum, shift=1.0, first-order multiples; deconvolution / low-pass as sci_rf_model_raysum.synth_rfs,
then Gaussian a=2.5 as the data) at 36 back-azimuths, p = 0.06 s/km, for
  (i) the isotropic dipping model, (ii) LVL anisotropy 2/5/10 % with a horizontal fast axis (trend 293 = LFE-fault
  strike, and 0), (iii) LVL anisotropy 5 % with the axis plunging 40 deg (dipping-foliation proxy),
  (iv) the 2-km layer above the LVL with 5 % horizontal-axis anisotropy.
Harmonic decomposition (same code / convention as sci_extras_rf_aniso.py) in the station's slab window ->
RMS k=1 and k=2 amplitudes; sensitivity d(RMS_k)/d(ani %) -> converts the observed detection limits.
-> data/sci/extras/rf_aniso/raysum_sensitivity.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sci_rf_model_raysum import synth_rfs, DT, NPTS  # noqa: E402

ROOT = '/home/seismic/LFE'
OUT = f'{ROOT}/data/sci/extras/rf_aniso'
STAS = ['PB.B049', 'NC.KMPB', 'PB.B047', 'PB.B932', 'PB.B935', 'NC.KMR']


def gauss(x, a=2.5):
    nf = 2 * x.shape[-1]
    f = np.fft.rfftfreq(nf, DT)
    g = np.exp(-(2 * np.pi * f) ** 2 / (4 * a * a))
    return np.fft.irfft(np.fft.rfft(x, nf) * g, nf)[..., :x.shape[-1]]


def model(layers, ani_layer=None, ani=0.0, trend=0.0, plunge=0.0):
    from pyraysum import Model
    L = [dict(l) for l in layers]
    flag = [1] * len(L); A = [0.0] * len(L); TR = [0.0] * len(L); PL = [0.0] * len(L)
    if ani_layer is not None and ani != 0:
        flag[ani_layer] = 0; A[ani_layer] = ani; TR[ani_layer] = trend; PL[ani_layer] = plunge
    return Model([l['thick_km'] * 1000 for l in L], [l['rho'] for l in L], [l['vp'] * 1000 for l in L],
                 [l['vs'] * 1000 for l in L], flag=flag, ani=A, trend=TR, plunge=PL,
                 strike=[l['strike'] for l in L], dip=[l['dip'] for l in L])


def harm(R, T, baz):
    ph = np.radians(baz)
    Gr = np.c_[np.ones_like(ph), np.cos(ph), np.sin(ph), np.cos(2 * ph), np.sin(2 * ph)]
    best = None
    for s in (1.0, -1.0):
        Gt = s * np.c_[np.zeros_like(ph), -np.sin(ph), np.cos(ph), -np.sin(2 * ph), np.cos(2 * ph)]
        G = np.r_[Gr, Gt]; D = np.r_[R, T]
        c, *_ = np.linalg.lstsq(G, D, rcond=None)
        mis = np.sum((D - G @ c) ** 2)
        if best is None or mis < best[1]:
            best = (c, mis, s)
    return best


def main():
    W = {}
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    L = pd.read_csv(f'{ROOT}/data/sci/rf_lvz/stations_lvz.csv').set_index('sta')
    rows = []
    baz = np.arange(0, 360, 10.0)
    p = np.full(len(baz), 0.06)   # s/km (PyRaysum Geometry units, as sci_rf_model_raysum)
    t = np.arange(NPTS) * DT - NPTS / 2 * DT
    for sta in STAS:
        fn = f'{ROOT}/data/sci/rf_timelapse/raysum/input_{sta}.json'
        if not os.path.exists(fn):
            continue
        layers = json.load(open(fn))['layers']
        st, sm = L.loc[sta, 'syn_t'], L.loc[sta, 'syn_m']
        t0 = max(st - 0.6, 1.0) if np.isfinite(st) else 1.5
        t1 = min((sm + 0.6) if np.isfinite(sm) else st + 2.0, 5.5) if np.isfinite(st) else 5.0
        w = (t >= t0) & (t <= t1)
        cases = [('iso', None, 0, 0, 0)]
        for a in (2, 5, 10):
            cases += [(f'lvl_h293_{a}', 2, a, 293, 0), (f'lvl_h0_{a}', 2, a, 0, 0)]
        cases += [('lvl_p40_5', 2, 5, 293, 40), ('above_h293_5', 1, 5, 293, 0)]
        ref = None
        for name, li, a, tr, pl in cases:
            R, T = synth_rfs(model(layers, li, -a if False else a, tr, pl), baz, p)
            R, T = gauss(R), gauss(T)
            c, mis, s = harm(R, T, baz)
            z1 = c[1] + 1j * c[2]; z2 = c[3] + 1j * c[4]
            d = dict(sta=sta, case=name, layer=li, ani_pct=a, trend=tr, plunge=pl, t0=t0, t1=t1, tsign=s,
                     k0_rms=np.sqrt(np.mean(c[0][w] ** 2)), k1_rms=np.sqrt(np.mean(np.abs(z1[w]) ** 2)),
                     k2_rms=np.sqrt(np.mean(np.abs(z2[w]) ** 2)), P0=np.abs(c[0][np.abs(t) < 0.3]).max())
            if ref is None:
                ref = (z1, z2)
            d['k1_diff_rms'] = np.sqrt(np.mean(np.abs(z1[w] - ref[0][w]) ** 2))
            d['k2_diff_rms'] = np.sqrt(np.mean(np.abs(z2[w] - ref[1][w]) ** 2))
            rows.append(d)
            print(sta, name, {k: round(v, 4) for k, v in d.items() if isinstance(v, float)}, flush=True)
        pd.DataFrame(rows).to_csv(f'{OUT}/raysum_sensitivity.csv', index=False)


if __name__ == '__main__':
    main()
