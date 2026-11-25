#!/usr/bin/env python3
"""RUN WITH THE 'rf' CONDA ENV (~/anaconda3/envs/rf/bin/python; PyRaysum 1.0.0).
LVZ top/bottom receiver-function study, synthetic part (sci_rf_lvz*.py).
Per station: the dipping layered model of sci_rf_raysum_prep.py (Bloch, Bostock & Audet 2023 depths/dips:
overriding plate, 2-km layer above the slab low-velocity layer (LVL), LVL, oceanic crust, mantle), PyRaysum
synthetic seismograms for the back-azimuth / slowness of every QC-accepted event (sci_rf_common.load_rf), deconvolved
(R/Z, water level 1e-4) and Gaussian filtered exactly like the data RFs (exp(-(2 pi f)^2 / 4a^2), unit peak;
a = 1, 2.5), cut to the data time axis (-5..30 s, 0.05 s), raw and Ps-moveout corrected to p = 0.06 s/km
(sci_rf_common.moveout, as the data).
Models: base; base without multiples (mults=0, phase identification); LVL Vs changed by -0.10, -0.05, +0.05,
+0.10 km/s (changes the S-velocity contrast across the top of the LVL by the opposite amount; Gosselin et al. 2020
scale) and by -4, -2, +2, +4 % (Herath & Audet scale); Vp fixed (Vp/Vs changes).
-> data/sci/rf_lvz/synth_{STA}.npz (R[model, gauss, event, nt] moveout-corrected, Rraw same not corrected,
   models, gauss, t, evid)
Usage: sci_rf_lvz_synth.py STA1,STA2,...
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sci_rf_common import OUT as RFOUT, load_rf, moveout  # noqa: E402

OUT = '/home/seismic/LFE/data/sci/rf_lvz'
RDIR = os.path.join(RFOUT, 'raysum')
GAUSS = (1.0, 2.5)
MODELS = [('base', None, 0.0), ('nomult', None, 0.0),
          ('lvl_dvs-0.10', 'abs', -0.10), ('lvl_dvs-0.05', 'abs', -0.05),
          ('lvl_dvs+0.05', 'abs', 0.05), ('lvl_dvs+0.10', 'abs', 0.10),
          ('lvl_pct-4', 'pct', -0.04), ('lvl_pct-2', 'pct', -0.02),
          ('lvl_pct+2', 'pct', 0.02), ('lvl_pct+4', 'pct', 0.04)]
LVL = 2          # index of the low-velocity layer in the input layer list


def build_model(layers, kind=None, d=0.0):
    from pyraysum import Model
    L = [dict(l) for l in layers]
    if kind == 'abs':
        L[LVL]['vs'] += d
    elif kind == 'pct':
        L[LVL]['vs'] *= (1 + d)
    return Model([l['thick_km'] * 1000 for l in L], [l['rho'] for l in L], [l['vp'] * 1000 for l in L],
                 [l['vs'] * 1000 for l in L], strike=[l['strike'] for l in L], dip=[l['dip'] for l in L])


def synth(model, baz, p, t_out, mults=2, dtf=0.0125, tlen=40.0):
    from pyraysum import prs, Geometry, Control
    n = int(round(tlen / dtf))
    rc = Control(npts=n, dt=dtf, mults=mults, rot=1, verbose=0, shift=1.0)   # shift=None drops the direct P (PyRaysum 1.0.0)
    nf = 2 * n
    f = np.fft.rfftfreq(nf, dtf)
    lags = np.round(t_out / dtf).astype(int) % nf
    out = np.zeros((len(GAUSS), len(baz), len(t_out)))
    for i0 in range(0, len(baz), 300):
        res = prs.run(model, Geometry(list(baz[i0:i0 + 300]), list(p[i0:i0 + 300])), rc, rf=False, mode='bare')
        for j, st in enumerate(res.streams):
            Z = np.fft.rfft(st[2].data, nf)
            den = np.abs(Z) ** 2
            den = den + 1e-4 * den.max()
            rz = np.fft.irfft(Z * np.conj(Z) / den, nf)
            Rspec = np.fft.rfft(st[0].data, nf) * np.conj(Z) / den
            # P arrival = peak of the (unfiltered) Z/Z autocorrelation-like response -> lag 0 by construction;
            # locate the direct P on R/Z via the Z autodeconvolution peak (index 0)
            k0 = int(np.argmax(np.abs(rz)))
            for ig, a in enumerate(GAUSS):
                G = np.exp(-(2 * np.pi * f) ** 2 / (4 * a * a))
                g0 = np.fft.irfft(G, nf)[0]
                y = np.fft.irfft(Rspec * G, nf) / g0
                out[ig, i0 + j] = np.roll(y, -k0)[lags]   # R/Z: direct P positive (checked), no sign flip needed
    return out


def main():
    sts = sys.argv[1].split(',')
    for code in sts:
        fn = os.path.join(RDIR, f'input_{code}.json')
        if not os.path.exists(fn):
            print(code, 'no raysum input'); continue
        I = json.load(open(fn))
        m, rf, _, t, g, _ = load_rf(code)
        baz, p = m.baz.values.astype(float), m.p.values.astype(float)
        Rraw = np.zeros((len(MODELS), len(GAUSS), len(m), len(t)), np.float32)
        Rmo = np.zeros_like(Rraw)
        for im, (name, kind, d) in enumerate(MODELS):
            x = synth(build_model(I['layers'], kind, d), baz, p, t, mults=0 if name == 'nomult' else 2)
            Rraw[im] = x
            for ig in range(len(GAUSS)):
                for k in range(len(m)):
                    Rmo[im, ig, k] = moveout(x[ig, k], t, p[k])
            print(code, name, flush=True)
        np.savez_compressed(os.path.join(OUT, f'synth_{code}.npz'), R=Rmo, Rraw=Rraw, t=t, gauss=np.array(GAUSS),
                            models=np.array([x[0] for x in MODELS]), evid=m.evid.values.astype(str),
                            layers=json.dumps(I['layers']))


if __name__ == '__main__':
    main()
