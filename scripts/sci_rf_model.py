#!/usr/bin/env python3
"""Stage 8: sensitivity of the slab-window RF power (1.25-3 Hz) to velocity changes of the slab low-velocity
layer (LVL) and of a 2-km layer above it -- a SIMPLIFIED stand-in for the PyRaysum modelling of Herath & Audet
(2024): kept as a cross-check of the PyRaysum modelling (sci_rf_model_raysum.py).
Model: flat layers from the Bloch et al. (2023) model at the station (grid depths of t, c, m; layer Vs and Vp/Vs
from the nearest control point or regional medians; mantle Vs 4.5 km/s, Vp/Vs 1.8; density from Vp after Brocher
2005), with a 2-km layer above the slab top (initially identical to the overriding plate).
Synthetic radial RF = primary Ps conversions only (no multiples, no dip/anisotropy, no free-surface factor):
amplitude = linearised P-to-S transmission coefficient (Aki & Richards 1980, weak contrast) at p = 0.06 s/km,
times from Ps delays; spikes Fourier-shifted, low-passed exactly as the modified Wiener RFs; PSD and band power
computed with the same code as sci_rf_power.py.
Perturbations: Vs of the LVL / of the 2-km layer by -10..+10 % at fixed Vp (i.e. Vp/Vs change).
Observed sensitivity includes dilution by noise: dB_obs = 10 log10(1 + delta * S / (S + N)) with S, N the station's
mean slab-window and pre-P-noise-window band powers.  Minimum detectable Vs change = 2-sigma detection limit (dB,
long-term contrast) / sensitivity.
-> data/sci/rf_timelapse/model_sensitivity_flat.csv, figures/sci_rf_power_model_flat.png (superseded by the PyRaysum
   dipping-layer modelling in sci_rf_model_raysum.py, which runs in the separate 'rf' conda env)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sci_rf_common import OUT, FIG, STATIONS, coords, P_REF  # noqa: E402
import sci_rf_power as PW  # noqa: E402
from sci_rf_static import bloch_at  # noqa: E402

DT = 0.05
T = np.arange(-100, 601) * DT


def rho(vp):
    return 1.6612 * vp - 0.4721 * vp ** 2 + 0.0671 * vp ** 3 - 0.0043 * vp ** 4 + 0.000106 * vp ** 5


def tps_lin(a1, b1, r1, a2, b2, r2, p=P_REF):
    """weak-contrast P->S transmission (upper medium 1, lower 2), sign: velocity increase downward -> positive."""
    a, b, r = (a1 + a2) / 2, (b1 + b2) / 2, (r1 + r2) / 2
    ci = np.sqrt(1 - (p * a) ** 2); cj = np.sqrt(1 - (p * b) ** 2)
    k = ci / a * cj / b
    return (p * a / (2 * cj)) * ((1 - 2 * b * b * p * p + 2 * b * b * k) * (r2 - r1) / r -
                                 (4 * b * b * p * p - 4 * b * b * k) * (b2 - b1) / b)


def layers(B, dvs_lvl=0.0, dvs_above=0.0, h_above=2.0):
    """-> list of (thickness, vp, vs) from the surface down to the mantle half-space."""
    zt, zc, zm = B['z_t'], B['z_c'], B['z_m']
    vs0, vs1, vs2 = B['vs0'], B['vs1'], B['vs2']
    vp0, vp1, vp2 = vs0 * B['vpvs0'], vs1 * B['vpvs1'], vs2 * B['vpvs2']
    L = [(zt - h_above, vp0, vs0), (h_above, vp0, vs0 * (1 + dvs_above)),
         (max(zc - zt, 0.3), vp1, vs1 * (1 + dvs_lvl)), (max(zm - zc, 0.5), vp2, vs2), (np.inf, 8.1, 4.5)]
    return L


def synth(L, p=P_REF):
    rf = np.zeros(len(T))
    nf = 2 * len(T)
    f = np.fft.rfftfreq(nf, DT)
    lp = np.clip((6.0 - f) / 2.0, 0, 1); lp = 0.5 - 0.5 * np.cos(np.pi * lp)
    t_acc = 0.0
    spikes = [(0.0, 0.3)]
    for i in range(len(L) - 1):
        h, vp, vs = L[i]
        t_acc += h * (np.sqrt(1 / vs ** 2 - p ** 2) - np.sqrt(1 / vp ** 2 - p ** 2))
        _, vp2, vs2 = L[i + 1]
        A = tps_lin(vp, vs, rho(vp), vp2, vs2, rho(vp2))
        spikes.append((t_acc, A))
    X = np.zeros(len(f), complex)
    for tt, A in spikes:
        X += A * np.exp(-2j * np.pi * f * (tt - T[0])) / DT
    return np.fft.irfft(X * lp, nf)[:len(T)] * DT, spikes


def band_power(x, win):
    w = (T >= win[0]) & (T <= win[1])
    f, p = PW.mt_psd(x[w], DT)
    b = PW.band_avg(f, p)
    return b[(PW.FB >= PW.BAND[0]) & (PW.FB <= PW.BAND[1])].sum()


def main():
    C = coords()
    rows = []
    curves = {}
    for code in STATIONS:
        fn = os.path.join(OUT, f'power_{code}.csv')
        if not os.path.exists(fn):
            continue
        B = bloch_at(code, C)
        if not np.isfinite(B.get('z_m', np.nan)) or not np.isfinite(B.get('z_c', np.nan)):
            continue
        win, src = PW.windows_for(code)
        x0, sp = synth(layers(B))
        P0 = band_power(x0, win['slab'])
        ser = pd.read_csv(os.path.join(OUT, f'power_series_{code}.csv'))
        S = PW.PStation(code)
        Sp = S.total(S.psd.mean(0))          # (nwin, 2)
        snr_frac = max(Sp[0, 0] - Sp[2, 0], 0) / Sp[0, 0]
        T_ = pd.read_csv(fn)
        det = T_[(T_.contrast == 'longterm_post2021') & (T_.window == 'slab') & (T_.comp == 'R')].det2sigma
        det = float(det.iloc[0]) if len(det) else np.nan
        grid = np.linspace(-0.10, 0.10, 21)
        for kind in ('lvl', 'above'):
            db = []
            for g in grid:
                x, _ = synth(layers(B, **({'dvs_lvl': g} if kind == 'lvl' else {'dvs_above': g})))
                d = band_power(x, win['slab']) / P0 - 1
                db.append(10 * np.log10(max(1 + d * snr_frac, 1e-6)))
            db = np.array(db)
            curves[(code, kind)] = db
            slope = np.polyfit(grid[8:13] * 100, db[8:13], 1)[0]      # dB per % Vs near 0
            rows.append(dict(sta=code, layer=kind, slope_dB_per_pct=slope, dB_at_minus5pct=db[5], dB_at_plus5pct=db[15],
                             signal_fraction=snr_frac, det2sigma_longterm_dB=det,
                             min_detectable_dVs_pct=det / abs(slope) if abs(slope) > 1e-6 else np.inf,
                             z_t=B['z_t'], z_c=B['z_c'], z_m=B['z_m'], vs1=B['vs1'], vpvs1=B['vpvs1'], vel_src=B['vel_src']))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, 'model_sensitivity_flat.csv'), index=False)
    print(R.round(3).to_string())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    grid = np.linspace(-10, 10, 21)
    for ax, kind, title in [(axs[0], 'lvl', 'Vs of the slab low-velocity layer (Vp fixed)'),
                            (axs[1], 'above', 'Vs of a 2-km layer above the slab top (Vp fixed)')]:
        for (code, k), db in curves.items():
            if k == kind:
                ax.plot(grid, db, label=code)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xlabel('Vs change (%)'); ax.set_ylabel('predicted change of 1.25-3 Hz slab-window radial power (dB)')
        ax.set_title(f'Perturbing {title}', fontsize=10)
    axs[0].legend(fontsize=7, ncol=2)
    fig.suptitle('Simplified flat-layer primary-conversion sensitivity (cross-check of the PyRaysum modelling), diluted by each '
                 "station's noise level", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(FIG, 'sci_rf_power_model_flat.png'), dpi=100)


if __name__ == '__main__':
    main()
