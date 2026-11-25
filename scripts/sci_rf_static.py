#!/usr/bin/env python3
"""Stage 4: static receiver-function images per station.
  - QC-selected events (sci_rf_common.load_rf), Ps-moveout corrected to p = 0.06 s/km
  - radial RF section sorted by back-azimuth; 10-deg back-azimuth stacks (radial, transverse)
  - harmonic decomposition (Bianchi et al. 2010; joint radial + transverse, k = 0, 1, 2): constant term A0
    (isotropic, flat-layer part), first-harmonic amplitude (dipping interface / tilted-axis anisotropy)
  - all-event mean stacks for Gaussian a = 1, 2.5, 5, 10 with bootstrap 2-sigma bands
  - phase picks on the a=2.5 constant term with bootstrap uncertainty; predicted Ps times from the
    Bloch, Bostock & Audet (2023) slab model (t = top of the low-velocity layer, c = its base,
    m = slab Moho), using grid depths at the station and regional layer velocities from their control points
-> figures/sci_rf_static_{STA}.png, data/sci/rf_timelapse/static_{STA}.npz, static_picks.csv
Usage: sci_rf_static.py [--stations A,B]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sci_rf_common import OUT, FIG, STATIONS, PILOT, coords, load_rf, predicted_ps, P_REF, QC  # noqa: E402

TSIGN = 1.0     # sign of the transverse harmonic rows; chosen empirically in harmonics() (smaller misfit)


def bloch_at(code, C):
    """grid depths (km below sea level) of t, c, m at the station; layer velocities from the nearest Bloch
    control point within 10 km (else regional medians)."""
    import h5py
    from scipy.interpolate import RegularGridInterpolator
    la, lo = C[code][:2]
    z = {}
    for k in 'tcm':
        f = h5py.File(os.path.join('/home/seismic/LFE/data/slab/bloch2023/grids_masked', f'{k}_depth_masked.nc'), 'r')
        I = RegularGridInterpolator((f['lat'][:], f['lon'][:]), f['z'][:], bounds_error=False)
        z[k] = float(I((la, lo)))
        if not np.isfinite(z[k]):
            f2 = '/home/seismic/LFE/data/slab/bloch2023/grids_unmasked/t_depth.nc'
            if k == 't' and os.path.exists(f2):
                g = h5py.File(f2, 'r')
                z[k] = float(RegularGridInterpolator((g['lat'][:], g['lon'][:]), g['z'][:], bounds_error=False)((la, lo)))
    cp = pd.read_csv('/home/seismic/LFE/data/slab/bloch2023/control_points/control-points.csv')
    reg = cp[(cp.latitude.between(39.5, 41.3)) & (cp.longitude.between(-124.6, -122.8))]
    dkm = np.hypot((cp.latitude - la) * 111.2, (cp.longitude - lo) * 111.2 * np.cos(np.radians(la)))
    near = cp[dkm < 10]
    vel = {k: float(reg[k].median()) for k in ['vs0', 'vs1', 'vs2', 'vpvs0', 'vpvs1', 'vpvs2']}
    src = 'regional median'
    if len(near):
        r = near.iloc[np.argmin(dkm[dkm < 10].values)]
        if np.isfinite(r.vs0):
            vel = {k: float(r[k]) for k in vel}; src = f'control point {r.station}'
    elev = C[code][2] / 1000.0
    out = dict(z_t=z['t'], z_c=z['c'], z_m=z['m'], vel_src=src, **vel)
    if np.isfinite(z['t']):
        h0 = z['t'] + elev
        out['tp_t'] = predicted_ps([h0], [vel['vs0']], [vel['vpvs0']])
        if np.isfinite(z['c']):
            h1 = max(z['c'] - z['t'], 0)
            out['tp_c'] = out['tp_t'] + predicted_ps([h1], [vel['vs1']], [vel['vpvs1']])
            if np.isfinite(z['m']):
                out['tp_m'] = out['tp_c'] + predicted_ps([max(z['m'] - z['c'], 0)], [vel['vs2']], [vel['vpvs2']])
        # sensitivity of the predicted times to the overriding-plate Vp/Vs (1.70 / 1.90)
        for r in (1.70, 1.90):
            out[f'tp_t_vpvs{r:.2f}'] = predicted_ps([h0], [vel['vs0']], [r])
    return out


def harmonics(R, T, baz, tsign=None):
    """Joint R/T harmonic decomposition, k=0,1,2. -> coeffs (5, nt), tsign used, misfit."""
    ph = np.radians(baz)
    Gr = np.c_[np.ones_like(ph), np.cos(ph), np.sin(ph), np.cos(2 * ph), np.sin(2 * ph)]
    best = None
    for s in ([tsign] if tsign is not None else [1.0, -1.0]):
        Gt = s * np.c_[np.zeros_like(ph), -np.sin(ph), np.cos(ph), -np.sin(2 * ph), np.cos(2 * ph)]
        G = np.r_[Gr, Gt]
        D = np.r_[R, T]
        c, *_ = np.linalg.lstsq(G, D, rcond=None)
        mis = np.sum((D - G @ c) ** 2)
        if best is None or mis < best[2]:
            best = (c, s, mis)
    return best


def pick(trace, t, t0, t1, sign):
    w = (t >= t0) & (t <= t1)
    if not w.any():
        return np.nan, np.nan
    y = sign * trace[w]
    k = np.argmax(y)
    tt = t[w]
    if not (0 < k < len(y) - 1) or y[k] <= 0:
        return np.nan, np.nan             # extremum at the window edge (or wrong polarity): no phase
    a, b, c = y[k - 1], y[k], y[k + 1]
    den = a - 2 * b + c
    dk = 0.5 * (a - c) / den if den != 0 else 0.0
    return float(tt[k] + dk * (t[1] - t[0])), float(trace[w][k])


def static_station(code, C, rng):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m, rf, rf_raw, t, g, mall = load_rf(code, verbose=True)
    n = len(m)
    if n < 10:
        print(code, 'too few RFs', n); return None
    ig = g.index(2.5)
    R, T = rf[:, ig, 0].astype(float), rf[:, ig, 1].astype(float)
    baz = m.baz.values
    c, ts, mis = harmonics(R, T, baz)
    # bootstrap of the constant term / mean stack
    nb = 200
    A0b = np.empty((nb, len(t))); MB = np.empty((nb, len(g), len(t)))
    for b in range(nb):
        k = rng.integers(0, n, n)
        A0b[b] = harmonics(R[k], T[k], baz[k], ts)[0][0]
        MB[b] = rf[k, :, 0].mean(0)
    A0 = c[0]; A0se = A0b.std(0)
    mean = rf[:, :, 0].mean(0); mse = MB.std(0)
    B = bloch_at(code, C)
    # picks on A0 (a=2.5): direct P, strongest positive / negative in 0.1-1 s, and phases near predictions
    picks = {}
    picks['P'] = pick(A0, t, -0.5, 0.5, 1)
    picks['shallow+'] = pick(A0, t, 0.3, 1.2, 1)
    picks['shallow-'] = pick(A0, t, 0.3, 1.2, -1)
    for ph, sgn, half in [('t', -1, 1.0), ('c', 1, 0.8), ('m', 1, 1.0)]:
        tp = B.get(f'tp_{ph}', np.nan)
        if np.isfinite(tp):
            picks[ph] = pick(A0, t, tp - half, tp + half, sgn)
    rows = []
    for k, (tt, aa) in picks.items():
        bt = []
        for b in range(nb):
            win = {'P': (-0.5, 0.5, 1), 'shallow+': (0.3, 1.2, 1), 'shallow-': (0.3, 1.2, -1)}.get(k)
            if win is None:
                tp = B[f'tp_{k}']; half = {'t': 1.0, 'c': 0.8, 'm': 1.0}[k]; sgn = -1 if k == 't' else 1
                win = (tp - half, tp + half, sgn)
            bt.append(pick(A0b[b], t, *win)[0])
        rows.append(dict(sta=code, phase=k, t_obs=tt, t_se=float(np.nanstd(bt)), amp=aa,
                         amp_se=float(A0se[np.argmin(np.abs(t - tt))]) if np.isfinite(tt) else np.nan,
                         t_pred=B.get(f'tp_{k}', np.nan) if k in 'tcm' else np.nan,
                         z_model=B.get(f'z_{k}', np.nan) if k in 'tcm' else np.nan))
    np.savez_compressed(os.path.join(OUT, f'static_{code}.npz'), t=t, A0=A0, A0se=A0se, harm=c, mean=mean,
                        mean_se=mse, gauss=np.array(g), n=n, tsign=ts)
    # ------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(17, 11))
    xl = (-1, 8)
    ax1 = fig.add_axes([0.05, 0.08, 0.20, 0.82])
    o = np.argsort(baz)
    sc = max(1.2, n / 120) / max(np.percentile(np.abs(R), 99), 1e-6)
    for j, k in enumerate(o):
        y = j + R[k] * sc
        ax1.fill_between(t, j, y, where=y > j, color='r', lw=0); ax1.fill_between(t, j, y, where=y < j, color='b', lw=0)
    ax1.set_xlim(xl); ax1.set_ylim(-1, n + 1)
    yt = np.linspace(0, n - 1, 8).astype(int)
    ax1.set_yticks(yt); ax1.set_yticklabels([f'{baz[o][i]:.0f}' for i in yt])
    ax1.set_ylabel('back-azimuth (deg), events sorted'); ax1.set_xlabel('time after P (s)')
    ax1.set_title(f'Radial RFs, Gaussian a=2.5 (n={n})', fontsize=10)
    # baz-binned stacks
    for col, (comp, lab) in enumerate([(R, 'radial'), (T, 'transverse')]):
        ax = fig.add_axes([0.30 + 0.17 * col, 0.08, 0.15, 0.82])
        for b0 in range(0, 360, 10):
            k = (baz >= b0) & (baz < b0 + 10)
            if k.sum() == 0:
                continue
            s = comp[k].mean(0) * 4
            ax.fill_between(t, b0 + 5, b0 + 5 + s * 10, where=s > 0, color='r', lw=0)
            ax.fill_between(t, b0 + 5, b0 + 5 + s * 10, where=s < 0, color='b', lw=0)
            ax.text(xl[1] - 0.1, b0 + 5, str(int(k.sum())), fontsize=6, ha='right', va='center')
        ax.set_xlim(xl); ax.set_ylim(0, 360); ax.set_xlabel('time after P (s)')
        ax.set_ylabel('back-azimuth bin (deg)') if col == 0 else None
        ax.set_title(f'{lab}, 10-deg back-azimuth stacks\n(number of events at right)', fontsize=10)
    # harmonics + stacks
    ax = fig.add_axes([0.66, 0.56, 0.32, 0.36])
    ax.fill_between(t, A0 - 2 * A0se, A0 + 2 * A0se, color='0.8')
    ax.plot(t, A0, 'k', lw=1.5, label='constant term (k=0)')
    ax.plot(t, np.hypot(c[1], c[2]), 'C1', lw=1, label='first-harmonic amplitude (k=1)')
    ax.plot(t, np.hypot(c[3], c[4]), 'C2', lw=1, label='second-harmonic amplitude (k=2)')
    cols = {'t': 'royalblue', 'c': 'm', 'm': 'navy'}
    names = {'t': 'top of slab low-velocity layer', 'c': 'base of low-velocity layer', 'm': 'slab Moho'}
    for ph in 'tcm':
        tp = B.get(f'tp_{ph}', np.nan)
        if np.isfinite(tp):
            ax.axvline(tp, color=cols[ph], ls='--', lw=1, label=f'predicted {names[ph]} ({B[f"z_{ph}"]:.1f} km)')
    for r in rows:
        if np.isfinite(r['t_obs']):
            ax.errorbar(r['t_obs'], r['amp'], xerr=2 * r['t_se'], fmt='o', ms=3, color='k')
    ax.axhline(0, color='k', lw=0.5); ax.set_xlim(xl); ax.legend(fontsize=7, loc='upper right')
    ax.set_title('Harmonic decomposition, a=2.5 (grey: 2-sigma bootstrap); dots: picks', fontsize=10)
    ax.set_xlabel('time after P (s), moveout-corrected to p=0.06 s/km')
    ax = fig.add_axes([0.66, 0.08, 0.32, 0.38])
    for i, a in enumerate(g):
        off = -0.6 * i
        ax.fill_between(t, off + mean[i] - 2 * mse[i], off + mean[i] + 2 * mse[i], color='0.8')
        ax.plot(t, off + mean[i], 'k', lw=1)
        ax.text(xl[0] + 0.1, off + 0.15, f'a={a:g}', fontsize=8)
    ax.set_xlim(xl); ax.set_yticks([]); ax.set_xlabel('time after P (s)')
    ax.set_title('All-event mean radial RF by Gaussian width a (2-sigma bootstrap)', fontsize=10)
    dep = C[code][3]
    extra = f', sensor depth {dep:.0f} m (borehole: surface ghost at ~2h/Vs)' if dep > 50 else ''
    fig.suptitle(f'{code} static P receiver functions, {m.time.min():%Y-%m} to {m.time.max():%Y-%m}, '
                 f'{n} of {len(mall)} records pass quality control{extra}', fontsize=12)
    fig.savefig(os.path.join(FIG, f'sci_rf_static_{code.split(".")[1]}.png'), dpi=110)
    plt.close(fig)
    for r in rows:
        r.update(n=n, n_all=len(mall), tsign=ts, vel_src=B['vel_src'], tp_t_vpvs1_70=B.get('tp_t_vpvs1.70', np.nan),
                 tp_t_vpvs1_90=B.get('tp_t_vpvs1.90', np.nan))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stations', default='pilot')
    a = ap.parse_args()
    sts = PILOT if a.stations == 'pilot' else (list(STATIONS) if a.stations == 'all' else a.stations.split(','))
    C = coords()
    rng = np.random.default_rng(1)
    fn = os.path.join(OUT, 'static_picks.csv')
    old = pd.read_csv(fn) if os.path.exists(fn) else pd.DataFrame(columns=['sta'])
    rows = []
    for s in sts:
        if not os.path.exists(os.path.join(OUT, f'rf_{s}.npz')):
            continue
        r = static_station(s, C, rng)
        if r:
            rows += r
    new = pd.DataFrame(rows)
    out = pd.concat([old[~old.sta.isin(new.sta)], new]) if len(new) else old
    out.to_csv(fn, index=False)
    print(new.round(3).to_string())


if __name__ == '__main__':
    main()
