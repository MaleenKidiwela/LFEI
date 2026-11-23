#!/usr/bin/env python3
"""Pick P and S on a family's master stacks and locate the family by grid search in
precomputed travel-time volumes (3D Furlong 2024 or its 1D starting model).

Picking (per station, times relative to catalog time):
  S: horizontals if available (E/N or 1/2), else vertical. Summed smoothed envelope ->
     peak in [-3, 28] s; AIC onset in [peak-2.5, peak+0.1] s on the strongest component.
  P: vertical; AIC in [S - 1.45*SPpred, S - 0.55*SPpred] where SPpred is the predicted S-P
     from the starting location; kept only if onset energy ratio >= P_MIN_RATIO.
  sigma from stack SNR (S: 0.05/0.10/0.20 s; P: 0.10/0.15/0.25 s); SNR < SNR_MIN rejected.
Location: L1 misfit with free origin time (weighted median of residuals) over every node;
bootstrap (resampling picks) for uncertainty.
"""
import argparse
import json
import os

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.signal import hilbert

SNR_MIN, P_MIN_RATIO = 5.0, 2.5
EXCLUDE = {'NC.KPP..SHZ', 'NC.KCPB..HHN', 'NC.KCPB..HHE', 'NC.KBS..SHZ'}


def aic_pick(x):
    n = len(x)
    k = np.arange(1, n - 1)
    cs = np.cumsum(x ** 2)
    v1 = cs[k - 1] / k
    v2 = (cs[-1] - cs[k - 1]) / (n - k)
    aic = k * np.log(v1 + 1e-30) + (n - k - 1) * np.log(v2 + 1e-30)
    return int(k[np.argmin(aic)])


def smooth(x, n):
    return np.convolve(x, np.ones(n) / n, mode='same')


def pick_station(comps, t, sr, sp_pred):
    """comps: dict channel-code -> stack (normalised). Returns dict with picks."""
    hor = [c for c in comps if c[-1] in 'EN12']
    ver = [c for c in comps if c[-1] == 'Z']
    s_set = hor if hor else ver
    env = sum(smooth(np.abs(hilbert(comps[c])) ** 2, int(0.2 * sr)) for c in s_set)
    win = (t > -3) & (t < 28)
    noise = (t > -15) & (t < -5)
    pk = np.where(win)[0][np.argmax(env[win])]
    snr_s = np.sqrt(env[pk] / np.mean(env[noise]))
    out = dict(snr_s=snr_s)
    i0, i1 = pk - int(2.5 * sr), pk + int(0.1 * sr)
    best = max(s_set, key=lambda c: np.max(np.abs(comps[c][i0:i1])))
    ks = i0 + aic_pick(comps[best][i0:i1])
    out['S'] = t[ks]
    out['S_comp'] = 'H' if hor else 'Z'
    out['sig_S'] = 0.05 if snr_s > 20 else 0.10 if snr_s > 8 else 0.20
    if snr_s < SNR_MIN:
        out['S'] = np.nan
    if ver and np.isfinite(out['S']):
        z = comps[ver[0]]
        j0 = np.searchsorted(t, out['S'] - 1.45 * sp_pred)
        j1 = np.searchsorted(t, out['S'] - 0.55 * sp_pred)
        if j1 - j0 > int(0.5 * sr):
            kp = j0 + aic_pick(z[j0:j1])
            a = np.sqrt(np.mean(z[kp:kp + int(0.5 * sr)] ** 2))
            b = np.sqrt(np.mean(z[max(0, kp - int(1.0 * sr)):kp] ** 2)) + 1e-20
            out['p_ratio'] = a / b
            if a / b >= P_MIN_RATIO:
                out['P'] = t[kp]
                out['sig_P'] = 0.10 if a / b > 6 else 0.15 if a / b > 4 else 0.25
    return out


def repick_s(comps, t, sr, t_pred, half=1.5):
    """Re-pick S with AIC inside [t_pred-half, t_pred+half/2] on horizontals (else vertical)."""
    hor = [c for c in comps if c[-1] in 'EN12']
    s_set = hor if hor else [c for c in comps if c[-1] == 'Z']
    i0, i1 = np.searchsorted(t, t_pred - half), np.searchsorted(t, t_pred + half / 2)
    if i1 - i0 < int(0.5 * sr):
        return np.nan
    best = max(s_set, key=lambda c: np.max(np.abs(comps[c][i0:i1])))
    return t[i0 + aic_pick(comps[best][i0:i1])]


def locate(obs, TT, nboot=300, rng=None):
    """obs: list of (key, t_obs, sigma); TT: dict key -> tt volume. L1 with free origin."""
    keys = [o[0] for o in obs]
    tobs = np.array([o[1] for o in obs])[:, None, None, None]
    w = 1.0 / np.array([o[2] for o in obs])[:, None, None, None]
    T = np.stack([TT[k] for k in keys])            # (n, nx, ny, nz)
    R = tobs - T

    def solve(idx):
        r, ww = R[idx], w[idx]
        t0 = np.median(r, axis=0)                  # free origin (L1 optimum ~ median)
        mis = np.sum(ww * np.abs(r - t0[None]), axis=0) / np.sum(ww)
        b = np.unravel_index(np.argmin(mis), mis.shape)
        return b, mis, t0
    best, mis, t0 = solve(np.arange(len(obs)))
    rng = rng or np.random.default_rng(0)
    boots = []
    for _ in range(nboot):
        idx = rng.integers(0, len(obs), len(obs))
        if len(np.unique(idx)) < 4:
            continue
        boots.append(solve(idx)[0])
    return best, mis, t0[best], np.array(boots), (R[(slice(None),) + best] - t0[best]).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', type=int, default=23)
    ap.add_argument('--templates', default=None)
    ap.add_argument('--model', default='3d')
    ap.add_argument('--lat', type=float, default=None, help='start location; default Shelly family location')
    ap.add_argument('--lon', type=float, default=None)
    ap.add_argument('--dep', type=float, default=None)
    ap.add_argument('--tag', default='')
    ap.add_argument('--statics', default=None,
                    help='station residual table (phase,sta,median,count) from validate_location_eq.py; '
                         'medians with count>=5 are subtracted from observed times')
    ap.add_argument('--allow-vertical-s', dest='s_horizontal_only', action='store_false',
                    help='also use S picked on vertical-only stations (unreliable; off by default)')
    ap.add_argument('--s-scales', type=float, nargs='+',
                    default=[1.0, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12],
                    help='factors applied to S travel times (= Vs scaled by 1/f) to test Vp/Vs')
    a = ap.parse_args()
    F = a.family
    fl = pd.read_csv('data/families/shelly_family_locations.csv').set_index('idnum').loc[F]
    a.lat = fl.lat if a.lat is None else a.lat
    a.lon = fl.lon if a.lon is None else a.lon
    a.dep = fl.depth_km if a.dep is None else a.dep
    print(f'family {F}: start (Shelly) {a.lat:.4f} {a.lon:.4f} {a.dep:.1f} km', flush=True)
    T = h5py.File(a.templates or f'data/templates/fam{F}_templates.h5', 'r')
    sr, pre = T.attrs['sr'], T.attrs['pre']
    g = json.load(open(f'data/velocity/tt_{a.model}/grid.json'))
    gx, gy, gz = map(np.array, (g['sub_x'], g['sub_y'], g['sub_z']))
    tr = Transformer.from_crs('EPSG:4326', g['proj'], always_xy=True)
    inv = Transformer.from_crs(g['proj'], 'EPSG:4326', always_xy=True)
    fx, fy = tr.transform(a.lon, a.lat)
    i0, j0, k0 = np.argmin(abs(gx - fx)), np.argmin(abs(gy - fy)), np.argmin(abs(gz - a.dep))

    by_sta = {}
    for cid in T:
        if cid in EXCLUDE:
            continue
        n, s, l, c = cid.split('.')
        st = T[cid]['stack'][:].astype(float)
        by_sta.setdefault(f'{n}.{s}', {})[c] = st / np.max(np.abs(st))
    t = np.arange(len(next(iter(T.values()))['stack'])) / sr + pre

    picks, TTp, TTs = [], {}, {}
    for sta, comps in sorted(by_sta.items()):
        f = f'data/velocity/tt_{a.model}/{sta}.npz'
        if not os.path.exists(f):
            continue
        z = np.load(f)
        TTp[sta], TTs[sta] = z['tp'], z['ts']
        sp_pred = z['ts'][i0, j0, k0] - z['tp'][i0, j0, k0]
        p = pick_station(comps, t, sr, sp_pred)
        p['sta'] = sta
        picks.append(p)
    pk = pd.DataFrame(picks)
    obs = [(('S', r.sta), r.S, r.sig_S) for r in pk.itertuples()
           if np.isfinite(r.S) and not (a.s_horizontal_only and r.S_comp == 'Z')]
    if 'P' in pk:
        obs += [(('P', r.sta), r.P, r.sig_P) for r in pk.itertuples() if np.isfinite(getattr(r, 'P', np.nan))]
    statics = {}
    if a.statics:
        st_ = pd.read_csv(a.statics)
        statics = {(r.phase, r.sta): r['median'] for _, r in st_.iterrows() if r['count'] >= 5}
        print(f'applying {len(statics)} station statics', flush=True)

    def corr(obs):
        return [(k, tt - statics.get(k, 0.0), sg) for k, tt, sg in obs]

    # pass 1 at s-scale 1: locate, then re-pick all S near predicted, then search s-scale
    def run(obs, f):
        obs = corr(obs)
        TT = {('S', s): v * f for s, v in TTs.items()}
        TT.update({('P', s): v for s, v in TTp.items()})
        return locate(obs, TT, nboot=0)
    best, mis, t0, _, res = run(obs, 1.0)
    scan = []
    for f in a.s_scales:
        for it in range(2):
            newobs = [o for o in obs if o[0][0] == 'P']
            for r in pk.itertuples():
                if r.snr_s < SNR_MIN or (a.s_horizontal_only and r.S_comp == 'Z'):
                    continue
                tp = TTs[r.sta][best] * f + t0
                ts = repick_s(by_sta[r.sta], t, sr, tp)
                if np.isfinite(ts):
                    newobs.append((('S', r.sta), ts, r.sig_S))
            best, mis, t0, _, res = run(newobs, f)
        scan.append((f, float(mis[best]), best, newobs))
        print(f'S-scale {f:.2f}: L1 misfit {mis[best]:.3f} s, best node {best}', flush=True)
    f_best, _, _, obs = min(scan, key=lambda x: x[1])
    TT = {('S', s): v * f_best for s, v in TTs.items()}
    TT.update({('P', s): v for s, v in TTp.items()})
    best, mis, t0, boots, res = locate(corr(obs), TT)
    pk['S_final'] = pk.sta.map({o[0][1]: o[1] for o in obs if o[0][0] == 'S'})
    bx, by, bz = gx[best[0]], gy[best[1]], gz[best[2]]
    blon, blat = inv.transform(bx, by)
    bxy = np.c_[gx[boots[:, 0]], gy[boots[:, 1]], gz[boots[:, 2]]]
    sd = bxy.std(0)
    dshelly = np.hypot(bx - fx, by - fy)
    out = dict(family=F, model=a.model, lat=blat, lon=blon, depth_km=float(bz),
               origin_rel_catalog_s=float(t0), n_S=sum(1 for o in obs if o[0][0] == 'S'),
               n_P=sum(1 for o in obs if o[0][0] == 'P'), sd_x_km=float(sd[0]), sd_y_km=float(sd[1]),
               sd_z_km=float(sd[2]), l1_misfit_s=float(mis[best]),
               horiz_offset_from_shelly_km=float(dshelly), dz_from_shelly_km=float(bz - a.dep),
               rms_res_s=float(np.sqrt(np.mean(res ** 2))), s_scale=f_best,
               vpvs_scan=[(f, m) for f, m, _, _ in scan])
    os.makedirs('data/locations', exist_ok=True)
    tag = f'fam{F}_{a.model}{a.tag}'
    json.dump(out, open(f'data/locations/{tag}.json', 'w'), indent=1)
    pk.assign(res=np.nan).to_csv(f'data/locations/{tag}_picks.csv', index=False)
    rr = pd.DataFrame([(o[0][0], o[0][1], o[1], o[2], r) for o, r in zip(obs, res)],
                      columns=['phase', 'sta', 't_obs', 'sigma', 'residual'])
    rr.to_csv(f'data/locations/{tag}_residuals.csv', index=False)
    print(json.dumps(out, indent=1))
    print(rr.sort_values(['phase', 'residual']).to_string(index=False))

    # QC figures: picks on stacks; misfit slices
    fig, ax = plt.subplots(1, 3, figsize=(16, 9), gridspec_kw=dict(width_ratios=[2, 1, 1]))
    order = sorted(pk.sta, key=lambda s: TTs[s][best])
    for n, sta in enumerate(order):
        comps = by_sta[sta]
        for k, (c, x) in enumerate(sorted(comps.items())):
            ax[0].plot(t, x * 0.3 + n + (k - 1) * 0.12, lw=0.4, color=['k', 'C0', 'C2'][k % 3])
        r = pk[pk.sta == sta].iloc[0]
        if np.isfinite(r.S_final):
            ax[0].plot([r.S_final] * 2, [n - 0.45, n + 0.45], 'r', lw=1.5)
        if np.isfinite(r.get('P', np.nan)):
            ax[0].plot([r.P] * 2, [n - 0.45, n + 0.45], 'b', lw=1.5)
        ax[0].plot(TTs[sta][best] * f_best + t0, n, 'r^', ms=4)
        ax[0].plot(TTp[sta][best] + t0, n, 'b^', ms=4)
        ax[0].text(29, n, f'{sta} S-SNR {r.snr_s:.0f}', fontsize=6, va='center')
    ax[0].set_xlim(-6, 35)
    ax[0].set_yticks([])
    ax[0].set_xlabel('time rel. catalog time (s)')
    ax[0].set_title(f'picks (lines) and predicted from best location (triangles), {a.model}')
    ext = [gx[0] - fx, gx[-1] - fx, gy[0] - fy, gy[-1] - fy]
    ax[1].imshow(mis[:, :, best[2]].T, origin='lower', extent=ext, cmap='viridis_r')
    ax[1].plot(0, 0, 'w*', ms=12)
    ax[1].plot(bx - fx, by - fy, 'r+', ms=12)
    ax[1].set_title(f'misfit map at z={bz:.1f} km\n(white star = Shelly centroid)')
    ax[1].set_xlabel('E (km)')
    ax[1].set_ylabel('N (km)')
    ax[2].imshow(mis[:, best[1], :].T, origin='upper', aspect='auto',
                 extent=[gx[0] - fx, gx[-1] - fx, gz[-1], gz[0]], cmap='viridis_r')
    ax[2].plot(0, a.dep, 'w*', ms=12)
    ax[2].plot(bx - fx, bz, 'r+', ms=12)
    ax[2].set_title('misfit E-W section')
    ax[2].set_xlabel('E (km)')
    ax[2].set_ylabel('depth (km)')
    fig.tight_layout()
    fig.savefig(f'figures/{tag}_location.png', dpi=120)


if __name__ == '__main__':
    main()
