#!/usr/bin/env python3
"""True LFE patch geometry from split-station relocations (Fable ideas 2, 3, 5, 8).

Each event relocated twice with disjoint azimuth-interleaved station halves A and B
(data/relocations/fam{F}_events_3d_half{A,B}.csv). Errors of A and B are independent, so
  C_true = sym <(xA - mean)(xB - mean)^T>     (real spread, unbiased by error)
  C_err  = cov(xA - xB) / 2                   (error per half)
Per family: C_true eigen-decomposition (patch axes, normal -> plane strike/dip), Shelly frame
(strike 293, dip 40 NE) projections, bootstrap CIs, shuffled-pairing null.
Idea 3: consecutive-event spatial memory rho_k = <dA_i . dB_{i+k}> / tr(C_true) (k = 1..5) by time gap
class; negative = next event avoids the spot that just slipped (stress shadow), positive = persistent
sub-asperity. Null: shuffled order.
Idea 8: where did each family restart after the M7 silence? centroid of first 20 relocated events after the
M7 (mean of A and B) relative to the long-term centroid, in the Shelly frame, with error.
Idea 5: M7 cessation vs patch length, with rate control.
Outputs data/patches/patch_tensors.csv, data/patches/memory.csv, figures/patch_tensors.png
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

M7 = pd.Timestamp('2024-12-05T18:44')
rng = np.random.default_rng(0)


def shelly_frame(strike=293.0, dip=40.0):
    st, dp = np.radians(strike), np.radians(dip)
    s = np.array([np.sin(st), np.cos(st), 0.0])                                  # along strike (E,N,Down)
    d = np.array([np.sin(st + np.pi / 2) * np.cos(dp), np.cos(st + np.pi / 2) * np.cos(dp), np.sin(dp)])  # down-dip
    return np.vstack([s, d, np.cross(s, d)])


def ctrue(XA, XB):
    a, b = XA - XA.mean(0), XB - XB.mean(0)
    return (a.T @ b + b.T @ a) / (2 * len(a))


def plane_from_normal(n):
    n = n / np.linalg.norm(n)
    if n[2] < 0:
        n = -n                                     # point down
    dip = np.degrees(np.arccos(abs(n[2])))
    dip = 90 - dip if False else np.degrees(np.arcsin(np.hypot(n[0], n[1])))
    dipdir = np.degrees(np.arctan2(-n[0], -n[1])) % 360      # horizontal direction the plane dips toward
    return (dipdir - 90) % 360, dip, dipdir


def main():
    os.makedirs('data/patches', exist_ok=True)
    R = shelly_frame()
    summ = {int(json.load(open(f))['family']): json.load(open(f)) for f in
            [f'data/reports/fam{F}_summary.json' for F in range(1, 28)] if os.path.exists(f)}
    rows, mem = [], []
    for F in range(1, 28):
        try:
            A = pd.read_csv(f'data/relocations/fam{F}_events_3d_halfA.csv')
            B = pd.read_csv(f'data/relocations/fam{F}_events_3d_halfB.csv')
        except Exception:
            continue
        m = A.merge(B, on=['ev', 'time_utc'], suffixes=('_a', '_b'))
        m = m[(m.sd_x_a < 0.5) & (m.sd_y_a < 0.5) & (m.sd_z_a < 1) & (m.sd_x_b < 0.5) & (m.sd_y_b < 0.5) & (m.sd_z_b < 1)]
        m = m.assign(t=pd.to_datetime(m.time_utc, format='mixed')).sort_values('t').reset_index(drop=True)
        n = len(m)
        if n < 40:
            rows.append(dict(family=F, n=n, note='too few events in both halves'))
            continue
        XA, XB = m[['dx_a', 'dy_a', 'dz_a']].values, m[['dx_b', 'dy_b', 'dz_b']].values
        C = ctrue(XA, XB)
        Ce = np.cov((XA - XB).T) / 2
        w, v = np.linalg.eigh(C)
        boot = []
        for _ in range(300):
            i = rng.integers(0, n, n)
            boot.append(np.linalg.eigvalsh(ctrue(XA[i], XB[i])))
        boot = np.array(boot)
        null = np.array([np.linalg.eigvalsh(ctrue(XA, XB[rng.permutation(n)])) for _ in range(300)])
        Cr = R @ C @ R.T
        strike, dip, dipdir = plane_from_normal(v[:, 0])
        L = lambda lam: 2 * 1.645 * np.sqrt(max(lam, 0)) * 1000   # noqa: E731  5-95% extent (m) if Gaussian
        r = dict(family=F, n=n,
                 L1_m=L(w[2]), L2_m=L(w[1]), L3_m=L(w[0]),
                 L1_lo=L(np.percentile(boot[:, 2], 16)), L1_hi=L(np.percentile(boot[:, 2], 84)),
                 L1_null95=L(np.percentile(null[:, 2], 95)), L2_null95=L(np.percentile(null[:, 1], 95)),
                 resolved=bool(w[2] > np.percentile(null[:, 2], 95)),
                 axis1_az=float(np.degrees(np.arctan2(v[0, 2], v[1, 2])) % 180),
                 axis1_plunge=float(np.degrees(np.arcsin(abs(v[2, 2])))),
                 plane_strike=strike, plane_dip=dip, plane_dipdir=dipdir,
                 std_strike_m=np.sqrt(max(Cr[0, 0], 0)) * 1000, std_downdip_m=np.sqrt(max(Cr[1, 1], 0)) * 1000,
                 std_normal_m=np.sqrt(max(Cr[2, 2], 0)) * 1000,
                 err_h_m=np.sqrt((Ce[0, 0] + Ce[1, 1]) / 2) * 1000, err_z_m=np.sqrt(Ce[2, 2]) * 1000)
        # idea 3: consecutive-event memory
        dA, dB = XA - XA.mean(0), XB - XB.mean(0)
        tr = np.trace(C)
        gap = m.t.diff().dt.total_seconds().values
        for k in range(1, 6):
            prod = np.sum(dA[:-k] * dB[k:], axis=1)
            prod2 = np.sum(dB[:-k] * dA[k:], axis=1)
            pr = (prod + prod2) / 2
            g = pd.Series(m.t.values[k:] - m.t.values[:-k]).dt.total_seconds().values
            for lab, sel in (('<1min', g < 60), ('1min-1h', (g >= 60) & (g < 3600)), ('1h-1d', (g >= 3600) & (g < 86400)),
                             ('>1d', g >= 86400), ('all', np.ones(len(g), bool))):
                if sel.sum() < 30:
                    continue
                rho = pr[sel].mean() / tr
                se = pr[sel].std() / np.sqrt(sel.sum()) / tr
                mem.append(dict(family=F, k=k, gap=lab, n=int(sel.sum()), rho=rho, se=se))
        # idea 8: restart location after the M7 silence (first 20 events after M7)
        Xm = (XA + XB) / 2
        post = m.t >= M7
        if post.sum() >= 20:
            i0 = np.where(post)[0][:20]
            c_first = (Xm[i0] - Xm.mean(0)) @ R.T * 1000
            err = np.sqrt(np.diag(R @ (Ce / 2 + C) @ R.T) / 20) * 1000
            r.update(restart_strike_m=c_first[:, 0].mean(), restart_downdip_m=c_first[:, 1].mean(),
                     restart_normal_m=c_first[:, 2].mean(), restart_se_strike_m=err[0], restart_se_downdip_m=err[1])
        s = summ.get(F, {})
        r.update(m7_gapfull_d=s.get('m7_gapfull_d'), m7_t10_d=s.get('m7_t10_d'), rate_per_yr=s.get('rate_per_yr'),
                 depth_km=s.get('loc3d_depth_km'), lon=s.get('loc3d_lon'))
        rows.append(r)
        print(f"fam {F:2d} n={n:5d}  L1 {r['L1_m']:4.0f} m [{r['L1_lo']:.0f}-{r['L1_hi']:.0f}] (null95 {r['L1_null95']:.0f})  "
              f"L2 {r['L2_m']:4.0f}  L3 {r['L3_m']:4.0f}  axis1 az {r['axis1_az']:3.0f} plunge {r['axis1_plunge']:2.0f}  "
              f"plane strike {strike:3.0f} dip {dip:2.0f}  Shelly-frame std s/d/n {r['std_strike_m']:.0f}/{r['std_downdip_m']:.0f}/{r['std_normal_m']:.0f} m",
              flush=True)
    P = pd.DataFrame(rows)
    P.to_csv('data/patches/patch_tensors.csv', index=False)
    M = pd.DataFrame(mem)
    M.to_csv('data/patches/memory.csv', index=False)
    ok = P[P.get('resolved', False) == True] if 'resolved' in P else P.iloc[0:0]  # noqa: E712
    print(f'\nresolved patches: {len(ok)} of {P.n.notna().sum()}')
    if len(ok):
        print(f"median L1 {ok.L1_m.median():.0f} m, L2 {ok.L2_m.median():.0f} m; long axis azimuth median {ok.axis1_az.median():.0f} "
              f"(Shelly strike 293/113)")
        ax = np.radians(2 * ok.axis1_az)
        print(f"axial mean of long axis: {np.degrees(np.arctan2(np.sin(ax).mean(), np.cos(ax).mean())) / 2 % 180:.0f} deg, "
              f"R={np.hypot(np.sin(ax).mean(), np.cos(ax).mean()):.2f}")
        # idea 5: cessation vs size, rate-controlled: expected events during the gap = gap * rate
        q = ok.dropna(subset=['m7_gapfull_d', 'rate_per_yr'])
        q = q.assign(gap_norm=q.m7_gapfull_d * q.rate_per_yr / 365.25)
        for y in ('m7_gapfull_d', 'gap_norm', 'm7_t10_d'):
            rr, p = stats.spearmanr(q.L1_m, q[y])
            print(f'  M7 {y} vs patch length L1: spearman r={rr:+.2f} p={p:.3f} (n={len(q)})')
        rr, p = stats.spearmanr(q.L1_m * q.L2_m, q.m7_gapfull_d)
        print(f'  M7 gap vs patch area L1*L2: r={rr:+.2f} p={p:.3f}')
    if len(M):
        a = M[M.gap == 'all'].groupby('k').apply(lambda g: pd.Series(dict(
            rho=np.average(g.rho, weights=1 / g.se ** 2), se=1 / np.sqrt((1 / g.se ** 2).sum()))), include_groups=False)
        print('\nconsecutive-event memory (all families, inverse-variance): \n', a.round(4).to_string())
        b = M[M.k == 1].groupby('gap').apply(lambda g: pd.Series(dict(
            rho=np.average(g.rho, weights=1 / g.se ** 2), se=1 / np.sqrt((1 / g.se ** 2).sum()), n=g.n.sum())), include_groups=False)
        print('k=1 by time gap:\n', b.round(4).to_string())
    if 'restart_strike_m' in P:
        q = P.dropna(subset=['restart_strike_m'])
        zs = q.restart_strike_m / q.restart_se_strike_m
        zd = q.restart_downdip_m / q.restart_se_downdip_m
        print(f'\nM7 restart location (first 20 events) rel. long-term centroid, {len(q)} families: '
              f'along-strike median {q.restart_strike_m.median():+.0f} m (|z|>2 in {(abs(zs) > 2).sum()}), '
              f'down-dip median {q.restart_downdip_m.median():+.0f} m (|z|>2 in {(abs(zd) > 2).sum()})')
        print(q[['family', 'restart_strike_m', 'restart_se_strike_m', 'restart_downdip_m', 'restart_se_downdip_m']].round(0).to_string(index=False))
    # figure: patch ellipses (strike vs down-dip) per family at their positions
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    lat0, lon0 = 40.06, -123.625
    for _, r in P.dropna(subset=['L1_m']).iterrows():
        j = json.load(open(f"data/locations/fam{int(r.family)}_3d.json"))
        x, y = (j['lon'] - lon0) * 111.19 * np.cos(np.radians(lat0)), (j['lat'] - lat0) * 111.19
        th = np.radians(90 - r.axis1_az)
        tt = np.linspace(0, 2 * np.pi, 60)
        a1, a2 = r.L1_m / 2000 * 2, r.L2_m / 2000 * 2      # exaggerate x2 for visibility
        ex = x + a1 * np.cos(tt) * np.cos(th) - a2 * np.sin(tt) * np.sin(th)
        ey = y + a1 * np.cos(tt) * np.sin(th) + a2 * np.sin(tt) * np.cos(th)
        ax[0].plot(ex, ey, color='C0' if r.resolved else '0.7', lw=1.2)
        ax[0].annotate(str(int(r.family)), (x, y), fontsize=7, ha='center')
    st = np.radians(90 - 293)
    ax[0].plot([-6, 6], [-6 * np.tan(st), 6 * np.tan(st)], 'r--', lw=0.8, label='Shelly strike 293')
    ax[0].set_aspect('equal')
    ax[0].set_xlabel('E (km)')
    ax[0].set_ylabel('N (km)')
    ax[0].legend(fontsize=8)
    ax[0].set_title('True patch extent (5-95%, x2 exaggerated); grey = not resolved')
    if len(M):
        for g, c in (('<1min', 'C0'), ('1min-1h', 'C1'), ('1h-1d', 'C2'), ('>1d', 'C3')):
            s = M[M.gap == g].groupby('k').apply(lambda q: pd.Series(dict(
                rho=np.average(q.rho, weights=1 / q.se ** 2), se=1 / np.sqrt((1 / q.se ** 2).sum()))), include_groups=False)
            if len(s):
                ax[1].errorbar(s.index + {'<1min': -0.15, '1min-1h': -0.05, '1h-1d': 0.05, '>1d': 0.15}[g], s.rho, yerr=s.se,
                               fmt='o-', color=c, label=f'gap {g}')
        ax[1].axhline(0, color='k', lw=0.6)
        ax[1].set_xlabel('event lag k (k-th next relocated event)')
        ax[1].set_ylabel('spatial memory rho_k = <dA_i . dB_i+k> / tr(C)')
        ax[1].legend(fontsize=8)
        ax[1].set_title('Does the next LFE go to the same spot? (+ same, - avoids)')
    fig.tight_layout()
    fig.savefig('figures/patch_tensors.png', dpi=130)


if __name__ == '__main__':
    main()
