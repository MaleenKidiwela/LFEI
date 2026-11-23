#!/usr/bin/env python3
"""Per-family LFE density maps on the fault plane (along-strike x down-dip).

Events: data/relocations/fam{F}_events_3d.csv, well constrained (sd_x,sd_y<0.5 km, sd_z<1 km).
Frame: strike 293 / dip 40 NE (patch_tensors.shelly_frame); + along-strike = toward az 293 (WNW),
+ down-dip = toward az 23 and deeper.
Density: Gaussian KDE (Scott bandwidth), contours at 20/40/60/80% of peak.
Overlays: black ellipse = true patch (1 sigma, from split-station AxB covariance, data/patches/patch_tensors.csv);
grey dashed circle = typical location error of a full-station event (split-half error / sqrt 2).
The raw cloud = true patch convolved with location error, so structure smaller than the grey circle is not real.
Also: before vs after the 2024-12-05 M7 (contours of each, 50% of peak).
Output figures/cloud_density_fault_plane.png, figures/cloud_density_m7.png
"""
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Ellipse
from scipy.stats import gaussian_kde

sys.path.insert(0, 'scripts')
from patch_tensors import shelly_frame  # noqa: E402

M7 = pd.Timestamp('2024-12-05T18:44')
R = shelly_frame(293.0, 40.0)        # rows: along-strike, down-dip, normal  (E, N, Down)
P = pd.read_csv('data/patches/patch_tensors.csv').set_index('family')
LIM = 1.2                            # km


def load(F):
    e = pd.read_csv(f'data/relocations/fam{F}_events_3d.csv')
    e['time_utc'] = pd.to_datetime(e.time_utc, format='mixed')
    e = e[(e.sd_x < 0.5) & (e.sd_y < 0.5) & (e.sd_z < 1.0)].copy()
    X = e[['dx', 'dy', 'dz']].values @ R.T
    X -= np.median(X, axis=0)
    e['s'], e['d'], e['n'] = X[:, 0], X[:, 1], X[:, 2]
    return e


def kde_grid(s, d, ng=121):
    g = np.linspace(-LIM, LIM, ng)
    S, D = np.meshgrid(g, g)
    k = gaussian_kde(np.vstack([s, d]))
    Z = k(np.vstack([S.ravel(), D.ravel()])).reshape(S.shape)
    return S, D, Z / Z.max(), k.factor * np.array([s.std(), d.std()])


def overlays(a, F):
    if F in P.index and np.isfinite(P.loc[F, 'std_strike_m']):
        r = P.loc[F]
        a.add_patch(Ellipse((0, 0), 2 * r.std_strike_m / 1e3, 2 * r.std_downdip_m / 1e3, fill=False,
                            ec='k', lw=1.2, ls='-' if r.resolved else ':'))
        a.add_patch(Circle((0.8, -0.8), r.err_h_m / 1e3 / np.sqrt(2), fill=False, ec='0.4', ls='--', lw=0.9))


def main():
    fig, axs = plt.subplots(4, 7, figsize=(21, 13), sharex=True, sharey=True)
    fig2, axs2 = plt.subplots(4, 7, figsize=(21, 13), sharex=True, sharey=True)
    rows = []
    for F in range(1, 28):
        a, a2 = axs.flat[F - 1], axs2.flat[F - 1]
        e = load(F)
        n = len(e)
        for ax in (a, a2):
            ax.set_title(f'fam {F}  (n={n})', fontsize=9)
            ax.set_aspect('equal')
            ax.axhline(0, color='0.85', lw=0.5); ax.axvline(0, color='0.85', lw=0.5)
        if n < 30:
            a.text(0, 0, 'too few', ha='center'); a2.text(0, 0, 'too few', ha='center')
            continue
        S, D, Z, bw = kde_grid(e.s.values, e.d.values)
        a.contourf(S, D, Z, levels=[0.05, 0.2, 0.4, 0.6, 0.8, 1.0], cmap='magma_r', alpha=0.9)
        a.plot(e.s, e.d, ',', color='0.3', alpha=0.25)
        overlays(a, F)
        b, c = e[e.time_utc < M7], e[e.time_utc >= M7]
        row = dict(family=F, n=n, n_pre=len(b), n_post=len(c), bw_s_m=bw[0] * 1e3, bw_d_m=bw[1] * 1e3,
                   raw_std_s_m=e.s.std() * 1e3, raw_std_d_m=e.d.std() * 1e3)
        for sub, col, lab in [(b, 'C0', 'pre-M7'), (c, 'C3', 'post-M7')]:
            if len(sub) >= 30:
                S2, D2, Z2, _ = kde_grid(sub.s.values, sub.d.values, 81)
                a2.contour(S2, D2, Z2, levels=[0.5], colors=col, linewidths=1.3)
                a2.contour(S2, D2, Z2, levels=[0.9], colors=col, linewidths=0.7, linestyles='--')
        if len(c) >= 30:
            ds, dd = c.s.mean() - b.s.mean(), c.d.mean() - b.d.mean()
            se_s = np.hypot(b.s.std() / np.sqrt(len(b)), c.s.std() / np.sqrt(len(c)))
            se_d = np.hypot(b.d.std() / np.sqrt(len(b)), c.d.std() / np.sqrt(len(c)))
            row.update(post_minus_pre_s_m=ds * 1e3, se_s_m=se_s * 1e3, post_minus_pre_d_m=dd * 1e3, se_d_m=se_d * 1e3)
        rows.append(row)
    for ax in list(axs.flat) + list(axs2.flat):
        ax.set_xlim(-LIM, LIM); ax.set_ylim(LIM, -LIM)   # down-dip increases downward
    axs.flat[-1].axis('off'); axs2.flat[-1].axis('off')
    axs.flat[-1].text(0, 0.5, 'x: along strike (km), + = toward 293° (WNW)\ny: down-dip (km), + = deeper (NNE)\n'
                      'colour: event density (KDE, % of peak)\nblack ellipse: true patch, 1σ (dotted = not resolved)\n'
                      'grey dashed circle: typical location error\n(raw cloud = patch ⊛ error)', fontsize=9, va='center')
    axs2.flat[-1].text(0, 0.5, 'blue: before M7, red: after M7\nsolid = 50% of peak density, dashed = 90%\n'
                       'x: along strike (km, + WNW)\ny: down-dip (km, + deeper)', fontsize=9, va='center')
    fig.suptitle('LFE density on the fault plane (strike 293°, dip 40° NE), well-located events per family')
    fig2.suptitle('Density before vs after the 2024 M7, per family')
    fig.tight_layout(); fig2.tight_layout()
    fig.savefig('figures/cloud_density_fault_plane.png', dpi=110)
    fig2.savefig('figures/cloud_density_m7.png', dpi=110)
    T = pd.DataFrame(rows)
    T.to_csv('data/patches/cloud_density_summary.csv', index=False)
    print(T.round(0).to_string(index=False))


if __name__ == '__main__':
    main()
