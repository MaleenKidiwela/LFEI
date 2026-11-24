#!/usr/bin/env python3
"""Source-region dv/v, optional item: scan vertical-component LFE stacks for a SOURCE-SIDE converted phase.

A velocity interface a height h above the LFEs converts the up-going S to P (Sp) or the up-going P to S (Ps):
   Sp precursor: t = T_S - h/cos(i) * (1/Vs - 1/Vp)        Ps: t = T_P + h/cos(i) * (1/Vs - 1/Vp)
cos(i) and the source Vp, Vs per family-station from the 3D travel-time grids (geometry_3d.csv), so the lead time
has a station-dependent MOVEOUT (rays leave the source at 60-80 deg from vertical here).
Data: family template stacks (data/templates/fam{F}_templates.h5, 2-8 Hz, all events), vertical channels with an
S pick (and a P pick for Ps), template SNR >= 20.  Envelope normalised by the direct-S (Sp) or direct-P (Ps) peak.
For h = 0.25..10 km: mean normalised envelope at the predicted time over all family-channels (boreholes, surface,
and two family halves separately).  Significance: the same statistic with the moveout replaced by a random
permutation of the channels' cos(i) (500 draws) and with a constant (no-moveout) lead; a real conversion must
beat both, appear in both halves and in boreholes and surface alike.
Outputs data/sci/dvv_source_region/sp_scan.csv, figures/sci_dvv_source_region_spscan.png
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.signal import hilbert

ROOT = '/home/seismic/LFE'
sys.path.insert(0, f'{ROOT}/scripts')
OUT = f'{ROOT}/data/sci/dvv_source_region'
H = np.arange(0.25, 10.01, 0.25)


def main():
    G = pd.read_csv(f'{OUT}/geometry_3d.csv')
    recs = []
    for F in range(1, 28):
        picks = pd.read_csv(f'{ROOT}/data/locations/fam{F}_3d_picks.csv').set_index('sta')
        g = G[G.F == F].set_index('sta')
        with h5py.File(f'{ROOT}/data/templates/fam{F}_templates.h5', 'r') as T:
            sr, pre = float(T.attrs['sr']), float(T.attrs['pre'])
            for cid in T:
                if cid[-1] != 'Z' or T[cid].attrs['snr'] < 20:
                    continue
                sta = '.'.join(cid.split('.')[:2])
                if sta not in picks.index or sta not in g.index:
                    continue
                pk = picks.loc[sta]
                S = pk.S_final if np.isfinite(pk.S_final) else (pk.S if pk.sig_S <= 0.1 else np.nan)
                if not np.isfinite(S):
                    continue
                x = T[cid]['stack'][:].astype(float)
                t = np.arange(len(x)) / sr + pre
                env = np.abs(hilbert(x))
                # refine S on the envelope (max within +-0.5 s)
                m = np.abs(t - S) <= 0.5
                S2 = t[m][np.argmax(env[m])]
                P = pk.P if np.isfinite(pk.P) and pk.sig_P <= 0.25 else np.nan
                if np.isfinite(P):
                    mp = np.abs(t - P) <= 0.4
                    P2 = t[mp][np.argmax(env[mp])]
                else:
                    P2 = np.nan
                noise = np.median(env[(t > -15) & (t < -5)])
                ks, kp = g.loc[sta, 'KS_sph5'] / 5, g.loc[sta, 'KP_sph5'] / 5
                recs.append(dict(F=F, chan=cid, sta=sta, bh=sta.startswith('PB.'), S=S2, P=P2, t=t, env=env,
                                 noise=noise, envS=env[np.argmin(np.abs(t - S2))],
                                 envP=env[np.argmin(np.abs(t - P2))] if np.isfinite(P2) else np.nan,
                                 cos=max(g.loc[sta, 'cos_S'], 0.15), dsl=ks - kp))
    print(len(recs), 'family-channels', flush=True)
    rng = np.random.default_rng(1)

    def stat(sel, phase, cosv, moveout=True):
        out = np.zeros(len(H))
        n = 0
        for r, c in zip(sel, cosv):
            if phase == 'Sp':
                ref, norm = r['S'], r['envS']
                sgn = -1
            else:
                if not np.isfinite(r['P']):
                    continue
                ref, norm = r['P'], r['envP']
                sgn = +1
            lead = H * r['dsl'] / (c if moveout else 0.3)
            tt = ref + sgn * lead
            v = np.interp(tt, r['t'], r['env']) / norm
            # stay away from the direct phase (+-0.25 s) and before P for Sp
            bad = (lead < 0.25) | ((phase == 'Sp') & np.isfinite(r['P']) & (tt < r['P'] + 0.5)) | \
                  ((phase == 'Ps') & (tt > r['S'] - 0.5))
            v[bad] = np.nan
            out = out + np.nan_to_num(v)
            n += 1
        return out / max(n, 1)
    rows = []
    for phase in ('Sp', 'Ps'):
        for name, sel in (('all', recs), ('boreholes', [r for r in recs if r['bh']]),
                          ('surface', [r for r in recs if not r['bh']]),
                          ('fam_odd', [r for r in recs if r['F'] % 2]), ('fam_even', [r for r in recs if not r['F'] % 2])):
            cosv = np.array([r['cos'] for r in sel])
            real = stat(sel, phase, cosv)
            flat = stat(sel, phase, cosv, moveout=False)
            null = np.array([stat(sel, phase, rng.permutation(cosv)) for _ in range(200)])
            mu, sd = null.mean(0), null.std(0) + 1e-12
            z = (real - mu) / sd
            for i, h in enumerate(H):
                rows.append(dict(phase=phase, subset=name, h_km=h, amp=real[i], amp_flat=flat[i], null_mean=mu[i],
                                 null_sd=sd[i], z=z[i], n=len(sel)))
            k = int(np.nanargmax(z))
            print(phase, name, len(sel), f'max z {z[k]:.2f} at h={H[k]} km (amp {real[k]:.3f}, null {mu[k]:.3f})', flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(f'{OUT}/sp_scan.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for a_, ph in zip(ax, ('Sp', 'Ps')):
        for nm in ('all', 'boreholes', 'surface', 'fam_odd', 'fam_even'):
            g = D[(D.phase == ph) & (D.subset == nm)]
            a_.plot(g.h_km, g.z, label=nm)
        a_.axhline(3, color='k', ls=':')
        a_.axhline(-3, color='k', ls=':')
        a_.set_xlabel('interface height above LFEs h (km)')
        a_.set_title(f'{ph} conversion scan (z vs permuted moveout)')
    ax[0].set_ylabel('z')
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{ROOT}/figures/sci_dvv_source_region_spscan.png', dpi=130)


if __name__ == '__main__':
    main()
