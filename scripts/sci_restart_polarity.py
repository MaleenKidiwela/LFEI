#!/usr/bin/env python
"""Polarity of the first LFEs after each quake-induced silence vs the events just before it.

Uses data/sci/remote_check/perchan/fam{F}.npz: 'ccs' = signed template CC at the |cc| extremum within
+-3 samples of zero lag (S window S-0.5..S+3 s, 2-8 Hz), i.e. a per-channel polarity measure for every
catalog event (template = long-term family stack). n0_ccs = same statistic in a pre-P noise window.

Per family and quake: PRE = last N_PRE events before the quake, POST = first N_POST events after it.
Per channel: mean ccs (PRE, POST), t-test of the difference. A polarity flip = channel whose PRE mean is
clearly positive (>= 0.10) and POST mean is negative with t <= -2.
Network statistics: (1) number of flipped channels, (2) median over channels of the POST/PRE mean ccs ratio,
(3) across-channel correlation of PRE and POST mean-ccs patterns (radiation pattern similarity).
Null: 300 placebo reference times (2009-2024, >=90 d from the three quakes) with the same N_PRE/N_POST.
Outputs data/sci/restart_polarity/{per_channel.csv, summary.csv}, figures/sci_restart_polarity.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'data/sci/restart_polarity'
os.makedirs(OUT, exist_ok=True)
N_PRE, N_POST = 50, 20
DELAY = float(os.environ.get('POST_DELAY_D', 0))   # POST starts this many days after the reference time
OUT = OUT + (f'_delay{DELAY:g}d' if DELAY else '')
os.makedirs(OUT, exist_ok=True)
QUAKES = {'Petrolia M6.1': '2021-12-20 20:10', 'Ferndale M6.4': '2022-12-20 10:34', 'M7.0': '2024-12-05 18:44'}
rng = np.random.default_rng(7)


def load(F):
    z = np.load(f'data/sci/remote_check/perchan/fam{F}.npz')
    t = pd.to_datetime(z['time_utc']).values
    o = np.argsort(t)
    return z['chans'], t[o], z['ccs'][o], z['n0_ccs'][o]


def stats(t, C, ref, keep_rows=False):
    ref = np.datetime64(pd.Timestamp(ref))
    i = np.searchsorted(t, ref)
    j = np.searchsorted(t, ref + np.timedelta64(int(DELAY * 86400), 's'))
    if i < N_PRE or j + N_POST > len(t):
        return None
    A, B = C[i - N_PRE:i], C[j:j + N_POST]
    na, nb = np.sum(np.isfinite(A), 0), np.sum(np.isfinite(B), 0)
    ma, mb = np.nanmean(A, 0), np.nanmean(B, 0)
    sa, sb = np.nanstd(A, 0, ddof=1), np.nanstd(B, 0, ddof=1)
    tt = (mb - ma) / np.sqrt(sa ** 2 / np.maximum(na, 1) + sb ** 2 / np.maximum(nb, 1))
    ok = (na >= 15) & (nb >= 8) & (ma >= 0.10)
    flip = ok & (mb < 0) & (tt <= -2)
    r = np.corrcoef(ma[ok], mb[ok])[0, 1] if ok.sum() >= 5 else np.nan
    return dict(n_ok=int(ok.sum()), n_flip=int(flip.sum()), ratio=float(np.nanmedian(mb[ok] / ma[ok])) if ok.any() else np.nan,
                pattern_r=r, frac_pos_post=float(np.mean(mb[ok] > 0)) if ok.any() else np.nan,
                gap_d=float((t[j] - ref) / np.timedelta64(1, 'D'))), (ma, mb, tt, ok, flip)


excl = [pd.Timestamp(v) for v in QUAKES.values()]
cands = pd.date_range('2009-06-01', '2024-09-01', freq='6h')
cands = cands[[min(abs((c - e).days) for e in excl) >= 90 for c in cands]]
PLAC = rng.choice(cands, 300, replace=False)

rows, prow, plac = [], [], []
for F in range(1, 28):
    if not os.path.exists(f'data/sci/remote_check/perchan/fam{F}.npz'):
        continue
    ch, t, C, N0 = load(F)
    for q, ref in QUAKES.items():
        res = stats(t, C, ref)
        if res is None:
            continue
        s, (ma, mb, tt, ok, flip) = res
        rows.append(dict(family=F, quake=q, **s))
        sn = stats(t, N0, ref)
        for c in np.where(ok)[0]:
            prow.append(dict(family=F, quake=q, chan=ch[c], pre=ma[c], post=mb[c], t=tt[c], flip=bool(flip[c])))
    for k, ref in enumerate(PLAC):
        res = stats(t, C, ref)
        if res is not None:
            plac.append(dict(family=F, k=k, **res[0]))
R, P, PL = pd.DataFrame(rows), pd.DataFrame(prow), pd.DataFrame(plac)
P.to_csv(f'{OUT}/per_channel.csv', index=False)
R.to_csv(f'{OUT}/per_family.csv', index=False)
PL.to_csv(f'{OUT}/placebo_per_family.csv', index=False)

# network aggregates (sum of flips, median ratio, median pattern r over families) vs placebo distribution
def agg(D):
    return pd.Series(dict(n_fam=len(D), n_ok=D.n_ok.sum(), n_flip=D.n_flip.sum(), flip_frac=D.n_flip.sum() / max(D.n_ok.sum(), 1),
                          ratio=D.ratio.median(), pattern_r=D.pattern_r.median(), frac_pos=D.frac_pos_post.median()))
PA = PL.groupby('k').apply(agg)
S = []
for q in QUAKES:
    a = agg(R[R.quake == q])
    d = dict(quake=q, **a.to_dict())
    for m in ('flip_frac', 'ratio', 'pattern_r', 'frac_pos'):
        v = PA[m].dropna()
        d[f'{m}_plac_med'] = v.median()
        d[f'{m}_plac_p05'] = v.quantile(0.05)
        d[f'{m}_plac_p95'] = v.quantile(0.95)
        d[f'{m}_pct_rank'] = float((v < a[m]).mean() * 100)
    S.append(d)
S = pd.DataFrame(S)
S.to_csv(f'{OUT}/summary.csv', index=False)
print(S.T.to_string())
print('placebo per-family flip fraction mean', (PL.n_flip.sum() / PL.n_ok.sum()))

fig, axs = plt.subplots(1, 3, figsize=(14, 4.3))
for ax, q in zip(axs, QUAKES):
    g = P[P.quake == q]
    ax.scatter(g.pre, g.post, s=6, c=np.where(g.flip, 'r', 'k'), alpha=0.5)
    ax.axhline(0, color='0.5', lw=0.6); ax.plot([0, 1], [0, 1], 'b--', lw=0.7)
    s = S[S.quake == q].iloc[0]
    ax.set_title(f'{q}: {int(s.n_flip)}/{int(s.n_ok)} channels flipped\n'
                 f'pattern r {s.pattern_r:.2f} (placebo 5-95% {s.pattern_r_plac_p05:.2f}-{s.pattern_r_plac_p95:.2f}), '
                 f'post/pre {s.ratio:.2f}', fontsize=8.5)
    ax.set_xlabel(f'mean signed CC, last {N_PRE} events before'); ax.set_ylabel(f'first {N_POST} events after')
    ax.set_xlim(0, 0.8); ax.set_ylim(-0.4, 0.8); ax.grid(alpha=0.3)
fig.suptitle('S-wave polarity per channel (signed template CC, 2-8 Hz), all 27 families; red = significant sign flip', fontsize=10)
fig.tight_layout()
fig.savefig('figures/sci_restart_polarity' + (f'_delay{DELAY:g}d' if DELAY else '') + '.png', dpi=140)
