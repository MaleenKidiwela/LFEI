#!/usr/bin/env python3
"""Follow-up checks on burst cascades (see burst_cascades.py):
 a) leader statistics normalised by participation (is the array END more often first than its share?)
 b) front vs diffusion: in significant episodes, onset lag vs distance from the leader: fit lag = a*r and lag = b*r^2
 c) depth-level lead/lag: mean onset of shallow (<28 km) vs deep (>=28 km) families per episode
 d) time-shuffle null: shift each family's burst onsets by a random +-30 d offset (preserves rates), recompute
    fraction of significant episodes
Writes data/cascades/checks.json"""
import json
import numpy as np
import pandas as pd
import burst_cascades as bc

rng = np.random.default_rng(1)
d = pd.read_csv('data/shelly2026_MTJ_v2/LFEcatV2_MTJ_2008-2026.csv')
d['t'] = (pd.to_datetime(dict(year=d['%year'], month=d.month, day=d.day)) - pd.Timestamp('1970-01-01')).dt.total_seconds() + d.s_of_day
lat0, lon0 = 40.06, -123.625
J = {F: json.load(open(f'data/locations/fam{F}_3d.json')) for F in range(1, 28)}
pos = {F: ((j['lon'] - lon0) * 111.19 * np.cos(np.radians(lat0)), (j['lat'] - lat0) * 111.19) for F, j in J.items()}
depth = {F: j['depth_km'] for F, j in J.items()}


def onsets(shift=False):
    on = []
    for F, g in d.groupby('idnum'):
        t = np.sort(g.t.values)
        s = np.r_[0, np.where(np.diff(t) > bc.GAP)[0] + 1]
        e = np.r_[s[1:], len(t)]
        off = rng.uniform(-30, 30) * 86400 if shift else 0.0
        on += [(F, t[a] + off, b - a) for a, b in zip(s, e)]
    on = pd.DataFrame(on, columns=['fam', 'onset', 'n']).sort_values('onset').reset_index(drop=True)
    on['ep'] = np.r_[0, np.cumsum(np.diff(on.onset.values) > bc.EP_GAP)]
    return on


def episodes(on, nperm=100):
    out = []
    for e, g in on.groupby('ep'):
        g = g.sort_values('onset').drop_duplicates('fam')
        if len(g) < bc.MINFAM:
            continue
        X = np.array([pos[F] for F in g.fam])
        t = (g.onset.values - g.onset.min()) / 3600.0
        m, rss, tss = bc.fit(t, X)
        r2 = 1 - rss / tss if tss > 0 else np.nan
        null = [(lambda q: 1 - q[1] / q[2] if q[2] > 0 else np.nan)(bc.fit(t, X[rng.permutation(len(X))], iters=2)) for _ in range(nperm)]
        p = (np.sum(np.array(null) >= r2) + 1) / (nperm + 1)
        out.append(dict(ep=e, fams=list(g.fam), t=list(t), p=p))
    return out


res = {}
on = onsets()
E = episodes(on)
sig = [e for e in E if e['p'] < 0.05]
res['n_episodes'], res['n_sig'] = len(E), len(sig)
# a) leader share vs participation share
part = pd.Series([F for e in sig for F in e['fams']]).value_counts()
lead = pd.Series([e['fams'][0] for e in sig]).value_counts()
exp = part / part.sum() * len(sig) * (len(sig) / sum(len(e['fams']) for e in sig)) * (sum(len(e['fams']) for e in sig) / len(sig))
# expected leads if leader random among participants of each episode:
expl = pd.Series(0.0, index=part.index)
for e in sig:
    for F in e['fams']:
        expl[F] += 1 / len(e['fams'])
ratio = (lead.reindex(expl.index).fillna(0) / expl).sort_values(ascending=False)
res['leader_ratio'] = {int(F): round(float(v), 2) for F, v in ratio.items()}
res['leader_ratio_vs_E'] = float(np.corrcoef([pos[F][0] for F in ratio.index], ratio.values)[0, 1])
res['leader_ratio_vs_absE'] = float(np.corrcoef([abs(pos[F][0]) for F in ratio.index], ratio.values)[0, 1])
# b) lag vs distance from leader
r_all, l_all = [], []
for e in sig:
    x0 = np.array(pos[e['fams'][0]])
    for F, tt in zip(e['fams'][1:], e['t'][1:]):
        r_all.append(np.hypot(*(np.array(pos[F]) - x0)))
        l_all.append(tt)
r_all, l_all = np.array(r_all), np.array(l_all)
ok = l_all < 6
a = np.sum(r_all[ok] * l_all[ok]) / np.sum(r_all[ok] ** 2)
b = np.sum(r_all[ok] ** 2 * l_all[ok]) / np.sum(r_all[ok] ** 4)
res['front_rss'] = float(np.sum((l_all[ok] - a * r_all[ok]) ** 2))
res['diff_rss'] = float(np.sum((l_all[ok] - b * r_all[ok] ** 2) ** 2))
res['front_speed_km_per_day'] = float(24 / a)
res['diffusivity_km2_per_day'] = float(24 / b / 4)        # lag = r^2/(4D)
# binned median lag vs distance
bins = np.arange(0, 18, 2)
res['lag_vs_r_median_h'] = {f'{lo}-{lo + 2}': float(np.median(l_all[(r_all >= lo) & (r_all < lo + 2)])) for lo in bins
                            if ((r_all >= lo) & (r_all < lo + 2)).sum() > 30}
# c) depth-level lead/lag
dl = []
for e in sig:
    sh = [tt for F, tt in zip(e['fams'], e['t']) if depth[F] < 28]
    dp = [tt for F, tt in zip(e['fams'], e['t']) if depth[F] >= 28]
    if sh and dp:
        dl.append(np.mean(dp) - np.mean(sh))
dl = np.array(dl)
res['deep_minus_shallow_onset_h_median'] = float(np.median(dl))
res['deep_minus_shallow_frac_positive'] = float(np.mean(dl > 0))
res['deep_minus_shallow_n'] = int(len(dl))
# d) time-shuffle null (3 realisations)
fr = []
for k in range(3):
    Es = episodes(onsets(shift=True), nperm=50)
    fr.append(np.mean([e['p'] < 0.05 for e in Es]) if Es else np.nan)
    print(f'shuffle {k}: {len(Es)} episodes, frac significant {fr[-1]:.2f}', flush=True)
res['shuffle_frac_sig'] = [float(x) for x in fr]
res['real_frac_sig'] = len(sig) / len(E)
json.dump(res, open('data/cascades/checks.json', 'w'), indent=1)
for k, v in res.items():
    print(k, v)
