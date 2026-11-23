#!/usr/bin/env python3
"""Before/after the post-M7 silence (Fable ideas 1, 3, 5) for one family.

Groups (v2 events):  pre = last N before the mainshock; restart = first N after it;
                     late = N events starting LATE_START (later baseline).
Per channel, from data/waveforms/fam{F}.h5 (1-15 Hz, raw counts, not normalised):
 (1) healing: per-event noise-corrected S-window RMS amplitude -> log10 relative to the
     station's pre-group median -> per-event relative magnitude = median over stations
     (>= MINSTA). Group stacks -> spectral ratio restart/pre and late/pre (S window).
 (3) Vp/Vs: P and S time shifts of restart/late stacks relative to pre stack (CC with
     parabolic interpolation, 2-8 Hz) -> d(tS - tP) per station. Bootstrap over events.
 (5) rake: P/S (vertical P RMS / horizontal S RMS) and SH/SV (transverse/radial S RMS,
     horizontals rotated with channel azimuths and source->station back-azimuth), per group.
All uncertainties: bootstrap (resample events within each group, NBOOT times).
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
from scipy.signal import butter, sosfiltfilt

MAIN = pd.Timestamp('2024-12-05T18:44')
N, NBOOT, MINSTA = 500, 200, 4
LATE_START = pd.Timestamp('2025-03-01')
S_WIN, P_WIN, NOISE = (-0.5, 2.5), (-0.3, 1.2), (-15.0, -5.0)


def win(t, a, b):
    return (t >= a) & (t < b)


def lag_cc(ref, x, t, w, sr, maxlag=0.15):
    i = np.where(w)[0]
    ml = int(maxlag * sr)
    best, bl = -2, 0
    ccs = []
    r = ref[i] - ref[i].mean()
    for L in range(-ml, ml + 1):
        seg = x[i + L] - x[i + L].mean()
        ccs.append(float(r @ seg / (np.linalg.norm(r) * np.linalg.norm(seg) + 1e-20)))
    ccs = np.array(ccs)
    k = int(np.argmax(ccs))
    frac = 0.0
    if 0 < k < len(ccs) - 1:
        y0, y1, y2 = ccs[k - 1:k + 2]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if den else 0.0
    return (k - ml + frac) / sr, ccs[k]


def load_azimuths():
    az = {}
    for f in ['data/stations/cha_ncedc_2008_2026.txt', 'data/stations/cha_iris_PB_2008_2026.txt']:
        for l in open(f):
            if l.startswith('#') or '|' not in l:
                continue
            p = [x.strip() for x in l.split('|')]
            az[(p[0], p[1], p[3])] = float(p[8] or 0)      # latest epoch wins
    return az


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', type=int, default=23)
    ap.add_argument('--model', default='3d')
    ap.add_argument('--main', default=str(MAIN), help='mainshock time (default 2024 M7)')
    ap.add_argument('--late-start', default=str(LATE_START))
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    main_t, late_t = pd.Timestamp(a.main), pd.Timestamp(a.late_start)
    F = a.family
    h = h5py.File(f'data/waveforms/fam{F}.h5', 'r')
    sr, pre = h.attrs['sr'], h.attrs['pre']
    t_ev = pd.to_datetime(h['events/time_utc'][:].astype(str))
    npts = h[f'chan/{next(iter(h["chan"]))}/data'].shape[1]
    t = np.arange(npts) / sr + pre
    pk = pd.read_csv(f'data/locations/fam{F}_{a.model}_picks.csv').set_index('sta')
    loc = json.load(open(f'data/locations/fam{F}_{a.model}.json'))
    order = np.argsort(t_ev.values)
    before = order[t_ev.values[order] < main_t.to_datetime64()]
    after = order[t_ev.values[order] >= main_t.to_datetime64()]
    late = order[t_ev.values[order] >= late_t.to_datetime64()]
    groups = {'pre': before[-N:], 'restart': after[:N], 'late': late[:N]}
    for k, g in groups.items():
        print(f'{k:8s} n={len(g)}  {t_ev[g].min()} .. {t_ev[g].max()}', flush=True)
    sos28 = butter(4, [2, 8], btype='band', fs=sr, output='sos')
    az = load_azimuths()
    stas = pd.read_csv('data/stations/candidate_channels_65km_2008_2026.csv').groupby(['net', 'sta'])[['lat', 'lon']].first()
    rng = np.random.default_rng(0)

    amp_rows, shift_rows, ratio_rows, spec = [], [], [], {}
    by_sta = {}
    for cid in h['chan']:
        n, s, l, c = cid.split('.')
        by_sta.setdefault(f'{n}.{s}', []).append(cid)
    for sta, cids in sorted(by_sta.items()):
        if sta not in pk.index:
            continue
        r = pk.loc[sta]
        S = r['S_final'] if np.isfinite(r.get('S_final', np.nan)) else r['S']
        P = r.get('P', np.nan)
        if not np.isfinite(S):
            continue
        sw, nw = win(t, S + S_WIN[0], S + S_WIN[1]), win(t, *NOISE)
        pw = win(t, P + P_WIN[0], P + P_WIN[1]) if np.isfinite(P) else None
        data = {}
        for cid in cids:
            ev = h[f'chan/{cid}/ev'][:]
            pos = {e: i for i, e in enumerate(ev)}
            sel = {k: [pos[e] for e in g if e in pos] for k, g in groups.items()}
            if min(len(v) for v in sel.values()) < 50:
                continue
            X = h[f'chan/{cid}/data'][:].astype('float64')
            data[cid] = (ev, pos, sel, X)
        if not data:
            continue
        # (1) per-event amplitudes on horizontals (else vertical)
        use = [c for c in data if c[-1] in 'EN12'] or list(data)
        for cid in use:
            ev, pos, sel, X = data[cid]
            Xf = sosfiltfilt(sos28, X, axis=1)
            # matched-filter amplitude: projection on the pre-group stack (unbiased by noise)
            m = Xf[sel['pre']].mean(0)[sw]
            A = Xf[:, sw] @ m / (m @ m)
            ref = np.median(A[sel['pre']])
            for k, idx in sel.items():
                for i in idx:
                    amp_rows.append(dict(sta=sta, chan=cid, group=k, ev=int(ev[i]), amp=A[i] / ref))
            # spectra of group stacks (S window, 1-15 Hz data)
            for k, idx in sel.items():
                stck = X[idx].mean(0)
                seg = stck[sw] * np.hanning(sw.sum())
                spec.setdefault(k, []).append(np.abs(np.fft.rfft(seg, 256)))
        # (3) P and S shifts restart/late vs pre (2-8 Hz stacks), bootstrap
        for cid, (ev, pos, sel, X) in data.items():
            Xf = sosfiltfilt(sos28, X, axis=1)
            for ph, w in [('S', sw), ('P', pw)]:
                if w is None or (ph == 'S' and cid[-1] == 'Z' and any(c[-1] in 'EN12' for c in data)) \
                        or (ph == 'P' and cid[-1] != 'Z'):
                    continue
                refstack = Xf[sel['pre']].mean(0)
                for k in ['restart', 'late']:
                    dt0, cc0 = lag_cc(refstack, Xf[sel[k]].mean(0), t, w, sr)
                    bs = []
                    for _ in range(NBOOT // 4):
                        i1 = rng.choice(sel['pre'], len(sel['pre']))
                        i2 = rng.choice(sel[k], len(sel[k]))
                        bs.append(lag_cc(Xf[i1].mean(0), Xf[i2].mean(0), t, w, sr)[0])
                    shift_rows.append(dict(sta=sta, chan=cid, phase=ph, group=k, dt=dt0, cc=cc0, sd=np.std(bs)))
        # (5) amplitude ratios per group: P/S and SH/SV
        hz = [c for c in data if c[-1] in 'EN12']
        vz = [c for c in data if c[-1] == 'Z']
        if len(hz) == 2 and vz and (sta.split('.')[0], sta.split('.')[1]) in stas.index:
            slat, slon = stas.loc[(sta.split('.')[0], sta.split('.')[1])]
            baz = np.degrees(np.arctan2((loc['lon'] - slon) * np.cos(np.radians(slat)), loc['lat'] - slat)) % 360
            c1, c2 = sorted(hz)
            a1, a2 = az.get((sta.split('.')[0], sta.split('.')[1], c1.split('.')[3]), np.nan), \
                az.get((sta.split('.')[0], sta.split('.')[1], c2.split('.')[3]), np.nan)
            if np.isfinite(a1) and np.isfinite(a2):
                common = {k: sorted(set(data[c1][2][k]) & set(data[c2][2][k])) for k in groups}
                for k in groups:
                    i1 = [data[c1][1][data[c1][0][i]] for i in data[c1][2][k]]
                    ev1 = data[c1][0][data[c1][2][k]]
                    ev2 = set(data[c2][0][data[c2][2][k]])
                    evz = set(data[vz[0]][0][data[vz[0]][2][k]])
                    evc = [e for e in ev1 if e in ev2 and e in evz]
                    if len(evc) < 50:
                        continue
                    vals = []
                    for b in range(NBOOT // 4 + 1):
                        es = evc if b == 0 else list(rng.choice(evc, len(evc)))
                        h1 = sosfiltfilt(sos28, data[c1][3][[data[c1][1][e] for e in es]].mean(0))
                        h2 = sosfiltfilt(sos28, data[c2][3][[data[c2][1][e] for e in es]].mean(0))
                        zz = sosfiltfilt(sos28, data[vz[0]][3][[data[vz[0]][1][e] for e in es]].mean(0))
                        # rotate to N,E then to radial (source->station) / transverse
                        N_ = h1 * np.cos(np.radians(a1)) + h2 * np.cos(np.radians(a2))
                        E_ = h1 * np.sin(np.radians(a1)) + h2 * np.sin(np.radians(a2))
                        ba = np.radians(baz)
                        R = -N_ * np.cos(ba) - E_ * np.sin(ba)
                        T_ = N_ * np.sin(ba) - E_ * np.cos(ba)
                        rms = lambda x, w: np.sqrt(np.mean(x[w] ** 2))
                        ps = rms(zz, pw) / np.hypot(rms(R, sw), rms(T_, sw)) if pw is not None else np.nan
                        vals.append((rms(T_, sw) / rms(R, sw), ps))
                    v = np.array(vals)
                    ratio_rows.append(dict(sta=sta, group=k, sh_sv=v[0, 0], sh_sv_sd=v[1:, 0].std(),
                                           p_s=v[0, 1], p_s_sd=v[1:, 1].std(), n=len(evc)))
        print(f'{sta:10s} done', flush=True)

    os.makedirs('data/silent', exist_ok=True)
    amp = pd.DataFrame(amp_rows)
    ev_mag = amp.groupby(['group', 'ev']).agg(rel_amp=('amp', 'median'), nsta=('sta', 'nunique')).reset_index()
    ev_mag['rel_mag'] = np.log10(ev_mag.rel_amp.clip(lower=1e-3))   # for plotting only
    ev_mag = ev_mag[ev_mag.nsta >= MINSTA]
    ev_mag['time_utc'] = t_ev[ev_mag.ev].astype(str)
    ev_mag.to_csv(f'data/silent/fam{F}{a.tag}_relative_magnitudes.csv', index=False)
    sh = pd.DataFrame(shift_rows)
    sh.to_csv(f'data/silent/fam{F}{a.tag}_phase_shifts.csv', index=False)
    rt = pd.DataFrame(ratio_rows)
    rt.to_csv(f'data/silent/fam{F}{a.tag}_amplitude_ratios.csv', index=False)

    # summaries
    print('\n(1) relative amplitude (matched-filter amplitude / pre median; median over stations):')
    for k in groups:
        x = ev_mag[ev_mag.group == k].rel_amp
        bs = [rng.choice(x, len(x)).mean() for _ in range(NBOOT)]
        print(f'   {k:8s} n={len(x):4d}  mean {x.mean():.3f} ± {np.std(bs):.3f}  median {x.median():.3f}')
    r0 = ev_mag[ev_mag.group == 'restart'].sort_values('time_utc')
    print('   first 20 restart events rel_amp:', np.round(r0.rel_amp.values[:20], 2))
    if len(sh):
        good = sh[(sh.cc >= 0.6) & (sh.sd <= 0.02)]          # quality cut (cycle-skip / noise)
        print(f'\n(3) phase shifts passing CC>=0.6 & sd<=20 ms: {len(good)} of {len(sh)} '
              f'(S {int((good.phase=="S").sum())}, P {int((good.phase=="P").sum())})')
        S_ = good[good.phase == 'S'].groupby(['sta', 'group']).dt.median()
        P_ = good[good.phase == 'P'].groupby(['sta', 'group']).dt.median()
        both = pd.concat([S_.rename('dtS'), P_.rename('dtP')], axis=1).dropna()
        both['d_SminusP_ms'] = (both.dtS - both.dtP) * 1000
        print('\n(3) d(tS - tP) relative to pre (ms):')
        print(both.round(4).to_string())
        both.to_csv(f'data/silent/fam{F}{a.tag}_dSP.csv')
    if len(rt):
        print('\n(5) amplitude ratios:')
        print(rt.round(3).to_string(index=False))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for k, c in zip(groups, ['k', 'r', 'C0']):
        x = ev_mag[ev_mag.group == k].rel_amp
        ax[0].hist(x.clip(-2, 6), bins=60, histtype='step', color=c, density=True, label=f'{k} (n={len(x)})')
    ax[0].set_xlabel('amplitude rel. pre median (matched filter)')
    ax[0].legend(fontsize=8)
    ax[0].set_title('(1) relative amplitudes')
    f = np.fft.rfftfreq(256, 1 / sr)
    pre_sp = np.median(np.array(spec['pre']), 0)
    for k, c in [('restart', 'r'), ('late', 'C0')]:
        ax[1].semilogy(f, np.median(np.array(spec[k]), 0) / pre_sp, c, label=f'{k}/pre')
    ax[1].set_xlim(1, 15)
    ax[1].set_xlabel('Hz')
    ax[1].legend(fontsize=8)
    ax[1].set_title('(1) S-window spectral ratio (median over channels)')
    ax[2].plot(pd.to_datetime(r0.time_utc, format='mixed'), r0.rel_amp, '.', ms=3, color='r')
    ax[2].axhline(1, color='k', lw=0.5)
    ax[2].set_title('(1) restart events: rel. amplitude vs time')
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(f'figures/fam{F}_silent_period{a.tag}.png', dpi=130)


if __name__ == '__main__':
    main()
