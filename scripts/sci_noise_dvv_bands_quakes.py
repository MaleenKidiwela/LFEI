#!/usr/bin/env python3
"""Quick single-station ambient-noise dv/v in three frequency bands (0.1-1, 1-3, 3-5 Hz) around Petrolia (2021-12-20),
Ferndale (2022-12-20) and the M7 (2024-12-05), from the continuous days already on disk (data/continuous/<day>/<NET.STA>.mseed,
downloaded for the matched-filter scans: about -40..+50 d around each quake).

Per station-day: merge (gaps zero-filled), decimate to 20 Hz, components Z/N/E (1/2 = N/E for PB boreholes; no rotation,
single-station pairs only), per band: bandpass -> one-bit -> segment cross-correlation (whole day), 6 pairs ZZ NN EE EN EZ NZ.
Daily CCFs are stretched against a pre-quake reference (mean of pre-quake days) on both coda sides:
  0.1-1 Hz |lag| 8-50 s (300-s segments), 1-3 Hz 3-25 s (120 s), 3-5 Hz 2-15 s (120 s).
Pair dv/v combined per day with CC^2/(1-CC^2) weights. Step = mean(post days 1..15) - mean(pre days -30..-1);
'silence' = M7 +1..+25 d. Error: moving-block bootstrap (5-d blocks) of the daily series, and a placebo spread from
all 15-d vs 30-d splits inside the pre-quake window (limited: ~30-40 pre days).
No response removal: 0.1-1 Hz at short-period (1-2 Hz) sensors is weak; check CC.
Outputs data/sci/noise_dvv_bands_quakes/{ccf_<sta>_<quake>.npz, daily.csv, steps.csv}, figures/sci_noise_dvv_bands_quakes.png
Usage: python scripts/sci_noise_dvv_bands_quakes.py correlate|dvv|plot [--procs 12]
"""
import argparse
import glob
import os
from multiprocessing import get_context

import numpy as np
import pandas as pd

OUT = 'data/sci/noise_dvv_bands_quakes'
FS = 20.0
QUAKES = {'petrolia': '2021-12-20', 'ferndale': '2022-12-20', 'm7': '2024-12-05'}
BANDS = {'0.1-1Hz': (0.1, 1.0, 300.0, 60.0, (8.0, 50.0)),       # f1, f2, segment s, maxlag s, coda |lag| window
         '1-3Hz': (1.0, 3.0, 120.0, 30.0, (3.0, 25.0)),
         '3-5Hz': (3.0, 5.0, 120.0, 30.0, (2.0, 15.0))}
PAIRS = [('Z', 'Z'), ('N', 'N'), ('E', 'E'), ('E', 'N'), ('E', 'Z'), ('N', 'Z')]
COMP = {'Z': 'Z', 'N': 'N', '1': 'N', 'E': 'E', '2': 'E'}
EPS = np.arange(-0.01, 0.01 + 1e-12, 0.0002)


def window_days(q):
    t = pd.Timestamp(QUAKES[q])
    days = sorted(d for d in os.listdir('data/continuous') if -35 <= (pd.Timestamp(d) - t).days <= 30)
    return days


def load_day(path):
    from obspy import read
    st = read(path)
    st.merge(fill_value=0)
    out = {}
    for tr in st:
        c = COMP.get(tr.stats.channel[-1])
        if c is None or c in out or tr.stats.channel[:2] not in ('HH', 'EH', 'BH', 'HN') or tr.stats.channel[:2] == 'HN':
            continue
        tr.detrend('demean')
        if abs(tr.stats.sampling_rate - 100.0) < 1e-3:
            tr.filter('lowpass', freq=8.0, corners=4, zerophase=True)
            tr.data = tr.data[::5]
            tr.stats.sampling_rate = FS
        elif abs(tr.stats.sampling_rate - 40.0) < 1e-3:
            tr.filter('lowpass', freq=8.0, corners=4, zerophase=True)
            tr.data = tr.data[::2]
            tr.stats.sampling_rate = FS
        else:
            tr.resample(FS)
        day0 = tr.stats.starttime.__class__(tr.stats.starttime.date)
        i0 = int(round((tr.stats.starttime - day0) * FS))
        x = np.zeros(int(86400 * FS), dtype=np.float32)
        n = min(len(tr.data), len(x) - i0)
        if i0 < 0 or n <= 0:
            continue
        x[i0:i0 + n] = tr.data[:n]
        out[c] = x
    return out


def correlate_day(arrs):
    from scipy.signal import butter, sosfiltfilt
    res = {}
    for b, (f1, f2, seg, maxlag, _) in BANDS.items():
        sos = butter(4, [f1, f2], btype='band', fs=FS, output='sos')
        ns, nl = int(seg * FS), int(maxlag * FS)
        nfft = int(2 ** np.ceil(np.log2(2 * ns)))
        spec = {}
        for c, x in arrs.items():
            y = sosfiltfilt(sos, x.astype(float))
            nseg = len(y) // ns
            Y = y[:nseg * ns].reshape(nseg, ns)
            live = (np.abs(x[:nseg * ns].reshape(nseg, ns)) > 0).mean(1) > 0.95
            Y = np.sign(Y)
            spec[c] = (np.fft.rfft(Y, nfft, axis=1), live)
        cc = np.full((len(PAIRS), 2 * nl + 1), np.nan, dtype=np.float32)
        for k, (a, bb) in enumerate(PAIRS):
            if a not in spec or bb not in spec:
                continue
            (A, la), (B, lb) = spec[a], spec[bb]
            ok = la & lb
            if ok.sum() < 0.5 * len(ok):
                continue
            c = np.fft.irfft((A[ok] * np.conj(B[ok])).mean(0), nfft)
            c = np.concatenate([c[-nl:], c[:nl + 1]])
            cc[k] = c / (np.abs(c).max() + 1e-12)
        res[b] = cc
    return res


def _job(args):
    sta, q = args
    out = f'{OUT}/ccf_{sta}_{q}.npz'
    if os.path.exists(out):
        return sta, q, 'cached'
    days, store = [], {b: [] for b in BANDS}
    for d in window_days(q):
        p = f'data/continuous/{d}/{sta}.mseed'
        if not os.path.exists(p):
            continue
        try:
            arrs = load_day(p)
        except Exception:
            continue
        if len(arrs) < 1:
            continue
        r = correlate_day(arrs)
        days.append(d)
        for b in BANDS:
            store[b].append(r[b])
    if len(days) < 20:
        return sta, q, f'only {len(days)} days'
    np.savez_compressed(out, days=np.array(days), **{b: np.array(v) for b, v in store.items()})
    return sta, q, f'{len(days)} days'


def stage_correlate(a):
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for q in QUAKES:
        stas = set()
        for d in window_days(q):
            stas |= {os.path.basename(p)[:-6] for p in glob.glob(f'data/continuous/{d}/*.mseed')}
        jobs += [(s, q) for s in sorted(stas) if s != 'BK.JEWT']       # JEWT handled by sci_noise_dvv_jewt.py
    print(len(jobs), 'station-quake jobs', flush=True)
    with get_context('spawn').Pool(a.procs) as pool:
        for sta, q, msg in pool.imap_unordered(_job, jobs):
            print(sta, q, msg, flush=True)


def stretch(ref, X, lags, win):
    m = (np.abs(lags) >= win[0]) & (np.abs(lags) <= win[1])
    tl = lags[m]
    R = np.array([np.interp(tl * (1 + e), lags, ref) for e in EPS])        # ref stretched: arrivals at t(1+e)
    R = R - R.mean(1, keepdims=True)
    R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-12
    Y = X[:, m]
    Y = Y - Y.mean(1, keepdims=True)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12
    C = Y @ R.T                                     # (ndays, neps)
    k = np.nanargmax(np.nan_to_num(C, nan=-2), 1)
    return EPS[k], C[np.arange(len(k)), k]          # current(t) = ref(t(1+e)) -> dv/v = +e (faster if e>0)


def stage_dvv(a):
    rows = []
    for f in sorted(glob.glob(f'{OUT}/ccf_*.npz')):
        z = np.load(f)
        sta, q = os.path.basename(f)[4:-4].rsplit('_', 1)
        t0 = pd.Timestamp(QUAKES[q])
        days = pd.to_datetime(z['days'])
        rel = np.array([(d - t0).days for d in days])
        pre = (rel < 0)
        for b, (f1, f2, seg, maxlag, win) in BANDS.items():
            X = z[b].astype(float)                   # (nday, npair, nlag)
            lags = (np.arange(X.shape[2]) - X.shape[2] // 2) / FS
            for k, (p1, p2) in enumerate(PAIRS):
                Xk = X[:, k]
                good = np.isfinite(Xk).all(1)
                if (good & pre).sum() < 10 or (good & ~pre).sum() < 10:
                    continue
                ref = Xk[good & pre].mean(0)
                e, c = stretch(ref, np.nan_to_num(Xk), lags, win)
                for i in np.where(good)[0]:
                    rows.append(dict(sta=sta, quake=q, band=b, pair=p1 + p2, day=days[i], rel=rel[i], dvv=e[i] * 100, cc=c[i]))
    D = pd.DataFrame(rows)
    D.to_csv(f'{OUT}/daily_pairs.csv', index=False)
    # combine pairs per day (CC weights, pairs with median CC >= 0.3 only)
    medcc = D.groupby(['sta', 'quake', 'band', 'pair']).cc.transform('median')
    # single-station CROSS-COMPONENT pairs only (EN, EZ, NZ); 0.1-1 Hz autocorrelation coda is ~ the filter response anyway
    auto = D.pair.isin(['ZZ', 'NN', 'EE'])
    E = D[(medcc >= 0.3) & ~auto].copy()
    E['w'] = E.cc.clip(0, 0.99) ** 2 / (1 - E.cc.clip(0, 0.99) ** 2)
    G = (E.assign(wd=E.w * E.dvv).groupby(['sta', 'quake', 'band', 'day', 'rel'])
         .agg(wd=('wd', 'sum'), w=('w', 'sum'), cc=('cc', 'mean'), npair=('pair', 'size')).reset_index())
    G['dvv'] = G.wd / G.w
    G.drop(columns=['wd', 'w']).to_csv(f'{OUT}/daily.csv', index=False)
    rng = np.random.default_rng(0)
    out = []
    for (sta, q, b), g in G.groupby(['sta', 'quake', 'band']):
        s = g.set_index('rel').dvv
        pre, post = s[(s.index >= -30) & (s.index <= -1)], s[(s.index >= 1) & (s.index <= 15)]
        sil = s[(s.index >= 1) & (s.index <= 25)]
        if len(pre) < 10 or len(post) < 5:
            continue
        step = post.mean() - pre.mean()
        # moving-block bootstrap of residuals about the two means
        res = np.r_[pre.values - pre.mean(), post.values - post.mean()]
        bs = []
        for _ in range(400):
            idx = np.concatenate([np.arange(i, i + 5) for i in rng.integers(0, len(res) - 4, len(res) // 5 + 1)])[:len(res)]
            r = res[idx]
            bs.append(r[len(pre):].mean() - r[:len(pre)].mean())
        # placebo inside pre window: 15-d after vs preceding days, all split points with >=8 days each side
        pv = s[s.index <= -1].values
        plc = [pv[i:i + 15].mean() - pv[:i].mean() for i in range(8, len(pv) - 8)]
        se = max(np.std(bs), np.std(plc) if len(plc) > 3 else 0)
        out.append(dict(sta=sta, quake=q, band=b, step_pct=step, se_boot=np.std(bs), sd_placebo=np.std(plc) if len(plc) > 3 else np.nan,
                        se=se, z=step / se if se > 0 else np.nan, silence_mean_pct=(sil.mean() - pre.mean()) if q == 'm7' else np.nan,
                        n_pre=len(pre), n_post=len(post), mean_cc=g.cc.mean()))
    S = pd.DataFrame(out)
    S.to_csv(f'{OUT}/steps.csv', index=False)
    with pd.option_context('display.width', 200, 'display.max_rows', 300):
        print(S.round(3).to_string(index=False))


def stage_plot(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    S = pd.read_csv(f'{OUT}/steps.csv')
    G = pd.read_csv(f'{OUT}/daily.csv')
    bands = list(BANDS)
    fig, axs = plt.subplots(2, 3, figsize=(17, 9))
    for j, q in enumerate(QUAKES):
        ax = axs[0, j]
        for i, b in enumerate(bands):
            s = S[(S.quake == q) & (S.band == b)].sort_values('sta')
            x = np.arange(len(s)) + (i - 1) * 0.25
            ax.errorbar(x, s.step_pct, 2 * s.se, fmt='o', ms=4, label=b, capsize=0)
        s = S[(S.quake == q) & (S.band == bands[1])].sort_values('sta')
        ax.set_xticks(np.arange(len(s)))
        ax.set_xticklabels([v.split('.')[1] for v in s.sta], rotation=90, fontsize=7)
        ax.axhline(0, color='k', lw=0.6)
        ax.set_ylim(-0.6, 0.4)
        ax.set_title(f'{q}: step (post 1-15 d minus pre 30 d), 2 SE')
        ax.set_ylabel('dv/v (%)')
        ax.legend(fontsize=7)
        ax = axs[1, j]
        for b in bands:
            g = G[(G.quake == q) & (G.band == b)]
            m = g.groupby('rel').dvv.median()
            m = m - m[m.index < 0].mean()
            ax.plot(m.index, m.values, '.-', ms=3, lw=0.8, label=f'{b} (median over stations)')
        ax.axvline(0, color='k', ls='--', lw=0.7)
        if q == 'm7':
            ax.axvspan(0, 25, color='0.85', zorder=0, label='LFE silence')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xlabel(f'days from {QUAKES[q]}')
        ax.set_ylabel('dv/v (%)')
        ax.set_ylim(-0.4, 0.3)
        ax.legend(fontsize=7)
    fig.suptitle('Single-station cross-component (EN, EZ, NZ) ambient-noise dv/v by band around the three quakes (data on disk)')
    fig.tight_layout()
    fig.savefig('figures/sci_noise_dvv_bands_quakes.png', dpi=120)
    print('figures/sci_noise_dvv_bands_quakes.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['correlate', 'dvv', 'plot'])
    ap.add_argument('--procs', type=int, default=12)
    a = ap.parse_args()
    dict(correlate=stage_correlate, dvv=stage_dvv, plot=stage_plot)[a.stage](a)
