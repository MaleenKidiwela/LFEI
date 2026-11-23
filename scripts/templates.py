#!/usr/bin/env python3
"""Master templates per channel + yearly-stack alignment QC for one family.

Master stack (per channel, 2-8 Hz):
  1. seed = top SEED_FRAC of events by v2 ccsum; RMS-normalise each trace in the signal
     window; linear stack.
  2. N_ITER times: cross-correlate every event with the stack (lag <= MAXLAG), keep events
     with CC >= CC_KEEP (capped to the best KEEP_FRAC), shift and restack.
Yearly stacks: per calendar year, stack of catalog-aligned traces (no shifts), CC vs master
-> lag(year) and CC(year). A drifting lag flags clock / instrument / template-time issues.

Outputs data/templates/fam{F}_templates.h5 and data/templates/fam{F}_yearly_qc.csv,
figure figures/fam{F}_yearly_lag_qc.png.
"""
import argparse
import os

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

SEED_FRAC, N_ITER, MAXLAG, CC_KEEP, KEEP_FRAC = 0.2, 2, 0.3, 0.1, 0.5
SIG = (-3.0, 25.0)      # s rel. catalog time: correlation / normalisation window
NOISE = (-15.0, -5.0)   # s rel. catalog time: noise window


def xcorr_lag(a, B, maxlag):
    """Normalised CC of template a with each row of B, lags in [-maxlag, maxlag] samples.
    Returns best lag (samples, positive = B later than a) and CC."""
    n = a.size
    nfft = int(2 ** np.ceil(np.log2(2 * n)))
    A = np.fft.rfft(a, nfft)
    X = np.fft.rfft(B, nfft, axis=1)
    cc = np.fft.irfft(X * np.conj(A)[None, :], nfft, axis=1)
    cc = np.concatenate([cc[:, -maxlag:], cc[:, :maxlag + 1]], axis=1)
    norm = np.linalg.norm(a) * np.linalg.norm(B, axis=1)
    cc = cc / norm[:, None]
    k = np.argmax(cc, axis=1)
    return k - maxlag, cc[np.arange(len(B)), k]


def xcorr_lag_frac(a, B, maxlag):
    """As xcorr_lag but with parabolic sub-sample interpolation of the peak (float lags)."""
    n = a.size
    nfft = int(2 ** np.ceil(np.log2(2 * n)))
    cc = np.fft.irfft(np.fft.rfft(B, nfft, axis=1) * np.conj(np.fft.rfft(a, nfft))[None, :], nfft, axis=1)
    cc = np.concatenate([cc[:, -maxlag:], cc[:, :maxlag + 1]], axis=1)
    cc /= (np.linalg.norm(a) * np.linalg.norm(B, axis=1))[:, None]
    k = np.argmax(cc, axis=1)
    r = np.arange(len(B))
    km, kp = np.clip(k - 1, 0, 2 * maxlag), np.clip(k + 1, 0, 2 * maxlag)
    y0, y1, y2 = cc[r, km], cc[r, k], cc[r, kp]
    den = y0 - 2 * y1 + y2
    frac = np.where((k > 0) & (k < 2 * maxlag) & (den != 0), 0.5 * (y0 - y2) / den, 0.0)
    return k + frac - maxlag, y1


def shift_rows(D, lags):
    out = np.zeros_like(D)
    for i, L in enumerate(lags):
        out[i] = np.roll(D[i], -L)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', type=int, default=23)
    ap.add_argument('--h5', default=None)
    ap.add_argument('--fmin', type=float, default=2.0)
    ap.add_argument('--fmax', type=float, default=8.0)
    ap.add_argument('--min-events', type=int, default=50)
    ap.add_argument('--tmin', default=None, help='only events at/after this time (era-matched templates)')
    ap.add_argument('--tmax', default=None, help='only events before this time')
    ap.add_argument('--out', default=None, help='output templates h5 (default data/templates/fam{F}_templates.h5)')
    a = ap.parse_args()
    F = a.family
    h5 = a.h5 or f'data/waveforms/fam{F}.h5'
    os.makedirs('data/templates', exist_ok=True)

    h = h5py.File(h5, 'r')
    sr, pre = h.attrs['sr'], h.attrs['pre']
    t_ev = pd.to_datetime(h['events/time_utc'][:].astype(str))
    ccsum = h['events/ccsum_v2'][:]
    sos = butter(4, [a.fmin, a.fmax], btype='band', fs=sr, output='sos')
    i_sig = slice(int((SIG[0] - pre) * sr), int((SIG[1] - pre) * sr))
    i_noi = slice(int((NOISE[0] - pre) * sr), int((NOISE[1] - pre) * sr))
    maxlag = int(MAXLAG * sr)

    era = np.ones(len(t_ev), bool)
    if a.tmin:
        era &= t_ev >= pd.Timestamp(a.tmin)
    if a.tmax:
        era &= t_ev < pd.Timestamp(a.tmax)
    out = h5py.File(a.out or f'data/templates/fam{F}_templates.h5', 'w')
    out.attrs['tmin'] = str(a.tmin)
    out.attrs['tmax'] = str(a.tmax)
    out.attrs.update(sr=sr, pre=pre, fmin=a.fmin, fmax=a.fmax, sig=SIG, noise=NOISE)
    qc = []
    for cid in sorted(h['chan']):
        ev = h[f'chan/{cid}/ev'][:]
        rows_ = np.where(era[ev])[0]
        if len(rows_) < a.min_events:
            continue
        ev = ev[rows_]
        D = sosfiltfilt(sos, h[f'chan/{cid}/data'][rows_].astype('float64'), axis=1)
        D /= (np.sqrt(np.mean(D[:, i_sig] ** 2, axis=1)) + 1e-20)[:, None]
        seed = np.argsort(ccsum[ev])[::-1][:max(a.min_events // 2, int(SEED_FRAC * len(ev)))]
        stack = D[seed].mean(0)
        lags = np.zeros(len(ev), int)
        for _ in range(N_ITER):
            lags, cc = xcorr_lag(stack[i_sig], D[:, i_sig], maxlag)
            ok = np.where(cc >= CC_KEEP)[0]
            ok = ok[np.argsort(cc[ok])[::-1][:max(a.min_events // 2, int(KEEP_FRAC * len(ev)))]]
            if len(ok) < 10:
                break
            stack = shift_rows(D[ok], lags[ok]).mean(0)
        snr = np.max(np.abs(stack[i_sig])) / (np.sqrt(np.mean(stack[i_noi] ** 2)) + 1e-20)
        g = out.create_group(cid)
        g['stack'] = stack.astype('float32')
        g['ev'] = ev
        g['lag_s'] = (lags / sr).astype('float32')
        g['cc'] = cc.astype('float32')
        g.attrs.update(n=len(ev), n_used=len(ok), snr=snr)
        # yearly catalog-aligned stacks vs master
        yrs = t_ev[ev].year
        for y in np.unique(yrs):
            sel = np.where(yrs == y)[0]
            if len(sel) < 20:
                continue
            ys = D[sel].mean(0)
            L, C = xcorr_lag_frac(stack[i_sig], ys[None, i_sig], maxlag)
            ysnr = np.max(np.abs(ys[i_sig])) / (np.sqrt(np.mean(ys[i_noi] ** 2)) + 1e-20)
            qc.append(dict(chan=cid, year=int(y), n=len(sel), lag_s=L[0] / sr, cc=C[0], snr=ysnr))
        print(f'{cid:20s} n={len(ev):6d} used={len(ok):6d} SNR={snr:6.1f} '
              f'median CC={np.median(cc):.2f}', flush=True)
    out.close()
    q = pd.DataFrame(qc)
    if a.tmin or a.tmax:          # era templates: skip yearly QC outputs
        print('era templates written'); return
    q.to_csv(f'data/templates/fam{F}_yearly_qc.csv', index=False)

    # QC figure: lag vs year per station (vertical or first component), only reliable years
    q['sta'] = q.chan.str.split('.').str[:2].str.join('.')
    good = q[(q.cc >= 0.3)]
    stas = sorted(good.sta.unique())
    fig, ax = plt.subplots(figsize=(12, max(4, 0.28 * len(stas))))
    for j, s in enumerate(stas):
        g = good[good.sta == s]
        ax.scatter(g.year, np.full(len(g), j) + g.lag_s.clip(-0.1, 0.1) / 0.2, c=g.cc, cmap='viridis',
                   vmin=0.3, vmax=1, s=12)
        ax.axhline(j, color='0.85', lw=0.5, zorder=0)
    ax.set_yticks(range(len(stas)))
    ax.set_yticklabels(stas, fontsize=7)
    ax.set_xlabel('year')
    ax.set_title(f'Family {F}: yearly-stack lag vs master (offset from row line; ±0.1 s = ±0.5 row), '
                 'colour = CC')
    fig.tight_layout()
    fig.savefig(f'figures/fam{F}_yearly_lag_qc.png', dpi=130)
    print('wrote data/templates/, figures/fam%d_yearly_lag_qc.png' % F)


if __name__ == '__main__':
    main()
