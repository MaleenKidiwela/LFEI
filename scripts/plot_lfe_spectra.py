#!/usr/bin/env python3
"""LFE S-wave spectra (absolute shape) from the source-spectra extraction (data/sci/source_spectra/raw/fam{F}.h5).

SP / NP: per-event S-window and pre-event noise power in 12 log-spaced bands (1-15 Hz) per channel.
(a) family 23: median signal and noise power per band on its 6 highest-SNR channels (signal above noise only ~2-7 Hz).
(b) all families: median noise-corrected S spectrum (sqrt power, i.e. amplitude) on one common high-SNR channel,
    normalised at 2.2 Hz, with Brune omega-2 shapes (fc = 2, 4, 8 Hz) times a path attenuation term exp(-pi f t*), t* = 0.04 s,
    for reference. Path and site are not removed here; within-family comparisons in sci_source_spectra.py cancel them.
Velocity counts (no response removal; 1-15 Hz bandpass at dataset build) are divided by f to give displacement shape.
Output figures/lfe_spectra.png
"""
import glob
import re

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RAW = 'data/sci/source_spectra/raw'


def load(F):
    f = h5py.File(f'{RAW}/fam{F}.h5', 'r')
    e = f.attrs['edges']
    fc = np.sqrt(e[:-1] * e[1:])
    chan = [c.decode() for c in f['chan'][:]]
    return f, fc, chan


def med_spec(f, k, nmax=4000):
    n = f['SP'].shape[0]
    idx = np.sort(np.random.default_rng(0).choice(n, min(n, nmax), replace=False))
    sp = f['SP'][idx, k, :].astype(float)
    nz = f['NP'][idx, k, :].astype(float)
    ok = np.isfinite(sp).all(1) & np.isfinite(nz).all(1) & (sp.sum(1) > 0)
    return np.median(sp[ok], 0), np.median(nz[ok], 0), ok.sum()


def main():
    fams = sorted(int(re.search(r'fam(\d+)\.h5', p).group(1)) for p in glob.glob(f'{RAW}/fam*.h5') if '_mf' not in p)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    # (a) family 23
    f, fc, chan = load(23)
    snr = f['snr'][:]
    for k in np.argsort(-np.nan_to_num(snr))[:6]:
        s, n, m = med_spec(f, k)
        l, = ax[0].loglog(fc, s, '-o', ms=3, label=f'{chan[k]} (S)')
        ax[0].loglog(fc, n, '--', color=l.get_color(), lw=0.8)
    ax[0].axvspan(2, 6.5, color='0.9', zorder=0)
    ax[0].set_xlabel('frequency (Hz)'); ax[0].set_ylabel('median power per band (counts², uncorrected)')
    ax[0].set_title('(a) Family 23, 6 best channels: S window (solid)\nvs pre-event noise (dashed); shaded = usable band')
    ax[0].legend(fontsize=7)
    # (b) all families on a common channel: choose the channel present in most families with highest median SNR
    cnt = {}
    for F in fams:
        g, _, ch = load(F)
        for c, v in zip(ch, g['snr'][:]):
            cnt.setdefault(c, []).append(v)
    best = max((c for c in cnt if len(cnt[c]) >= 0.8 * len(fams)), key=lambda c: np.nanmedian(cnt[c]))
    cmap = plt.get_cmap('viridis')
    for i, F in enumerate(fams):
        g, fcF, ch = load(F)
        if best not in ch:
            continue
        s, n, m = med_spec(g, ch.index(best))
        a = np.sqrt(np.clip(s - n, 0, None)) / fcF          # velocity counts -> displacement shape (divide by f)
        good = (fcF > 1.8) & (fcF < 7.5) & (s > 2 * n)
        if good.sum() < 3:
            continue
        ref = np.interp(2.2, fcF[good], a[good])
        ax[1].loglog(fcF[good], a[good] / ref, '-', color=cmap(i / len(fams)), lw=1, alpha=0.8)
    ff = np.logspace(np.log10(1.8), np.log10(7.5), 50)
    for fcB, ls in [(2, ':'), (4, '--'), (8, '-.')]:
        b = 1 / (1 + (ff / fcB) ** 2) * np.exp(-np.pi * ff * 0.04)     # displacement
        ax[1].loglog(ff, b / np.interp(2.2, ff, b), 'k', ls=ls, lw=1.2, label=f'Brune fc = {fcB} Hz, t* = 0.04 s')
    ff2 = np.logspace(np.log10(1.8), np.log10(7.5), 50)
    ax[1].loglog(ff2, (ff2 / 2.2) ** -1, color='0.6', lw=0.8, label='f^-1 (reference)')
    ax[1].loglog(ff2, (ff2 / 2.2) ** -2, color='0.6', lw=0.8, ls='--', label='f^-2 (reference)')
    ax[1].set_xlabel('frequency (Hz)'); ax[1].set_ylabel('noise-corrected S DISPLACEMENT amplitude (normalised at 2.2 Hz)')
    ax[1].set_title(f'(b) All families on {best} (borehole): median S displacement\n'
                    'spectrum per family (velocity / f; no attenuation correction)')
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig('figures/lfe_spectra.png', dpi=130)
    print('common channel', best, '-> figures/lfe_spectra.png')


if __name__ == '__main__':
    main()
