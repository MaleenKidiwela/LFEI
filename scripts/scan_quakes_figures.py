#!/usr/bin/env python3
"""Figures for the quake matched-filter scans (outputs of scan_quakes_analysis.py).

figures/sci_scan_quakes_threshold.png  hourly CC-sum MAD (= adaptive threshold) and channel RMS
                                       relative to pre-quake, -3..+10 d, three quakes
figures/sci_scan_quakes_rates.png      daily rates: catalog vs our fixed-threshold scan (and reversed
                                       false alarms), -30..+30 d, normalised to pre-quake rate
figures/sci_scan_quakes_ratios.png     24 h / 2 d / 30 d rate ratios: catalog vs scan (fixed, adaptive),
                                       and injection detection efficiency
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from scan_quakes_analysis import QUAKES, catalog  # noqa: E402

SUFFIX = os.environ.get('SCAN_SUFFIX', '_fc')     # '_fc' = fixed channel set (primary); '' = all channels
TAGS = [t + SUFFIX for t in ['petrolia', 'ferndale', 'm7'] if os.path.exists(f'data/scan_quakes/{t}{SUFFIX}_summary.json')]
NAME0 = {'petrolia': 'Petrolia M6.2 2021-12-20', 'ferndale': 'Ferndale M6.4 2022-12-20', 'm7': 'M7.0 2024-12-05'}
NAME = {t: NAME0[t.split('_')[0]] for t in TAGS}
COL = {'cat': '#444444', 'fixed8': '#1f77b4', 'daily': '#ff7f0e', 'neg': '#d62728', 'hourly': '#2ca02c'}


def fig_threshold():
    fig, ax = plt.subplots(len(TAGS), 1, figsize=(9, 2.6 * len(TAGS)), sharex=True, squeeze=False)
    for i, tag in enumerate(TAGS):
        h = pd.read_csv(f'data/scan_quakes/{tag}_hourly_threshold.csv')
        x = h.hours_since / 24
        a = ax[i, 0]
        a.plot(x, h.mad_rel_med, color='k', lw=1, label='CC-sum MAD (adaptive threshold), median family')
        a.plot(x, h.mad_rel_p90, color='0.6', lw=0.7, label='90th pct family')
        a.plot(x, h.nchan_rel_med, color='#2ca02c', lw=0.8, ls='--', label='channels available')
        a2 = a.twinx()
        a2.plot(x, h.chan_rms_rel_med, color='#d62728', lw=0.7, alpha=0.8, label='2-8 Hz RMS, median channel (right, log)')
        a2.set_yscale('log')
        a2.set_ylim(0.3, 300)
        a2.set_ylabel('RMS / pre-quake', color='#d62728', fontsize=8)
        a.axvline(0, color='b', lw=0.8)
        a.axhline(1, color='0.5', lw=0.5)
        a.set_ylim(0.7, 1.5)
        a.set_xlim(-3, 10)
        a.set_ylabel('CC-sum MAD / pre-quake')
        a.set_title(NAME[tag], fontsize=10, loc='left')
        if i == 0:
            h1, l1 = a.get_legend_handles_labels()
            h2, l2 = a2.get_legend_handles_labels()
            a.legend(h1 + h2, l1 + l2, fontsize=7, ncol=2, loc='upper right')
    ax[-1, 0].set_xlabel('days since mainshock')
    fig.tight_layout()
    fig.savefig(f'figures/sci_scan_quakes_threshold{SUFFIX}.png', dpi=150)


def fig_rates():
    c = catalog()
    fig, ax = plt.subplots(len(TAGS), 1, figsize=(9, 2.6 * len(TAGS)), sharex=True, squeeze=False)
    for i, tag in enumerate(TAGS):
        s = json.load(open(f'data/scan_quakes/{tag}_summary.json'))
        Q = pd.Timestamp(s['quake'])
        t0, t1 = pd.Timestamp(s['window'][0]), pd.Timestamp(s['window'][1])
        D = pd.read_csv(f'data/scan_quakes/{tag}_detections.csv', parse_dates=['t'])
        a = ax[i, 0]
        edges = np.arange(np.floor((t0 - Q).total_seconds() / 86400), (t1 - Q).total_seconds() / 86400 + 1e-6, 1.0)
        def hist(tt):
            d = (tt - Q).dt.total_seconds() / 86400
            return np.histogram(d, edges)[0]
        cc = c[(c.t >= t0) & (c.t < t1)]
        pre = edges[:-1] < 0
        fa = D.fa.astype(str) == 'True'
        npos = hist(D[(D.rule == 'fixed10') & (D.sign == 1)].t)
        for lab, n, col, ls in [('catalog', hist(cc.t), COL['cat'], '-'),
                                ('scan fixed 10xMAD0 (+)', npos, COL['fixed8'], '-'),
                                ('scan daily 8xMAD (+)', hist(D[(D.rule == 'daily') & (D.sign == 1)].t), COL['daily'], '-')]:
            a.step(edges[:-1], n / max(n[pre].mean(), 1e-9), where='post', color=col, lw=1, ls=ls, label=lab)
        nn = hist(D[(D.rule == 'fixed10') & (D.sign == -1) & fa].t)
        a.step(edges[:-1], nn / max(npos[pre].mean(), 1e-9), where='post', color=COL['neg'], lw=0.8,
               label='scan fixed 10x, reversed false alarms')
        a.axvline(0, color='b', lw=0.8)
        a.axhline(1, color='0.5', lw=0.5)
        a.set_ylabel('daily rate / pre-quake')
        a.set_ylim(0, 3)
        a.set_title(NAME[tag], fontsize=10, loc='left')
        if i == 0:
            a.legend(fontsize=7, ncol=2, loc='upper left')
    ax[-1, 0].set_xlabel('days since mainshock')
    fig.tight_layout()
    fig.savefig(f'figures/sci_scan_quakes_rates{SUFFIX}.png', dpi=150)


def fig_ratios():
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    wins = ['24h', '2d', '30d']
    w = 0.16
    series = [('catalog', lambda s, k: s['catalog'][k]['ratio'], COL['cat']),
              ('catalog, aftershock-associated removed', lambda s, k: s['catalog'][k].get('ratio_clean', np.nan), '0.65'),
              ('scan fixed 10xMAD0 (FA-corr.)', lambda s, k: s['fixed10'][k]['ratio_fa_corr'], COL['fixed8']),
              ('scan fixed 10x, efficiency-corr.', lambda s, k: s.get('corrected', {}).get('fixed10', {}).get(k, {}).get('ratio_eff_corr', np.nan), '#9467bd'),
              ('scan daily 8xMAD, efficiency-corr.', lambda s, k: s.get('corrected', {}).get('daily', {}).get(k, {}).get('ratio_eff_corr', np.nan), COL['daily'])]
    for i, tag in enumerate(TAGS):
        s = json.load(open(f'data/scan_quakes/{tag}_summary.json'))
        for j, (lab, fn, col) in enumerate(series):
            xs = np.arange(3) + i * 3.8 + (j - 2) * w
            ax[0].bar(xs, [max(fn(s, k), 0) for k in wins], w, color=col, label=lab if i == 0 else None)
    ax[0].set_xticks([k + i * 3.8 for i in range(len(TAGS)) for k in range(3)])
    ax[0].set_xticklabels([f'{k}\n{NAME[tag].split()[0]}' if k == '2d' else k for tag in TAGS for k in wins], fontsize=8)
    ax[0].axhline(1, color='0.5', lw=0.5)
    ax[0].set_ylim(0, 1.6)
    ax[0].set_ylabel('post / pre rate ratio')
    ax[0].legend(fontsize=7, loc='upper left')
    for i, tag in enumerate(TAGS):
        p = f'data/scan_quakes/{tag}_injection.csv'
        if not os.path.exists(p):
            continue
        I = pd.read_csv(p)
        e0 = I.det_daily[I.h < 0].mean()
        e = np.array([0, 6, 24, 48, 96, 168, 336, 720])
        x, y, er = [], [], []
        for a0, a1 in zip(e[:-1], e[1:]):
            k = (I.h >= a0) & (I.h < a1)
            if k.sum() < 20:
                continue
            p_ = I.det_daily[k].mean()
            x.append(np.sqrt(a0 * a1) if a0 > 0 else 3)
            y.append(p_ / e0)
            er.append(np.sqrt(p_ * (1 - p_) / k.sum()) / e0)
        ax[1].errorbar(np.array(x) / 24, y, yerr=er, marker='o', ms=3, lw=1, capsize=2, label=NAME[tag])
    ax[1].set_xscale('log')
    ax[1].set_xticks([0.1, 0.3, 1, 3, 10, 30])
    ax[1].set_xticklabels(['0.1', '0.3', '1', '3', '10', '30'])
    ax[1].minorticks_off()
    ax[1].axhline(1, color='0.5', lw=0.5)
    ax[1].set_xlabel('days since mainshock (bin centre)')
    ax[1].set_ylabel('relative detection efficiency\n(injected LFEs, daily 8xMAD rule)')
    ax[1].set_ylim(0, 1.5)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f'figures/sci_scan_quakes_ratios{SUFFIX}.png', dpi=150)


if __name__ == '__main__':
    fig_threshold()
    fig_rates()
    fig_ratios()
