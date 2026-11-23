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

TAGS = [t for t in ['petrolia', 'ferndale', 'm7'] if os.path.exists(f'data/scan_quakes/{t}_summary.json')]
NAME = {'petrolia': 'Petrolia M6.2 2021-12-20', 'ferndale': 'Ferndale M6.4 2022-12-20', 'm7': 'M7.0 2024-12-05'}
COL = {'cat': '#444444', 'fixed8': '#1f77b4', 'daily': '#ff7f0e', 'neg': '#d62728', 'hourly': '#2ca02c'}


def fig_threshold():
    fig, ax = plt.subplots(len(TAGS), 1, figsize=(9, 2.6 * len(TAGS)), sharex=True, squeeze=False)
    for i, tag in enumerate(TAGS):
        h = pd.read_csv(f'data/scan_quakes/{tag}_hourly_threshold.csv')
        x = h.hours_since / 24
        a = ax[i, 0]
        a.plot(x, h.mad_rel_med, color='k', lw=1, label='CC-sum MAD (adaptive threshold), median family')
        a.plot(x, h.mad_rel_p90, color='0.6', lw=0.7, label='90th pct family')
        a.plot(x, h.chan_rms_rel_med, color='#d62728', lw=0.8, label='2-8 Hz RMS, median channel')
        a.plot(x, h.nchan_rel_med, color='#2ca02c', lw=0.8, ls='--', label='channels available')
        a.axvline(0, color='b', lw=0.8)
        a.axhline(1, color='0.5', lw=0.5)
        a.set_ylim(0.5, 3)
        a.set_xlim(-3, 10)
        a.set_ylabel('relative to pre-quake')
        a.set_title(NAME[tag], fontsize=10, loc='left')
        if i == 0:
            a.legend(fontsize=7, ncol=2, loc='upper right')
    ax[-1, 0].set_xlabel('days since mainshock')
    fig.tight_layout()
    fig.savefig('figures/sci_scan_quakes_threshold.png', dpi=150)


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
        for lab, n, col, ls in [('catalog', hist(cc.t), COL['cat'], '-'),
                                ('scan fixed 8xMAD0 (+)', hist(D[(D.rule == 'fixed8') & (D.sign == 1)].t), COL['fixed8'], '-'),
                                ('scan daily 8xMAD (+)', hist(D[(D.rule == 'daily') & (D.sign == 1)].t), COL['daily'], '-')]:
            a.step(edges[:-1], n / max(n[pre].mean(), 1e-9), where='post', color=col, lw=1, ls=ls, label=lab)
        nn = hist(D[(D.rule == 'fixed8') & (D.sign == -1)].t)
        npos = hist(D[(D.rule == 'fixed8') & (D.sign == 1)].t)
        a.step(edges[:-1], nn / max(npos[pre].mean(), 1e-9), where='post', color=COL['neg'], lw=0.8,
               label='scan fixed, reversed (false alarms)')
        a.axvline(0, color='b', lw=0.8)
        a.axhline(1, color='0.5', lw=0.5)
        a.set_ylabel('daily rate / pre-quake')
        a.set_ylim(0, 2.5)
        a.set_title(NAME[tag], fontsize=10, loc='left')
        if i == 0:
            a.legend(fontsize=7, ncol=2, loc='upper left')
    ax[-1, 0].set_xlabel('days since mainshock')
    fig.tight_layout()
    fig.savefig('figures/sci_scan_quakes_rates.png', dpi=150)


def fig_ratios():
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    wins = ['24h', '2d', '30d']
    w = 0.2
    for i, tag in enumerate(TAGS):
        s = json.load(open(f'data/scan_quakes/{tag}_summary.json'))
        for j, (lab, val, col) in enumerate([
                ('catalog', [s['catalog'][k]['ratio'] for k in wins], COL['cat']),
                ('scan fixed8 (FA-corr.)', [s['fixed8'][k]['ratio_fa_corr'] for k in wins], COL['fixed8']),
                ('scan fixed8 / inj. eff.', [s.get('corrected', {}).get('fixed8', {}).get(k, np.nan) for k in wins], '#9467bd'),
                ('scan daily 8xMAD', [s['daily'][k]['ratio'] for k in wins], COL['daily'])]):
            xs = np.arange(3) + i * 3.6 + (j - 1.5) * w
            ax[0].bar(xs, val, w, color=col, label=lab if i == 0 else None)
    ax[0].set_xticks([k + i * 3.6 for i in range(len(TAGS)) for k in range(3)])
    ax[0].set_xticklabels([f'{k}\n{tag}' if k == '2d' else k for tag in TAGS for k in wins], fontsize=8)
    ax[0].axhline(1, color='0.5', lw=0.5)
    ax[0].set_ylabel('post / pre rate ratio')
    ax[0].legend(fontsize=7)
    for i, tag in enumerate(TAGS):
        p = f'data/scan_quakes/{tag}_injection.csv'
        if not os.path.exists(p):
            continue
        I = pd.read_csv(p)
        e = np.array([-48, -24, 0, 3, 6, 12, 24, 36, 48, 72, 96])
        x, y, lo, hi = [], [], [], []
        for a0, a1 in zip(e[:-1], e[1:]):
            k = (I.h >= a0) & (I.h < a1)
            if k.sum() < 5:
                continue
            p_ = I.det_fixed8[k].mean()
            se = np.sqrt(p_ * (1 - p_) / k.sum())
            x.append((a0 + a1) / 2)
            y.append(p_)
            lo.append(se)
        ax[1].errorbar(x, y, yerr=lo, marker='o', ms=3, lw=1, capsize=2, label=NAME[tag])
    ax[1].axvline(0, color='b', lw=0.8)
    ax[1].set_xlabel('hours since mainshock')
    ax[1].set_ylabel('injected-event detection efficiency (fixed 8xMAD0)')
    ax[1].set_ylim(0, 1)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig('figures/sci_scan_quakes_ratios.png', dpi=150)


if __name__ == '__main__':
    fig_threshold()
    fig_rates()
    fig_ratios()
