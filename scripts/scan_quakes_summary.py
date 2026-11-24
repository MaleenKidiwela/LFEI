#!/usr/bin/env python3
"""Three-quake summary table from data/scan_quakes/{tag}_summary.json (scan_quakes_analysis.py).

Output data/scan_quakes/summary_three_quakes.csv: one row per quake x config x window x rule.
"""
import json
import os

import pandas as pd

ROWS = []
for base in ['petrolia', 'ferndale', 'm7']:
    for cfg in ['_fc', '']:
        f = f'data/scan_quakes/{base}{cfg}_summary.json'
        if not os.path.exists(f):
            continue
        s = json.load(open(f))
        pf = f'data/scan_quakes/{base}{cfg}_popeff.json'
        pop = json.load(open(pf)) if os.path.exists(pf) else {}
        for k in ['6h', '24h', 'd1_2', '2d', 'd2_7', '30d']:
            c = s['catalog'][k]
            for rn in ['fixed10', 'fixed8', 'daily']:
                x = s[rn][k]
                co = s.get('corrected', {}).get(rn, {}).get(k, {})
                th = s['threshold'].get(k, {})
                ROWS.append(dict(
                    quake=base, channels='fixed set' if cfg else 'all', window=k, rule=rn,
                    cat_n=c['n'], cat_ratio=c['ratio'], cat_nfam=c['nfam'], cat_n_eqassoc=c.get('n_eqassoc'),
                    cat_n_clean=c.get('n_clean'), cat_ratio_clean=c.get('ratio_clean'),
                    cat_nfam_clean=c.get('nfam_clean'),
                    scan_pre_pos=s[rn]['pre_pos'], scan_pre_fa=s[rn]['pre_neg'],
                    scan_pos=x['pos'], scan_fa=x['neg'], scan_catmatch=x['catmatch'],
                    scan_pos_eqassoc=x.get('pos_eqassoc'), scan_nfam_pos=x['nfam_pos'], scan_nfam_fa=x['nfam_neg'],
                    scan_ratio_raw=x['ratio'], scan_ratio_fa_corr=x['ratio_fa_corr'],
                    rel_efficiency=co.get('mean_rel_eff'), scan_observed_net=co.get('observed_net'),
                    scan_expected_if_unchanged=co.get('expected_if_unchanged'),
                    scan_ratio_eff_corr=co.get('ratio_eff_corr'), scan_p_empirical=co.get('p_empirical'),
                    cat_ratio_eff_corr=co.get('catalog_ratio_eff_corr'),
                    cat_expected_if_unchanged=co.get('catalog_expected_if_unchanged'),
                    cat_p_empirical=co.get('catalog_p_empirical'),
                    pop_rel_eff=pop.get(rn, {}).get(k, {}).get('mean_rel_eff'),
                    pop_scan_ratio_eff_corr=pop.get(rn, {}).get(k, {}).get('ratio_eff_corr'),
                    pop_cat_clean_ratio_eff_corr=pop.get(rn, {}).get(k, {}).get('catalog_clean_ratio_eff_corr'),
                    cat_clean_ratio_eff_corr=co.get('catalog_ratio_eff_corr', float('nan')) * (c.get('ratio_clean') or 0) / max(c['ratio'], 1e-9),
                    recall_pre=s[rn]['recall']['pre']['any'], recall_win=s[rn]['recall'][k]['any'],
                    mad_rel_median=th.get('mad_rel_median'), mad_rel_max=th.get('mad_rel_max'),
                    chan_rms_rel_median=th.get('chan_rms_rel_median'), nchan_rel=th.get('nchan_rel_median')))
df = pd.DataFrame(ROWS)
df.to_csv('data/scan_quakes/summary_three_quakes.csv', index=False)
pd.set_option('display.width', 250)
cols = ['quake', 'channels', 'window', 'rule', 'cat_n', 'cat_ratio', 'cat_ratio_clean', 'scan_pos', 'scan_fa',
        'scan_ratio_fa_corr', 'rel_efficiency', 'scan_ratio_eff_corr', 'scan_p_empirical', 'cat_clean_ratio_eff_corr',
        'cat_p_empirical', 'pop_rel_eff', 'pop_scan_ratio_eff_corr', 'pop_cat_clean_ratio_eff_corr',
        'mad_rel_median', 'chan_rms_rel_median']
print(df[df.window.isin(['24h', '2d', '30d']) & df.rule.isin(['fixed10', 'daily'])][cols].round(2).to_string(index=False))
