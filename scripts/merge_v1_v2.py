#!/usr/bin/env python3
"""Merge Shelly LFE catalogs v1 (2025, 2018-2024) and v2 (2026, 2008-2026) into one
per-family catalog where each physical event appears once.

Rules
-----
* v2 is the base catalog: its event times and family assignments are kept.
* Each family's template start time differs by a near-constant offset between versions
  (v1 - v2, up to ~±1.1 s). Offsets are estimated per family from same-family matches.
* A v1 event is "matched" if
    (a) a same-family v2 event lies within MATCH_TOL of (t_v1 - offset_family), or else
    (b) any v2 event (any family) lies within DEDUP_WIN -- the catalogs' own
        cross-family duplicate window -- in which case it is flagged as reassigned.
* Unmatched v1 events are added to their v1 family, time-shifted onto the v2 time base.

Output columns: time fields (v2 time base), idnum (final family), ID, source
(both | v2_only | v1_only), idnum_v1 (v1 family, if in v1), reassigned, dt_v1_v2
(raw v1 - v2 time, s), and per-version detection stats (*_v1, *_v2).
"""
import argparse
import numpy as np
import pandas as pd

COLS = ['year', 'month', 'day', 'hour', 'minute', 'second', 's_of_day', 'ccsum',
        'thresh', 'cc_div_thr', 'meancc', 'nchan', 'ID', 'idnum']
STATS = ['ccsum', 'thresh', 'cc_div_thr', 'meancc', 'nchan']
COARSE_WIN = 2.0   # s, window used only to estimate per-family offsets
MATCH_TOL = 0.25   # s, same-family match tolerance after offset correction
DEDUP_WIN = 4.0    # s, cross-family window (same as the catalogs' duplicate removal)


def load(path):
    d = pd.read_csv(path)
    d.columns = COLS
    day = pd.to_datetime(dict(year=d.year, month=d.month, day=d.day))
    d['t'] = day + pd.to_timedelta(d.s_of_day, unit='s')
    d['ts'] = (d.t - pd.Timestamp('1970-01-01')) / pd.Timedelta(seconds=1)   # unit-safe (pandas>=3 uses us)
    return d


def nearest(t_sorted, t_query):
    """Index of nearest element in sorted t_sorted for each t_query, and signed dt."""
    if len(t_sorted) == 0:
        return np.full(len(t_query), -1), np.full(len(t_query), np.inf)
    i = np.clip(np.searchsorted(t_sorted, t_query), 1, max(len(t_sorted) - 1, 1))
    i0 = np.clip(i - 1, 0, len(t_sorted) - 1)
    i1 = np.clip(i, 0, len(t_sorted) - 1)
    pick = np.where(np.abs(t_query - t_sorted[i0]) <= np.abs(t_query - t_sorted[i1]), i0, i1)
    return pick, t_query - t_sorted[pick]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v1', default='data/shelly2025_MTJ_v1/LFEcat_MTJ_2018-2024.csv')
    ap.add_argument('--v2', default='data/shelly2026_MTJ_v2/LFEcatV2_MTJ_2008-2026.csv')
    ap.add_argument('--out', default='data/merged/LFEcat_MTJ_merged_v1v2.csv')
    ap.add_argument('--offsets-out', default='data/merged/family_time_offsets_v1_minus_v2.csv')
    a = ap.parse_args()

    v1, v2 = load(a.v1), load(a.v2)
    v2 = v2.sort_values('ts').reset_index(drop=True)

    # --- per-family offsets (v1 - v2) from coarse same-family matches
    offsets = {}
    for f, g1 in v1.groupby('idnum'):
        g2 = v2[v2.idnum == f]
        _, dt = nearest(g2.ts.values, g1.ts.values)
        dt = dt[np.abs(dt) <= COARSE_WIN]
        offsets[f] = float(np.median(dt))
    off_tab = pd.DataFrame({'idnum': list(offsets), 'offset_s': list(offsets.values())})

    # --- match each v1 event to a v2 event
    v1['v2_idx'] = -1
    v1['reassigned'] = False
    v1['dt_raw'] = np.nan
    for f, g1 in v1.groupby('idnum'):
        g2 = v2[v2.idnum == f]
        pick, dt = nearest(g2.ts.values, g1.ts.values - offsets[f])
        ok = np.abs(dt) <= MATCH_TOL
        v1.loc[g1.index[ok], 'v2_idx'] = g2.index.values[pick[ok]]
    # fallback: any-family match within the duplicate window
    rest = v1.index[v1.v2_idx < 0]
    pick, dt = nearest(v2.ts.values, v1.loc[rest, 'ts'].values)
    ok = np.abs(dt) <= DEDUP_WIN
    v1.loc[rest[ok], 'v2_idx'] = pick[ok]
    v1.loc[rest[ok], 'reassigned'] = True

    # resolve many-to-one: keep only the closest v1 event per v2 event
    m = v1[v1.v2_idx >= 0].copy()
    m['dt_raw'] = m.ts.values - v2.ts.values[m.v2_idx.values]
    m['resid'] = np.abs(m.dt_raw - np.where(m.reassigned, 0, m.idnum.map(offsets)))
    m = m.sort_values('resid').drop_duplicates('v2_idx', keep='first')
    v1.loc[:, 'v2_idx'] = -1
    v1.loc[m.index, 'v2_idx'] = m.v2_idx
    v1.loc[m.index, 'dt_raw'] = m.dt_raw
    v1.loc[~v1.index.isin(m.index), 'reassigned'] = False

    # --- assemble merged catalog
    out = v2[['ts', 'ID', 'idnum'] + STATS].rename(columns={s: s + '_v2' for s in STATS})
    out['source'] = 'v2_only'
    out['idnum_v1'] = pd.array([pd.NA] * len(out), dtype='Int64')
    out['reassigned'] = False
    out['dt_v1_v2'] = np.nan
    for s in STATS:
        out[s + '_v1'] = np.nan
    mm = v1[v1.v2_idx >= 0]
    idx = mm.v2_idx.values.astype(int)
    out.loc[idx, 'source'] = 'both'
    out.loc[idx, 'idnum_v1'] = mm.idnum.values
    out.loc[idx, 'reassigned'] = mm.reassigned.values
    out.loc[idx, 'dt_v1_v2'] = mm.dt_raw.values
    for s in STATS:
        out.loc[idx, s + '_v1'] = mm[s].values

    only = v1[v1.v2_idx < 0]
    add = pd.DataFrame({
        'ts': only.ts.values - only.idnum.map(offsets).values,   # shift to v2 time base
        'ID': only.ID.values, 'idnum': only.idnum.values, 'source': 'v1_only',
        'idnum_v1': only.idnum.values, 'reassigned': False, 'dt_v1_v2': np.nan})
    for s in STATS:
        add[s + '_v2'] = np.nan
        add[s + '_v1'] = only[s].values
    # drop v1-only events that fall within DEDUP_WIN of any v2 event (catalog duplicate rule)
    _, dt = nearest(v2.ts.values, add.ts.values)
    dup = np.abs(dt) <= DEDUP_WIN
    n_dup = int(dup.sum())
    add = add[~dup]
    out = pd.concat([out, add], ignore_index=True).sort_values('ts').reset_index(drop=True)

    out['reassigned'] = (out.source == 'both') & (out.idnum_v1 != out.idnum).fillna(False)
    t = pd.to_datetime(out.ts, unit='s').dt.round('10ms')
    out.insert(0, 'year', t.dt.year)
    out.insert(1, 'month', t.dt.month)
    out.insert(2, 'day', t.dt.day)
    out.insert(3, 'hour', t.dt.hour)
    out.insert(4, 'minute', t.dt.minute)
    out.insert(5, 'second', (t.dt.second + t.dt.microsecond / 1e6).round(2))
    out.insert(6, 's_of_day', ((t - t.dt.normalize()).dt.total_seconds()).round(2))
    out.insert(7, 'time_utc', t.dt.strftime('%Y-%m-%dT%H:%M:%S.%f').str[:-4])
    out = out.drop(columns='ts')
    out['idnum_v1'] = out['idnum_v1'].astype('Int64')
    out['dt_v1_v2'] = out['dt_v1_v2'].round(2)

    import os
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for c in ['thresh_v1', 'thresh_v2', 'ccsum_v1', 'ccsum_v2', 'cc_div_thr_v1', 'cc_div_thr_v2']:
        out[c] = out[c].round(2)
    for c in ['meancc_v1', 'meancc_v2']:
        out[c] = out[c].round(3)
    for c in ['nchan_v1', 'nchan_v2']:
        out[c] = out[c].astype('Int64')
    out.to_csv(a.out, index=False)
    off_tab.round(3).to_csv(a.offsets_out, index=False)

    # --- summary
    print(f"v1 {len(v1)}  v2 {len(v2)}  merged {len(out)}")
    print(out.source.value_counts().to_string())
    print(f"v1-only events dropped as duplicates (within {DEDUP_WIN} s of a v2 event): {n_dup}")
    print("reassigned (v1 family != v2 family):", int(out.reassigned.sum()))
    print(f"\n{'fam':>3} {'merged':>7} {'both':>6} {'v2only':>7} {'v1only':>7} {'offset_s':>9}")
    for f, g in out.groupby('idnum'):
        c = g.source.value_counts()
        print(f"{f:3d} {len(g):7d} {c.get('both', 0):6d} {c.get('v2_only', 0):7d} "
              f"{c.get('v1_only', 0):7d} {offsets[f]:+9.2f}")


if __name__ == '__main__':
    main()
