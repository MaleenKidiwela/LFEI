#!/usr/bin/env python3
"""EQ-above step 2a/2c: relative relocation of catalog earthquake-family events, in-situ Vp/Vs (Lin & Shearer style)
and time-lapse P, S, S-P changes per station from the earthquake families.

Input: meas_cat.parquet (sci_eq_above_measure.py), families.csv, 3D travel-time tables (data/velocity/tt_3d).
geom  : per family centroid (median catalog hypocentre): g_P, g_S (s/km), T_P, T_S per station (station_geom of
        sci_dvv_source_region) -> geom_eq.csv
reloc : per family, rows with cc >= CCMIN_S / CCMIN_P:  y_ec = tau_e + g_c . dx_e + off_c + r_ec
        (IRLS Huber; channel offsets regularised to zero-mean; sigma per station-phase = robust residual scatter;
        events with >= 6 rows) -> reloc_events.csv (tau, dx, dy, dz + formal se), reloc_resid.parquet
vpvs  : Lin & Shearer (2007) style: for every event, demean P and S delays separately over its stations that have
        both phases; robust slope of dS on dP (orthogonal regression, IRLS) = in-situ Vp/Vs of the family volume.
        Through time: per era (pre-Petrolia / Petrolia-M7 / post-M7) and before/after each quake (+-3 yr); bootstrap
        over events; placebo: random split dates -> vpvs.csv
lapse : residual r_ec (location- and origin-free part of the path delay) per station and phase, fitted with the time
        model of sci_dvv_source_region (annual, Petrolia/Ferndale/M7 steps, 60-d early windows, ramp from 2021-06);
        also S-P per event-station.  Errors: bootstrap over events (B=300) and placebo date shifts
        (5.25..12.25 yr, pre-2021-06 data only) -> lapse_station.csv, lapse_network.csv
Usage: nice -n 10 python scripts/sci_eq_above_lapse.py --stage geom|reloc|vpvs|lapse
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/seismic/LFE'
os.chdir(ROOT)
OUT = f'{ROOT}/data/sci/eq_above'
sys.path.insert(0, f'{ROOT}/scripts')
import sci_dvv_source_region as SR_  # noqa: E402
from sci_dvv_instrument_check import BREAKS2, EXCL  # noqa: E402,F401

CCMIN_S, CCMIN_P = 0.6, 0.5
RNG = np.random.default_rng(11)
T_Q = SR_.T_Q
YR = SR_.YR


def fam_centroids():
    c = pd.read_csv(f'{OUT}/families.csv')
    from pyproj import Transformer
    tr = Transformer.from_crs('EPSG:4326', SR_.TM, always_xy=True)
    g = c[c.efam > 0].groupby('efam').agg(lat=('Latitude', 'median'), lon=('Longitude', 'median'),
                                          z=('depth', 'median'), n=('EventID', 'size'))
    x, y = tr.transform(g.lon.values, g.lat.values)
    g['x'], g['y'] = x, y
    return g


def stage_geom(a):
    grid = SR_.Grid('3d')
    import glob
    stas = sorted(os.path.basename(f)[:-4] for f in glob.glob('data/velocity/tt_3d/*.npz'))
    rows = []
    for E, r in fam_centroids().iterrows():
        src = np.array([r.x, r.y, np.clip(r.z, 12.5, 41.5)])
        for s in stas:
            g = SR_.station_geom(grid, s, src)
            if g is None:
                continue
            d = dict(efam=E, sta=s, x=src[0], y=src[1], z=src[2])
            for k, v in g.items():
                if isinstance(v, np.ndarray):
                    for i, cc in enumerate('xyz'):
                        d[f'{k}{cc}'] = v[i]
                else:
                    d[k] = v
            rows.append(d)
    G = pd.DataFrame(rows)
    G.to_csv(f'{OUT}/geom_eq.csv', index=False)
    print(G.groupby('efam')[['T_P', 'T_S', 'cos_S']].median().round(2).to_string())


def excl_eq(ch, t):
    """EXCL windows (sci_dvv_instrument_check) for channel names NET.STA.C (C = Z/1/2)."""
    sta, comp = ch.rsplit('.', 1)
    m = np.zeros(len(t), bool)
    for k, wins in EXCL.items():
        if k == sta or (k.startswith(sta + '..') and k[-1] == comp.replace('1', '1').replace('2', '2')[-1]) \
                or (k.startswith(sta + '..') and comp in 'NE12' and k[-1] in 'NE12' and
                    {'N': '1', 'E': '2'}.get(k[-1], k[-1]) == comp):
            for a_, b_ in wins:
                m |= (t >= SR_._ep(a_)) & (t < SR_._ep(b_))
    return m


def load_meas():
    d = pd.read_parquet(f'{OUT}/meas_cat.parquet')
    ok = ((d.phase == 'S') & (d.cc >= CCMIN_S)) | ((d.phase == 'P') & (d.cc >= CCMIN_P))
    d = d[ok & (np.abs(d.y) < 0.6)].copy()
    keep = []
    for ch, g in d.groupby('chan'):
        g = g[~excl_eq(ch, g.t.values)]
        keep.append(g)
    d = pd.concat(keep, ignore_index=True)
    br = {s: np.array([SR_._ep(b) for b in v]) for s, v in BREAKS2.items()}
    d['seg'] = [int(np.searchsorted(br[s], t, side='right')) if s in br else 0 for s, t in zip(d.sta, d.t)]
    return d


def reloc_family(d, G, huber=2.0, irls=4):
    """d rows of one family; G geometry rows (index sta).  Returns events DataFrame, residual rows."""
    d = d[d.sta.isin(G.index)].copy()
    evs, ei = np.unique(d.ev.values, return_inverse=True)
    key = d.chan.values + '|' + d.phase.values + '|' + d.seg.astype(str).values
    ks, ki = np.unique(key, return_inverse=True)
    ne, nk = len(evs), len(ks)
    n = len(d)
    A = np.zeros((n, 4 * ne + nk))
    A[np.arange(n), 4 * ei] = 1.0
    for j, cc in enumerate('xyz'):
        gS = G[f'gS{cc}'].reindex(d.sta).values
        gP = G[f'gP{cc}'].reindex(d.sta).values
        A[np.arange(n), 4 * ei + 1 + j] = np.where(d.phase.values == 'S', gS, gP)
    A[np.arange(n), 4 * ne + ki] = 1.0
    y = d.y.values
    sp = d.sta.values + '|' + d.phase.values
    sig = pd.Series(np.where(d.phase.values == 'S', 0.01, 0.02), index=d.index)
    w = 1.0 / sig.values ** 2
    reg = np.zeros(A.shape[1])
    reg[4 * ne:] = 1e-2                                        # offsets weakly damped (degenerate with event mean)
    for it in range(irls + 1):
        N = (A * w[:, None]).T @ A
        sc = np.median(np.diag(N)[np.diag(N) > 0])
        x = np.linalg.solve(N + np.diag(reg * sc + 1e-9 * sc), A.T @ (w * y))
        r = y - A @ x
        # sigma per station-phase from robust scatter
        rs = pd.Series(r).groupby(sp).transform(lambda v: 1.4826 * np.median(np.abs(v - np.median(v))) + 1e-3)
        z = np.abs(r) / rs.values
        w = (1.0 / rs.values ** 2) * np.where(z <= huber, 1.0, huber / np.maximum(z, 1e-9))
    cov = np.linalg.inv(N + np.diag(reg * sc + 1e-9 * sc))
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    ev = pd.DataFrame(dict(ev=evs, tau=x[0:4 * ne:4], dx=x[1:4 * ne:4], dy=x[2:4 * ne:4], dz=x[3:4 * ne:4],
                           dx_se=se[1:4 * ne:4], dy_se=se[2:4 * ne:4], dz_se=se[3:4 * ne:4],
                           nrow=np.bincount(ei, minlength=ne)))
    d['res'] = r
    d['sig'] = rs.values
    d['w'] = w
    d['off'] = x[4 * ne + ki]
    return ev, d


def stage_reloc(a):
    d = load_meas()
    G = pd.read_csv(f'{OUT}/geom_eq.csv')
    evr, rr = [], []
    for E, g in d.groupby('efam'):
        ok = g.groupby('ev').y.transform('size') >= 6
        g = g[ok]
        if g.ev.nunique() < 5:
            continue
        ev, res = reloc_family(g, G[G.efam == E].set_index('sta'))
        ev['efam'] = E
        evr.append(ev)
        rr.append(res)
        print(f'E{E}: {len(ev)} events, {len(res)} rows, median sigma S '
              f'{1e3 * res[res.phase == "S"].sig.median():.1f} ms, P {1e3 * res[res.phase == "P"].sig.median():.1f} ms,'
              f' dz se {1e3 * ev.dz_se.median():.0f} m', flush=True)
    E = pd.concat(evr, ignore_index=True)
    R = pd.concat(rr, ignore_index=True)
    E.to_csv(f'{OUT}/reloc_events.csv', index=False)
    R.to_parquet(f'{OUT}/reloc_resid.parquet')
    s = R.groupby(['sta', 'phase']).agg(n=('res', 'size'), sig_ms=('sig', lambda v: 1e3 * np.median(v)),
                                        nfam=('efam', 'nunique')).unstack()
    print(s.to_string())


# --------------------------------------------------------------------------------------------- Vp/Vs
def vpvs_fit(d):
    """d: rows with ev, sta, phase, y (one per ev-sta-phase; components averaged).  Lin & Shearer demeaned delays."""
    p = d.pivot_table(index=['ev', 'sta'], columns='phase', values='y', aggfunc='mean').dropna()
    if len(p) < 20:
        return np.nan, 0
    p = p[p.groupby(level=0).S.transform('size') >= 3]
    dP = p.P - p.P.groupby(level=0).transform('mean')
    dS = p.S - p.S.groupby(level=0).transform('mean')
    x, y = dP.values, dS.values
    w = np.ones(len(x))
    k = 1.75
    for _ in range(6):
        # orthogonal (total least squares) slope with weights
        xm, ym = np.average(x, weights=w), np.average(y, weights=w)
        C = np.cov(np.c_[x - xm, y - ym].T, aweights=w)
        ev_, V = np.linalg.eigh(C)
        v = V[:, 1]
        k = v[1] / v[0] if v[0] != 0 else np.nan
        r = (y - ym - k * (x - xm)) / np.sqrt(1 + k * k)
        s = 1.4826 * np.median(np.abs(r)) + 1e-6
        z = np.abs(r) / s
        w = np.where(z <= 2, 1.0, 2 / z)
    return float(k), int(p.index.get_level_values(0).nunique())


def stage_vpvs(a):
    d = load_meas()
    d['t'] = d.t.astype(float)
    rr = pd.read_parquet(f'{OUT}/reloc_resid.parquet')[['efam', 'ev', 'chan', 'phase', 'res', 'sig']]
    d = d.merge(rr, on=['efam', 'ev', 'chan', 'phase'])
    d = d[np.abs(d.res) <= 3 * d.sig]                 # relocation-consistent picks only (no cycle skips)
    rows = []
    q = {'pet': T_Q['pet'], 'fer': T_Q['fer'], 'm7': T_Q['m7']}
    t_pre = SR_._ep('2009-01-01')
    for E, g in list(d.groupby('efam')) + [('all', d.assign(ev=d.efam * 100000 + d.ev))]:
        evt = g.groupby('ev').t.first()
        base, nev = vpvs_fit(g)
        if not np.isfinite(base):
            continue
        evs = evt.index.values

        def boot(gg, nb=200):
            ev_ = gg.ev.unique()
            out = []
            for _ in range(nb):
                pick = RNG.choice(ev_, len(ev_))
                cnt = pd.Series(pick).value_counts()
                gb = gg[gg.ev.isin(cnt.index)]
                # weight duplicates by repeating
                rep = gb.ev.map(cnt).values
                gb = gb.loc[gb.index.repeat(rep)].copy()
                gb['ev'] = gb.ev.astype(str) + '_' + gb.groupby(['ev', 'chan', 'phase']).cumcount().astype(str)
                out.append(vpvs_fit(gb)[0])
            return np.nanstd(out)
        rows.append(dict(efam=E, set='all', vpvs=base, nev=nev, se=boot(g, 100)))
        for qn, tq in q.items():
            pre = g[(g.t < tq) & (g.t >= tq - 3 * YR)]
            post = g[(g.t >= tq) & (g.t < tq + 3 * YR)]
            v1, n1 = vpvs_fit(pre)
            v2, n2 = vpvs_fit(post)
            if not (np.isfinite(v1) and np.isfinite(v2)) or n1 < 5 or n2 < 5:
                continue
            # placebo: random split dates 2010-2021 with +-3 yr windows
            pl = []
            for _ in range(100):
                ds = RNG.uniform(SR_._ep('2012-01-01'), SR_._ep('2018-06-01'))
                a1 = vpvs_fit(g[(g.t < ds) & (g.t >= ds - 3 * YR)])
                a2 = vpvs_fit(g[(g.t >= ds) & (g.t < ds + 3 * YR)])
                if a1[1] >= 5 and a2[1] >= 5:
                    pl.append(a2[0] - a1[0])
            pl = np.array(pl)
            rows.append(dict(efam=E, set=f'{qn}_pm3yr', vpvs_pre=v1, n_pre=n1, vpvs_post=v2, n_post=n2,
                             d=v2 - v1, se_boot=np.hypot(boot(pre, 60), boot(post, 60)),
                             se_plac=float(np.nanstd(pl)) if len(pl) > 5 else np.nan, n_plac=len(pl)))
        print(rows[-1] if rows else None, flush=True)
    V = pd.DataFrame(rows)
    V.to_csv(f'{OUT}/vpvs.csv', index=False)
    pd.set_option('display.width', 220)
    print(V.round(4).to_string())


# --------------------------------------------------------------------------------------------- time lapse
PN = SR_.PN


def tfit(t, y, w, s=0.0, cols=None):
    """weighted LSQ of y on [1, tcols(t, s)] restricted to columns with support; returns params (9,), se, used mask."""
    X = SR_.tcols(t, s)
    use = np.array([X[:, j].std() > 0 and np.sum(X[:, j] != 0) >= 3 for j in range(X.shape[1])])
    if cols is not None:
        use &= cols
    A = np.c_[np.ones(len(t)), X[:, use]]
    N = (A * w[:, None]).T @ A
    try:
        x = np.linalg.solve(N + 1e-9 * np.eye(len(N)) * np.median(np.diag(N)), A.T @ (w * y))
    except np.linalg.LinAlgError:
        return None
    r = y - A @ x
    chi = np.sum(w * r ** 2) / max(len(y) - A.shape[1], 1)
    cov = np.linalg.pinv(N) * max(chi, 1.0)
    p = np.full(len(use), np.nan)
    se = np.full(len(use), np.nan)
    p[use] = x[1:]
    se[use] = np.sqrt(np.clip(np.diag(cov)[1:], 0, None))
    return p, se, use


def stage_lapse(a):
    R = pd.read_parquet(f'{OUT}/reloc_resid.parquet')
    R['t'] = R.t.astype(float)
    # per event x station x phase: average components
    g = R.groupby(['efam', 'ev', 'sta', 'phase']).agg(t=('t', 'first'), r=('res', 'mean'), sig=('sig', 'median'),
                                                       nc=('res', 'size')).reset_index()
    g['w'] = g.nc / g.sig ** 2
    sp = g.pivot_table(index=['efam', 'ev', 'sta'], columns='phase', values='r').dropna().reset_index()
    sp['t'] = g.groupby(['efam', 'ev']).t.first().reindex(pd.MultiIndex.from_arrays([sp.efam, sp.ev])).values
    sp['r'] = sp.S - sp.P
    sp['phase'] = 'SmP'
    sig_sp = g.pivot_table(index=['efam', 'ev', 'sta'], columns='phase', values='sig').dropna()
    sp['w'] = 1.0 / (sig_sp.S ** 2 + sig_sp.P ** 2).values
    rows_all = pd.concat([g[['efam', 'ev', 'sta', 'phase', 't', 'r', 'w']], sp[['efam', 'ev', 'sta', 'phase', 't', 'r', 'w']]])
    out = []
    want = {'pet': 2, 'fer': 3, 'm7': 4, 'ramp': 8}
    for (sta, ph), d in rows_all.groupby(['sta', 'phase']):
        if d.ev.nunique() < 15:
            continue
        # family offsets: demean residuals per family (residual is already offset-free; keep simple)
        t, y, w = d.t.values, d.r.values, d.w.values
        cols = np.zeros(9, bool)
        cols[[2, 3, 4, 8]] = True                         # steps + ramp only (too few events for annual/early)
        f = tfit(t, y, w, 0.0, cols)
        if f is None:
            continue
        p, se, use = f
        # bootstrap over events
        bs = []
        evs = d.ev.unique()
        for _ in range(200):
            pick = RNG.choice(len(d), len(d))
            fb = tfit(t[pick], y[pick], w[pick], 0.0, use)
            if fb is not None:
                bs.append(fb[0])
        bs = np.array(bs)
        # placebo shifts on pre-2021-06 data
        pre = t < SR_.T_RAMP
        pl = []
        for s in SR_.SHIFTS:
            fp = tfit(t[pre], y[pre], w[pre], s, cols)
            if fp is not None:
                pl.append(fp[0])
        pl = np.array(pl) if pl else np.full((1, 9), np.nan)
        r = dict(sta=sta, phase=ph, n=len(d), nev=len(evs), nfam=d.efam.nunique(),
                 n_post_m7=int((t >= T_Q['m7']).sum()), n_pre_pet=int((t < T_Q['pet']).sum()))
        for nm, j in want.items():
            r[nm] = p[j] * 1e3
            r[nm + '_se_boot'] = np.nanstd(bs[:, j]) * 1e3 if len(bs) else np.nan
            r[nm + '_se_plac'] = np.nanstd(pl[:, j]) * 1e3 if np.isfinite(pl[:, j]).sum() >= 4 else np.nan
        out.append(r)
    L = pd.DataFrame(out)
    L.to_csv(f'{OUT}/lapse_station.csv', index=False)
    pd.set_option('display.width', 250)
    print(L.round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True)
    a = ap.parse_args()
    {'geom': stage_geom, 'reloc': stage_reloc, 'vpvs': stage_vpvs, 'lapse': stage_lapse}[a.stage](a)


if __name__ == '__main__':
    main()
