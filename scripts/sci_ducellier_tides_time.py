#!/usr/bin/env python
"""Tidal sensitivity of the Ducellier main-band LFE families through time, from our matched-filter scan
(sci_ducellier_mf.py tag 'scan'). Revised 2026-11-25 after independent code review.

Units: filtered detections (cc*nchannel >= thr_perm), subduction families, declustered per family (first
detection after a >1 h gap), as in sci_ducellier_compare.py.
Estimator (per constituent): signed, day-block-debiased squared amplitude
    a2 = (|sum S|^2 - sum_b |S_b|^2) / (n^2 - sum_b n_b^2),  S = exp(i*phase),  b = UTC day blocks;
    reported amplitude = sign(a2) * 2*sqrt(|a2|)   (no clipping at 0, so nulls are unbiased).
Per-family version: a2 computed per family and combined with weights (n_f^2 - sum n_fb^2), so families with
different tidal phases cannot cancel (phase-free), alongside the network (single-phase) version.
Uncertainty: block bootstrap over 'episode blocks' (tremor episodes; outside episodes, clusters of scanned days
separated by >7 d), 1000 draws; contrasts use the bootstrap of the signed difference.
Off-tide null: the same estimator at 50 non-tidal periods (10.6-11.7 h and 13.0-14.2 h) -> 95% null band.
Leakage controls: M2, N2, S2, K1, O1, S1 all reported (the scan's every-14th-day days alias S2->M2 and K1->O1;
they are reported separately). Validation: the same estimator on the published catalog vs our scan on the
same days and families (2004-2011) before any statement about change through time.
Epochs: per year; ETS phase (inside PNSN tremor episodes / 0-30 d after / >30 d after); 60 d before vs after
each quake. Caveat: scanned days are not uniform in time.
Outputs data/sci/ducellier/tides_time.csv, tides_time_contrasts.csv, tides_time_validation.csv;
figures/sci_ducellier_tides_time.png"""
import glob, sys
import numpy as np, pandas as pd
sys.path.insert(0, '/home/seismic/LFE/scripts')
import sci_ducellier_compare as C   # decluster(), load_duc(), TREF (main() is guarded)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

DUC = '/data/wsd02/maleen_data/ducellier'; OUT = '/home/seismic/LFE/data/sci/ducellier'; FIG = '/home/seismic/LFE/figures'
PER_H = {'M2': 12.4206012, 'N2': 12.65834751, 'S2': 12.0, 'K1': 23.93447213, 'O1': 25.81933871, 'S1': 24.0}
OFF = np.r_[np.linspace(10.6, 11.7, 25), np.linspace(13.0, 14.2, 25)]
QK = {'M6.8_2014': '2014-03-10 05:18', 'Petrolia_2021': '2021-12-20 20:10', 'Ferndale_2022': '2022-12-20 10:34', 'M7_2024': '2024-12-05 18:44'}
rng = np.random.default_rng(7)
NB = 1000


def load_scan():
    fs = []
    for f in sorted(glob.glob(f'{DUC}/mf/scan/det_*.csv')):
        try:
            d = pd.read_csv(f, dtype={'template': str}, usecols=['template', 'time', 'cc', 'nchannel', 'thr_perm'])
        except (pd.errors.EmptyDataError, ValueError):
            continue
        d = d[(d.thr_perm > 0) & (d.cc * d.nchannel >= d.thr_perm) & ~d.template.str.startswith(('080326', '080328'))]
        if len(d):
            fs.append(d)
    D = pd.concat(fs, ignore_index=True)
    D['t'] = pd.to_datetime(D.time); D['fam'] = D.template; D['tsec'] = (D.t - C.TREF).dt.total_seconds()
    return C.decluster(D, 'fam', 3600.0)


def parts(tsec, per_h, fam=None):
    """per (family) the resultant pieces: |sum S|^2, sum_b |S_b|^2, n^2, sum_b n_b^2 (b = day blocks)."""
    phi = 2 * np.pi * np.mod(tsec / 3600.0 / per_h, 1.0)
    S = np.exp(1j * phi); day = np.floor(tsec / 86400).astype(np.int64)
    key = day if fam is None else pd.factorize(pd.Series(fam).astype(str) + '_' + pd.Series(day).astype(str))[0]
    _, blk = np.unique(key, return_inverse=True)
    Sb = np.bincount(blk, S.real) + 1j * np.bincount(blk, S.imag); nb = np.bincount(blk)
    if fam is None:
        return abs(S.sum()) ** 2 - (abs(Sb) ** 2).sum(), len(S) ** 2 - (nb ** 2).sum()
    fid = pd.factorize(pd.Series(fam).astype(str))[0]
    fb = np.zeros(len(nb), int); fb[blk] = fid
    Sf = np.bincount(fid, S.real) + 1j * np.bincount(fid, S.imag); nf = np.bincount(fid)
    num = abs(Sf) ** 2 - np.bincount(fb, abs(Sb) ** 2, len(nf)); den = nf ** 2 - np.bincount(fb, nb ** 2, len(nf))
    return num.sum(), den.sum()


def amp_signed(num, den):
    a2 = num / den if den > 0 else np.nan
    return np.sign(a2) * 2 * np.sqrt(abs(a2))


def blocks(sub):
    """episode blocks: tremor episodes, else clusters of scanned days separated by >7 d."""
    lab = np.full(len(sub), -1, int); tt = sub.t.values
    for k, (s, e) in enumerate(zip(EP.s.values, EP.e.values)):
        lab[(tt >= s) & (tt <= e + np.timedelta64(30, 'D'))] = k
    rest = lab < 0
    if rest.any():
        d = np.sort(np.unique(sub.t[rest].dt.normalize().values))
        cl = np.r_[0, np.cumsum(np.diff(d) > np.timedelta64(7, 'D'))]
        m = dict(zip(d, cl + 100000))
        lab[rest] = sub.t[rest].dt.normalize().map(m).values
    return lab


def estimate(sub, const, perfam=False, boot=True):
    ts = sub.tsec.to_numpy(); fam = sub.fam.to_numpy() if perfam else None
    num, den = parts(ts, PER_H[const], fam); a = amp_signed(num, den)
    out = dict(const=const, perfam=perfam, n=len(sub), ndays=sub.t.dt.normalize().nunique(), a=a)
    if boot:
        lab = blocks(sub); ub, inv = np.unique(lab, return_inverse=True)
        # pre-compute per-block pieces then resample blocks
        P = [parts(ts[inv == k], PER_H[const], None if fam is None else fam[inv == k]) for k in range(len(ub))]
        phi = 2 * np.pi * np.mod(ts / 3600.0 / PER_H[const], 1.0); S = np.exp(1j * phi)
        if perfam:
            fid = pd.factorize(pd.Series(fam).astype(str))[0]; nf = fid.max() + 1
            R = np.zeros((len(ub), nf), complex); np.add.at(R, (inv, fid), S)
            Q = np.array([p[0] for p in P]); Dd = np.array([p[1] for p in P])
            NBF = np.zeros((len(ub), nf)); np.add.at(NBF, (inv, fid), 1)
        else:
            R = np.bincount(inv, S.real) + 1j * np.bincount(inv, S.imag)
        within = np.array([p for p in P])  # (num_within, den_within) per block, num includes block self-terms
        bs = []
        for _ in range(NB):
            w = rng.poisson(1.0, len(ub)).astype(float)
            if perfam:
                Rf = w @ R; nfw = w @ NBF
                # |sum_k w_k R_kf|^2 minus each resampled block's own day-block self terms
                num_b = (abs(Rf) ** 2).sum() - (w ** 2) @ (abs(R) ** 2).sum(1) + w @ Q   # remove copy-copy self terms, keep within-block cross-day signal
                den_b = (nfw ** 2).sum() - (w ** 2) @ (NBF ** 2).sum(1) + w @ Dd
            else:
                Rw = w @ R; nb_ = np.bincount(inv).astype(float)
                num_b = abs(Rw) ** 2 - (w ** 2) @ (abs(R) ** 2) + w @ within[:, 0]
                den_b = (w @ nb_) ** 2 - (w ** 2) @ nb_ ** 2 + w @ within[:, 1]
            bs.append(amp_signed(num_b, den_b))
        bs = np.array(bs)
        out.update(a_lo=np.nanpercentile(bs, 2.5), a_hi=np.nanpercentile(bs, 97.5), a_sd=np.nanstd(bs), _boot=bs)
    if const == 'M2':   # off-tide null band
        nz = [amp_signed(*parts(ts, p, fam)) for p in OFF]
        out.update(null_p95=np.nanpercentile(np.abs(nz), 95))
    phi = 2 * np.pi * np.mod(ts / 3600.0 / PER_H[const], 1.0)
    out['phase_deg'] = float(np.rad2deg(np.angle(np.exp(1j * phi).sum())) % 360)
    return out


EP = pd.read_csv(f'{OUT}/tremor_episodes.csv'); EP['s'] = pd.to_datetime(EP['start']); EP['e'] = pd.to_datetime(EP['end'])


def ets_phase(t):
    lab = np.full(len(t), 'inter', dtype=object); tt = t.values
    for s, e in zip(EP.s.values, EP.e.values):
        post = (tt > e) & (tt <= e + np.timedelta64(30, 'D')) & (lab != 'ETS')
        lab[post] = 'post'; lab[(tt >= s) & (tt <= e)] = 'ETS'
    return lab


if __name__ == '__main__':
    U = load_scan(); U['ets'] = ets_phase(U.t)
    every14 = set(pd.to_datetime(pd.read_csv(f'{OUT}/scan_daylist.txt', header=None)[0]))  # all scanned days
    print('declustered units', len(U), 'days', U.t.dt.normalize().nunique())
    rows, boots = [], {}
    def add(name, sub, consts=('M2', 'N2', 'S2', 'K1', 'O1', 'S1'), min_n=100):
        if len(sub) < min_n:
            rows.append(dict(epoch=name, const='M2', n=len(sub), note='too few units')); return
        for c in consts:
            for pf in (False, True):
                r = estimate(sub, c, pf); boots[(name, c, pf)] = r.pop('_boot', None); r['epoch'] = name; rows.append(r)
    add('all', U)
    for k in ('ETS', 'post', 'inter'):
        add(f'ETS phase: {k}', U[U.ets == k])
    for y, g in U.groupby(U.t.dt.year):
        add(f'year {y}', g, consts=('M2', 'O1'))
    for q, tq in QK.items():
        tq = pd.Timestamp(tq)
        add(f'{q} before 60d', U[(U.t >= tq - pd.Timedelta('60D')) & (U.t < tq)], consts=('M2',), min_n=50)
        add(f'{q} after 60d', U[(U.t > tq) & (U.t <= tq + pd.Timedelta('60D'))], consts=('M2',), min_n=50)
    T = pd.DataFrame(rows); T.to_csv(f'{OUT}/tides_time.csv', index=False)
    show = T[(T.const.isin(['M2', 'O1', 'K1', 'S1']))]
    print(show[['epoch', 'const', 'perfam', 'n', 'a', 'a_lo', 'a_hi', 'null_p95', 'phase_deg']].round(3).to_string())

    # contrasts: bootstrap of signed differences (independent groups)
    cr = []
    def contrast(a, b, c='M2'):
        for pf in (False, True):
            A, B = boots.get((a, c, pf)), boots.get((b, c, pf))
            if A is None or B is None:
                continue
            ra = T[(T.epoch == a) & (T.const == c) & (T.perfam == pf)].a.iloc[0]; rb = T[(T.epoch == b) & (T.const == c) & (T.perfam == pf)].a.iloc[0]
            d = B - A
            cr.append(dict(contrast=f'{b} minus {a}', const=c, perfam=pf, diff=rb - ra, lo=np.nanpercentile(d, 2.5), hi=np.nanpercentile(d, 97.5),
                           p_two=float(2 * min((d <= 0).mean(), (d >= 0).mean()))))
    contrast('ETS phase: inter', 'ETS phase: ETS'); contrast('ETS phase: inter', 'ETS phase: post')
    CR = pd.DataFrame(cr); CR.to_csv(f'{OUT}/tides_time_contrasts.csv', index=False)
    print(CR.round(3).to_string())

    # validation: published catalog vs our scan, same days and families (2004-2011)
    _, cat = C.load_duc(); cat = cat[cat['sub']].copy(); cat['fam'] = cat.template
    cat = C.decluster(cat, 'fam', 3600.0)
    days = sorted(set(U.t.dt.normalize()) & set(cat.t.dt.normalize()))
    fams = sorted(set(U.fam) & set(cat.fam))
    V = []
    for nm, X in (('catalog', cat), ('our scan', U)):
        X = X[X.t.dt.normalize().isin(days) & X.fam.isin(fams)]
        X = X.assign(ets=ets_phase(X.t))
        for grp, sub in (('all common days', X), ('common ETS days', X[X.ets == 'ETS'])):
            if len(sub) < 100:
                continue
            for pf in (False, True):
                r = estimate(sub, 'M2', pf); r.pop('_boot', None); r.update(source=nm, group=grp); V.append(r)
    V = pd.DataFrame(V); V.to_csv(f'{OUT}/tides_time_validation.csv', index=False)
    print('common days', len(days), 'families', len(fams)); print(V[['source', 'group', 'perfam', 'n', 'a', 'a_lo', 'a_hi', 'null_p95']].round(3).to_string())

    yy = T[T.epoch.str.startswith('year') & (T.const == 'M2') & (T.perfam == True) & T.a.notna()]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    x = yy.epoch.str[5:].astype(int)
    ax[0].errorbar(x, yy.a, [yy.a - yy.a_lo, yy.a_hi - yy.a], fmt='o-', label='per-family M2 (signed)')
    ax[0].plot(x, yy.null_p95, 'k:', label='off-tide 95% |null|'); ax[0].plot(x, -yy.null_p95, 'k:')
    ax[0].axhline(0, color='0.5', lw=.6); ax[0].set_ylabel('M2 modulation amplitude'); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[0].set_title('Main-band LFEs (our scan): M2 tidal sensitivity per year', fontsize=10)
    v = V[V.perfam == True]
    ax[1].errorbar(np.arange(len(v)), v.a, [v.a - v.a_lo, v.a_hi - v.a], fmt='o'); ax[1].set_xticks(np.arange(len(v)))
    ax[1].set_xticklabels([f'{s}\n{g}' for s, g in zip(v.source, v.group)], fontsize=7); ax[1].axhline(0, color='0.5', lw=.6)
    ax[1].set_title('Validation: catalog vs our scan, same days and families', fontsize=10); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f'{FIG}/sci_ducellier_tides_time.png', dpi=130)
