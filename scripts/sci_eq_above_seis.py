#!/usr/bin/env python3
"""EQ-above step 3: seismicity of the 15-35 km crust around the LFEs as a stress gauge.

Catalog: NCEDC NCSS (data/sci/eq_above/raw/ncedc_cat_all.txt), all magnitudes 2008-2026.  Mechanisms: NCSN FPFIT
via ComCat (data/sci/eq_above/focal_mechanisms.csv, scripts/sci_eq_above_fm.py).
Volumes (epicentral r from 40.06 N 123.62 W):  ABOVE r<20 km, 18-26 km (the ~22 km cluster above the LFEs);
                                               MID   r<30 km, 15-26 km;  DEEP r<30 km, 26-35 km;  ALL r<30, 15-35.
1. Mc (max curvature + 0.2) per era; common Mc for rates = max over eras.
2. Rates (M >= Mc): post/pre counts in +-30/90/365 d around Petrolia 2021-12-20, Ferndale 2022-12-20, M7 2024-12-05;
   placebo = 2000 random dates 2009-2026 (windows not overlapping a real quake) -> empirical p of the log ratio.
   Era rates: 2009-01..2021-12-20 | ..2024-12-05 | after.
3. b-value (Aki-Utsu, dM bin correction, M >= Mc): +-2 yr around each quake and the three eras; bootstrap SE;
   placebo b-difference (random split dates, same window lengths).
4. Mechanisms: Michael (1984) linear stress inversion (random nodal plane per bootstrap draw; 1000 draws);
   eras pre-Petrolia | Petrolia-M7 | post-M7 and +-2 yr around M7; sigma1/sigma3 azimuth & plunge, R; angle between
   eras vs placebo splits.  Also rake classes and P-axis azimuths.
Outputs data/sci/eq_above/seis_*.csv, seis_summary.json; figures/sci_eq_above_seismicity.png
"""
import json
import sys

import numpy as np
import pandas as pd

ROOT = '/home/seismic/LFE'
OUT = f'{ROOT}/data/sci/eq_above'
sys.path.insert(0, f'{ROOT}/scripts')
from sci_eq_above_dl import read_cat  # noqa: E402

QUAKES = {'petrolia': '2021-12-20T20:10', 'ferndale': '2022-12-20T10:34', 'm7': '2024-12-05T18:44'}
VOLS = {'ABOVE': (20, 18, 26, None), 'MID': (30, 15, 26, None), 'DEEP': (30, 26, 35, None), 'ALL': (30, 15, 35, None),
        'ABOVE_m17': (20, 18, 26, 1.7), 'DEEP_m17': (30, 26, 35, 1.7), 'ALL_m17': (30, 15, 35, 1.7)}
T0, T1 = pd.Timestamp('2009-01-01'), pd.Timestamp('2026-11-20')
RNG = np.random.default_rng(7)


def mc_maxc(m, dm=0.1):
    b = np.round(m / dm) * dm
    v, n = np.unique(b, return_counts=True)
    return float(v[np.argmax(n)]) + 0.2


def bval(m, mc, dm=0.01):
    m = m[m >= mc - 1e-9]
    if len(m) < 10:
        return np.nan, np.nan, len(m)
    b = np.log10(np.e) / (m.mean() - (mc - dm / 2))
    bs = [np.log10(np.e) / (x.mean() - (mc - dm / 2)) for x in (RNG.choice(m, len(m)) for _ in range(500))]
    return float(b), float(np.std(bs)), len(m)


def sdr_to_ns(strike, dip, rake):
    f, d, l = np.radians(strike), np.radians(dip), np.radians(rake)
    n = np.c_[-np.sin(d) * np.sin(f), np.sin(d) * np.cos(f), -np.cos(d)]
    s = np.c_[np.cos(l) * np.cos(f) + np.cos(d) * np.sin(l) * np.sin(f),
              np.cos(l) * np.sin(f) - np.cos(d) * np.sin(l) * np.cos(f), -np.sin(l) * np.sin(d)]
    return n, s


def michael(n, s):
    """linear stress inversion (Michael 1984); n, s (k,3) in N,E,Down. Returns 3x3 deviatoric stress (tension +)."""
    A, b = [], []
    for (n1, n2, n3), sl in zip(n, s):
        A.append([n1 - n1 ** 3 + n1 * n3 ** 2, n2 - 2 * n1 ** 2 * n2, n3 - 2 * n1 ** 2 * n3,
                  -n1 * n2 ** 2 + n1 * n3 ** 2, -2 * n1 * n2 * n3])
        A.append([-n2 * n1 ** 2 + n2 * n3 ** 2, n1 - 2 * n1 * n2 ** 2, -2 * n1 * n2 * n3,
                  n2 - n2 ** 3 + n2 * n3 ** 2, n3 - 2 * n2 ** 2 * n3])
        A.append([-n3 * n1 ** 2 - n3 + n3 ** 3, -2 * n1 * n2 * n3, n1 - 2 * n1 * n3 ** 2,
                  -n3 * n2 ** 2 - n3 + n3 ** 3, n2 - 2 * n2 * n3 ** 2])
        b.extend(sl)
    x = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
    return np.array([[x[0], x[1], x[2]], [x[1], x[3], x[4]], [x[2], x[4], -x[0] - x[3]]])


def principal(T):
    """-> sigma1 (most compressive) and sigma3 unit vectors, R = (s1-s2)/(s1-s3), tension positive."""
    w, v = np.linalg.eigh(T)                           # ascending: w[0] most compressive (tension +)
    s1, s2, s3 = w[0], w[1], w[2]
    R = (s1 - s2) / (s1 - s3)
    return v[:, 0], v[:, 2], R


def azpl(v):
    v = v if v[2] >= 0 else -v                         # down-pointing
    az = np.degrees(np.arctan2(v[1], v[0])) % 360
    pl = np.degrees(np.arcsin(np.clip(v[2], -1, 1)))
    return az, pl


def ang(u, v):
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(u, v)), 0, 1))))


def stress_boot(f, nb=1000):
    n1, s1 = sdr_to_ns(f.strike1.values, f.dip1.values, f.rake1.values)
    n2, s2 = sdr_to_ns(f.strike2.values, f.dip2.values, f.rake2.values)
    out = []
    for b in range(nb):
        k = RNG.integers(0, len(f), len(f)) if b else np.arange(len(f))
        pick = RNG.random(len(f)) < 0.5
        n = np.where(pick[:, None], n1, n2)[k]
        s = np.where(pick[:, None], s1, s2)[k]
        out.append(principal(michael(n, s)))
    return out


def test_michael():
    """synthetic check: sigma1 N-S horizontal, sigma3 E-W -> recovered?"""
    T = np.diag([-1.0, 0.0, 1.0])                      # N compression, Down tension... rotate: s1=N, s3=E
    T = np.diag([-1.0, 1.0, 0.0])
    ns, ss = [], []
    for _ in range(200):
        n = RNG.normal(size=3)
        n /= np.linalg.norm(n)
        t = T @ n
        tau = t - (t @ n) * n
        ns.append(n)
        ss.append(tau / np.linalg.norm(tau))
    s1, s3, R = principal(michael(np.array(ns), np.array(ss)))
    return azpl(s1), azpl(s3), R


def main():
    c = read_cat()
    c['t'] = pd.to_datetime(c.Time)
    c = c[(c.t >= T0) & (c.t < T1)]
    fm = pd.read_csv(f'{OUT}/focal_mechanisms.csv', parse_dates=['time'])
    qt = {k: pd.Timestamp(v) for k, v in QUAKES.items()}
    eras = [('pre_petrolia', T0, qt['petrolia']), ('petrolia_m7', qt['petrolia'], qt['m7']), ('post_m7', qt['m7'], T1)]
    summ = {'michael_selftest': test_michael()}
    rate_rows, b_rows, st_rows = [], [], []
    for vn, (R, z0, z1, mforce) in VOLS.items():
        s = c[(c.r < R) & (c.depth >= z0) & (c.depth < z1)]
        mcs = {en: mc_maxc(s[(s.t >= a) & (s.t < b)].Magnitude.values) for en, a, b in eras}
        mc = max(max(mcs.values()), 1.5) if mforce is None else mforce
        summ[f'{vn}_mc_eras'] = mcs
        summ[f'{vn}_mc'] = mc
        sc = s[s.Magnitude >= mc - 1e-9]
        tt = sc.t.values.astype('datetime64[s]').astype(float)
        # declustered: first event of each episode (no earlier event within 3 d and 10 km)
        xy = np.c_[(sc.Latitude.values - 40.06) * 111.2, (sc.Longitude.values + 123.62) * 85.1]
        ind = np.array([not np.any((tt[:i] > tt[i] - 3 * 86400) &
                                   (np.hypot(*(xy[:i] - xy[i]).T) < 10)) for i in range(len(tt))], bool)
        ttd = tt[ind]
        yrs = (T1 - T0).days / 365.25
        for en, a, b in eras:
            n = ((sc.t >= a) & (sc.t < b)).sum()
            dur = (b - a).days / 365.25
            rate_rows.append(dict(vol=vn, kind='era', name=en, mc=mc, n=int(n), years=dur, rate=n / dur))
            bb, be, nb = bval(s[(s.t >= a) & (s.t < b)].Magnitude.values, mc)
            b_rows.append(dict(vol=vn, kind='era', name=en, mc=mc, b=bb, b_se=be, n=nb))
        # quake windows + placebo
        for W in (30, 90, 365):
            ws = W * 86400.0
            lo = T0.value / 1e9 + ws
            hi = T1.value / 1e9 - ws
            qs = np.array([v.value / 1e9 for v in qt.values()])
            pl = []
            while len(pl) < 2000:
                d = RNG.uniform(lo, hi)
                if np.all(np.abs(d - qs) > 2 * ws):
                    pl.append(d)
            pl = np.array(pl)

            def lr(d, x):
                post = ((x >= d) & (x < d + ws)).sum()
                pre = ((x >= d - ws) & (x < d)).sum()
                return post, pre, np.log((post + 0.5) / (pre + 0.5))
            for lab, x in (('all', tt), ('declust', ttd)):
                plv = np.array([lr(d, x)[2] for d in pl])
                for qn, q in qt.items():
                    post, pre, v = lr(q.value / 1e9, x)
                    rate_rows.append(dict(vol=vn, kind=f'win{W}d_{lab}', name=qn, mc=mc, n_pre=int(pre),
                                          n_post=int(post), ratio=(post + 0.5) / (pre + 0.5),
                                          p_high=float((plv >= v).mean()), p_low=float((plv <= v).mean()),
                                          plac_sd=float(plv.std())))
        # b around quakes +-2 yr, placebo
        for qn, q in qt.items():
            pre = s[(s.t >= q - pd.Timedelta(days=730)) & (s.t < q)].Magnitude.values
            post = s[(s.t >= q) & (s.t < q + pd.Timedelta(days=730))].Magnitude.values
            b1, e1, n1 = bval(pre, mc)
            b2, e2, n2 = bval(post, mc)
            pdl = []
            for _ in range(300):
                d = pd.Timestamp(RNG.uniform((T0 + pd.Timedelta(days=730)).value, (T1 - pd.Timedelta(days=730)).value))
                x1 = s[(s.t >= d - pd.Timedelta(days=730)) & (s.t < d)].Magnitude.values
                x2 = s[(s.t >= d) & (s.t < d + pd.Timedelta(days=730))].Magnitude.values
                if (x1 >= mc).sum() >= 10 and (x2 >= mc).sum() >= 10:
                    pdl.append(bval(x2, mc)[0] - bval(x1, mc)[0])
            pdl = np.array(pdl)
            b_rows.append(dict(vol=vn, kind='pm2yr', name=qn, mc=mc, b_pre=b1, b_pre_se=e1, n_pre=n1, b_post=b2,
                               b_post_se=e2, n_post=n2, db=b2 - b1, db_plac_sd=float(np.nanstd(pdl)) if len(pdl) else np.nan,
                               p_two=float((np.abs(pdl) >= abs(b2 - b1)).mean()) if len(pdl) else np.nan))
    # mechanisms
    fm = fm.dropna(subset=['strike1', 'dip1', 'rake1', 'strike2', 'dip2', 'rake2']).reset_index(drop=True)
    # P and T axes from nodal plane 1 (ComCat FPFIT products carry no axis fields): P = (n - s)/sqrt2, T = (n + s)/sqrt2
    n_, s_ = sdr_to_ns(fm.strike1.values, fm.dip1.values, fm.rake1.values)
    for nm, v in (('p', (n_ - s_) / np.sqrt(2)), ('t', (n_ + s_) / np.sqrt(2))):
        ap_ = np.array([azpl(x) for x in v])
        fm[f'{nm}_az'], fm[f'{nm}_pl'] = ap_[:, 0], ap_[:, 1]
    fm['rclass'] = pd.cut(((fm.rake1 + 180) % 360) - 180, [-181, -135, -45, 45, 135, 181],
                          labels=['SS', 'N', 'SS2', 'R', 'SS3']).astype(str).replace({'SS2': 'SS', 'SS3': 'SS'})
    sets = {'all': fm}
    for en, a, b in eras:
        sets[en] = fm[(fm.time >= a) & (fm.time < b)]
    q = qt['m7']
    sets['m7_pre2yr'] = fm[(fm.time >= q - pd.Timedelta(days=730)) & (fm.time < q)]
    sets['m7_post2yr'] = fm[(fm.time >= q) & (fm.time < q + pd.Timedelta(days=730))]
    sets['ABOVE'] = fm[(fm.r_km < 20) & (fm.depth >= 18) & (fm.depth < 26)]
    sets['DEEP'] = fm[(fm.depth >= 26)]
    res = {}
    for k, f in sets.items():
        if len(f) < 8:
            continue
        bs = stress_boot(f, 500)
        s1, s3, R = bs[0]
        a1 = [ang(s1, x[0]) for x in bs[1:]]
        res[k] = bs
        az1, pl1 = azpl(s1)
        az3, pl3 = azpl(s3)
        st_rows.append(dict(set=k, n=len(f), s1_az=az1, s1_pl=pl1, s3_az=az3, s3_pl=pl3, R=R,
                            s1_conf95_deg=float(np.percentile(a1, 95)),
                            frac_SS=float((f.rclass == 'SS').mean()), frac_R=float((f.rclass == 'R').mean()),
                            frac_N=float((f.rclass == 'N').mean()),
                            paxis_az_circmean=float(np.degrees(np.angle(np.mean(np.exp(2j * np.radians(f.p_az))))) / 2 % 180)))
    # era-difference vs placebo splits (all mechanisms, split at random dates, same counts on either side)
    def s1_of(f):
        return stress_boot(f, 1)[0][0]
    comp = []
    for qn in ('petrolia', 'm7'):
        qd = qt[qn]
        a_ = fm[(fm.time >= qd - pd.Timedelta(days=1460)) & (fm.time < qd)]
        b_ = fm[(fm.time >= qd) & (fm.time < qd + pd.Timedelta(days=1460))]
        if len(a_) < 8 or len(b_) < 8:
            continue
        obs = ang(s1_of(a_), s1_of(b_))
        pl = []
        for _ in range(300):
            d = pd.Timestamp(RNG.uniform(pd.Timestamp('2012-01-01').value, pd.Timestamp('2021-06-01').value))
            x = fm[(fm.time >= d - pd.Timedelta(days=1460)) & (fm.time < d)]
            y = fm[(fm.time >= d) & (fm.time < d + pd.Timedelta(days=1460))]
            if len(x) >= 8 and len(y) >= 8:
                pl.append(ang(s1_of(x), s1_of(y)))
        comp.append(dict(quake=qn, n_pre=len(a_), n_post=len(b_), s1_angle_deg=obs, plac_med=float(np.median(pl)),
                         plac_p=float((np.array(pl) >= obs).mean()), n_plac=len(pl)))
    pd.DataFrame(rate_rows).to_csv(f'{OUT}/seis_rates.csv', index=False)
    pd.DataFrame(b_rows).to_csv(f'{OUT}/seis_bvalue.csv', index=False)
    pd.DataFrame(st_rows).to_csv(f'{OUT}/seis_stress.csv', index=False)
    pd.DataFrame(comp).to_csv(f'{OUT}/seis_stress_change.csv', index=False)
    json.dump(summ, open(f'{OUT}/seis_summary.json', 'w'), indent=1, default=float)
    pd.set_option('display.width', 250)
    print(pd.DataFrame(rate_rows).round(3).to_string())
    print(pd.DataFrame(b_rows).round(3).to_string())
    print(pd.DataFrame(st_rows).round(2).to_string())
    print(pd.DataFrame(comp).round(2).to_string())
    print(summ)
    figs(c, fm, qt)


def figs(c, fm, qt):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    s = c[(c.r < 30) & (c.depth >= 15) & (c.depth < 35)]
    ax[0].scatter(s.t, s.depth, s=4 * 2 ** s.Magnitude, c=s.r, cmap='viridis', lw=0)
    ax[0].invert_yaxis()
    ax[0].set_ylabel('depth (km)')
    ax[0].set_title('NCSS earthquakes r<30 km of the LFEs, 15-35 km (colour = epicentral distance)')
    for vn, col in (('ABOVE', 'C3'), ('DEEP', 'C0'), ('ALL', 'k')):
        R, z0, z1, _ = VOLS[vn]
        x = c[(c.r < R) & (c.depth >= z0) & (c.depth < z1) & (c.Magnitude >= 1.8)].sort_values('t')
        ax[1].step(x.t, np.arange(len(x)), where='post', color=col, label=f'{vn} M>=1.8')
    ax[1].legend()
    ax[1].set_ylabel('cumulative count')
    ax[2].scatter(fm.time, fm.p_az % 180, s=10, c=fm.depth, cmap='plasma')
    ax[2].set_ylabel('P-axis azimuth (deg, mod 180)')
    for a in ax:
        for v in qt.values():
            a.axvline(v, color='r', lw=0.8, ls='--')
    fig.tight_layout()
    fig.savefig(f'{ROOT}/figures/sci_eq_above_seismicity.png', dpi=130)


if __name__ == '__main__':
    main()
