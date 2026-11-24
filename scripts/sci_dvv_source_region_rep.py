#!/usr/bin/env python3
"""Source-region dv/v: ordinary repeating earthquakes as the RECEIVER/SHALLOW reference for S-P and splitting.

Input: data/sci/dvv_deep_shallow/rep/{clusters.csv, wf/{evid}.mseed} (NCSN M>=1.5 2015-2026, CC>=0.95 clusters,
9 stations: NC.KMR NC.KMPB PB.B045 B046 B047 B049 B932 BK.BRIC BK.HAYF; 100 Hz, origin-10 .. +60 s).

stage 'meas'  per cluster (cl95 >= 0) and station:
   * 2-8 Hz; components Z, H1, H2 (PB EH1/EH2 rotated to N/E with the epoch azimuth; NC/BK HHN/HHE).
   * initial alignment of each event on the cluster medoid: CC lag (+-1 s) over [P-1, S+5] s, 3-comp mean.
   * reference = mean of aligned events; P / S picked on the reference (STA/LTA near the 1D-predicted times).
   * per event: dP = CC lag vs reference on Z in [P-0.3, P+1.2] s; dS = CC lag on N and E in [S-0.4, S+1.6] s
     (CC-weighted mean of the two); 4x upsampled (400 Hz) parabolic lags.  dSP = dS - dP (origin/clock free).
   * splitting: event N/E vs reference rotated to fast/slow at the station reference phi (data/sci/sws/
     reference_splitting.csv, b28): ddt = lag(slow) - lag(fast)  (fast-minus-slow delay change, s).
   -> data/sci/dvv_source_region/rep/rep_event_meas.parquet
stage 'fit'   per station and depth class (deep: 15-30 km [median 19]; shallow: < 10 km):
   y = c_cluster + annual + steps (Petrolia, Ferndale, M7) + M7 60-d window + ramp from 2021-06, robust WLS
   (CC weights, Huber); D = mean 2025 - mean 2021.  Bootstrap over clusters (B=200); placebo: 40 random fake M7 dates
   2016-2021 on pre-2021-06 data (M7 column only).  Observables: dSP, dS_rel (dS minus the event's station-mean dS),
   dP_rel, ddt.  -> rep/rep_station_fit.csv
Usage: nice -n 10 python scripts/sci_dvv_source_region_rep.py meas|fit --procs 10
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt

ROOT = '/home/seismic/LFE'
REP = f'{ROOT}/data/sci/dvv_deep_shallow/rep'
OUT = f'{ROOT}/data/sci/dvv_source_region/rep'
sys.path.insert(0, f'{ROOT}/scripts')
STAS = {'NC.KMR': (40.202175, -123.708633), 'NC.KMPB': (40.417194, -124.120758), 'PB.B045': (40.436001, -123.996498),
        'PB.B046': (40.321972, -124.284302), 'PB.B047': (40.249933, -124.169848), 'PB.B049': (40.237499, -123.819702),
        'PB.B932': (40.275002, -124.221199), 'BK.BRIC': (40.11223, -123.8965), 'BK.HAYF': (40.5227, -123.23198)}
SL = list(STAS)
SR = 100.0
NS = 7000
UP = 4
VP, VS = 6.0, 3.5


def dist_km(lat1, lon1, lat2, lon2):
    return np.hypot((lat1 - lat2) * 111.2, (lon1 - lon2) * 111.2 * np.cos(np.radians(40.3)))


_AZ = None


def azimuths():
    """PB EH1/EH2 azimuth epochs from the station metadata (via sci_sws.read_meta)."""
    global _AZ
    if _AZ is None:
        from sci_sws import read_meta
        m = read_meta()
        m = m[m.net.isin(['PB', 'NC', 'BK']) & m.cha.str.match(r'^(EH[12]|HH[NE])$')]
        m = m.assign(ns=m.net + '.' + m.sta)
        _AZ = m[m.ns.isin(SL)][['ns', 'cha', 'az', 'start', 'end']].reset_index(drop=True)
    return _AZ


def az_at(ns, cha, t):
    A = azimuths()
    g = A[(A.ns == ns) & (A.cha == cha) & (A.start <= t) & (A.end > t)]
    if len(g) == 0:
        return {'1': 0.0, 'N': 0.0, '2': 90.0, 'E': 90.0}[cha[-1]]
    return float(g.az.iloc[0])


def load_event(evid, t0):
    """-> (nsta, 3, NS) [Z, N, E] (NaN if missing), raw counts (gain irrelevant for lags/splitting shape)."""
    from obspy import read
    A = np.full((len(SL), 3, NS), np.nan, np.float32)
    f = f'{REP}/wf/{evid}.mseed'
    if not os.path.exists(f):
        return A
    try:
        st = read(f)
    except Exception:                                  # noqa: BLE001
        return A
    st0 = min(tr.stats.starttime for tr in st)
    tt = pd.Timestamp(str(t0))
    for s_i, sta in enumerate(SL):
        net, code = sta.split('.')
        comps = {}
        for tr in st.select(network=net, station=code):
            if tr.stats.sampling_rate != SR:
                continue
            x = tr.data.astype(float)
            if len(x) < NS * 0.9 or np.std(x) == 0:
                continue
            k0 = int(round((tr.stats.starttime - st0) * SR))
            y = np.zeros(NS)
            n = min(NS - k0, len(x))
            if k0 < 0 or n < NS * 0.9:
                continue
            y[k0:k0 + n] = x[:n] - x[:n].mean()
            comps[tr.stats.channel] = y
        z = [c for c in comps if c[-1] == 'Z']
        if z:
            A[s_i, 0] = comps[z[0]]
        h = sorted(c for c in comps if c[-1] in '12NE')
        if len(h) == 2:
            a1, a2 = az_at(sta, h[0], tt), az_at(sta, h[1], tt)
            x1, x2 = comps[h[0]], comps[h[1]]
            # x = x1 * u(a1) + x2 * u(a2) with u(a) = (N, E) = (cos a, sin a)
            M = np.array([[np.cos(np.radians(a1)), np.cos(np.radians(a2))],
                          [np.sin(np.radians(a1)), np.sin(np.radians(a2))]])
            NE = M @ np.vstack([x1, x2])
            A[s_i, 1], A[s_i, 2] = NE[0], NE[1]
    return A


def subpk(y, k):
    if 0 < k < len(y) - 1:
        y0, y1, y2 = y[k - 1:k + 2]
        den = y0 - 2 * y1 + y2
        f = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        return float(np.clip(f, -0.5, 0.5)), float(y1 - 0.25 * (y0 - y2) * f)
    return 0.0, float(y[k])


def lag1(ref, x, i0, i1, ml):
    """lag (samples, + = x late) of x vs ref in window [i0, i1), max |lag| ml; -> lag, cc."""
    r = ref[i0:i1] - ref[i0:i1].mean()
    r = r / (np.linalg.norm(r) + 1e-20)
    L = np.arange(-ml, ml + 1)
    c = np.empty(len(L))
    for j, l in enumerate(L):
        s = x[i0 + l:i1 + l]
        s = s - s.mean()
        c[j] = s @ r / (np.linalg.norm(s) + 1e-20)
    k = int(np.argmax(c))
    f, v = subpk(c, k)
    if k in (0, len(L) - 1):
        return np.nan, -1.0
    return L[k] + f, v


def stalta_pick(x, sr, t_pred, half, sta=0.3, lta=2.0):
    e = x ** 2
    ns, nl = int(sta * sr), int(lta * sr)
    c = np.concatenate([[0.0], np.cumsum(e)])
    i = np.arange(nl, len(x) - ns)
    r = ((c[i + ns] - c[i]) / ns) / ((c[i] - c[i - nl]) / nl + 1e-30)
    t = i / sr
    m = np.abs(t - t_pred) <= half
    if not m.any():
        return t_pred
    return float(t[m][np.argmax(r[m])])


def cluster_job(args):
    cl, rows, ref_phi = args
    sos = butter(4, [2.0, 8.0], btype='band', fs=SR, output='sos')
    W = np.array([load_event(r['EventID'], r['Time']) for r in rows])      # (ne, nsta, 3, NS)
    out = []
    srU = SR * UP
    for s_i, sta in enumerate(SL):
        X = W[:, s_i]
        okz = np.isfinite(X[:, 0, 100])
        okh = np.isfinite(X[:, 1, 100]) & np.isfinite(X[:, 2, 100])
        ok = okz & okh
        if ok.sum() < 2:
            continue
        ii = np.where(ok)[0]
        Xf = sosfiltfilt(sos, np.nan_to_num(X[ii]), axis=-1)
        Xf /= np.std(Xf, axis=-1, keepdims=True).mean(1, keepdims=True) + 1e-20
        la, lo = STAS[sta]
        d = dist_km(np.mean([r['Latitude'] for r in rows]), np.mean([r['Longitude'] for r in rows]), la, lo)
        z = np.mean([r['depth'] for r in rows])
        R = np.hypot(d, z)
        Pp, Sp = 10 + R / VP, 10 + R / VS
        if Sp + 5 > NS / SR - 2:
            continue
        # initial alignment on the medoid (3-comp, [P-1, S+5])
        i0, i1 = int((Pp - 1.5) * SR), int((Sp + 5) * SR)
        ml = int(1.0 * SR)
        if i0 - ml < 0 or i1 + ml > NS:
            continue
        C = np.zeros((len(ii), len(ii)))
        for a in range(len(ii)):
            for b in range(len(ii)):
                if a < b:
                    v = np.mean([lag1(Xf[a, c], Xf[b, c], i0, i1, ml)[1] for c in range(3)])
                    C[a, b] = C[b, a] = v
        med = int(np.argmax(C.sum(1)))
        t = np.arange(NS) / SR
        Al = np.empty_like(Xf)
        for a in range(len(ii)):
            lg = np.nanmedian([lag1(Xf[med, c], Xf[a, c], i0, i1, ml)[0] for c in range(3)])
            lg = 0.0 if not np.isfinite(lg) else lg
            Al[a] = np.array([np.interp(t + lg / SR, t, Xf[a, c]) for c in range(3)])
        ref = Al.mean(0)
        P = stalta_pick(ref[0], SR, Pp, 1.5)
        S = stalta_pick(np.hypot(ref[1], ref[2]), SR, Sp, 2.0)
        if S - P < 1.0:
            continue
        refU = resample_poly(ref, UP, 1, axis=-1)
        AlU = resample_poly(Al, UP, 1, axis=-1)
        pw = (int((P - 0.3) * srU), int((P + 1.2) * srU))
        sw = (int((S - 0.4) * srU), int((S + 1.6) * srU))
        mlU = int(0.12 * srU)
        if sw[1] + mlU >= AlU.shape[-1]:
            continue
        # splitting: rotate N/E to fast/slow at the station reference phi
        phi = ref_phi.get(sta, np.nan)
        for a in range(len(ii)):
            lp, cp = lag1(refU[0], AlU[a, 0], pw[0], pw[1], mlU)
            ls_ = [lag1(refU[c], AlU[a, c], sw[0], sw[1], mlU) for c in (1, 2)]
            wv = np.array([max(c_, 0) ** 2 for _, c_ in ls_])
            lv = np.array([l_ for l_, _ in ls_])
            okc = np.isfinite(lv) & (wv > 0)
            dS = np.sum(wv[okc] * lv[okc]) / wv[okc].sum() if okc.any() else np.nan
            ccS = np.sqrt(np.sum(wv[okc]) / max(okc.sum(), 1)) if okc.any() else np.nan
            ddt = np.nan
            ccsw = np.nan
            if np.isfinite(phi):
                ph = np.radians(phi)
                f_r = refU[1] * np.cos(ph) + refU[2] * np.sin(ph)
                s_r = -refU[1] * np.sin(ph) + refU[2] * np.cos(ph)
                f_e = AlU[a, 1] * np.cos(ph) + AlU[a, 2] * np.sin(ph)
                s_e = -AlU[a, 1] * np.sin(ph) + AlU[a, 2] * np.cos(ph)
                lf, cf = lag1(f_r, f_e, sw[0], sw[1], mlU)
                lsl, csl = lag1(s_r, s_e, sw[0], sw[1], mlU)
                ddt = (lsl - lf) / srU
                ccsw = min(cf, csl)
            r = rows[ii[a]]
            out.append(dict(cl=cl, EventID=r['EventID'], t=r['Time'], depth=r['depth'], mag=r['Magnitude'],
                            lat=r['Latitude'], lon=r['Longitude'], sta=sta, n_cl=len(ii), P=P, S=S,
                            dP=lp / srU, ccP=cp, dS=dS / srU, ccS=ccS, ddt=ddt, ccsw=ccsw))
    return out


def stage_meas(a):
    os.makedirs(OUT, exist_ok=True)
    C = pd.read_csv(f'{REP}/clusters.csv')
    C = C[C.cl95 >= 0]
    rp = pd.read_csv(f'{ROOT}/data/sci/sws/reference_splitting.csv')
    rp = rp[rp.band == 'b28'].set_index('sta').phi0.to_dict()
    azimuths()
    jobs = [(cl, g.to_dict('records'), rp) for cl, g in C.groupby('cl95')]
    rows = []
    with Pool(a.procs) as p:
        for k, r in enumerate(p.imap_unordered(cluster_job, jobs, chunksize=4)):
            rows += r
            if k % 50 == 0:
                print(k, len(jobs), len(rows), flush=True)
    D = pd.DataFrame(rows)
    D.to_parquet(f'{OUT}/rep_event_meas.parquet')
    print(D.describe().T.round(4).to_string())


# ------------------------------------------------------------------------------------------- fit
def fcols(t, m7=None):
    from sci_dvv_deep_shallow import cols
    from dvv_cwire import epoch_s
    ts = epoch_s(pd.DatetimeIndex(t))
    C = cols(ts)
    if m7 is not None:
        tm = epoch_s([m7])[0]
        C[:, 4] = (ts >= tm).astype(float)
        C[:, 5] = ((ts >= tm) & (ts < tm + 60 * 86400)).astype(float)
    else:
        tm = epoch_s([pd.Timestamp('2024-12-05T18:44')])[0]
        C[:, 5] = ((ts >= tm) & (ts < tm + 60 * 86400)).astype(float)    # M7 60-d window (restart) indicator
    return C        # sin cos pet fer m7 early60 ramp


PN = ['sin', 'cos', 'pet', 'fer', 'm7', 'early', 'ramp']


def fit_events(g, col, wcol, m7=None, irls=3):
    """y_e = c_cluster + f(t_e); returns params (7,) with nan for unidentifiable."""
    y = g[col].values
    cc = np.clip(g[wcol].values, 0.05, 0.995)
    w0 = cc ** 2 / (1 - cc ** 2)
    F = fcols(g.t, m7)
    cls, ci = np.unique(g.cl.values, return_inverse=True)
    A0 = np.zeros((len(g), len(cls)))
    A0[np.arange(len(g)), ci] = 1
    # identifiable columns: need variation within clusters
    use = []
    for j in range(7):
        v = F[:, j]
        within = np.bincount(ci, weights=v) / np.bincount(ci)
        use.append(np.abs(v - within[ci]).sum() > 1e-6 * len(v) and (np.abs(v - within[ci]) > 0.05).sum() >= 4)
    use = np.array(use)
    A = np.c_[A0, F[:, use]]
    w = w0.copy()
    for it in range(irls):
        Wq = np.sqrt(w)[:, None]
        x, *_ = np.linalg.lstsq(A * Wq, y * Wq[:, 0], rcond=None)
        r = y - A @ x
        z = np.abs(r) * np.sqrt(w0)
        z = z / (1.4826 * np.median(z) + 1e-20)
        w = w0 * np.where(z <= 2, 1, 2 / np.maximum(z, 1e-12))
    out = np.full(7, np.nan)
    out[use] = x[len(cls):]
    return out


def decline_vec():
    G21 = pd.DatetimeIndex([f'2021-{m:02d}-15' for m in range(1, 13)])
    G25 = pd.DatetimeIndex([f'2025-{m:02d}-15' for m in range(1, 13)])
    Cd = fcols(G25).mean(0) - fcols(G21).mean(0)
    Cd[[0, 1, 5]] = 0
    return Cd


def prep(D):
    D = D.copy()
    D['t'] = pd.to_datetime(D.t)
    D['dSP'] = D.dS - D.dP
    D['ccSP'] = np.minimum(D.ccS, D.ccP)
    # event origin term: CC-weighted mean over stations of (dS, dP) -> relative terms
    for c_ in ('dS', 'dP'):
        w = np.clip(D['cc' + c_[1]], 0, 0.99) ** 2
        D['_w'] = w
        D['_wy'] = w * D[c_]
        m = D.groupby('EventID')[['_w', '_wy']].transform('sum')
        D[c_ + '_rel'] = D[c_] - m['_wy'] / m['_w'].clip(1e-9)
    D['cls'] = np.where(D.depth < 10, 'shallow', np.where((D.depth > 15) & (D.depth <= 30), 'deep',
                                                          np.where(D.depth > 30, 'vdeep', 'mid')))
    return D.drop(columns=['_w', '_wy'])


def qc(D, col):
    wcol = {'dSP': 'ccSP', 'dS_rel': 'ccS', 'dP_rel': 'ccP', 'ddt': 'ccsw'}[col]
    m = np.isfinite(D[col]) & (D[wcol] > (0.7 if col != 'ddt' else 0.6)) & (np.abs(D[col]) < 0.05)
    # clusters with >= 2 good events at the station
    g = D[m]
    n = g.groupby(['sta', 'cl']).EventID.transform('size')
    return g[n >= 2], wcol


def stage_fit(a):
    D = prep(pd.read_parquet(f'{OUT}/rep_event_meas.parquet'))
    Cd = decline_vec()
    rng = np.random.default_rng(11)
    rows, reps = [], {}
    for col in ['dSP', 'dS_rel', 'dP_rel', 'ddt']:
        G, wcol = qc(D, col)
        for (sta, cls), g in G.groupby(['sta', 'cls']):
            if cls in ('mid', 'vdeep') or g.cl.nunique() < 8:
                continue
            x = fit_events(g, col, wcol)
            ids = g.cl.unique()
            byc = {k: v for k, v in g.groupby('cl')}
            B = []
            for b in range(200):
                pick = rng.choice(ids, len(ids))
                gb = pd.concat([byc[k].assign(cl=f'{k}_{i}') for i, k in enumerate(pick)])
                B.append(fit_events(gb, col, wcol, irls=2))
            B = np.array(B)
            PL = []
            gp = g[g.t < '2021-06-01']
            if gp.cl.nunique() >= 8:
                for k in range(40):
                    d = pd.Timestamp('2016-01-01') + pd.Timedelta(days=int(rng.integers(0, 365 * 5)))
                    PL.append(fit_events(gp, col, wcol, m7=d, irls=2)[4])
            Dq = np.nan_to_num(x) @ Cd
            DB = np.nan_to_num(B) @ Cd
            reps[(col, sta, cls)] = dict(x=x, B=B, PL=np.array(PL))
            span = ((g.groupby('cl').t.min() < '2024-12-05') & (g.groupby('cl').t.max() > '2024-12-06')).sum()
            rows.append(dict(obs=col, sta=sta, cls=cls, n_ev=len(g), n_cl=len(ids), n_cl_span_m7=int(span),
                             depth_med=g.depth.median(),
                             **{f'{p}_ms': x[i] * 1e3 for i, p in enumerate(PN)},
                             m7_se_boot_ms=np.nanstd(B[:, 4]) * 1e3, early_se_boot_ms=np.nanstd(B[:, 5]) * 1e3,
                             fer_se_boot_ms=np.nanstd(B[:, 3]) * 1e3,
                             m7_se_plac_ms=np.nanstd(PL) * 1e3 if PL else np.nan,
                             D_ms=Dq * 1e3, D_se_boot_ms=np.std(DB) * 1e3))
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(f'{OUT}/rep_station_fit.csv', index=False)
    import pickle
    pickle.dump(reps, open(f'{OUT}/rep_station_fit_reps.pkl', 'wb'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['meas', 'fit'])
    ap.add_argument('--procs', type=int, default=4)
    a = ap.parse_args()
    {'meas': stage_meas, 'fit': stage_fit}[a.stage](a)


if __name__ == '__main__':
    main()
