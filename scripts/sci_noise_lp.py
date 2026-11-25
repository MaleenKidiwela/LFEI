#!/usr/bin/env python3
"""Long-period INTER-STATION ambient-noise cross-correlations (1-sps LH data), 2008-2026.

Stages
  meta       site list (aliases: TA.O02D -> site PETY, NC.KCPC -> site KCPB), coordinates, pair list
             (3-150 km) -> data/sci/noise_lp/sites.csv, pairs.csv
  correlate  per day (<= 4 processes): per station: gaps -> mask, demean/detrend, response removal to
             velocity (pre-filter 0.01-0.02-0.40-0.45 Hz, sub-sample start-time correction), rotation to ZNE,
             temporal normalisation = running absolute mean (128 s) of the 15-50 s earthquake band, common to
             all 3 components; 3600-s segments, 50 % overlap; segment rejection (gaps; eq-band RMS > 5x the
             day median); spectral whitening by the 3-component mean amplitude spectrum (rotation-consistent),
             0.03-0.40 Hz; ZNE cross-spectra averaged over common segments -> rotated to ZRT per pair.
             Stored: ZZ, RR, TT, ZR, RZ, lags -300..300 s at 1 s.
             -> /data/wsd02/maleen_data/MTJ_LH/ccf/YYYY/YYYY-MM-DD.npy (+ _meta.npz)
             station-day QC -> /data/wsd02/maleen_data/MTJ_LH/qc/YYYY-MM-DD.csv
  assemble   per pair -> /data/wsd02/maleen_data/MTJ_LH/pairs/<A>_<B>.npz (days, ccf, nseg)
Convention: C_ab(tau) = <a(t) b(t+tau)>; positive lag = energy travelling a -> b (causal).
"""
import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import numpy as np
import pandas as pd

ROOT = '/home/seismic/LFE'
OUT = os.path.join(ROOT, 'data/sci/noise_lp')
BASE = '/data/wsd02/maleen_data/MTJ_LH'
RAW = os.path.join(BASE, 'raw')
RESP = os.path.join(BASE, 'resp')
CCF = os.path.join(BASE, 'ccf')
CCF_EXTRA = os.path.join(BASE, 'ccf_extra')
PAIRD = os.path.join(BASE, 'pairs')
QC = os.path.join(BASE, 'qc')
DAY0 = pd.Timestamp('2008-01-01')
DAY1 = pd.Timestamp('2026-11-23')
NDAY = 86400
NFFT_DAY = 2 ** 17
SEG = 3600
STEP = 1800
NFFT = 8192
MAXLAG = 300
COMPS = ['ZZ', 'RR', 'TT', 'ZR', 'RZ']
ALIAS = {('TA', 'O02D'): 'PETY', ('NC', 'KCPC'): 'KCPB'}
PRE = (0.01, 0.02, 0.40, 0.45)
WHITE = (0.02, 0.03, 0.40, 0.45)
EQB = (0.02, 1 / 15.0)
RAM_WIN = 128


def stations():
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    from sci_noise_lp_download import STATIONS
    return STATIONS


def site_of(n, s):
    return ALIAS.get((n, s), s)


# ------------------------------------------------------------------------------------------- meta
def stage_meta(args):
    from obspy import read_inventory
    from obspy.geodetics import gps2dist_azimuth
    rows = []
    for n, s in stations():
        fn = os.path.join(RESP, f'{n}.{s}.xml')
        if not os.path.exists(fn):
            continue
        inv = read_inventory(fn)
        st = inv[0][0]
        rows.append(dict(net=n, sta=s, site=site_of(n, s), lat=st.latitude, lon=st.longitude, elev=st.elevation,
                         kind='geophone' if n == 'PB' else 'broadband'))
    S = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    S.to_csv(os.path.join(OUT, 'sites_raw.csv'), index=False)
    sites = S.groupby('site').agg(lat=('lat', 'mean'), lon=('lon', 'mean'), kind=('kind', 'first'),
                                  members=('sta', lambda x: '+'.join(x))).reset_index()
    sites.to_csv(os.path.join(OUT, 'sites.csv'), index=False)
    P = []
    for i in range(len(sites)):
        for j in range(i + 1, len(sites)):
            a, b = sites.iloc[i], sites.iloc[j]
            d, az, baz = gps2dist_azimuth(a.lat, a.lon, b.lat, b.lon)
            d /= 1e3
            if 3 <= d <= 150 or {a.site, b.site} == {'JCC', 'GASB'}:   # JCC-GASB (172 km): only pre-2010 pair
                P.append(dict(a=a.site, b=b.site, dist_km=d, az=az, baz=baz,
                              mid_lat=(a.lat + b.lat) / 2, mid_lon=(a.lon + b.lon) / 2,
                              kind='BB-BB' if a.kind == b.kind == 'broadband' else (
                                  'GP-GP' if a.kind == b.kind == 'geophone' else 'BB-GP')))
    P = pd.DataFrame(P)
    P['pid'] = np.arange(len(P))
    P.to_csv(os.path.join(OUT, 'pairs.csv'), index=False)
    print(len(sites), 'sites', len(P), 'pairs', P.kind.value_counts().to_dict())


# ------------------------------------------------------------------------------------------- preprocessing
_INV = {}
_RCACHE = {}


def inventory(n, s):
    from obspy import read_inventory
    k = (n, s)
    if k not in _INV:
        _INV[k] = read_inventory(os.path.join(RESP, f'{n}.{s}.xml'))
    return _INV[k]


def cos_taper(f, c):
    f1, f2, f3, f4 = c
    t = np.zeros_like(f)
    m = (f >= f2) & (f <= f3)
    t[m] = 1
    m = (f > f1) & (f < f2)
    t[m] = 0.5 * (1 - np.cos(np.pi * (f[m] - f1) / (f2 - f1)))
    m = (f > f3) & (f < f4)
    t[m] = 0.5 * (1 + np.cos(np.pi * (f[m] - f3) / (f4 - f3)))
    return t


def response(inv, seed_id, t):
    ch = inv.select(network=seed_id.split('.')[0], station=seed_id.split('.')[1],
                    location=seed_id.split('.')[2], channel=seed_id.split('.')[3], time=t)
    cha = ch[0][0][0]
    key = (seed_id, str(cha.start_date))
    if key not in _RCACHE:
        f = np.fft.rfftfreq(NFFT_DAY, 1.0)
        r = cha.response.get_evalresp_response_for_frequencies(f[1:], output='VEL')
        R = np.r_[r[0] if np.isfinite(r[0]) else 1.0, r]
        _RCACHE[key] = (R, cha.azimuth, cha.dip)
    return _RCACHE[key]


def movmean(x, n):
    c = np.cumsum(np.r_[0.0, x])
    h = n // 2
    i0 = np.clip(np.arange(len(x)) - h, 0, len(x))
    i1 = np.clip(np.arange(len(x)) + h + 1, 0, len(x))
    return (c[i1] - c[i0]) / np.maximum(i1 - i0, 1)


def load_station_day(n, s, day):
    """-> (comp dict Z/N/E -> velocity (86400,), valid mask (86400,) bool per comp) or None."""
    from obspy import read, UTCDateTime
    from obspy.signal.rotate import rotate2zne
    fn = os.path.join(RAW, f'{n}.{s}', f'{day:%Y}', f'{n}.{s}.{day:%Y-%m-%d}.mseed')
    if not os.path.exists(fn):
        return None
    try:
        st = read(fn)
    except Exception:
        return None
    inv = inventory(n, s)
    u0 = UTCDateTime(day.to_pydatetime())
    # choose one location code per channel (most samples)
    by = {}
    for tr in st:
        if abs(tr.stats.sampling_rate - 1.0) > 1e-3:
            continue
        ch = tr.stats.channel
        by.setdefault(ch, {}).setdefault(tr.stats.location, []).append(tr)
    f = np.fft.rfftfreq(NFFT_DAY, 1.0)
    pre = cos_taper(f, PRE)
    chans = {}
    for ch, locs in by.items():
        loc = max(locs, key=lambda L: sum(t.stats.npts for t in locs[L]))
        trs = locs[loc]
        x = np.zeros(NDAY)
        m = np.zeros(NDAY, bool)
        big = max(trs, key=lambda t: t.stats.npts)
        off = float(big.stats.starttime - u0)
        frac = off - np.round(off)
        for tr in trs:
            k0 = int(np.round(float(tr.stats.starttime - u0)))
            d = tr.data.astype(float)
            a, b = max(k0, 0), min(k0 + len(d), NDAY)
            if b <= a:
                continue
            x[a:b] = d[a - k0:b - k0]
            m[a:b] = True
        if m.sum() < 4 * SEG:
            continue
        seed = f'{n}.{s}.{loc}.{ch}'
        try:
            R, az, dip = response(inv, seed, u0 + 43200)
        except Exception:
            continue
        idx = np.where(m)[0]
        p = np.polyfit(idx, x[idx], 1)
        x = np.where(m, x - np.polyval(p, np.arange(NDAY)), 0.0)
        # taper 60 s at each gap edge
        edge = np.diff(np.r_[0, m.astype(int), 0])
        tap = np.ones(NDAY)
        for e0, e1 in zip(np.where(edge == 1)[0], np.where(edge == -1)[0]):
            L = min(60, (e1 - e0) // 4)
            if L > 1:
                w = 0.5 * (1 - np.cos(np.pi * np.arange(L) / L))
                tap[e0:e0 + L] *= w
                tap[e1 - L:e1] *= w[::-1]
        x = x * tap
        X = np.fft.rfft(x, NFFT_DAY)
        X = X * pre / np.where(np.abs(R) > 0, R, 1.0) * np.exp(-2j * np.pi * f * frac)
        v = np.fft.irfft(X, NFFT_DAY)[:NDAY]
        # mask: grow invalid zones by 300 s (deconvolution ringing)
        bad = ~m
        if bad.any():
            bad = movmean(bad.astype(float), 601) > 0
        chans[ch] = (v, ~bad, az, dip)
    if not chans:
        return None
    zc = [c for c in chans if c.endswith('Z')]
    hc = [c for c in chans if not c.endswith('Z')]
    if not zc:
        return None
    z = chans[zc[0]]
    out = {'Z': (z[0], z[1])}
    if len(hc) >= 2:
        h1, h2 = chans[hc[0]], chans[hc[1]]
        zz, nn, ee = rotate2zne(z[0], z[2], z[3], h1[0], h1[2], h1[3], h2[0], h2[2], h2[3])
        mm = z[1] & h1[1] & h2[1]
        out = {'Z': (zz, z[1]), 'N': (nn, mm), 'E': (ee, mm)}
    return out


def spectra(day_data):
    """-> F (3, nseg, nf) complex64 (whitened), valid (nseg,) bool for Z, validH (nseg,) bool; qc dict."""
    f = np.fft.rfftfreq(NFFT_DAY, 1.0)
    eqt = cos_taper(f, (0.01, EQB[0], EQB[1], 0.09))
    comps = ['Z', 'N', 'E']
    have = [c in day_data for c in comps]
    xs, ms, eqs = [], [], []
    for c in comps:
        if c in day_data:
            v, m = day_data[c]
        else:
            v, m = np.zeros(NDAY), np.zeros(NDAY, bool)
        xs.append(v)
        ms.append(m)
        eqs.append(np.fft.irfft(np.fft.rfft(v, NFFT_DAY) * eqt, NFFT_DAY)[:NDAY])
    W = np.zeros(NDAY)
    for c in range(3):
        if have[c]:
            W = np.maximum(W, movmean(np.abs(eqs[c]), RAM_WIN))
    med = np.median(W[ms[0]]) if ms[0].any() else 1.0
    W = np.maximum(W, 1e-3 * med)
    starts = np.arange(0, NDAY - SEG + 1, STEP)
    ns = len(starts)
    win = np.hanning(SEG)
    ff = np.fft.rfftfreq(NFFT, 1.0)
    band = cos_taper(ff, WHITE)
    F = np.zeros((3, ns, len(ff)), np.complex64)
    valid = np.zeros((3, ns), bool)
    erms = np.full((3, ns), np.nan)
    for c in range(3):
        if not have[c]:
            continue
        y = xs[c] / W
        for k, s0 in enumerate(starts):
            mk = ms[c][s0:s0 + SEG]
            if mk.all():
                valid[c, k] = True
                erms[c, k] = np.std(eqs[c][s0:s0 + SEG])
                F[c, k] = np.fft.rfft(y[s0:s0 + SEG] * win, NFFT)
    # earthquake/transient rejection: eq-band RMS > 5x day median
    for c in range(3):
        if valid[c].any():
            md = np.nanmedian(erms[c][valid[c]])
            valid[c] &= ~(erms[c] > 5 * md)
    vZ = valid[0]
    vH = valid[0] & valid[1] & valid[2]
    # whitening by the mean amplitude of the available components (smoothed over 33 bins ~0.004 Hz)
    nav = 1 + 2 * int(have[1] and have[2])
    A = np.abs(F[0]) + (np.abs(F[1]) + np.abs(F[2]) if nav == 3 else 0)
    A = A / nav
    k = np.ones(33) / 33
    for s in range(ns):
        if vZ[s]:
            a = np.convolve(A[s], k, mode='same')
            a = np.maximum(a, 1e-6 * a.max() + 1e-30)
            F[:, s] = F[:, s] * (band / a)[None, :]
    # QC: microseism-band (0.1-0.3 Hz) RMS (pre-normalisation) of Z
    zband = cos_taper(f, (0.08, 0.1, 0.3, 0.35))
    zm = np.fft.irfft(np.fft.rfft(xs[0], NFFT_DAY) * zband, NFFT_DAY)[:NDAY]
    lp = np.fft.irfft(np.fft.rfft(xs[0], NFFT_DAY) * cos_taper(f, (0.03, 0.05, 0.1, 0.12)), NFFT_DAY)[:NDAY]
    qc = dict(nseg_z=int(vZ.sum()), nseg_h=int(vH.sum()), ms_rms=float(np.std(zm[ms[0]])) if ms[0].any() else np.nan,
              lp_rms=float(np.std(lp[ms[0]])) if ms[0].any() else np.nan, have_h=bool(have[1] and have[2]))
    return F, vZ, vH, qc


def rotmat(phi):
    """rows: Z, R, T from columns Z, N, E; phi = azimuth of R (deg)."""
    c, s = np.cos(np.radians(phi)), np.sin(np.radians(phi))
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def correlate_day(day, pid_min=None):
    """pid_min: supplementary pass for pairs added later (pid >= pid_min) -> CCF_EXTRA."""
    day = pd.Timestamp(day)
    odir = CCF if pid_min is None else CCF_EXTRA
    fo = os.path.join(odir, f'{day:%Y}', f'{day:%Y-%m-%d}.npy')
    if os.path.exists(fo) or os.path.exists(fo + '.none'):
        return 0
    os.makedirs(os.path.dirname(fo), exist_ok=True)
    P = pd.read_csv(os.path.join(OUT, 'pairs.csv'))
    S = pd.read_csv(os.path.join(OUT, 'sites_raw.csv'))
    if pid_min is not None:
        P = P[P.pid >= pid_min]
        S = S[S.site.isin(set(P.a) | set(P.b))]
    spec = {}
    qcs = []
    for _, r in S.iterrows():
        if r.site in spec:
            continue
        try:
            dd = load_station_day(r.net, r.sta, day)
        except Exception as e:
            print('load fail', r.net, r.sta, day.date(), e, flush=True)
            dd = None
        if dd is None:
            continue
        F, vZ, vH, qc = spectra(dd)
        if vZ.sum() < 8:
            continue
        spec[r.site] = (F, vZ, vH)
        qc.update(site=r.site, net=r.net, sta=r.sta, day=f'{day:%Y-%m-%d}')
        qcs.append(qc)
    if pid_min is None:
        os.makedirs(os.path.join(QC, f'{day:%Y}'), exist_ok=True)
        pd.DataFrame(qcs).to_csv(os.path.join(QC, f'{day:%Y}', f'{day:%Y-%m-%d}.csv'), index=False)
    pids, ccfs, nsz, nsh = [], [], [], []
    lag_idx = np.r_[np.arange(NFFT - MAXLAG, NFFT), np.arange(0, MAXLAG + 1)]
    for _, p in P.iterrows():
        if p.a not in spec or p.b not in spec:
            continue
        Fa, vZa, vHa = spec[p.a]
        Fb, vZb, vHb = spec[p.b]
        cz = vZa & vZb
        ch = vHa & vHb
        if cz.sum() < 8:
            continue
        out = np.full((5, 2 * MAXLAG + 1), np.nan, np.float32)
        zz = np.fft.irfft(np.sum(np.conj(Fa[0, cz]) * Fb[0, cz], axis=0) / cz.sum(), NFFT)
        out[0] = zz[lag_idx]
        if ch.sum() >= 8:
            C = np.einsum('isf,jsf->ijf', np.conj(Fa[:, ch]), Fb[:, ch]) / ch.sum()
            Ra, Rb = rotmat(p.az), rotmat(p.baz + 180.0)
            Cr = np.einsum('ai,ijf,bj->abf', Ra, C, Rb)
            c = np.fft.irfft(Cr, NFFT, axis=-1)[..., lag_idx]
            out[1] = c[1, 1]
            out[2] = c[2, 2]
            out[3] = c[0, 1]
            out[4] = c[1, 0]
        pids.append(p.pid)
        ccfs.append(out)
        nsz.append(cz.sum())
        nsh.append(ch.sum())
    if not pids:
        open(fo + '.none', 'w').close()
        return 0
    np.savez(fo.replace('.npy', '_meta.npz'), pid=np.array(pids, np.int32), nseg_z=np.array(nsz, np.int16),
             nseg_h=np.array(nsh, np.int16))
    np.save(fo, np.array(ccfs, np.float32))
    return len(pids)


def ready_months():
    """months whose downloads are complete for every station (download markers)."""
    from sci_noise_lp_download import tasks
    T = tasks()
    ok = {}
    for _, n, s, m in T:
        done = os.path.exists(os.path.join(RAW, f'{n}.{s}', '.done', f'{m}'))
        ok[m] = ok.get(m, True) and done
    return {m for m, v in ok.items() if v}


def stage_correlate(args):
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    days = pd.date_range(args.start or DAY0, args.end or DAY1, freq='D')
    if args.reverse:
        days = days[::-1]
    t0 = time.time()
    done = set()
    with ProcessPoolExecutor(args.workers) as ex:
        while True:
            rm = ready_months()
            todo = [d for d in days if d not in done and d.to_period('M') in rm]
            if not todo:
                if len(done) == len(days):
                    break
                time.sleep(300)
                continue
            for i, n in enumerate(ex.map(correlate_day, todo, chunksize=1)):
                done.add(todo[i])
                if i % 20 == 0:
                    print(f'{time.strftime("%H:%M:%S")} {todo[i].date()} {len(done)}/{len(days)} pairs {n} '
                          f'{(time.time() - t0) / len(done):.1f} s/day', flush=True)
            if not args.wait:
                break


def stage_addsite(args):
    """append sites/pairs for stations added to the download list later, keeping existing pair ids."""
    from obspy import read_inventory
    from obspy.geodetics import gps2dist_azimuth
    SR = pd.read_csv(os.path.join(OUT, 'sites_raw.csv'))
    P = pd.read_csv(os.path.join(OUT, 'pairs.csv'))
    new = []
    for n, s in stations():
        if ((SR.net == n) & (SR.sta == s)).any():
            continue
        st = read_inventory(os.path.join(RESP, f'{n}.{s}.xml'))[0][0]
        new.append(dict(net=n, sta=s, site=site_of(n, s), lat=st.latitude, lon=st.longitude, elev=st.elevation,
                        kind='geophone' if n == 'PB' else 'broadband'))
    if not new:
        return
    N = pd.DataFrame(new)
    sites = SR.groupby('site').agg(lat=('lat', 'mean'), lon=('lon', 'mean'), kind=('kind', 'first')).reset_index()
    rows = []
    pid = P.pid.max() + 1
    for _, a in N.iterrows():
        for _, b in sites.iterrows():
            d, az, baz = gps2dist_azimuth(b.lat, b.lon, a.lat, a.lon)
            d /= 1e3
            if 3 <= d <= args.maxdist:
                rows.append(dict(a=b.site, b=a.site, dist_km=d, az=az, baz=baz, mid_lat=(a.lat + b.lat) / 2,
                                 mid_lon=(a.lon + b.lon) / 2,
                                 kind='BB-BB' if b.kind == 'broadband' else 'BB-GP', pid=pid))
                pid += 1
    pd.concat([SR, N]).to_csv(os.path.join(OUT, 'sites_raw.csv'), index=False)
    pd.concat([P, pd.DataFrame(rows)]).to_csv(os.path.join(OUT, 'pairs.csv'), index=False)
    print('added', N.sta.tolist(), len(rows), 'pairs')


def _cd_extra(day):
    return correlate_day(day, pid_min=EXTRA_PID)


def stage_correlate_extra(args):
    global EXTRA_PID
    EXTRA_PID = args.pid_min
    days = pd.date_range(args.start or DAY0, args.end or DAY1, freq='D')
    with ProcessPoolExecutor(args.workers) as ex:
        for i, n in enumerate(ex.map(_cd_extra, days, chunksize=4)):
            if i % 100 == 0:
                print(time.strftime('%H:%M:%S'), days[i].date(), n, flush=True)


EXTRA_PID = None


def stage_assemble(args):
    P = pd.read_csv(os.path.join(OUT, 'pairs.csv'))
    files = sorted(glob.glob(os.path.join(CCF, '*', '*-*-*.npy')) + glob.glob(os.path.join(CCF_EXTRA, '*', '*-*-*.npy')))
    os.makedirs(PAIRD, exist_ok=True)
    metas = []
    for fn in files:
        mt = np.load(fn.replace('.npy', '_meta.npz'))
        dn = (pd.Timestamp(os.path.basename(fn)[:10]) - DAY0).days
        metas.append((fn, dn, mt['pid'], mt['nseg_z'], mt['nseg_h']))
    chunk = args.chunk
    for c0 in range(0, len(P), chunk):
        pset = set(range(c0, min(c0 + chunk, len(P))))
        acc = {p: ([], [], [], []) for p in pset}
        for fn, dn, pid, nz, nh in metas:
            sel = [i for i, p in enumerate(pid) if p in pset]
            if not sel:
                continue
            arr = np.load(fn, mmap_mode='r')
            for i in sel:
                a = acc[pid[i]]
                a[0].append(dn)
                a[1].append(np.array(arr[i]))
                a[2].append(nz[i])
                a[3].append(nh[i])
        for p, a in acc.items():
            if not a[0]:
                continue
            r = P.iloc[p]
            np.savez(os.path.join(PAIRD, f'{r.a}_{r.b}.npz'), days=np.array(a[0], np.int32),
                     ccf=np.array(a[1], np.float32), nseg_z=np.array(a[2]), nseg_h=np.array(a[3]))
        print(time.strftime('%H:%M:%S'), 'assembled pairs', c0, '..', c0 + chunk, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage')
    ap.add_argument('--start')
    ap.add_argument('--end')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--reverse', action='store_true')
    ap.add_argument('--wait', action='store_true')
    ap.add_argument('--pid-min', dest='pid_min', type=int)
    ap.add_argument('--maxdist', type=float, default=150.0)
    ap.add_argument('--chunk', type=int, default=80)
    a = ap.parse_args()
    globals()['stage_' + a.stage](a)
