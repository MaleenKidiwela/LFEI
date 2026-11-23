#!/usr/bin/env python
"""sci_dyn_trigger: measured teleseismic peak ground velocity / dynamic stress at the MTJ LFE source
region and the dependence of LFE burst triggering on it.

Stages
  download : BK.JCC and BK.HOPS LH? (1 Hz, STS-2 / STS-1/STS-7) for every M>=6.5 quake in
             data/sci/cascades2/remote_quakes.csv, window P-1200 s .. P+5700 s (P from iasp91 at the LFE
             centroid). Sequential NCEDC requests, ~1/s.  -> data/sci/dyn_trigger/wf/{STA}/{i:04d}.mseed
  measure  : response removal to velocity (pre_filt 0.004/0.008/0.40/0.45 Hz), bandpass 0.01-0.45 Hz
             (0.5 Hz is the LH Nyquist; band otherwise as requested) and a 0.02-0.1 Hz surface-wave-band
             variant; rotate to Z/R/T; PGV in [P, P+1.5 h]; pre-P noise; dominant period at the peak.
             Dynamic stress  sigma_d = mu*PGV/c  (mu=30 GPa, c=3.5 km/s).
             Fault-resolved: plane-wave horizontal strain e_ij = -(p_i v_j + p_j v_i)/2, p = k/c
             (k = propagation direction at the LFEs), plane stress (sigma_iz = 0, nu=0.25, same
             convention as data/sci/tidal_stress), resolved on strike 293 dip 45 NE rake 180:
             tau = s.sigma.n (+ = promotes right-lateral slip), sigma_n = n.sigma.n (+ = unclamping).
             Love-only (T) and Rayleigh-only (R) contributions computed separately.
  analyze  : real-only LFE burst onsets (2-h gap) in [t_surf, t_surf+1 h) vs the cascades2 tidal-phase
             and time-of-day matched controls (same rng seed -> same controls as remote_check);
             binned ratios, step / ramp / power models with bootstrap over quakes, control-draw nulls,
             wave type / geometry / period / tidal-stress dependence; local-quake comparison.
  fig      : figures/sci_dyn_trigger_*.png
Usage: python scripts/sci_dyn_trigger.py [download|measure|analyze|fig] ...
"""
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import warnings

import numpy as np
import pandas as pd

warnings.simplefilter('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = 'data/sci/dyn_trigger'
WF = f'{OUT}/wf'
STAS = ['JCC', 'HOPS']
LAT0, LON0, DEP0 = 40.06, -123.625, 27.0
MU, CS, NU = 30e9, 3500.0, 0.25
STRIKE, DIP, RAKE = 293.0, 45.0, 180.0
EPOCH = pd.Timestamp('1970-01-01')
PRE, POST = 1200.0, 5700.0
ASG = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]           # a*sigma grid (kPa)
TAG = [10, 30, 60, 300, 900, 1800, 3600, 3 * 3600, 10 * 3600, 86400, 3 * 86400, 10 * 86400]   # t_a grid (s)
for d_ in [OUT, WF] + [f'{WF}/{s}' for s in STAS]:
    os.makedirs(d_, exist_ok=True)


def load_quakes():
    Q = pd.read_csv('data/sci/cascades2/remote_quakes.csv')
    Q['pgv_rank'] = Q.pgv_proxy_um_s.rank(ascending=False)
    return Q


# ----------------------------------------------------------------------------- download
def p_times(Q):
    from obspy.taup import TauPyModel
    m = TauPyModel('iasp91')
    tp = []
    for r in Q.itertuples():
        arr = m.get_travel_times(source_depth_in_km=min(max(r.depth, 0.0), 700.0), distance_in_degree=r.delta_deg,
                                 phase_list=['P', 'Pdiff', 'PKP', 'PKIKP', 'PKiKP'])
        tp.append(r.to + (min(a.time for a in arr) if arr else r.delta_deg * 111.19 / 8.0))
    return np.array(tp)


def stage_download():
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    Q = load_quakes()
    fq = f'{OUT}/quakes_ptime.csv'
    if os.path.exists(fq):
        tp = pd.read_csv(fq).t_p.values
    else:
        tp = p_times(Q)
        pd.DataFrame({'i': np.arange(len(Q)), 'id': Q.id, 't_p': tp}).to_csv(fq, index=False)
    c = Client('NCEDC', timeout=120)
    if not os.path.exists(f'{OUT}/inv_LH.xml'):
        inv = c.get_stations(network='BK', station=','.join(STAS), channel='LH?', level='response',
                             starttime=UTCDateTime('2008-12-01'))
        inv.write(f'{OUT}/inv_LH.xml', format='STATIONXML')
        time.sleep(1)
    log = open(f'{OUT}/download.log', 'a')
    for i in range(len(Q)):
        todo = [s for s in STAS if not (os.path.exists(f'{WF}/{s}/{i:04d}.mseed') or os.path.exists(f'{WF}/{s}/{i:04d}.mseed.none'))]
        if not todo:
            continue
        t0 = UTCDateTime(float(tp[i]) - PRE)
        try:
            st = c.get_waveforms('BK', ','.join(todo), '*', 'LH?', t0, t0 + PRE + POST)
        except Exception as e:  # no data for any station
            st, err = None, str(e)[:200]
        for s in todo:
            fn = f'{WF}/{s}/{i:04d}.mseed'
            ss = st.select(station=s) if st is not None else []
            if len(ss):
                ss.write(fn, format='MSEED')
                log.write(f'{i} {s} ok {len(ss)}\n')
            else:
                open(fn + '.none', 'w').write('no data' if st is not None else err)
                log.write(f'{i} {s} none\n')
        log.flush()
        time.sleep(1.0)
        if i % 50 == 0:
            print(i, flush=True)


# ----------------------------------------------------------------------------- measure
def fault_vectors():
    a = np.array([np.sin(np.radians(STRIKE)), np.cos(np.radians(STRIKE)), 0.0])          # E,N,U
    d = np.array([np.sin(np.radians(STRIKE + 90)), np.cos(np.radians(STRIKE + 90)), 0.0])
    dl = np.radians(DIP)
    n = np.sin(dl) * d + np.cos(dl) * np.array([0, 0, 1.0])       # normal into hanging wall
    downdip = np.cos(dl) * d - np.sin(dl) * np.array([0, 0, 1.0])
    s = np.cos(np.radians(RAKE)) * a - np.sin(np.radians(RAKE)) * downdip     # HW slip direction
    return n, s


NF, SF = fault_vectors()


def fault_stress(vE, vN, prop_az):
    """plane-wave, plane-stress fault-resolved shear and normal stress (Pa) from horizontal velocity (m/s)."""
    k = np.array([np.sin(np.radians(prop_az)), np.cos(np.radians(prop_az))])
    v = np.vstack([vE, vN])                                    # (2, nt)
    eps = -(k[:, None, None] * v[None, :, :] + k[None, :, None] * v[:, None, :]) / (2 * CS)   # (2,2,nt)
    lam_ps = 2 * MU * NU / (1 - NU)                              # plane-stress lambda
    tr = eps[0, 0] + eps[1, 1]
    sig = 2 * MU * eps
    sig[0, 0] += lam_ps * tr
    sig[1, 1] += lam_ps * tr
    n2, s2 = NF[:2], SF[:2]                                     # sigma_iz = 0
    tau = np.einsum('i,ijt,j->t', s2, sig, n2)
    sn = np.einsum('i,ijt,j->t', n2, sig, n2)
    return tau, sn


def dom_period(x, ipk, fs=1.0):
    """2 x interval between the zero crossings bracketing the peak sample."""
    sgn = np.sign(x)
    j1 = ipk
    while j1 > 0 and sgn[j1 - 1] == sgn[ipk]:
        j1 -= 1
    j2 = ipk
    while j2 < len(x) - 1 and sgn[j2 + 1] == sgn[ipk]:
        j2 += 1
    return 2.0 * (j2 - j1 + 1) / fs


def measure_one(args):
    i, s, tp, baz, prop_az = args
    from obspy import read, read_inventory
    fn = f'{WF}/{s}/{i:04d}.mseed'
    out = dict(i=i, sta=s)
    if not os.path.exists(fn):
        return out
    try:
        inv = read_inventory(f'{OUT}/inv_LH.xml')
        st = read(fn)
        st.merge(method=1, fill_value=None)
        st = st.split() if any(np.ma.is_masked(tr.data) for tr in st) else st
        # keep longest segment per channel
        best = {}
        for tr in st:
            k = tr.stats.channel
            if k not in best or tr.stats.npts > best[k].stats.npts:
                best[k] = tr
        from obspy import Stream
        st = Stream(list(best.values()))
        if len(st) < 3:
            out['err'] = f'nch={len(st)}'
            return out
        st.detrend('demean')
        st.detrend('linear')
        st.taper(0.03)
        st.remove_response(inventory=inv, output='VEL', pre_filt=(0.004, 0.008, 0.40, 0.45), zero_mean=True, taper=False)
        st.rotate('->ZNE', inventory=inv)
        t_start = max(tr.stats.starttime for tr in st)
        t_end = min(tr.stats.endtime for tr in st)
        st.trim(t_start, t_end)
        from obspy import UTCDateTime
        tP = UTCDateTime(float(tp))
        cov = (min(t_end, tP + 5400) - max(t_start, tP)) / 5400.0
        out['coverage'] = float(cov)
        if cov < 0.8:
            out['err'] = 'coverage'
            return out
        for band, (f1, f2) in {'bb': (0.01, 0.45), 'sw': (0.02, 0.1)}.items():
            sb = st.copy()
            sb.filter('bandpass', freqmin=f1, freqmax=f2, corners=4, zerophase=True)
            Z = sb.select(component='Z')[0]
            E = sb.select(component='E')[0]
            N = sb.select(component='N')[0]
            nmin = min(len(Z.data), len(E.data), len(N.data))
            z, e, nn = Z.data[:nmin], E.data[:nmin], N.data[:nmin]
            t = Z.stats.starttime.timestamp + np.arange(nmin) * Z.stats.delta
            b = np.radians(baz)
            r = -e * np.sin(b) - nn * np.cos(b)          # radial (away from source)
            tt = -e * np.cos(b) + nn * np.sin(b)         # transverse
            w = (t >= tP.timestamp) & (t <= tP.timestamp + 5400)
            wn = (t >= tP.timestamp - 1100) & (t <= tP.timestamp - 60)
            h = np.hypot(e, nn)
            v3 = np.sqrt(e ** 2 + nn ** 2 + z ** 2)
            ip = np.argmax(np.where(w, v3, 0))
            out[f'{band}_pgv3'] = float(v3[ip])
            out[f'{band}_t_pk'] = float(t[ip])
            out[f'{band}_pgvh'] = float(h[w].max())
            out[f'{band}_pgvz'] = float(np.abs(z[w]).max())
            out[f'{band}_pgvr'] = float(np.abs(r[w]).max())
            out[f'{band}_pgvt'] = float(np.abs(tt[w]).max())
            iz, ih = np.argmax(np.where(w, np.abs(z), 0)), np.argmax(np.where(w, np.abs(tt), 0))
            out[f'{band}_T_z'] = dom_period(z, iz)
            out[f'{band}_T_t'] = dom_period(tt, ih)
            comp = np.array([np.abs(z[ip]), np.abs(r[ip]), np.abs(tt[ip])])
            out[f'{band}_pk_comp'] = 'ZRT'[int(np.argmax(comp))]
            out[f'{band}_T_pk'] = dom_period([z, r, tt][int(np.argmax(comp))], ip)
            out[f'{band}_noise3'] = float(v3[wn].max()) if wn.sum() > 100 else np.nan
            out[f'{band}_noise3_rms'] = float(np.sqrt(np.mean(v3[wn] ** 2))) if wn.sum() > 100 else np.nan
            # fault-resolved stresses (total, Love-only, Rayleigh-only)
            pa = np.radians(prop_az)
            tau, sn = fault_stress(e[w], nn[w], prop_az)
            out[f'{band}_tau_pos_kPa'] = float(tau.max() / 1e3)
            out[f'{band}_tau_abs_kPa'] = float(np.abs(tau).max() / 1e3)
            out[f'{band}_sn_abs_kPa'] = float(np.abs(sn).max() / 1e3)
            out[f'{band}_cfs04_pos_kPa'] = float((tau + 0.4 * sn).max() / 1e3)
            # Love only: transverse motion only;  Rayleigh only: radial motion only
            tE, tN = -tt[w] * np.cos(b), tt[w] * np.sin(b)
            rE, rN = -r[w] * np.sin(b), -r[w] * np.cos(b)
            tauL, _ = fault_stress(tE, tN, prop_az)
            tauR, snR = fault_stress(rE, rN, prop_az)
            out[f'{band}_tauL_abs_kPa'] = float(np.abs(tauL).max() / 1e3)
            out[f'{band}_tauR_abs_kPa'] = float(np.abs(tauR).max() / 1e3)
            out[f'{band}_snR_abs_kPa'] = float(np.abs(snR).max() / 1e3)
            # rate-state (Dieterich 1994) transient integral X = int (exp(tau(t)/a sigma) - 1) dt  [s], t << t_a:
            # triggered count during the waves = r t_a ln(1 + X/t_a) ~ r X for X << t_a
            if band == 'bb':
                from scipy.special import logsumexp
                for asg in ASG:
                    for pre, x_ in [('lnX', tau / (asg * 1e3)), ('lnXcfs', (tau + 0.4 * sn) / (asg * 1e3))]:
                        out[f'{pre}_as{asg}'] = float(np.log(max(np.sum(np.expm1(x_)), 1e-9)) if x_.max() < 30 else logsumexp(x_))
            # duration above half of peak |tau| (s) and number of half-cycles above 1/2 peak
            ta = np.abs(tau)
            out[f'{band}_dur_half_s'] = float((ta >= 0.5 * ta.max()).sum())
            del pa
    except Exception as ex:
        out['err'] = repr(ex)[:200]
    return out


def stage_measure():
    from obspy.geodetics import gps2dist_azimuth
    Q = load_quakes()
    tp = pd.read_csv(f'{OUT}/quakes_ptime.csv').t_p.values
    stll = {'JCC': (40.817, -124.030), 'HOPS': (38.994, -123.072)}
    jobs = []
    for i, r in Q.iterrows():
        _, az_lfe2q, baz_q2lfe = gps2dist_azimuth(LAT0, LON0, r.latitude, r.longitude)
        prop_az = (az_lfe2q + 180.0) % 360.0       # wave propagation direction at the LFEs
        for s in STAS:
            _, az_s, _ = gps2dist_azimuth(stll[s][0], stll[s][1], r.latitude, r.longitude)
            jobs.append((i, s, tp[i], az_s, prop_az))
    with Pool(4) as p:
        res = p.map(measure_one, jobs, chunksize=8)
    M = pd.DataFrame(res)
    M.to_csv(f'{OUT}/pgv_raw_measurements.csv', index=False)
    print(M.describe().T.to_string())
    print(M.err.value_counts() if 'err' in M else 'no errors')


# ----------------------------------------------------------------------------- analyze
def build_table():
    from obspy.geodetics import gps2dist_azimuth
    Q = load_quakes()
    M = pd.read_csv(f'{OUT}/pgv_raw_measurements.csv')
    tp = pd.read_csv(f'{OUT}/quakes_ptime.csv').t_p.values
    T = Q[['id', 'time', 'latitude', 'longitude', 'depth', 'mag', 'place', 'name', 'dist_km', 'delta_deg', 'to', 't_surf',
           'pgv_proxy_um_s', 'pgv_rank']].copy()
    T['t_p'] = tp
    az = [gps2dist_azimuth(LAT0, LON0, r.latitude, r.longitude)[1] for r in Q.itertuples()]
    T['baz_lfe'] = az
    T['prop_az'] = (np.array(az) + 180) % 360
    keep = [c for c in M.columns if c not in ('i', 'sta')]
    for s in STAS:
        Ms = M[M.sta == s].set_index('i')
        for c in keep:
            T[f'{s}_{c}'] = Ms[c].reindex(T.index).values
    for s in STAS:
        ok = T[f'{s}_bb_pgv3'].notna()
        if f'{s}_err' in T:
            ok &= T[f'{s}_err'].isna()
        T[f'{s}_ok'] = ok
    # primary PGV: JCC; HOPS where JCC missing.  Both in um/s.
    for k in (['bb_pgv3', 'bb_pgvh', 'bb_pgvz', 'bb_pgvr', 'bb_pgvt', 'sw_pgv3', 'bb_noise3', 'sw_noise3', 'bb_T_pk', 'sw_T_pk',
              'bb_tau_abs_kPa', 'bb_tau_pos_kPa', 'bb_sn_abs_kPa', 'bb_cfs04_pos_kPa', 'bb_tauL_abs_kPa', 'bb_tauR_abs_kPa',
              'sw_tau_abs_kPa', 'bb_T_t', 'bb_T_z', 'bb_t_pk', 'bb_dur_half_s']
              + [f'{p_}_as{x}' for p_ in ['lnX', 'lnXcfs'] for x in ASG]):
        a = T[f'JCC_{k}'].where(T.JCC_ok)
        b = T[f'HOPS_{k}'].where(T.HOPS_ok)
        T[k] = a.fillna(b)
    T['pgv_sta'] = np.where(T.JCC_ok, 'JCC', np.where(T.HOPS_ok, 'HOPS', ''))
    T['pk_comp'] = np.where(T.JCC_ok, T.JCC_bb_pk_comp, T.HOPS_bb_pk_comp)
    for k in ['bb_pgv3', 'bb_pgvh', 'bb_pgvz', 'bb_pgvr', 'bb_pgvt', 'sw_pgv3', 'bb_noise3', 'sw_noise3']:
        T[k] = T[k] * 1e6                                         # m/s -> um/s
    T['pgv_um_s'] = T.bb_pgv3
    T['snr_bb'] = T.bb_pgv3 / T.bb_noise3
    T['snr_sw'] = T.sw_pgv3 / T.sw_noise3
    T['sigma_d_kPa'] = MU * T.pgv_um_s * 1e-6 / CS / 1e3
    T['sigma_d_sw_kPa'] = MU * T.sw_pgv3 * 1e-6 / CS / 1e3
    T['love_frac'] = T.bb_pgvt / np.hypot(T.bb_pgvt, T.bb_pgvr)
    # pure geometric (unit-amplitude) Love and Rayleigh fault-resolution factors
    th = np.radians(T.prop_az.values)
    k = np.vstack([np.sin(th), np.cos(th)])
    tdir = np.vstack([np.cos(th), -np.sin(th)])
    gL, gR = [], []
    for j in range(len(T)):
        tl, _ = fault_stress(np.array([tdir[0, j]]), np.array([tdir[1, j]]), T.prop_az.values[j])
        tr_, _ = fault_stress(np.array([k[0, j]]), np.array([k[1, j]]), T.prop_az.values[j])
        gL.append(abs(tl[0]) / (MU / CS))
        gR.append(abs(tr_[0]) / (MU / CS))
    T['geomL'] = gL
    T['geomR'] = gR
    # overlap: another listed quake with larger measured PGV whose surface waves arrive within 2 h
    ts_ = T.t_surf.values
    pg = T.bb_pgv3.fillna(0).values
    T['overlap_larger'] = [bool(np.any((np.abs(ts_ - ts_[j]) < 7200) & (pg > pg[j]) & (np.arange(len(T)) != j)))
                           for j in range(len(T))]
    # HOPS vs JCC consistency
    return T


def tidal_at(t):
    D = pd.read_csv('data/sci/tidal_stress/tidal_stress_hourly.csv.gz', usecols=['time_utc', 'tau_RL_d45_kPa', 'sn_d45_kPa'])
    tt = (pd.to_datetime(D.time_utc) - EPOCH).dt.total_seconds().values
    tau = np.interp(t, tt, D.tau_RL_d45_kPa.values)
    dtau = (np.interp(t + 1800, tt, D.tau_RL_d45_kPa.values) - np.interp(t - 1800, tt, D.tau_RL_d45_kPa.values))
    amp = pd.Series(D.tau_RL_d45_kPa.values).rolling(25 * 1, center=True).apply(lambda x: x.max() - x.min(), raw=True).values
    return tau, dtau, np.interp(t, tt, amp) / 2, float(np.nanstd(D.tau_RL_d45_kPa.values)), float(np.nanmax(np.abs(D.tau_RL_d45_kPa.values)))


def poisson_ll(o, mu):
    from scipy.special import gammaln
    mu = np.maximum(mu, 1e-12)
    return float(np.sum(o * np.log(mu) - mu - gammaln(o + 1)))


def fit_models(o, e, S, grid, full=True):
    """Poisson models for counts o with matched expectation e as a function of stress S (log10 kPa).
       M0: o ~ P(r0 e);  step: P(e (r0 + r1 [S>=S0]));  additive step: P(r0 e + a [S>=S0]);
       log-ramp: P(r0 e (1 + b max(0, S-S0)));  power: P(r0 e 10^(b S))"""
    res = {}
    r0 = o.sum() / e.sum()
    res['M0'] = dict(r0=r0, LL=poisson_ll(o, r0 * e), k=1)
    best = None
    for S0 in grid:
        hi = S >= S0
        if hi.sum() < 5 or (~hi).sum() < 5:
            continue
        rl = o[~hi].sum() / e[~hi].sum()
        rh = o[hi].sum() / e[hi].sum()
        ll = poisson_ll(o, np.where(hi, rh, rl) * e)
        if best is None or ll > best[0]:
            best = (ll, S0, rl, rh, int(hi.sum()), float(o[hi].sum()), float(e[hi].sum()))
    res['step'] = dict(LL=best[0], S0_log10kPa=best[1], S0_kPa=10 ** best[1], r_below=best[2], r_above=best[3], n_above=best[4],
                       obs_above=best[5], exp_above=best[6], k=3)
    from scipy.optimize import minimize
    f = lambda p: -poisson_ll(o, np.exp(p[0] + p[1] * S) * e)
    r = minimize(f, [0.0, 0.0], method='Nelder-Mead')
    res['power'] = dict(LL=-r.fun, r_at_1kPa=np.exp(r.x[0]), exponent=r.x[1], k=2)
    if not full:
        return res
    # additive excess (extra onsets per quake above threshold, background ratio r0 below)
    best = None
    for S0 in grid:
        hi = S >= S0
        if hi.sum() < 5 or (~hi).sum() < 5:
            continue
        f = lambda p: -poisson_ll(o, np.exp(p[0]) * e + np.where(hi, np.exp(p[1]), 0))
        r = minimize(f, [np.log(max(r0, .1)), np.log(.3)], method='Nelder-Mead')
        if best is None or -r.fun > best[0]:
            best = (-r.fun, S0, np.exp(r.x[0]), np.exp(r.x[1]), int(hi.sum()))
    res['add_step'] = dict(LL=best[0], S0_log10kPa=best[1], S0_kPa=10 ** best[1], r_bg=best[2], extra_onsets_per_quake=best[3],
                           n_above=best[4], k=3)
    best = None
    for S0 in grid:
        f = lambda p: -poisson_ll(o, np.exp(p[0]) * e * (1 + np.exp(p[1]) * np.maximum(0, S - S0)))
        r = minimize(f, [np.log(max(r0, .1)), 0.0], method='Nelder-Mead')
        if best is None or -r.fun > best[0]:
            best = (-r.fun, S0, np.exp(r.x[0]), np.exp(r.x[1]))
    res['ramp'] = dict(LL=best[0], S0_log10kPa=best[1], S0_kPa=10 ** best[1], r0=best[2], slope_per_decade=best[3], k=3)
    for m in res:
        res[m]['AIC'] = 2 * res[m]['k'] - 2 * res[m]['LL']
    return res


def stage_analyze():
    import sci_cascades2 as C
    from scipy import stats
    T = build_table()
    Q = load_quakes()
    # -------- real-only burst onsets (identical construction to remote_check excess_tests 'real')
    m = pd.read_parquet('data/sci/remote_check/catalog_classified.parquet', columns=['t', 'idnum', 'real', 'meancc'])
    C.T_END = m.t.max()
    base = m[m.t >= C.ts(C.T_START)]
    dsub = base[base.real == 1][['t', 'idnum']]
    Bo = np.sort(C.make_bursts(dsub).onset.values)
    rng = np.random.default_rng(44)
    t_lo, t_hi = C.ts(C.T_START), dsub.t.max()
    ctrl = Q.t_surf.values[:, None] + rng.choice(C.SHIFTS, (len(Q), 300))
    ctrl = np.where((ctrl > t_lo) & (ctrl < t_hi - 86400), ctrl, Q.t_surf.values[:, None] - rng.choice(np.abs(C.SHIFTS), (len(Q), 300)))

    def cnt(arr, a, b):
        return np.searchsorted(arr, b) - np.searchsorted(arr, a)
    H = 3600.0
    o = cnt(Bo, Q.t_surf.values, Q.t_surf.values + H)
    cc = cnt(Bo, ctrl, ctrl + H)                 # (nq, 300)
    e = cc.mean(1)
    T['obs_1h'] = o
    T['exp_1h'] = e
    T['ratio_1h'] = o / np.where(e > 0, e, np.nan)
    T['p_any_ctrl'] = (cc > 0).mean(1)
    T['pctl_1h'] = (cc < o[:, None]).mean(1) + 0.5 * (cc == o[:, None]).mean(1)
    # window aligned on measured peak (variant)
    tpk = T.bb_t_pk.fillna(T.t_surf).values
    o_pk = cnt(Bo, tpk - 600, tpk - 600 + H)
    cpk = cnt(Bo, ctrl + (tpk - Q.t_surf.values)[:, None] - 600, ctrl + (tpk - Q.t_surf.values)[:, None] - 600 + H)
    T['obs_pk1h'] = o_pk
    T['exp_pk1h'] = cpk.mean(1)
    # tidal stress at arrival
    tau_t, dtau_t, amp_t, tid_sd, tid_max = tidal_at(Q.t_surf.values)
    T['tidal_tau_kPa'] = tau_t
    T['tidal_dtau_dt_kPa_per_h'] = dtau_t
    summ = dict(check_reproduce_remote_check=dict(obs_all=int(o.sum()), exp_all=float(e.sum()),
                                                  obs_top30=int(o[Q.pgv_rank.values <= 30].sum()),
                                                  exp_top30=float(e[Q.pgv_rank.values <= 30].sum())))
    print(summ)
    T.to_csv(f'{OUT}/per_quake_pgv_stress_triggering.csv', index=False)

    # -------- PGV measurement summary
    ok = T.pgv_um_s.notna()
    both = T.JCC_ok & T.HOPS_ok
    summ['measurement'] = dict(
        stations='BK.JCC (STS-2, 40.817N 124.030W, ~85 km NNW of LFE centroid) primary; BK.HOPS (STS-1/STS-7, 39.0N 123.07W, ~128 km SE) where JCC missing',
        channels='LH? 1 Hz; response removed to velocity; bandpass 0.01-0.45 Hz (LH Nyquist 0.5 Hz); PGV = max 3-C vector |v| in [P, P+1.5 h]',
        n_quakes=len(T), n_measured=int(ok.sum()), n_JCC=int(T.JCC_ok.sum()), n_HOPS_only=int((~T.JCC_ok & T.HOPS_ok).sum()),
        jcc_over_hops_median=float(np.median(T.JCC_bb_pgv3[both] / T.HOPS_bb_pgv3[both])),
        jcc_vs_hops_log10_corr=float(np.corrcoef(np.log10(T.JCC_bb_pgv3[both]), np.log10(T.HOPS_bb_pgv3[both]))[0, 1]),
        jcc_hops_log10_scatter_sd=float(np.std(np.log10(T.JCC_bb_pgv3[both] / T.HOPS_bb_pgv3[both]))),
        measured_over_proxy_median=float(np.median(T.pgv_um_s[ok] / T.pgv_proxy_um_s[ok])),
        measured_vs_proxy_log10_corr=float(np.corrcoef(np.log10(T.pgv_um_s[ok]), np.log10(T.pgv_proxy_um_s[ok]))[0, 1]),
        pgv_um_s_quantiles={q: float(T.pgv_um_s[ok].quantile(q)) for q in [0, .1, .25, .5, .75, .9, 1]},
        noise_um_s_median=float(T.bb_noise3[ok].median()),
        frac_snr_gt3=float((T.snr_bb[ok] > 3).mean()),
        sigma_d_kPa_quantiles={q: float(T.sigma_d_kPa[ok].quantile(q)) for q in [0, .1, .25, .5, .75, .9, 1]},
        fault_tau_over_sigma_d_median=float(np.median(T.bb_tau_abs_kPa[ok] / T.sigma_d_kPa[ok])),
        peak_component_counts=T.pk_comp[ok].value_counts().to_dict(),
        median_T_at_peak_s=float(T.bb_T_pk[ok].median()),
        formula='sigma_d = mu*PGV/c, mu=30 GPa, c=3.5 km/s => 1 um/s -> 8.57 Pa; fault tau from plane-wave plane-stress strain on 293/45NE rake 180',
        top10=T[ok].sort_values('pgv_um_s', ascending=False).head(10)[
            ['time', 'mag', 'place', 'dist_km', 'pgv_proxy_um_s', 'pgv_um_s', 'sigma_d_kPa', 'bb_tau_abs_kPa', 'pk_comp', 'bb_T_pk',
             'obs_1h', 'exp_1h']].round(3).to_dict('records'))

    # -------- triggering vs stress
    U = T[ok & (T.exp_1h > 0)].copy()
    U['S'] = np.log10(U.sigma_d_kPa)
    oo, ee, SS = U.obs_1h.values.astype(float), U.exp_1h.values, U.S.values
    ccU = cc[U.index.values]
    grid = np.arange(-1.6, 1.21, 0.05)
    fits = fit_models(oo, ee, SS, grid)
    summ['fits_sigma_d'] = fits
    # null for dLL(step - M0): replace obs by one random control draw per quake
    rngn = np.random.default_rng(7)
    dnull, dnull_pow, rho_null = [], [], []
    for b in range(1000):
        on = ccU[np.arange(len(U)), rngn.integers(0, 300, len(U))].astype(float)
        f_ = fit_models(on, ee, SS, grid, full=False)
        dnull.append(f_['step']['LL'] - f_['M0']['LL'])
        dnull_pow.append(f_['power']['LL'] - f_['M0']['LL'] if f_['power']['exponent'] > 0 else 0.0)
        rho_null.append(stats.spearmanr(SS, (on - ee) / np.sqrt(ee))[0])
    dobs = fits['step']['LL'] - fits['M0']['LL']
    dobs_r = fits['power']['LL'] - fits['M0']['LL'] if fits['power']['exponent'] > 0 else 0.0
    rho = stats.spearmanr(SS, (oo - ee) / np.sqrt(ee))[0]
    summ['threshold_tests'] = dict(
        dLL_step_obs=dobs, p_step_vs_ctrl_null=float((np.sum(np.array(dnull) >= dobs) + 1) / (len(dnull) + 1)),
        dLL_power_pos_obs=dobs_r, p_power_pos_vs_ctrl_null=float((np.sum(np.array(dnull_pow) >= dobs_r) + 1) / (len(dnull_pow) + 1)),
        spearman_S_vs_pearson_resid=float(rho), p_spearman_ctrl_null=float((np.sum(np.array(rho_null) >= rho) + 1) / (len(rho_null) + 1)),
        note='nulls: obs replaced by a random tidal/time-of-day matched control count for each quake, refit (1000 draws)')
    # bootstrap over quakes
    rngb = np.random.default_rng(11)
    bs = []
    for b in range(1000):
        j = rngb.integers(0, len(U), len(U))
        f_ = fit_models(oo[j], ee[j], SS[j], grid, full=False)
        bs.append((f_['step']['S0_log10kPa'], f_['step']['r_above'], f_['step']['r_below'], f_['power']['exponent']))
    bs = np.array(bs)
    q = lambda x: [float(np.quantile(x, .025)), float(np.quantile(x, .16)), float(np.median(x)), float(np.quantile(x, .84)), float(np.quantile(x, .975))]
    summ['bootstrap_over_quakes'] = dict(
        n=len(bs), quantiles='2.5,16,50,84,97.5',
        step_S0_kPa=[10 ** v for v in q(bs[:, 0])], step_r_above=q(bs[:, 1]), step_r_below=q(bs[:, 2]),
        power_exponent=q(bs[:, 3]),
        frac_step_S0_below_0p5kPa=float(np.mean(bs[:, 0] < np.log10(0.5))),
        frac_power_exponent_gt0=float(np.mean(bs[:, 3] > 0)))
    np.save(f'{OUT}/bootstrap_fits.npy', bs)
    # binned (fixed stress bins) ratios with control-draw nulls + triggering probability
    edges = [0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 1e3]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (U.sigma_d_kPa >= lo) & (U.sigma_d_kPa < hi)
        if sel.sum() == 0:
            continue
        cs = ccU[sel.values]
        jj = rngn.integers(0, 300, (4000, sel.sum()))
        nul = cs[np.arange(sel.sum())[None, :], jj].sum(1)
        ob = int(U.obs_1h[sel].sum())
        # bootstrap over quakes for the ratio
        bb = []
        oi, ei = U.obs_1h[sel].values, U.exp_1h[sel].values
        for _ in range(2000):
            k_ = rngn.integers(0, len(oi), len(oi))
            bb.append(oi[k_].sum() / max(ei[k_].sum(), 1e-9))
        p_any_obs = float((U.obs_1h[sel] > 0).mean())
        p_any_exp = float(U.p_any_ctrl[sel].mean())
        rows.append(dict(lo_kPa=lo, hi_kPa=hi, n=int(sel.sum()), median_kPa=float(U.sigma_d_kPa[sel].median()), obs=ob,
                         exp=float(U.exp_1h[sel].sum()), ratio=ob / float(U.exp_1h[sel].sum()),
                         ratio_boot_ci68=[float(np.quantile(bb, .16)), float(np.quantile(bb, .84))],
                         null_ci95=[float(np.quantile(nul, .025)), float(np.quantile(nul, .975))],
                         p_upper=float((np.sum(nul >= ob) + 1) / (len(nul) + 1)),
                         frac_quakes_ge1=p_any_obs, frac_ctrl_ge1=p_any_exp,
                         trig_prob=(p_any_obs - p_any_exp) / (1 - p_any_exp)))
    Bn = pd.DataFrame(rows)
    Bn.to_csv(f'{OUT}/binned_ratio_vs_stress.csv', index=False)
    summ['binned_sigma_d'] = Bn.round(4).to_dict('records')
    print(Bn.round(3).to_string())

    # cumulative-threshold curve: ratio for quakes with sigma_d >= x
    cum = []
    for x in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 2, 3, 5]:
        sel = (U.sigma_d_kPa >= x).values
        if sel.sum() < 3:
            continue
        cs = ccU[sel]
        jj = rngn.integers(0, 300, (4000, sel.sum()))
        nul = cs[np.arange(sel.sum())[None, :], jj].sum(1)
        ob = int(oo[sel].sum())
        cum.append(dict(min_kPa=x, n=int(sel.sum()), obs=ob, exp=float(ee[sel].sum()), ratio=ob / ee[sel].sum(),
                        p_upper=float((np.sum(nul >= ob) + 1) / (len(nul) + 1)),
                        n_quakes_obs_gt_exp=int((oo[sel] > ee[sel]).sum())))
    summ['cumulative_above_threshold'] = cum
    print(pd.DataFrame(cum).round(3).to_string())

    # -------- other covariates: Poisson regression with offset (log link), bootstrap CIs over quakes
    def pois_irls(y, X, off, it=50):
        b = np.zeros(X.shape[1])
        b[0] = np.log(max(y.sum(), 0.5) / np.exp(off).sum())
        for _ in range(it):
            mu = np.exp(off + X @ b)
            Hm = X.T @ (X * mu[:, None])
            step = np.linalg.solve(Hm + 1e-9 * np.eye(len(b)), X.T @ (y - mu))
            b = b + step
            if np.max(np.abs(step)) < 1e-8:
                break
        mu = np.exp(off + X @ b)
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * mu[:, None]))))
        return b, se

    def glm(cols, data, name):
        X = np.column_stack([np.ones(len(data))] + [data[c].values.astype(float) for c in cols])
        y, off = data.obs_1h.values.astype(float), np.log(data.exp_1h.values)
        params, bse = pois_irls(y, X, off)
        # quake bootstrap
        bb = []
        rr = np.random.default_rng(5)
        for _ in range(500):
            j = rr.integers(0, len(data), len(data))
            try:
                bb.append(pois_irls(y[j], X[j], off[j])[0])
            except Exception:
                pass
        bb = np.array(bb)
        return {name: dict(cols=cols, coef=[float(x) for x in params], se_poisson=[float(x) for x in bse],
                           boot_ci95=[[float(np.quantile(bb[:, k], .025)), float(np.quantile(bb[:, k], .975))] for k in range(bb.shape[1])],
                           n=len(data))}
    U['logS'] = U.S
    U['logtauF'] = np.log10(U.bb_tau_abs_kPa)
    U['love_frac_c'] = U.love_frac - U.love_frac.mean()
    U['logT'] = np.log10(U.bb_T_pk)
    U['tid'] = U.tidal_tau_kPa
    U['tid_rate'] = U.tidal_dtau_dt_kPa_per_h
    U['geomL_c'] = U.geomL
    U['above'] = (U.sigma_d_kPa >= 0.5).astype(float)
    cov = {}
    cov.update(glm(['logS'], U, 'logS'))
    cov.update(glm(['logtauF'], U, 'log_fault_tau'))
    cov.update(glm(['logS', 'love_frac'], U, 'logS+love_frac'))
    cov.update(glm(['logS', 'logT'], U, 'logS+logPeriod'))
    cov.update(glm(['logS', 'tid'], U, 'logS+tidal_tau'))
    cov.update(glm(['logS', 'tid_rate'], U, 'logS+tidal_dtau_dt'))
    cov.update(glm(['logS', 'geomL'], U, 'logS+Love_geometry_factor'))
    cov.update(glm(['above', 'tid'], U, 'above0.5kPa+tidal_tau'))
    # split tests among quakes above 0.3 kPa (where any triggering appears)
    A = U[U.sigma_d_kPa >= 0.3]
    splits = {}
    for nm, sel in {'tidal_tau_pos': A.tidal_tau_kPa > 0, 'tidal_tau_neg': A.tidal_tau_kPa <= 0,
                    'love_dominant(pk T)': A.pk_comp == 'T', 'rayleigh_dominant(pk Z/R)': A.pk_comp != 'T',
                    'love_geom_favourable(geomL>0.5)': A.geomL > 0.5, 'love_geom_unfavourable(geomL<=0.5)': A.geomL <= 0.5,
                    'period_ge_20s': A.bb_T_pk >= 20, 'period_lt_20s': A.bb_T_pk < 20,
                    'fault_tau_ge_median': A.bb_tau_abs_kPa >= A.bb_tau_abs_kPa.median(),
                    'fault_tau_lt_median': A.bb_tau_abs_kPa < A.bb_tau_abs_kPa.median()}.items():
        splits[nm] = dict(n=int(sel.sum()), obs=int(A.obs_1h[sel].sum()), exp=float(A.exp_1h[sel].sum()),
                          ratio=float(A.obs_1h[sel].sum() / A.exp_1h[sel].sum()) if A.exp_1h[sel].sum() > 0 else None)
    summ['covariate_glm'] = cov
    summ['splits_sigma_d_ge_0p3kPa'] = splits
    # fault-resolved stress as the predictor
    fitsF = fit_models(oo, ee, np.log10(U.bb_tau_abs_kPa.values), grid)
    summ['fits_fault_tau'] = fitsF
    # surface-wave-band (0.02-0.1 Hz) variant and peak-aligned window variant
    fitsSW = fit_models(oo, ee, np.log10(U.sigma_d_sw_kPa.values), grid)
    summ['fits_sigma_d_sw_band'] = fitsSW
    Upk = U[U.exp_pk1h > 0]
    summ['fits_sigma_d_peak_window'] = fit_models(Upk.obs_pk1h.values.astype(float), Upk.exp_pk1h.values, Upk.S.values, grid)
    # proxy for comparison
    summ['fits_pgv_proxy'] = fit_models(oo, ee, np.log10(MU * U.pgv_proxy_um_s.values * 1e-6 / CS / 1e3), grid)
    # high-SNR only
    hs = (U.snr_bb > 3).values
    summ['fits_sigma_d_snr_gt3'] = fit_models(oo[hs], ee[hs], SS[hs], grid)
    nov = ~U.overlap_larger.values.astype(bool)
    summ['fits_sigma_d_no_overlap'] = fit_models(oo[nov], ee[nov], SS[nov], grid)
    summ['n_overlap_larger'] = int((~nov).sum())
    # rate-state (Dieterich 1994) prediction from the measured fault tau(t): during a transient short compared
    # with t_a, the count triggered is r t_a ln(1 + X/t_a), X = int(exp(tau/a sigma) - 1) dt; ratio over 1 h:
    #   ratio_pred = 1 + (t_a/3600) ln(1 + X/t_a)   (shadow after the waves neglected within the hour)
    rows = []
    top = (U.sigma_d_kPa >= 0.5).values
    for pre in ['lnX', 'lnXcfs']:
        for asg in ASG:
            lnX = U[f'{pre}_as{asg}'].values
            for ta in TAG:
                rp = 1 + (ta / 3600.0) * np.logaddexp(0, lnX - np.log(ta))
                rows.append(dict(stress=pre[2:], a_sigma_kPa=asg, t_a_s=ta, tau_dot_kPa_per_day=asg / ta * 86400,
                                 LL=poisson_ll(oo, ee * rp), pred_ratio_all=float((ee * rp).sum() / ee.sum()),
                                 pred_ratio_ge0p5=float((ee * rp)[top].sum() / ee[top].sum()),
                                 pred_ratio_max=float(rp.max())))
    RS = pd.DataFrame(rows)
    RS['dLL_vs_null'] = RS.LL - poisson_ll(oo, ee)
    RS.to_csv(f'{OUT}/rate_state_grid.csv', index=False)
    bestRS = RS.loc[RS[RS.stress == 'X'].LL.idxmax()]
    rs = dict(best_tau=bestRS.to_dict(), best_cfs=RS.loc[RS[RS.stress == 'Xcfs'].LL.idxmax()].to_dict(),
              at_a_sigma_0p5={str(int(ta)): RS[(RS.stress == 'X') & (RS.a_sigma_kPa == 0.5) & (RS.t_a_s == ta)].iloc[0][
                  ['LL', 'dLL_vs_null', 'pred_ratio_all', 'pred_ratio_ge0p5', 'pred_ratio_max']].to_dict() for ta in TAG},
              obs_ratio_all=float(oo.sum() / ee.sum()), obs_ratio_ge0p5=float(oo[top].sum() / ee[top].sum()),
              note='Poisson LL is anti-conservative (onsets cluster); use for relative comparison only')
    summ['rate_state_prediction'] = rs
    for lab, row in [('rs_pred_best', bestRS),
                     ('rs_pred_as0.5', RS.loc[RS[(RS.stress == 'X') & (RS.a_sigma_kPa == 0.5)].LL.idxmax()])]:
        lnX = T[f"lnX_as{row.a_sigma_kPa}"].values
        T[lab] = 1 + (row.t_a_s / 3600.0) * np.logaddexp(0, lnX - np.log(row.t_a_s))
    summ['rate_state_prediction']['best_at_a_sigma_0p5'] = RS.loc[RS[(RS.stress == 'X') & (RS.a_sigma_kPa == 0.5)].LL.idxmax()].to_dict()
    T.to_csv(f'{OUT}/per_quake_pgv_stress_triggering.csv', index=False)
    print(RS.sort_values('LL', ascending=False).head(12).round(3).to_string())

    # -------- tidal comparison and local quakes
    D = pd.read_csv('data/sci/dose_response/per_quake_drivers_responses.csv')
    D['sigma_d_kPa'] = MU * D.pgv_rrup * 1e-2 / CS / 1e3
    D['sigma_d_from_strain_kPa'] = MU * D.dyn_strain_ue * 1e-6 / 1e3
    loc = local_quake_windows(Bo, C, t_hi)
    summ['local_quakes'] = dict(
        table=D[['quake', 'dist_km', 'mag', 'pgv_rrup', 'sigma_d_kPa', 'dtau', 'dCFS04', 'ratio30_all']].round(3).to_dict('records'),
        pgv_units='cm/s (BSSA14 at Rrup)', static_dtau_units='kPa (Okada, from dose_response)',
        early_windows=loc)
    summ['tidal'] = dict(tau_RL_d45_sd_kPa=tid_sd, tau_RL_d45_max_abs_kPa=tid_max,
                         note='body-tide plane stress, same convention as the dynamic fault-resolved stress here')
    json.dump(summ, open(f'{OUT}/summary.json', 'w'), indent=1, default=float)
    print(json.dumps({k: summ[k] for k in ['fits_sigma_d', 'threshold_tests', 'bootstrap_over_quakes', 'splits_sigma_d_ge_0p3kPa']},
                     indent=1, default=float))


def local_quake_windows(Bo, C, t_hi):
    """real-only onsets in [t0+60 s, t0+1 h) and [t0, t0+24 h) after the local M>=5.6 quakes, vs the same
       tidal/time-of-day matched controls (excluding controls within 3 d of the quake)."""
    import json as _j
    out = []
    rng = np.random.default_rng(3)
    for f in sorted(glob.glob('data/sci/dose_response/usgs_raw/nc*.json')):
        try:
            j = _j.load(open(f))
            p = j['properties'] if 'properties' in j else j
            t0 = p['time'] / 1000.0
            mag = p['mag']
        except Exception:
            continue
        if t0 < C.ts(C.T_START) or t0 > t_hi - 86400:
            continue
        sh = rng.choice(C.SHIFTS, 300)
        ct = t0 + sh
        ct = ct[(ct > C.ts(C.T_START)) & (ct < t_hi - 86400)]
        r = dict(id=os.path.basename(f)[:-5], mag=mag, time=str(pd.Timestamp(t0, unit='s')))
        for lab, a, b in [('0-1h', 60, 3600), ('1-24h', 3600, 86400)]:
            o = int(np.searchsorted(Bo, t0 + b) - np.searchsorted(Bo, t0 + a))
            c = np.searchsorted(Bo, ct + b) - np.searchsorted(Bo, ct + a)
            r[f'obs_{lab}'] = o
            r[f'exp_{lab}'] = float(c.mean())
            r[f'p_upper_{lab}'] = float((np.sum(c >= o) + 1) / (len(c) + 1))
            r[f'p_lower_{lab}'] = float((np.sum(c <= o) + 1) / (len(c) + 1))
        out.append(r)
    return out


# ----------------------------------------------------------------------------- figures
def stage_fig():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    T = pd.read_csv(f'{OUT}/per_quake_pgv_stress_triggering.csv')
    S = json.load(open(f'{OUT}/summary.json'))
    Bn = pd.read_csv(f'{OUT}/binned_ratio_vs_stress.csv')
    ok = T.pgv_um_s.notna()
    # ---- figure 1: measurement
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    a = ax[0]
    sc = a.scatter(T.pgv_proxy_um_s[ok], T.pgv_um_s[ok], s=9, c=np.log10(T.snr_bb[ok]), cmap='viridis', vmin=0, vmax=2)
    lim = [1, 2e4]
    a.plot(lim, lim, 'k--', lw=.8)
    a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(lim); a.set_ylim(lim)
    a.set_xlabel('Ms-type PGV proxy (um/s, cascades2)')
    a.set_ylabel('measured PGV, 3-C, 0.01-0.45 Hz (um/s)\n[x 8.57e-3 = mu PGV/c in kPa]')
    plt.colorbar(sc, ax=a, label='log10 SNR (vs pre-P noise)')
    a.set_title(f'{ok.sum()} M>=6.5 teleseisms, BK.JCC (HOPS fallback)')
    a = ax[1]
    both = T.JCC_ok & T.HOPS_ok
    a.loglog(T.JCC_bb_pgv3[both] * 1e6, T.HOPS_bb_pgv3[both] * 1e6, '.', ms=3)
    a.plot([1, 2e4], [1, 2e4], 'k--', lw=.8)
    a.set_xlabel('PGV JCC (um/s)'); a.set_ylabel('PGV HOPS (um/s)')
    a.set_title(f'station consistency (n={both.sum()})')
    a = ax[2]
    a.loglog(T.sigma_d_kPa[ok], T.bb_tauL_abs_kPa[ok], '.', ms=3, label='Love-only fault tau')
    a.loglog(T.sigma_d_kPa[ok], T.bb_tauR_abs_kPa[ok], '.', ms=3, label='Rayleigh(radial)-only fault tau')
    a.loglog(T.sigma_d_kPa[ok], T.bb_tau_abs_kPa[ok], 'k.', ms=2, label='total fault tau')
    a.plot([1e-3, 1e2], [1e-3, 1e2], 'k--', lw=.8)
    a.set_xlabel('sigma_d = mu PGV/c (kPa)'); a.set_ylabel('peak |tau| on 293/45NE, rake 180 (kPa)')
    a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig('figures/sci_dyn_trigger_pgv.png', dpi=140)
    plt.close(fig)

    # ---- figure 2: triggering vs stress
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    a = ax[0]
    x = Bn.median_kPa.values
    a.errorbar(x, Bn.ratio, yerr=[Bn.ratio - [v[0] for v in map(json.loads, Bn.ratio_boot_ci68)],
                                  [v[1] for v in map(json.loads, Bn.ratio_boot_ci68)] - Bn.ratio], fmt='o', color='C0', capsize=3,
               label='obs/exp, real-only onsets 0-1 h (68% quake bootstrap)')
    for xi, r_, n_ in zip(x, Bn.ratio, Bn.n):
        a.annotate(f'n={n_}', (xi, r_), textcoords='offset points', xytext=(4, 6), fontsize=7)
    nl = np.array([json.loads(v) for v in Bn.null_ci95])
    a.fill_between(x, nl[:, 0] / Bn.exp, nl[:, 1] / Bn.exp, color='0.8', alpha=.6, label='95% matched-control null')
    f = S['fits_sigma_d']
    xx = np.logspace(-2.5, 1.7, 200)
    a.plot(xx, np.where(xx >= f['step']['S0_kPa'], f['step']['r_above'], f['step']['r_below']), 'C3-',
           label=f"step fit: S0={f['step']['S0_kPa']:.2f} kPa")
    xr = xx[(xx >= T.sigma_d_kPa[ok].min()) & (xx <= T.sigma_d_kPa[ok].max())]
    a.plot(xr, f['power']['r_at_1kPa'] * xr ** f['power']['exponent'], 'C2--', label=f"power fit: exponent {f['power']['exponent']:.2f}")
    rsb = S['rate_state_prediction']['best_tau']
    for col, lab, sty in [('rs_pred_best', f"rate-state best (a-sigma {rsb['a_sigma_kPa']} kPa, t_a {rsb['t_a_s']:.0f} s)", 'C1s:'),
                          ('rs_pred_as0.5', 'rate-state, a-sigma 0.5 kPa (best t_a)', 'C4^:')]:
        pr = []
        for lo, hi in zip(Bn.lo_kPa, Bn.hi_kPa):
            sel = ok & (T.sigma_d_kPa >= lo) & (T.sigma_d_kPa < hi) & (T.exp_1h > 0)
            pr.append((T.exp_1h[sel] * T[col][sel]).sum() / T.exp_1h[sel].sum())
        a.plot(x, pr, sty, ms=5, label=lab)
    bq = S['bootstrap_over_quakes']['step_S0_kPa']
    a.axvspan(bq[1], bq[3], color='C3', alpha=.12, label='S0 68% bootstrap')
    a.axhline(1, color='k', lw=.6)
    a.set_xscale('log'); a.set_xlabel('measured dynamic stress mu PGV/c (kPa)'); a.set_ylabel('observed / expected onsets')
    a.legend(fontsize=6.5, loc='upper left')
    a.set_title('triggering vs dynamic stress (teleseisms)')
    a = ax[1]
    a.plot(Bn.median_kPa, Bn.trig_prob, 'o-')
    a.axhline(0, color='k', lw=.6)
    a.set_xscale('log'); a.set_xlabel('dynamic stress (kPa)'); a.set_ylabel('excess P(>=1 onset in 1 h)')
    a.set_title('triggering probability (vs matched controls)')
    # ---- stress scale comparison
    a = ax[2]
    D = pd.DataFrame(S['local_quakes']['table'])
    items = [('tidal tau (sd)', S['tidal']['tau_RL_d45_sd_kPa'], 'C0'), ('tidal tau (max)', S['tidal']['tau_RL_d45_max_abs_kPa'], 'C0'),
             ('a-sigma (tides, M7)', 0.5, 'C1'),
             ('teleseism step S0', f['step']['S0_kPa'], 'C3'),
             ('teleseism median', float(T.sigma_d_kPa[ok].median()), 'C3'),
             ('largest teleseism (El Mayor 2010)', float(T.sigma_d_kPa[ok].max()), 'C3')]
    for r_ in D.itertuples():
        items.append((f'{r_.quake} dyn', r_.sigma_d_kPa, 'k'))
    for r_ in D[D.quake.isin(['2021-12-20 M6.2', '2022-12-20 M6.4', '2024-12-05 M7.0'])].itertuples():
        items.append((f'{r_.quake} static dtau', abs(r_.dtau), '0.5'))
    items = sorted(items, key=lambda z: z[1])
    a.barh(range(len(items)), [z[1] for z in items], color=[z[2] for z in items])
    a.set_yticks(range(len(items))); a.set_yticklabels([z[0] for z in items], fontsize=6.5)
    a.set_xscale('log'); a.set_xlabel('stress (kPa)')
    a.set_title('stress scales at the LFE source')
    fig.tight_layout()
    fig.savefig('figures/sci_dyn_trigger_threshold.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    stages = sys.argv[1:] or ['download', 'measure', 'analyze', 'fig']
    for s in stages:
        {'download': stage_download, 'measure': stage_measure, 'analyze': stage_analyze, 'fig': stage_fig}[s]()
