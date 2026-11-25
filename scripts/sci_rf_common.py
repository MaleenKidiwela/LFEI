#!/usr/bin/env python3
"""Shared configuration / helpers for the time-lapse teleseismic P receiver-function study (scripts/sci_rf_*.py).

Stages:
  sci_rf_catalog.py   USGS event catalogue (M>=5.5, 30-95 deg) + iasp91 P times / ray parameters
  sci_rf_download.py  event-window waveforms (P-30..P+90 s) -> /data/wsd02/maleen_data/MTJ_RF/wf/{evid}/
  sci_rf_compute.py   preprocessing, rotation ZNE->ZRT, deconvolution (iterative time domain), QC -> rf_{STA}.npz
  sci_rf_static.py    static images (back-azimuth stacks, harmonic decomposition, phase picks vs slab model)
  sci_rf_timelapse.py time-lapse comparison at matched geometry, bootstrap / placebo nulls, injection tests
"""
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import os
import numpy as np
import pandas as pd

ROOT = '/home/seismic/LFE'
OUT = os.environ.get('RF_OUT', f'{ROOT}/data/sci/rf_timelapse')     # override for tests
FIG = os.environ.get('RF_FIG', f'{ROOT}/figures')
WF = '/data/wsd02/maleen_data/MTJ_RF/wf'
INV_XML = f'{ROOT}/data/sci/noise_dvv_sc_full/stations_ncedc.xml'   # cached NCEDC StationXML with responses
CENTER = (40.30, -123.90)          # reference point (lat, lon) for catalogue distance selection
T0, T1 = '2008-01-01', '2026-11-23'
PRE, POST = 30.0, 90.0             # window around P (s)

# net, sta, start, end, channel pattern.  NC broadband HH data exist from 2008 (earlier NCEDC epochs), so the
# NC stations are used from 2008; instrument epochs are treated as possible breaks.
STATIONS = {
    'PB.B049': ('2008-11-01', None, 'EH?'), 'PB.B934': ('2008-10-01', None, 'EH?'),
    'PB.B933': ('2008-09-01', None, 'EH?'), 'PB.B045': ('2008-09-01', None, 'EH?'),
    'PB.B047': ('2008-09-01', None, 'EH?'), 'PB.B046': ('2008-08-01', None, 'EH?'),
    'PB.B932': ('2008-10-01', None, 'EH?'), 'PB.B935': ('2008-10-01', None, 'EH?'),
    'NC.KMR': ('2008-01-01', '2025-09-17T23:18', 'HH?'), 'NC.KMPB': ('2008-01-01', '2026-07-01', 'HH?'),
    'NC.KCPB': ('2008-01-01', '2022-11-15T18:19', 'HH?'), 'NC.KCPC': ('2023-06-01', None, 'HH?'),
    'BK.BRIC': ('2019-02-28', None, 'HH?'), 'BK.RVIT': ('2019-10-01', None, 'HH?'),
    'BK.HALS': ('2020-06-19', None, 'HH?'), 'BK.RBOW': ('2020-06-18', None, 'HH?'),
    'BK.HAYF': ('2019-11-19', None, 'HH?'), 'BK.BJES': ('2019-08-22', None, 'HH?'),
    'BK.SPRL': ('2023-09-13', None, 'HH?'), 'BK.USAL': ('2024-03-21', None, 'HH?'),
    'BK.BLAS': ('2022-06-29', None, 'HH?'), 'BK.GHOP': ('2022-05-05', None, 'HH?'),
    'BK.HUNT': ('2022-06-28', None, 'HH?'), 'BK.PRDS': ('2022-09-09', None, 'HH?'),
    'BK.SHWD': ('2023-02-02', None, 'HH?'), 'BK.ETSL': ('2024-07-30', '2025-08-14', 'HH?'),
    'BK.JEWT': ('2023-08-23', None, 'HH?'),
}
PILOT = ['PB.B049', 'NC.KMR', 'PB.B047', 'NC.KMPB', 'BK.BRIC', 'BK.JEWT']

QUAKES = {'Petrolia': '2021-12-20T20:10', 'Ferndale': '2022-12-20T10:34', 'M7': '2024-12-05T18:44'}
QUAKE_LONG = {'Petrolia': '2021-12-20 Petrolia M6.2', 'Ferndale': '2022-12-20 Ferndale M6.4',
              'M7': '2024-12-05 offshore Cape Mendocino M7.0'}
# instrument breaks used elsewhere in this project + metadata epochs; instrument-problem windows from the
# shear-wave-splitting orientation monitor (scripts/sci_sws.py EXCLUDE; NOTES.md)
BREAKS = {'PB.B933': ['2021-11-01'], 'BK.BRIC': ['2024-05-22'], 'BK.RVIT': ['2024-06-21'],
          'BK.BJES': ['2024-05-23'], 'NC.KMPB': ['2023-10-17', '2010-11-02', '2015-10-27'],
          'NC.KMR': ['2009-09-16', '2015-10-27', '2025-09-17'], 'NC.KCPB': ['2010-11-01', '2015-10-29']}
SWS_EXCLUDE = {'PB.B045': [('2000-01-01', '2011-01-01')],        # EH1 polarity reversal 2009-2010
               'PB.B933': [('2017-01-01', '2021-11-01'), ('2024-08-01', '2030-01-01')],
               'PB.B935': [('2024-05-01', '2030-01-01')],
               'PB.B047': [('2000-01-01', '2009-03-01')],
               'NC.KMPB': [('2000-01-01', '2010-11-03')],
               'NC.KMR': [('2000-01-01', '2009-09-17'), ('2025-09-17', '2030-01-01')],
               'BK.BRIC': [('2022-12-01', '2024-05-22')],
               'BK.RVIT': [('2022-12-20T10:34', '2024-06-21')],
               'BK.BJES': [('2022-12-20T10:34', '2024-05-23')]}


def station_active(code, t):
    s0, s1, _ = STATIONS[code]
    t = pd.Timestamp(t)
    return t >= pd.Timestamp(s0) and (s1 is None or t <= pd.Timestamp(s1))


def excluded(code, t):
    t = pd.Timestamp(t)
    return any(pd.Timestamp(a) <= t < pd.Timestamp(b) for a, b in SWS_EXCLUDE.get(code, []))


_INV = None


def inventory():
    global _INV
    if _INV is None:
        from obspy import read_inventory
        _INV = read_inventory(INV_XML)
    return _INV


def coords():
    """{code: (lat, lon, elev_m, depth_m)} from the cached inventory (latest channel epoch)."""
    out = {}
    inv = inventory()
    for code in STATIONS:
        n, s = code.split('.')
        sel = inv.select(network=n, station=s)
        ch = [c for nn in sel for ss in nn for c in ss if c.code[-1] == 'Z']
        if not ch:
            continue
        c = sorted(ch, key=lambda c: c.start_date)[-1]
        out[code] = (c.latitude, c.longitude, c.elevation, c.depth)
    return out


def ev_dir(evid):
    return os.path.join(WF, evid)


# ----------------------------------------------------------------------------- RF loading / QC / moveout
QC = dict(snr_z=3.0, fit=80.0, p0_min=0.05, p0_max=1.2, pre_max=0.5, cc_min=0.3, rms_mad=4.0)
P_REF = 0.06                       # s/km, reference ray parameter for moveout correction
# 1D model for Ps moveout (forearc crust above the slab; Vp/Vs 1.8): (top depth km, Vp km/s)
MO_MODEL = [(0.0, 3.5), (1.0, 5.0), (5.0, 6.0), (20.0, 6.6), (30.0, 7.2), (45.0, 8.0)]
MO_VPVS = 1.80


def _tps_table(p, zmax=80.0, dz=0.05):
    z = np.arange(0, zmax + dz, dz)
    tops = np.array([m[0] for m in MO_MODEL]); vps = np.array([m[1] for m in MO_MODEL])
    vp = vps[np.searchsorted(tops, z, side='right') - 1]
    vs = vp / MO_VPVS
    q = np.sqrt(1 / vs ** 2 - p ** 2) - np.sqrt(1 / vp ** 2 - p ** 2)
    return z, np.concatenate([[0], np.cumsum(q[:-1] * dz)])


def moveout(rf, t, p, p_ref=P_REF):
    """Ps moveout correction of RF(s) (..., nt) at ray parameter p (s/km) to p_ref (t>=0 only)."""
    z, tp = _tps_table(p)
    _, tr = _tps_table(p_ref)
    pos = t >= 0
    t_src = np.interp(t[pos], tr, tp)                 # time at p of the conversion depth that has time t at p_ref
    out = np.array(rf, dtype=np.float64, copy=True)
    flat = out.reshape(-1, out.shape[-1])
    src = np.array(rf, dtype=np.float64).reshape(-1, out.shape[-1])
    for k in range(flat.shape[0]):
        flat[k, pos] = np.interp(t_src, t, src[k])
    return out


def predicted_ps(depth, vs, vpvs, p=P_REF):
    """Ps delay (s) for a stack of layers (thickness km, Vs, Vp/Vs)."""
    t = 0.0
    for h, s, r in zip(depth, vs, vpvs):
        t += h * (np.sqrt(1 / s ** 2 - p ** 2) - np.sqrt(1 / (s * r) ** 2 - p ** 2))
    return t


def load_rf(code, qc=QC, gauss_sel=2.5, excl=True, verbose=False):
    """-> meta DataFrame (selected events, with QC columns), rf array (n, ngauss, 2, nt) moveout-corrected,
    raw rf array, t, gauss.  Selection: SNR_Z, fit at gauss_sel, direct-P amplitude, pre-P energy, CC with the
    back-azimuth-sector median, RMS outliers; SWS instrument-problem windows and clipped records removed."""
    import json as _json
    fn = os.path.join(OUT, f'rf_{code}.npz')
    d = np.load(fn)
    t = d['t']; g = list(d['gauss']); rf = d['rf']
    m = pd.DataFrame(_json.loads(str(d['meta'])))
    m['time'] = pd.to_datetime(m['time'])
    ig = g.index(gauss_sel)
    R = rf[:, ig, 0]
    i0 = np.argmin(np.abs(t))
    p0 = R[:, (t >= -0.5) & (t <= 0.5)].max(1)
    pre = np.abs(R[:, t < -1.0]).max(1)
    m['p0'] = p0; m['pre_ratio'] = pre / np.maximum(p0, 1e-6)
    keep = (m.snr_z >= qc['snr_z']) & (m[f'fit_r{gauss_sel:g}'] >= qc['fit']) & \
           (m.p0 >= qc['p0_min']) & (m.p0 <= qc['p0_max']) & (m.pre_ratio <= qc['pre_max'])
    if 'clip' in m:
        keep &= ~m['clip'].fillna(False).astype(bool)
    if excl:
        keep &= ~m.excl.astype(bool)
    m['k_basic'] = keep.values
    # outliers: CC (0-10 s, after moveout) with the median of the same 30-deg back-azimuth sector (all kept events)
    Rm = moveout(R, t, P_REF)  # placeholder shape for indexing; replaced per event below
    for k in range(len(m)):
        Rm[k] = moveout(R[k], t, m.p.iloc[k])
    w = (t >= 0) & (t <= 10)
    sect = (m.baz // 30).astype(int).values
    cc = np.full(len(m), np.nan); rms = np.sqrt(np.mean(Rm[:, w] ** 2, 1))
    for s in np.unique(sect):
        idx = np.where((sect == s) & keep.values)[0]
        if len(idx) < 3:
            idx_ref = np.where(keep.values)[0]
        else:
            idx_ref = idx
        med = np.median(Rm[idx_ref][:, w], 0)
        for k in np.where(sect == s)[0]:
            a = Rm[k, w] - Rm[k, w].mean(); b = med - med.mean()
            cc[k] = np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b) + 1e-30)
    m['cc_med'] = cc; m['rms'] = rms
    kr = keep.values
    med_r = np.median(rms[kr]) if kr.any() else 0
    mad = np.median(np.abs(rms[kr] - med_r)) * 1.4826 if kr.any() else 1
    keep &= (m.cc_med >= qc['cc_min']) & (m.rms <= med_r + qc['rms_mad'] * mad)
    m['keep'] = keep.values
    if verbose:
        print(code, len(m), 'basic', int(m.k_basic.sum()), 'final', int(keep.sum()))
    sel = np.where(keep.values)[0]
    rfm = np.empty((len(sel),) + rf.shape[1:], np.float32)
    for j, k in enumerate(sel):
        rfm[j] = moveout(rf[k], t, m.p.iloc[k])
    return m.iloc[sel].reset_index(drop=True), rfm, rf[sel], t, g, m
