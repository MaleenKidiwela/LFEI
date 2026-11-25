#!/usr/bin/env python3
"""Cross-check of our modified Wiener deconvolution (sci_rf_compute.wiener_mod) against RfPy 0.1.2
(RFData.deconvolve(method='wiener'), pre-event-noise-regularised spectral division, Audet 2010).
  --dump  (lfe env): for one station, the 40 highest-SNR events: gain-corrected, 20-Hz Z/N/E (P-30..P+90 s) as
          produced by sci_rf_compute.prep_station (ZRT rotated back to ZNE) + our Wiener RFs -> rfpy_check_{STA}.npz
  --check (rf env): RfPy RFData (iasp91 P time, ZRT rotation, 'wiener'), both RFs low-passed identically
          (cosine 4-6 Hz); per-event and stacked correlation (0-8 s), amplitude ratio, and the 1.25-3 Hz slab-window
          power of both -> rfpy_check_{STA}.csv / .png
Usage: lfe: sci_rf_rfpy_check.py --dump NC.KMPB ; rf: sci_rf_rfpy_check.py --check NC.KMPB
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = '/home/seismic/LFE'
OUT = os.environ.get('RF_OUT', f'{ROOT}/data/sci/rf_timelapse')
FIG = os.environ.get('RF_FIG', f'{ROOT}/figures')


def dump(code, nev=40):
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    import sci_rf_compute as R
    from sci_rf_common import coords, inventory, ev_dir
    from obspy import read, Stream, UTCDateTime
    from obspy.geodetics import gps2dist_azimuth, locations2degrees
    d = np.load(os.path.join(OUT, f'rf_{code}.npz'))
    m = pd.DataFrame(json.loads(str(d['meta'])))
    m = m[~m.excl.astype(bool)].sort_values('snr_z', ascending=False).head(nev)
    E = pd.read_csv(os.path.join(OUT, 'events.csv')).set_index('evid')
    C = coords(); la, lo = C[code][:2]
    inv = inventory()
    out = dict(Z=[], N=[], E=[], rfw=[], t0=[], evid=[], lat=[], lon=[], dep=[], mag=[], otime=[])
    for r in m.itertuples():
        ev = E.loc[r.evid]
        st = Stream()
        for nm in ('ncedc', 'pnw'):
            f = os.path.join(ev_dir(r.evid), f'{nm}.mseed')
            if os.path.exists(f):
                st += read(f)
        t0 = UTCDateTime(pd.Timestamp(ev.time).to_pydatetime())
        dist = locations2degrees(la, lo, ev.lat, ev.lon)
        baz = gps2dist_azimuth(la, lo, ev.lat, ev.lon)[1]
        tp = t0 + ev.tP + ev.p_sdeg * (dist - ev.dist)
        x, why = R.prep_station(st, code, tp, baz, inv)
        if x is None:
            continue
        b = np.radians(baz)
        N = -x['R'] * np.cos(b) + x['T'] * np.sin(b)
        Ee = -x['R'] * np.sin(b) - x['T'] * np.cos(b)
        out['Z'].append(x['Z']); out['N'].append(N); out['E'].append(Ee); out['rfw'].append(R.wiener_mod(x))
        out['t0'].append(float(tp - R.PRE)); out['evid'].append(r.evid); out['lat'].append(ev.lat); out['lon'].append(ev.lon)
        out['dep'].append(ev.depth); out['mag'].append(ev.mag); out['otime'].append(float(t0.timestamp))
    np.savez(os.path.join(OUT, f'rfpy_check_{code}.npz'), **{k: np.array(v) for k, v in out.items()},
             sta_lat=la, sta_lon=lo, t_rf=R.T_RF)
    print('dumped', len(out['evid']))


def check(code):
    import scipy.signal, scipy.signal.windows
    if not hasattr(scipy.signal, 'hann'):         # obspy in the rf env still calls scipy.signal.hann (removed in scipy)
        scipy.signal.hann = scipy.signal.windows.hann
    from types import SimpleNamespace
    from obspy import Stream, Trace, UTCDateTime
    from obspy.core.event import Event, Origin, Magnitude
    from rfpy import RFData
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    import sci_rf_model_raysum as MR
    d = np.load(os.path.join(OUT, f'rfpy_check_{code}.npz'))
    sta = SimpleNamespace(latitude=float(d['sta_lat']), longitude=float(d['sta_lon']), station=code.split('.')[1],
                          network=code.split('.')[0], location=[''], channel='HH', azcorr=0., status='open')
    trf = d['t_rf']
    rows, A, B = [], [], []
    for i in range(len(d['evid'])):
        ev = Event(origins=[Origin(time=UTCDateTime(float(d['otime'][i])), latitude=float(d['lat'][i]),
                                   longitude=float(d['lon'][i]), depth=float(d['dep'][i]) * 1000)],
                   magnitudes=[Magnitude(mag=float(d['mag'][i]))])
        rfd = RFData(sta)
        if not rfd.add_event(ev, returned=True):
            continue
        st = Stream()
        for c in 'ZNE':
            st += Trace(np.array(d[c][i], float), header=dict(network=sta.network, station=sta.station,
                                                              channel=f'HH{c}', sampling_rate=20.0,
                                                              starttime=UTCDateTime(float(d['t0'][i]))))
        rfd.add_data(st, new_sr=20.0)
        rfd.rotate(align='ZRT')
        rfd.calc_snr()
        rfd.deconvolve(method='wiener')
        rr = rfd.rf.select(component='R')[0].data
        n = len(rr)
        tt = (np.arange(n) - n // 2) / 20.0                      # ifftshift: zero lag at the centre
        k = np.argmax(rfd.rf.select(component='Z')[0].data)
        rr = np.interp(trf, tt - tt[k], rr) / rfd.rf.select(component='Z')[0].data[k]
        a = MR.lowpass(rr)
        b = np.array(d['rfw'][i][0], float)
        w = (trf >= 0) & (trf <= 8)
        cc = np.corrcoef(a[w], b[w])[0, 1]
        ws = (trf >= 1.5) & (trf <= 4.5)
        pa = MR.band_power(a, trf, (1.5, 4.5), (1.25, 3.0)); pb = MR.band_power(b, trf, (1.5, 4.5), (1.25, 3.0))
        rows.append(dict(evid=d['evid'][i], cc_0_8s=cc, amp_ratio=np.dot(a[w], b[w]) / np.dot(b[w], b[w]),
                         power_dB_rfpy_minus_ours=10 * np.log10(pa / pb)))
        A.append(a); B.append(b)
    D = pd.DataFrame(rows)
    A, B = np.array(A), np.array(B)
    w = (trf >= 0) & (trf <= 8)
    D.attrs['stack_cc'] = float(np.corrcoef(A.mean(0)[w], B.mean(0)[w])[0, 1])
    D.to_csv(os.path.join(OUT, f'rfpy_check_{code}.csv'), index=False)
    print(D.describe().round(3).to_string()); print('stack cc 0-8 s', round(D.attrs['stack_cc'], 3))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(trf, B.mean(0), 'k', label='this study (modified Wiener)')
    ax.plot(trf, A.mean(0), 'C3--', label='RfPy 0.1.2 (Wiener, pre-event noise)')
    ax.set_xlim(-2, 10); ax.set_xlabel('time after P (s)'); ax.legend()
    ax.set_title(f'{code}: mean radial RF of {len(D)} events, both low-passed 4-6 Hz; stack correlation 0-8 s '
                 f'{D.attrs["stack_cc"]:.3f}, median per-event {D.cc_0_8s.median():.2f}', fontsize=9)
    fig.savefig(os.path.join(FIG, f'sci_rf_rfpy_check_{code.split(".")[1]}.png'), dpi=100)


if __name__ == '__main__':
    (dump if sys.argv[1] == '--dump' else check)(sys.argv[2])
