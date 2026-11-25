#!/usr/bin/env python
"""sci_ducellier_mf.py - network matched filter for the Ducellier & Creager (2022) southern-Cascadia
main-band LFE families, reproducing their detector on permanent NCEDC stations, processing
continuous data on the fly (one day in memory; no raw continuous data stored).

Template source: Ducellier's templates_v1 (20 Hz, response-removed velocity, 1.5-9 Hz, 20-50 s long;
loaded with a restricted unpickler: scripts/sci_ducellier_safe_pickle.py).
Preprocessing (as theirs, src/get_data.py): linear detrend, 5-s Hann taper, remove_response(VEL,
pre_filt 0.2/0.5/10/15, water_level 80), bandpass 1.5-9 Hz zero-phase, Lanczos (a=10) to 20 Hz.
Horizontals rotated into the template's orientation (their rotate_data).
Detection (theirs, src/find_all_LFEs_parallel.py): per hour, mean normalized CC across channels;
threshold 8 * median(|cc - mean(cc)|); peaks grouped within 1 s; keep cc*nchannel >= threshold_perm.

Usage:
  python sci_ducellier_mf.py --days 2008-04-20,2009-10-10 --tag pilot [--stations all|core]
Outputs (per day): /data/wsd02/maleen_data/ducellier/mf/<tag>/det_<day>.csv (all 8xMAD detections,
with cc, nchannel, per-channel cc kept for the strongest), timing log.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys, io, time, glob, json, argparse
import numpy as np
import pandas as pd
import requests
from math import cos, pi
from scipy.signal import fftconvolve, oaconvolve
import obspy
from obspy import read, read_inventory, UTCDateTime, Stream
sys.path.insert(0, "/home/seismic/LFE/scripts")
from sci_ducellier_safe_pickle import load as safe_load

DUC = "/data/wsd02/maleen_data/ducellier"
REPO = f"{DUC}/gh_catalog/catalog-master"
TDIR = f"{REPO}/src/templates_v1"
RESP = f"{DUC}/resp"
DS_URL = "https://service.ncedc.org/fdsnws/dataselect/1/query"
ST_URL = "https://service.ncedc.org/fdsnws/station/1/query"
os.makedirs(RESP, exist_ok=True)
DT = 0.05
FILT = (1.5, 9.0)
PAD = 120.0        # s of data before/after the day (template length + taper)
NET = {}           # station -> network


def station_table():
    s = pd.read_csv(f"{REPO}/data/Ducellier/stations_permanent.txt", sep=r"\s+", header=None,
                    names=["station", "network", "channels", "location", "server", "lat", "lon", "on", "off", "dt"])
    return s.drop_duplicates("station", keep="last").set_index("station")


def families():
    f = pd.read_csv(f"{REPO}/data/Ducellier/families_permanent.txt", sep=r"\s+", header=None,
                    names=["template", "stations", "duration"], dtype={"template": str})
    th = pd.read_csv(f"{REPO}/data/Ducellier/threshold_cc.txt", sep=r"\s+", header=None,
                     names=["template", "thr_fame", "thr_perm"], dtype={"template": str})
    return f.merge(th, on="template")


def load_templates(template, stations):
    out = {}
    for sta in stations:
        for p in sorted(glob.glob(f"{TDIR}/{template}/{sta}_*.pkl")):
            tr, ang = safe_load(p)
            out[f"{sta}_{tr.stats.channel}"] = (tr, ang)
    return out


def get_inventory(net, sta):
    f = f"{RESP}/{net}_{sta}.xml"
    if not os.path.exists(f):
        r = requests.get(ST_URL, params=dict(net=net, sta=sta, level="response", format="xml"), timeout=300)
        r.raise_for_status()
        open(f, "wb").write(r.content)
    return read_inventory(f)


_RCACHE = {}


def fast_remove_response(tr, inv, pre_filt=(0.2, 0.5, 10.0, 15.0), water_level=80.0):
    """obspy-equivalent remove_response(output=VEL) with the response evaluated on a coarse grid
    (cached per channel epoch) and interpolated (amplitude in log, phase unwrapped) to FFT freqs."""
    from scipy.fft import rfft, irfft, next_fast_len
    from obspy.signal.invsim import cosine_sac_taper
    resp = inv.get_response(tr.id, tr.stats.starttime + 1)
    key = (tr.id, id(resp), tr.stats.sampling_rate)
    if key not in _RCACHE:
        fg = np.concatenate([[0.0], np.logspace(-3, np.log10(tr.stats.sampling_rate / 2), 4000)])
        r = resp.get_evalresp_response_for_frequencies(fg[1:], output="VEL")
        r = np.concatenate([[r[0]], r])
        _RCACHE[key] = (fg, np.log(np.abs(r) + 1e-300), np.unwrap(np.angle(r)))
    fg, la, ph = _RCACHE[key]
    n = tr.stats.npts
    nfft = next_fast_len(n)
    f = np.fft.rfftfreq(nfft, tr.stats.delta)
    R = np.exp(np.interp(f, fg, la)) * np.exp(1j * np.interp(f, fg, ph))
    # water level (obspy invert_spectrum semantics)
    amax = np.abs(R).max()
    wl = amax * 10 ** (-water_level / 20.0)
    a = np.abs(R)
    R = np.where(a < wl, R * wl / np.maximum(a, 1e-300), R)
    X = rfft(tr.data.astype(np.float64), nfft)
    X *= cosine_sac_taper(f, flimit=pre_filt)
    X = np.where(np.abs(R) > 0, X / R, 0)
    tr.data = irfft(X, nfft)[:n]
    return tr


def fetch_station_day(net, sta, chans, day, tries=3):
    """download one (padded) day; return preprocessed 20-Hz Stream (in memory) and seconds spent."""
    t0 = UTCDateTime(day) - PAD
    t1 = UTCDateTime(day) + 86400 + PAD
    lines = "\n".join(f"{net} {sta} * {c} {t0.strftime('%Y-%m-%dT%H:%M:%S')} {t1.strftime('%Y-%m-%dT%H:%M:%S')}" for c in chans)
    tic = time.time()
    st = None
    for a in range(tries):
        try:
            r = requests.post(DS_URL, data=lines, timeout=900)
            if r.status_code == 204:
                return Stream(), time.time() - tic
            r.raise_for_status()
            st = read(io.BytesIO(r.content))
            break
        except Exception:
            time.sleep(20)
    if st is None:
        return None, time.time() - tic
    tdl = time.time() - tic
    inv = get_inventory(net, sta)
    # keep one location code per channel (most samples)
    keep = Stream()
    for c in set(tr.stats.channel for tr in st):
        sub = st.select(channel=c)
        locs = {}
        for tr in sub:
            locs[tr.stats.location] = locs.get(tr.stats.location, 0) + tr.stats.npts
        loc = max(locs, key=locs.get)
        keep += sub.select(location=loc)
    st = keep
    st.merge(method=1, fill_value=None)
    out = Stream()
    for tr in st:
        try:
            if isinstance(tr.data, np.ma.MaskedArray):
                # gaps: split, process pieces, merge back with zeros (zeros give cc=0 -> excluded by npts check per hour)
                pieces = tr.split()
            else:
                pieces = Stream([tr])
            pp = Stream()
            for p in pieces:
                if p.stats.npts * p.stats.delta < 600:
                    continue
                p = p.copy()
                p.detrend("linear")
                p.taper(type="hann", max_percentage=None, max_length=5.0)
                fast_remove_response(p, inv)
                p.filter("bandpass", freqmin=FILT[0], freqmax=FILT[1], zerophase=True)
                # data already band-limited to 9 Hz: shift to the 20-Hz grid (sub-sample, Fourier) and decimate
                fs = p.stats.sampling_rate
                fac = int(round(fs * DT))
                tgrid = UTCDateTime(np.ceil(p.stats.starttime.timestamp / DT) * DT)
                shift = tgrid - p.stats.starttime            # >= 0, < DT
                k0 = int(np.floor(shift * fs + 1e-9)); frac = shift * fs - k0
                x = p.data[k0:]
                if frac > 1e-6:
                    from scipy.fft import rfft, irfft, next_fast_len
                    nf = next_fast_len(len(x)); f = np.fft.rfftfreq(nf)
                    x = irfft(rfft(x, nf) * np.exp(2j * np.pi * f * frac), nf)[:len(x)]
                from fractions import Fraction
                fr_ = Fraction(int(round(1.0 / DT)), int(round(fs)))
                if fr_.numerator == 1:
                    y = x[::fr_.denominator]
                else:
                    from scipy.signal import resample_poly
                    y = resample_poly(x, fr_.numerator, fr_.denominator)
                p.data = np.ascontiguousarray(y)
                p.stats.sampling_rate = 1.0 / DT
                p.stats.starttime = tgrid
                pp += p
            if len(pp):
                pp.merge(method=1, fill_value=0)
                for p in pp:
                    p.stats.response = None
                    p.stats["az"] = inv.get_orientation(p.id, p.stats.starttime + 3600)["azimuth"]
                out += pp
        except Exception as e:
            print("proc fail", tr.id, e, flush=True)
    return out, tdl


def rotate_to_template(st, sta, tmpl):
    """rotate horizontals into each template channel's azimuth (Ducellier rotate_data)."""
    res = {}
    hz = [tr for tr in st if tr.stats.channel[-1] in "ENZ12"]
    comps = {tr.stats.channel: tr for tr in hz}
    for key, (ttr, ang) in tmpl.items():
        s, ch = key.split("_")
        if s != sta:
            continue
        # match by component letter on any band code present (SHZ template <- EHZ data after 2011, HH<->BH)
        cand = [c for c in comps if c[-1] == ch[-1]]
        if ch[-1] in "Z":
            if cand:
                res[key] = comps[cand[0]].data.astype(np.float64), comps[cand[0]].stats.starttime
            continue
        pair = {"E": "N", "N": "E", "1": "2", "2": "1"}
        c1 = cand[0] if cand else None
        if c1 is None:
            continue
        c2s = [c for c in comps if c[-1] == pair[c1[-1]] and c[:2] == c1[:2]]
        if not c2s:
            continue
        a, b = comps[c1], comps[c2s[0]]
        t0 = max(a.stats.starttime, b.stats.starttime)
        n = min(int((a.stats.endtime - t0) / DT), int((b.stats.endtime - t0) / DT))
        ia, ib = int(round((t0 - a.stats.starttime) / DT)), int(round((t0 - b.stats.starttime) / DT))
        da, db = a.data[ia:ia + n].astype(np.float64), b.data[ib:ib + n].astype(np.float64)
        # da is the component with the same letter as the template channel
        if c1[-1] in "E1":
            dE, dN, aE, aN = da, db, a.stats.az, b.stats.az
        else:
            dE, dN, aE, aN = db, da, b.stats.az, a.stats.az
        tE = np.deg2rad(ang["azimuth"])
        out = cos(np.deg2rad(aE) - tE) * dE + cos(np.deg2rad(aN) - tE) * dN
        res[key] = out, t0
    return res


_SDC = {}


def ncc(tpl, data, ckey=None):
    """Ducellier correlate.optimized via overlap-add FFT; sliding sums cached per (channel, M)."""
    M = len(tpl)
    tpl = tpl - tpl.mean()
    st_ = np.sqrt(M * np.sum(tpl ** 2))
    if ckey is not None and (ckey, M) in _SDC:
        sd = _SDC[(ckey, M)]
    else:
        xs = np.concatenate([[0.0], np.cumsum(data)])
        x2 = np.concatenate([[0.0], np.cumsum(data ** 2)])
        xsum = xs[M:] - xs[:-M]
        x2sum = x2[M:] - x2[:-M]
        sd = np.sqrt(np.maximum(M * x2sum - xsum ** 2, 0))
        if ckey is not None:
            _SDC[(ckey, M)] = sd
    num = M * oaconvolve(data, tpl[::-1], mode="valid")
    with np.errstate(invalid="ignore", divide="ignore"):
        cc = np.where(sd > 1e-6 * np.median(sd[sd > 0]) if np.any(sd > 0) else False, num / (st_ * sd), 0.0)
    return cc


def detect_family(fam_row, tmpl, chdata, day, mad_mult=8.0, freq0=1.0):
    """hourly Ducellier detection; chdata: key -> (array, starttime). returns DataFrame"""
    T0 = UTCDateTime(day)
    dur = fam_row.duration
    rows = []
    # full-day cc per channel on a common grid starting at T0
    ccs = {}
    for key, (ttr, ang) in tmpl.items():
        if key not in chdata:
            continue
        d, ts = chdata[key]
        i0 = int(round((T0 - ts) / DT))
        n = int(round((86400 + dur) / DT)) + 1
        if i0 < 0 or i0 + n > len(d):
            seg = np.zeros(n)
            a0, b0 = max(0, i0), min(len(d), i0 + n)
            if b0 > a0:
                seg[a0 - i0:b0 - i0] = d[a0:b0]
        else:
            seg = d[i0:i0 + n]
        tpl = ttr.data.astype(np.float64)
        cc = ncc(tpl, seg, ckey=(key, round(float(ang["azimuth"]), 2), len(seg)))
        ccs[key] = (cc, seg)
    if not ccs:
        return pd.DataFrame()
    keys = list(ccs)
    nh = int(dur / DT)
    for h in range(24):
        a = int(h * 3600 / DT)
        b = a + int(3600 / DT) + 1
        good = []
        for k in keys:
            seg = ccs[k][1][a:a + int((3600 + dur) / DT) + 1]
            if np.count_nonzero(seg) > 0.999 * len(seg):   # complete hour (their npts check)
                good.append(k)
        if not good:
            continue
        M = np.vstack([ccs[k][0][a:b] for k in good])
        mcc = M.mean(0)
        mad = np.median(np.abs(mcc - np.mean(mcc)))
        idx = np.where(mcc >= mad_mult * mad)[0]
        if len(idx) == 0:
            continue
        # group within 1 s, take max
        groups = np.split(idx, np.where(np.diff(idx) > int(1.0 / (DT * freq0)))[0] + 1)
        for g in groups:
            i = g[np.argmax(mcc[g])]
            rows.append(dict(template=fam_row.template, time=(T0 + (a + i) * DT).datetime, cc=float(mcc[i]),
                             nchannel=len(good), mad=float(mad), thr_perm=float(fam_row.thr_perm),
                             chans=",".join(good)))
    return pd.DataFrame(rows)


def build_requests(fams, stab, station_subset=None):
    need = set()
    for x in fams.stations:
        need |= set(x.split(","))
    if station_subset is not None:
        need &= set(station_subset)
    req = {}
    for sta in sorted(need):
        if sta not in stab.index:
            continue
        net = stab.loc[sta, "network"]
        ch = stab.loc[sta, "channels"].split(",")
        if all(c[:2] == "SH" for c in ch):
            ch = ["SHZ", "EHZ"]
        elif ch[0][:2] == "EH" and net == "PB":
            ch = ["EH1", "EH2", "EHZ"]
        elif ch[0][:2] == "BH":
            ch = ["BHE", "BHN", "BHZ", "BH1", "BH2"]
        else:
            ch = ["HHE", "HHN", "HHZ", "HH1", "HH2"]
        req[sta] = (net, ch)
    return req


WIN_PRE, WIN_POST, MAX_WIN = 10.0, 70.0, 40


def run_day(day, fams, stab, outdir, station_subset=None, save_windows=True):
    """sequential downloads (one request at a time per worker), detection, short event windows kept."""
    tic = time.time()
    os.makedirs(outdir, exist_ok=True)
    req = build_requests(fams, stab, station_subset)
    data, tdl = {}, {}
    for sta in req:
        net, ch = req[sta]
        st, t = fetch_station_day(net, sta, ch, day)
        tdl[sta] = t
        if st is not None and len(st):
            data[sta] = st
    t_dl = time.time() - tic
    _SDC.clear()
    t_proc = time.time()
    dets, wins, widx = [], {}, []
    for fr in fams.itertuples():
        stas = [x for x in fr.stations.split(",") if x in data]
        if not stas:
            continue
        tmpl = load_templates(fr.template, stas)
        chdata = {}
        for x in stas:
            chdata.update(rotate_to_template(data[x], x, tmpl))
        d = detect_family(fr, tmpl, chdata, day)
        if len(d):
            dets.append(d)
            if save_windows:
                g = d[d.cc * d.nchannel >= d.thr_perm].sort_values("cc", ascending=False).head(MAX_WIN)
                keys = sorted(chdata)
                for r in g.itertuples():
                    t0 = UTCDateTime(r.time) - WIN_PRE
                    n = int((WIN_PRE + WIN_POST) / DT) + 1
                    arr = np.zeros((len(keys), n), np.float32)
                    for k, key in enumerate(keys):
                        dd, ts = chdata[key]
                        i0 = int(round((t0 - ts) / DT))
                        if 0 <= i0 and i0 + n <= len(dd):
                            arr[k] = dd[i0:i0 + n]
                    wid = f"w{len(widx):05d}"
                    wins[wid] = arr
                    widx.append(dict(wid=wid, template=fr.template, time=r.time, cc=r.cc, nchannel=r.nchannel,
                                     chans=",".join(keys)))
    det = pd.concat(dets, ignore_index=True) if dets else pd.DataFrame(columns=["template", "time", "cc", "nchannel", "mad", "thr_perm", "chans"])
    if wins:
        np.savez_compressed(f"{outdir}/win_{day}.npz", **wins)
        pd.DataFrame(widx).to_csv(f"{outdir}/winidx_{day}.csv", index=False)
    log = dict(day=day, n_sta=len(data), stations=sorted(data), t_download_s=t_dl, t_detect_s=time.time() - t_proc, t_total_s=time.time() - tic,
               per_station_s=tdl, n_det=int(len(det)), n_win=len(widx))
    json.dump(log, open(f"{outdir}/log_{day}.json", "w"), indent=1, default=str)
    det.to_csv(f"{outdir}/det_{day}.csv", index=False)   # written last = day complete
    return det, log


def _worker(args):
    day, tag, sub = args
    import warnings
    warnings.filterwarnings("ignore")
    outdir = f"{DUC}/mf/{tag}"
    if os.path.exists(f"{outdir}/det_{day}.csv"):
        return day, None
    fams = families()
    fams = fams[(fams.thr_perm > 0) & ~fams.template.str.startswith(("080326", "080328"))]   # subduction families only
    try:
        det, log = run_day(day, fams, station_table(), outdir, sub)
        return day, dict(n_sta=log["n_sta"], n_det=log["n_det"], n_win=log["n_win"], t=round(log["t_total_s"]))
    except Exception as e:
        return day, f"ERROR {e!r}"


if __name__ == "__main__":
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", default=None, help="comma list")
    ap.add_argument("--daylist", default=None, help="file with one YYYY-MM-DD per line")
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--stations", default="all")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    days = a.days.split(",") if a.days else [l.strip() for l in open(a.daylist) if l.strip()]
    sub = None if a.stations == "all" else a.stations.split(",")
    with Pool(a.workers) as pool:
        for day, res in pool.imap_unordered(_worker, [(d, a.tag, sub) for d in days]):
            print(time.strftime("%H:%M:%S"), day, res, flush=True)
