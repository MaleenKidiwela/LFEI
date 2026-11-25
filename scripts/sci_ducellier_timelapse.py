#!/usr/bin/env python
"""sci_ducellier_timelapse.py - time-lapse tests at depth with the Ducellier main-band families, from
the matched-filter scan (sci_ducellier_mf.py, tag 'scan'): 20-Hz, 1.5-9 Hz velocity windows
(-10..+70 s around the detection / template-start time) for up to 40 best detections per family-day.

Per family x channel:
  * reference stack = all windows outside the epoch (leave-epoch-out); picks on the all-window stack (linear, each window RMS-normalised over 0..60 s);
  * P / S picks on the reference stack: S = envelope max on the channel inside the template span,
    P = envelope max before S - 2 s on Z (only S-P on Z+horizontal pairs of the same station is used);
  * epoch stacks (per epoch, >= NMIN windows): delays at P (+/-1.5 s window) and S (+/-2.5 s) vs the
    reference by cross-correlation (parabolic sub-sample), -> d(S-P)/(S-P) = d(ts - tp)/(ts - tp)
    ~ -dVs/Vs-ish path average (P and S picks on the same station);
  * coda dv/v by stretching (S+4 s .. S+30 s), +/-1 %, vs reference.
  Network estimate per epoch = weighted median across family-channels (weights: CC^2/(1-CC^2)).
Epochs: (a) per calendar year; (b) ETS phase: 'ETS' (detections inside PNSN-tremor episodes),
  'post' (0-30 d after an episode end), 'inter' (>30 d after); (c) quake steps: 60 d before vs 60 d
  after each quake (2014 M6.8, Petrolia, Ferndale, 2024 M7) - 2005/2010 quakes need the 2004-2011
  validation subset only, which is too sparse.
Nulls: (1) placebo = labels shuffled among detections (same epoch sizes), 300 draws;
  (2) injection = stretch a random half of the windows by eps and re-measure (recovery factor and
  minimum detectable change at 2x the placebo 95% width, since formal errors understate the scatter).
Outputs: data/sci/ducellier/timelapse_*.csv/json, figures/sci_ducellier_timelapse*.png
Status: written 2026-11-25 while the scan runs; run when the scan has finished.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
import glob, json, sys
import numpy as np
import pandas as pd
from scipy.signal import hilbert

DUC = "/data/wsd02/maleen_data/ducellier"
OUT = "/home/seismic/LFE/data/sci/ducellier"
FIG = "/home/seismic/LFE/figures"
TAG = sys.argv[1] if len(sys.argv) > 1 else "scan"
DT, PRE = 0.05, 10.0
NMIN = 30
rng = np.random.default_rng(11)
QUAKES = {"M6.8_2014": "2014-03-10T05:18", "Petrolia_2021": "2021-12-20T20:10",
          "Ferndale_2022": "2022-12-20T10:34", "M7_2024": "2024-12-05T18:44"}


def load_windows(tag):
    idx, arrs = [], {}
    for f in sorted(glob.glob(f"{DUC}/mf/{tag}/winidx_*.csv")):
        day = os.path.basename(f)[7:17]
        w = pd.read_csv(f, dtype={"template": str}, parse_dates=["time"])
        z = np.load(f"{DUC}/mf/{tag}/win_{day}.npz")
        for r in w.itertuples():
            a = z[r.wid]
            for k, ch in enumerate(r.chans.split(",")):
                if np.any(a[k]):
                    arrs.setdefault((r.template, ch), []).append((r.time, r.cc, a[k]))
    return arrs


def norm(x):
    s = np.sqrt(np.mean(x[int(PRE / DT):int((PRE + 60) / DT)] ** 2))
    return x / s if s > 0 else x


def xcorr_delay(a, b, i0, i1, maxlag=10):
    """delay of b relative to a in window [i0,i1) (samples, parabolic), and cc."""
    seg = a[i0:i1]
    best, bl = -2, 0
    ccs = []
    for L in range(-maxlag, maxlag + 1):
        s2 = b[i0 + L:i1 + L]
        c = np.dot(seg, s2) / (np.linalg.norm(seg) * np.linalg.norm(s2) + 1e-20)
        ccs.append(c)
    ccs = np.array(ccs); k = int(np.argmax(ccs))
    if 0 < k < len(ccs) - 1:
        d = 0.5 * (ccs[k - 1] - ccs[k + 1]) / (ccs[k - 1] - 2 * ccs[k] + ccs[k + 1])
    else:
        d = 0.0
    return (k - maxlag + d) * DT, ccs[k]


def stretch_dvv(ref, cur, i0, i1, eps_max=0.01, n=81):
    t = np.arange(len(ref)) * DT
    seg_t = t[i0:i1]
    best = (-2, 0.0)
    for e in np.linspace(-eps_max, eps_max, n):
        # cur(t*(1+e)) compared with ref(t): dv/v = e (positive = faster)
        c = np.interp(seg_t * (1 + e), t, cur)
        cc = np.dot(ref[i0:i1], c) / (np.linalg.norm(ref[i0:i1]) * np.linalg.norm(c) + 1e-20)
        if cc > best[0]:
            best = (cc, e)
    return best[1], best[0]


REPO = f"{DUC}/gh_catalog/catalog-master"
_ST = pd.read_csv(f"{REPO}/data/Ducellier/stations_permanent.txt", sep=r"\s+", header=None,
                  names=["station", "network", "channels", "location", "server", "lat", "lon", "on", "off", "dt"]).drop_duplicates("station").set_index("station")
_FAM = pd.read_csv(f"{OUT}/families_summary.csv", dtype={"template": str}).set_index("template")
VP, VS = 6.5, 3.75


def predicted_sp(tpl, ch):
    sta = ch.split("_")[0]
    f, s = _FAM.loc[tpl], _ST.loc[sta]
    dx = (s.lon - f.lon) * 111.2 * np.cos(np.deg2rad(f.lat)); dy = (s.lat - f.lat) * 111.2
    d = np.sqrt(dx ** 2 + dy ** 2 + f.depth ** 2)
    return d * (1 / VS - 1 / VP)


def picks(ref, ch, tsp):
    """P,S pair on the reference stack with S-P = predicted +/- 1.5 s maximising the summed
    (median-normalised) envelope; P searched from -8 s to +40 s of the window start."""
    env = np.abs(hilbert(ref)); env = env / (np.median(env) + 1e-20)
    k0 = int(PRE / DT)
    best = (-1, None, None)
    nsp = int(round(tsp / DT)); tol = int(1.5 / DT)
    for pidx in range(k0 - int(8 / DT), min(len(ref) - nsp - tol - 1, k0 + int(40 / DT))):
        seg = env[pidx + nsp - tol:pidx + nsp + tol + 1]
        k = int(np.argmax(seg))
        sc = env[pidx] + seg[k]
        if sc > best[0]:
            best = (sc, pidx, pidx + nsp - tol + k)
    return best[1], best[2]


def measure(arrs, labels_fn, min_n=NMIN):
    """labels_fn(time array) -> epoch labels; returns rows per family-channel-epoch."""
    rows = []
    for (tpl, ch), L in arrs.items():
        if len(L) < 2 * min_n:
            continue
        times = np.array([x[0] for x in L], dtype="datetime64[ns]")
        X = np.vstack([norm(x[2].astype(np.float64)) for x in L])
        p, s = picks(X.mean(0), ch, predicted_sp(tpl, ch))
        lab = labels_fn(pd.DatetimeIndex(times))
        for e in pd.unique(lab[pd.notna(lab)]):
            m = lab == e
            if m.sum() < min_n:
                continue
            st = X[m].mean(0)
            if (~m).sum() < min_n:
                continue
            ref = X[~m].mean(0)          # leave-epoch-out reference (avoids self-bias toward zero)
            ds, cs = xcorr_delay(ref, st, s - int(2.5 / DT), s + int(2.5 / DT))
            dv, cv = stretch_dvv(ref, st, s + int(4 / DT), min(len(ref) - 20, s + int(30 / DT)))
            row = dict(template=tpl, ch=ch, epoch=e, n=int(m.sum()), dS=ds, ccS=cs, dvv=dv, ccv=cv, tS=(s * DT - PRE))
            if p is not None and ch.endswith("Z"):
                dp, cp = xcorr_delay(ref, st, p - int(1.5 / DT), p + int(1.5 / DT))
                row.update(dP=dp, ccP=cp, tSP=(s - p) * DT)
            rows.append(row)
    return pd.DataFrame(rows)


def network(df):
    out = []
    for e, g in df.groupby("epoch"):
        w = np.clip(g.ccv, 0, 0.99) ** 2 / (1 - np.clip(g.ccv, 0, 0.99) ** 2)
        o = dict(epoch=e, n_fc=len(g), n_win=int(g.n.sum()), dvv_med=float(np.median(g.dvv)),
                 dvv_wmean=float(np.sum(w * g.dvv) / np.sum(w)))
        if "dP" in g:
            h = g.dropna(subset=["dP"])
            h = h[(h.ccP > 0.5) & (h.ccS > 0.5)]
            if len(h):
                o["dSP_rel_med"] = float(np.median((h.dS - h.dP) / h.tSP))
        out.append(o)
    return pd.DataFrame(out)


def main():
    arrs = load_windows(TAG)
    print("family-channels with windows:", len(arrs), "total windows:", sum(len(v) for v in arrs.values()))
    tr = pd.read_csv(f"{OUT}/tremor_episodes.csv", parse_dates=["start", "end"])
    def ets_lab(t):
        lab = np.array(["inter"] * len(t), dtype=object)
        for a, b in zip(tr.start, tr.end):
            lab[(t >= b) & (t < b + pd.Timedelta(days=30))] = "post"
        for a, b in zip(tr.start, tr.end):
            lab[(t >= a) & (t < b)] = "ETS"
        lab[t < pd.Timestamp("2012-01-01")] = None
        return lab
    def year_lab(t):
        return np.asarray(t.year.astype(str), dtype=object)
    res = {}
    for name, fn in [("year", year_lab), ("ets_phase", ets_lab)] + [
            (q, (lambda T: (lambda t: np.where((t >= T - pd.Timedelta(days=60)) & (t < T), "before",
                                               np.where((t >= T) & (t < T + pd.Timedelta(days=60)), "after", None)).astype(object)))(pd.Timestamp(v)))
            for q, v in QUAKES.items()]:
        df = measure(arrs, fn)
        if df.empty:
            continue
        df.to_csv(f"{OUT}/timelapse_{name}_famchan.csv", index=False)
        net = network(df)
        net.to_csv(f"{OUT}/timelapse_{name}_network.csv", index=False)
        res[name] = net.to_dict("records")
        # placebo: shuffle labels within each family-channel
        if name != "year":
            nulls = []
            for _ in range(100):
                def shuf(t, fn=fn):
                    lab = fn(t)
                    m = pd.notna(lab)
                    lab2 = lab.copy(); lab2[m] = rng.permutation(lab[m])
                    return lab2
                d2 = measure(arrs, shuf)
                if not d2.empty:
                    nulls.append(network(d2).set_index("epoch").dvv_med)
            if nulls:
                N = pd.concat(nulls, axis=1)
                res[name + "_placebo95_halfwidth"] = (N.quantile(0.975, axis=1) - N.quantile(0.025, axis=1)).div(2).to_dict()
        print(name, net.to_string())
    json.dump(res, open(f"{OUT}/timelapse_summary.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
