#!/usr/bin/env python
"""sci_ducellier_tremor.py - main-band ETS episodes 2009-2026 from the public PNSN tremor catalog,
and their relation to our Shelly MTJ LFE families (extends the 2008-2011 catalog comparison of
sci_ducellier_compare.py to 2026 without waveforms).

Regions: MAIN = main-band box lat 39.8-41.8, lon -123.45..-122.2 (Ducellier families 40.1-41.7N,
         -123.47..-122.43); NEAR = tremor within 30 km of the Shelly families (40.06N, -123.62).
Episodes: days with >= 20 MAIN tremor windows (2.5 min each = 50 min), merged across gaps <= 3 d,
          duration >= 3 d.
Tests: (1) Shelly units (meancc >= 0.137, first per family burst gap > 1 h) superposed on episode
       starts (pre 30 d / during / after 30 d), placebo = random starts with the same durations,
       excluding windows that touch +/-60 d of the 2024 M7 (and Shelly data after 2026-06-30);
       per-episode lead/lag: Shelly cumulative-rate change point vs episode start; migration speed.
       (2) daily CCF Shelly vs MAIN tremor (log1p, 61-d high-pass), circular-shift null.
       (3) M2/O1 tidal modulation of MAIN tremor (in episodes) and Shelly, same years, astronomical phase.
       (4) quakes: MAIN tremor and NEAR tremor 30 d after/before, time to next episode vs the
           inter-episode distribution, Shelly silence for comparison.
Outputs: data/sci/ducellier/tremor_*.csv, tremor_summary.json; figures/sci_ducellier_tremor_*.png
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
import sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/seismic/LFE/scripts")
from sci_ducellier_compare import load_shelly, decluster, daily, sea, tide, wrap, C1, C2, C3, GREY

ROOT = "/home/seismic/LFE"
OUT = f"{ROOT}/data/sci/ducellier"
FIG = f"{ROOT}/figures"
rng = np.random.default_rng(20261126)
T0, T1 = pd.Timestamp("2009-09-01"), pd.Timestamp("2026-06-30")   # Shelly v2 ends 2026-06-30
M7 = pd.Timestamp("2024-12-05T18:44")
QUAKES = {"M6.5_2010-01-10": pd.Timestamp("2010-01-10T00:27:39"), "M6.8_2014-03-10": pd.Timestamp("2014-03-10T05:18:13"),
          "M6.2_Petrolia_2021-12-20": pd.Timestamp("2021-12-20T20:10"), "M6.4_Ferndale_2022-12-20": pd.Timestamp("2022-12-20T10:34"),
          "M7.0_2024-12-05": M7}
LAT0, LON0 = 40.06, -123.62


def episodes_from_tremor(tm, t0, t1, min_n=20, gap=3, min_dur=3):
    c = daily(tm.time, t0, t1)
    act = (c >= min_n).to_numpy()
    idx = c.index
    eps, i = [], 0
    while i < len(act):
        if act[i]:
            j = i
            while True:
                nxt = np.where(act[j + 1:j + 2 + gap])[0]
                if len(nxt) == 0:
                    break
                j = j + 1 + nxt[0]
            eps.append((idx[i], idx[j] + pd.Timedelta(days=1)))
            i = j + 1
        else:
            i += 1
    rows = []
    for a, b in eps:
        if (b - a).days < min_dur:
            continue
        w = tm[(tm.time >= a) & (tm.time < b)]
        tt = (w.time - a).dt.total_seconds() / 86400
        sl = np.polyfit(tt, w.lat * 111.2, 1)[0] if len(w) > 20 else np.nan
        rows.append(dict(start=a, end=b, dur_d=(b - a).days, n_win=len(w), lat_min=w.lat.quantile(0.02),
                         lat_max=w.lat.quantile(0.98), migr_kmd=sl,
                         t_south=w[w.lat < 40.4].time.min() if (w.lat < 40.4).any() else pd.NaT))
    return pd.DataFrame(rows), c


def main():
    tr = pd.read_csv(f"{OUT}/pnsn_tremor_south.csv", parse_dates=["time"])
    main_ = tr[(tr.lat >= 39.8) & (tr.lat <= 41.8) & (tr.lon >= -123.45) & (tr.lon <= -122.2)].copy()
    dx = (tr.lon - LON0) * 111.2 * np.cos(np.deg2rad(LAT0)); dy = (tr.lat - LAT0) * 111.2
    near = tr[np.hypot(dx, dy) <= 30].copy()
    eps, cm = episodes_from_tremor(main_, T0 - pd.Timedelta(days=30), pd.Timestamp("2026-11-24"))
    eps.to_csv(f"{OUT}/tremor_episodes.csv", index=False)
    S = {"n_main_windows": int(len(main_)), "n_near_windows": int(len(near)), "n_episodes": int(len(eps))}
    ii = np.diff(eps.start.values).astype("timedelta64[D]").astype(int)
    S["inter_episode_days_median"] = float(np.median(ii)); S["inter_episode_days_iqr"] = [float(np.percentile(ii, 25)), float(np.percentile(ii, 75))]
    big = eps[eps.n_win >= eps.n_win.quantile(0.5)]
    iib = np.diff(big.start.values).astype("timedelta64[D]").astype(int)
    S["big_episodes"] = int(len(big)); S["big_inter_episode_days_median"] = float(np.median(iib))

    sh = load_shelly(); shu = decluster(sh, "fam")
    shd = daily(shu.t, T0, T1).astype(float)
    S["shelly_units_in_period"] = int(shd.sum())

    # ---------- (1) SEA
    def ok(a, b):
        return (a - pd.Timedelta(days=30) >= T0) and (b + pd.Timedelta(days=30) <= T1) and not (
            a - pd.Timedelta(days=30) < M7 + pd.Timedelta(days=60) and b + pd.Timedelta(days=30) > M7 - pd.Timedelta(days=60))
    res = {}
    for lab, E in [("all", eps), ("big", big)]:
        E = E[[ok(a, b) for a, b in zip(E.start, E.end)]]
        r = sea(shd, E.start, E.dur_d)
        od = np.log(r[:, 1].sum() / r[:, 0].sum()); oa = np.log(r[:, 2].sum() / r[:, 0].sum())
        # per-episode log ratios, t-like stat
        per_d = np.log((r[:, 1] + 0.1) / (r[:, 0] + 0.1))
        # placebo
        durs = E.dur_d.to_numpy()
        pdl, pal, pmed = [], [], []
        for _ in range(3000):
            st = []
            for d in durs:
                while True:
                    s = T0 + pd.Timedelta(days=int(rng.integers(30, (T1 - T0).days - int(d) - 30)))
                    if ok(s, s + pd.Timedelta(days=int(d))):
                        break
                st.append(s)
            rr = sea(shd, st, durs)
            pdl.append(np.log(rr[:, 1].sum() / rr[:, 0].sum())); pal.append(np.log(rr[:, 2].sum() / rr[:, 0].sum()))
            pmed.append(np.median(np.log((rr[:, 1] + 0.1) / (rr[:, 0] + 0.1))))
        pdl, pal, pmed = map(np.array, (pdl, pal, pmed))
        res[lab] = dict(n=int(len(r)), during_over_pre=float(np.exp(od)), after_over_pre=float(np.exp(oa)),
                        placebo_during_95=[float(np.exp(np.percentile(pdl, 2.5))), float(np.exp(np.percentile(pdl, 97.5)))],
                        placebo_after_95=[float(np.exp(np.percentile(pal, 2.5))), float(np.exp(np.percentile(pal, 97.5)))],
                        p_during=float(2 * min((pdl >= od).mean(), (pdl <= od).mean())),
                        p_after=float(2 * min((pal >= oa).mean(), (pal <= oa).mean())),
                        median_episode_during_ratio=float(np.exp(np.median(per_d))),
                        p_median=float(2 * min((pmed >= np.median(per_d)).mean(), (pmed <= np.median(per_d)).mean())))
        if lab == "all":
            Eall = E; r_all = r
    S["sea"] = res
    # lag curve: stacked Shelly rate around starts, normalized per episode by its pre-30-d mean
    lags = np.arange(-60, 61)
    curves = []
    for s in Eall.start:
        seg = shd.reindex(s.floor("D") + pd.to_timedelta(lags, unit="D")).to_numpy(float)
        curves.append(seg / np.nanmean(seg[30:60]))
    curves = np.array(curves)
    stack = np.nanmean(curves, 0)
    pd.DataFrame({"lag_d": lags, "stack": stack}).to_csv(f"{OUT}/tremor_sea_stack.csv", index=False)
    # placebo envelope for the stack (5-d smoothed)
    env = []
    for _ in range(500):
        cs = []
        for d in Eall.dur_d:
            while True:
                s = T0 + pd.Timedelta(days=int(rng.integers(60, (T1 - T0).days - 60)))
                if ok(s, s + pd.Timedelta(days=int(d))):
                    break
            seg = shd.reindex(s + pd.to_timedelta(lags, unit="D")).to_numpy(float)
            cs.append(seg / np.nanmean(seg[30:60]))
        env.append(pd.Series(np.nanmean(cs, 0)).rolling(5, center=True).mean().to_numpy())
    env = np.array(env)
    # episode-start relative to tremor reaching the south end (closest to our families)
    eps["south_lag_d"] = (eps.t_south - eps.start).dt.total_seconds() / 86400

    # ---------- (2) CCF daily, Shelly vs MAIN tremor
    tmd = daily(main_.time, T0, T1).astype(float)
    nrd = daily(near.time, T0, T1).astype(float)
    def hp(v, w=61):
        return v - np.convolve(v, np.ones(w) / w, mode="same")
    def ccf(x, y, L):
        x = (x - x.mean()) / x.std(); y = (y - y.mean()) / y.std()
        return np.array([np.mean(x[max(0, -l):len(x) - max(0, l)] * y[max(0, l):len(y) - max(0, -l)]) for l in L])
    msk = ~((shd.index > M7 - pd.Timedelta(days=60)) & (shd.index < M7 + pd.Timedelta(days=120)))
    x = hp(np.log1p(tmd.to_numpy()))[msk]; y = hp(np.log1p(shd.to_numpy()))[msk]; z = hp(np.log1p(nrd.to_numpy()))[msk]
    L = np.arange(-30, 31)
    c_main = ccf(x, y, L); c_near = ccf(z, y, L); c_mn = ccf(x, z, L)
    null = np.array([ccf(x, np.roll(y, int(rng.integers(90, len(y) - 90))), [0])[0] for _ in range(2000)])
    nulln = np.array([ccf(z, np.roll(y, int(rng.integers(90, len(y) - 90))), [0])[0] for _ in range(2000)])
    pd.DataFrame({"lag_d": L, "ccf_main_tremor_vs_shelly": c_main, "ccf_near_tremor_vs_shelly": c_near,
                  "ccf_main_vs_near_tremor": c_mn}).to_csv(f"{OUT}/tremor_ccf_daily.csv", index=False)
    S["ccf"] = dict(main_lag0=float(c_main[30]), main_peak_lag=int(L[np.argmax(np.abs(c_main))]), main_peak=float(c_main[np.argmax(np.abs(c_main))]),
                    null95=[float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))],
                    near_lag0=float(c_near[30]), near_peak_lag=int(L[np.argmax(np.abs(c_near))]), near_peak=float(c_near[np.argmax(np.abs(c_near))]),
                    near_null95=[float(np.percentile(nulln, 2.5)), float(np.percentile(nulln, 97.5))],
                    main_vs_near_lag0=float(c_mn[30]))

    # ---------- (3) tides
    trows = []
    ine = np.zeros(len(main_), bool)
    for a, b in zip(eps.start, eps.end):
        ine |= ((main_.time >= a) & (main_.time < b)).to_numpy()
    sets = {"main_tremor_all": main_.time, "main_tremor_in_episodes": main_.time[ine], "main_tremor_outside": main_.time[~ine],
            "near_tremor_all": near.time,
            "shelly_2009_2026": shu.t[(shu.t >= T0) & (shu.t < T1)]}
    insh = np.zeros(len(shu), bool)
    for a, b in zip(eps.start, eps.end):
        insh |= ((shu.t >= a) & (shu.t < b)).to_numpy()
    sets["shelly_during_main_episodes"] = shu.t[insh & (shu.t >= T0).to_numpy()]
    sets["shelly_outside_main_episodes"] = shu.t[~insh & (shu.t >= T0).to_numpy() & (shu.t < T1).to_numpy()]
    for lab, tt in sets.items():
        for c in ["M2", "O1", "N2", "S2", "K1"]:
            r = tide(tt.reset_index(drop=True), c, 1, B=500, M=1000); r["set"] = lab; trows.append(r)
    tdf = pd.DataFrame(trows); tdf.to_csv(f"{OUT}/tremor_tides.csv", index=False)

    # ---------- (4) quakes
    qrows = []
    for q, tq in QUAKES.items():
        row = dict(quake=q)
        for lab, tm in [("main", main_.time), ("near", near.time), ("shelly", shu.t)]:
            t = tm.sort_values().to_numpy()
            b = ((t >= np.datetime64(tq - pd.Timedelta(days=30))) & (t < np.datetime64(tq))).sum()
            a = ((t >= np.datetime64(tq)) & (t < np.datetime64(tq + pd.Timedelta(days=30)))).sum()
            row[f"{lab}_before30"] = int(b); row[f"{lab}_after30"] = int(a)
        nxt = eps[eps.start >= tq]; prv = eps[eps.start < tq]
        row["prev_episode_start"] = prv.start.max(); row["next_episode_start"] = nxt.start.min()
        row["days_since_prev_ep"] = (tq - prv.start.max()).days if len(prv) else np.nan
        row["days_to_next_ep"] = (nxt.start.min() - tq).days if len(nxt) else np.nan
        # in-episode at quake?
        row["in_episode"] = bool(((eps.start <= tq) & (eps.end > tq)).any())
        # big-episode interval containing the quake vs median
        nb = big[big.start >= tq]; pb = big[big.start < tq]
        if len(nb) and len(pb):
            row["big_interval_containing_d"] = (nb.start.min() - pb.start.max()).days
        qrows.append(row)
    qdf = pd.DataFrame(qrows); qdf.to_csv(f"{OUT}/tremor_quakes.csv", index=False)
    S["big_interval_median_d"] = float(np.median(iib)); S["big_interval_p10_p90"] = [float(np.percentile(iib, 10)), float(np.percentile(iib, 90))]
    # main-band tremor 60 d after M7 vs all 60-d windows
    w60 = main_.time
    cnt = daily(w60, T0, pd.Timestamp("2026-11-24")).rolling(60).sum().shift(-59)
    m7c = float(cnt.loc[M7.floor("D")])
    S["M7_main_tremor_60d_after"] = m7c; S["M7_main_tremor_60d_percentile"] = float((cnt.dropna() < m7c).mean())

    with open(f"{OUT}/tremor_summary.json", "w") as f:
        json.dump(S, f, indent=1, default=str)

    # ---------- figures
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ax[0].scatter(main_.time, main_.lat, s=0.3, c="k", alpha=0.2, lw=0)
    ax[0].scatter(near.time, near.lat, s=1, c=C2, lw=0)
    ax[0].set_ylabel("tremor lat"); ax[0].set_title("PNSN tremor: main band (black) and within 30 km of our families (orange); blue = episodes")
    for a, b in zip(eps.start, eps.end):
        for k in range(3):
            ax[k].axvspan(a, b, color=C1, alpha=0.2, lw=0)
    ax[1].plot(cm.index, cm.values, color="k", lw=0.5); ax[1].set_ylabel("main tremor windows/day"); ax[1].set_yscale("symlog")
    shd_all = daily(shu.t, T0, T1)
    ax[2].plot(shd_all.index, shd_all.values, color=GREY, lw=0.3); ax[2].plot(shd_all.index, shd_all.rolling(15, center=True).mean(), color="k")
    ax[2].set_ylabel("Shelly units/day")
    for q, tq in QUAKES.items():
        for k in range(3):
            ax[k].axvline(tq, color="r", ls="--", lw=0.8)
    fig.tight_layout(); fig.savefig(f"{FIG}/sci_ducellier_tremor_timeline.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].fill_between(lags, np.nanpercentile(env, 2.5, 0), np.nanpercentile(env, 97.5, 0), color=GREY, alpha=0.3, label="placebo 95%")
    ax[0].plot(lags, pd.Series(stack).rolling(5, center=True).mean(), color="k", lw=2, label=f"{len(curves)} episodes")
    ax[0].axvline(0, color=C1); ax[0].axhline(1, color="k", lw=0.5); ax[0].legend(fontsize=8)
    ax[0].set_xlabel("days from main-band episode start"); ax[0].set_ylabel("Shelly rate / pre-30-d mean (5-d smooth)")
    ax[1].plot(L, c_main, color=C1, label="main tremor vs Shelly"); ax[1].plot(L, c_near, color=C2, label="near tremor vs Shelly")
    ax[1].plot(L, c_mn, color=C3, label="main vs near tremor")
    ax[1].axhspan(*np.percentile(null, [2.5, 97.5]), color=C1, alpha=0.15)
    ax[1].set_xlabel("lag (d), + = second series later"); ax[1].legend(fontsize=8); ax[1].set_title("daily CCF (log1p, 61-d high-pass)")
    for c, col in [("M2", C1), ("O1", C2)]:
        sub = tdf[tdf.const == c]
        ax[2].errorbar(sub.phase_deg, sub.a, yerr=[sub.a - sub.a_lo, sub.a_hi - sub.a], fmt="o", color=col, label=c)
        for _, r in sub.iterrows():
            ax[2].annotate(r.set.replace("_", " ")[:22], (r.phase_deg, r.a), fontsize=6)
    ax[2].set_xlabel("astronomical phase (deg)"); ax[2].set_ylabel("modulation amplitude"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/sci_ducellier_tremor_sea_ccf_tides.png", dpi=130); plt.close(fig)

    pd.set_option("display.width", 250)
    print(json.dumps(S, indent=1, default=str))
    print(eps.to_string())
    print(tdf[tdf.const.isin(["M2", "O1", "K1", "S2"])][["set", "const", "n", "a", "a_lo", "a_hi", "a_null95", "p", "phase_deg", "phase_sd_deg"]].to_string())
    print(qdf.to_string())


if __name__ == "__main__":
    main()
