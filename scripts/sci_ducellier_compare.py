#!/usr/bin/env python
"""sci_ducellier_compare.py - Ducellier & Creager (2022) main-band (ETS-driven) LFE families vs
our Shelly MTJ families (near-continuous), over the 2008-2011 overlap (catalog-level; no waveforms).

Inputs : /data/wsd02/maleen_data/ducellier/catalog_csv/ducellier_2004_2011_all_raw.parquet
         (from sci_ducellier_catalog.py; pickles read with a restricted unpickler)
         data/shelly2026_MTJ_v2/LFEcatV2_MTJ_2008-2026.csv
         Boyarko et al. 2015 tremor (from the Ducellier repo; 2005.67-2010.94)
Filter : Ducellier's own (cc * nchannel >= threshold_perm; families with threshold_perm = 0 dropped,
         as in their plot_tremor_nbLFEs.py). Subduction families = Plourde numeric families 001-034.
Units  : first detection of each family burst (gap > 1 h), for both catalogs (as in sci_tides.py;
         Shelly units meancc >= 0.137).
Tests  : (1) ETS episodes from the Ducellier detections; superposed-epoch Shelly rate (pre/during/after)
             with a random-start placebo (same durations), plus per-episode lead/lag;
         (2) daily and hourly rate cross-correlation over the overlap, null = random circular shifts
             by whole days (daily) / whole days (hourly; keeps time-of-day pattern);
         (3) M2/O1/N2 tidal amplitude + astronomical phase, both populations, same period, ETS vs
             inter-ETS for the main band;
         (4) quakes inside the overlap/2004-2011: rate ratio 30 d after / 30 d before, silence length,
             vs random-time placebo.
Outputs: data/sci/ducellier/compare_*.csv, compare_summary.json; figures/sci_ducellier_compare_*.png
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/seismic/LFE"
OUT = f"{ROOT}/data/sci/ducellier"
FIG = f"{ROOT}/figures"
DUC = "/data/wsd02/maleen_data/ducellier"
REPO = f"{DUC}/gh_catalog/catalog-master"
rng = np.random.default_rng(20261125)
TREF = pd.Timestamp("2000-01-01T12:00:00")
PER_H = {"M2": 12.4206012, "N2": 12.65834751, "O1": 25.81933871, "S2": 12.0, "K1": 23.93447213}
OV0, OV1 = pd.Timestamp("2008-06-01"), pd.Timestamp("2012-01-01")   # overlap (Duc ends 2011-12-31);
# Shelly NSta 13 -> 16 on 2008-04-30 and sparse before mid-2008: start the overlap 2008-06-01
QUAKES = {"Gorda_M7.2_2005": pd.Timestamp("2005-06-15T02:50:54"),
          "M6.5_2010": pd.Timestamp("2010-01-10T00:27:39"),
          "M5.9_2010-02": pd.Timestamp("2010-02-04T20:20:21"),
          "M5.4_2008-04-30": pd.Timestamp("2008-04-30T03:03:06")}
C1, C2, C3, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def decluster(df, key="fam", gap=3600.0):
    df = df.sort_values([key, "t"])
    g = df.groupby(key).tsec.diff()
    return df[(g.isna() | (g > gap)).to_numpy()].sort_values("t").reset_index(drop=True)


# ------------------------------------------------------------------ load
def load_duc():
    raw = pd.read_parquet(f"{DUC}/catalog_csv/ducellier_2004_2011_all_raw.parquet")
    fs = pd.read_csv(f"{OUT}/families_summary.csv", dtype={"template": str, "fam": str})
    raw = raw.merge(fs[["template", "fam", "lat", "lon", "depth", "thr_b"]], on="template")
    raw = raw[raw.thr_b > 0]
    keep = raw.cc * raw.nchannel >= raw.thr_b
    d = raw[keep].copy()
    d["sub"] = d.fam.str.isnumeric()
    d["t"] = pd.to_datetime(d.time)
    d["tsec"] = (d.t - TREF).dt.total_seconds()
    return raw, d


def load_shelly():
    d = pd.read_csv(f"{ROOT}/data/shelly2026_MTJ_v2/LFEcatV2_MTJ_2008-2026.csv")
    d.columns = [c.lstrip("%") for c in d.columns]
    d["t"] = pd.to_datetime(dict(year=d.year, month=d.month, day=d.day)) + pd.to_timedelta(d.s_of_day, unit="s")
    d["fam"] = d.idnum.astype(int)
    d = d[d.meancc >= 0.137].copy()
    d["tsec"] = (d.t - TREF).dt.total_seconds()
    return d


def daily(t, t0, t1):
    idx = pd.date_range(t0, t1 - pd.Timedelta(days=1), freq="D")
    c = pd.Series(1, index=t.dt.floor("D")).groupby(level=0).sum().reindex(idx, fill_value=0)
    return c


def hourly(t, t0, t1):
    idx = pd.date_range(t0, t1 - pd.Timedelta(hours=1), freq="h")
    return pd.Series(1, index=t.dt.floor("h")).groupby(level=0).sum().reindex(idx, fill_value=0)


# ------------------------------------------------------------------ ETS episodes
def episodes(dsub, t0, t1, min_fams=4, min_ev=3, merge_gap=5, min_dur=3):
    """day is 'active' if >= min_fams subduction families each have >= min_ev units that day."""
    x = dsub.assign(day=dsub.t.dt.floor("D")).groupby(["day", "fam"]).size()
    nf = (x >= min_ev).groupby(level=0).sum()
    idx = pd.date_range(t0, t1 - pd.Timedelta(days=1), freq="D")
    nf = nf.reindex(idx, fill_value=0)
    act = (nf >= min_fams).to_numpy()
    eps = []
    i = 0
    while i < len(act):
        if act[i]:
            j = i
            while j + 1 < len(act) and (act[j + 1] or (j + 1 + merge_gap < len(act) and act[j + 1:j + 2 + merge_gap].any())):
                j += 1
            while not act[j]:
                j -= 1
            eps.append((idx[i], idx[j] + pd.Timedelta(days=1)))
            i = j + 1
        else:
            i += 1
    eps = [(a, b) for a, b in eps if (b - a).days >= min_dur]
    rows = []
    for a, b in eps:
        w = dsub[(dsub.t >= a) & (dsub.t < b)]
        # along-strike migration: latitude vs time slope (km/day) of family-day centroids
        fd = w.assign(day=(w.t - a).dt.total_seconds() / 86400).groupby("fam").agg(lat=("lat", "first"), tmed=("day", "median"), n=("day", "size"))
        slope = np.polyfit(fd.tmed, fd.lat * 111.2, 1)[0] if len(fd) >= 4 else np.nan
        south = w[w.lat <= 40.2]
        rows.append(dict(start=a, end=b, dur_d=(b - a).days, n_units=len(w), n_fams=w.fam.nunique(),
                         lat_min=w.lat.min(), lat_max=w.lat.max(), migr_kmd=slope,
                         south_first=south.t.min() if len(south) else pd.NaT))
    return pd.DataFrame(rows), nf


# ------------------------------------------------------------------ superposed epoch
def sea(rate, starts, durs, pre=30, post=30, nboot=0):
    """ratio during/pre and after/pre of mean daily rate, per episode; rate is a daily Series."""
    out = []
    for s, d in zip(starts, durs):
        s = pd.Timestamp(s).floor("D")
        pr = rate[s - pd.Timedelta(days=pre): s - pd.Timedelta(days=1)]
        du = rate[s: s + pd.Timedelta(days=int(d) - 1)]
        af = rate[s + pd.Timedelta(days=int(d)): s + pd.Timedelta(days=int(d) + post - 1)]
        if len(pr) < pre * 0.8 or len(du) < d * 0.8 or len(af) < post * 0.8:
            continue
        out.append((pr.mean(), du.mean(), af.mean()))
    return np.array(out)


def placebo_sea(rate, durs, t0, t1, avoid, M=4000, pre=30, post=30):
    """random start days (uniform in [t0+pre, t1-dur-post]); same durations; returns stats arrays."""
    days = rate.index
    res_d, res_a = [], []
    for _ in range(M):
        st = []
        for d in durs:
            for _try in range(50):
                s = t0 + pd.Timedelta(days=int(rng.integers(pre, (t1 - t0).days - int(d) - post)))
                if not any((s < b + pd.Timedelta(days=post)) and (s + pd.Timedelta(days=int(d)) > a - pd.Timedelta(days=pre)) for a, b in avoid):
                    break
            st.append(s)
        r = sea(rate, st, durs, pre, post)
        if len(r) == 0:
            continue
        res_d.append(np.log(r[:, 1].sum() / r[:, 0].sum()))
        res_a.append(np.log(r[:, 2].sum() / r[:, 0].sum()))
    return np.array(res_d), np.array(res_a)


# ------------------------------------------------------------------ tides
def tide(t, const, blockday, B=1000, M=2000):
    """network modulation amplitude (debiased, day-block clustered) and astronomical phase."""
    tsec = (t - TREF).dt.total_seconds().to_numpy()
    phi = 2 * np.pi * np.mod(tsec / 3600.0 / PER_H[const], 1.0)
    day = np.floor(tsec / 86400).astype(int)
    ub, blk = np.unique(day // blockday, return_inverse=True)
    nb = len(ub)
    S = np.bincount(blk, np.cos(phi), nb) + 1j * np.bincount(blk, np.sin(phi), nb)
    n = np.bincount(blk, None, nb).astype(float)
    def amp(Ss, N, Q):
        return 2 * np.sqrt(np.maximum(np.abs(Ss) ** 2 - Q, 0)) / N
    a = amp(S.sum(), n.sum(), (np.abs(S) ** 2).sum())
    W = rng.poisson(1.0, (B, nb)).astype(float)
    ab = amp(W @ S, W @ n, (W ** 2) @ np.abs(S) ** 2)
    phb = np.angle(W @ S)
    k = rng.integers(1, 401, (M, nb)) * rng.choice([-1, 1], (M, nb))
    E = np.exp(1j * 2 * np.pi * np.mod(24.0 * k / PER_H[const], 1.0))
    az = amp(E @ S, n.sum(), (np.abs(S) ** 2).sum())
    ph = np.angle(S.sum())
    return dict(const=const, n=int(n.sum()), a=float(a), a_lo=float(np.percentile(ab, 16)), a_hi=float(np.percentile(ab, 84)),
                a_null95=float(np.percentile(az, 95)), p=float((1 + (az >= a).sum()) / (M + 1)),
                phase_deg=float(np.rad2deg(ph) % 360), phase_sd_deg=float(np.rad2deg(np.std(wrap(phb - ph)))))


# ------------------------------------------------------------------ quake response
def quake_resp(units, tq, t0, t1, win=30, M=4000):
    """log ratio of counts (after win d / before win d) and longest silence (h) starting within 2 d after tq;
    placebo = same statistics at random times."""
    t = units.t.sort_values().to_numpy()
    def stats(tt):
        tt = np.datetime64(tt)
        b = ((t >= tt - np.timedelta64(win, "D")) & (t < tt)).sum()
        a = ((t >= tt) & (t < tt + np.timedelta64(win, "D"))).sum()
        nxt = t[t >= tt]
        sil = (nxt[0] - tt) / np.timedelta64(1, "h") if len(nxt) else np.nan
        return np.log((a + 0.5) / (b + 0.5)), sil, a, b
    lr, sil, a, b = stats(tq)
    lo, hi = t0 + pd.Timedelta(days=win), t1 - pd.Timedelta(days=win)
    rs = [stats(lo + pd.Timedelta(seconds=float(rng.uniform(0, (hi - lo).total_seconds())))) for _ in range(M)]
    rs = np.array([(x[0], x[1]) for x in rs])
    return dict(n_before=int(b), n_after=int(a), logratio=float(lr), p_drop=float((1 + (rs[:, 0] <= lr).sum()) / (M + 1)),
                p_rise=float((1 + (rs[:, 0] >= lr).sum()) / (M + 1)), first_after_h=float(sil),
                p_silence=float((1 + (rs[:, 1] >= sil).sum()) / (M + 1)))


def main():
    raw, d = load_duc()
    summ = {"duc_raw_n": int(len(raw)), "duc_filtered_n": int(len(d))}
    dsub_all = d[d["sub"]]
    # dedupe templates of the same Plourde family: units = first of burst per family (gap > 1 h)
    du_units = decluster(dsub_all, "fam")
    # dedupe templates of the same Plourde family (detections within 3 s = same LFE)
    du = decluster(dsub_all, "fam", gap=3.0)
    summ["duc_sub_detections"] = int(len(dsub_all)); summ["duc_sub_dedup"] = int(len(du)); summ["duc_sub_units_1h"] = int(len(du_units))
    summ["duc_sub_families"] = int(du.fam.nunique())
    sh = load_shelly()
    shu = decluster(sh, "fam")
    summ["shelly_units_2008_2026"] = int(len(shu))

    # ---------------- ETS episodes 2004-2011 from main band
    t0, t1 = pd.Timestamp("2004-01-01"), pd.Timestamp("2012-01-01")
    eps, nf = episodes(du, t0, t1)
    eps.to_csv(f"{OUT}/compare_ets_episodes.csv", index=False)
    nf.rename("n_active_fams").to_csv(f"{OUT}/compare_active_families_daily.csv")
    summ["n_episodes_2004_2011"] = int(len(eps))
    # tremor check (Boyarko 2015, south of 42N)
    tr = pd.read_csv(f"{REPO}/data/tremor/tremor.xyddtt", sep=r"\s+", header=None, names=["lon", "lat", "x", "y", "dy", "ep"])
    tr = tr[tr.lat < 42]
    tr["t"] = pd.to_datetime(tr.ep, unit="s")
    trd = daily(tr.t, pd.Timestamp("2005-09-01"), pd.Timestamp("2010-12-01"))
    eps["tremor_n"] = [int(((tr.t >= a) & (tr.t < b)).sum()) for a, b in zip(eps.start, eps.end)]
    # fraction of tremor (south of 42N) inside episodes
    intr = np.zeros(len(tr), bool)
    for a, b in zip(eps.start, eps.end):
        intr |= ((tr.t >= a) & (tr.t < b)).to_numpy()
    summ["tremor_frac_in_episodes"] = float(intr.mean())
    eps.to_csv(f"{OUT}/compare_ets_episodes.csv", index=False)

    # ---------------- (1) superposed epoch of Shelly rates around ETS episodes (overlap)
    sh_daily = daily(shu.t, OV0, OV1)
    du_daily_sub = daily(du.t, t0, t1)
    ov_eps = eps[(eps.start >= OV0 + pd.Timedelta(days=30)) & (eps.end <= OV1 - pd.Timedelta(days=30))]
    big = ov_eps[ov_eps.n_fams >= 25]
    res = {}
    for lab, E in [("all", ov_eps), ("big", big)]:
        r = sea(sh_daily, E.start, E.dur_d)
        if len(r) == 0:
            continue
        obs_d = float(np.log(r[:, 1].sum() / r[:, 0].sum())); obs_a = float(np.log(r[:, 2].sum() / r[:, 0].sum()))
        avoid = list(zip(eps.start, eps.end))
        pd_, pa_ = placebo_sea(sh_daily, E.dur_d.to_numpy(), OV0, OV1, avoid=[], M=3000)
        res[lab] = dict(n_eps=int(len(r)), during_over_pre=float(np.exp(obs_d)), after_over_pre=float(np.exp(obs_a)),
                        placebo_during_95=[float(np.exp(np.percentile(pd_, 2.5))), float(np.exp(np.percentile(pd_, 97.5)))],
                        placebo_after_95=[float(np.exp(np.percentile(pa_, 2.5))), float(np.exp(np.percentile(pa_, 97.5)))],
                        p_during_two_sided=float(2 * min((pd_ >= obs_d).mean(), (pd_ <= obs_d).mean())),
                        p_after_two_sided=float(2 * min((pa_ >= obs_a).mean(), (pa_ <= obs_a).mean())),
                        per_episode=[dict(start=str(s.date()), dur=int(dd), pre=float(x[0]), during=float(x[1]), after=float(x[2]))
                                     for s, dd, x in zip(E.start, E.dur_d, r)])
    summ["sea_shelly_vs_ets"] = res
    # stacked curve -60..+60 d around starts, normalized by the episode's pre-30-d mean
    lags = np.arange(-60, 61)
    curves = []
    for s in ov_eps.start:
        seg = sh_daily.reindex(pd.Timestamp(s).floor("D") + pd.to_timedelta(lags, unit="D")).to_numpy(float)
        base = np.nanmean(seg[30:60])
        curves.append(seg / base)
    curves = np.array(curves)
    pd.DataFrame(curves.T, index=lags, columns=[str(s.date()) for s in ov_eps.start]).to_csv(f"{OUT}/compare_sea_shelly_curves.csv")

    # ---------------- (2) cross-correlation of rates (overlap)
    dd = daily(du.t, OV0, OV1).astype(float)
    sd = sh_daily.astype(float)
    def ccf(x, y, lags):
        x = (x - x.mean()) / x.std(); y = (y - y.mean()) / y.std()
        return np.array([np.mean(x[max(0, -L):len(x) - max(0, L)] * y[max(0, L):len(y) - max(0, -L)]) for L in lags])
    # log-ish transform to reduce dominance of ETS bursts
    xd, yd = np.log1p(dd.to_numpy()), np.log1p(sd.to_numpy())
    Ld = np.arange(-30, 31)
    cd = ccf(xd, yd, Ld)   # positive lag L: Shelly at t+L vs Duc at t (Shelly lags)
    nulld = np.array([ccf(xd, np.roll(yd, int(rng.integers(60, len(yd) - 60))), [0])[0] for _ in range(2000)])
    # detrend version: remove 61-d running mean from both (removes slow station-network changes)
    def hp(v, w=61):
        k = np.ones(w) / w
        return v - np.convolve(v, k, mode="same")
    cdh = ccf(hp(xd), hp(yd), Ld)
    nulldh = np.array([ccf(hp(xd), np.roll(hp(yd), int(rng.integers(60, len(yd) - 60))), [0])[0] for _ in range(2000)])
    hx = hourly(du.t, OV0, OV1).to_numpy(float); hy = hourly(shu.t, OV0, OV1).to_numpy(float)
    Lh = np.arange(-72, 73)
    ch = ccf(hx, hy, Lh)
    nullh = np.array([ccf(hx, np.roll(hy, 24 * int(rng.integers(5, len(hy) // 24 - 5))), [0])[0] for _ in range(1000)])
    # hourly, excluding ETS episode days (inter-ETS coupling)
    hidx = pd.date_range(OV0, OV1 - pd.Timedelta(hours=1), freq="h")
    inep = np.zeros(len(hidx), bool)
    for a, b in zip(eps.start, eps.end):
        inep |= (hidx >= a) & (hidx < b)
    chi = ccf(hx[~inep], hy[~inep], Lh)
    pd.DataFrame({"lag_d": Ld, "ccf_daily_log": cd, "ccf_daily_log_hp61": cdh}).to_csv(f"{OUT}/compare_ccf_daily.csv", index=False)
    pd.DataFrame({"lag_h": Lh, "ccf_hourly": ch, "ccf_hourly_interETS": chi}).to_csv(f"{OUT}/compare_ccf_hourly.csv", index=False)
    summ["ccf"] = dict(daily_lag0=float(cd[30]), daily_peak_lag=int(Ld[np.argmax(cd)]), daily_peak=float(cd.max()),
                       daily_null95=[float(np.percentile(nulld, 2.5)), float(np.percentile(nulld, 97.5))],
                       daily_hp_lag0=float(cdh[30]), daily_hp_peak_lag=int(Ld[np.argmax(cdh)]), daily_hp_peak=float(cdh.max()),
                       daily_hp_null95=[float(np.percentile(nulldh, 2.5)), float(np.percentile(nulldh, 97.5))],
                       hourly_lag0=float(ch[72]), hourly_peak_lag=int(Lh[np.argmax(ch)]), hourly_peak=float(ch.max()),
                       hourly_null95=[float(np.percentile(nullh, 2.5)), float(np.percentile(nullh, 97.5))],
                       hourly_interETS_lag0=float(chi[72]), hourly_interETS_peak_lag=int(Lh[np.argmax(chi)]))
    # coincidence within +/-10 min (common false detections, e.g. regional earthquakes)
    tsh = np.sort(shu.tsec.to_numpy())
    tdu = du[(du.t >= OV0) & (du.t < OV1)].tsec.to_numpy()
    j = np.searchsorted(tsh, tdu)
    near = np.minimum(np.abs(tdu - tsh[np.clip(j - 1, 0, len(tsh) - 1)]), np.abs(tsh[np.clip(j, 0, len(tsh) - 1)] - tdu))
    obs_c = float((near <= 60).mean())
    shift = []
    for _ in range(200):
        k = 86400 * rng.integers(3, 400) * rng.choice([-1, 1])
        jj = np.searchsorted(tsh, tdu + k)
        nn = np.minimum(np.abs(tdu + k - tsh[np.clip(jj - 1, 0, len(tsh) - 1)]), np.abs(tsh[np.clip(jj, 0, len(tsh) - 1)] - tdu - k))
        shift.append((nn <= 60).mean())
    summ["coincidence_60s"] = dict(obs=obs_c, null_mean=float(np.mean(shift)), null95=float(np.percentile(shift, 95)))

    # ---------------- (3) tides
    trows = []
    inep_u = np.zeros(len(du), bool)
    for a, b in zip(eps.start, eps.end):
        inep_u |= ((du.t >= a) & (du.t < b)).to_numpy()
    sets = {"duc_sub_2004_2011_all": du.t, "duc_sub_ETS": du.t[inep_u], "duc_sub_interETS": du.t[~inep_u],
            "duc_sub_2008_2011": du.t[(du.t >= OV0) & (du.t < OV1)],
            "duc_sub_units1h_2004_2011": du_units.t,
            "shelly_2008_2011": shu.t[(shu.t >= OV0) & (shu.t < OV1)], "shelly_2008_2026": shu.t}
    # shelly during main-band ETS vs not (overlap)
    insh = np.zeros(len(shu), bool)
    for a, b in zip(eps.start, eps.end):
        insh |= ((shu.t >= a) & (shu.t < b)).to_numpy()
    ovm = ((shu.t >= OV0) & (shu.t < OV1)).to_numpy()
    sets["shelly_2008_2011_during_mainband_ETS"] = shu.t[insh & ovm]
    sets["shelly_2008_2011_outside_mainband_ETS"] = shu.t[~insh & ovm]
    for lab, tt in sets.items():
        for c in ["M2", "N2", "O1", "S2", "K1"]:
            r = tide(tt.reset_index(drop=True), c, 1)
            r["set"] = lab
            trows.append(r)
    tdf = pd.DataFrame(trows)
    tdf.to_csv(f"{OUT}/compare_tides.csv", index=False)

    # ---------------- (4) quakes
    qrows = []
    for q, tq in QUAKES.items():
        for lab, u, a0, a1 in [("duc_sub", du, t0, t1), ("shelly", shu, OV0, pd.Timestamp("2026-07-01"))]:
            if not (a0 + pd.Timedelta(days=30) < tq < a1 - pd.Timedelta(days=30)):
                continue
            r = quake_resp(u, tq, a0, a1)
            r.update(quake=q, pop=lab)
            qrows.append(r)
    qdf = pd.DataFrame(qrows)
    qdf.to_csv(f"{OUT}/compare_quakes.csv", index=False)

    with open(f"{OUT}/compare_summary.json", "w") as f:
        json.dump(summ, f, indent=1, default=str)

    # ---------------- figures
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ax[0].scatter(du.t, du.lat, s=1, c="k", alpha=0.2, lw=0)
    for a, b in zip(eps.start, eps.end):
        for k in range(3):
            ax[k].axvspan(a, b, color=C2, alpha=0.25, lw=0)
    ax[0].set_ylabel("main-band lat (deg)")
    ax[0].set_title("Ducellier & Creager main band (filtered, burst units) vs Shelly MTJ families; orange = ETS episodes")
    ax[1].plot(du_daily_sub.index, du_daily_sub.values, color=C1, lw=0.6)
    ax[1].set_ylabel("main band units/day"); ax[1].set_yscale("symlog")
    ax[1].plot(trd.index, trd.values, color=C3, lw=0.6, label="tremor S of 42N (Boyarko)")
    ax[1].legend(loc="upper left", fontsize=8)
    shd_full = daily(shu.t, OV0, OV1)
    ax[2].plot(shd_full.index, shd_full.values, color=GREY, lw=0.4)
    ax[2].plot(shd_full.index, shd_full.rolling(15, center=True).mean(), color="k", lw=1)
    ax[2].set_ylabel("Shelly units/day")
    for q, tq in QUAKES.items():
        for k in range(3):
            ax[k].axvline(tq, color="r", lw=0.8, ls="--")
    ax[2].set_xlim(t0, t1)
    fig.tight_layout(); fig.savefig(f"{FIG}/sci_ducellier_compare_timeline.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    if len(curves):
        for c in curves:
            ax[0].plot(lags, pd.Series(c).rolling(5, center=True).mean(), color=GREY, lw=0.5, alpha=0.6)
        ax[0].plot(lags, pd.Series(np.nanmean(curves, 0)).rolling(5, center=True).mean(), color="k", lw=2)
    ax[0].axvline(0, color=C2); ax[0].axhline(1, color="k", lw=0.5)
    ax[0].set_xlabel("days from main-band ETS start"); ax[0].set_ylabel("Shelly rate / pre-30-d mean")
    ax[0].set_title(f"Superposed epoch, {len(curves)} episodes 2008-11")
    ax[1].plot(Ld, cd, color=C1, label="daily log1p"); ax[1].plot(Ld, cdh, color=C2, label="daily, 61-d high-pass")
    ax[1].axhspan(*np.percentile(nulldh, [2.5, 97.5]), color=C2, alpha=0.15)
    ax[1].axhspan(*np.percentile(nulld, [2.5, 97.5]), color=C1, alpha=0.15)
    ax[1].set_xlabel("lag (d), + = Shelly later"); ax[1].legend(fontsize=8); ax[1].set_title("Rate CCF 2008-11")
    ax[2].plot(Lh, ch, color=C1, label="all"); ax[2].plot(Lh, chi, color=C3, label="outside ETS")
    ax[2].axhspan(*np.percentile(nullh, [2.5, 97.5]), color=C1, alpha=0.15)
    ax[2].set_xlabel("lag (h), + = Shelly later"); ax[2].legend(fontsize=8); ax[2].set_title("Hourly CCF")
    fig.tight_layout(); fig.savefig(f"{FIG}/sci_ducellier_compare_sea_ccf.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), subplot_kw=dict(projection="polar"))
    cols = {"duc_sub_2004_2011_all": C1, "duc_sub_ETS": C2, "duc_sub_interETS": C3, "shelly_2008_2011": "k",
            "shelly_2008_2026": GREY, "duc_sub_2008_2011": "#8a4fd6"}
    for k, c in enumerate(["M2", "O1"]):
        for lab, col in cols.items():
            r = tdf[(tdf.set == lab) & (tdf.const == c)].iloc[0]
            ax[k].plot([0, np.deg2rad(r.phase_deg)], [0, r.a], color=col, lw=2, label=f"{lab} (n={r.n})")
            ax[k].plot(np.deg2rad(r.phase_deg), r.a, "o", color=col)
        ax[k].set_title(f"{c}: amplitude vs astronomical phase")
    ax[0].legend(fontsize=7, loc="lower left", bbox_to_anchor=(-0.3, -0.35))
    fig.tight_layout(); fig.savefig(f"{FIG}/sci_ducellier_compare_tides.png", dpi=130); plt.close(fig)

    pd.set_option("display.width", 250)
    print(json.dumps(summ, indent=1, default=str)[:6000])
    print(eps.to_string())
    print(tdf[["set", "const", "n", "a", "a_lo", "a_hi", "a_null95", "p", "phase_deg", "phase_sd_deg"]].to_string())
    print(qdf.to_string())


if __name__ == "__main__":
    main()
