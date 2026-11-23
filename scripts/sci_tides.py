#!/usr/bin/env python
"""sci_tides.py - Tidal (M2, N2, O1; MSf) and seasonal sensitivity of MTJ LFEs through time.

Question: does the tidal modulation amplitude of LFE occurrence change through time and at the
2021 M6.2 Petrolia, 2022 M6.4 Ferndale and 2024 M7.0 Mendocino earthquakes?  (Effective-stress
test: higher effective normal stress -> smaller tidal modulation.)  Plus annual LFE-rate cycle vs
coda dv/v (fastest ~day 250).

Design (incl. reviewer advice):
  * No tidal-strain package (pygtide etc.) is installed -> astronomical phases of individual
    constituents, phi = 2*pi*frac((t - 2000-01-01T12:00 UTC)/P).  Equilibrium (body-tide potential)
    phase also given for M2/N2/MSf.  Phase is NOT Coulomb-stress phase (ocean loading near coast);
    only the modulation AMPLITUDE is interpreted.
  * Conclusions from M2, N2, O1 only.  S2, K1, S1 are computed only as detection-cycle controls.
  * Units: events with meancc >= 0.137 (catalog median; fixed threshold that gives the most stable
    yearly counts), declustered per family -> first event of each burst (gap > 1 h).
  * Modulation: rate ~ 1 + a cos(phi - phi0);  a = 2|mean exp(i phi)|, debiased with the
    clustered-Schuster expectation sum_b |S_b|^2 (blocks = UTC days, joint across families).
  * Uncertainty: Poisson day-block bootstrap (joint across families).
  * Null: each day-block shifted by an independent random INTEGER number of solar days
    (|k| = 1..400): preserves time-of-day detection pattern, scrambles lunar phase.
  * Pooling across families: Neff-weighted mean of debiased per-family amplitudes ("pooled"),
    and the network sum assuming a common phase ("network").
Outputs: data/sci/tides/*.csv, summary.json; figures/sci_tides_*.png
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
import json
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.stats import binomtest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/seismic/LFE"
OUT = f"{ROOT}/data/sci/tides"
FIG = f"{ROOT}/figures"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261122)

TREF = pd.Timestamp("2000-01-01T12:00:00")
T0 = pd.Timestamp("2008-01-01")
LAT, LON = 40.06, -123.62
PER_H = {"M2": 12.4206012, "N2": 12.65834751, "O1": 25.81933871,
         "S2": 12.0, "K1": 23.93447213, "S1": 24.0, "MSf": 14.765294 * 24}
PRIMARY = ["M2", "N2", "O1"]
CONTROL = ["S2", "K1", "S1"]
QUAKES = {"Petrolia": pd.Timestamp("2021-12-20T20:10"),
          "Ferndale": pd.Timestamp("2022-12-20T10:34"),
          "Mendocino": pd.Timestamp("2024-12-05T18:44")}
NF = 27
NMIN = 20
MEANCC_THR = 0.137
CCTHR_ALT = 1.5
GAP_S = 3600.0
B_BOOT, M_NULL = 1000, 2000
DVV_FAST_DOY = 250.0
C1, C2, C3, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


# ----------------------------------------------------------------------------------- data
def load():
    d = pd.read_csv(f"{ROOT}/data/shelly2026_MTJ_v2/LFEcatV2_MTJ_2008-2026.csv")
    d.columns = [c.lstrip("%") for c in d.columns]
    t = pd.to_datetime(dict(year=d.year, month=d.month, day=d.day)) + pd.to_timedelta(d.s_of_day, unit="s")
    d["t"] = t
    d["fam"] = d.idnum.astype(int) - 1
    d["tsec"] = (t - TREF).dt.total_seconds()
    d["day"] = np.floor((t - T0).dt.total_seconds() / 86400).astype(int)
    d["doy"] = t.dt.day_of_year
    d = d.sort_values("t").reset_index(drop=True)
    for c, P in PER_H.items():
        d["ph_" + c] = 2 * np.pi * np.mod(d.tsec / 3600.0 / P, 1.0)
    # equilibrium (Doodson) arguments; potential max at argument 0 for semidiurnals (cos^2 lat > 0)
    jd = d.tsec / 86400.0  # days from J2000 (UT ~ TT here; 64 s -> <0.01 rad for M2)
    T = jd / 36525.0
    s = np.deg2rad(218.3164477 + 481267.88123421 * T)
    h = np.deg2rad(280.46646 + 36000.76983 * T)
    p = np.deg2rad(83.3532465 + 4069.0137287 * T)
    gmst = np.deg2rad(280.46061837 + 360.98564736629 * jd)
    tau = gmst + np.deg2rad(LON) - s
    eq = {"M2": 2 * tau, "N2": 2 * tau - s + p, "MSf": 2 * (s - h)}
    off = {}
    for c, a in eq.items():
        dd = wrap(a - d["ph_" + c])
        off[c] = float(np.angle(np.mean(np.exp(1j * dd))))
        off[c + "_spread_deg"] = float(np.rad2deg(np.std(wrap(dd - off[c]))))
    return d, off


def decluster(df):
    df = df.sort_values(["fam", "t"])
    g = df.groupby("fam").tsec.diff()
    first = g.isna() | (g > GAP_S)
    out = df[first.to_numpy()].copy()
    return out.sort_values("t").reset_index(drop=True)


def add_msf_weight(df):
    """weight = 1 / (31-day centred running mean of the family's daily unit count)."""
    nd = int(df.day.max()) + 1
    cnt = np.zeros((NF, nd))
    np.add.at(cnt, (df.fam.to_numpy(), df.day.to_numpy()), 1.0)
    m = uniform_filter1d(cnt, 31, axis=1, mode="constant")
    df = df.copy()
    df["w_msf"] = 1.0 / m[df.fam.to_numpy(), df.day.to_numpy()]
    return df


# ----------------------------------------------------------------------------------- core stats
def damp(Ssum, N, Q):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(N > 0, 2 * np.sqrt(np.maximum(np.abs(Ssum) ** 2 - Q, 0)) / N, np.nan)


def tidal_block(df, const, blockdays=1, weight=None, B=B_BOOT, M=M_NULL):
    """Per-family + network sums, bootstrap and integer-day-shift null for one constituent."""
    phi = df["ph_" + const].to_numpy()
    fam = df.fam.to_numpy()
    ub, blk = np.unique(df.day.to_numpy() // blockdays, return_inverse=True)
    nb = len(ub)
    w = np.ones(len(df)) if weight is None else df[weight].to_numpy()
    idx = fam * nb + blk
    S = (np.bincount(idx, w * np.cos(phi), NF * nb) + 1j * np.bincount(idx, w * np.sin(phi), NF * nb)).reshape(NF, nb)
    n = np.bincount(idx, w, NF * nb).reshape(NF, nb)
    cnt = np.bincount(fam, minlength=NF)
    Snet, nnet = S.sum(0), n.sum(0)
    A2 = np.abs(S) ** 2
    W = rng.poisson(1.0, (B, nb)).astype(float)
    W2 = W ** 2
    k = rng.integers(1, 401, (M, nb)) * rng.choice([-1, 1], (M, nb))
    E = np.exp(1j * 2 * np.pi * np.mod(24.0 * k / PER_H[const], 1.0))
    r = dict(const=const, cnt=cnt, nb=nb,
             Ssum=S.sum(1), N=n.sum(1), Q=A2.sum(1),
             Neff=np.where(n.sum(1) > 0, n.sum(1) ** 2 / np.maximum((n ** 2).sum(1), 1e-30), 0),
             bS=W @ S.T, bN=W @ n.T, bQ=W2 @ A2.T,
             zS=E @ S.T,
             net_S=Snet.sum(), net_N=nnet.sum(), net_Q=(np.abs(Snet) ** 2).sum(),
             net_bS=W @ Snet, net_bN=W @ nnet, net_bQ=W2 @ np.abs(Snet) ** 2,
             net_zS=E @ Snet)
    r["a"] = damp(r["Ssum"], r["N"], r["Q"])
    r["ab"] = damp(r["bS"], r["bN"], r["bQ"])            # (B, NF)
    r["az"] = damp(r["zS"], r["N"][None, :], r["Q"][None, :])  # (M, NF)
    r["net_a"] = float(damp(r["net_S"], r["net_N"], r["net_Q"]))
    r["net_ab"] = damp(r["net_bS"], r["net_bN"], r["net_bQ"])
    r["net_az"] = damp(r["net_zS"], r["net_N"], r["net_Q"])
    return r


def pool(r, mask, wts):
    w = np.where(mask, wts, 0.0)
    w = w / w.sum()
    a = np.nansum(w * np.nan_to_num(r["a"]))
    ab = (np.nan_to_num(r["ab"]) * w[None, :]).sum(1)
    az = (np.nan_to_num(r["az"]) * w[None, :]).sum(1)
    return a, ab, az


def pval(obs, null):
    return (1 + np.sum(null >= obs)) / (len(null) + 1)


def circ_sd_deg(angles):
    R = np.abs(np.mean(np.exp(1j * angles)))
    return float(np.rad2deg(np.sqrt(-2 * np.log(max(R, 1e-12)))))


def summarize(r, label, off):
    """Rows: per family, pooled, network."""
    c = r["const"]
    rows = []
    mask = r["cnt"] >= NMIN
    for f in range(NF):
        if r["cnt"][f] == 0:
            continue
        ph = np.angle(r["Ssum"][f])
        rows.append(dict(window=label, const=c, family=f + 1, n_units=int(r["cnt"][f]), Neff=r["Neff"][f],
                         a_raw=2 * abs(r["Ssum"][f]) / r["N"][f], a=r["a"][f],
                         a_se=np.nanstd(r["ab"][:, f]), a_lo=np.nanpercentile(r["ab"][:, f], 2.5),
                         a_hi=np.nanpercentile(r["ab"][:, f], 97.5),
                         a_null95=np.nanpercentile(r["az"][:, f], 95), p_null=pval(r["a"][f], r["az"][:, f]),
                         log10p_schuster=-(abs(r["Ssum"][f]) ** 2 / r["Q"][f]) / np.log(10) if r["Q"][f] > 0 else np.nan,
                         phase_deg=np.rad2deg(ph) % 360,
                         phase_sd_deg=circ_sd_deg(np.angle(r["bS"][:, f])),
                         phase_eq_deg=(np.rad2deg(ph + off[c]) % 360) if c in off else np.nan))
    a, ab, az = pool(r, mask, r["Neff"])
    rows.append(dict(window=label, const=c, family="pooled", n_units=int(r["cnt"][mask].sum()),
                     Neff=float(r["Neff"][mask].sum()), a=a, a_se=np.std(ab), a_lo=np.percentile(ab, 2.5),
                     a_hi=np.percentile(ab, 97.5), a_null95=np.percentile(az, 95), p_null=pval(a, az),
                     n_fam=int(mask.sum())))
    ph = np.angle(r["net_S"])
    rows.append(dict(window=label, const=c, family="network", n_units=int(r["cnt"].sum()),
                     a_raw=2 * abs(r["net_S"]) / r["net_N"], a=r["net_a"], a_se=np.nanstd(r["net_ab"]),
                     a_lo=np.nanpercentile(r["net_ab"], 2.5), a_hi=np.nanpercentile(r["net_ab"], 97.5),
                     a_null95=np.nanpercentile(r["net_az"], 95), p_null=pval(r["net_a"], r["net_az"]),
                     log10p_schuster=-(abs(r["net_S"]) ** 2 / r["net_Q"]) / np.log(10),
                     phase_deg=np.rad2deg(ph) % 360, phase_sd_deg=circ_sd_deg(np.angle(r["net_bS"])),
                     phase_eq_deg=(np.rad2deg(ph + off[c]) % 360) if c in off else np.nan))
    return rows


def compare(rb, ra, name, lb, la):
    """Change in modulation between two windows, same family set, fixed weights."""
    mask = (rb["cnt"] >= NMIN) & (ra["cnt"] >= NMIN)
    wts = np.where(mask, 1.0 / (1.0 / np.maximum(rb["Neff"], 1e-9) + 1.0 / np.maximum(ra["Neff"], 1e-9)), 0)
    Ab, abb, azb = pool(rb, mask, wts)
    Aa, aba, aza = pool(ra, mask, wts)
    d = Aa - Ab
    se = np.sqrt(np.var(abb) + np.var(aba))
    ratio_b = aba / np.where(abb > 0, abb, np.nan)
    dfam = (ra["a"] - rb["a"])[mask]
    nneg = int(np.sum(dfam < 0))
    php_b, php_a = np.angle(rb["net_S"]), np.angle(ra["net_S"])
    dph_b = wrap(np.angle(ra["net_bS"]) - np.angle(rb["net_bS"]))
    return dict(comparison=name, const=rb["const"], before=lb, after=la, n_fam=int(mask.sum()),
                n_before=int(rb["cnt"][mask].sum()), n_after=int(ra["cnt"][mask].sum()),
                A_before=Ab, A_before_se=np.std(abb), A_before_p=pval(Ab, azb),
                A_after=Aa, A_after_se=np.std(aba), A_after_p=pval(Aa, aza),
                dA=d, dA_se=se, z=d / se, ratio=Aa / Ab if Ab > 0 else np.nan,
                ratio_lo=np.nanpercentile(ratio_b, 2.5), ratio_hi=np.nanpercentile(ratio_b, 97.5),
                fam_n_decrease=nneg, fam_sign_p=binomtest(nneg, int(mask.sum()), 0.5).pvalue if mask.sum() else np.nan,
                net_A_before=rb["net_a"], net_A_after=ra["net_a"],
                net_dA=ra["net_a"] - rb["net_a"],
                net_dA_se=np.sqrt(np.nanvar(rb["net_ab"]) + np.nanvar(ra["net_ab"])),
                net_phase_before=np.rad2deg(php_b) % 360, net_phase_after=np.rad2deg(php_a) % 360,
                dphase_deg=np.rad2deg(wrap(php_a - php_b)), dphase_sd_deg=circ_sd_deg(dph_b))


def win(df, t0, t1):
    return df[(df.t >= t0) & (df.t < t1)]


# ----------------------------------------------------------------------------------- seasonal
def annual_fit(y, doy):
    ok = np.isfinite(y)
    if ok.sum() < 6:
        return np.nan, np.nan
    om = 2 * np.pi * doy[ok] / 365.25
    X = np.c_[np.cos(om), np.sin(om)]
    a, b = np.linalg.lstsq(X, y[ok], rcond=None)[0]
    return float(np.hypot(a, b)), float((np.arctan2(b, a) / (2 * np.pi) * 365.25) % 365.25)


def seasonal(units, label, months, excl):
    """units: df with t, fam. Returns per-family + network rows and monthly table."""
    mi = ((units.t.dt.year - 2008) * 12 + units.t.dt.month - 1).to_numpy()
    nm = len(months)
    C = np.zeros((NF, nm))
    np.add.at(C, (units.fam.to_numpy(), mi), 1.0)
    series = {f + 1: C[f] for f in range(NF)}
    series["network"] = C.sum(0)
    doy = months.doy.to_numpy().astype(float)
    yr = months.year.to_numpy()
    use = (yr >= 2009) & ~excl
    rows, monthly = [], {}
    for key, c in series.items():
        rm = pd.Series(c).rolling(24, center=True, min_periods=18).mean().to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            rr = c / rm - 1.0
        rr[~use | ~np.isfinite(rr)] = np.nan
        monthly[key] = rr
        A, dmax = annual_fit(rr, doy)
        # year matrix for nulls / bootstrap
        yrs = np.arange(2009, 2027)
        Y = np.full((len(yrs), 12), np.nan)
        for i, y in enumerate(yrs):
            sel = yr == y
            Y[i, months.month.to_numpy()[sel] - 1] = rr[sel]
        D = np.tile(months.doy.to_numpy()[:12].astype(float), (len(yrs), 1))  # mid-month doys (2008 grid)
        nullA = np.empty(5000)
        for b in range(5000):
            Ys = np.array([np.roll(row, rng.integers(0, 12)) for row in Y])
            nullA[b] = annual_fit(Ys.ravel(), D.ravel())[0]
        bootA, bootD = np.empty(1000), np.empty(1000)
        for b in range(1000):
            ii = rng.integers(0, len(yrs), len(yrs))
            bootA[b], bootD[b] = annual_fit(Y[ii].ravel(), D[ii].ravel())
        # per-year phases -> Rayleigh
        ph = []
        for row in Y:
            if np.isfinite(row).sum() >= 10:
                ph.append(annual_fit(row, D[0])[1])
        ph = np.array(ph)
        Rbar = np.abs(np.mean(np.exp(2j * np.pi * ph / 365.25)))
        nY = len(ph)
        pray = np.exp(np.sqrt(1 + 4 * nY + 4 * (nY ** 2 - (nY * Rbar) ** 2)) - (1 + 2 * nY))
        dsd = np.rad2deg(np.sqrt(-2 * np.log(max(np.abs(np.mean(np.exp(2j * np.pi * bootD / 365.25))), 1e-12)))) * 365.25 / 360
        rows.append(dict(subset=label, series=key, n_units=int(c.sum()), A=A, A_se=np.nanstd(bootA),
                         A_lo=np.nanpercentile(bootA, 2.5), A_hi=np.nanpercentile(bootA, 97.5),
                         A_null95=np.nanpercentile(nullA, 95), p_null=pval(A, nullA),
                         doy_max=dmax, doy_max_sd=dsd,
                         doy_min=(dmax + 365.25 / 2) % 365.25,
                         dmax_minus_dvvfast=((dmax - DVV_FAST_DOY + 182.625) % 365.25) - 182.625,
                         n_years=nY, rayleigh_R=Rbar, rayleigh_p=pray))
    return rows, monthly


# ----------------------------------------------------------------------------------- main
def main():
    d, off = load()
    print("equilibrium offsets (rad) and spreads:", off)
    allunits = decluster(d)
    units = decluster(d[d.meancc >= MEANCC_THR])
    units_cc = decluster(d[d["cc/thr"] >= CCTHR_ALT])
    units = add_msf_weight(units)
    print(f"events {len(d)}, declustered all {len(allunits)}, meancc>={MEANCC_THR} units {len(units)}, "
          f"cc/thr>={CCTHR_ALT} units {len(units_cc)}")
    summ = dict(n_events=len(d), n_units_primary=len(units), n_units_all_declustered=len(allunits),
                n_units_ccthr=len(units_cc), unit_definition=f"meancc>={MEANCC_THR}, first event per family burst (gap>1h)",
                equilibrium_offset_deg={k: float(np.rad2deg(v)) for k, v in off.items() if "spread" not in k},
                pygtide_available=False)
    yc = units.groupby(units.t.dt.year).size()
    summ["units_per_year"] = {int(k): int(v) for k, v in yc.items()}

    # ---- (a) whole period, all constituents, subsets
    rows = []
    res_whole = {}
    for lab, df in [("primary", units), ("all_declustered", allunits), ("ccthr1.5", units_cc), ("all_events", d)]:
        for c in PRIMARY + CONTROL:
            r = tidal_block(df, c)
            rows += [dict(subset=lab, **x) for x in summarize(r, "2008-2026", off)]
            if lab == "primary":
                res_whole[c] = r
        r = tidal_block(add_msf_weight(df) if lab != "primary" else df, "MSf", blockdays=7, weight="w_msf")
        rows += [dict(subset=lab, **x) for x in summarize(r, "2008-2026", off)]
        if lab == "primary":
            res_whole["MSf"] = r
    whole = pd.DataFrame(rows)
    whole.to_csv(f"{OUT}/tidal_whole_period.csv", index=False)
    pw = whole[whole.family.isin(["pooled", "network"])]
    print(pw[["subset", "const", "family", "n_units", "a", "a_se", "a_null95", "p_null", "log10p_schuster", "phase_deg", "phase_eq_deg"]].to_string())
    # family phase coherence for M2
    fm = whole[(whole.subset == "primary") & (whole.const == "M2") & ~whole.family.isin(["pooled", "network"])]
    fm = fm[fm.n_units >= NMIN]
    zz = np.exp(1j * np.deg2rad(fm.phase_deg.to_numpy().astype(float)))
    summ["M2_family_phase_coherence_R"] = float(np.abs(zz.mean()))
    summ["M2_family_phase_mean_deg"] = float(np.rad2deg(np.angle(zz.mean())) % 360)
    summ["M2_n_families_p_lt_0.05"] = int((fm.p_null < 0.05).sum())
    pp = pw[(pw.subset == "primary") & (pw.family == "pooled")].set_index("const").a
    summ["amplitude_ratios_obs_vs_equilibrium"] = dict(
        N2_M2=[float(pp["N2"] / pp["M2"]), 0.191], K1_O1=[float(pp["K1"] / pp["O1"]), 1.408],
        S2_M2=[float(pp["S2"] / pp["M2"]), 0.466], note="equilibrium-tide coefficient ratios (Cartwright-Tayler); S2,K1 are detection-cycle-contaminated controls")
    summ["whole_period"] = pw.replace({np.nan: None}).to_dict(orient="records")

    # ---- (b) per calendar year
    rows = []
    for y in range(2008, 2027):
        for lab, df in [("primary", units), ("ccthr1.5", units_cc)]:
            sub = win(df, pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 1}-01-01"))
            for c in PRIMARY + (["S2", "K1"] if lab == "primary" else []):
                r = tidal_block(sub, c, B=500, M=1000)
                rows += [dict(subset=lab, year=y, **x) for x in summarize(r, str(y), off) if x["family"] in ("pooled", "network")]
            if lab == "primary":
                r = tidal_block(sub, "MSf", blockdays=7, weight="w_msf", B=500, M=1000)
                rows += [dict(subset=lab, year=y, **x) for x in summarize(r, str(y), off) if x["family"] in ("pooled", "network")]
    byyear = pd.DataFrame(rows)
    byyear.to_csv(f"{OUT}/tidal_by_year.csv", index=False)
    print(byyear[(byyear.subset == "primary") & (byyear.const == "M2")][["year", "family", "n_units", "a", "a_se", "p_null", "phase_deg"]].to_string())

    # ---- (c) quake windows
    yr = pd.Timedelta(days=365.25)
    hy = pd.Timedelta(days=182.625)
    m7 = QUAKES["Mendocino"]
    restart = d[d.t > m7].groupby("fam").t.min()
    restart_net = units[units.t > m7].groupby("fam").t.min().median()
    summ["M7_restart_median_family"] = str(restart_net)
    comps, famrows, wins = [], [], []

    def restart_window(df, dt):
        keep = np.zeros(len(df), bool)
        for f, t0 in restart.items():
            keep |= ((df.fam == f) & (df.t >= t0) & (df.t < t0 + dt)).to_numpy()
        return df[keep]

    def first_last(df, tq, nw, after=True):
        if after:
            return df[df.t >= tq].iloc[:nw]
        return df[df.t < tq].iloc[-nw:]

    NW = 1500
    windows = []
    for q, tq in QUAKES.items():
        windows.append((f"{q}_1yr", lambda df, tq=tq: win(df, tq - yr, tq), lambda df, tq=tq: win(df, tq, tq + yr),
                        f"[{q}-1yr,{q})", f"[{q},{q}+1yr)"))
        ta = restart_net if q == "Mendocino" else tq
        windows.append((f"{q}_equalcount{NW}", lambda df, tq=tq: first_last(df, tq, NW, False),
                        lambda df, ta=ta: first_last(df, ta, NW, True), f"last {NW} before", f"first {NW} after"))
    windows.append(("M7_restart_6mo", lambda df: win(df, m7 - hy, m7), lambda df: restart_window(df, hy),
                    "[M7-6mo,M7)", "[restart_f, restart_f+6mo)"))
    windows.append(("M7_restart_60d", lambda df: win(df, m7 - pd.Timedelta(days=60), m7),
                    lambda df: restart_window(df, pd.Timedelta(days=60)), "[M7-60d,M7)", "[restart_f, restart_f+60d)"))
    for name, fb, fa, lb, la in windows:
        for lab, df in [("primary", units), ("ccthr1.5", units_cc)]:
            db, da = fb(df), fa(df)
            consts = PRIMARY + (["MSf"] if lab == "primary" else [])
            for c in consts:
                kw = dict(blockdays=7, weight="w_msf") if c == "MSf" else {}
                rb, ra = tidal_block(db, c, **kw), tidal_block(da, c, **kw)
                comps.append(dict(subset=lab, t_before=f"{db.t.min()}..{db.t.max()}", t_after=f"{da.t.min()}..{da.t.max()}",
                                  **compare(rb, ra, name, lb, la)))
                if lab == "primary":
                    for x in summarize(rb, name + "_before", off) + summarize(ra, name + "_after", off):
                        famrows.append(x)
    comp = pd.DataFrame(comps)
    comp.to_csv(f"{OUT}/tidal_quake_changes.csv", index=False)
    pd.DataFrame(famrows).to_csv(f"{OUT}/tidal_quake_windows_by_family.csv", index=False)
    print(comp[["subset", "comparison", "const", "n_fam", "n_before", "n_after", "A_before", "A_before_p", "A_after", "A_after_p",
                "dA", "dA_se", "z", "ratio", "ratio_lo", "ratio_hi", "fam_n_decrease", "dphase_deg", "dphase_sd_deg"]].to_string())

    # ---- (c') equal-count sliding (non-overlapping) windows, segmented at the quakes
    bounds = [units.t.min(), QUAKES["Petrolia"], QUAKES["Ferndale"], m7, units.t.max() + pd.Timedelta(seconds=1)]
    srows = []
    for i in range(len(bounds) - 1):
        seg = win(units, bounds[i], bounds[i + 1])
        if i == len(bounds) - 2:
            seg = seg[seg.t >= restart_net]  # post-M7 segment starts at the network restart
        nwin = max(1, int(round(len(seg) / NW)))
        edges = np.linspace(0, len(seg), nwin + 1).astype(int)
        for j in range(nwin):
            sub = seg.iloc[edges[j]:edges[j + 1]]
            row = dict(segment=i, t_start=sub.t.min(), t_end=sub.t.max(), t_mid=sub.t.min() + (sub.t.max() - sub.t.min()) / 2,
                       n_units=len(sub))
            for c in PRIMARY:
                r = tidal_block(sub, c, B=400, M=1000)
                mask = r["cnt"] >= NMIN
                a, ab, az = pool(r, mask, r["Neff"])
                row.update({f"{c}_A": a, f"{c}_A_se": np.std(ab), f"{c}_A_null95": np.percentile(az, 95),
                            f"{c}_p": pval(a, az), f"{c}_net": r["net_a"], f"{c}_net_se": np.nanstd(r["net_ab"]),
                            f"{c}_net_null95": np.nanpercentile(r["net_az"], 95), f"{c}_net_p": pval(r["net_a"], r["net_az"]),
                            f"{c}_net_phase": np.rad2deg(np.angle(r["net_S"])) % 360, f"{c}_nfam": int(mask.sum())})
            srows.append(row)
    sl = pd.DataFrame(srows)
    sl.to_csv(f"{OUT}/tidal_equalcount_windows.csv", index=False)
    print(sl[["t_start", "t_end", "n_units", "M2_A", "M2_A_se", "M2_A_null95", "M2_p", "M2_net", "M2_net_p", "M2_net_phase"]].to_string())
    # empirical null for step changes: differences of adjacent windows not straddling a quake
    eq = {}
    for c in PRIMARY:
        A = sl[f"{c}_A"].to_numpy()
        seg = sl.segment.to_numpy()
        same = seg[1:] == seg[:-1]
        dif = np.diff(A)
        d_in = dif[same]
        cross = {}
        for k in np.where(~same)[0]:
            cross[str(sl.t_start.iloc[k + 1].date())] = dict(dA=float(dif[k]), z_emp=float(dif[k] / d_in.std()),
                                                             frac_abs_ge=float(np.mean(np.abs(d_in) >= abs(dif[k]))))
        pre = sl[sl.segment < 3][f"{c}_A"]
        post = sl[sl.segment == 3][f"{c}_A"]
        # linear trend of amplitude with time (pre-M7 windows)
        tt = (sl.t_mid - T0).dt.total_seconds().to_numpy() / 86400 / 365.25
        msk = (sl.segment < 3).to_numpy()
        X = np.c_[np.ones(msk.sum()), tt[msk]]
        coef, res, *_ = np.linalg.lstsq(X, A[msk], rcond=None)
        resid = A[msk] - X @ coef
        cov = np.linalg.inv(X.T @ X) * resid.var(ddof=2)
        eq[c] = dict(adjacent_diff_sd=float(d_in.std()), quake_boundaries=cross,
                     pre_M7_mean=float(pre.mean()), pre_M7_sd=float(pre.std()), post_restart_mean=float(post.mean()),
                     post_restart_first=float(post.iloc[0]) if len(post) else None,
                     post_first_z=float((post.iloc[0] - pre.mean()) / pre.std()) if len(post) else None,
                     post_first_rank_frac_pre_below=float(np.mean(pre < post.iloc[0])) if len(post) else None,
                     trend_per_yr=float(coef[1]), trend_se=float(np.sqrt(cov[1, 1])),
                     frac_windows_p_lt_0p05=float(np.mean(sl[f"{c}_p"] < 0.05)),
                     chi2_const_all=float(np.sum(((A - np.average(A, weights=sl[f"{c}_A_se"] ** -2)) / sl[f"{c}_A_se"]) ** 2)),
                     chi2_dof=int(len(A) - 1))
        from scipy.stats import chi2 as _chi2
        eq[c]["chi2_const_p"] = float(_chi2.sf(eq[c]["chi2_const_all"], eq[c]["chi2_dof"]))
    summ["equal_count_windows"] = dict(NW=NW, n_windows=len(sl), stats=eq)
    print(json.dumps(eq, indent=1, default=str))

    # ---- seasonal tidal sensitivity: M2 amplitude in fast-dv/v half-year vs slow half
    dd = ((units.doy - DVV_FAST_DOY + 182.625) % 365.25) - 182.625
    fast, slow = units[np.abs(dd) <= 91.3], units[np.abs(dd) > 91.3]
    srow = []
    for c in PRIMARY:
        srow.append(dict(subset="primary", **compare(tidal_block(slow, c), tidal_block(fast, c), "season_fast_vs_slow_dvv",
                                                     "slow-dv/v half (doy 342-159)", "fast-dv/v half (doy 159-341)")))
    seas_tide = pd.DataFrame(srow)
    seas_tide.to_csv(f"{OUT}/tidal_by_season.csv", index=False)
    print(seas_tide[["const", "A_before", "A_after", "dA", "dA_se", "z", "ratio", "ratio_lo", "ratio_hi"]].to_string())

    # ---- (d) seasonal LFE rate
    months = pd.DataFrame(dict(date=pd.date_range("2008-01-01", "2026-06-01", freq="MS")))
    months["year"], months["month"] = months.date.dt.year, months.date.dt.month
    months["doy"] = (months.date + pd.Timedelta(days=14.5)).dt.dayofyear
    excl = ((months.date >= pd.Timestamp("2024-12-01")) & (months.date < pd.Timestamp("2025-02-01"))).to_numpy()
    srows, monthly_all = [], {}
    for lab, df in [("primary", units), ("all_declustered", allunits), ("ccthr1.5", units_cc)]:
        rr, mo = seasonal(df, lab, months, excl)
        srows += rr
        monthly_all[lab] = mo
    # noise proxy: daily network-median detection threshold (MAD-based) -> monthly mean
    thr = d.groupby("day").thresh.median()
    tday = T0 + pd.to_timedelta(thr.index, unit="D")
    mth = pd.Series(thr.to_numpy(), index=tday).resample("MS").mean().reindex(months.date)
    rm = mth.rolling(24, center=True, min_periods=18).mean()
    thr_r = np.array((mth / rm - 1).to_numpy(), dtype=float)
    thr_r[(months.year < 2009).to_numpy() | excl] = np.nan
    A, dm = annual_fit(thr_r, months.doy.to_numpy().astype(float))
    srows.append(dict(subset="noise_proxy_thresh", series="network", A=A, doy_max=dm))
    seas = pd.DataFrame(srows)
    seas.to_csv(f"{OUT}/seasonal_annual_fit.csv", index=False)
    mon = months[["date", "year", "month", "doy"]].copy()
    for lab, mo in monthly_all.items():
        mon[f"{lab}_network_ratio_minus1"] = mo["network"]
    mon["thresh_ratio_minus1"] = thr_r
    mon.to_csv(f"{OUT}/seasonal_monthly.csv", index=False)
    print(seas[seas.series == "network"].to_string())
    sf = seas[(seas.subset == "primary") & (seas.series != "network")]
    zf = np.exp(2j * np.pi * sf.doy_max.to_numpy().astype(float) / 365.25)
    summ["seasonal"] = dict(network=seas[seas.series == "network"].replace({np.nan: None}).to_dict(orient="records"),
                            n_families_p_lt_0p05=int((sf.p_null < 0.05).sum()),
                            family_doy_max_circmean=float((np.angle(zf.mean()) / (2 * np.pi) * 365.25) % 365.25),
                            family_doy_max_R=float(np.abs(zf.mean())),
                            noise_proxy=dict(A=A, doy_max=dm))
    print("families seasonal p<0.05:", summ["seasonal"]["n_families_p_lt_0p05"], "circmean", summ["seasonal"]["family_doy_max_circmean"],
          "R", summ["seasonal"]["family_doy_max_R"])

    summ["quake_changes_primary"] = comp[comp.subset == "primary"].replace({np.nan: None}).to_dict(orient="records")
    summ["quake_changes_ccthr"] = comp[comp.subset == "ccthr1.5"].replace({np.nan: None}).to_dict(orient="records")
    summ["season_tidal"] = seas_tide.replace({np.nan: None}).to_dict(orient="records")
    with open(f"{OUT}/summary.json", "w") as fh:
        json.dump(summ, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    # ----------------------------------------------------------------------------------- figures
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    # Fig 1: amplitude through time
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for c, col in zip(PRIMARY, [C1, C2, C3]):
        ax[0].errorbar(sl.t_mid, sl[f"{c}_A"], yerr=sl[f"{c}_A_se"], fmt="o-", ms=4, lw=1, color=col, label=f"{c} pooled (families)")
        ax[0].plot(sl.t_mid, sl[f"{c}_A_null95"], ":", color=col, lw=1)
    ax[0].set_ylabel("modulation amplitude a\n(equal-count windows, %d units)" % NW)
    ax[0].legend(loc="upper left", frameon=False, ncol=3)
    ax[0].set_title("Tidal modulation of LFE occurrence (declustered, meancc>=%.3f); dotted = 95%% of integer-day-shift null" % MEANCC_THR)
    yy = byyear[(byyear.subset == "primary") & (byyear.family == "pooled")]
    yc_ = byyear[(byyear.subset == "ccthr1.5") & (byyear.family == "pooled")]
    for c, col in zip(PRIMARY, [C1, C2, C3]):
        s = yy[yy.const == c]
        ax[1].errorbar(pd.to_datetime(s.year.astype(str) + "-07-01"), s.a, yerr=s.a_se, fmt="o-", ms=4, lw=1, color=col, label=c)
    s = yc_[yc_.const == "M2"]
    ax[1].errorbar(pd.to_datetime(s.year.astype(str) + "-07-01"), s.a, yerr=s.a_se, fmt="s--", ms=3, lw=1, color=GREY, label="M2, cc/thr>=1.5 subset")
    ax[1].set_ylabel("per calendar year\npooled amplitude a")
    ax[1].legend(loc="upper left", frameon=False, ncol=4)
    s = yy[yy.const == "M2"]
    net = byyear[(byyear.subset == "primary") & (byyear.family == "network") & (byyear.const == "M2")]
    ax[2].errorbar(pd.to_datetime(net.year.astype(str) + "-07-01"), net.phase_deg, yerr=net.phase_sd_deg, fmt="o", color=C1)
    ax[2].set_ylabel("M2 network phase of max rate\n(deg, astronomical ref.)")
    ax[2].set_ylim(0, 360)
    for a_ in ax:
        for q, tq in QUAKES.items():
            a_.axvline(tq, color=GREY, lw=0.8, ls="--")
    for q, tq in QUAKES.items():
        ax[0].text(tq, ax[0].get_ylim()[1], q, rotation=90, va="top", ha="right", fontsize=7, color=GREY)
    fig.tight_layout()
    fig.savefig(f"{FIG}/sci_tides_amplitude_through_time.png", dpi=150)
    plt.close(fig)

    # Fig 2: whole-period rate vs phase + per-family vectors + quake comparisons
    fig = plt.figure(figsize=(11, 7.5))
    for i, c in enumerate(["M2", "N2", "O1", "MSf"]):
        axh = fig.add_subplot(2, 4, i + 1)
        ph = units["ph_" + c].to_numpy()
        w = units.w_msf.to_numpy() if c == "MSf" else np.ones(len(ph))
        h, e = np.histogram(ph, bins=18, range=(0, 2 * np.pi), weights=w)
        cen = 0.5 * (e[1:] + e[:-1])
        axh.bar(np.rad2deg(cen), h / h.mean(), width=18, color=C1, edgecolor="white", lw=1)
        r = res_whole[c]
        aa, ph0 = 2 * abs(r["net_S"]) / r["net_N"], np.angle(r["net_S"])
        xx = np.linspace(0, 2 * np.pi, 200)
        axh.plot(np.rad2deg(xx), 1 + aa * np.cos(xx - ph0), color=C2, lw=2)
        axh.set_ylim(0.6, 1.4)
        rw = pw[(pw.subset == "primary") & (pw.const == c) & (pw.family == "pooled")].iloc[0]
        axh.set_title(f"{c}: a={rw.a:.3f}±{rw.a_se:.3f}, p={rw.p_null:.1g}", fontsize=8)
        axh.set_xlabel("phase (deg)")
        if i == 0:
            axh.set_ylabel("relative rate (network)")
    axp = fig.add_subplot(2, 4, 5, projection="polar")
    for _, rw in fm.iterrows():
        col = C2 if rw.p_null < 0.05 else GREY
        axp.plot([0, np.deg2rad(rw.phase_deg)], [0, rw.a], color=col, lw=1.2)
    axp.set_title("M2 per family (a, phase)\norange: p<0.05", fontsize=8)
    axc = fig.add_subplot(2, 4, (6, 8))
    cp = comp[(comp.subset == "primary") & (comp.const == "M2")].reset_index(drop=True)
    x = np.arange(len(cp))
    axc.errorbar(x - 0.12, cp.A_before, yerr=cp.A_before_se, fmt="o", color=C1, label="before")
    axc.errorbar(x + 0.12, cp.A_after, yerr=cp.A_after_se, fmt="o", color=C2, label="after")
    axc.set_xticks(x)
    axc.set_xticklabels([n.replace("_", "\n") for n in cp.comparison], fontsize=7)
    for k, rw in cp.iterrows():
        axc.text(k, 0.62, f"z={rw.z:.1f}", ha="center", fontsize=7)
    axc.set_ylabel("pooled M2 amplitude a")
    axc.set_ylim(0, 0.66)
    axc.legend(frameon=False)
    axc.set_title("M2 modulation before vs after each earthquake (bootstrap 1σ)", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIG}/sci_tides_phase_and_quakes.png", dpi=150)
    plt.close(fig)

    # Fig 3: seasonal
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for lab, col in [("primary", C1), ("ccthr1.5", C2), ("all_declustered", C3)]:
        y = mon[f"{lab}_network_ratio_minus1"]
        g = y.groupby(mon.month)
        ax[0].errorbar(mon.month.unique() * 30.44 - 15, g.mean().to_numpy(), yerr=(g.std() / np.sqrt(g.count())).to_numpy(), fmt="o", color=col, ms=4, label=lab)
        s = seas[(seas.subset == lab) & (seas.series == "network")].iloc[0]
        xx = np.arange(366)
        ax[0].plot(xx, s.A * np.cos(2 * np.pi * (xx - s.doy_max) / 365.25), color=col, lw=1.5)
    ax[0].plot(mon.month.unique() * 30.44 - 15, pd.Series(thr_r).groupby(mon.month).mean().to_numpy(), "x--", color=GREY, label="noise proxy (threshold)")
    ax[0].axvline(DVV_FAST_DOY, color="k", ls="--", lw=1)
    ax[0].text(DVV_FAST_DOY + 3, ax[0].get_ylim()[1] * 0.9, "dv/v fastest", fontsize=7)
    ax[0].axhline(0, color=GREY, lw=0.5)
    ax[0].set_xlabel("day of year")
    ax[0].set_ylabel("burst rate / 24-mo running mean - 1")
    ax[0].legend(frameon=False, fontsize=7)
    ax[0].set_title("Annual LFE-rate cycle (network, 2009-2026; Dec24-Jan25 excluded)", fontsize=9)
    sf2 = seas[(seas.subset == "primary") & (seas.series != "network")]
    ax[1].scatter(sf2.doy_max, sf2.A, c=np.where(sf2.p_null < 0.05, C2, GREY), s=25)
    ax[1].axvline(DVV_FAST_DOY, color="k", ls="--", lw=1)
    ax[1].set_xlim(0, 366)
    ax[1].set_xlabel("day of maximum LFE rate")
    ax[1].set_ylabel("annual amplitude (fraction)")
    ax[1].set_title("Per family (orange: per-year-shift null p<0.05)", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIG}/sci_tides_seasonal.png", dpi=150)
    plt.close(fig)
    print("done")


if __name__ == "__main__":
    main()
