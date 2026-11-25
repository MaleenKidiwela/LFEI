#!/usr/bin/env python
"""Compare our re-detections (sci_ducellier_mf.py output) with the published Ducellier catalog
for the same days: recall / precision per template, raw (8xMAD) and filtered (cc*nchannel >= thr).
Usage: python -I sci_ducellier_validate.py <tag> [tol_s]"""
import sys, glob, os
import numpy as np, pandas as pd
DUC = "/data/wsd02/maleen_data/ducellier"
tag = sys.argv[1]; tol = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
raw = pd.read_parquet(f"{DUC}/catalog_csv/ducellier_2004_2011_all_raw.parquet")
fs = pd.read_csv("/home/seismic/LFE/data/sci/ducellier/families_summary.csv", dtype={"template": str})
raw = raw.merge(fs[["template", "thr_b"]], on="template")
raw = raw[raw.thr_b > 0]
raw["t"] = pd.to_datetime(raw.time)
rows = []
for f in sorted(glob.glob(f"{DUC}/mf/{tag}/det_*.csv")):
    day = pd.Timestamp(os.path.basename(f)[4:14])
    try:
        d = pd.read_csv(f, dtype={"template": str}, parse_dates=["time"])
    except pd.errors.EmptyDataError:
        continue
    c = raw[(raw.t >= day) & (raw.t < day + pd.Timedelta(days=1))]
    for tpl in sorted(set(c.template) | set(d.template)):
        a = c[c.template == tpl]; b = d[d.template == tpl] if len(d) else d
        af = a[a.cc * a.nchannel >= a.thr_b]
        bf = b[b.cc * b.nchannel >= b.thr_perm] if len(b) else b
        def match(x, y):
            if len(x) == 0 or len(y) == 0:
                return 0, np.array([])
            ty = np.sort(y.time.values.astype("datetime64[ns]").astype(np.int64)) / 1e9
            tx = x.time.values.astype("datetime64[ns]").astype(np.int64) / 1e9 if "time" in x else x.t.values.astype("datetime64[ns]").astype(np.int64) / 1e9
            j = np.clip(np.searchsorted(ty, tx), 1, len(ty) - 1) if len(ty) > 1 else np.zeros(len(tx), int)
            dd = np.minimum(np.abs(tx - ty[j - 1 if len(ty) > 1 else 0]), np.abs(ty[j] - tx))
            return int((dd <= tol).sum()), dd
        a2 = a.assign(time=a.t); af2 = af.assign(time=af.t)
        m_raw, _ = match(a2, b)
        m_f, ddf = match(af2, bf)
        p_f, _ = match(bf, af2) if len(bf) else (0, None)
        rows.append(dict(day=day.date(), template=tpl, cat_raw=len(a), ours_raw=len(b), recall_raw=m_raw / len(a) if len(a) else np.nan,
                         cat_filt=len(af), ours_filt=len(bf), recall_filt=m_f / len(af) if len(af) else np.nan,
                         precision_filt=p_f / len(bf) if len(bf) else np.nan,
                         nch_cat=a.nchannel.median() if len(a) else np.nan, nch_ours=b.nchannel.median() if len(b) else np.nan,
                         cc_cat=a.cc.median() if len(a) else np.nan, cc_ours=b.cc.median() if len(b) else np.nan))
r = pd.DataFrame(rows)
r.to_csv(f"/home/seismic/LFE/data/sci/ducellier/validate_{tag}.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print(r.round(3).to_string())
tot = r.groupby("day")[["cat_raw", "ours_raw", "cat_filt", "ours_filt"]].sum()
print(tot)
for day, g in r.groupby("day"):
    w = g.cat_filt.sum()
    print(day, "pooled recall_filt", (g.recall_filt * g.cat_filt).sum() / max(w, 1),
          "pooled precision_filt", (g.precision_filt.fillna(0) * g.ours_filt).sum() / max(g.ours_filt.sum(), 1),
          "pooled recall_raw", (g.recall_raw * g.cat_raw).sum() / max(g.cat_raw.sum(), 1))
