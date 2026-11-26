#!/usr/bin/env python3
"""TT-vs-LP tension on the regional terms (requested by review R3).  LP-only cannot constrain the three regional
terms separately (LP-only stage-2 is degenerate), so the tension is measured in data space: the LP band x
distance-bin observations (0.1-0.3 Hz; 0.05-0.1 Hz as variant) are predicted from the stage-1 travel-time
estimates of R1_10, R10_20 and the 20-31 km layer (with their honest covariance, shallow terms set to 0), and
chi2 = r^T (C_LP + A C_TT A^T)^-1 r is compared with chi2 of the zero model; dchi2 = chi2(TT) - chi2(0) > 0 means
the LP data disfavour the TT regional values.  Output data/sci/dvv_3d/tt_lp_tension.csv."""
import json
import os

import numpy as np
import pandas as pd

OUT = '/home/seismic/LFE/data/sci/dvv_3d'
BIG = '/data/wsd02/maleen_data/dvv_3d'
R = pd.read_csv(os.environ.get('TT_RESULTS', f'{OUT}/tt_results.csv'))
LP = pd.read_csv(f'{OUT}/coda_obs_lp.csv')
LPK = json.load(open(f'{OUT}/lp_depth_kernels.json'))
rows = []
for run in ['clean_R_lay_3d', 'clean_R_lay_1d', 'joint_R_lay_3d', 'joint_R_lay_1d']:
    g = R[R.run == run]
    if g.empty:
        continue
    vols = [v for v in g.vol.unique()]
    for q in ['m7', 'pet', 'fer', 'D']:
        est = np.array([g[(g.vol == v) & (g.quantity == q)].est_pct.iloc[0] for v in vols])
        se = np.array([g[(g.vol == v) & (g.quantity == q)].se.iloc[0] for v in vols])
        for bands, lab in ((('0.1-0.2', '0.2-0.3'), 'LP_0.1-0.3'), (('0.05-0.1', '0.1-0.2', '0.2-0.3'), 'LP_all')):
            d = LP[(LP.quantity == q) & LP.band.isin(bands)]
            A = np.array([[sum(LPK[b]['layer_frac'][3:6]), sum(LPK[b]['layer_frac'][6:8]), sum(LPK[b]['layer_frac'][8:10])]
                          for b in d.band])
            C = np.diag(d.se.values ** 2)
            Ct = A @ np.diag(se ** 2) @ A.T
            r1 = d.est.values - A @ est
            c1 = float(r1 @ np.linalg.solve(C + Ct, r1))
            c0 = float(d.est.values @ np.linalg.solve(C, d.est.values))
            rows.append(dict(run=run, quantity=q, lp=lab, n=len(d), chi2_TTpred=c1, chi2_zero=c0, dchi2_tension=c1 - c0,
                             tt_R1_10=est[0], tt_R10_20=est[1], tt_lay=est[2]))
T = pd.DataFrame(rows)
T.to_csv(f'{OUT}/tt_lp_tension.csv', index=False, float_format='%.4g')
print(T.round(3).to_string())
