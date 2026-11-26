#!/usr/bin/env python3
"""Joint 3-D dv/v validation (revised after code review): injections through the REAL estimators, information
absorbed by nuisances, averaging kernels.

Stage 1 (travel times): the stage-1 joint fit (sci_dvv_3d_tt.Joint, 3-D-traced geometry columns attached) is fitted to
the real data; its final IRLS weights define the linear estimator.  Synthetic delay columns are passed through
J.solve with those weights, i.e. origin, location, channel offsets and free receiver terms are all re-estimated.
NOT tautological: deep truths are generated with one geometry (3-D-traced or 1-D) and estimated with the other
(and with soft / no LVZ shallow structure); laterally varying mid-crust truths (west / east of the family centroid,
LFE-only / earthquake-only paths) test the vertical-contrast nature of the layer term.
Information absorbed: SE of each deep term with free receiver terms vs without receiver terms vs a naive
single-column regression (variance inflation).
Stage 2: synthetic stage-1 output + synthetic coda rows (adversarial kernels: pure surface, l* = 5 km, no LVZ,
4x more near-surface sensitive; for deep truths the coda sees f20_31 x amplitude of the long-l* kernel) through
sci_dvv_3d_gls.Problem with the REAL profiled kappa of the primary configuration.
Averaging kernels: a(cell) = sum_i h_i (-T_i frac_i(cell)) f_q(t_i) on the grid of sci_dvv_3d_G (1-D rays).
Outputs: data/sci/dvv_3d/inj_results.csv, var_inflation.csv, ak_summary.csv; ak_maps.npz on /data/wsd02.
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

ROOT = '/home/seismic/LFE'
OUT = f'{ROOT}/data/sci/dvv_3d'
sys.path.insert(0, f'{ROOT}/scripts')
import sci_dvv_3d_tt as TT  # noqa: E402
import sci_dvv_3d_gls as GL  # noqa: E402
import sci_dvv_3d_G as GG  # noqa: E402
import sci_dvv_3d_kernels as KK  # noqa: E402
import sci_dvv_source_region as SR_  # noqa: E402

NPROC = int(os.environ.get('NPROC', '6'))
RUNS = [('clean_lay_3d', ['lay3'], TT.CLEAN), ('clean_lay_1d', ['lay'], TT.CLEAN),
        ('clean_R_lay_3d', ['R1_10_3', 'R10_20_3', 'lay3'], TT.CLEAN), ('clean_R_lay_1d', ['R1_10', 'R10_20', 'lay'], TT.CLEAN),
        ('clean_box_3d', ['box3'], TT.CLEAN), ('clean_sph_3d', ['sph3'], TT.CLEAN), ('clean_slab_3d', ['slab3'], TT.CLEAN),
        ('clean_sph_1d', ['sph'], TT.CLEAN), ('clean_box_1d', ['box'], TT.CLEAN),
        ('joint_lay_3d', ['lay3'], 'all'), ('joint_lay_1d', ['lay'], 'all')]
NNOISE = 20


def build_joint(vols, sub='all'):
    D = TT.subset(TT.attach(TT.load_rows(), 'ref'), sub)
    J = TT.Joint(D, vols)
    xg, cov, res, chi, gain, w = J.fit(want_gain=True)
    J.w_main = w
    return J, xg, cov, gain


def geo_cols(D, lvz):
    G = pd.read_csv(f'{OUT}/tt_geometry.csv')
    g = G[G.lvz == lvz].set_index(['fam', 'sta', 'phase'])
    return g.reindex(pd.MultiIndex.from_arrays([D.fam, D.sta, D.phase])).fillna(0.0)


def truths(J):
    D = J.D
    F = (D.t.values >= SR_.T_Q['m7']).astype(float)
    isL = (D.kind == 'L').values
    out = {}
    for nm, c1, c3, amp in [('lay', 'lay', 'lay3', -0.1), ('box', 'box', 'box3', -0.1), ('sph', 'sph', 'sph3', -0.3),
                            ('slab', 'slab', 'slab3', -0.3)]:
        out[f'deep_{nm}|1d'] = (-D[c1].values * amp / 100 * F, dict(kind='deep', vol=nm, amp=amp, geom='1d'))
        out[f'deep_{nm}|3d'] = (-D[c3].values * amp / 100 * F, dict(kind='deep', vol=nm, amp=amp, geom='3d'))
    for g_, sfx in (('1d', ''), ('3d', '_3')):
        for nm, col in (('mid', 'R10_20'), ('up', 'R1_10')):
            K = D[col + sfx].values
            KW = D[col + sfx + 'W'].values            # path time WEST of the family centroid (ray geometry)
            out[f'{nm}_uniform|{g_}'] = (-K * (-0.1) / 100 * F, dict(kind='mid', vol=nm, amp=-0.1, geom=g_))
            out[f'{nm}_west|{g_}'] = (-KW * (-0.1) / 100 * F, dict(kind='mid', vol=nm + '_west', amp=-0.1, geom=g_))
            out[f'{nm}_LFEpaths|{g_}'] = (-K * isL * (-0.1) / 100 * F, dict(kind='mid', vol=nm + '_L', amp=-0.1, geom=g_))
            out[f'{nm}_EQpaths|{g_}'] = (-K * ~isL * (-0.1) / 100 * F, dict(kind='mid', vol=nm + '_E', amp=-0.1, geom=g_))
    for lvz in ('ref', 'soft', 'none'):
        g = geo_cols(D, lvz)
        sh = g['T0_0.1'].values + g['T0.1_1'].values
        out[f'shallow_all|{lvz}'] = (-sh * (-0.5) / 100 * F, dict(kind='shallow', amp=-0.5, geom=lvz))
        out[f'shallow_top100|{lvz}'] = (-g['T0_0.1'].values * (-2.0) / 100 * F, dict(kind='shallow100', amp=-2.0, geom=lvz))
    rng = np.random.default_rng(7)
    stas = sorted(D.sta.unique())
    gs = geo_cols(D, 'soft')
    for k in range(5):
        a = dict(zip(stas, rng.normal(0, 0.5, len(stas))))
        sh = gs['T0_0.1'].values + gs['T0.1_1'].values
        out[f'shallow_rand{k}|soft'] = (-sh * D.sta.map(a).values / 100 * F, dict(kind='shallow_rand', amp=0.5, geom='soft', stamp=a))
        c = dict(zip(stas, rng.normal(0, 0.003, len(stas))))
        out[f'clock{k}|-'] = (D.sta.map(c).values * F, dict(kind='clock', amp=3.0, geom='-'))
    return out


def coda_rows(pr, q, meta, variant):
    """synthetic coda rows (same rows / SEs as the real data) under the truth `meta` with kernel `variant`."""
    kf = pr.kf
    kind = meta['kind']
    res = {}

    def sh_vec(sta):
        if kind == 'shallow':
            return np.array([-0.5, -0.5, 0.0])
        if kind == 'shallow100':
            return np.array([-2.0, 0.0, 0.0])
        if kind == 'shallow_rand':
            a = meta['stamp'].get(sta, 0.0)
            return np.array([a, a, 0.0])
        return np.zeros(3)

    def frac(ds, sta, band):
        var = variant
        if ds == 'LFE' and variant not in ('x4',):
            var = 'default'
        if kind in ('deep', 'mid'):
            var = 'l50' if ds == 'SC' else 'default'
        k = (ds, sta, band, 'default' if var == 'x4' else var)
        if k not in kf.index:
            return None
        f = kf.loc[k].astype(float)
        if var == 'x4':
            v = np.array([f.f0, f.f1, f.f2, f.fdeep]) * np.array([4, 4, 1, 1])
            v = v / v.sum()
            f = f.copy()
            f['f0'], f['f1'], f['f2'] = v[0], v[1], v[2]
        return f
    for name, df, ds in (('SC', pr.sc, 'SC'), ('LFE', pr.lfe, 'LFE'), ('REP', pr.rep, 'LFE')):
        g = df[df.quantity == q].copy()
        vals = []
        for r in g.itertuples():
            band = r.band if ds == 'SC' else '2-8Hz'
            f = frac(ds, r.sta, band)
            if f is None:
                vals.append(0.0)
                continue
            if kind == 'deep':
                v = f.f20_31 * meta['amp']
            elif kind == 'mid':
                v = (f.f10_20 if meta['vol'].startswith('mid') else f.f1_10) * meta['amp']
            else:
                v = np.array([f.f0, f.f1, f.f2]) @ sh_vec(r.sta)
            vals.append(v)
        g['est'] = vals
        res[name] = g
    return res


def _inj_job(args):
    run, vols, sub = args
    os.environ['OMP_NUM_THREADS'] = '2'
    J, xg, cov, gain = build_joint(vols, sub)
    Tr = truths(J)
    names = list(Tr)
    Y = np.column_stack([Tr[n][0] for n in names])
    # noise: NNOISE Gaussian realisations with the row SEs x sqrt(chi2r) (no signal) -> spread of the estimator
    rngn = np.random.default_rng(3)
    _, _, res0, chi0, _, _ = J.fit(irls=0)
    En = rngn.normal(0, 1, (len(J.D), NNOISE)) * (J.D.se.values * np.sqrt(max(chi0, 1.0)))[:, None]
    Xg, *_ = J.solve(np.column_stack([Y, En]), J.w_main)
    Xn = Xg[:, len(names):]
    Xg = Xg[:, :len(names)]
    noise_sd = {v: float(np.std([TT.qx('m7') @ J.gam(Xn[:, k])[iv] for k in range(NNOISE)]) * 100) for iv, v in enumerate(vols)}
    rows = []
    for iv, v in enumerate(vols):
        rows.append(dict(run=run, truth='noise_only', geom='-', kind='noise', true_amp=0.0, stage='TT', coda='-', vol=v,
                         rec_pct=0.0, frac=np.nan, noise_sd_pct=noise_sd[v]))
    gr = pd.read_csv(f'{OUT}/gls_results.csv')
    gk = gr[(gr.run == run) & (gr.cfg == 'primary')]
    import ast
    kap = ast.literal_eval(gk.kappa.iloc[0]) if len(gk) else (1.0, 1.0, 1.0)
    pr = GL.Problem(run)
    for k, n in enumerate(names):
        meta = Tr[n][1]
        gam = J.gam(Xg[:, k])
        base = dict(run=run, truth=n.split('|')[0], geom=meta['geom'], kind=meta['kind'], true_amp=meta['amp'])
        for iv, v in enumerate(vols):
            rec = float(TT.qx('m7') @ gam[iv]) * 100
            rows.append(dict(**base, stage='TT', coda='-', vol=v, rec_pct=rec, frac=rec / meta['amp'], noise_sd_pct=noise_sd[v]))
        for cv in (['default', 'surf', 'l5', 'lvz_none', 'x4'] if meta['kind'] not in ('deep', 'mid') else ['l50']):
            ov = coda_rows(pr, 'm7', meta, cv)
            sol = pr.solve('m7', kap, tt_override=Xg[:, k], coda_override=ov)
            ix = sol['info']['ix']
            for v in vols:
                i = ix[f'g|{v}']
                rows.append(dict(**base, stage='GLS', coda=cv, vol=v, rec_pct=float(sol['x'][i]),
                                 sd_pct=float(np.sqrt(sol['C'][i, i])), frac=float(sol['x'][i]) / meta['amp']))
    # variance inflation: deep-term SE with free receiver terms / without receiver terms / naive one-column fit
    vi = []
    Jn = TT.Joint(J.D, vols, beta=False)
    _, covn, *_ = Jn.fit()
    w = J.w_main
    F = SR_.tcols(J.D.t.values)
    for iv, v in enumerate(vols):
        se_free = TT._se(cov, J.nb, len(vols), iv, TT.qx('m7')) * 100
        se_nob = TT._se(covn, Jn.nb, len(vols), iv, TT.qx('m7')) * 100
        x = -J.D[v].values * F[:, 4]
        xd = x - np.average(x, weights=w)
        se_naive = 100 / np.sqrt(np.sum(w * xd ** 2))
        vi.append(dict(run=run, vol=v, se_free_receiver=se_free, se_no_receiver=se_nob, se_naive=se_naive,
                       inflation_free_vs_naive=se_free / se_naive))
    print(run, 'injections done', flush=True)
    return rows, vi


def stage_inject():
    runs = [r for r in RUNS if os.path.exists(f'{TT.BIG}/tt_{r[0]}.npz')]
    with Pool(NPROC) as pool:
        res = pool.map(_inj_job, runs, chunksize=1)
    R = pd.DataFrame([r for a, b in res for r in a])
    V = pd.DataFrame([r for a, b in res for r in b])
    R.to_csv(f'{OUT}/inj_results.csv', index=False, float_format='%.5g')
    V.to_csv(f'{OUT}/var_inflation.csv', index=False, float_format='%.5g')
    with pd.option_context('display.width', 250):
        print(V.round(4).to_string())
        print(R[R.kind.isin(['deep', 'mid'])].pivot_table(index=['run', 'stage', 'truth', 'geom'], columns='vol', values='rec_pct').round(4).to_string())
        print(R[~R.kind.isin(['deep', 'mid'])].groupby(['run', 'stage', 'kind', 'geom', 'coda', 'vol']).rec_pct.agg(lambda x: np.sqrt(np.mean(x ** 2))).round(4).unstack('vol').to_string())


_VMC = {}


def _vm(sta, lat, lon):
    if sta not in _VMC:
        _VMC[sta] = KK.station_vmodel(lat, lon, 'ref')
    return _VMC[sta]


def _ak_job(args):
    """averaging kernels of the saved stage-1 estimator (gain = linear map data -> gamma, final IRLS weights) in the
    run's OWN geometry: 3-D runs use the 3-D-traced + 1-D-continued path rows (grid_rows_3d.npz, seconds per cell);
    1-D runs trace 1-D rays (time fraction x T)."""
    run, vols, sub = args
    os.environ['OMP_NUM_THREADS'] = '2'
    # project-internal npz written by sci_dvv_3d_tt.run (string columns stored as object arrays; trusted)
    Z = np.load(f'{TT.BIG}/tt_{run}.npz', allow_pickle=True)
    gain = Z['gain'].astype(float)
    fam, sta, ph, t = Z['fam'].astype(str), Z['sta'].astype(str), Z['phase'].astype(str), Z['t']
    keys = sorted(set(zip(fam, sta, ph)))
    is3 = run.endswith('_3d')
    Fm = np.zeros((len(keys), GG.GR.npar), np.float32)                   # seconds per cell for each path
    if is3:
        R3 = np.load(f'{TT.BIG}/grid_rows_3d.npz', allow_pickle=False)
        idx = {k: i for i, k in enumerate(R3['keys'])}
        rows3 = R3['rows']
        for k, key in enumerate(keys):
            j = idx.get('|'.join(key))
            if j is not None:
                Fm[k] = rows3[j]
    else:
        S = TT.sources()
        ST = TT.stations_all()
        for k, (f_, s_, p_) in enumerate(keys):
            st = ST.loc[s_]
            src = S.loc[f_]
            xr, yr = GG.xy(st.lat, st.lon)
            X = float(np.hypot(xr - src.x, yr - src.y))
            zc, h, dt, v = TT.ray_phase(_vm(s_, st.lat, st.lon), src.z, st.z0, X, p_, n=300)
            ux, uy = ((xr - src.x) / X, (yr - src.y) / X) if X > 0 else (1.0, 0.0)
            Fm[k] = GG.points_to_row(src.x + ux * h, src.y + uy * h, zc, dt)
    gidx = pd.Series(range(len(keys)), index=pd.MultiIndex.from_tuples(keys))
    gi = gidx.reindex(pd.MultiIndex.from_arrays([fam, sta, ph])).values
    PI = GG.GR.param_info()
    inbox = ((PI.layer.isin([8, 9])) & ((PI.x - TT.BOXC[0]).abs() <= 10) & ((PI.y - TT.BOXC[1]).abs() <= 10)).values
    summ, maps = [], {}
    nvt = TT.NJX
    for q, col in (('m7', 4), ('D', 8), ('pet', 2), ('fer', 3)):
        Qv = TT.qx(q)
        f = SR_.tcols(t)[:, col]
        if q == 'D':
            f = f / SR_.dvec()[8]
        for iv, v in enumerate(vols):
            h = Qv @ gain[iv * nvt:(iv + 1) * nvt]
            a_g = np.bincount(gi, weights=-h * f, minlength=len(keys))
            A = a_g @ Fm
            maps[f'{run}|{v}|{q}'] = A.astype(np.float32)
            lay = [A[GG.GR.off[l]:GG.GR.off[l + 1]].sum() + A[GG.GR.next0 + l] for l in range(GG.NL)]
            summ.append(dict(run=run, vol=v, quantity=q, ak_total=float(A.sum()), in_box2031=float(A[inbox].sum()),
                             abs_sum=float(np.abs(A).sum()), **{f'L{GG.LAYNAME[l]}': lay[l] for l in range(GG.NL)}))
    print(run, 'ak done', flush=True)
    return summ, maps


def stage_ak():
    runs = [r for r in RUNS if os.path.exists(f'{TT.BIG}/tt_{r[0]}.npz')]
    with Pool(NPROC) as pool:
        res = pool.map(_ak_job, runs, chunksize=1)
    T = pd.DataFrame([r for a, b in res for r in a])
    maps = {k: v for a, b in res for k, v in b.items()}
    T.to_csv(f'{OUT}/ak_summary.csv', index=False, float_format='%.4g')
    np.savez_compressed(f'{TT.BIG}/ak_maps.npz', **maps)
    with pd.option_context('display.width', 250):
        print(T[T.quantity == 'm7'].round(3).to_string())


if __name__ == '__main__':
    for st in sys.argv[1:]:
        {'inject': stage_inject, 'ak': stage_ak}[st]()
