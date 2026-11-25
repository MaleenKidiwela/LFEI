#!/usr/bin/env python3
"""Joint 3-D dv/v validation: injections through the REAL estimator, averaging kernels, absorption.

TT stage: the stage-1 joint fit (sci_dvv_3d_tt.Joint) is refitted on the real data to obtain its final IRLS
weights; synthetic delay columns are then passed through the same linear estimator (origin, location, channel
offsets, free receiver terms all re-estimated), i.e. exactly what the real data experience.  Second stage: the
synthetic stage-1 output and synthetic coda rows go through sci_dvv_3d_gls.Problem (default kernels).

Truths (M7 step pattern, t >= M7):
  deep:    box (-0.1 % at 20-31 km within 10 km of the centroid), lay (-0.1 % 20-31 km everywhere), sph (-0.3 %
           within 10 km of the families), mid (-0.1 % at 10-20 km), up (-0.1 % at 1-10 km)
           generated with the reference rays AND with adversarial ray models (soft / no LVZ) and, for 'lay',
           the straight-ray 3-D take-off geometry of sci_eq_above_diff (geom_layer K).
  shallow: -0.5 % in 0-1 km beneath every station; random N(0, 0.5 %) per station; -2 % in 0-0.1 km;
           clock steps N(0, 3 ms) per station (S = P).
  coda rows for shallow truths are generated with ADVERSARIAL kernels (pure surface, l* = 5 km, no LVZ, and
  4x more near-surface sensitive) and inverted with the default kernels; for deep truths the coda rows get the
  deep part of the long-l* (50 km) body kernel (coda that does see depth).
Averaging kernels: a_q(cell) = sum_i h_i (-T_i frac_i(cell)) f_q(t_i) on the 3-D grid of sci_dvv_3d_G (rays of the
reference model), for each deep volume of the primary runs; summarised per layer and inside/outside the target.
Outputs data/sci/dvv_3d/inj_results.csv, ak_summary.csv, ak_maps.npz.
"""
import os
import sys

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

NJ = TT.NJ
rng = np.random.default_rng(7)


def build_joint(run_vols):
    G = pd.read_csv(f'{OUT}/tt_geometry.csv')
    D = TT.attach(TT.load_rows(), G, 'ref')
    J = TT.Joint(D, run_vols)
    xg, cov, res, chi, gain = J.fit(want_gain=True)
    return J, G, gain


def truths(J, G):
    """synthetic delay columns (s) for M7-step truths; returns dict name -> (Y, meta)."""
    D = J.D
    F = (D.t.values >= SR_.T_Q['m7']).astype(float)
    out = {}
    keys = ['fam', 'sta', 'phase']
    geo = {}
    for lvz in ('ref', 'soft', 'none'):
        g = G[G.lvz == lvz].set_index(keys)
        geo[lvz] = g.reindex(pd.MultiIndex.from_arrays([D.fam, D.sta, D.phase])).fillna(0.0)
    for lvz in ('ref', 'soft', 'none'):
        g = geo[lvz]
        for nm, col, amp in [('box', 'box', -0.1), ('lay', 'lay', -0.1), ('sph', 'sph', -0.3), ('mid', 'R10_20', -0.1),
                             ('up', 'R1_10', -0.1)]:
            out[f'deep_{nm}|{lvz}'] = (-g[col].values * amp / 100 * F, dict(kind='deep', vol=nm, amp=amp, geom=lvz))
        sh = g['T0_0.1'].values + g['T0.1_1'].values
        out[f'shallow_all|{lvz}'] = (-sh * (-0.5) / 100 * F, dict(kind='shallow', amp=-0.5, geom=lvz))
        out[f'shallow_top100|{lvz}'] = (-g['T0_0.1'].values * (-2.0) / 100 * F, dict(kind='shallow100', amp=-2.0, geom=lvz))
    stas = sorted(D.sta.unique())
    for k in range(5):
        a = dict(zip(stas, rng.normal(0, 0.5, len(stas))))
        sh = geo['soft']['T0_0.1'].values + geo['soft']['T0.1_1'].values
        out[f'shallow_rand{k}|soft'] = (-sh * D.sta.map(a).values / 100 * F, dict(kind='shallow_rand', amp=0.5, geom='soft', stamp=a))
        c = dict(zip(stas, rng.normal(0, 0.003, len(stas))))
        out[f'clock{k}|-'] = (D.sta.map(c).values * F, dict(kind='clock', amp=3.0, geom='-'))
    # eq_above straight-ray layer geometry (LFE rows only; 22 km .. family depth)
    out['deep_layED|ED'] = (-D['K'].values * (-0.1) / 100 * F, dict(kind='deep', vol='layED', amp=-0.1, geom='ED'))
    return out


def coda_rows(pr, q, truth_meta, variant):
    """synthetic coda rows for the GLS (same rows/SDs as the real data)."""
    kf = pr.kf.set_index(['dataset', 'sta', 'band', 'variant'])
    res = {}
    kind = truth_meta['kind']
    sc = pr.sc[(pr.sc.quantity == q) & np.isfinite(pr.sc.se) & (pr.sc.se > 0)].copy()
    lfe = pr.lfe[pr.lfe.quantity == q].copy()
    rep = pr.rep[np.isfinite(pr.rep.m7_pct)].copy()

    def sh_vec(sta):
        if kind == 'shallow':
            return np.array([-0.5, -0.5, 0.0])
        if kind == 'shallow100':
            return np.array([-2.0, 0.0, 0.0])
        if kind == 'shallow_rand':
            a = truth_meta['stamp'].get(sta, 0.0)
            return np.array([a, a, 0.0])
        return np.zeros(3)

    def frac(ds, sta, band, var):
        k = (ds, sta, band, var)
        if var == 'x4':
            k = (ds, sta, band, 'default')
        if k not in kf.index:
            return None
        f = kf.loc[k, ['f0', 'f1', 'f2', 'fdeep']].values.astype(float)
        if var == 'x4':
            f = f * np.array([4, 4, 1, 1])
            f = f / f.sum()
        return f
    for name, g, ds, bandcol in (('SC', sc, 'SC', 'band'), ('LFE', lfe, 'LFE', None), ('REP', rep, 'LFE', None)):
        vals = []
        for r in g.itertuples():
            band = getattr(r, 'band') if bandcol else '2-8Hz'
            var = variant if ds == 'SC' or variant == 'x4' else ('default' if variant != 'x4' else 'x4')
            if ds == 'LFE' and variant not in ('x4',):
                var = 'default'
            f = frac(ds, r.sta, band, var if not (kind == 'deep' and ds == 'SC') else 'l50')
            if f is None:
                vals.append(0.0)
                continue
            v = f[:3] @ sh_vec(r.sta)
            if kind == 'deep':
                # deep truth seen by the coda through the part of its kernel below 3 km (scaled to the 20-31 km share)
                v = f[3] * truth_meta['amp'] * 0.15
            vals.append(v)
        g = g.copy()
        if name == 'REP':
            g['m7_pct'] = vals
        else:
            g['est'] = vals
        res[name] = g
    return res


def stage_inject():
    rows = []
    for run, vols in [('joint_R_lay', ['R1_10', 'R10_20', 'lay']), ('joint_lay', ['lay']), ('joint_box', ['box']),
                      ('joint_sph', ['sph']), ('joint_slab', ['slab']), ('joint_R_sph', ['R1_10', 'R10_20', 'sph'])]:
        J, G, gain = build_joint(vols)
        Tr = truths(J, G)
        names = list(Tr)
        Y = np.column_stack([Tr[n][0] for n in names])
        Xg, *_ = J.solve(Y, J.w_final)
        Q = TT.qvecs()['m7']
        pr = GL.Problem(run) if os.path.exists(f'{TT.BIG}/tt_{run}.npz') else None
        for k, n in enumerate(names):
            meta = Tr[n][1]
            gam = J.gam(Xg[:, k])
            base = dict(run=run, truth=n.split('|')[0], geom=meta['geom'], kind=meta['kind'], true_amp=meta['amp'])
            for iv, v in enumerate(vols):
                rec = float(Q @ gam[iv]) * 100
                rows.append(dict(**base, stage='TT', coda='-', vol=v, rec_pct=rec,
                                 frac=rec / meta['amp'] if meta['kind'] == 'deep' and meta.get('vol') in (v, 'layED' if v == 'lay' else None) else np.nan))
            if pr is None:
                continue
            for cv in (['default', 'surf', 'l5', 'lvz_none', 'x4'] if meta['kind'] != 'deep' else ['l50']):
                ov = coda_rows(pr, 'm7', meta, cv)
                sol = pr.solve('m7', (1.0, 1.0, 1.0), tt_override=Xg[:, k], coda_override=ov)
                for iv, v in enumerate(vols):
                    i = sol['info']['o_g'] + iv
                    rec = float(sol['x'][i])
                    rows.append(dict(**base, stage='GLS', coda=cv, vol=v, rec_pct=rec, sd_pct=float(np.sqrt(sol['C'][i, i])),
                                     frac=rec / meta['amp'] if meta['kind'] == 'deep' and meta.get('vol') in (v, 'layED' if v == 'lay' else None) else np.nan))
        print(run, 'done', flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(f'{OUT}/inj_results.csv', index=False, float_format='%.5g')
    with pd.option_context('display.width', 250):
        print(R[R.kind.isin(['deep'])].groupby(['run', 'stage', 'truth', 'geom', 'vol']).rec_pct.mean().round(4).unstack('vol').to_string())
        print(R[~R.kind.isin(['deep'])].groupby(['run', 'stage', 'kind', 'geom', 'coda', 'vol']).rec_pct.agg(lambda x: np.sqrt(np.mean(x ** 2))).round(4).unstack('vol').to_string())


def ray_frac_rows(G, S, ST, keys):
    """3-D grid rows (time fractions) for straight 1-D rays (reference model) per (fam, sta, phase)."""
    rows = {}
    for fam, sta, ph in keys:
        if sta not in ST.index or fam not in S.index:
            continue
        st = ST.loc[sta]
        vm = TT_vm(sta, st.lat, st.lon)
        xr, yr = GG.xy(st.lat, st.lon)
        src = S.loc[fam]
        X = float(np.hypot(xr - src.x, yr - src.y))
        zc, h, dt, v = TT.ray_phase(vm, src.z, st.z0, X, ph, n=300)
        ux, uy = ((xr - src.x) / X, (yr - src.y) / X) if X > 0 else (1.0, 0.0)
        rows[(fam, sta, ph)] = (GG.points_to_row(src.x + ux * h, src.y + uy * h, zc, dt / dt.sum()).astype(np.float32), dt.sum())
    return rows


_VMC = {}


def TT_vm(sta, lat, lon):
    if sta not in _VMC:
        _VMC[sta] = KK.station_vmodel(lat, lon, 'ref')
    return _VMC[sta]


def stage_ak():
    S = TT.sources()
    ST = TT.stations_all()
    summ = []
    maps = {}
    PI = GG.GR.param_info()
    for run, vols in [('joint_R_lay', ['R1_10', 'R10_20', 'lay']), ('joint_box', ['box']), ('joint_sph', ['sph']),
                      ('joint_slab', ['slab']), ('joint_lay', ['lay'])]:
        J, G, gain = build_joint(vols)
        D = J.D
        keys = sorted(set(zip(D.fam, D.sta, D.phase)))
        FR = ray_frac_rows(G, S, ST, keys)
        gidx = pd.Series(range(len(keys)), index=pd.MultiIndex.from_tuples(keys))
        gi = gidx.reindex(pd.MultiIndex.from_arrays([D.fam, D.sta, D.phase])).values
        Fm = np.zeros((len(keys), GG.GR.npar), np.float32)
        Tg = np.zeros(len(keys))
        for k, key in enumerate(keys):
            if key in FR:
                Fm[k], Tg[k] = FR[key]
        for q, col in (('m7', 4), ('D', 8), ('pet', 2), ('fer', 3)):
            Qv = TT.qvecs()[q]
            f = SR_.tcols(D.t.values)[:, col]
            if q == 'D':
                f = f / SR_.dvec()[8]
            for iv, v in enumerate(vols):
                h = Qv @ gain[iv * NJ:(iv + 1) * NJ]           # (n,)
                a_rows = h * (-D['T'].values) * f
                a_g = np.bincount(gi, weights=a_rows, minlength=len(keys))
                A = a_g @ Fm                                   # per-cell response to a unit dv/v (fraction) in the cell
                maps[f'{run}|{v}|{q}'] = A.astype(np.float32)
                tot = A.sum()
                lay = np.array([A[GG.GR.off[l]:GG.GR.off[l + 1]].sum() + A[GG.GR.next0 + l] for l in range(GG.NL)])
                rec = dict(run=run, vol=v, quantity=q, ak_total=tot, **{f'L{GG.LAYNAME[l]}': lay[l] for l in range(GG.NL)})
                # inside the 20-31 km 20-km box
                m = (PI.layer.isin([8, 9])) & (PI.x.abs() <= 10) & (PI.y.abs() <= 10)
                rec['in_box2031'] = float(A[m.values].sum())
                rec['abs_sum'] = float(np.abs(A).sum())
                summ.append(rec)
        print(run, 'ak done', flush=True)
    T = pd.DataFrame(summ)
    T.to_csv(f'{OUT}/ak_summary.csv', index=False, float_format='%.4g')
    np.savez_compressed(f'{TT.BIG}/ak_maps.npz', **maps)
    with pd.option_context('display.width', 250):
        print(T[T.quantity == 'm7'].round(3).to_string())


if __name__ == '__main__':
    for st in sys.argv[1:]:
        {'inject': stage_inject, 'ak': stage_ak}[st]()
