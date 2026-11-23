#!/usr/bin/env python3
"""Fault geometry from LFE families (reproduces the 2026-11-22 19:00 result).
(a) plane z = a + b*E + c*N through the 27 family centroids (ours 3D and Shelly's), bootstrap over families;
(b) pooled within-patch A x B covariance over resolved families (data/patches/patch_tensors.csv):
    principal axes, plane normal, bootstrap over families, shuffled-pairing null.
Output data/patches/fault_geometry.json"""
import json
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
lat0, lon0 = 40.06, -123.625
out = {}


def plane(X, Y, Z):
    A = np.c_[np.ones(len(X)), X, Y]
    m, *_ = np.linalg.lstsq(A, Z, rcond=None)
    gx, gy = m[1], m[2]
    return float(np.degrees(np.arctan(np.hypot(gx, gy)))), float(np.degrees(np.arctan2(gx, gy)) % 360), float(np.std(Z - A @ m))


sh = pd.read_csv('data/families/shelly_family_locations.csv').set_index('idnum')
for lab, get in (('ours_3d', lambda F: json.load(open(f'data/locations/fam{F}_3d.json'))),
                 ('shelly', lambda F: dict(lat=sh.loc[F].lat, lon=sh.loc[F].lon, depth_km=sh.loc[F].depth_km))):
    L = [get(F) for F in range(1, 28)]
    X = np.array([(l['lon'] - lon0) * 111.19 * np.cos(np.radians(lat0)) for l in L])
    Y = np.array([(l['lat'] - lat0) * 111.19 for l in L])
    Z = np.array([l['depth_km'] for l in L])
    dip, dd, res = plane(X, Y, Z)
    bs = np.array([plane(X[i], Y[i], Z[i])[:2] for i in [rng.integers(0, 27, 27) for _ in range(500)]])
    out[f'centroid_plane_{lab}'] = dict(dip=dip, dip_dir=dd, resid_km=res, dip_68=[float(np.percentile(bs[:, 0], 16)), float(np.percentile(bs[:, 0], 84))])

P = pd.read_csv('data/patches/patch_tensors.csv')
res = P[P.resolved == True].family.astype(int).tolist()  # noqa: E712
data = {}
for F in res:
    A = pd.read_csv(f'data/relocations/fam{F}_events_3d_halfA.csv')
    B = pd.read_csv(f'data/relocations/fam{F}_events_3d_halfB.csv')
    m = A.merge(B, on='ev', suffixes=('_a', '_b'))
    m = m[(m.sd_x_a < 0.5) & (m.sd_y_a < 0.5) & (m.sd_z_a < 1) & (m.sd_x_b < 0.5) & (m.sd_y_b < 0.5) & (m.sd_z_b < 1)]
    a, b = m[['dx_a', 'dy_a', 'dz_a']].values, m[['dx_b', 'dy_b', 'dz_b']].values
    data[F] = (a - a.mean(0), b - b.mean(0))


def pooled(keys, perm=False):
    S, n = np.zeros((3, 3)), 0
    for F in keys:
        a, b = data[F]
        if perm:
            b = b[rng.permutation(len(b))]
        S += (a.T @ b + b.T @ a) / 2
        n += len(a)
    return S / n


def normal_plane(v):
    n = v / np.linalg.norm(v)
    n = n if n[2] > 0 else -n
    dipdir = (np.degrees(np.arctan2(n[0], n[1])) + 180) % 360
    return float(np.degrees(np.arccos(abs(n[2])))), float(dipdir)


w, v = np.linalg.eigh(pooled(res))
dip, dipdir = normal_plane(v[:, 0])
bd = []
for _ in range(300):
    ww, vv = np.linalg.eigh(pooled([res[i] for i in rng.integers(0, len(res), len(res))]))
    bd.append((normal_plane(vv[:, 0])[0], np.degrees(np.arctan2(vv[0, 2], vv[1, 2])) % 180))
bd = np.array(bd)
null = np.mean([np.linalg.eigvalsh(pooled(res, perm=True)) for _ in range(100)], axis=0)
out['pooled_patch'] = dict(families=res, eig_std_m=[float(x) for x in np.sqrt(np.clip(w, 0, None)) * 1000],
                           null_eig_std_m=[float(x) for x in np.sqrt(np.clip(null, 0, None)) * 1000],
                           long_axis_az=float(np.degrees(np.arctan2(v[0, 2], v[1, 2])) % 180),
                           long_axis_plunge=float(np.degrees(np.arcsin(abs(v[2, 2])))),
                           plane_dip=dip, plane_dip_dir=dipdir, plane_strike=(dipdir - 90) % 360,
                           dip_68=[float(np.percentile(bd[:, 0], 16)), float(np.percentile(bd[:, 0], 84))],
                           long_axis_az_68=[float(np.percentile(bd[:, 1], 16)), float(np.percentile(bd[:, 1], 84))])
json.dump(out, open('data/patches/fault_geometry.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
