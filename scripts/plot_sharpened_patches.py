#!/usr/bin/env python3
"""All families with sharpened (deconvolved, B_full) patch images placed at their 3D locations.

Images: data/sci/patch_images/ovl_incl/images_real.npz (scripts/sci_patch_images.py), 81x81 grid,
25 m pixels, +-1 km, in the fault frame (s = along strike toward az 293, d = down dip), centred on
the median of each family's well-located events. Each image is placed at the family location
(data/locations/fam{F}_3d.json) plus that median relative offset, and drawn as contours at 20/50/80%
of its peak, transformed into E, N, depth.
Panels: (a) map, (b) view onto the fault plane (along strike x down dip), (c) section along strike
(edge-on view, across-plane vs down-dip). Families without an image (4, 9, 11) are shown as x.
Output figures/sharpened_patches_all_families.png
"""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon

sys.path.insert(0, 'scripts')
from patch_tensors import shelly_frame  # noqa: E402

R = shelly_frame(293.0, 40.0)            # rows s, d, n in (E, N, Down)
L, PIX = 1000.0, 25.0
GX = (np.arange(81) * PIX - L) / 1e3     # km
LEVELS = [0.2, 0.5, 0.8]
lat0, lon0 = 40.06, -123.625
kx, ky = 111.19 * np.cos(np.radians(lat0)), 111.19
im = np.load('data/sci/patch_images/ovl_incl/images_real.npz')
RS = pd.read_csv('data/sci/patch_images/ovl_incl/real_summary.csv').set_index('F')


def centre(F):
    loc = json.load(open(f'data/locations/fam{F}_3d.json'))
    e = pd.read_csv(f'data/relocations/fam{F}_events_3d.csv')
    e = e[(e.sd_x < 0.5) & (e.sd_y < 0.5) & (e.sd_z < 1.0)]
    off = np.median(e[['dx', 'dy', 'dz']].values, axis=0) if len(e) else np.zeros(3)
    return np.array([(loc['lon'] - lon0) * kx, (loc['lat'] - lat0) * ky, loc['depth_km']]) + off


def contours(img):
    """contour polygons of a normalised image in (s, d) km, per level"""
    f = img / img.max()
    cs = plt.figure().add_subplot().contour(GX, GX, f.T, levels=LEVELS)
    segs = cs.allsegs
    plt.close()
    return segs


def slab_models():
    """plate-interface depth (km, positive down) as functions of local E, N (km): McCrory et al. 2012 (USGS DS633,
    modeled slab) and Slab2 Cascadia (Hayes et al. 2018). Files in data/slab/."""
    from scipy.interpolate import LinearNDInterpolator
    out = {}
    m = pd.read_csv('data/slab/mccrory/DS633_Data/DS633_ASCII_GIS_Data/Modeled_slab_XYZ_Grid_GCS_WGS84.txt')
    m = m[(m.Latitude.between(39.0, 41.5)) & (m.Longitude.between(-125.0, -122.0))]
    s2 = pd.read_csv('data/slab/cas_slab2_dep_02.24.18.xyz', header=None, names=['lon', 'lat', 'z']).dropna()
    s2['lon'] = np.where(s2.lon > 180, s2.lon - 360, s2.lon)
    s2 = s2[(s2.lat.between(39.0, 41.5)) & (s2.lon.between(-125.0, -122.0))]
    for name, lon, lat, z in [('McCrory et al. 2012', m.Longitude, m.Latitude, m.Depth_km),
                              ('Slab2 (Hayes et al. 2018)', s2.lon, s2.lat, s2.z)]:
        xy = np.c_[(lon.values - lon0) * kx, (lat.values - lat0) * ky]
        out[name] = LinearNDInterpolator(xy, np.abs(z.values))
    # Bloch, Bostock & Audet 2023 (G3) receiver-function slab model: t = top of the low-velocity layer, m = slab Moho
    import h5py
    for k, name in [('t', 'Bloch et al. 2023 top (t)'), ('m', 'Bloch et al. 2023 slab Moho (m)')]:
        f = h5py.File(f'data/slab/bloch2023/grids_masked/{k}_depth_masked.nc', 'r')
        la, lo, z = f['lat'][:], f['lon'][:], f['z'][:]
        i, j = (la > 39.0) & (la < 41.5), (lo > -125.0) & (lo < -122.0)
        LO, LA = np.meshgrid(lo[j], la[i]); Z = z[np.ix_(i, j)]
        ok = np.isfinite(Z)
        out[name] = LinearNDInterpolator(np.c_[(LO[ok] - lon0) * kx, (LA[ok] - lat0) * ky], np.abs(Z[ok]))
    return out


SLAB_STYLE = {'McCrory et al. 2012': dict(color='k', ls='-'), 'Slab2 (Hayes et al. 2018)': dict(color='crimson', ls='--'),
              'Bloch et al. 2023 top (t)': dict(color='royalblue', ls='-.'),
              'Bloch et al. 2023 slab Moho (m)': dict(color='royalblue', ls='-')}


def main():
    slabs = slab_models()
    C = {F: centre(F) for F in range(1, 28)}
    c0 = np.mean(np.array(list(C.values())), axis=0)
    depths = np.array([C[F][2] for F in C])
    norm = plt.Normalize(depths.min(), depths.max())
    cmap = plt.get_cmap('viridis_r')
    fig = plt.figure(figsize=(18, 15))
    axm = fig.add_axes([0.05, 0.56, 0.42, 0.40])
    axf = fig.add_axes([0.52, 0.56, 0.42, 0.40])
    axz = fig.add_axes([0.05, 0.06, 0.50, 0.42])
    axs = fig.add_axes([0.60, 0.06, 0.37, 0.42])
    AZX = np.radians(23.0)                          # cross-section direction = dip direction (NNE)
    xsec = lambda E, N: (E - c0[0]) * np.sin(AZX) + (N - c0[1]) * np.cos(AZX)
    ZS, ZD = (-3.6, 2.6), (0.4, 3.9)
    for F in range(1, 28):
        c = C[F]
        col = cmap(norm(c[2]))
        rel = c - c0
        s0, d0, n0 = R @ np.array([rel[0], rel[1], rel[2]])
        key = f'{F}_B_full'
        if key not in im:
            for ax, xy in [(axm, (c[0], c[1])), (axf, (s0, d0)), (axs, (xsec(c[0], c[1]), c[2])), (axz, (s0, d0))]:
                ax.plot(*xy, 'x', color=col, ms=7, mew=1.5)
                ax.annotate(str(F), xy, fontsize=7, color='0.4', xytext=(3, 3), textcoords='offset points')
            continue
        segs = contours(im[key])
        for li, lev in enumerate(segs):
            alpha = [0.25, 0.55, 0.95][li]
            for seg in lev:
                s, d = seg[:, 0], seg[:, 1]
                X = c[:, None] + (np.outer(R[0], s) + np.outer(R[1], d))        # (3, n): E, N, depth
                axm.add_patch(Polygon(np.c_[X[0], X[1]], closed=True, fc=col, ec='none', alpha=alpha))
                axf.add_patch(Polygon(np.c_[s0 + s, d0 + d], closed=True, fc=col, ec='none', alpha=alpha))
                axz.add_patch(Polygon(np.c_[s0 + s, d0 + d], closed=True, fc=col, ec=col if li == 0 else 'none', lw=0.4, alpha=alpha))
                axs.add_patch(Polygon(np.c_[xsec(X[0], X[1]), X[2]], closed=True, fc=col, ec=col if li == 0 else 'none', lw=0.6, alpha=alpha))
        r = RS.loc[F]
        flag = '*' if r.get('B_full_p_dip', 1) <= 0.05 else ''
        axm.annotate(f'{F}{flag}', (c[0], c[1]), fontsize=8, weight='bold', xytext=(6, 6), textcoords='offset points')
        axf.annotate(f'{F}{flag}', (s0, d0), fontsize=8, weight='bold', xytext=(8, -10), textcoords='offset points')
        axs.annotate(str(F), (xsec(c[0], c[1]), c[2]), fontsize=7, xytext=(5, -2), textcoords='offset points')
        if ZS[0] < s0 < ZS[1] and ZD[0] < d0 < ZD[1]:
            axz.annotate(f'{F}{flag}', (s0, d0), fontsize=10, weight='bold', xytext=(0, -16), textcoords='offset points', ha='center')
    for ax in (axm, axf):
        ax.set_aspect('equal'); ax.autoscale_view(); ax.margins(0.06)
    # --- plate interface
    xl, yl = axm.get_xlim(), axm.get_ylim()
    gE, gN = np.meshgrid(np.linspace(xl[0], xl[1], 200), np.linspace(yl[0], yl[1], 200))
    for name, f in slabs.items():
        if 'Moho' in name:
            continue
        st_ = SLAB_STYLE[name]
        cs = axm.contour(gE, gN, f(gE, gN), levels=np.arange(16, 40, 2), colors=st_['color'], linestyles=st_['ls'], linewidths=0.8)
        axm.clabel(cs, fmt='%d km', fontsize=7)
        axm.plot([], [], label=f'plate interface, {name}', **st_)
    axm.legend(loc='upper right', fontsize=7)
    # intersection of the 293/40 plane with the interface, in fault-plane coordinates
    gs, gd = np.meshgrid(np.linspace(-11, 11, 300), np.linspace(-9, 6, 250))
    PE = c0[0] + gs * R[0, 0] + gd * R[1, 0]; PN = c0[1] + gs * R[0, 1] + gd * R[1, 1]; PZ = c0[2] + gs * R[0, 2] + gd * R[1, 2]
    for name, f in slabs.items():
        st_ = SLAB_STYLE[name]
        for ax in (axf, axz):
            ax.contour(gs, gd, PZ - f(PE, PN), levels=[0], colors=st_['color'], linestyles=st_['ls'], linewidths=1.6)
        axf.plot([], [], lw=1.6, label=f'{name}: where the 293/40 plane crosses it', **st_)
    axf.legend(loc='upper left', fontsize=7)
    # cross-section profiles of the interface along the dip direction, at three along-strike offsets
    xs_ = np.linspace(-8, 8, 200)
    for name, f in slabs.items():
        st_ = SLAB_STYLE[name]
        for k, off in enumerate([-6.0, 0.0, 6.0]):
            E = c0[0] + xs_ * np.sin(AZX) + off * R[0, 0]; N = c0[1] + xs_ * np.cos(AZX) + off * R[0, 1]
            axs.plot(xs_, f(E, N), color=st_['color'], ls=st_['ls'], lw=1.6 if off == 0 else 0.7, alpha=1 if off == 0 else 0.6,
                     label=f'{name}' if off == 0 else None)
    axs.plot(xs_, c0[2] + xs_ * np.tan(np.radians(40)), color='0.5', ls=':', lw=1.2, label='293°/40° plane through centroid')
    sh = pd.read_csv('data/families/shelly_family_locations.csv')
    axs.plot(xsec((sh.lon - lon0) * kx, (sh.lat - lat0) * ky), sh.depth_km, '+', color='0.45', ms=6, label='Shelly et al. 2025 locations')
    axs.set_xlim(-7, 7); axs.set_ylim(36, 15); axs.set_aspect('equal')
    axs.set_xlabel('distance along dip direction (km, + = N23°E)'); axs.set_ylabel('depth (km)')
    axs.set_title('(d) Cross-section along dip (N23°E), all families projected; slab interfaces from three models\n'
                  '(thick = section through the array centre, thin = ±6 km along strike)', fontsize=10)
    axs.legend(loc='lower left', fontsize=7)
    axm.set_xlabel(f'E (km from {lon0})'); axm.set_ylabel(f'N (km from {lat0})')
    axm.set_title('(a) Map view: sharpened patches at their 3D locations (colour = depth)')
    # strike arrow
    st = np.radians(293)
    xl, yl = axm.get_xlim(), axm.get_ylim()
    ax0, ay0 = xl[0] + 0.12 * (xl[1] - xl[0]), yl[0] + 0.1 * (yl[1] - yl[0])
    axm.annotate('', (ax0 + 2 * np.sin(st), ay0 + 2 * np.cos(st)), (ax0, ay0), arrowprops=dict(arrowstyle='->'))
    axm.text(ax0 + 2 * np.sin(st), ay0 + 2 * np.cos(st) + 0.3, 'strike 293°', fontsize=8, ha='center')
    axf.invert_yaxis()
    axf.set_xlabel('along strike (km, + = WNW toward 293°)'); axf.set_ylabel('down dip (km, + = deeper, NNE)')
    axf.set_title('(b) View onto the fault plane (strike 293°, dip 40° NE), true scale')
    axf.add_patch(plt.Rectangle((ZS[0], ZD[0]), ZS[1] - ZS[0], ZD[1] - ZD[0], fill=False, ec='k', ls=':', lw=0.8))
    axz.set_xlim(*ZS); axz.set_ylim(ZD[1], ZD[0]); axz.set_aspect('equal')
    axz.set_xlabel('along strike (km, + WNW)'); axz.set_ylabel('down dip (km, + deeper)')
    axz.set_title('(c) Zoom on the central cluster (dotted box in b), fault-plane view, true scale; contours 20/50/80% of peak')
    axz.plot([ZS[0] + 0.15, ZS[0] + 0.65], [ZD[1] - 0.2] * 2, 'k-', lw=2); axz.text(ZS[0] + 0.4, ZD[1] - 0.27, '500 m', ha='center', fontsize=9)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=[axm, axf], fraction=0.02, pad=0.01)
    cb.set_label('depth (km)'); cb.ax.invert_yaxis()
    fig.text(0.05, 0.003, 'Contours at 20/50/80% of each patch\'s peak density after deconvolving the empirical location error '
             '(resolution ~110 m along strike, ~90 m down dip). * = flatter than Gaussian (p<=0.05). x = too few events to image (4, 9, 11). Our absolute depths may be ~2.5 km too deep (earthquake test); Shelly locations shown for comparison in (d). Slab models: McCrory et al. 2012 (USGS DS633); Slab2 (Hayes et al. 2018); Bloch, Bostock & Audet 2023 (G3, receiver functions; t = top of low-velocity layer, m = slab Moho).',
             fontsize=8, wrap=True)
    fig.savefig('figures/sharpened_patches_all_families.png', dpi=140)
    print('wrote figures/sharpened_patches_all_families.png')


if __name__ == '__main__':
    main()
