#!/usr/bin/env python3
"""Export the sharpened (deconvolved) patch images in shareable form.

From data/sci/patch_images/ovl_incl/images_real.npz (sci_patch_images.py) writes
  data/patches/sharpened/sharpened_images.npz  : B_full and A_het images per family (81x81, 25 m pixels,
                                                  along-strike x down-dip, +-1 km about the event median), plus axes
  data/patches/sharpened/sharpened_patches_geo.csv : every pixel with density >= 5% of the family peak, with its
                                                  along-strike / down-dip offset (m), lon, lat, depth (km) and
                                                  normalised density (B_full)
Frame: strike 293 / dip 40 NE (+ along strike = toward az 293, + down dip = deeper). Placement as in
plot_sharpened_patches.py (family 3D location + median relative offset of well-located events).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from plot_sharpened_patches import GX, R, centre, kx, ky, lat0, lon0  # noqa: E402

OUT = 'data/patches/sharpened'


def main():
    os.makedirs(OUT, exist_ok=True)
    im = np.load('data/sci/patch_images/ovl_incl/images_real.npz')
    keep, rows = {}, []
    for F in range(1, 28):
        if f'{F}_B_full' not in im:
            continue
        b = im[f'{F}_B_full']; keep[f'fam{F}_B_full'] = b.astype(np.float32)
        keep[f'fam{F}_A_het'] = im[f'{F}_A_het'].astype(np.float32)
        c = centre(F)
        f = b / b.max()
        i, j = np.where(f >= 0.05)
        s, d = GX[i], GX[j]                                  # km; image indexed [s, d]
        X = c[:, None] + np.outer(R[0], s) + np.outer(R[1], d)
        rows.append(pd.DataFrame(dict(family=F, along_strike_m=s * 1e3, down_dip_m=d * 1e3,
                                      lon=X[0] / kx + lon0, lat=X[1] / ky + lat0, depth_km=X[2], density=f[i, j])))
    np.savez_compressed(f'{OUT}/sharpened_images.npz', along_strike_km=GX, down_dip_km=GX, **keep)
    T = pd.concat(rows, ignore_index=True)
    T.round(dict(along_strike_m=1, down_dip_m=1, lon=6, lat=6, depth_km=4, density=4)).to_csv(
        f'{OUT}/sharpened_patches_geo.csv', index=False)
    print(f'{len(keep) // 2} families; {len(T)} pixels -> {OUT}/')


if __name__ == '__main__':
    main()
