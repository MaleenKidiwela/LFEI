#!/usr/bin/env python
"""Summary of every observable tested for a change at depth (2008-2026): table + depth-limit figure.
Numbers are copied from the individual analyses (see NOTES.md, 2026-11-23..25); this script only collates them.
Outputs data/sci/depth_summary.csv, figures/sci_depth_summary.png"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# observable, group, depth range km (z0, z1), M7 result, long-term result, 95% limit (text), limit for figure (% dv/v, or None), verdict
R = [
 ('Direct-S travel times, LFE→station (relative)', 'velocity', 1, 28, '−0.002 ± 0.009% (under re-check: station terms tied to coda)', 'decline −0.002 ± 0.009%', '±0.02% → under re-check; free station terms give several× larger', 0.02, 'no change (limit under re-check)'),
 ('LFE − earthquake differential S−P (layer above the LFEs)', 'velocity', 22, 31, 'no change seen; earlier ±0.008% WITHDRAWN (kernels > travel time on ~20% of paths)', 'free station terms: 1σ ~0.12–0.17%', '~±0.25–0.35% (revised fit pending)', 0.3, 'no change (weak limit)'),
 ('S−P around the sources (Vp, Vs, Vp/Vs)', 'velocity', 20, 40, '−0.07 ± 0.14%; Vp/Vs +0.06 ± 0.14%', 'decline +0.04 ± 0.07%', '±0.3–0.4% (10-km sphere)', 0.35, 'no change'),
 ('Coda: per-family source terms (1.83 M measurements)', 'velocity', 20, 32, 'family scatter < 0.03%', 'family scatter < 0.06%', 'common deep term −0.10…+0.02%', 0.10, 'no change'),
 ('Ordinary repeating earthquakes, deep (19 km) vs shallow (8 km)', 'velocity', 8, 19, 'deep − shallow +0.020 ± 0.011%', 'deep − shallow +0.006 ± 0.006%', '±0.02% (decline)', 0.02, 'no source-depth dependence'),
 ('Inter-station noise 0.05–0.1 Hz (56% of sensitivity at 10–30 km)', 'velocity', 10, 30, '−0.06% (13th placebo pct)', '−0.0003 ± 0.0055 %/yr; 2021→23 same-pair −0.05% (one-sided = source)', '~0.3% if confined to 10–30 km', 0.3, 'no change'),
 ('Single-station noise, 27 stations, 0.1–0.3 → 3–5 Hz', 'velocity', 0, 3, 'step grows with frequency: 0.1–0.3 Hz −0.07% (0 stations |z|>3) → 1–3 Hz −0.13% (11)', 'borehole decline 0.3–5 Hz, weak at 0.1–0.3 Hz', '—', None, 'shallow only'),
 ('Receiver functions: top of low-velocity layer (A_t)', 'structure', 20, 30, 'A_t −20 ± 21%', 'contrast +0.10 ± 0.07 km/s (1.4σ)', '5–6% Vs (long term), 10–15% (quakes)', 5.5, 'no change'),
 ('Receiver functions: slab power (Herath method)', 'structure', 20, 35, 'best ~10% Vs', 'best KMPB 4.7% Vs', '~5% Vs (2σ)', 4.7, 'no change'),
 ('Receiver-function anisotropy (k=1, 2)', 'structure', 5, 30, '17–27% amplitude', 'k1 10–20%, k2 15–17%, 3–5° rotation', '3–8% anisotropy', None, 'no change'),
 ('LFE-stack secondary phases (in-layer reflections)', 'structure', 25, 31, 'no phase found', '0/34 hypotheses robust', 'in-layer dv/v ±0.5–2%', 1.0, 'no phase'),
 ('Attenuation: Δt*_S, Δt*_P, P−S', 'attenuation', 0, 28, 'all |z| < 1.4', 'Δt*_S −0.2 ms', 'Δt*_S ±10 ms (path 1/Qs < 7e-4)', None, 'no change'),
 ('Coda Q (decay rate)', 'attenuation', 0, 30, '+0.7% (1 yr)', 'borehole −4% (z −1.7, receiver side)', '~7% (M7, 1 yr)', None, 'no deep change'),
 ('Restart LFE polarity / radiation pattern', 'source', 28, 31, '0.6% flips (44th pct); post/pre 1.00', '—', 'pattern r 0.81 (93rd pct)', None, 'no change'),
 ('Source spectra, duration, size', 'source', 28, 31, 'corner within ±20%', 'none in both station groups', 'stress drop < 2×', None, 'no change'),
 ('Tidal sensitivity (frictional state)', 'source', 28, 31, 'unchanged', 'unchanged', 'drop > 55% excluded', None, 'no change'),
 ('Shear-wave splitting', 'structure', 0, 30, '|z| ≤ 1.5', 'no decline counterpart', 'delay ±1–2 ms', None, 'no change'),
 ('LFE locations at restart / migration', 'source', 28, 31, 'same place to 5–10 m', '< ~1 m/yr between families', '—', None, 'no change'),
 ('Seismicity 15–35 km: stress inversion, rates', 'stress', 15, 35, 'σ1 rotation 16° (p 0.81)', 'σ3 stable E–W', '—', None, 'no change'),
 ('GNSS (56 stations) and borehole strain', 'deformation', 20, 35, 'no LFE-zone afterslip (< Mw 5.5–5.6)', 'slip-rate change |z| ≤ 1.7', 'Mw 4.4 per episode (strain)', None, 'no slip detected'),
 ('Main-band ETS LFEs/tremor (Ducellier & Creager; PNSN)', 'context', 28, 47, 'main band not silenced by the M7', 'Shelly rate during/before ETS 1.02 (0.96–1.04)', 'coupling < ~4%', None, 'not coupled'),
]
D = pd.DataFrame(R, columns=['observable', 'group', 'z0_km', 'z1_km', 'M7', 'long_term', 'limit95', 'limit_pct', 'verdict'])
D.to_csv('data/sci/depth_summary.csv', index=False)

fig, ax = plt.subplots(figsize=(10, 5.6))
v = D.dropna(subset=['limit_pct'])
col = {'velocity': 'C0', 'structure': 'C2'}
XO = {'Ordinary repeating earthquakes, deep (19 km) vs shallow (8 km)': 1.15}   # avoid overlap with direct S
LY = {'Direct-S travel times, LFE→station (relative)': 3, 'Ordinary repeating earthquakes, deep (19 km) vs shallow (8 km)': 9,
      'LFE − earthquake differential S−P (layer above the LFEs)': 21, 'Coda: per-family source terms (1.83 M measurements)': 18.5,
      'S−P around the sources (Vp, Vs, Vp/Vs)': 34.5, 'Inter-station noise 0.05–0.1 Hz (56% of sensitivity at 10–30 km)': 11,
      'LFE-stack secondary phases (in-layer reflections)': 24, 'Receiver functions: top of low-velocity layer (A_t)': 14.5,
      'Receiver functions: slab power (Herath method)': 33.5}
for i, r in v.reset_index(drop=True).iterrows():
    x = r.limit_pct * XO.get(r.observable, 1)
    ax.plot([x] * 2, [r.z0_km, r.z1_km], '-', lw=7, color=col.get(r.group, 'C1'), alpha=0.3 if 're-check' in r.verdict else 0.75, solid_capstyle='butt')
    ha = 'right' if x > 3 else 'left'
    ax.text(x * (0.9 if ha == 'right' else 1.12), LY.get(r.observable, (r.z0_km + r.z1_km) / 2),
            (r.observable.split(' (')[0] + (' (under re-check)' if 're-check' in r.verdict else '')).replace('Ordinary repeating earthquakes, deep', 'Repeating earthquakes 8 vs 19 km'),
            fontsize=7.5, va='center', ha=ha)
ax.axvspan(0.0004, 0.004, color='0.85', zorder=0)
ax.text(0.0013, 2, 'expected from the\nM7 static strain', fontsize=7.5, ha='center', va='top', color='0.3')
ax.axhspan(28, 31, color='#fde0dd', zorder=0); ax.text(0.00045, 29.5, 'LFEs', fontsize=8, va='center', color='#a3321f')
ax.axhspan(0, 0.8, color='#e3eef6', zorder=0); ax.text(0.00045, 0.4, 'observed changes (top ~0.1–0.8 km)', fontsize=7.5, va='center', color='#1f5f8b')
ax.set_xscale('log'); ax.set_xlim(3e-4, 40); ax.set_ylim(36, 0)
ax.set_xlabel('95% (2σ) limit on a velocity change at that depth (%; blue = dv/v, green = layer Vs)')
ax.set_ylabel('depth range sampled (km)')
ax.set_title('What could have changed at depth and not been seen, 2008–2026', fontsize=10)
ax.grid(alpha=0.3, which='both')
fig.tight_layout(); fig.savefig('figures/sci_depth_summary.png', dpi=150)
print(D[['observable', 'verdict']].to_string())
