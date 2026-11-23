#!/usr/bin/env python3
"""Build the shareable HTML report (report/index.html + report/figs/*) from the per-family
summaries, the pooled per-station dv/v and the cross-family tables."""
import glob
import html
import json
import os

import numpy as np
import pandas as pd
from PIL import Image

OUT = 'report'
os.makedirs(f'{OUT}/figs', exist_ok=True)


def fig(src, name, width=1400):
    if not os.path.exists(src):
        return None
    im = Image.open(src).convert('RGB')
    if im.width > width:
        im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
    im.save(f'{OUT}/figs/{name}.jpg', quality=84, optimize=True)
    return f'figs/{name}.jpg'


def f(x, fmt='{:.3f}', na='–'):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return na
        return fmt.format(x)
    except (ValueError, TypeError):
        return html.escape(str(x))


def sig(v, e, fmt='{:+.3f} ± {:.3f}', unit=''):
    """value ± se with a significance chip."""
    try:
        v, e = float(v), float(e)
    except (TypeError, ValueError):
        return '<span class="na">–</span>'
    if v is None or e is None or not np.isfinite(v) or not np.isfinite(e) or e <= 0:
        return '<span class="na">–</span>'
    z = v / e
    cls = 'strong' if abs(z) >= 3 else ('weak' if abs(z) >= 2 else 'none')
    return f'<span class="val">{fmt.format(v, e)}{unit}</span> <span class="z {cls}">{z:+.1f}σ</span>'


rows = [json.load(open(p)) for p in sorted(glob.glob('data/reports/fam*_summary.json'))]
S = pd.DataFrame(rows).set_index('family').sort_index()
for c in S.columns:
    if S[c].dtype == object:
        conv = pd.to_numeric(S[c], errors='coerce')
        if conv.notna().sum() == S[c].notna().sum():
            S[c] = conv
g = lambda F, k: S.loc[F].get(k) if k in S.columns else None  # noqa: E731

# ---------------- cross-family tables
ct = pd.read_csv('data/reports/all_families_summary.csv') if os.path.exists('data/reports/all_families_summary.csv') else None
qr = pd.read_csv('data/reports/quake_response_catalog.csv')
qt = qr.groupby(['quake', 'dist_km'], sort=False).agg(
    fam_top10=('gap_rank', lambda x: int((x <= 10).sum())), median_gap=('gap_d', 'median'),
    max_gap=('gap_d', 'max'), post=('post30', 'sum'), pre=('pre365', 'sum')).reset_index()
qt['ratio'] = qt.post / (qt.pre * 30 / 365)
pool = pd.read_csv('data/dvv/pool/params_by_channel.csv')
pool['type'] = np.where(pool.chan.str.startswith('PB'), 'Borehole (PB)', 'Surface (BK, NC)')


def wmean(gr, c):
    w = 1 / gr[c + '_se'] ** 2
    m = (gr[c] * w).sum() / w.sum()
    chi = ((gr[c] - m) ** 2 * w).sum() / max(len(gr) - 1, 1)
    return m, np.sqrt(max(chi, 1)) / np.sqrt(w.sum())


pool_rows = []
for t, gr in pool.groupby('type', sort=False):
    r = dict(type=t, n=len(gr), ann=gr.annual_amp_pct.median(), doy=gr.annual_peak_doy.median())
    for c in ('step_pet_pct', 'step_fer_pct', 'step_m7_pct', 'transient60_pct'):
        r[c] = wmean(gr, c)
    pool_rows.append(r)
sta = pool.assign(sta=pool.chan.str.rsplit('.', n=2).str[0]).groupby('sta').agg(
    type=('type', 'first'), dist=('dist_km', 'first'), nch=('chan', 'size'), nfam=('n_fam', 'median'),
    m7=('step_m7_pct', 'median'), m7se=('step_m7_pct_se', 'median'), tr=('transient60_pct', 'median'),
    trse=('transient60_pct_se', 'median'), ann=('annual_amp_pct', 'median'), doy=('annual_peak_doy', 'median')
).sort_values('m7')

figs = {k: fig(v, k) for k, v in {
    'clouds': 'figures/all_families_clouds.png', 'poolmap': 'figures/dvv_pool_map.png',
    'poolcurves': 'figures/dvv_pool_channels.png', 'm7': 'figures/all_families_m7.png',
    'fam23_dvv': 'figures/fam23_dvv_cwire.png', 'patches': 'figures/patch_tensors.png',
    'cascades': 'figures/burst_cascades.png', 'longterm': 'figures/dvv_longterm.png',
    'longterm_model': 'figures/dvv_longterm_model.png',
    'noise_ts': 'figures/sci_noise_dvv_timeseries.png', 'noise_steps': 'figures/sci_noise_dvv_steps.png',
    'strain': 'figures/sci_strain_stacks.png',
    'sharpened': 'figures/sharpened_patches_all_families.png',
    'patchimg': 'figures/sci_patch_images_families_B_full.png', 'spectra': 'figures/sci_source_spectra_quakes.png', 'lfespec': 'figures/lfe_spectra.png', 'instr': 'figures/sci_dvv_instrument_check.png', 'poolclean': 'figures/dvv_pool_channels_clean.png', 'famspec': 'figures/sci_family_spectra_scaling.png', 'sws': 'figures/sci_sws_timeseries.png', 'poro': 'figures/sci_tidal_spotl_poro.png',
    'diprake': 'figures/sci_tidal_spotl_dip_rake.png', 'gnss_q2': 'figures/sci_gnss_q2_postseismic.png',
    'density': 'figures/cloud_density_fault_plane.png', 'density_m7': 'figures/cloud_density_m7.png',
    'tidal_fits': 'figures/sci_tidal_spotl_fits.png', 'gnss_q1': 'figures/sci_gnss_q1_rate_peaks.png',
    'gnss_q3': 'figures/sci_gnss_q3_velocity.png', 'strain_null': 'figures/sci_strain_null_potency.png',
    'dyn_trigger': 'figures/sci_dyn_trigger_threshold.png', 'recovery': 'figures/sci_recovery_model_fits.png',
    'moments': 'figures/sci_moments_cumulative.png', 'coupling': 'figures/sci_cascades2_network.png',
    'dose': 'figures/sci_dose_response_drivers.png', 'seasonal': 'figures/sci_tides_seasonal.png'}.items()}
for F in S.index:
    fig(f'figures/fam{F}_relocations_3d.png', f'fam{F}_reloc')
    fig(f'figures/fam{F}_dvv_cwire.png', f'fam{F}_dvv')


def img(name, alt):
    p = f'figs/{name}.jpg'
    if not os.path.exists(f'{OUT}/{p}'):
        return f'<p class="na">Figure not available.</p>'
    return f'<figure><img src="{p}" alt="{html.escape(alt)}" loading="lazy"><figcaption>{html.escape(alt)}</figcaption></figure>'


# ---------------- page
H = []
H.append('''<title>MTJ LFE Families</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500;600&display=swap">
<style>
/* layout: single reading column with wide figure/table bands; sticky family index */
:root{
  --bg:#f6f7f8; --surface:#ffffff; --fg:#1b2026; --muted:#5b6672; --line:#d9dee4;
  --accent:#1f5f8b; --accent-soft:#e3eef6; --strong:#a3321f; --weak:#a86c12; --ok:#2f6f4f;
  --display:"IBM Plex Serif", Georgia, serif; --body:"IBM Plex Sans", system-ui, sans-serif;
  --mono:"IBM Plex Mono", ui-monospace, monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#12161a; --surface:#1a2026; --fg:#e4e8ec; --muted:#9aa6b2; --line:#2c353e;
  --accent:#7fb6dc; --accent-soft:#1d2e3b; --strong:#f08a74; --weak:#e2b05c; --ok:#7cc59f; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#12161a; --surface:#1a2026; --fg:#e4e8ec; --muted:#9aa6b2; --line:#2c353e;
  --accent:#7fb6dc; --accent-soft:#1d2e3b; --strong:#f08a74; --weak:#e2b05c; --ok:#7cc59f; color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:15px/1.6 var(--body);
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:24px 64px}
main{max-width:1080px;margin:0 auto;display:flex;flex-direction:column;gap:40px}
h1,h2,h3{font-family:var(--display);font-weight:600;text-wrap:balance;line-height:1.2;margin:0}
h1{font-size:2.1rem} h2{font-size:1.5rem;padding-bottom:6px;border-bottom:1px solid var(--line)} h3{font-size:1.15rem}
p{margin:0;max-width:72ch} section{display:flex;flex-direction:column;gap:16px;min-width:0}
.lede{color:var(--muted);max-width:72ch}
.eyebrow{font:500 .75rem var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--accent)}
.findings{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}
.finding{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:16px;display:flex;flex-direction:column;gap:8px;min-width:0}
.finding h3{font-size:1.02rem}
.finding p{font-size:.93rem}
.tbl{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:.86rem;font-variant-numeric:tabular-nums}
th,td{padding:6px 10px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{font-weight:600;color:var(--muted);background:var(--accent-soft);position:sticky;top:0}
tr:last-child td{border-bottom:none}
td a{color:var(--accent);text-decoration:none;font-weight:600}
.val{font-family:var(--mono);font-size:.82rem}
.z{font:500 .72rem var(--mono);padding:1px 5px;border-radius:4px}
.z.strong{color:var(--strong);background:color-mix(in srgb,var(--strong) 14%,transparent)}
.z.weak{color:var(--weak);background:color-mix(in srgb,var(--weak) 14%,transparent)}
.z.none{color:var(--muted)}
.na{color:var(--muted)}
figure{margin:0;display:flex;flex-direction:column;gap:6px}
figure img{width:100%;height:auto;border:1px solid var(--line);border-radius:6px;background:#fff}
figcaption{font-size:.82rem;color:var(--muted)}
.famnav{display:flex;flex-wrap:wrap;gap:6px}
.famnav a{font:500 .85rem var(--mono);color:var(--accent);background:var(--surface);border:1px solid var(--line);
  border-radius:6px;padding:3px 9px;text-decoration:none}
.famnav a:hover,.famnav a:focus-visible{background:var(--accent-soft);outline:2px solid var(--accent);outline-offset:1px}
.fam{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:20px;display:flex;flex-direction:column;gap:14px}
.fam h3{font-size:1.3rem}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px 24px}
.kv div{display:flex;flex-direction:column;gap:2px;min-width:0}
.kv .k{font:500 .72rem var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.kv .v{font-size:.9rem}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
ul{margin:0;padding-left:1.2em;display:flex;flex-direction:column;gap:6px;max-width:80ch}
.note{font-size:.88rem;color:var(--muted)}
@media (prefers-reduced-motion: reduce){*{scroll-behavior:auto}}
</style>
<main>
<header style="display:flex;flex-direction:column;gap:10px">
  <span class="eyebrow">Mendocino triple junction · Shelly v2 LFE catalog · 2008–2026</span>
  <h1>MTJ LFE families</h1>
  <p class="lede">Per-family results for all 27 low-frequency-earthquake families: 3D locations, relative relocations
  through time, coda-wave dv/v measured CWIRE-style on stacks of repeats, and the response to the 2021 M6.2 Petrolia,
  2022 M6.4 Ferndale and 2024 M7.0 Mendocino earthquakes. dv/v values are in %, positive = faster.
  ± values are bootstrap/jackknife standard errors; chips give the significance.</p>
</header>
''')

# ---------------- findings
pr = {r['type']: r for r in pool_rows}
sf, bh = pr.get('Surface (BK, NC)'), pr.get('Borehole (PB)')
m7q = qt[qt.quake.str.contains('2024-12-05')].iloc[0]
fer = qt[qt.quake.str.contains('2022-12-20')].iloc[0]
pet = qt[qt.quake.str.contains('M6.2')].iloc[0]
H.append('''<section><h2>What we learned</h2>
<p class="lede">The M7.0 acted as a natural clamp-and-hold experiment on a very weak, deep strike-slip fault. These are the results
that survived null tests and controls, grouped by how new they are.</p>
<h3>New</h3><ul>
<li><b>Only the M7 lowered pore pressure at the LFE fault, and only the M7 silenced it.</b> Of 12 nearby quake dates, Coulomb stress and
clamping do not single out the M7 (Ferndale was larger on both); the undrained pore-pressure drop does, in every model variant.</li>
<li><b>The fault's frictional state did not change.</b> Tidal sensitivity (±28% for M2) stayed the same across all three quakes (a drop of
more than ~55% is excluded), LFE sizes did not change and there is no healing with hold time (≤3.5% per decade). The missing moment was
repaid within ~60–90 days. With ~3.5 kPa of extra effective clamping, the unchanged tidal response implies an effective normal stress of
at least ~6 kPa (rough): a fault held near lithostatic pore pressure.</li>
<li><b>The tides and the M7 cannot be explained by one static fault response.</b> With a converged ocean-tide load at LFE depth (SPOTL,
layered Earth), the LFE rate follows right-lateral tidal shear on the 293°/40° plane (no bootstrap replicate prefers left-lateral), with a low
effective friction μ′ = 0.09 ± 0.01 and aσ ≈ 0.63 kPa (with the local rigidity). Written in Shelly &amp; Materna’s undrained poroelastic form,
the tides want a small Skempton B (0–0.5) and μ ≈ 0.1, as Lu, Ide &amp; Ren found for tremor, and reject Shelly &amp; Materna’s B ≈ 1, μ = 0.6
in 24 of 26 families. Yet making the M7 discourage slip needs B ≈ 1 and μ ≳ 0.5, and the tidal-best parameters predict a ~25× rate
increase after the M7. No single static (B, μ) fits both, so the silencing needs a timescale-dependent or nonlinear mechanism
(for example drained vs undrained behaviour, a permeability change, or a threshold response to a large step).</li>
<li><b>The LFEs switched back on abruptly, ~18 days after the M7, not diffusively.</b> Per-family rate fits give a median half-recovery
time of 18 d (9–30 d). A step switch-on or a rate-and-state negative step fits best; half-space diffusion is worse in 27 of 27 families, and a
network-scale diffusive front would need a diffusivity of ~180–1500 m²/s, over 100× fault-zone values. The restart was synchronous
(17 of 27 first events within 1.3 d) with no along-strike or depth order. In rate-and-state terms the silence matches the ~3.5 kPa effective
clamping with loading of ~71 kPa/yr (41–93), the rate expected from cm/yr slip across the whole LFE zone, not a single patch.</li>
<li><b>LFE patches are resolved and small:</b> ~560 × 350 m, elongated along strike and lying in Shelly's NE-dipping fault plane, smaller
than Cascadia patches. Deconvolved images show patches elongated along strike and flatter-topped than a Gaussian (5–7 families
significant), with no resolvable sub-asperities or rings above ~150–200 m and no shape change at the M7. Bursts re-rupture one sub-region
within minutes; patch size does not explain how long each family stayed silent.</li>
<li><b>Families talk to each other.</b> A burst in one family makes a burst in another ~9× more likely within 10 minutes. Coupling decays with
distance and is stronger along strike. Network episodes sweep inward from both ends of the 15-km array at ~80–100 km/day, in the range of
rapid tremor reversals, and the M7 did not change this.</li>
<li><b>Surface waves from distant large earthquakes trigger LFEs</b> (tentative). Burst onsets in the hour after teleseismic surface waves
are 1.44× the tide- and time-of-day-matched expectation (p = 0.0015), and 3.2× for the 30 quakes with the strongest expected shaking (p = 0.013).
A waveform check classifies 82% of these detections as real LFEs (87% for ordinary detections at the same quality). Thirteen quakes drive the
strong-shaking result, led by the 2020 Perryville and 2021 Chignik earthquakes in Alaska. Measured at a local broadband station (789 quakes),
triggering switches on above ~0.6 kPa of dynamic stress (68% range 0.2–1.4 kPa; 2.6× above vs 1.1× below), the same scale as the tidal
stresses and aσ, and then saturates: local quakes with 100–1000× more dynamic stress trigger about as much in the first hour. Shear stress
resolved on the fault predicts slightly better (threshold ~0.36 kPa). Rate-and-state with the measured stress and aσ ≈ 0.5 kPa predicts only
1.2× where 2.4× is seen, so this looks like threshold-and-saturation release of patches near failure. Dynamic stress
does not predict which local quake silenced the LFEs (Petrolia 173 kPa: rate up; M7 229 kPa: silenced), so the silencing needs a static
cause such as the pore-pressure drop.</li>

<li><b>The LFE rate has an annual cycle</b> of ~5–7%, peaking in August–September, in phase with the fastest coda velocities.</li>
<li><b>Coda dv/v from repeating LFEs gives an 18-year record</b> of the crust near the stations. Calibrated against random-date placebos:
an annual cycle (surface 0.045%, borehole 0.020%; z ≈ 4–10) and a decline since 2021 that has not recovered (surface −0.08 to −0.12%, borehole −0.06%
by 2025; z ≈ −6 to −7 on unsmoothed monthly data) are robust. The decline is the same for every LFE family, somewhat larger at surface sites than in boreholes (1.2–1.5×; an earlier factor of 2 depended on instrument breaks), starts no earlier than the Petrolia quake and is not explained by rainfall, so it is a near-surface change tied to
the 2021–24 quakes rather than a change at the LFE sources. The annual cycle tracks rainfall and is likely hydrological. Velocity drops at the quakes are real but modest (M7 surface −0.05%, z ≈ −2 to −3; Ferndale and
borehole steps marginal; Petrolia not significant).
<b>Ambient noise confirms the decline independently</b> (no LFE source involved): 2021→2025 change −0.27% at KMPB, −0.17% at BRIC,
−0.32% at B045 (1–4 Hz; LFE coda −0.24, −0.19, −0.13%) and none at the control station HAYF (−0.01%), against ±0.07% in pre-2021 placebo
years. Noise and LFE monthly series correlate at r = 0.6–0.75. Noise also shows clear drops at all three quakes at BRIC and B045
(M7 −0.3 to −0.75%; Ferndale at B045 z −5 to −9), while KMPB declines gradually from 2022 without a jump at the M7. After the instrument check, the KMPB and B045 declines and the BRIC and B045 M7 drops survive on the vertical component alone; BRIC’s 1–4 Hz decline does not survive an offset at its 2024 sensor reinstall, and the Ferndale drop at B045 is not seen on the vertical.</li>
</ul>
<h3>Confirms earlier work</h3><ul>
<li>All 27 families stopped after the M7 for ~7–25 days (Shelly &amp; Materna 2026), as did tremor in the same zone (Lu &amp; Ide 2026;
Lu, Ide &amp; Ren 2026); no other M≥5.5 within 200 km since 2008 did this to the LFEs.</li>
<li>The families define a plane dipping ~45° NNE (toward 28°), consistent with the dipping strike-slip fault of Shelly et al. (2025, 2026),
whose hypocentral plane is 293°/48° and composite P mechanism 292/34/177.</li>
<li>Right-lateral slip on the 293°/40° plane from the tides (Shelly et al. 2026, Science), now with a converged ocean load. Repeating their
dip-and-rake scan gives the maximum at dip 35°, rake 180°, with 40°/180° close; the left-lateral peak is negligible and appears only when
the ocean load is too strong relative to the body tide, which is the likely origin of their secondary left-lateral peak (unconfirmed without
their supplement).</li>
<li>Strong tidal modulation (Shelly et al. 2026); in our data the M2 tide modulates the rate by ±28% and the response is linear across
constituents (N2/M2 and K1/O1 ratios at their equilibrium values).</li>
</ul>
<h3>Tested and not supported</h3><ul>
<li>Fault healing during the silence (an apparent effect was a detection-threshold artefact).</li>
<li>Source migration at the M7 or since 2020 (common to all families and dependent on the station set), and relative migration between families
faster than ~1 m/yr.</li>
<li>A velocity change at the depth of the LFEs. The coseismic drops scale with local shaking only weakly (slope significant on all 14 surface
stations, but not on the 6 with long records), and the post-2021 decline does not scale with shaking beyond what random pre-2021 windows
produce; where the changes occur is therefore not established. A 60-day early-coda drop after the
M7 survived location and event-mix controls in three families, but across 18 families it is not significant against random-date placebos
(early minus late z = −1.2), is absent in boreholes, and similar drops follow Petrolia and Ferndale. Formal dv/v errors understate
the real scatter by 2–3×.</li>
<li>Detectable aseismic slip during LFE episodes on borehole strainmeters (B045, B933–B935): stacking 132 clean episodes gives nothing
(p = 0.70 vs 32,000 random windows). This only excludes episodes with more than ~180× the LFE moment (Mw 4.4 per episode); the expected
slow slip is ~15× below that bound, so the strainmeters are not sensitive enough to test it. GNSS likewise detects no slip on the LFE zone
(bounds Mw 5.1–5.6; see Newest results).</li>
<li>Fronts sweeping through individual patches (explained by overlapping LFEs pulling locations by ~50 m), tidal shifts in where LFEs occur,
and a restart after the M7 that propagates as a front. An apparent east–west gradient in restart delay was an artefact of normalising by
each family’s background rate; the restart time itself has no along-strike trend.</li>
<li>A single static poroelastic Coulomb model for both the tides and the M7 (see above), and diffusive re-pressurisation as the recovery
mechanism.</li>
<li>Changes across the silence in Vp/Vs (bound ~0.3%) or in slip direction (SH/SV and P/S amplitude ratios), and reversed-polarity (left-lateral)
LFEs during the silence (Vp/Vs and amplitude ratios tested on family 23; the reversed-polarity scan on every family).</li>
<li>Patch “breathing” (size changes with tide, season or the M7) is inconclusive: too few events are well located in both station halves.</li>

</ul></section>''')
H.append("""<section><h2>Newest results (round 4, 23 Nov 2026)</h2>
<div class="findings">
<div class="finding"><span class="eyebrow">Tides</span><h3>Right-lateral slip confirmed; one static pore-pressure model cannot explain both the tides and the M7</h3>
<p>With a converged ocean-tide load at LFE depth (SPOTL with the OSU usawest + TPXO7.2 model, layered Earth; validated on 11 tide gauges and
4 strainmeters), right-lateral slip on the 293°/40° plane wins decisively (no bootstrap replicate prefers left-lateral; robust to 9 ocean models).
Effective friction μ′ = 0.09 ± 0.01 and aσ ≈ 0.63 kPa (with the local rigidity at 29 km). Fitting the tides with the undrained poroelastic
form of Shelly &amp; Materna (2026), CFS = τ + μ(σn − B·mean stress), gives a best fit at small B (0–0.5) and μ ≈ 0.1. Their B ≈ 1, μ = 0.6 is
rejected (24 of 26 families). Yet making the M7 discourage slip needs B ≈ 1 and μ ≳ 0.5, and the tidal-best parameters predict a ~25×
rate increase after the M7 instead of the observed silence. No single static (B, μ) fits both, so the M7 silencing needs a mechanism beyond the
static stress-and-fluid response that governs the tides (for example timescale-dependent fluid behaviour, a permeability change, or a
nonlinear response to a large step).</p></div>
<div class="finding"><span class="eyebrow">Tides</span><h3>Details, and one open issue</h3>
<p>The LFE rate is in phase with stress, not stressing rate (it leads total right-lateral shear by 13 ± 2.5°; 3° with the body tide alone),
shear matters far more than normal stress, and an exponential response beats a linear one. Per constituent, aσ is 0.72 kPa (M2), 0.57 (N2),
0.22 (K1) and 0.23 (O1): the diurnal tides need ~3× more sensitivity than M2. All 26 fitted families are right-lateral with positive shear
sensitivity (median aσ 0.71 kPa), but lag and μ′ increase westward (lag vs longitude ρ = −0.59, p = 0.002); the ocean load does not explain this.
<b>Open issue:</b> the body tide alone fits better than body tide plus the modelled ocean load (the best ocean-load scale is 0; no bootstrap
replicate prefers the ocean load; 11 of 26 families individually fit better with the body tide only). Either the load at depth is overestimated, the body tide is underestimated, or this is an
LFE property. Relative to Shelly et al. (2026, Science), new here are the phase, shear vs normal stress, aσ and μ′, per-family fits, the
explicit ocean-load test and the time-resolved sensitivity (unchanged across the M7).</p></div>
<div class="finding"><span class="eyebrow">Patches</span><h3>Sharpened images: elongated, flat-topped patches without sub-asperities</h3>
<p>Deconvolving the empirical (non-Gaussian) location error makes the images 2–3× sharper than raw densities (resolution ~110 m along strike,
~90 m down dip; two sources 200 m apart are always separated in synthetic tests). 22 families could be imaged. Patches are elongated along
strike and flatter than Gaussian (significant in 5–7 families; binomial p ≤ 0.004): family 23 is a ~550 m × ≤125 m band, families 18 and 20
are similar, and family 8 has two lobes ~250 m apart (two sub-patches or one flat-topped patch). No family shows a ring or separated
sub-asperities, and patch shape does not change at the M7 or between early and late years beyond chance.</p></div>
<div class="finding"><span class="eyebrow">Geodesy</span><h3>No aseismic slip detected with GNSS or strainmeters</h3>
<p>56 GNSS stations within 110 km (NGL daily series) show no slip on the LFE zone during the 20 largest multi-week LFE-rate peaks
(stack p = 0.80; 95% bound Mw 5.1 for the average peak). Regressing the LFE-patch motion on the cumulative excess LFE count over 2009–2026 gives
86 ± 108 m³ of aseismic potency per LFE, so &lt; ~300 m³ (95%), ~40× the seismic potency of an Mw 1.5 LFE: if slip tracks LFE activity, the
aseismic/seismic ratio on the patch is below ~40. There is no LFE-zone signal in the M7 afterslip (bound Mw 5.5–5.6 at 3–90 days, nothing at
the ~18-day restart; the regional postseismic signal is clear). The patch slip rate after 2021 did not change detectably (−27 to +62 mm/yr
across model variants, all |z| ≤ 1.7; 95% limits 1.4–8× the plate rate), and the 2-year patch velocity does not track the LFE rate (r = −0.35,
p = 0.28). Borehole strainmeters bound slip during individual LFE episodes to below Mw 4.4.</p></div>
<div class="finding"><span class="eyebrow">Literature</span><h3>Graded response to the three quakes, not "ceased after each"</h3>
<p>Lu, Ide &amp; Ren (2026) report tremor stopping after each of the 2021, 2022 and 2024 quakes. In the LFE catalogue Petrolia caused only a 1–2-day
dip (4 LFEs in the first day; 30-day count unchanged), Ferndale a sustained halving (no shutdown), and only the M7 a full 7–25-day silence,
consistent with the M7 being the only quake that lowered pore pressure at the LFEs. (Measured against the preceding year, the Petrolia
30-day rate was 1.32×; against the 30 days before, it was unchanged.) The restart after the M7 shows no along-strike order (t50 vs along-strike
position p ≥ 0.38; first-event time vs longitude ρ = 0.04): Lu et al. report east-to-west recovery, Shelly &amp; Materna suggest, if anything, the
opposite. Lu, Ide &amp; Ren’s tidal intrinsic friction of ~0.1 agrees with our tidal fit; Shelly &amp; Materna’s 0.6 is what the M7 requires.
Whether the Petrolia/Ferndale dips are partly detection loss is being tested with our own matched-filter scan (in progress).</p></div>
<div class="finding"><span class="eyebrow">Setting</span><h3>Where the LFEs sit relative to the slab</h3>
<p>In McCrory et al. (2012) and Slab2 the LFEs lie 0.5–3 km above the plate interface. In the newest, receiver-function model (Bloch, Bostock &amp;
Audet 2023), the McCrory interface corresponds to the slab Moho (~30 km) and all 27 families lie inside the low-velocity layered package
above it (3–8 km below its top at ~20 km), the zone Shelly et al. (2026) interpret as material accreted to the southern edge of the Gorda slab.</p></div>
</div>
<div class="finding"><span class="eyebrow">Sources</span><h3>The M7 changed when the patches slip, not how they rupture</h3>
<p>Source spectra of all 27 families were compared within each family on the same channel and instrument epoch (matched-filter band amplitudes,
2–6.5 Hz, amplitude-matched, borehole and surface stations separately, 200 placebo dates). Spectral shape, S-wave duration and size do not change at
the M7, Ferndale or Petrolia (all |z| &lt; 1.7; 7 of 142 tests exceed |z| = 2, as expected by chance): corner frequency is unchanged within about
±20%, i.e. stress drop at fixed moment by less than ~2×. Restart events after the M7 look larger only because post-quake noise lets mainly larger
events be detected; size-matched, their spectra are ordinary. No source change over 18 years that both station groups agree on. Duration barely
grows with size (2–4%), favouring near-constant duration, though self-similar scaling is not excluded. Individual LFEs genuinely differ in spectral
shape (up to ±0.2 log10, reproduced on independent stations). Tentative: LFEs within bursts have slightly shorter S envelopes (−8 ms, z −4).</p></div>
<div class="finding"><span class="eyebrow">Sources</span><h3>Families differ only slightly in spectrum; LFE size is not set by patch size</h3>
<p>Comparing families with each other on the same stations (neighbours closer than 3 km share the path, so the ratio cancels it) shows small real
differences between patches: about 8% in corner frequency and 11% in typical LFE amplitude (borehole and surface stations agree for the corner).
Typical LFE amplitude does not grow with patch size, which rules out LFEs rupturing the whole patch at constant stress drop or constant slip. The
corner frequency tends to drop as the patch gets larger (sign as fc ∝ 1/L in 12 of 13 variants, but p = 0.10). Suggestive only (about 16 variables
tested): families that recur more often have higher-frequency LFEs (ρ = −0.5 to −0.6), hinting that loading rate, not geometry, sets LFE dynamics.
The size range between families is too narrow to tell self-similar from constant-duration scaling.</p></div>
<div class="finding"><span class="eyebrow">Anisotropy</span><h3>Shear-wave splitting did not change; it exposed instrument problems instead</h3>
<p>Monthly S-wave splitting from isolated LFEs at 28 stations (Silver–Chan, 2–8 and 2–5 Hz, bootstrap, half-splits; precision ~1–2 ms in delay and
1–2° in fast direction; an injection test recovers rotations fully and delay changes at about half size). No change at Petrolia, Ferndale or the
M7 (network |z| ≤ 1.5; delay bound ~1–2 ms), no counterpart of the post-2021 dv/v decline, no coherent annual cycle, and no difference between
borehole and surface stations. The method proved a sensitive check of horizontal components: it found instrument problems at B933 (2017–2021, and
from 2024-08), B935 (from 2024-05), BRIC (Dec 2022 to its May 2024 reinstall), RVIT and BJES (until their 2024 reinstalls), KMR and KMPB. Re-running every dv/v claim with these windows excluded (and with free offsets at the reinstalls): the annual cycle, the borehole decline, the decline onset after mid-2021 and the M7 surface step survive; the surface decline is ~20% smaller; the surface/borehole factor of 2 drops to 1.2–1.5×; the 1–4 Hz noise decline at BRIC does not survive an offset at its 2024 reinstall (its 2–8 Hz decline and M7 drop do, also on the vertical alone); Ferndale steps at RVIT, BRIC and BJES cannot be separated from instrument changes that began with that quake. Absolute splitting values are ambiguous (delays near half a period).</p></div>
</div>
<p class="note">In progress: our own
matched-filter scan of continuous data around Petrolia and Ferndale (real rate change vs detection loss).</p>""")
H.append(img('lfespec', 'LFE S-wave spectra. (a) Family 23 on its six best channels: S window vs pre-event noise; the LFE signal clearly exceeds noise only at about 2-7 Hz. (b) Median displacement spectrum of every family on borehole PB.B049 (velocity divided by f, normalised at 2.2 Hz): the falloff over 2-7 Hz lies between Brune shapes with corner 4 and 8 Hz for t* = 0.04 s; the corner frequency trades off with the unknown attenuation, so the absolute corner is not resolved.'))
H.append(img('spectra', 'Source-spectral indices (mid- and high-frequency ratios, low-frequency size proxy, S-envelope width) before and after each quake and in the first weeks after the M7, pooled over 27 families, compared with 200 random placebo dates: no change.'))
H.append(img('famspec', 'Relative source spectra between families (within clusters of neighbouring families): high-frequency falloff and typical amplitude versus patch size, recurrence interval and depth, with permutation nulls.'))
H.append(img('sws', 'Shear-wave splitting through time (delay along the reference fast direction and fast-direction change) per station; shaded windows are instrument problems found by this analysis; dashed lines mark the three quakes.'))
H.append(img('tidal_fits', 'Tidal fits with the converged ocean load (log-likelihood gain over a constant rate). Top left: fit vs effective friction mu\' for several load models (solid right-lateral, dashed left-lateral; right-lateral wins in every model). Top right: fit vs ocean-load scale factor and phase shift; the best fit is at scale 0, i.e. body tide only (the open issue). Bottom left: strike scan at dip 40 deg, right-lateral (dashed line = 293 deg). Bottom right: phase of the LFE rate relative to right-lateral shear per constituent (0 = in phase with right-lateral shear, +-180 = left-lateral).'))
H.append(img('poro', 'Poroelastic test. Left/centre: tidal fit quality over Skempton B and intrinsic friction (yellow = best; the Shelly & Materna point B=1, mu=0.6 is the red star). Right: fraction of families for which the M7 discourages slip. The regions allowed by the tides and required by the M7 do not overlap.'))
H.append(img('diprake', 'Shelly et al. (2026) tidal test repeated with the converged ocean load: the right-lateral optimum on 293° holds; a secondary left-lateral peak appears only when the ocean load is too strong relative to the body tide.'))
H.append(img('patchimg', 'Deconvolved (sharpened) patch image of each family in its fault plane (x along strike, + WNW; y down dip). Cyan: 50% contour of the alternative method; green dashed: split-station 1-sigma ellipse.'))
H.append(img('gnss_q1', 'GNSS vs LFE activity. Top: 15-day LFE rate relative to the 1-year median, with the 20 largest peaks (red; dashed lines = the three quakes). Middle: stacked LFE-patch potency around the peaks vs the 95% placebo band, the stack against random dates (p = 0.80; 95% bound Mw 5.1) and per-peak estimates. Bottom: patch projection vs cumulative excess LFE count, giving 86 +- 108 m3 per LFE (95% bound ~300 m3, ~40x the LFE seismic potency).'))
H.append(img('gnss_q2', 'GNSS after the M7: regional postseismic motion is clear, but no slip on the LFE zone is required beyond the placebo level.'))
H.append(img('gnss_q3', 'GNSS velocity changes after 2021. Top: change of velocity between Ferndale and the M7 (left) and after the M7 (right) relative to 2010-2021 (red box = LFE zone). Bottom left: change of LFE-patch slip rate for each model variant with 95% placebo ranges (green band = +-plate rate 2.5 cm/yr). Bottom right: rolling 2-year patch slip-rate anomaly (blue) vs LFE rate (red); the drop in windows spanning the M7 is postseismic leakage.'))
H.append(img('strain_null', 'Borehole strain stack on 132 clean LFE episodes. Left: strain step during the episode per gauge (B045, B933-B935; areal and shear) vs the 95% null from random windows matched in time of day and M2 phase; red = prediction for 1e4 m3 of right-lateral potency. Right: joint four-station potency estimate (293/40, 28 km) for step and 1-day-ramp variants; orange = median LFE potency per episode, green dotted = slip budget per episode at 2.5 cm/yr.'))
H.append('</section>')
H.append('<section><h2>Main findings</h2><div class="findings">')
H.append(f'''<div class="finding"><span class="eyebrow">Activity</span><h3>The M7.0 is the only quake that silenced the system</h3>
<p>Of 13 M≥5.5 earthquakes within 200 km since 2008, only the M7.0 shut the LFEs down: the 30-day rate of all families
fell to {m7q.ratio:.2f}× normal (no random date in 3,000 reached this), and {int(m7q.fam_top10)} of 27 families had one of
their 10 longest quiet periods since 2009 span it. After Ferndale the rate fell to {fer.ratio:.2f}× without long gaps;
after Petrolia it rose to {pet.ratio:.2f}×.</p></div>''')
H.append(f'''<div class="finding"><span class="eyebrow">dv/v</span><h3>Modest velocity drops at the quakes</h3>
<p>Pooling all families per station (CWIRE layout), surface stations dropped {sf["step_m7_pct"][0]:+.3f} ± {sf["step_m7_pct"][1]:.3f}% at the M7
and boreholes {bh["step_m7_pct"][0]:+.3f} ± {bh["step_m7_pct"][1]:.3f}%. Against random-date placebos the M7 surface drop is
z ≈ −2 to −3, the Ferndale and borehole drops are marginal and Petrolia is not significant (formal errors understate the real scatter
by 1.5–3.5×). The M7 drop scales with measured local shaking across 14 surface stations (−0.12 ± 0.05% per tenfold PGV, p = 0.02),
but not significantly on the 6 stations with long pre-quake records, so near-station damage is plausible but not established. A further 60-day recovering drop follows the M7 ({sf["transient60_pct"][0]:+.3f} ± {sf["transient60_pct"][1]:.3f}% surface,
{bh["transient60_pct"][0]:+.3f} ± {bh["transient60_pct"][1]:.3f}% borehole). It sits in the early coda and does not scale with shaking, so it is the only
possible change near the LFE sources; it survives location and event-mix controls in three families but is not significant across 18 families against random-date placebos.</p></div>''')
H.append('''<div class="finding"><span class="eyebrow">dv/v</span><h3>Stable for 13 years, then a decline since 2021 that has not recovered</h3>
<p>Yearly means (seasonal cycle removed) are flat from 2009 to 2021, then fall by about 0.06% in boreholes and 0.1–0.14% at surface
stations by 2025. This is far outside the 4-year changes seen before 2021 (all within ±0.022%;
z ≈ −7 surface, −6 borehole on unsmoothed monthly data). It starts no earlier than the Petrolia quake; at surface stations it is mostly a
drop at the M7 (−0.10%, z −4.3). It is the same for every LFE family (no family effect, z −2.2), so it is receiver-side, not at the sources;
it is only modestly larger at surface sites (1.2–1.5× once the instrument breaks at BRIC, RVIT and KMPB are given free offsets; the earlier factor of 2 does not hold). It is largest at KMPB, followed by BRIC, RVIT, RBOW and the borehole B045. A rainfall-storage model
consistent with 2009–2021 explains at most a third of it, and no instrument change lines up with it. It is most consistent with a
near-surface change tied to the 2021–24 quakes; whether that is site damage or a regional shallow change is open. Borehole data from the local archive and from NCEDC are byte-identical, so the 2023 data-source switch
is not the cause. <b>Ambient-noise dv/v at the same stations reproduces it</b> (KMPB −0.27%, BRIC −0.17%, B045 −0.32%, control HAYF
−0.01%; monthly r = 0.6–0.75 with the LFE curves), so it is a real crustal change, not an LFE-source or catalog artefact. Noise does not
reproduce the surface-vs-borehole contrast (B045 changes most), as expected because 1–8 Hz noise samples the top ~1 km regardless of sensor depth.</p></div>''')
H.append(f'''<div class="finding"><span class="eyebrow">dv/v</span><h3>An annual cycle every year since 2008</h3>
<p>Velocity is fastest around day {sf["doy"]:.0f}–{bh["doy"]:.0f} (early to mid September). The amplitude is
{sf["ann"]:.3f}% at surface stations and {bh["ann"]:.3f}% in boreholes, so the seasonal signal is mostly shallow.
It reproduces in independent halves of the LFEs.</p></div>''')
H.append(f'''<div class="finding"><span class="eyebrow">Locations</span><h3>27 families located in 3D</h3>
<p>Median horizontal offset from Shelly's locations is {S.loc3d_horiz_offset_from_shelly_km.median():.2f} km. Ours are
{S.loc3d_dz_from_shelly_km.median():.1f} km deeper on median, of which ~2.5 km is our model bias found with earthquakes.
The families sit at two depth levels across ~15 km east–west.</p></div>''')
H.append('''<div class="finding"><span class="eyebrow">Relocation</span><h3>Patches resolved: ~0.5–0.7 km long along the fault strike, thin across it</h3>
<p>Relocating each event twice with two independent halves of the stations separates real spread from error: in families 23, 8
and 27 the true spread is 160–210 m (standard deviation) along Shelly's strike of 293°, with little or none across the plane.
Raw single-event clouds (~1 km) are dominated by the near-vertical error ellipse and should not be read directly.
Bin centroids of 150 repeats are stable to 10–40 m over 18 years, and families move relative to each other by less than ~1 m/yr.
An apparent ~10 m shallowing at the M7 is common to all families, changes with the station set and shrinks to −3.5 ± 3.2 m
after reweighting by signal quality: it is not source motion (it matches a path-averaged velocity drop of ~0.03%).</p></div>''')
H.append(f'''<div class="finding"><span class="eyebrow">Silent period</span><h3>No sign of hidden or reversed LFEs during the silence</h3>
<p>Reverse-polarity detections in the silence ({S.rev_rate_silence_per_day.median():.2f}/day, median over families) occur at
or below the pre-M7 noise rate ({S.rev_rate_pre_per_day.median():.2f}/day) and are mostly incoherent across stations or
coincide with local earthquakes. The scan recovers {S.scan_recovery_pre.median():.0%} of catalog events before the M7.</p></div>''')
H.append('</div></section>')

# ---------------- activity table
H.append('<section><h2>LFE activity after every nearby earthquake</h2>'
         '<p class="lede">All 27 families from Shelly\'s v2 catalog, for every M≥5.5 within 200 km. "Ratio" is the 30-day count after the '
         'quake divided by the rate over the preceding year. "Top-10 gaps" counts families whose quiet period spanning the quake ranks among their '
         '10 longest since 2009.</p><div class="tbl"><table><tr><th>Earthquake</th><th>Distance (km)</th><th>30-day ratio</th>'
         '<th>Top-10 gaps</th><th>Median gap (d)</th><th>Max gap (d)</th></tr>')
for _, r in qt.iterrows():
    H.append(f'<tr><td>{html.escape(r.quake)}</td><td>{r.dist_km:.0f}</td><td>{r.ratio:.2f}</td><td>{int(r.fam_top10)}</td>'
             f'<td>{r.median_gap:.1f}</td><td>{r.max_gap:.1f}</td></tr>')
H.append('</table></div></section>')

# ---------------- dv/v pooled
H.append('<section><h2>dv/v per station, all families pooled</h2>'
         '<p class="lede">The CWIRE layout (Hotovec-Ellis, USGS): one damped least-squares dv/v curve per station-channel, using stack pairs '
         'from every family. Stacks are 50 isolated LFEs (25 in the 6 months after each quake), aligned on direct S and stretched in the 2–8 Hz coda. '
         'Steps and the 60-day transient are fitted directly to the pair measurements. A synthetic −0.05% step was recovered at ~70%, because the '
         'late coda and 4–8 Hz recover only 33–47% (early coda and 2–4 Hz 70–78%), so steps here are if anything underestimated. '
         'The significance chips in these tables use formal errors, which pooled understate the real scatter by 1.5–3.5×; the placebo-calibrated '
         'values are given in the text above.</p>')
H.append('<div class="tbl"><table><tr><th>Stations</th><th>Channels</th><th>Petrolia step</th><th>Ferndale step</th><th>M7 step</th>'
         '<th>M7 60-day transient</th><th>Annual amp.</th><th>Fastest (day)</th></tr>')
for r in pool_rows:
    H.append(f'<tr><td>{r["type"]}</td><td>{r["n"]}</td><td>{sig(*r["step_pet_pct"])}</td><td>{sig(*r["step_fer_pct"])}</td>'
             f'<td>{sig(*r["step_m7_pct"])}</td><td>{sig(*r["transient60_pct"])}</td><td>{r["ann"]:.3f}</td><td>{r["doy"]:.0f}</td></tr>')
H.append('</table></div>')
H.append(img('poolmap', 'M7 step, Ferndale step and annual amplitude per station (median over components); red star marks the LFE families. '
             'SPRL, USAL and HALS start after 2021, so their Petrolia and Ferndale steps cannot be separated.'))
H.append('<div class="tbl"><table><tr><th>Station</th><th>Type</th><th>Distance (km)</th><th>Channels</th><th>Families</th>'
         '<th>M7 step</th><th>M7 transient</th><th>Annual amp.</th><th>Fastest (day)</th></tr>')
for s, r in sta.iterrows():
    H.append(f'<tr><td>{s}</td><td>{r.type.split(" ")[0]}</td><td>{f(r.dist, "{:.0f}")}</td><td>{r.nch}</td><td>{r.nfam:.0f}</td>'
             f'<td>{sig(r.m7, r.m7se)}</td><td>{sig(r.tr, r.trse)}</td><td>{r.ann:.3f}</td><td>{r.doy:.0f}</td></tr>')
H.append('</table></div>')
H.append(img('poolclean', 'Instrument-cleaned curves for the channels with instrument problems found by the shear-wave-splitting analysis (grey: original; colour: cleaned). Excluded windows are shaded red; at sensor reinstalls (B933, BRIC, RVIT, BJES, KMPB maintenance) each instrument segment is inverted separately with its own level, so reinstall jumps cannot appear as velocity changes. The table and map above use the original curves; for these stations use the cleaned values (data/dvv/pool_clean/).'))
H.append(img('poolcurves', 'dv/v curves per station-channel (all families pooled); dashed lines mark the three quakes.'))
H.append(img('longterm', 'Yearly-mean dv/v (seasonal cycle averaged out), median and interquartile range over channels.'))
H.append(img('instr', 'dv/v claims re-run with the instrument-problem windows excluded and with free offsets at sensor reinstalls: the decline, its onset after mid-2021, the M7 surface step and the annual cycle survive; the surface/borehole contrast shrinks to 1.2-1.5x.'))
H.append(img('noise_ts', 'Ambient-noise single-station dv/v (1–4 and 2–8 Hz, 2019–2026) at KMPB, BRIC, B045 and HAYF, compared with the LFE-coda curves; dashed lines mark the three quakes.'))
H.append(img('noise_steps', 'Ambient-noise dv/v steps at Petrolia, Ferndale and the M7, per station and pair type, against the placebo spread.'))
H.append(img('longterm_model', 'Per station: total dv/v change from 2019–20 to 2025 vs the summed measured PGV of the three quakes (left); residual post-2021 ramp after steps and recovery terms (right).'))
H.append('</section>')

# ---------------- patch physics (A x B split-station relocations, burst cascades)
if os.path.exists('data/patches/patch_tensors.csv'):
    PT = pd.read_csv('data/patches/patch_tensors.csv')
    ck = json.load(open('data/cascades/checks.json')) if os.path.exists('data/cascades/checks.json') else {}
    rs = PT[PT.get('resolved', False) == True]  # noqa: E712
    H.append('<section><h2>Patch physics and fault response</h2>'
             '<p class="lede">Every event was relocated twice with two disjoint, azimuth-interleaved halves of the stations. Their errors are '
             'independent, so the covariance of the two locations measures the real extent of the source patch and half the variance of their '
             'difference measures the error. A patch counts as resolved when its long axis exceeds the 95th percentile of a shuffled-pairing null.</p>')
    H.append('<div class="findings">')
    H.append(f'''<div class="finding"><span class="eyebrow">Patch size</span><h3>{len(rs)} of 27 patches resolved, ~560 × 350 m</h3>
<p>Median 5–95% length {rs.L1_m.median():.0f} m and width {rs.L2_m.median():.0f} m (range {rs.L1_m.min():.0f}–{rs.L1_m.max():.0f} m).
Pooling the resolved patches, the long axis is near-horizontal at azimuth 102 deg (68%: 74–110) and the patches lie in a plane striking
286 deg and dipping 58 deg NNE (68%: 28–73 deg), consistent with Shelly's 293/40 deg NE fault. Smaller than Cascadia patches
(1–2 km; Sweet et al. 2014, Chestler &amp; Creager 2017).</p></div>''')
    H.append('''<div class="finding"><span class="eyebrow">Fault geometry</span><h3>The 27 families define a plane dipping ~45 deg NNE</h3>
<p>A plane through our 3D family locations dips 45 deg (68%: 38–52) toward 28 deg; through Shelly's locations, 46 deg toward 24 deg.
Two independent location sets and the within-patch shapes all agree on the NE-dipping strike-slip fault of Shelly et al. (2025, 2026).</p></div>''')
    H.append('''<div class="finding"><span class="eyebrow">Shelly &amp; Materna test</span><h3>M7 cessation time does not track patch size or pre-quake state</h3>
<p>Shelly &amp; Materna suggest the 7–25-day spread in cessation time may reflect patch size or proximity to failure. Across the 11 resolved
families, the gap after the M7 shows no correlation with patch length or area (Spearman p &gt; 0.4), with or without normalising by each
family's rate. Across all families, the time to the first post-M7 LFE does not correlate with patch length (ρ +0.14), area (+0.26), time since
the last pre-M7 LFE (+0.25), the 30-day pre-M7 rate (−0.29), longitude (+0.04) or depth (+0.17); all p &gt; 0.1. The time to the 10th event
simply follows each family's background rate. The size range is narrow (less than 2×), so the size test is weak.</p></div>''')
    H.append('''<div class="finding"><span class="eyebrow">Spatial memory</span><h3>Bursts re-rupture one sub-region</h3>
<p>LFEs less than a minute apart occur at the same spot (memory +0.16 ± 0.06). Events more than a day apart show no memory, and there
is no clear sign that the next event avoids the spot that just slipped. The mean squared separation between events is flat from 1 day
to 10 years, so each patch is fully sampled within minutes at 100–200 m precision.</p></div>''')
    H.append('''<div class="finding"><span class="eyebrow">Restart after the M7</span><h3>No resolvable change in where patches restarted</h3>
<p>The first 20 events after the silence sit +26 ± 11 m down-dip of each family's long-term centroid (15 of 24 families; sign test p = 0.31),
with no along-strike offset. Later tests do not confirm this: the first-60-day centroids of families 23, 8 and 27 are within the placebo
spread, and before/after density maps and deconvolved images show no family-specific shift beyond chance. Restart events are not larger once only
complete (above-threshold) events are used, so there is no sign of fault healing during the hold.</p></div>''')
    H.append('''<div class="finding"><span class="eyebrow">Moment budget</span><h3>The M7 deficit was repaid within ~60–90 days; no healing</h3>
<p>Relative moments from matched-filter S amplitudes on the 15 long-running stations (complete events only): the ~16 days of
missing moment during the silence were made up by a 1.48× moment rate in the first 30 days after restart, bringing the cumulative
moment back to the pre-M7 trend (offset +0.6%, 95% −17 to +21%, at 90 days). Petrolia, with no silence, also raised the rate (1.32×),
so this may be a triggered response rather than a repaid deficit. Healing with hold time is at most ~3.5% per decade.
Assuming a median LFE of Mw 1.5, patch slip rates are ~2–3 cm/yr, comparable to plate rates (uncertain by ~5×). LFE sizes did not change
at the M7 (characteristic amplitude 1.02×, 95% 0.96–1.09). Event counts overshoot by only 1.11× in the first 30 days, so the extra moment is
mostly not in event numbers.</p></div>''')
    H.append('''<div class="finding"><span class="eyebrow">Recovery model</span><h3>An abrupt switch-on at ~18 days, not diffusion</h3>
<p>Per-family fits of the post-M7 rate (rate-and-state negative step, on/off box, slab diffusion, half-space diffusion; compared by
overdispersion-corrected AIC): half-recovery at a median 18 d (9–30 d); the step and box models tie; half-space diffusion is worse in all
27 families because its slow tail is not seen. Recovery time has no relation to along-strike position, depth or distance (|ρ| ≤ 0.18), and
the restart was as synchronous as ordinary quiet gaps (placebo p = 0.69). At fault-zone diffusivities (0.001–1 m²/s) fluids move only
0.08–2.5 km in 18 d. The silence length matches rate-and-state with ~3.5 kPa of effective clamping and loading of ~71 kPa/yr (41–93),
consistent with zone-scale loading; family 23 alone gives aσ ≈ 0.5 kPa (0.08–1.6), compatible with the tides.</p></div>''')
    H.append('''<div class="finding"><span class="eyebrow">Why only the M7</span><h3>It was the only quake that lowered pore pressure at the LFEs</h3>
<p>Static stresses from Okada dislocations for 12 quake dates: Coulomb stress and clamping do not single out the M7 (Ferndale was
larger on both). The undrained pore-pressure change does, in every model variant: the M7 put the LFE zone in its dilatational
quadrant (−2.3 kPa), while Ferndale and the Gorda-slab quakes raised pore pressure. One positive case among 12, so this supports
Shelly &amp; Materna (2026) rather than proving a threshold. Measured shaking (PGV) ranks the rate drops better but has no clean threshold,
and Petrolia (small pore-pressure drop, rate up) breaks a simple dose-response. Across families within the M7, modelled stress does not
explain the cessation times (its range is smaller than the model uncertainty).</p></div>''')
    H.append('''<div class="finding"><span class="eyebrow">Tides and seasons</span><h3>Tidal sensitivity unchanged by the M7; an annual LFE-rate cycle</h3>
<p>The M2 tide modulates LFE rate by ±28% (20 of 27 families), with a linear response across constituents, and this did not change
at any quake (M7 after/before 0.85, 95% 0.47–1.44): a drop of more than ~55% is excluded. With ~3.5 kPa of extra effective clamping
from the M7, that implies an effective normal stress of at least ~6 kPa (rough). LFE rate also varies by ~5–7% over the year
(7.2 ± 0.9% in the main fit, 4.6% in a simpler independent check; 21 of 27 families significant; consistent phase over 17 years),
peaking in late August to mid-September, in phase with the fastest dv/v rather than opposite to it. A detection-threshold cycle (1.1%) cannot
produce it; a shared hydrological driver is speculative.</p></div>''')
    if ck:
        H.append(f'''<div class="finding"><span class="eyebrow">Cascades</span><h3>Bursts sweep the array from both ends at ~80–100 km/day</h3>
<p>{ck['real_frac_sig']:.0%} of {ck['n_episodes']:,} network episodes show significant spatial ordering of burst onsets
(time-shuffled catalogs: {np.mean(ck['shuffle_frac_sig']):.0%}). End-of-array families lead 2–3× more often than chance
(correlation with distance from the array centre r = {ck['leader_ratio_vs_absE']:.2f}); delay grows linearly with distance
(~0.3 h/km), fitting a front better than diffusion. The shallow and deep families start together. In a constant-completeness
catalog (mean CC ≥ 0.13, from 2009) 55% of 2,618 episodes propagate (median 91 km/day) with no trend through time; the rise in the full
catalog is a station artefact. Pairwise triggering gain is ~9× at 10 minutes, decays with distance, is stronger along strike than across,
and did not change after the M7 (p = 0.85).</p></div>''')
    H.append('</div>')
    H.append(img('patches', 'Left: true patch extent per family (5–95%, drawn 2× larger; blue = resolved). Right: spatial memory between an LFE and the k-th next one, by time gap.'))
    H.append(img('cascades', 'Burst cascades: per-episode permutation p-values (uniform = no propagation), propagation azimuths, apparent speeds through time.'))
    H.append(img('coupling', 'Inter-family coupling. Left: pairwise triggering gain (log2, relative to a null of circular shifts by whole M2 cycles) vs separation at 10 min, 1 h and 6 h. Centre: significant directed lead relations at 1 h on the family map, coloured by depth. Right: median coupling by year; squares and diamonds are the year before and after the M7 (no change).'))
    H.append(img('dose', 'Dose-response for 12 quake dates (13 M>=5.5 within 200 km; the two 2021-12-20 events merged). Rows: all-family 30-day rate ratio, number of families with a top-10 gap, median gap relative to the expected interval. Columns: modelled drivers at the LFE fault (Coulomb stress, shear, normal stress, undrained pore pressure, effective normal stress, octahedral shear, PGV). Red = M7, orange = Ferndale, green = Petrolia; bars = model-variant range. Only the pore-pressure drop sets the M7 apart.'))
    H.append(img('moments', 'Moment budget. Top: cumulative relative moment of all 27 families (events above completeness, fixed long-running stations). Bottom: residual from the pre-quake trend in days of pre-quake rate for Petrolia, Ferndale and the M7; dotted = level if the deficit were never repaid, grey = trend uncertainty. The M7 deficit (16.3 days) is made up within ~60-90 days.'))
    H.append(img('recovery', 'Post-M7 recovery per family (panels ordered WNW to ESE; title gives along-strike position and half-recovery time t50). Grey: daily counts (3-day means); red: rate-and-state negative step; green dashed: slab diffusion; blue dotted: half-space diffusion. dQAIC is relative to a constant rate.'))
    H.append(img('seasonal', 'Annual LFE-rate cycle (network, 2009-2026, M7 silence excluded). Left: monthly rate relative to a 24-month running mean for three event selections with sinusoid fits; dashed grey = detection-threshold proxy; vertical line = day of fastest dv/v. Right: per-family amplitude vs day of maximum (orange = significant).'))
    H.append(img('dyn_trigger', 'Dynamic triggering by teleseismic surface waves. Left: observed/expected LFE burst onsets in the first hour vs dynamic stress measured at a local broadband (grey = 95% matched-control null; red = step fit, threshold 0.63 kPa, pink band = its 68% range; green = power law; orange and purple = rate-and-state predictions). Centre: excess probability of at least one onset in the hour. Right: stress scales at the LFEs; the a-sigma bar is the earlier ~0.5 kPa estimate (0.63 kPa with the converged ocean load).'))
    cols = ['family', 'n', 'L1_m', 'L2_m', 'L1_null95', 'axis1_az', 'std_strike_m', 'std_downdip_m', 'std_normal_m', 'err_h_m', 'err_z_m']
    H.append('<div class="tbl"><table><tr>' + ''.join(f'<th>{c}</th>' for c in
             ['Family', 'Events (both halves)', 'Length (m)', 'Width (m)', 'Null 95% (m)', 'Long-axis azimuth', 'Std along strike (m)',
              'Std down-dip (m)', 'Std normal (m)', 'Error horiz. (m)', 'Error vert. (m)', 'Resolved']) + '</tr>')
    for _, r in PT.dropna(subset=['L1_m']).iterrows():
        H.append('<tr>' + ''.join(f'<td>{f(r[c], "{:.0f}")}</td>' for c in cols) + f'<td>{"yes" if r.resolved else "no"}</td></tr>')
    H.append('</table></div></section>')

# ---------------- locations & clouds
H.append('<section><h2>Where the families are</h2>')
H.append(img('clouds', 'Relative relocations of each family placed at its 3D location (+ = Shelly). Map view, E–W section and N–S section. '
             'Absolute depths are ~2.5 km too deep. The ~1 km raw cloud size is mostly single-event location error; the true patches, measured with independent station halves, are ~0.5–0.7 km along strike.'))
H.append(img('sharpened', 'All families with sharpened (deconvolved) patch images at their 3D locations: (a) map, (b) view onto the 293°/40° fault plane, (c) zoom on the central cluster, (d) cross-section along the dip direction. Slab interfaces: McCrory et al. (2012; black), Slab2 (2018; red dashed) and the receiver-function model of Bloch, Bostock & Audet (2023; blue: top of the low-velocity layer t at ~19-21 km and slab Moho m at ~30 km). In the 2023 model all 27 families lie between t and m (3-8 km below t, 1.5-7 km above m); the McCrory interface coincides with the Bloch slab Moho here. Contours at 20/50/80% of peak; resolution ~110 m along strike, ~90 m down dip; * = flatter than Gaussian.'))
H.append(img('density', 'LFE density on the fault plane per family (strike 293°, dip 40° NE; + along strike = WNW, + down-dip = deeper). Black ellipse = true patch (1σ, split-station A×B; dotted = not resolved); grey circle = typical location error. Every family is a single compact peak; the raw density is the true patch blurred by location error, and the diagonal elongation of several clouds follows the error, not the patch.'))
H.append(img('density_m7', 'Density before (blue) and after (red) the 2024 M7: contours at 50% and 90% of peak. Patch shape and size are unchanged; the small common shift of the centroids is the network-change artefact found earlier.'))
H.append('</section>')

# ---------------- family index table
H.append('<section><h2>All families at a glance</h2><p class="lede">Select a family number to jump to its report.</p>'
         '<div class="tbl"><table><tr><th>Family</th><th>Events</th><th>Depth (km)</th><th>Offset from Shelly (km)</th>'
         '<th>M7 gap (d)</th><th>Gap rank</th><th>30-day ratio after M7</th><th>dv/v channels</th><th>dv/v M7 step</th>'
         '<th>dv/v M7 transient</th><th>Annual amp.</th></tr>')
for F in S.index:
    meas = bool(g(F, 'cw_measurable'))
    H.append(f'<tr><td><a href="#fam-{F}">{F}</a></td><td>{f(g(F, "n_events"), "{:,.0f}")}</td><td>{f(g(F, "loc3d_depth_km"), "{:.1f}")}</td>'
             f'<td>{f(g(F, "loc3d_horiz_offset_from_shelly_km"), "{:.2f}")}</td><td>{f(g(F, "m7_gapfull_d"), "{:.1f}")}</td>'
             f'<td>{f(g(F, "m7_gap_rank"), "{:.0f}")}</td><td>{f(g(F, "m7_ratio30"), "{:.2f}")}</td><td>{f(g(F, "cw_nch"), "{:.0f}")}</td>'
             + (f'<td>{sig(g(F, "cw_step_m7_pct"), g(F, "cw_step_m7_pct_se"))}</td><td>{sig(g(F, "cw_transient60_pct"), g(F, "cw_transient60_pct_se"))}</td>'
                f'<td>{f(g(F, "cw_annual_amp_pct"))}</td>' if meas else '<td colspan="3" class="na">too few clean channels</td>')
             + '</tr>')
H.append('</table></div></section>')

# ---------------- per-family sections
H.append('<section><h2>Family reports</h2><nav class="famnav" aria-label="Families">'
         + ''.join(f'<a href="#fam-{F}">{F}</a>' for F in S.index) + '</nav>')
QN = {'pet': 'Petrolia 2021', 'fer': 'Ferndale 2022', 'm7': 'M7.0 2024'}
for F in S.index:
    H.append(f'<article class="fam" id="fam-{F}"><h3>Family {F}</h3><div class="kv">')
    kv = [('Events (v2)', f'{f(g(F, "n_events"), "{:,.0f}")} · {f(g(F, "rate_per_yr"), "{:.0f}")}/yr'),
          ('3D location', f'{f(g(F, "loc3d_lat"), "{:.3f}")}°N {f(g(F, "loc3d_lon"), "{:.3f}")}°E, {f(g(F, "loc3d_depth_km"), "{:.1f}")} km '
                          f'(± {f(g(F, "loc3d_sd_x_km"), "{:.2f}")}/{f(g(F, "loc3d_sd_z_km"), "{:.2f}")} km)'),
          ('Offset from Shelly', f'{f(g(F, "loc3d_horiz_offset_from_shelly_km"), "{:.2f}")} km horizontal, '
                                 f'{f(g(F, "loc3d_dz_from_shelly_km"), "{:+.1f}")} km depth'),
          ('Picks P / S', f'{f(g(F, "loc3d_n_P"), "{:.0f}")} / {f(g(F, "loc3d_n_S"), "{:.0f}")}'),
          ('Relocated events', f'{f(g(F, "reloc_events_good"), "{:,.0f}")} well constrained of {f(g(F, "reloc_events"), "{:,.0f}")}'),
          ('Bin centroid scatter (E;N;Z m)', f(g(F, 'bin_std_m'), '{}')),
          ('Fixed vs all-station agreement r', f(g(F, 'core_vs_all_r'), '{}'))]
    for k, lab in QN.items():
        kv.append((f'Activity after {lab}', f'gap {f(g(F, f"{k}_gapfull_d"), "{:.1f}")} d (rank {f(g(F, f"{k}_gap_rank"), "{:.0f}")}), '
                                            f'30-day ratio {f(g(F, f"{k}_ratio30"), "{:.2f}")}'))
    kv.append(('Depth shift at M7 (fixed stations)', sig(g(F, 'step_core_m7_dz_m'), g(F, 'step_core_m7_dz_se'),
                                                         fmt='{:+.0f} ± {:.0f}', unit=' m')))
    kv.append(('Silence (M7 to 10th event)', f'{f(g(F, "silence_days"), "{:.1f}")} d; reversed detections '
                                             f'{f(g(F, "rev_rate_silence_per_day"), "{:.2f}")}/d vs {f(g(F, "rev_rate_pre_per_day"), "{:.2f}")}/d before'))
    for pre, lab in (('cw', 'dv/v all channels'), ('cwPB', 'dv/v borehole'), ('cwSurf', 'dv/v surface')):
        if g(F, f'{pre}_measurable'):
            kv.append((f'{lab} ({f(g(F, pre + "_nch"), "{:.0f}")} ch)',
                       f'M7 {sig(g(F, pre + "_step_m7_pct"), g(F, pre + "_step_m7_pct_se"))}<br>'
                       f'transient {sig(g(F, pre + "_transient60_pct"), g(F, pre + "_transient60_pct_se"))}<br>'
                       f'annual {f(g(F, pre + "_annual_amp_pct"))}% (day {f(g(F, pre + "_annual_peak_doy"), "{:.0f}")})'))
        elif g(F, f'{pre}_nch') is not None:
            kv.append((lab, f'<span class="na">not measurable ({f(g(F, pre + "_nch"), "{:.0f}")} clean channels)</span>'))
    for k, v in kv:
        H.append(f'<div><span class="k">{k}</span><span class="v">{v}</span></div>')
    H.append('</div><div class="two">')
    H.append(img(f'fam{F}_reloc', f'Family {F}: relocation of 150-event bins through time, and the well-constrained single events.'))
    H.append(img(f'fam{F}_dvv', f'Family {F}: CWIRE-style dv/v from all channels (band = stack bootstrap; red = parametric fit).'))
    H.append('</div></article>')
H.append('</section>')

# ---------------- methods
H.append('''<section><h2>Methods and caveats</h2><ul>
<li><b>Data.</b> Shelly v2 catalog (27 families, 2008–2026). Waveforms −15 to +45 s around every repeat on NC, BK and PB stations within 65 km
(PB 2008–2022 from the local PNWstore archive, everything else from NCEDC).</li>
<li><b>Locations.</b> Picks on stacked templates, 3D and 1D models, station statics from earthquakes. An earthquake test showed our depths ~2.5 km too deep.</li>
<li><b>Relocation.</b> Cross-correlation differential times of 150-event bins and single events relative to the family master; the fixed-station
runs use 15 stations recording 2008–2026, and an availability-matched subset tests station-outage artefacts. True patch size comes from
relocating every event twice with two azimuth-interleaved, disjoint station halves: the A×B covariance is the real spread, half the variance
of A−B is the error. Of 217,710 catalog LFEs, 113,249 (52%) relocated, 68,333 (31%) with errors below 0.5 km horizontal and 1 km depth.
Formal location errors overstate the error, so they must not be subtracted from raw clouds.</li>
<li><b>dv/v.</b> Isolated repeats (no LFE of any family within ±5 s), stacks of 50, alignment on direct S (removes catalog-time jitter), stretching
about S on a ±1% grid at 200 Hz, all stack pairs within 2 years plus each stack against the family reference. Curves: CWIRE design matrix,
control points every 30 days (10 days after quakes), first- and second-difference damping with breaks at the quakes, damping chosen by
leave-channels-out (or leave-families-out) cross-validation. Hypothesis tests are fitted directly to the measurements.</li>
<li><b>Fault frame.</b> Along-strike axes and receiver faults use strike 293°, dip 40° NE, rake 180° (right-lateral). This is the tidal optimum of
Shelly et al. (2026, Science), used as the receiver by Shelly &amp; Materna (2026, GRL); both papers confirm it. Their hypocentral plane is 293°/48°
and their composite P mechanism 292/34/177; Lu, Ide &amp; Ren’s very-low-frequency moment tensor is steeper (284/76/177). Our data agree: the
family-centroid plane gives strike ~294–298°, the pooled patch shape ~286°, the tidal strike scan peaks at 280–290°; results do not change
between 283° and 298°. Positive along-strike is toward 293° (WNW) and down-dip toward ~23° (NNE) in all analyses except two: the in-patch
dynamics analysis uses the pooled patch axis at 102°, so its positive along-strike points ESE (its west-led/east-led counts are labelled
geographically and are unaffected), and the early-coda source-location test uses 290°/45°.</li>
<li><b>Station detections.</b> A repeat counts as detected at a station when its template correlation exceeds that station’s 99th percentile
on noise windows (1% false alarms). Events have data at a median of ~20 stations but are individually detected at a median of only one
(best stations 15–22% of repeats), so locations, relocations and dv/v all rely on stacks or matched-filter measurements.</li>
<li><b>Tidal analysis.</b> One unit per burst (first event, gap &gt; 1 h, mean CC ≥ 0.137; 43,772 units); nulls from whole-day time shifts and phase
scrambling. Body tide from our own lunar/solar ephemeris (validated to 0.0006° and within 0.5% on 8 constituents) with degree-2 Love
numbers, as plane stress on the 293°/40° fault. Rates are fitted with an exponential (Poisson) model; overdispersion inflates log-likelihoods
~8×, so bootstrap ranges are quoted.</li>
<li><b>Ocean loading at depth.</b> SPOTL 3.3 with the OSU usawest regional model inside TPXO7.2 (best of 9 ocean models on 11 NOAA tide gauges:
M2 misfit 1.8 cm). Stress at the LFEs: a layered finite-element point-load Green’s function (Furlong 1-D model) within 2°, and SPOTL/Farrell
loading with a depth correction beyond; converged to 2–4% (normal and mean stress) and ~10% (shear). Validated on 4 PBO strainmeters (adding
the load cuts the tidal calibration residual from 0.36 to 0.21). A homogeneous half-space, used earlier, never converges. Local modulus
54.6 GPa at 29 km. Poroelastic test: CFS = τ + μ(σn − B·mean stress), tension positive, over a (B, μ) grid. Caveats: 1-D layering, plane-stress
body tide, instantaneous undrained poroelasticity.</li>
<li><b>Dose-response.</b> Okada (1992) dislocations ported to numpy and validated against analytic solutions. Receiver 293°/40° right-lateral;
12 quake dates (13 M≥5.5 within 200 km since 2008). USGS finite faults for the M7 and Ferndale, Wells &amp; Coppersmith rectangles otherwise;
an ensemble over nodal planes, source depth ±5 km, receiver geometry and Skempton B 0.5–0.9. PGV from BSSA14. Absolute stresses are uncertain
by ~2×.</li>
<li><b>Moments.</b> Relative moments from matched-filter projection of each event's S window onto the stack, on 15 long-running stations, with
gain screening; completeness from the noisiest epoch (~15% of events retained). Assumes moment ∝ amplitude.</li>
<li><b>Triggering.</b> 793 M≥6.5 quakes beyond 300 km, controls matched in tidal phase and time of day. A waveform check classifies every catalog
event as real or not (template correlation, polarity and lag consistency vs noise windows). Dynamic stress = μ·PGV/c, with PGV measured at
BK.JCC or BK.HOPS (0.01–0.45 Hz) for 789 quakes.</li>
<li><b>Strainmeters.</b> Gladwin strainmeters B045, B933, B934, B935 (EarthScope, 10-min gauges and barometer, 2008–2026); per window trend, tides
and pressure removed; step and ramp over the 200 largest episodes; null from 32,000 random windows; Okada potency inversion with unit gauge
coupling (probably conservative by up to ~2×).</li>
<li><b>GNSS.</b> NGL IGS20 daily positions at 56 stations within 110 km; per-station trajectory model (offsets, seasonal terms, log and velocity
terms at the three quakes); common mode = daily median of 100 far stations (150–400 km); right-lateral slip on a 16 × 15.6 km patch at
22–32 km; placebo dates for all errors.</li>
<li><b>Ambient-noise dv/v.</b> 2 h per night, 2019–2026, at KMPB, BRIC, B045 and HAYF; one-bit normalisation, whitening and notching of persistent
spectral lines; single-station cross-component and autocorrelations; stretching of the 2–20 s coda (1–4 and 2–8 Hz) against a 2019–Nov 2021
reference; block bootstrap and ~200 placebo dates.</li>
<li><b>dv/v placebo calibration.</b> Steps are compared with fits at 200 random pre-M7 dates, the annual cycle with 100 off-period sinusoids, the
long-term decline with all pre-2021 4-year windows. The pooled curves are smoothed except at quake dates, so step and decline significance is
taken from unsmoothed monthly refits.</li>
<li><b>Patch images.</b> Richardson–Lucy deconvolution of the all-station locations with a per-event kernel: scale from the formal error,
shape from the empirical split-station error, which is non-Gaussian (sharper core, heavier tails; Gaussian kernels give false sharpness).
Iterations by burst-grouped cross-validation. Synthetic tests: two sources 200 m apart always separated; resolution ~110 m along strike,
~85–90 m down dip; a uniform disc returns edge “horns”, so only a central dip &gt; ~0.6 means separate sub-patches. Run with and without
overlapping events.</li>
<li><b>Slab models.</b> McCrory et al. (2012), Slab2 (Hayes et al. 2018) and the receiver-function model of Bloch, Bostock &amp; Audet (2023, G3),
which has control stations at the LFEs. Interpretation of LFE depth relative to “the interface” depends on the model.</li>
<li><b>Shared stations.</b> All families are recorded on the same stations, so per-family dv/v and relocation errors are not independent; the
per-station pooled inversion is the primary dv/v result, and cross-family combinations of relocation shifts should be read as upper bounds on significance.</li>
<li><b>Small families.</b> Families with ~2,000–3,000 repeats (e.g. 9, 11) give too few clean channels for their own dv/v curve; they still contribute to the pooled curves.</li>
<li><b>Related work.</b> All four directly relevant papers have been read in full: Shelly &amp; Materna (2026, GRL); Shelly, Thomas, Materna &amp;
Skoumal (2026, Science; main text, the supplement with their tidal-stress method is not available); Lu &amp; Ide (2026, GRL; tremor across
California); and Lu, Ide &amp; Ren (2026, GRL). Shelly &amp; Materna attribute the post-M7 cessation to clamping plus an undrained pore-pressure
drop (B ≈ 1, intrinsic friction 0.6), not shear stress. The Science paper finds right-lateral slip on a fault dipping ~40° NE at the edge of a
captured slab fragment. Lu &amp; Ide map the same zone as tremor cluster I and its shutdown after the M7. Lu, Ide &amp; Ren find the tremor in
positive Coulomb stress, fit the tides with intrinsic friction ~0.1, report suppression after each of the three M6+ quakes and propose dilatancy,
drainage and recharge with east-to-west recovery. Relative to these, we: test the pore-pressure mechanism on all 13 M≥5.5 quakes (only the M7
lowers pore pressure at the LFEs, and only the M7 silenced them); show a graded response (Petrolia a 1–2-day dip, Ferndale a halving, only the
M7 a cessation); find that the tides favour friction ~0.1 and reject B ≈ 1, μ = 0.6 while the M7 needs them, so no single static model fits
both; find a synchronous restart with no east–west order; find that the 7–25-day spread in cessation time is not explained by patch size, time
since the last LFE, background rate, longitude or depth (all p &gt; 0.1); confirm right-lateral slip with a converged ocean load; and bound
dynamically triggered slip, Shelly &amp; Materna’s main caveat, by the strainmeter and GNSS limits and the repaid moment deficit.</li>
</ul></section>
</main>''')
open(f'{OUT}/index.html', 'w').write('\n'.join(H))
tot = sum(os.path.getsize(p) for p in glob.glob(f'{OUT}/figs/*')) / 1e6
print(f'wrote {OUT}/index.html ({os.path.getsize(OUT + "/index.html") / 1e3:.0f} kB), figures {tot:.1f} MB')
