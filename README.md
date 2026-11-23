# LFEI: Low-frequency earthquakes at the Mendocino triple junction

Code and documentation for an analysis of the 27 low-frequency-earthquake (LFE) families at the southern edge of Cascadia
(Mendocino triple junction, ~40.03–40.09°N, 22–32 km depth), 2008–2026, built on the USGS LFE catalog version 2
(Shelly, 2026, doi:[10.5066/P1ZVHNGV](https://doi.org/10.5066/P1ZVHNGV)).

The lab notebook with every result, test, null and correction is in [`docs/NOTES.md`](docs/NOTES.md); a rendered summary report
is in [`docs/report.html`](docs/report.html).

## What is in this repository

| Path | Contents |
|---|---|
| `scripts/` | all analysis code (Python 3.11; shell drivers) |
| `environment.yml` | conda environment (`conda env create -f environment.yml`) |
| `docs/NOTES.md` | lab notebook: methods, results, null tests, corrections, literature comparison |
| `docs/report.html` | summary report with figures |
| `data/stations/` | station/channel inventories used (NCEDC, IRIS/EarthScope) |
| `data/locations/` | our 3D (and 1D) absolute locations of the 27 families |
| `data/families/`, `data/catalogs/` | Shelly family locations; ComCat M≥5.5 list used for the quake-response analysis |
| `data/relocations/station_split_AB.json` | the two azimuth-interleaved station halves used for split-station (A×B) relocation |
| `data/patches/` | patch tensors, memory, density summaries |
| `results/<analysis>/` | summary JSON of each science analysis |
| `tools/spotl/LFE_INSTALL.txt` | how SPOTL (ocean-tide loading) was built for the tidal-stress model |

Not included (size or licence): raw and continuous waveforms, per-family HDF5 waveform sets, derived intermediate data
(~10 GB), velocity models and travel-time tables, slab models, the SPOTL build, and the papers. Download routes are listed below.

## External data needed to reproduce

- LFE catalog: Shelly (2026), USGS data release, doi:10.5066/P1ZVHNGV → `data/shelly2026_MTJ_v2/LFEcatV2_MTJ_2008-2026.csv`
- Waveforms: NCEDC (NC, BK) and EarthScope/IRIS (PB borehole, strainmeters) FDSN web services. EarthScope access uses a token read at
  runtime from `~/.earthscope/token` (never stored in code).
- 3D velocity model: Furlong et al. (2024, Tectonics) → `data/velocity/furlong2024/`; travel-time tables built with `scripts/traveltimes_3d.py`.
- GNSS: Nevada Geodetic Laboratory daily IGS20 tenv3 series and steps file (fetched by `scripts/sci_gnss.py download`).
- Ocean tides: NOAA CO-OPS harmonic constituents; SPOTL 3.3.0.2 (Agnew) with the OSU usawest + TPXO7.2 models.
- Slab models (for figures): McCrory et al. (2012; USGS DS633), Slab2 Cascadia (Hayes et al., 2018; ScienceBase),
  Bloch, Bostock & Audet (2023; GFZ doi:10.5880/FIDGEO.2023.033) → `data/slab/`.

## Pipeline (in order)

1. **Waveforms and datasets**: `download_windows.py` (per-event windows, −15 to +45 s, stations within 65 km),
   `availability_and_histogram.py`, `build_all_families.py` (one-pass HDF5 builder for all families; identical to `build_dataset.py`).
2. **Templates and absolute locations**: `templates.py`, `traveltimes_3d.py`, `locate_family.py` (3D and 1D, earthquake station statics),
   `validate_location_eq.py`.
3. **Relative relocation**: `relocate_relative.py` (cross-correlation differential times, 2–8 Hz; 150-event bins and single events,
   CC ≥ 0.5 for events, 0.3 for bins; weighted LSQ with Huber IRLS), `run_split_reloc.sh` (split-station A×B relocation),
   `patch_tensors.py` (true patch covariance C = sym⟨(xA−x̄)(xB−x̄)ᵀ⟩, error = cov(A−B)/2), `fault_geometry.py`, `plot_all_clouds.py`,
   `plot_cloud_density.py`, `sci_patch_images.py` (deconvolved patch images), `plot_sharpened_patches.py`.
   Drivers: `run_family.sh`, `run_family_full.sh`, `run_all_families.sh`.
4. **dv/v from repeating LFEs (CWIRE-style)**: `dvv_family.py` (stacks of 50 isolated LFEs aligned on direct S, stretching about S,
   ±1 % grid, 200 Hz; pair measurements), `dvv_cwire.py` (damped least-squares inversion on control points, 1st/2nd-difference damping with
   breaks at quakes, λ by leave-channels-out cross-validation, parametric fit, block bootstrap), `dvv_cwire_pool.py` (all families per
   station channel), `dvv_longterm.py`, `dvv_longterm_model.py`, `sci_dvv_placebo*.py` (random-date placebo calibration),
   `sci_noise_dvv.py` (independent ambient-noise single-station dv/v check).
5. **Science analyses** (`scripts/sci_*.py`, outputs in `data/sci/<name>/`, figures `figures/sci_*`):
   - activity and quakes: `quake_response_catalog.py`, `sci_dose_response.py` (Okada 1992 + undrained pore pressure), `sci_recovery_model.py`
     (rate-and-state), `sci_moments.py`, `sci_decline*.py`, `silent_period.py`, `scan_matched_filter.py`, `check_reversed.py`
   - interaction and migration: `burst_cascades*.py`, `sci_cascades2.py`, `sci_patch_dynamics.py`, `sci_centroid_kinematics.py`
   - tides: `sci_tides.py`, `sci_tidal_stress.py` (body tide), `sci_tidal_stress_ocean.py`, `sci_tidal_ocean_spotl.py` + `sci_tidal_spotl_*`
     (converged ocean load at depth, poroelastic test)
   - triggering: `sci_dyn_trigger.py`, `sci_remote_check.py`
   - geodesy: `sci_strain_stack.py` (PBO strainmeters), `sci_gnss.py`
   - source and path: `sci_transient_source*.py`, `sci_dvv_depth*.py`, `sci_source_spectra.py`, `sci_sws.py`
6. **Report**: `build_report_html.py`.

Most scripts take `--family F` or stage arguments; see each script's docstring. Paths are relative to the project root.

## Main results (details, uncertainties and null tests in `docs/NOTES.md`)

- Only the 2024 Mw 7.0 Mendocino earthquake, of 13 M≥5.5 quakes within 200 km since 2008, silenced all 27 families (7–25 days); it is the
  only one that lowered pore pressure at the LFEs. Petrolia (2021) gave a 1–2-day dip, Ferndale (2022) a ~50 % rate drop.
- Tides: strong modulation, right-lateral slip on the 293°/40° plane confirmed with a converged ocean load (SPOTL + layered Earth);
  μ′ ≈ 0.09, aσ ≈ 0.6 kPa; the undrained poroelastic model (B ≈ 1, μ = 0.6) is rejected by the tides, and no single static (B, μ) explains
  both the tides and the M7 cessation.
- Patches: resolved, ~560 × 350 m, elongated along strike, flat-topped; no sub-asperities above ~150–200 m; no shape change across the M7.
- Interaction: ~9× coupling between families at 10 min, stronger along strike; 31–55 % of episodes propagate at 80–100 km/day.
- dv/v: 18-year record from repeating LFEs; robust annual cycle; a non-recovering velocity decrease since 2021, receiver-side, confirmed
  independently with ambient noise.
- Geodesy: no aseismic slip detectable with strainmeters (< Mw 4.4 per episode) or GNSS (< Mw 5.1–5.6).

## Licence

MIT (see `LICENSE`).
