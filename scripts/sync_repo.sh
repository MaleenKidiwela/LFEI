#!/bin/bash
# Copy the reproducible code, docs and small inputs/results from /home/seismic/LFE into the
# GitHub staging clone (/home/seismic/LFEI_repo). Excludes raw data, waveforms, large derived data,
# copyrighted papers, logs and the SPOTL build. Run, review `git status`, then commit/push.
set -euo pipefail
SRC=/home/seismic/LFE
DST=${1:-/home/seismic/LFEI_repo}
mkdir -p "$DST"/{scripts,docs,data/stations,data/locations,data/families,data/catalogs,data/relocations,data/patches,results,tools/spotl}
rsync -a --delete --exclude '__pycache__' --exclude '*.pyc' "$SRC/scripts/" "$DST/scripts/"
# lab notebook, minus the line naming another user of the shared machine
grep -v 'Machine context: user' "$SRC/NOTES.md" > "$DST/docs/NOTES.md"
cp "$SRC/report/index.html" "$DST/docs/report.html" 2>/dev/null || true
rsync -a "$SRC/data/stations/" "$DST/data/stations/"
rsync -a "$SRC/data/locations/" "$DST/data/locations/"
rsync -a "$SRC/data/families/" "$DST/data/families/"
rsync -a "$SRC/data/catalogs/" "$DST/data/catalogs/"
cp "$SRC/data/relocations/station_split_AB.json" "$DST/data/relocations/"
rsync -a "$SRC/data/patches/" "$DST/data/patches/"
# small result summaries (json) from each science investigation
for d in "$SRC"/data/sci/*/; do
  n=$(basename "$d")
  find "$d" -maxdepth 2 -name '*summary*.json' -size -2M | while read -r f; do
    mkdir -p "$DST/results/$n"; cp "$f" "$DST/results/$n/"
  done
done
cp "$SRC/tools/spotl/LFE_INSTALL.txt" "$DST/tools/spotl/" 2>/dev/null || true
echo "synced to $DST"; du -sh "$DST" --exclude=.git
# sharpened patch images: tables and figures
mkdir -p "$DST/results/patch_images" "$DST/docs/figures" "$DST/docs/figs"
for v in ovl_incl ovl_excl; do
  mkdir -p "$DST/results/patch_images/$v"
  cp "$SRC/data/sci/patch_images/$v/"{real_summary.csv,real_error_models.json,time_summary.csv,synth_summary.csv} "$DST/results/patch_images/$v/" 2>/dev/null || true
done
cp "$SRC/data/sci/patch_images/population_structure.csv" "$DST/results/patch_images/"
cp "$SRC"/figures/sci_dvv_instrument_check*.png "$SRC"/figures/lfe_spectra.png "$SRC"/figures/sharpened_patches_all_families.png "$SRC"/figures/cloud_density_*.png "$SRC"/figures/sci_patch_images_*.png "$DST/docs/figures/"
# report figures so docs/report.html renders
rsync -a --delete "$SRC/report/figs/" "$DST/docs/figs/"
echo "added patch images + figures"; du -sh "$DST" --exclude=.git
# per-station-channel dv/v curves (all families pooled) and per-channel parameters
mkdir -p "$DST/results/dvv_pool"; rsync -a --delete "$SRC/data/dvv/pool/" "$DST/results/dvv_pool/"
cp "$SRC"/figures/dvv_pool_channels.png "$SRC"/figures/dvv_pool_map.png "$DST/docs/figures/"
# instrument-cleaned per-station dv/v curves
mkdir -p "$DST/results/dvv_pool_clean"; rsync -a --delete "$SRC/data/dvv/pool_clean/" "$DST/results/dvv_pool_clean/"
cp "$SRC"/figures/dvv_pool_channels_clean.png "$SRC"/figures/sci_family_spectra_*.png "$SRC"/figures/sci_dvv_depth2_*.png "$DST/docs/figures/"
cp "$SRC"/figures/sci_dvv_source_receiver_*.png "$DST/docs/figures/"
cp "$SRC"/figures/sci_scan_quakes_*.png "$DST/docs/figures/"
cp "$SRC"/figures/sci_dvv_deep_shallow_*.png "$DST/docs/figures/"
cp "$SRC"/figures/sci_dvv_source_region_*.png "$DST/docs/figures/"
mkdir -p "$DST/results/dvv_source_region"; cp "$SRC"/data/sci/dvv_source_region/{summary.json,timefit_network.csv,timefit_reccorr.csv,family_terms_sph5.csv,family_terms_sph10.csv,family_terms_Q.json,synthetic_recovery.csv,synthetic_waveform_injection.csv,placebo_repeaters_as_sources.csv} "$DST/results/dvv_source_region/" 2>/dev/null || true
mkdir -p "$DST/results/dvv_deep_shallow"; cp "$SRC"/data/sci/dvv_deep_shallow/{summary.json,deep_estimates.csv,station_directS_coda.csv,shallow_from_directS.csv,swin_station_terms.csv,clock_steps.csv,synthetic_recovery.csv} "$SRC"/data/sci/dvv_deep_shallow/tension_*.csv "$SRC"/data/sci/dvv_deep_shallow/rep/{clusters.csv,rep_fit.csv} "$DST/results/dvv_deep_shallow/" 2>/dev/null || true
mkdir -p "$DST/results/scan_quakes"; cp "$SRC"/data/scan_quakes/summary_three_quakes.csv "$SRC"/data/scan_quakes/*_summary.json "$SRC"/data/scan_quakes/*_family_table.csv "$SRC"/data/scan_quakes/*_popeff.json "$DST/results/scan_quakes/" 2>/dev/null || true
mkdir -p "$DST/results/dvv_source_receiver"; cp "$SRC"/data/sci/dvv_source_receiver/summary.json "$SRC"/data/sci/dvv_source_receiver/{family_table_balanced,regressions_balanced,common_bounds,stress_sensitivity_bounds}.csv "$DST/results/dvv_source_receiver/" 2>/dev/null || true
mkdir -p "$DST/data/maps"; cp "$SRC"/data/maps/coast_ne10m_mtj.csv "$DST/data/maps/"; mkdir -p "$DST/docs/patches3d"; cp "$SRC"/report/patches3d/index.html "$DST/docs/patches3d/index.html"
cp "$SRC"/figures/sci_noise_dvv_jewt_*.png "$SRC"/figures/sci_noise_dvv_bands_quakes.png "$DST/docs/figures/"
mkdir -p "$DST/results/noise_dvv_jewt" "$DST/results/noise_dvv_bands_quakes"
cp "$SRC"/data/sci/noise_dvv_jewt/*.csv "$SRC"/data/sci/noise_dvv_jewt/*.json "$DST/results/noise_dvv_jewt/" 2>/dev/null || true
cp "$SRC"/data/sci/noise_dvv_bands_quakes/steps.csv "$SRC"/data/sci/noise_dvv_bands_quakes/daily.csv "$DST/results/noise_dvv_bands_quakes/" 2>/dev/null || true
cp "$SRC"/figures/sci_noise_dvv_bands_ondisk*.png "$DST/docs/figures/"
mkdir -p "$DST/results/noise_dvv_bands_ondisk"; cp "$SRC"/data/sci/noise_dvv_bands_ondisk/steps.csv "$SRC"/data/sci/noise_dvv_bands_ondisk/daily.csv "$DST/results/noise_dvv_bands_ondisk/" 2>/dev/null || true
# 2026-11-25 deep-change tests: small CSV/JSON (<2 MB, top level only) + figures
for n in noise_lp noise_lp/check2022 noise_dvv_sc_full restart_polarity restart_polarity_delay7d rf_lvz rf_timelapse eq_above extras/codaq extras/quakes2 extras/rf_aniso extras/trigger ducellier lfe_phases; do
  [ -d "$SRC/data/sci/$n" ] || continue
  mkdir -p "$DST/results/$n"
  find "$SRC/data/sci/$n" -maxdepth 1 -type f \( -name '*.csv' -o -name '*.json' \) -size -2M -exec cp {} "$DST/results/$n/" \;
done
cp "$SRC"/data/sci/depth_summary.csv "$DST/results/"
cp "$SRC"/figures/sci_depth_summary.png "$SRC"/figures/sci_noise_lp_*.png "$SRC"/figures/sci_noise_dvv_sc_full_all.png "$SRC"/figures/sci_noise_dvv_sc_5back_all.png "$SRC"/figures/sci_noise_dvv_sc_5back_m7.png "$SRC"/figures/sci_noise_dvv_sc_10back_all.png "$SRC"/figures/sci_noise_dvv_sc_10back_m7.png "$SRC"/figures/sci_restart_polarity*.png \
   "$SRC"/figures/sci_rf_lvz_*.png "$SRC"/figures/sci_eq_above_*.png "$SRC"/figures/sci_lfe_phases_*.png "$SRC"/figures/sci_extras_*.png \
   "$SRC"/figures/sci_ducellier_*.png "$DST/docs/figures/" 2>/dev/null || true
echo "added deep-change tests"; du -sh "$DST" --exclude=.git
