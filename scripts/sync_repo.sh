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
