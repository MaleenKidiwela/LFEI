# dv/v attribution (2026-11-25) — plan

Goal: attribute dv/v at every station x band (SC ambient noise), LFE-coda channel, LP network series and
repeater station to causes, with honest uncertainties, so component amplitudes can be mapped to depth later.
Scripts: scripts/sci_dvv_attr*.py. Small outputs: data/sci/dvv_attr/. Bulk/raw: /data/wsd02/maleen_data/dvv_attr/.

## Data sources (checked 2026-11-25; all public, no keys, generic user agent)
- dv/v: SC 10-day backward stacks (primary; 5 bands, 27 sta), SC 30-day (check); LFE-coda data/dvv/pool_clean (monthly,
  regularised curves); LP network_series_10d; repeaters rep_pairs_dvv (pairwise design).
- Instrument windows/breaks: EXCL/BREAKS as in scripts/sci_dvv_3d_data.py / sci_dvv_instrument_check.py (copied, not edited).
- Precip/temperature: NOAA GHCN-Daily stations within ~80 km (ghcnd-stations.txt, per-station CSV), and Open-Meteo
  archive (ERA5/ERA5-Land reanalysis) at each seismic station: daily precip, T2m, soil moisture (0-7, 0-100, 100-255 cm),
  surface pressure, snow depth. Daymet timed out (not used); PRISM needs grid downloads (not used).
- Groundwater: USGS water services (NWIS/OGC API) wells in the bounding box — use if any has a daily/continuous record.
- Sea level: NOAA CO-OPS monthly MSL North Spit 9418767, Crescent City 9419750, Arena Cove 9416841 (optional loading proxy).
- Earthquakes: USGS ComCat M>=3.5 within 250 km (~620 events) 2008-2026; BSSA14 PGV at each station
  (bssa14_pgv from scripts/sci_dose_response.py; measured M7 PGV from data/sci/dvv_depth where available);
  teleseism PGV from data/sci/dyn_trigger (measured at JCC/HOPS).
- Geodesy (scope addition): NGL GNSS (data/sci/gnss near_clean: steps/seasonal/CME removed — need raw too, tenv3 in raw/):
  vertical loading (U) near each station; horizontal areal strain from the network (least-squares gradient of
  E,N within ~40 km); M7 afterslip time curve. PB strainmeters (B045, B933-935) areal strain, daily, tides/baro removed.
  Coseismic static volumetric strain at each station from Okada (finite fault for M7/Ferndale, W&C rectangles otherwise)
  using scripts/sci_dose_response.py.

## Model (per station x band; daily 10-day-back series, all regressors passed through the same 10-day boxcar)
dv/v = c + offsets at BREAKS + Sum_q quake terms + H(t) + T(t) + long-term + beta*log10(microseism power) [+ eta*strain]
- Quakes: big three (Petrolia 2021-12-20, Ferndale 2022-12-20, M7 2024-12-05) free permanent step + transient
  exp(-t/tau) (tau shared, grid). All other quakes: shared dose A*G(PGV) with same temporal shape (G = PGV cm/s, test
  alternative static strain). Steps ignored for PGV < 0.1 cm/s.
- Hydrology variants: (H1) exponential-drainage groundwater proxy (Sens-Schoenfelder & Wegler 2006 style) with tau_h grid;
  (H2) Talwani-type diffusion: Sum (P_i - Pbar) erfc(sqrt(tau_d/(t - t_i))) with tau_d = z^2/4D grid; (H3) soil moisture;
  (H4) GNSS vertical; (H5) GW level if found; empirical annual sinusoid as benchmark (not a cause).
- Thermoelastic: T2m low-passed by exp. filter / lagged, lag grid 0-120 d.
- Long-term: linear trend; hinge (slope change) at 2021-07-01; compare to "quakes only" for the decline.
- Nuisance: log10 microseism power in the matching band (power_B045_EHZ; 1-3/3-5 Hz: none available -> 0.5-1 proxy check).
- Strain: eta * areal strain (GNSS-interpolated or borehole) as alternative hydrology/thermo-driver.

## Fitting and validation
- Variable projection: grid over nonlinear params (tau_heal, tau_h, lag_T), linear LSQ (robust Huber) for amplitudes.
- Errors: moving-block bootstrap of residuals (block 120 d, refit), plus placebo quake dates (random pre-2021 dates)
  for step amplitudes and year-shifted climate (integer-year circular shifts) for hydro/thermal gains.
- Model selection: leave-one-water-year-out CV RMS and BIC with n_eff = N/10 (overlapping stacks); forward selection
  of component groups; variance fraction per component = var(component) and partial-R2 (drop-one).
- Out-of-sample: fit pre-2021-07 (long-record stations: PB, KMPB, KMR, KCPB, B-stations), predict 2021-2026 with
  PGV-scaled quake model calibrated on 2010/2014/2016 quakes + hydro + T; does it predict the decline?
- Output: components.csv (dataset, sta, chan/band, component, amp, sd, units, extra params), per-model scores,
  component time series (parquet on /data/wsd02), figures/sci_dvv_attr_*.png.

## Run time
Downloads ~30-60 min (GHCN ~20 stations, Open-Meteo 27 calls, ComCat 1 call, USGS few). Fits: ~135 series x ~200
grid points x linear LSQ, + bootstrap 200 -> < 1 h on 4 workers (nice 10).

## Status 2026-11-25 ~15:10 (code written, smoke-tested; WAITING FOR CODE REVIEW before production fits)
- Downloads done: ComCat (630 M>=3.5 local), GHCN-Daily (128 stations in box; ~10 used, nearest gauges per station),
  CO-OPS monthly MSL (3 gauges), USGS daily streamflow (6 Eel/Mattole gauges; no continuous groundwater wells exist
  in the box — only sparse periodic field measurements, not used), Open-Meteo ERA5/ERA5-Land at 5 sites (rate-limited:
  ~500 'calls' per 20-yr request; remaining sites GHOP-ERA5Land, KCPC, B046 scheduled ~15:55; rerun `drivers climate` after).
- Geodesy added (coordinator scope addition): GNSS vertical loading (seasonal kept), GNSS areal strain from local
  gradients, GNSS post-2021 strain (log + velocity-break terms), borehole areal strain (high-passed), Okada static
  volumetric strain per station per quake (finite fault M7/Ferndale; W&C rectangles), GNSS M7 afterslip time constant.
- Model selection: core (big3 perm+transient, PGV dose >= 0.5 cm/s, trend, ramp 2021-07) -> forward selection of
  environmental groups by leave-one-water-year-out CV (scored outside big-3 windows) AND BIC(n_eff) -> backward
  elimination of unneeded core groups by BIC; pruned groups reported from add-back fits.
- Production run plan: ~230 series x ~8 s / 10 workers ~ 5 min; then collect, oos, afterslip, figs.

## Revision 2026-11-25 ~15:30 after code review (NO-GO -> fixes M1-M6 + should-fixes; awaiting focused re-check)
- M1 precip: Open-Meteo best_match dropped (inhomogeneous). Primary = GHCN gauges, gaps filled with NASA POWER (MERRA-2)
  x local 2-yr gauge/POWER ratio; reanalysis alternative = POWER precip; T, pressure from POWER; soil = ERA5-Land layers
  + POWER GWETROOT/GWETPROF. gauge/POWER annual ratios now stable (no 2010->2022 drift).
- M2 big-3 (perm, trans) pairs sign-constrained (bounded LSQ); shared tau grid 5-150 d capped at span/3; unconstrained
  opposite-sign part reported as 'delayed post-quake change'.
- M3 core groups always kept; 16-subset BIC model averaging + min-max over dBIC<6; headline SD = full-model bootstrap;
  env selection forward+backward with identical CV+BIC criteria.
- M4 chg_identifiable flag (free offset inside 2021-2025) -> excluded from decline budgets/figures.
- M5 tau profiles with likelihood intervals; 'unconstrained'/bounds instead of grid-edge values.
- M6 dose transient-only for non-big-3 local events (no threshold), hypocentral distance, site factors (measured/BSSA,
  class medians), 30-d aftershock window, log10 PGV variant; pgv vs static-strain test restricted to the 13 modelled events.

## Production 2026-11-25 16:15-16:30 (re-check GO; should-fixes 1-7 applied first)
- 234 series (SC10 162 incl. 0.1-1 Hz check band, LFE 58, REP 9, LP10 5), no errors; CV-all sensitivity (SC) done.
- Outputs: components.csv (for_depth_mapping flag: drop0/level1yr per quake, chg_2125[POST21|TOTAL|ENV] identifiable only,
  annual_amp[ENV|TOTAL]), model_scores.csv, candidate_tests.csv, model_average.csv, permutation_network.csv,
  out_of_sample.csv, afterslip_tau.json; figures/sci_dvv_attr_*.png; bulk results/results_{main,cvall}.pkl,
  component_series_main.parquet.
