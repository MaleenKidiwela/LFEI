# Stage 3 - unified space-time dv/v model: plan (2026-11-25, for review)

Goal: one volume model dv/v(x,y,z,t) = sum_k F_k(x,y,z) D_k(t) (+ per-series nuisances) that explains all observed dv/v
series, with every field reported together with its resolution. Deep answers come from data, not priors.

## 1. Route: (A) two-step, made equivalent to the one-step fit
- Step 1 (per series, 234 attribution series: SC 5 bands x 27 stations, LFE coda channels, repeaters, LP network series).
  - Re-fit the observed series y(t) by linear least squares on:
    - the attribution's own selected driver shapes (component_series_main.parquet, unit-amplitude);
    - plus per-quake basis functions rebuilt exactly as in the attribution: H(t - t_q) and H exp(-(t - t_q)/tau) for
      Petrolia, Ferndale and M7, with each series' selected tau and passed through the same 10-day boxcar;
    - plus a constant and instrument-segment offsets.
  - Non-linear parameters (tau_q, hydrology lag/tau, thermal lag) stay fixed at the attribution's selected values.
  - Given those, the amplitudes are linear. Step 1 returns the FULL amplitude vector per series and its covariance:
    - moving-block bootstrap, 120-day blocks;
    - scaled up where the attribution's placebo SD is larger.
  - Statistically this is the same as one joint fit (B): the fields enter only through the amplitudes, and every
    within-series cross-component covariance is carried into step 2.
  - Cross-SERIES correlation (the regional common mode) is added explicitly as a per-dataset x band
    common-mode covariance, from simultaneous placebos (factor model as in stage 2).
  - (B) is not needed unless the non-linear parameters must be shared across series. As a variant: one network tau
    per quake.
- Step 2: all components are inverted JOINTLY on the grid. Data = stacked amplitude vectors (about 10 components x
  234 series). Covariance = block-diagonal per series (K x K cross-component) plus the common-mode term.
  Data-space Gaussian solver: about 2,000 data, about 10 x 9,732 parameters.

## 2. Components D_k (fields F_k)
| Field | Time function D_k(t) |
|---|---|
| ENV_cos, ENV_sin | site environmental component, reduced to its annual cos and sin parts (linear; gives wet- vs dry-season fields). Non-seasonal hydrology residual reported per series, not mapped. |
| drop0_q, level_q (Petrolia, Ferndale, M7) | linear recombination of the permanent and transient amplitudes (drop at t_q; permanent part) |
| ramp21 | post-2021-07 ramp (%/yr) |
| trend | secular trend (%/yr) |
- Nuisances: microseism, dose (non-big-3 quakes), offsets. Thermal and pressure are kept inside ENV (selected in only
  18 and 7 series); a separate thermal field is a variant if at least 15 stations support it.

## 3. Forward kernels (grid of sci_dvv_3d_G.py)
- SC: radial station kernels.
- LFE coda and repeaters: source-receiver kernels, MC-depth-reweighted.
- LP: band series = mean of the pair rows.
- Coda depth handling, as in stage 2:
  - The kernel mass below 3 km is moved to a deep-coda nuisance u_(k, dataset, band) (prior 1 %), so coda does not
    vote on depth below 3 km.
  - Coupling kappa per dataset: run with kappa = 1 and with the stage-2 profile (10 / 5 / 2.5; at the grid edge,
    reported).
- Deep constraints, as linear data on the fields:
  - stage-1 travel-time clean-subset estimates (layer contrast and absolute, box, sphere, slab; 1-D and 3-D
    geometry);
  - each row = the averaging kernel on the grid (ak_maps.npz; exact, from the saved gains); data value and honest SE
    from tt_results.csv, with the jackknife SE folded in (max);
  - mapping: TT M7 / Petrolia / Ferndale -> level_q; TT D -> sum of level steps + 4 x ramp21; TT annual sin/cos -> ENV;
  - they enter only once per volume type (the 1-D geometry is used as a variant, not stacked with 3-D).

## 4. Prior and regularisation (the reviewer's points on the oscillating 3-bin split are addressed here)
- Depth-dependent prior SD, per component scale s_k (set from the median observed amplitude):
  s_k x [1.0, 0.7, 0.4, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] over the layers 0-0.1, ... 31-35, 35+.
  Variants: x0.3 and x3 overall, and FLAT-DEEP (deep prior = shallow prior) to show that the deep is not prior-driven.
- Depth regularisation:
  - vertical smoothness: Gaussian correlation between adjacent layers in log-depth, rho 0.7 (variant 0);
  - a monotone / one-signed-in-depth variant for quake drops (damage decays with depth) via a sign-constrained solve;
  - no lateral correlation in the top 3 layers (cells independent);
  - 10 km below 1 km, 20 km below 10 km.
- What is reported:
  - DEPTH-INTEGRATED quantities: kernel-weighted 0-0.3, 0-1 and 0-3 km means per station / per cell;
  - posterior correlations between depth bins;
  - resolution with the ACTUAL data covariance: R = Cm G^T (G Cm G^T + Cd)^-1 G, diagonal and depth rows;
  - hit count (summed kernel mass) per cell;
  - prior-variant stability.
  - A depth split is called "resolved" only if it keeps its sign and stays within 1 SD under the x0.3 / x3 prior and
    rho = 0 / 0.7 variants, and the resolution diagonal is >= 0.5. Otherwise maps show only the depth-integrated
    value, labelled "depth not resolved".
- Deep layers (>= 3 km) are masked unless the resolution diagonal is >= 0.3 AND posterior SD / prior SD <= 0.7 under the
  FLAT-DEEP prior.

## 5. Products
1. Per-component volumes: mean, posterior SD, resolution diagonal, hit count.
   - /data/wsd02/maleen_data/dvv_3d/stage3/fields.npz (+ NetCDF if xarray is available);
   - summaries in data/sci/dvv_3d/stage3/.
2. Depth-slice maps per component for 0-0.1, 0.1-1, 1-3, 3-10, 10-20 and 20-31 km, masked where resolution is poor,
   plus maps of the depth-integrated values.
3. Predicted dv/v volumes at 2024-11-30, 2024-12-20, 2025-06-01 and 2025-12-01, plus wet (Feb) vs dry (Sep) season.
   - Time functions: network-median tau and the nearest station's environmental phase.
   - Predicted-vs-observed per series: amplitudes (G F vs a) and time series (RMS, R^2 against the attribution FIT).
4. Validation in the stage-3 geometry:
   - point-spread functions (cells at 0.05, 0.5, 2, 6, 15, 25 km under stations and between stations);
   - checkerboards (10 and 20 km, per layer group);
   - the reviewer's leakage tests: shallow-only truths generated with adversarial kernels (pure surface, l* = 5, no
     LVZ, coda 4x near-surface) -> spurious deep fields;
   - deep truths (-0.1 % at 20-31 km) -> recovered fraction with and without the travel-time constraints.
5. Viewer export: coarse JSON / npz with 10-km cells, 12 layers, per component and per date (< 5 MB).

## 6. Run time (8 workers, nice 10)
- Step 1 per-series refits with bootstrap: about 10 min.
- Kernels: SC radial (exists) about 2 min; LFE and repeater source-receiver rows about 20-30 min; LP rows about 10 min.
- Joint inversion with variants: about 10 x 9.7k parameters, about 2k data; data-space solve plus resolution and
  posterior diagonals; about 15 min for all variants.
- Validation: about 20 min. Maps and export: about 10 min.
- Total about 1.5 h. Disk: about 2 GB on wsd02, < 20 MB on /home.

## Revision after the plan review (adopted; replaces sections 1-4 where they conflict)
- M1: Step 1 fits the SAME component set in every series. That set is:
  - base, big-3 step + transient, PGV dose (nuisance), trend, ramp21;
  - ENV = representative hydrology (hyd_exp_rean, network tau_h) + annual and semi-annual sin/cos;
  - microseism where available.
  There is ONE network tau per quake (coordinate descent on the summed normalised SSE); variants tau x0.5 / x2.
  Cross-series and cross-component covariance comes from a JOINT moving-block bootstrap with COMMON calendar
  blocks for all time-series datasets, so the regional common mode and multiple LFE channels per station are
  included. Repeaters use a cluster bootstrap. The covariance is shrunk toward the per-series block diagonal
  (lambda 0.3 / 0.7 / 1.0). Whole-block placebo scaling: the network robust z-RMS of whole-model placebos (quake
  dates and hinge shifted back 5-12 yr, pre-2021-06 data).
- M2: PRIMARY parametrisation = per-station terms. There is no voxel grid. Maps use station symbols. A kriged map is
  drawn only if the variogram supports a correlation length. Everything is labelled "apparent", shown with kappa = 1
  and kappa = profile.
- M3: Two parameters per station and component: 0-0.3 km and 0.3-3 km.
  - Prior variants: x0.3, x3, INVERTED (SD increasing with depth).
  - Depth statement = one-parameter hypotheses (top 0.3 km / uniform 0-3 km / 1-3 km only) compared by dchi2. A
    verdict is given only if the same hypothesis wins by >= 4 for both kappas and both prior scales; else "not
    resolved".
  - Injected truths (uniform 0-3, 1-3 only, increasing with depth, top 0.3, deep 20-31 km), generated with default
    and adversarial kernels, show which verdicts can be recovered.
- M4: No travel-time rows enter the station model. "Deep: bounded by stage 1 only" - the stage-1 clean-subset results
  are printed (one target per line); no deep field is presented. Expected deep-truth recovery in the station terms is
  ~0, and this is stated up front.
- Predictions only at stations (1-3 Hz kernel, apparent). The ENV non-annual variance fraction is reported per series.
