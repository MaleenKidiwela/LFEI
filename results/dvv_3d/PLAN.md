# Joint 3-D relative dv/v inversion: plan (2026-11-25, awaiting go-ahead)

Status: the SC noise observables (with season-matched errors) and the kernel code exist and are tested.
`scripts/sci_dvv_3d_data.py` has run and written `data/sci/dvv_3d/obs_*.csv`. Its steps match `sc_steps.csv` `10day_back` to 3e-6 %.
`scripts/sci_dvv_3d_kernels.py` passes its tests. The 2-D diffusion kernel matches the numerical time integral exactly. Its body-kernel depth marginal reproduces `sci_dvv_depth2_kernels.noise_body_kernel`: z50 2.29 vs 2.36 km, top-1-km fraction 0.25 vs 0.24.
The kernel production run was stopped at the user's request. No inversion has been run yet.

## 1. Data inventory (per quantity: M7 step / Petrolia / Ferndale / 2021->2025 decline)

| # | Dataset (source files) | n obs M7 / Pet / Fer / Decl | Error calibration | Instrument windows | Season |
|---|---|---|---|---|---|
| N1 | SC single-station coda, 5 bands 0.1-5 Hz (`sc_10day_back.csv`) | 116 / 75 / 60 / 45 (+50 pre-2021 2016->2020 stability) | Season-matched placebo SD (event date +-30 d in all other years; quake-, window- and break-free) x sqrt(1+1/n_yr). Short records use a pooled band ratio. Inter-band correlation comes from the placebos. | EXCL removed (window +9 d); no statistic spans a break; KMPB decline is a variant | Step minus season-placebo mean (Nov-Dec drift, -0.014 % at 1-3 Hz, -0.04 % at 3-5 Hz). Decline uses calendar-matched days. |
| N2 | LP inter-station coda, 0.05-0.1 / 0.1-0.2 / 0.2-0.3 Hz, 734 pairs (`MTJ_LH/dvv/*.npz` bin10/bin30) | ~1,100 pair-bands (M7; fewer in 2021-22) | Per-pair season-matched placebos, as N1. Pair errors are correlated through shared stations: covariance from placebo realisations (shrunk), or an effective-N inflation. | Apply the same EXCL/BREAKS to LP sites; PB LH months already skipped | As N1 |
| L1 | LFE coda, per channel (`dvv_source_receiver/chanfit_calibrated.csv`, V2-cleaned) | 66 / ~66 / ~66 / 29 | Honest = max(bootstrap, placebo, permutation). Pet/Fer have bootstrap only, so x2, the measured placebo/bootstrap factor at the M7. Junk channels are removed by QC (e.g. BJES seg1). | V2 (EXCL dropped, free offsets at BREAKS2) | Annual terms fitted jointly in the time model |
| L2 | LFE family deviations dS_F (`family_table_balanced.csv`, gauge sum = 0) | 25 each | Honest SE from the table | V2 | As L1 |
| L3 | Direct-S delays per station (`dvv_deep_shallow/station_directS_coda.csv`) | 22 / - / - / 15 | max(bootstrap, placebo) | V2 | Fitted annual terms |
| L4 | Near-source S-P eps, 10-km sphere, receiver-corrected (`dvv_source_region/timefit_reccorr.csv`, `summary.json`) | 1 / - / 1 / 1 | Honest SE (placebo 1.5-2x bootstrap) | V2 | Fitted |
| L5 | LFE-minus-earthquake layer dv/v, 22-31 km (`eq_above/diff_results.csv`, joint) | 1 / 1 / 1 / 1 | Honest SE (+-0.006-0.008 %) | as in that study | Fitted |
| L6 | Repeater coda, deep (~19 km) and shallow (~8 km) clusters x 9 stations (`dvv_deep_shallow/rep/rep_fit.csv`, `clusters.csv`) | 18 / - / - / 18 | Cluster bootstrap x1.5 (heterogeneity: chi2 19/9 at 1.0x) | V2 | Fitted |

- Some of the LFE data are derived estimates: L4 and L5 are already inversions of S-P times, and L3's origin term is removed.
- Each of these enters as a linear functional. It gets its own averaging kernel and its published honest SE.
- For L3, the per-stack origin term becomes one free nuisance offset per quantity.
- None of the derived estimates double-counts L1, because the measurements are different (travel times vs coda stretching).
- The L3 deep estimate in `deep_estimates.csv` is NOT used: the per-station delays already carry that information.

**N3 (new option): inter-station CCFs at 0.1-5 Hz from the 20 Hz archive** (`/data/wsd02/maleen_data/MTJ_20Hz/`, 84,100 station-days, 340 GB)

- Size: 351 pairs and 599k pair-days in total; 130 pairs are under 40 km and 84 under 30 km. The median nearest-neighbour distance is 10 km. Most pair-days fall in 2022-2026.
- Cost of the full run, pairs under 40 km, 9 components, 2 products (0.1-1 Hz and 1-5 Hz):
  - Preprocessing: ~25 CPU-h (decode and whiten 84k station-days).
  - Cross-spectra: ~10-15 CPU-h.
  - Total: about 10-15 h wall at 4 workers, plus ~2 h of I/O.
  - dv/v measurement (stretching on 10-day backward stacks, 2 bands x 2 lapse windows): ~3-5 h.
  - Disk: ~15 GB of daily float16 CCFs, or ~8 GB of 10-day stacks, on wsd02 (238 GB free).
- Pilot: Jun 2023 to Jun 2025, 26 stations, pairs under 40 km. About 5 h wall. It tells us the inter-station coda CC and scatter at 1-5 Hz before we commit.
- Expected gain, between stations:
  - In the 0.3-3 km layers, lateral resolution between stations should improve: SC kernels at 1-5 Hz are ~2-5 km wide, so the gaps are currently unsampled.
  - Ballistic and early-coda surface waves at 0.1-1 Hz would add 1-10 km depth information with well-known kernels, perhaps a 1.5-2x lower posterior SD at 1-6 km in the core.
  - It adds essentially nothing at 10-35 km.
- Without N3: 1-10 km relies on the noisy, partition-dependent SC low bands and the sparse LP 0.2-0.3 Hz data.
- N3 does not change the deep (20-31 km) answer. That answer is set by L3 and L5.
- Recommendation: run the pilot in parallel with stage B. Decide on the full run from the pilot's CC and scatter.

## 2. Forward kernels (all linear: d = integral of K(x) dv/v(x) dV, with the integral of K equal to 1 for stretching data)

- **Velocity model.** Station-specific 1-D profiles: the Furlong et al. (2024) column at each station, to 60 km, plus a shallow LVZ (variants ref / soft / none). Families, clusters and paths use the Furlong 3-D model sampled along the paths.
- **N1, SC coincident source and receiver.**
  - K = sum over lapse time t of w(t) times the sum of two terms:
    - surface part: a_eff(t) * Klat2(rho,t) * Kz_sw(z);
    - body part: (1 - a_eff(t)) * Kbody(rho,z,t | z0).
  - Klat2 is the exact 2-D diffusion kernel (Pacheco & Snieder 2005), (1/2piD) exp(-rho^2/2Dt) K0(rho^2/2Dt), plus a single-scattering share exp(-Ut/l*).
  - Kz_sw is the mean of the Rayleigh and Love coda kernels (thin-layer solver of `sci_dvv_depth2_kernels`).
  - Kbody: half-space with an image source and a buried receiver. It uses the exact 3-D Gaussian time integral plus single scattering.
  - Lapse windows: 20-120 / 10-60 / 5-40 / 3-30 / 1-8 s; w(t) ~ t^0.5.
- **N2, LP inter-station.** Separated source and receiver: the 2-D coda kernel (diffusion plus single scattering, numerical time integral) times the LP depth kernel (`sci_noise_lp_kernels`, regional Furlong 1-D). Lapse windows are those stored in the npz (30-150 s, 22-112 s, 22-82 s).
- **L1, L2, L6: LFE and repeater coda, source to receiver.**
  - The same half-space diffusion and single-scattering formula applies with the source at the family or cluster location and the receiver at the station. The time integral for distinct source and receiver distances a and b has the closed form (a+b)/(4piDab) G(a+b,t).
  - Lapse is the S time plus 3 to ~20 s, at 2-8 Hz.
  - The depth profile is rescaled to the existing LFE kernel (`kernel_mc.csv` via `lfe_body_kernel`, with slowness weighting) so the shallow enhancement from the LVZ is kept.
  - A near-receiver surface-wave share a (0 / 0.25 / 0.5) is a variant. Earlier work found the coda more near-surface sensitive than the kernels (1.7-2.1 s vs ~8 s); the joint fit will test this.
  - L1 uses the family-averaged kernel per channel. L2 uses K_F minus the mean over families of K_F.
- **L3, direct S.**
  - Kernel: the 3-D Fresnel-volume-weighted ray kernel (1-D Furlong+LVZ ray, width sqrt(lambda L), 3 Hz). Units are ms per % (delay = -T * dv/v averaged along the ray).
  - The per-quantity origin nuisance is solved jointly.
  - Surface and borehole stations have separate geometry, so the borehole stations see the near-surface segment differently.
- **L4.** Functional = the S-minus-P path-length-weighted average over the 10-km sphere around the family centroids, as used to derive it.
- **L5.** Functional = the average over the 22-31 km band along the LFE-to-station paths: the near-horizontal path segments, 15-45 km long, toward the stations used.
- **Varied in sensitivity runs:**
  - mean free path l* (default 20 km x 1 Hz/f, clipped to 5-150 km; constant 5 / 20 / 50 km);
  - surface/body partition (default a(t) = 0.3 + 0.7 exp(-t/tau), tau = l*/U, after Obermann et al. 2013/2016; pure surface; pure body; 50/50);
  - LVZ variant (ref / soft / none);
  - LFE-coda surface share;
  - Fresnel width x0.5 / x2.

## 3. Model, prior, solver, weighting

- **Volume.** Box of about 170 x 175 km centred on the LFE centroid (40.05 N, -123.56 W), covering all stations, families and clusters plus 30 km.
- **Lateral cells.** 5 km down to 10 km depth, 10 km below. The LFE centroid sits on a cell corner, so the 20-km injection box is exactly 4x4 (or 2x2) cells.
- **Layers.** 0-0.1, 0.1-0.3, 0.3-1, 1-3, 3-6, 6-10, 10-15, 15-20, 20-25, 25-31, 31-35 km, plus an absorbing layer below 35 km. Each layer also has one exterior parameter, which takes the kernel mass of LP and low-frequency kernels outside the box.
- **Size.** About 9,000 parameters.
- **Prior.** Gaussian, zero mean, the same SD in every layer (0.3 %, so depth placement is data-driven, not prior-driven), lateral exponential correlation of 10 km (20 km below 10 km), no correlation between layers. Variants: SD x3 and /3, correlation length 5 or 20 km.
- **Solver.** Exact data-space Gaussian posterior (N_d up to ~3,000): m = Cm G^T (G Cm G^T + Cd)^-1 d. It gives the posterior SD, diag(R), and layer-average posterior SDs.
- **Weighting.** Each datum is weighted by its honest error. Correlated blocks get full Cd blocks:
  - SC bands within a station: placebo correlation;
  - LP pairs: placebo covariance, shrunk;
  - LFE channels of one station.
- **Avoiding domination by one dataset.**
  - (i) A per-dataset error scale s_k = max(1, sqrt(chi2r_k)), estimated from the first fit and applied once.
  - (ii) The model resolution splits exactly into dataset shares: R = sum over k of (Cm G^T S^-1)[:, k] G_k. Per layer, this shows which dataset drives each part.
  - (iii) Leave-one-dataset-out runs.
  - (iv) chi2r is reported per dataset.

## 4. Validation

- Resolution diag and posterior/prior SD per layer.
- Checkerboards: 10 and 20 km, one layer at a time.
- Layer-uniform injections giving the recovery fraction per layer, and the depth where it falls below 50 %.
- **Injection tests, each run as joint, noise-only (N1+N2), LFE-only (L1-L6), and with/without N3 if it is run:**
  - (a) -0.1 % at 20-31 km only, in a 20-km box around the LFEs: recovered fraction and where it leaks.
  - (b) -0.5 % in the top 300 m beneath single stations: leakage to depth.
  - (c) uniform 0-3 km change.
- Each test is noise-free and also run with 200 noise realisations drawn from Cd.
- Leave-one-dataset-out runs for all four quantities.
- Kernel-assumption runs: l*, partition, LVZ, LFE surface share, Fresnel width, prior.
- Pre-2021 stability inversion (2016->2020) as a null control.

## 5. Expected answer and limits

- **0-0.3 km beneath stations.** Resolved by SC 1-5 Hz, LFE coda, and surface-vs-borehole direct S. This is where the M7 drop and the decline should map.
- **0.3-3 km.** Partly resolved: SC 0.3-1 Hz, borehole sensors, LP 0.2-0.3 Hz.
- **3-20 km.** Poor. Only layer averages are constrained (SC low bands under body partition, LP 0.05-0.2 Hz, repeater deep-vs-shallow). These depths are partition- and l*-dependent.
- **20-31 km near the LFEs.** Constrained as a path or layer average by L5 (+-0.008 %) and L3 (+-0.009 %), and as a near-source average by L4 (+-0.14 %). There is no lateral detail. Expect injection (a) to be recovered well with the joint data and ~0 with noise only.
- **Failure modes.**
  - Kernel misspecification: the coda depth partition, and the known misfit between coda and direct S.
  - The 1-D per-station velocity model and the LVZ.
  - Trade-off between a shallow change under a station and a path change. Surface/borehole pairs and the spread of azimuths and families should break it.
  - Correlated or understated errors; derived functionals carrying their own model assumptions.
  - Season bias: all quakes fall in December. It is corrected with season placebos, but there are few years for the BK stations.
  - Instrument windows remove BRIC, RVIT and BJES from Ferndale and from the decline.
  - Uneven station coverage: the ESE/SW and offshore areas are unsampled.

## 6. Schedule (nice 10, at most 4 workers)

| Stage | Work | Wall time | Disk |
|---|---|---|---|
| A (done) | SC observables, kernel code and tests | - | <1 MB |
| B | Kernels: SC surface waves (27 stations x 3 LVZ, ~40 min), SC body (~10 min), LP inter-station (~20 min), LFE/repeater source-receiver (~30 min), direct-S Fresnel (~5 min), functionals | ~2 h | kbody ~0.4 GB + G matrices ~1 GB on wsd02; <20 MB on /home |
| C | LP per-pair steps and decline with season nulls, and LP covariance (~20 min); LFE observable extraction and QC (~10 min) | ~0.5 h | small |
| D | Inversions: 4 quantities x (joint + 7 leave-one-out + ~10 kernel/prior variants), plus injections and checkerboards | ~3-4 h | ~1 GB wsd02 |
| E | Figures (map slices per layer per event, resolution and per-dataset shares, injections, data fit), summary.json, NOTES entry | ~1 h | ~20 MB |
| F (option) | N3 pilot (~5 h), then the full 20 Hz inter-station run (~15-20 h) and re-inversion | 5 h / +20 h | 8-15 GB wsd02 |

Total without N3: about 7-8 h wall.

## Revision after review (2026-11-25, adopted)
1. PRIMARY = small physically parametrised GLS, not the 9,000-cell Gaussian model (kept for figures only, with a
   depth-dependent prior: 5 % top 100 m, 1 % 0.1-1 km, 0.3 % below; per-station shallow cells, no lateral
   correlation in the top layers). Parameters per quantity: per-station shallow terms 0-0.1, 0.1-1, 1-3 km
   (independent between stations, Gaussian prior 5 / 1 / 0.5 %); regional 1-10 km (1 term; variant: + gradient
   toward the M7 rupture); 10-20 km (1 term); deep target (one at a time): (i) uniform 20-31 km layer,
   (ii) 20-31 km in a 20-km box around the LFEs, (iii) union of 10-km spheres around the families,
   (iv) 5-km fault slab through the families. Deep terms have flat priors. Nuisances: per-source origin and
   dx/dy/dz (eliminated, Schur complement), per-station clock (S = P; Gaussian 3 ms), per-band common mode
   (variant), coda coupling factor per coda dataset (profiled 0.25-4x; variant free per station).
   Reported: deep estimates +- honest SD, covariance with the mean station term, shallow-only vs shallow+deep
   (F-test / BIC), Backus-Gilbert averaging kernels of every deep estimate on the 3-D grid.
2. Travel times are ingested ONCE at the raw level: LFE epoch-stack S and P delays per family x channel
   (dvv_source_region/meas, as in eq_above) and earthquake S and P delays (eq_above meas_cat), with
   per-source origin (common to S and P) and location nuisances, free receiver terms per station-phase, and
   the deep target term (extends the validated sci_eq_above_diff.Joint design; imported, not edited).
   L3/L4/L5 summary numbers are no longer used as data (only as cross-checks).
3. Coda data (SC noise, LFE coda, repeater coda) constrain ONLY the station terms <= 3 km (their kernels
   below 3 km are not used); no sqrt(chi2r) rescaling to hide kernel bias (dataset chi2 reported).
4. One time model for all data: annual sin/cos + Petrolia/Ferndale/M7 steps + 60-d windows after each quake
   + ramp from 2021-06 (sci_dvv_source_region.tcols). M7 = step after the 60-d window; D = mean 2025 - mean
   2021 of steps + ramp (dvec). SC noise and LP series are re-fitted with this model (10-day backward
   series dated at window centre; segments at instrument breaks; microseism-power regressor as variant).
5. Errors: SC covariance from simultaneous placebo dates (network correlation per band + inter-band
   correlation); season-matched placebos = whole-year date shifts (5..12 yr) of the whole fit on pre-2021
   data (~8 independent values); TT honest SE = max(bootstrap over sources, placebo). LP reduced to
   band x distance-bin means. L2 (family deviations) dropped: the raw S/P fit subsumes it.
6. Validation without inverse crime: (a) shallow-only truths generated with adversarial kernels/models
   (soft or no LVZ, pure-surface SC, l* = 5 km, coda 4x more near-surface sensitive, frequency-dependent
   shallow change) and inverted with the default; (b) deep truths (-0.1 % at 20-31 km in the 20-km box,
   etc.) injected into the RAW TT rows and passed through the full estimator (origin/location removal);
   (c) station and family jackknives; point-spread functions of the target volumes.
7. N3 (20 Hz inter-station pilot) STOPPED (judged not worth it for this question). Measured cost before the
   stop: 1.3 CPU-s per station-day preprocessing, 0.08 CPU-s per pair-day; full run ~30 CPU-h + dv/v,
   30-40 GB. Revisit only if (a) shows leakage from 1-10 km into the deep term.

## §5 rewritten after the focused re-review (replaces the expected-answer section above)
- The 20-31 km "layer" term from travel times is a CONTRAST: with free receiver terms and per-source origin and
  location nuisances, a uniform change of the whole crust is largely absorbed. The estimator measures 20-31 km
  relative to the overlying crust. A -0.1 % change at 10-20 km leaks +0.06-0.07 % into the layer term, even in
  the clean subset. Both the contrast (layer alone) and the absolute value (layer with R1_10 and R10_20) are
  reported.
- PRIMARY estimator: the clean subset (take-off cos >= 0.2 at the source, no lateral exit of the 3-D sub-grid
  below 20 km), in both geometries (1-D rays; 3-D-traced plus 1-D continuation). Cross-geometry recovery there is
  0.9-1.1, against 0.4-0.5 for all rows. All-rows estimates are secondary, with an explicit x2-3
  amplitude-calibration uncertainty.
- Significance: honest SE = max(formal, bootstrap, formal x placebo z-RMS), with t-quantiles on ~4-7 dof. With
  ~0.05-0.1 % honest SE on the layer, a -0.1 % deep change is NOT testable; the statement is a bound, or a
  detection only if |z| is large and the signal survives the subset, jackknife and phase checks. The number of
  looks is counted.
- 1-10 km and 10-20 km regional terms: honest SE ~0.2-0.4 %. They are noise and are not interpreted.
- Stage 2 (coda + travel-time receiver terms) is used for shallow attribution and the deep-coda nuisance u
  only. Its deep values are conditional and not for inference.
- Long-period 0.05-0.1 Hz data are kept out of the 10-31 km terms unless the ZZ/TT, causal/acausal, azimuth,
  power-regressor and station-cluster checks pass.
