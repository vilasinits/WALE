# Validating the LDT prediction of the wavelet ℓ₁-norm against simulations: methodology and results

**Lead author:** Andreas Tersenov  
**Implementation assistant:** Claude Code (Anthropic Claude Opus 4.7)  
**Project:** WALE — Wavelet ℓ₁-norm Estimator  
**Last revised:** 2026-05-20

## Scope and provenance

This document records, in detail, the full simulation-based-inference (SBI)
validation programme we implemented to test whether the
Large-Deviation-Theory (LDT) prediction of the **Difference-of-Top-Hats
(DoTH) wavelet ℓ₁-norm** reproduces the same cosmological posterior as a
matched simulation-based inference. The programme is intended to feed
directly into a manuscript for the Astronomy & Astrophysics journal. The
note covers everything from the basic data products (cosmoGRID Halofit
suite) up through the headline finding (scale-dependent LDT bias and a
data-driven mitigation strategy).

Code paths quoted below are relative to the WALE repository
(`github.com/vilasinits/WALE`). Where exact line numbers are given, they
refer to the snapshot at the time of writing.

---

## 1. Motivation and physical setting

The wavelet ℓ₁-norm of the weak-lensing convergence κ has emerged as an
attractive higher-order summary statistic: it is simple to compute on a
map (a single histogram weighted by |κ|), it captures information beyond
the two-point function, and it admits an analytic prediction through
Large-Deviation Theory once the convergence PDF has been computed
[Boyle, Uhlemann & Pichon, in prep.; cf. Bernardeau & Valageas 2000,
Codis et al. 2016, Uhlemann et al. 2018, 2023]. The deliverable of the
WALE project is a **theory-based, end-to-end forward model** of the
wavelet ℓ₁-norm that is accurate enough to drive cosmological inference
at the level of a single LSST/Euclid-like survey.

The proposition tested here is sharp:

> *At fixed cosmology and survey configuration, does the LDT prediction
> of the DoTH wavelet ℓ₁-norm produce the same Ωm–σ₈ posterior as a
> simulation-based likelihood-free inference trained on cosmoGRID
> realisations of the same statistic?*

If the answer is yes, then the LDT chain becomes a fully differentiable
forward model that bypasses the cosmoGRID simulation budget. If the
answer is no, we want to characterise *when* and *why* it fails, and
identify the regime in which it remains usable. This is the question
this validation programme answers.

---

## 2. Theoretical framework

### 2.1 The DoTH wavelet filter

We use the *Difference of Top-Hats* wavelet, a scale-localised filter
defined as

$$
W_{\rm DoTH}(\vec{\theta}; \theta_1, \theta_2) \;=\;
\frac{1}{\pi\theta_1^2}\,\mathbb{1}_{|\vec{\theta}|<\theta_1}
\;-\;
\frac{1}{\pi\theta_2^2}\,\mathbb{1}_{|\vec{\theta}|<\theta_2}\,,
$$

with $\theta_2 = R\,\theta_1$ and ratio $R=2$ used throughout this work.
The wavelet field is

$$
\kappa_{\rm DoTH}(\vec{\theta}) \;=\; (W_{\rm DoTH}\ast\kappa)(\vec{\theta})
\;=\;
\kappa_{\rm TH}(\vec{\theta};\theta_1) - \kappa_{\rm TH}(\vec{\theta};\theta_2),
$$
i.e. a difference of top-hat-smoothed convergence maps. The wavelet
$\ell_1$-norm we predict and measure is, per κ-bin,

$$
\boxed{\;L_1(\kappa) \;=\; P(\kappa) \cdot |\kappa|\;}
$$
where $P(\kappa)$ is the one-point PDF of the DoTH field $\kappa_{\rm DoTH}$.
Because the bin integral of $L_1(\kappa)\,\mathrm{d}\kappa$ equals the
mean of $|\kappa|$, the L₁-norm is a peak-style summary that
preferentially weights the non-Gaussian tails.

### 2.2 The LDT two-cell rate function

For each cosmology and each smoothing scale we evaluate the LDT
prediction of $P_{\rm LDT}(\kappa)$ at the same κ-grid as the simulations
through a two-cell rate function ($R_1=\theta_1$, $R_2=R\theta_1$):

$$
\Psi^{(2)}(\delta_1,\delta_2)
\;=\;\frac{\nu^2}{2}\,
\begin{pmatrix}\tau_1\\\tau_2\end{pmatrix}^\top
\mathbf{\Sigma}^{-1}(R_1,R_2)
\begin{pmatrix}\tau_1\\\tau_2\end{pmatrix},
$$

with $\tau_i(\delta_i)$ the spherical-collapse mapping from final to
linear contrast, and $\mathbf{\Sigma}$ the linear-theory annulus
covariance. The projected SCGF is then computed by saddle-point along
the lens line-of-sight, locating critical points of the action through a
combined marching-squares + Newton refinement, and the PDF recovered via
a Bromwich inverse Laplace transform. Implementation files:

| Stage | File | Key entry point |
|---|---|---|
| n(z), distances, lensing kernels | `src/wale/InitializeVariables.py` | `InitialiseVariables` |
| linear / non-linear annulus variance | `src/wale/VarianceCalculator.py` | `Variance` |
| filter functions, Hankel transforms | `src/wale/FilterFunctions.py` | `top_hat_filter`, `starlet_filter` |
| rate function and projection | `src/wale/RateFunction.py` | `get_psi_2cell`, `get_phi_projec_2cell`, `get_scaled_cgf` |
| critical-point finder | `src/wale/CriticalPoints.py` | `CriticalPointsFinder`, `find_critical_points_for_cosmo` |
| PDF via inverse Laplace | `src/wale/ComputePDF.py` | `computePDF` |
| ℓ₁-norm assembly | `src/wale/CommonUtils.py` | `get_l1_from_pdf` |
| cosmology grid driver | `src/wale/cosmogrid_fulldv.py` | `run_cosmogrid_fulldv`, `FullDVConfig` |

### 2.3 Variance recalibration ("recal")

The 2-cell LDT predicts the *unsmoothed continuous-field* variance
$\sigma_{\rm LDT}^2(R_1,R_2)$. Real simulation maps incur pixel-window
suppression and other small offsets that depress the measured variance
$\sigma_{\rm sim}^2$. To absorb this constant offset without disturbing
shape we apply a single multiplicative *recalibration*

$$
\text{recal} \;\equiv\; \frac{\sigma_{\rm LDT}^2}{\sigma_{\rm sim}^2},
$$

implemented in `src/wale/cosmogrid_fulldv.py:214` and toggleable via the
new `FullDVConfig.disable_recal` field. Disabling recal is one of the
two control experiments we run (Section 7).

---

## 3. Simulation infrastructure

All numbers in this work use the **cosmoGRID Halofit grid**: 186 unique
cosmologies × 7 permutations = **1299 lightcone realisations** per
smoothing scale, plus 200 realisations of the **fiducial cosmology**
$(Ω_m, σ_8, w_0, H_0, n_s, Ω_b) = (0.26, 0.84, -1.0, 67.36, 0.9649, 0.0493)$.
Lightcones are noiseless and projected onto a HEALPix nside=512 map,
using the redshift distribution of **stage-III tomographic bin 4 only**
(`data/nz/nz_stage3_4_GRID.txt`). The bin-4 restriction is propagated
through every product: every NPZ file in `data/l1/{simulations,theory}/`
carries `tomo_bin == 4` and the suffix `_bin4_` in its filename, and the
NPZ packer (`scripts/sim_processing/pack_l1_npz.py:50`) accepts only
bin 4 in the metadata.

### 3.1 Sim L1 generation

`scripts/sim_processing/l1_norm_processing_halofit.py` reads the
cosmoGRID HDF5 lightcones, applies the DoTH filter directly in Fourier
space (top-hat $\theta_1$ and $\theta_2$ closed-form transfer functions
in `src/wale/FilterFunctions.py`), and emits per-row arrays of

- the DoTH-filtered variance $\sigma_{\kappa}^2$ (no recal),
- the κ-bin histogram (PDF) and matching ℓ₁-norm bin vector.

A critical design choice was to force the histogram to share an
**identical κ-grid across all 1299 rows**. This is achieved by passing
`--kappa-min/--kappa-max --nbins` so that `np.histogram(...,
range=(kmin, kmax), bins=nbins)` always returns the same bin edges.
The downstream theory pipeline relies on this — it broadcasts the
single 1-D κ-grid to every cosmology so that the theory computes
exactly the same observable on the same grid.

### 3.2 Sim NPZ packing

To homogenise the data product (and to enforce post-hoc invariants), the
three companion .npy outputs are packed into a single NPZ per (θ, kind)
pair by `scripts/sim_processing/pack_l1_npz.py`:

```
params (N, 6)
param_names (6,)              # LaTeX-formatted, ordered
selected_indices (N,)         # halofit_cosmo_selection mapping; cosmo_id = idx // 7
kappa_bins (nbins,)           # 1-D shared grid; asserted identical across rows
l1_norms (N, nbins)           # observable
variances (N,)                # DoTH-filtered variance
theta, theta_ratio, tomo_bin  # scalar metadata
```

The packer fails loudly if the κ-rows are not identical across the
input file (max-deviation check at line 84) — this is the safeguard
against silently allowing per-cosmology grids and corrupting the
shared-grid contract.

### 3.3 Scale catalogue

We have processed five smoothing scales for this work, all at $R=2$:

| θ₁ [arcmin] | κ-range used | σ_κ measured | range / σ_κ | shape (halofit / fid) |
|---:|---|---:|---:|---|
| 30  | $[-0.0150, +0.0150]$ | 2.15·10⁻³ | 6.9 σ | (1299, 200) / (200, 200) |
| 40  | $[-0.0140, +0.0140]$ | 1.96·10⁻³ | 7.1 σ | (1299, 200) / (200, 200) |
| 50  | $[-0.0130, +0.0130]$ | 1.82·10⁻³ | 7.1 σ | (1299, 200) / (200, 200) |
| 60  | $[-0.0120, +0.0120]$ | 1.71·10⁻³ | 7.0 σ | (1299, 200) / (200, 200) |
| 100 | $[-0.0080, +0.0080]$ | 1.39·10⁻³ | 5.7 σ | (1299, 200) / (200, 200) |

The ranges were chosen as roughly $\pm 7\sigma_\kappa$ at each scale to
keep the populated bins in a similar relative envelope across θ. At
θ=100, where the field is narrower, the relative envelope is $\sim 5.7\sigma_\kappa$
— still wide enough to capture the full populated range.

---

## 4. Theory pipeline

### 4.1 The LDT engine on cosmoGRID

`scripts/theory_processing/compute_theory_l1_halofit.py` is the CLI
wrapper around `wale.cosmogrid_fulldv.run_cosmogrid_fulldv`. It:

1. Reads the packed sim NPZ from §3.2 and pulls $(\mathrm{params},
   \mathrm{kappa\_bins}, \mathrm{variances})$ — this gives, per
   cosmology row, the shared κ-grid and the *DoTH-filtered* simulation
   variance that the recal step needs.
2. Broadcasts the shared κ-grid to $N\times n_{\rm bins}$.
3. Calls `run_cosmogrid_fulldv` with `joblib.Parallel(n_jobs=50,
   backend='loky')`. Each worker, per cosmology, executes
   `_compute_one_cosmology`
   (`src/wale/cosmogrid_fulldv.py:147`):
   - builds the lensing kernel from the bin-4 n(z),
   - computes the linear/non-linear 2-cell variance,
   - applies `recal_value = σ²_LDT / σ²_sim` (line 214),
   - locates critical points,
   - computes the PDF via Bromwich on the **passed-in κ grid**,
   - returns the per-bin ℓ₁-norm `pdf * |κ|`.

A new optional flag, `FullDVConfig.disable_recal`, suppresses the
recal step (added at line 214); the CLI exposes this through
`--no-recal`. We use this for the no-recal control experiment of §7.

### 4.2 Output schema

The theory NPZ uses the same schema as the sim NPZ plus three theory-only
fields:

```
pdf_theory   (N, nbins)   # the LDT PDF before |κ| weighting
variance_ldt (N,)         # σ²_LDT before recal
recal_value  (N,)         # σ²_LDT / σ²_sim
```

Sanity checks documented in the plan note that the recal value is
expected to be O(1) (typically 0.8–1.5) for well-behaved cosmologies; we
verified this across all five smoothing scales.

### 4.3 Compute cost

On a 50-CPU node, per smoothing scale, the theory L1 computation
completes in **~40 min** for the full 1299-row grid (38–42 min observed
at θ ∈ {30, 40, 50, 60, 100}). The cost is dominated by the
Bromwich integral plus critical-point search, both of which JIT to
JAX-on-CPU.

---

## 5. Simulation-based inference architecture

### 5.1 Neural Posterior Estimation (NPE)

We use `jaxili`'s standardisation-wrapped Neural Posterior Estimation
(NPE) trained with the default normalising-flow head:

- training set: a stack of ℓ₁ data vectors with their attached
  cosmological parameters,
- standardisation: per-bin z-scoring over the training stack,
- evaluation: 3000 posterior samples drawn at a single fiducial
  observation.

The driver is `scripts/inference/run_npe_inference.py` (the
`--data-key l1_norms --axis-key kappa_bins` invocation; the same
script handles `--data-key cls` via the previously-developed C_ℓ path).
Every job is pinned to GPU 1 via `CUDA_VISIBLE_DEVICES=1` near the
script top.

### 5.2 Covariance-injection pseudo-realisations: apples-to-apples training

The LDT prediction is *deterministic* — there is one theory ℓ₁ vector
per cosmology, not a stochastic distribution like the simulation. To
train an NPE on the theory we must convert it to a likelihood, and to
make sim- and theory-trained NPEs comparable we apply the **same**
likelihood model to both. We use the cov-injected pseudo-realisation
trick:

1. From the 200 fiducial simulation realisations, estimate the full
   $n_{\rm bin}\times n_{\rm bin}$ data-vector covariance $\hat\Sigma_{\rm fid}$.
2. For each cosmology in the grid, take the **mean** ℓ₁ vector
   $\mu_c$ (theory: as-is per row; simulations: averaged over the 7
   perms of each unique cosmology).
3. Draw $n_{\rm draws}$ Gaussian samples
   $\tilde\ell_c \sim \mathcal{N}(\mu_c, \hat\Sigma_{\rm fid})$.
4. Stack these draws across all cosmologies — this is the training set
   for the NPE.

This is implicitly a Gaussian-likelihood inference: the NPE is
recovering the same posterior that an explicit
$\mathcal{N}(\mu_c, \hat\Sigma_{\rm fid})$ likelihood with the same
covariance would. Crucially, the covariance is *identical* between the
sim- and theory-trained NPEs, so any difference in the recovered
posterior is attributable to the mean-prediction difference, i.e. the
shape difference between the LDT and the simulated ℓ₁-norm.

Implementation: `scripts/inference/generate_l1_pseudorealizations.py`
with `--mean-mode {as-is, average-perms}`. For theory, every row is
already a single deterministic mean, so we use `as-is`. For sims, we
group rows by `selected_indices // 7` and average to get a per-cosmology
mean, then draw. Default `n_draws = 7` matches the seven-perm structure
of the original simulation grid.

### 5.3 The fiducial "observation"

The fiducial observation passed to the NPE at evaluation time is
**always** the mean of the 200 sim fiducial realisations. Both the
sim-trained and theory-trained NPEs are evaluated on this same
observation. This isolates the bias to the training-set difference, not
to the choice of observation.

### 5.4 NPE filename convention and cut tags

To organise the swept results, the NPE output stem encodes the cut
applied to the data vector:

```
samples_l1_norms_<sim_stem>_fid_<fid_stem><cut_tag>_npe.npy
```

The cut tag is built by `scripts/inference/run_npe_inference.py:118`
(`_format_cut_tag`) and uses

- `_amin{val}` and `_amax{val}` for axis cuts (κ-range for L₁,
  $\ell$-range for C_ℓ), with the value formatted as `{v:+.4g}` then
  `.` → `p`, `+` → `p`, `-` → `m`. E.g. `-0.0006` becomes `_aminm0p0006`.
- `_tf{thresh}` for transfer-function cuts (C_ℓ only).
- `_dcrm` if `--remove-dc` was set.

Files written by training variants are distinguished by their sim_stem:
`sim_doth_l1_bin4_realizations_fidcov_thetaX.X_ratio2.0` for the sim
training set, `theory_doth_l1_bin4_realizations_fidcov_thetaX.X_ratio2.0`
for theory, and `theory_..._norecal` for the no-recal control.

---

## 6. Diagnostic suite

A reliable validation programme needs diagnostics that distinguish
*where* and *how* the theory disagrees with the simulations. We
developed a five-script suite under `scripts/diagnostics/`:

### 6.1 Per-cosmology mean residuals — `compare_theory_vs_sim_l1.py`

For each of the 186 unique cosmologies, compute the mean ℓ₁ vector over
the 7 perms (sim and theory) and produce three side-by-side panels:

| Panel | y-axis | reference lines |
|---|---|---|
| Fractional | $(L_1^{\rm th}-L_1^{\rm sim})/L_1^{\rm sim}$ | ±2 %, ±5 % |
| σ per-cosmo | $(L_1^{\rm th}-L_1^{\rm sim})/\sigma_{\rm sim}^{\rm 7\,perms}$ | ±2σ, ±3σ |
| σ from fid cov | $(L_1^{\rm th}-L_1^{\rm sim})/\sigma_{\rm fid}$ | ±2σ, ±3σ |

Lines are coloured by $\Omega_m$ (configurable). This is the workhorse
diagnostic — its σ-fid panel is what feeds the cut-design algorithm of
§8.

The script also writes the per-θ fiducial covariance NPZ to
`data/l1/simulations/fid_cov_thetaT.npz` containing
`kappa_bins, cov, sigma_diag`. This is consumed downstream by the
cut-design helper.

### 6.2 Moment-by-moment diagnostic — `compare_theory_vs_sim_moments.py`

This was the diagnostic that *broke open* the analysis. Earlier, an
attempt to plot fractional/σ residuals of the **PDF** rather than the
ℓ₁-norm produced figures that were numerically identical to the ℓ₁
residual plots. The reason is straightforward: $L_1(\kappa) = P(\kappa)|\kappa|$
so

$$
\frac{L_1^{\rm th} - L_1^{\rm sim}}{L_1^{\rm sim}}(\kappa)
\;=\;
\frac{P^{\rm th}(\kappa) - P^{\rm sim}(\kappa)}{P^{\rm sim}(\kappa)},
$$

i.e. the $|\kappa|$ weight cancels mathematically. Bin-wise residuals
therefore say nothing about whether the bias is in the PDF "core" or
its $|\kappa|$-weighting tails. We needed a shape statistic that
integrates across bins. The moments do exactly this:

$$
\langle\kappa\rangle,\;\;
\sigma_\kappa^2,\;\;
S_3 \equiv \frac{\langle(\kappa-\langle\kappa\rangle)^3\rangle}{\sigma_\kappa^3},
\;\;
K_4 \equiv \frac{\langle(\kappa-\langle\kappa\rangle)^4\rangle}{\sigma_\kappa^4} - 3.
$$

Each is computed per cosmology from the bin-integrated PDF, and the
script plots a four-panel scatter (sim moment, theory moment, y=x line)
coloured by $\Omega_m$. The slope of points relative to y=x quantifies
the shape mismatch.

### 6.3 Side-by-side datavectors — `plot_l1_datavectors_overview.py`

For visual sanity: the theory and sim mean ℓ₁ vectors, one curve per
unique cosmology, on shared x- and y-axes. Useful to confirm that the
overall amplitude / κ-scale of the two pipelines agree (i.e. that the
recal step is working) before diving into residuals.

### 6.4 Single-cosmology debug — `plot_l1_single_cosmo_debug.py`

Five representative cosmologies (closest to fiducial, extreme $\Omega_m$,
extreme $\sigma_8$): per-cosmology, the simulation mean ±1σ band over
the 7 perms with the LDT prediction overlaid, plus a fractional-residual
sub-panel. The visual equivalent of "show me the bias on one κ-curve."

### 6.5 PDF datavectors overview — `compare_theory_vs_sim_pdf.py`

Sibling of §6.3 but for $P(\kappa)$ rather than $L_1(\kappa)$. Trimmed
to keep only the side-by-side overview after we confirmed that PDF
residuals are tautological with L₁ residuals.

---

## 7. The bias at θ=30 and its diagnosis

### 7.1 The headline shift

Running the SBI pipeline at θ₁=30′ with the cov-injected training set
for both sim and theory and the 200-realisation sim fiducial as the
observation, the posterior means differ between the sim-trained and
theory-trained NPEs by

$$
\Delta\mu \,/\, \sigma_{\rm sim}\;\big|_{\Omega_m,\sigma_8}^{\rm full\,\kappa}
\;\simeq\;
\big(\!-4.04,\,+3.64\big),
$$

with all other parameters consistent at $\lesssim 0.3\,\sigma$. The
shift saturates the prior at the ~$4\sigma$ level in $\Omega_m$.

### 7.2 Ruling out alternative explanations

We ran two control experiments before accepting "higher-moment shape
mismatch" as the answer.

**Control A — DC-offset removal.** Hypothesis: if the bias is driven by
a per-cosmology overall offset of the ℓ₁ curve (i.e. by $\langle |\kappa|\rangle$
disagreeing with the integral of the LDT prediction), then removing the
per-row mean from each training vector should kill it. We added a
`--remove-dc` flag to `run_npe_inference.py` that subtracts
`train.mean(axis=1, keepdims=True)` from the training set and
$\bar f$ from the fiducial before standardisation. Result: shifts
roughly unchanged (4–9σ across κ-cuts). **The bias is not a DC offset.**

**Control B — recal off.** Hypothesis: the bias might be an artefact of
the variance-recal step ($\sigma_{\rm LDT}^2 \to \sigma_{\rm sim}^2$);
removing recal should reveal whether the underlying LDT prediction is
correct in shape but only in normalisation. We re-ran the full theory
pipeline with `FullDVConfig(disable_recal=True)`. Result: bias slightly
*worse* at the full κ-range (4.04 → 5.98σ on $\Omega_m$), nearly
unchanged at tighter cuts. **Recal was a small second-order correction;
the dominant effect is shape, not normalisation.**

### 7.3 The smoking gun: moment ratios

Both controls pointed at shape. Computing the moments per cosmology
across 186 cosmologies at θ=30 gives the ratio
(theory mean / sim mean):

| Moment | Sim mean | Theory mean | Th/Sim ratio |
|---|---:|---:|---:|
| $\langle\kappa\rangle$ | ~0 (rounding) | ~0 (rounding) | — |
| $\sigma_\kappa^2$ | $4.642\times10^{-6}$ | $4.640\times10^{-6}$ | **1.00** |
| $S_3(\kappa)$ | $-0.270$ | $-0.182$ | **0.68** |
| $K_4(\kappa)$ | $+0.358$ | $+0.175$ | **0.50** |

Variance matches by construction (the recal does this); skewness and
excess kurtosis are *systematically underpredicted* by the LDT by
30 % and 50 % respectively. The DoTH κ field at $\theta_1=30^\prime$ is
non-Gaussian to a level the leading-order 2-cell LDT does not capture.
The 4–8 σ posterior shift is the NPE reading these higher-moment
differences as a cosmology offset.

---

## 8. Data-driven κ-cut design

### 8.1 Motivation

The natural mitigation, given a shape mismatch concentrated in the
tails, is to cut the κ-range used for inference. But picking the cut
*by eye* is slow and unmotivated. We want an algorithm that

1. **Reads** the simulation–theory residual at each κ-bin,
2. **Quantifies** it in fiducial-1σ units (the natural NPE-bias scale),
3. **Walks** outward from κ=0 to find the largest contiguous κ-range
   in which the residual stays below a target threshold,
4. **Reports** both the symmetric and asymmetric versions of that range
   so we can exploit the skew-induced asymmetry in the LDT residual.

### 8.2 Algorithm and implementation

`scripts/inference/design_l1_kappa_cuts.py` does exactly this. The
residual statistic is

$$
R(k) \;\equiv\;
\mathrm{median}_{\rm cosmo}\,
\frac{\big|\,\langle L_1^{\rm th}\rangle - \langle L_1^{\rm sim}\rangle\,\big|(k)}
     {\sigma_{\rm fid}(k)},
$$

where $\sigma_{\rm fid}(k)$ is the diagonal of the 200-realisation
fiducial covariance and the cosmology mean is the 7-perm average. The
median (rather than the mean) is chosen for robustness — a single
extreme cosmology should not dominate the cut.

The walker (`find_inner_range`) starts at the bin closest to $\kappa=0$
and extends as far left and right as $R(k) < T$ for thresholds
$T \in \{1, 2, 3\}$. The symmetric variant takes
$\min(|k_-|, |k_+|)$ as the half-width. A minimum-bin guardrail
(`--min-bins`, default 5) refuses cuts that would leave too few bins
for the NPE to train on. Bins with $\sigma_{\rm fid}=0$ — i.e. bins
outside the populated range — are treated as automatic failures of the
walker so the suggested ranges never extend beyond data.

Output: a row per (θ, threshold) appended to
`outputs/l1_cuts_design.csv`, a diagnostic plot of $R(k)$ with
threshold lines at $|\kappa|$-cuts in
`outputs/plots/l1/thetaT/diagnostics/kappa_cut_design_thetaT.pdf`, and
a CLI-ready list of cut strings printed to stdout.

### 8.3 The cut catalogue we obtained

| θ | T=1σ sym | T=2σ sym | T=3σ sym | T=1σ asym | T=2σ asym | T=3σ asym |
|---:|---|---|---|---|---|---|
| 30  | — (fail) | tiny (1 bin) | $\pm 0.0004$ (5 bin) | — (fail) | tiny (8 bin) | $[-0.0013,+0.0004]$ (12 bin) |
| 40  | tiny (1 bin) | $\pm 0.0006$ (10 bin) | $\pm 0.0113$ (162 bin) | $[-0.0008,-0.0001]$ (6 bin) | $[-0.0013,+0.0006]$ (15 bin) | $[-0.0139,+0.0113]$ (181 bin) |
| 50  | $\pm 0.0005$ (7 bin) | $\pm 0.0103$ (159 bin) | same as 2σ | $[-0.0008,+0.0005]$ (11 bin) | $[-0.0129,+0.0103]$ (180 bin) | same as 2σ |
| 60  | $\pm 0.0010$ (18 bin) | $\pm 0.0092$ (154 bin) | same as 2σ | $[-0.0015,+0.0010]$ (22 bin) | $[-0.0119,+0.0092]$ (177 bin) | same as 2σ |
| 100 | (broad — bias trivial across range) | | | | | |

The asymmetry — the T=1σ asym range at $\theta=40$ is **all on the
negative-κ side** — reflects that the LDT residual is not symmetric
about $\kappa=0$ when the underlying κ field is skewed. In the
intermediate θ range the negative-tail residual is sometimes the
limiting side and sometimes not, depending on the relative growth of
$|L_1^{\rm th}-L_1^{\rm sim}|(\kappa)$ versus $\sigma_{\rm fid}(\kappa)$.
We surface both variants from the design helper so the user can
pick whichever is informationally richer.

---

## 9. The scale-dependence of LDT bias

### 9.1 Moment ratios as a function of θ

Across our five smoothing scales:

| θ [arcmin] | var ratio | skew ratio | kurt ratio |
|---:|---:|---:|---:|
| 30  | 1.00 | 0.68 | 0.50 |
| 40  | 1.00 | 0.75 | 0.59 |
| 50  | 1.00 | 0.80 | 0.67 |
| 60  | 1.00 | 0.84 | 0.75 |
| 100 | 1.00 | 0.92 | 1.13 |

The variance is matched at all scales (the recal does its job). The
skew and kurtosis ratios march monotonically toward unity as θ grows.
This is consistent with the basic expectation: at larger smoothing
scales the convergence field becomes more Gaussian, the leading-order
LDT prediction tracks it better, and the residual non-Gaussian
information sits in higher-order corrections that are not captured by
the two-cell action.

### 9.2 Posterior bias on (Ωm, σ₈) — full κ range and best κ cut

The max-shift summary computed from the eight NPE samples per θ:

| θ | full κ max-shift | best κ-cut max-shift | best κ-range | bins used |
|---:|---:|---:|---|---:|
| 30  | **4.04 σ** | **4.04 σ** (no usable tight cut) | — | — |
| 40  | **3.41 σ** | **0.04 σ** | sym $[-0.0006, +0.0006]$ | 10 |
| 50  | **2.01 σ** | **0.05 σ** | asym $[-0.0008, +0.0005]$ | 11 |
| 60  | **1.83 σ** | **0.06 σ** | sym $[-0.0010, +0.0010]$ | 18 |
| 100 | **0.45 σ** | **0.23 σ** | sym $[-0.003, +0.003]$ | wider |

This is the headline numerical result.

### 9.3 The scale-summary plot

`scripts/inference/plot_l1_scale_summary.py` produces a two-panel PDF
(`outputs/plots/l1/l1_scale_summary.pdf`):

- **Left.** All per-cut curves, one curve per cut family, plotted
  against θ on a log-y axis. The full κ-range curve and the DC-removal
  curve both stay above 1 σ until θ=100. The narrow data-driven
  T=1–2 σ cut curves all sit below 0.1 σ from θ=40 upward.

- **Right.** A summary view: for each θ, the *best* cut (minimum over
  all available cuts) versus the full κ-range. The "best" cut drops
  from 4 σ at θ=30 to 0.04 σ at θ=40 — a cliff. Each point is annotated
  with the κ-range that achieved it.

### 9.4 Interpretation: where the LDT becomes usable

Taking 1 σ on a parameter posterior mean as the "usable" threshold:

- $\theta \le 30^\prime$: the LDT is **not usable** at any κ-cut we
  could find while keeping more than the 5-bin guardrail.
- $30^\prime < \theta \le 40^\prime$: **usable only with a tight,
  narrow κ-cut** (~10 bins, $|\kappa|<6\times10^{-4}$ ≈ 0.3 σ_κ). Below
  this cut the bias is sub-σ; above it the bias is ~3 σ.
- $40^\prime < \theta \le 60^\prime$: **usable in a broader window**, but
  the broad cut ($T=2\sigma$ on R(k)) still leaves ~1–2 σ of bias. The
  tight T=1σ cut (10–20 bins) is what removes the bias.
- $\theta \ge 100^\prime$: **usable at full κ-range** with sub-σ bias
  even before any cut.

The transition between the "tight cut required" and "broad cut required"
regimes happens between $\theta=40^\prime$ and $\theta=50^\prime$. Above
$\theta=60^\prime$ the field is sufficiently Gaussian that wide cuts
suffice. The data-driven cut design picks this up automatically — the
T=2σ-symmetric cut at θ=50 already covers 159 of 200 bins, where at θ=40
it covers only 10.

---

## 10. Code and data products

The repository state at the end of this work, organised by purpose:

### 10.1 Pipeline scripts (all in `scripts/`)

| Module | File | Role |
|---|---|---|
| sim_processing | `l1_norm_processing_halofit.py` | DoTH-filter sim maps, histogram into shared κ-grid (CPU) |
| sim_processing | `pack_l1_npz.py` | combine the 3 .npy companions into a single packed NPZ; assert κ-row identity |
| sim_processing | `run_l1_intermediate_theta_sims.sh` | batched 50-CPU driver for θ ∈ {40, 50, 60} sim L₁ |
| theory_processing | `compute_theory_l1_halofit.py` | LDT engine CLI, consumes packed sim NPZ |
| theory_processing | `run_theory_l1_intermediate.sh` | batched 50-CPU driver for θ ∈ {40, 50, 60} theory L₁ |
| inference | `generate_l1_pseudorealizations.py` | cov-injected pseudo-realisations, sim or theory |
| inference | `run_npe_inference.py` | NPE training with `--data-key l1_norms`, `--axis-key kappa_bins`, `--axis-min/-max`, `--remove-dc`, GPU pin |
| inference | `run_l1_kappa_cut_sweep.sh` | early sweep driver for θ=30 |
| inference | `run_l1_theta_npe.sh` | parameterised single-θ NPE sweep driver |
| inference | `run_l1_intermediate_theta_sweep.sh` | multi-θ driver reading cuts from CSV |
| inference | `design_l1_kappa_cuts.py` | data-driven κ-cut design (the algorithm of §8.2) |
| inference | `plot_l1_kappa_cut_comparison.py` | per-cut sim-vs-theory contour overlays (getdist) |
| inference | `plot_l1_scale_summary.py` | scale-summary plot (§9.3) |
| diagnostics | `compare_theory_vs_sim_l1.py` | three-panel residuals, writes per-θ fid cov NPZ |
| diagnostics | `compare_theory_vs_sim_moments.py` | the four-moment scatter |
| diagnostics | `compare_theory_vs_sim_pdf.py` | PDF-side datavectors overview |
| diagnostics | `plot_l1_datavectors_overview.py` | sim/theory mean ℓ₁ side-by-side |
| diagnostics | `plot_l1_single_cosmo_debug.py` | five representative cosmologies |

### 10.2 Library

`src/wale/cosmogrid_fulldv.py` is the joblib-parallelised driver that
combines every WALE sub-module into a single per-cosmology call. The
only modification we made to the engine for this work was the
`disable_recal: bool = False` flag on `FullDVConfig` (line ~108) plus
the conditional at line 214 that gates the recal assignment.

### 10.3 Data products

```
data/l1/simulations/
  sim_doth_l1_bin4_thetaT_ratio2.0_nobaryons.npz       # halofit grid (1299, 200)
  sim_doth_l1_bin4_thetaT_ratio2.0_fiducial.npz        # fiducial (200, 200)
  sim_doth_l1_bin4_realizations_fidcov_thetaT_ratio2.0.npz   # sim pseudo-realisations (1302, 200)
  fid_cov_thetaT.npz                                    # diagonal + full cov

data/l1/theory/
  theory_doth_l1_bin4_thetaT_ratio2.0_simbin.npz       # theory on shared grid
  theory_doth_l1_bin4_realizations_fidcov_thetaT_ratio2.0.npz   # theory pseudo-realisations
  theory_doth_l1_bin4_*_norecal.npz                    # the no-recal control products at θ=30
```

T runs over {30, 40, 50, 60, 100}.

### 10.4 Plots

```
outputs/plots/l1/
  thetaT/diagnostics/    # moments, residuals, datavectors overview, single-cosmo debug, cut design
  thetaT/overlays/       # per-cut sim-vs-theory contour comparisons
  thetaT/posteriors/     # individual NPE posteriors
  l1_scale_summary.pdf   # the headline plot (§9.3)
```

### 10.5 Tables

```
outputs/l1_cuts_design.csv    # all cut suggestions per (θ, threshold), sym and asym
```

---

## 11. Reproducibility recipe

To regenerate everything in this study from a clean checkout:

```bash
# 1. Environments
conda env create -f environment.yml         # `wale`
pip install -e .

# 2. Sim L1 at all five scales (≈ 1 hr CPU each at 50 workers, θ=30 to θ=100)
bash scripts/sim_processing/run_l1_intermediate_theta_sims.sh    # θ ∈ {40, 50, 60}
# (θ=30 and θ=100 used the same script with their own ranges)

# 3. Pack into NPZs
python scripts/sim_processing/pack_l1_npz.py --base <raw_basename>.npy ... \
       --output data/l1/simulations/sim_doth_l1_bin4_thetaT_ratio2.0_<kind>.npz

# 4. Theory L1 (≈ 40 min CPU per θ at 50 workers)
bash scripts/theory_processing/run_theory_l1_intermediate.sh

# 5. Diagnostics + cut design (≈ 1 min total per θ)
python scripts/diagnostics/compare_theory_vs_sim_l1.py --sim-npz ... --theory-npz ...
python scripts/diagnostics/compare_theory_vs_sim_moments.py --sim-npz ... --theory-npz ...
python scripts/inference/design_l1_kappa_cuts.py --sim-npz ... --theory-npz ... --fid-cov-npz ...

# 6. Pseudo-realisations
python scripts/inference/generate_l1_pseudorealizations.py --mean-source <theory.npz>  --mean-mode as-is        --cov-source <sim_fid.npz> --output <theory_reals.npz>
python scripts/inference/generate_l1_pseudorealizations.py --mean-source <sim.npz>     --mean-mode average-perms --cov-source <sim_fid.npz> --output <sim_reals.npz>

# 7. NPE sweep on GPU 1 (≈ 10-15 min per θ)
CUTS="full <kmin>:<kmax> ..." bash scripts/inference/run_l1_theta_npe.sh <theta>

# 8. Plot contours and scale summary
conda run -n jaxili python scripts/inference/plot_l1_kappa_cut_comparison.py --theta-tag thetaT
python scripts/inference/plot_l1_scale_summary.py --params Om sigma8
```

---

## 12. Limitations and future directions

- **No shape noise.** All sims are noiseless; in a survey-realistic
  analysis the negative-κ tails will be partially destroyed by noise
  and the LDT might "look" more accurate. Repeating this study with
  shape noise injection is the obvious next step.
- **Single tomographic bin.** We restricted to bin 4 throughout (the
  cleanest, highest-S/N redshift slice in stage-III). Cross-bin or
  lower-bin behaviour may differ; the field at lower z is less
  collapsed and may pose different challenges.
- **No baryons.** This is a non-baryonic Halofit grid. The interplay of
  LDT bias with baryonic feedback (also a higher-moment-shape effect
  on small scales) is not addressed here.
- **Approach A (per-cosmology) covariance was not used.** With 7 perms
  per cosmology the per-cosmo covariance is rank-deficient for a 200-bin
  vector. We use the fidcov approach throughout. If a future
  simulation suite provides ≥30 perms/cosmo, the per-cosmo arm becomes
  feasible and would let us check that the cov-injection assumption is
  not itself biased.
- **Beyond the two-cell action.** The diagnosed failure mode is
  precisely a missing higher-order correction. A three-cell or
  saddle-point-corrected LDT might recover the kurtosis at small
  scales. This is, in our view, the most promising follow-up.

---

## 13. Suggested paper outline

If we use this work as the spine of an A&A submission, the natural
section structure is:

1. **Introduction.** The wavelet ℓ₁-norm as a higher-order summary;
   need for a fast forward model; SBI as the comparison framework.
2. **Theoretical framework.** §2 of this note.
3. **Simulations and observables.** §3 of this note.
4. **The LDT prediction.** §4 of this note.
5. **SBI design with cov-injected pseudo-realisations.** §5.2 is the
   key methodological contribution and warrants its own subsection.
6. **Diagnostics — and why the natural ones fail.** §6.2 is the
   moment-by-moment insight; §6.1 is the workhorse residual.
7. **Results at θ=30: bias and its diagnosis.** §7.
8. **Scale-dependence and data-driven mitigation.** §8 + §9. The
   data-driven κ-cut design is the second methodological contribution.
9. **Discussion and outlook.** §12.

---

## Acknowledgement

This validation programme was carried out interactively with **Claude
Code (Anthropic Claude Opus 4.7)** as the implementation partner. All
scientific decisions, scope choices, control experiments, and
interpretation are due to A. Tersenov; Claude implemented the pipeline,
ran the diagnostics, executed the simulation/theory/SBI jobs, and
drafted this methodology note. The convergence on "higher-moment shape
mismatch" as the cause of the θ=30 bias and the data-driven κ-cut
design were arrived at iteratively in conversation between the two.
