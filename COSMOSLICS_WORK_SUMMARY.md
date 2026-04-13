# cosmoSLICS L1 Pipeline Implementation Summary

## Scope and goal

This work implemented a new cosmoSLICS-focused pipeline in WALE to:

- inspect/audit cosmoSLICS simulation structure,
- split realizations into setup groups,
- compute wavelet `L1` datavectors for tomographic bins **4** and **5** from precomputed mass maps using `wl_stats_torch`,
- export inference-ready `.npz` bundles with metadata and parameter alignment,
- provide a direct theory-vs-simulation comparison utility.


## What was added/changed

## 1) New processing script: `l1_norm_processing_cosmoslics_starlet.py`

Implemented a new root-level script that performs:

- **Layout audit**
  - scans cosmoSLICS directories and map availability,
  - detects missing maps and records structured issues.

- **Realization grouping**
  - supports `--group 1-10`, `--group 11-25`, `--group both`,
  - maps these to `group_1_10` and `group_11_25` labels in outputs.

- **Wavelet L1 extraction**
  - uses `wl_stats_torch.WLStatistics`,
  - computes L1 vectors for selected bins (default `4,5`),
  - supports configurable SNR range, scales, noise sigma, and quality policy.

- **Quality diagnostics**
  - checks non-finite/negative/wiggly vectors using existing `DatavectorDiagnostics`,
  - records per-bin quality arrays in output.

- **Output packaging**
  - writes canonical `.npz` with row metadata + datavectors + issue arrays,
  - writes manifest JSON summarizing counts, inputs, issue preview, and quality summary.


## 2) Parameter alignment improvements

Extended parameter loading in `l1_norm_processing_cosmoslics_starlet.py`:

- now supports `--params-file` in formats:
  - `.npy`, `.npz`, `.dat`, `.txt`, `.csv`
- robustly handles index-based tables:
  - detects ID/index column,
  - sorts by ID,
  - enforces required IDs `0..24` (for 25 cosmologies),
  - raises on missing IDs,
  - explicitly ignores extra IDs with a clear message.

For your provided file:

- input: `/home/tersenov/software/WALE/data/CosmoTable.dat`
- parsed parameter names from header:
  - `Om`, `h`, `w_0`, `sigma_8`, `Oc`


## 3) New comparison utility: `compare_theory_vs_cosmoslics_l1.py`

Added a standalone script to compare simulation L1 bundles vs theory:

- required: `--sim-npz`
- theory source modes (exactly one):
  - `--theory-npz` (separate file), or
  - `--theory-key-prefix` (arrays inside same NPZ)
- selection options:
  - `--bin {4,5}`
  - `--group {all,group_1_10,group_11_25}`
  - `--scale-indices ...`
- outputs:
  - `.npz` diagnostics
  - optional JSON manifest (`--manifest-output`)
- metrics:
  - `rmse`
  - `mean_abs_relative_error`
  - `max_abs_relative_error`
  - `mean_abs_residual`
- strict shape checks with explicit errors (no silent truncation).


## 4) README updates

`README.md` was updated with:

- cosmoSLICS L1 processing notes,
- accepted `--params-file` formats,
- an example command for simulation-vs-theory comparison.


## Simulation structure findings

From audit runs:

- Base path:
  - `/nas/tersenov/cosmoSLICS/cosmoSLICS`
- Expected structure:
  - 25 cosmologies × 2 sets (`_a`, `_f`) × 25 cones.
- One concrete data-gap found and now handled:
  - `02_a_cone15` missing mass maps in bins `[4,5]`.

The script records such cases in:

- `issue_row_key`
- `issue_status`
- `issue_reason`
- `issue_cosmo_label`
- `issue_cone`


## Validation performed

## Code/syntax checks

- `python -m py_compile l1_norm_processing_cosmoslics_starlet.py` ✅
- `python -m py_compile compare_theory_vs_cosmoslics_l1.py` ✅


## Processing smoke tests

Smoke outputs were generated under:

- `results/smoke_cosmoslics/`

Representative successful runs:

- `group_1_10_bins45_max2.npz` + manifest
- `group_11_25_bins45_max2.npz` + manifest
- `group_both_bins45_max2.npz` + manifest
- `with_params_1row.npz` + manifest (with `CosmoTable.dat`)

From `with_params_1row.npz`:

- `params.shape == (1, 5)`
- `param_names == ['Om', 'h', 'w_0', 'sigma_8', 'Oc']`

From `group_both_bins45_max2_manifest.json`:

- `rows_total = 1250`
- `rows_candidate = 1249`
- `rows_missing_maps = 1`
- issue preview identifies `02_a_cone15` missing bins 4/5.


## Comparison smoke test

Ran synthetic simulation/theory comparison and saved:

- `results/smoke_cosmoslics/compare_tmp_main.npz`
- `results/smoke_cosmoslics/compare_tmp_main.json`

Example metrics from the run:

- `rmse = 0.026989272360541917`
- `mean_abs_relative_error = 0.04999994910626806`
- `max_abs_relative_error = 0.04999997861155256`


## Notes on baseline repo checks

During baseline checks, pre-existing environment/repo issues were observed (not introduced by this work):

- `black --check src/` reports multiple files needing reformat,
- `pytest` fails in this environment due to missing `pyccl`.

These were left unchanged.


## Final state

All planned implementation todos for this task set were completed:

- cosmoSLICS L1 processing pipeline,
- parameter alignment integration for your `CosmoTable.dat`,
- theory-vs-simulation comparison utility,
- docs + validation.

