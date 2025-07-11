# WALE
# WALE

[![CI](https://github.com/vilasinits/WALE/actions/workflows/ci.yml/badge.svg)](https://github.com/vilasinits/WALE/actions) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

WALE ("Wavelet **\(\ell_1\)**-norm Estimator") is a Python-based toolkit for predicting and analyzing the one-point statistics of the wavelet ℓ₁-norm in cosmological density fields. WALE provides both theoretical predictions derived from one-point PDF expansions and direct measurements on simulated or observational data, enabling rigorous comparisons across scales.

## Installation

Install WALE via PyPI:

```bash
pip install wale-estimator
```

Or clone and install from source:

```bash
git clone https://github.com/vilasinits/WALE.git
cd WALE
pip install -e .
```

## Quickstart

Look at the notebooks on how this can be used.

## Citation

If you use WALE in your work, please cite:

```bibtex
@ARTICLE{2024A&A...691A..80S,
  author = {{Sreekanth}, Vilasini Tinnaneri and {Codis}, Sandrine and {Barthelemy}, Alexandre and {Starck}, Jean-Luc},
  title = "{Theoretical wavelet {\ensuremath{\ell}}$_{1}$-norm from one-point probability density function prediction}",
  journal = {Astronomy & Astrophysics},
  volume = {691},
  eid = {A80},
  pages = {A80},
  year = {2024},
  month = nov,
  doi = {10.1051/0004-6361/202450061},
  archivePrefix = {arXiv},
  eprint = {2406.10033},
  primaryClass = {astro-ph.CO}
}
```

