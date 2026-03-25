# WALE

[![CI](https://github.com/vilasinits/WALE/actions/workflows/ci.yml/badge.svg)](https://github.com/vilasinits/WALE/actions)  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  ![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)


**WALE** (pronounced *WAL-E*, short for **W**avelet **ℓ₁**-norm **E**stimator) is a Python package for predicting and analyzing the one-point statistics of the wavelet ℓ₁-norm in cosmological density fields. It supports both theoretical predictions based on large-deviation statistics and direct measurements from simulations or observational data, enabling cross-scale comparisons and validation.


- ✅ **CI-tested** via [GitHub Actions](https://github.com/vilasinits/WALE/actions)
- 🎯 **Code formatted** using [Black](https://black.readthedocs.io/)
- 🐳 **Docker-ready** — containerized image available for reproducible environments

> 📦 A pre-configured Dockerfile is included in the repository to allow easy containerization and deployment.


---

## Documentation

Comprehensive API reference and usage tutorials are available at:  
👉 [https://vilasinits.github.io/WALE/wale.html](https://vilasinits.github.io/WALE/wale.html)

---

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/vilasinits/WALE.git
cd WALE
pip install -e .
```

---

## Quickstart

Explore the example notebooks in the notebooks/ directory to get started with theory predictions or applications on your own data.

---

## Key Features

- Theory Predictions: Estimate the expected wavelet ℓ₁-norm at each scale from the one-point PDF using Large Deviation Theory.

- Stochastic Extensions: Preliminary support for stochastic modeling is included and currently under testing.

- Modular Design: Use individual components or the full pipeline depending on your use case.

Included Modules
  - Wavelet ℓ₁-norm computation

  - Cosmological data I/O and preprocessing

  - Covariance estimation and modeling

This toolkit supports reproducibility of results from our paper and can be extended for new simulations or observational datasets.

---

## Citation

If you use WALE in your work, please cite the following publication:


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

