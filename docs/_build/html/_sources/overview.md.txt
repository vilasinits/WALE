[![CI](https://github.com/vilasinits/WALE/actions/workflows/ci.yml/badge.svg)](https://github.com/vilasinits/WALE/actions)  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  ![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)

# WALE — Wavelet ℓ₁-norm Estimator

**WALE** (pronounced *WAL-E*) is a Python toolkit for predicting and analyzing the one-point statistics of the **wavelet ℓ₁-norm** in cosmological density fields. It combines theoretical predictions based on one-point PDF expansions with direct measurements on simulations or observational data, enabling robust multi-scale comparisons.

---

##  Repository

Source code and issue tracker:  
🔗 [https://github.com/vilasinits/WALE](https://github.com/vilasinits/WALE)

---

##  Features

- **Theoretical Predictions**  
  Derive analytical estimates of the wavelet ℓ₁-norm's mean and variance using one-point PDF expansions rooted in Large Deviation Theory.

- **Wavelet Decomposition**  
  Perform multi-scale analysis with wavelet bases such as top-hat and starlet to extract scale-resolved information.

- **ℓ₁-norm Measurements**  
  Compute the ℓ₁-norm of wavelet coefficients on weak lensing convergence fields efficiently and accurately.

- **Theory vs. Simulation Comparison**  
  Built-in routines to overlay theoretical predictions with simulation results, including visualization tools and diagnostic metrics.

- **Modular API**  
  Clean, extensible architecture with dedicated modules for theory, analysis, I/O, and utility functions.

- **Parallel Processing (Coming Soon)**  
  MPI-based support for handling large cosmological datasets in parallel.

- **JAX Integration (Coming Soon)**  
  Accelerated computation and auto-differentiation via JAX for high-performance workflows.

---

##  Installation

Clone and install in editable mode:

```bash
git clone https://github.com/vilasinits/WALE.git
cd WALE
pip install -e .
```

## Quickstart

Explore the example notebooks in the notebooks/ directory to see how WALE can be applied to theoretical predictions or real data analysis.

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


