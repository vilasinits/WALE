# Overview


The `WALE` package provides a theoretical framework to model the wavelet ℓ₁-norm of weak lensing convergence maps. This is achieved using predictions of the one-point probability distribution function (PDF) at each wavelet scale, based on large-deviation theory (LDT). The method enables simulation-free inference of higher-order statistics in cosmological fields.

📘 API Documentation: https://vilasinits.github.io/WALE/

This tool supports the analysis presented in the paper:
*Theoretical wavelet ℓ₁-norm from one-point probability density function prediction*, A&A 678, A116 (2024).  
A&A Article: https://www.aanda.org/articles/aa/full_html/2024/11/aa50061-24/aa50061-24.html

Development Status
------------------

The package is in its **initial phase** and several features are still being tested, including:

- A model for incorporating **stochasticity**
- Support for real data inputs
- Extension to additional filtering schemes

Key Features
------------

- Predicts the expected ℓ₁-norm at each wavelet scale from theory
- Includes a model for incorporating **stochasticity**, currently under testing
- Modular

Modules include:

- l1-norm computation
- Cosmology I/O and preprocessing
- Covariance modeling

Use this package for reproducibility of results in the paper or to apply the method to new data or simulations.

How to Cite
-----------

If you use this code for your research, please cite the following paper:

Vilasini Tinnaneri Sreekanth, *Theoretical wavelet ℓ₁-norm from one-point probability density function prediction*,  
Astronomy & Astrophysics, Volume 678, November 2024, A116.  
DOI: https://doi.org/10.1051/0004-6361/202450061

Contact
-------

Please feel free to reach out to **tsvilasini97@gmail.com** in case of any bugs or questions regarding usage.

## How to Install

Clone the repo and run:

```bash
pip install .