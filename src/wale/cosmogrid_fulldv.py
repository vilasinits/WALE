from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed

from .ComputePDF import computePDF
from .CriticalPoints import find_critical_points_for_cosmo
from .InitializeVariables import InitialiseVariables
from .VarianceCalculator import Variance

PARAM_NAMES_DEFAULT = np.array(
    [r"$\Omega_{m}$", r"$\sigma_8$", r"$w_0$", r"$H_0$", r"$n_s$", r"$\Omega_b$"]
)


@dataclass(frozen=True)
class FullDVConfig:
    theta1: float
    nz_file: str
    nplanes: int = 69
    kmin: float = 1e-3
    kmax: float = 1.0
    dk: float = 0.001
    nlambdas: int = 61
    filter_type: str = "tophat"
    ngrid_critical: int = 5
    min_z: int = 0
    max_z: int = 5
    lambda_fallback_min: float = -400.0
    lambda_fallback_max: float = 1400.0
    ells: tuple[float, ...] = tuple(
        np.unique(np.geomspace(10, 5000, 60).astype(int)).astype(float).tolist()
    )
    save_pk: bool = False
    disable_recal: bool = False  # if True: do not rescale the LDT action by σ²_LDT/σ²_sim


def build_ell_grid(
    ell_min: float = 10.0, ell_max: float = 5000.0, n_ell: int = 60
) -> np.ndarray:
    return np.unique(np.geomspace(ell_min, ell_max, n_ell).astype(int)).astype(float)


def infer_simulation_paths(
    data_dir: Path,
    tomo_bin: int,
    theta1: float,
    simulation_tag: str = "halofit_nobaryons",
) -> dict[str, Path]:
    data_dir = Path(data_dir)
    theta_tag = f"{float(theta1):.1f}"

    prefixes = {
        "kappa": f"all_kappas_{simulation_tag}",
        "l1": f"all_l1_norms_{simulation_tag}",
        "variance": f"all_variances_{simulation_tag}",
    }

    def _find_first_existing(prefix: str) -> Path:
        candidates = [
            data_dir / f"{prefix}_bin{tomo_bin}_theta{theta_tag}_ratio2.0.npy",
            data_dir / f"{prefix}_bin{tomo_bin}_theta{theta_tag}.npy",
            data_dir / f"{prefix}_bin{tomo_bin}.npy",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    return {
        "kappa": _find_first_existing(prefixes["kappa"]),
        "l1": _find_first_existing(prefixes["l1"]),
        "variance": _find_first_existing(prefixes["variance"]),
    }


def load_input_arrays(
    params_file: Path,
    kappa_file: Path,
    l1_file: Path,
    variance_file: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    params = np.load(params_file)
    kappa_sim = np.load(kappa_file)
    l1_norms_sim = np.load(l1_file)
    variance_sim = np.load(variance_file)

    if params.ndim != 2 or params.shape[1] != 6:
        raise ValueError(f"Expected params shape (N, 6), got {params.shape}.")
    if kappa_sim.ndim != 2:
        raise ValueError(f"Expected kappa_sim shape (N, nbins), got {kappa_sim.shape}.")
    if l1_norms_sim.ndim != 2:
        raise ValueError(
            f"Expected l1_norms_sim shape (N, nbins), got {l1_norms_sim.shape}."
        )
    if variance_sim.ndim != 1:
        raise ValueError(f"Expected variance_sim shape (N,), got {variance_sim.shape}.")
    if kappa_sim.shape != l1_norms_sim.shape:
        raise ValueError(
            "kappa_sim and l1_norms_sim must have identical shapes "
            f"(got {kappa_sim.shape} vs {l1_norms_sim.shape})."
        )

    return params, kappa_sim, l1_norms_sim, variance_sim


def _lookup_pk_slice(pnl: dict[float, np.ndarray], z: float) -> np.ndarray:
    if z in pnl:
        return np.asarray(pnl[z], dtype=float)

    for key in pnl.keys():
        try:
            if abs(float(key) - float(z)) < 1e-9:
                return np.asarray(pnl[key], dtype=float)
        except Exception:
            continue

    numeric_keys = []
    for key in pnl.keys():
        try:
            numeric_keys.append((abs(float(key) - float(z)), key))
        except Exception:
            continue
    if not numeric_keys:
        raise KeyError(f"No compatible P(k) entry found for z={z}.")
    _, nearest_key = min(numeric_keys, key=lambda item: item[0])
    return np.asarray(pnl[nearest_key], dtype=float)


def _compute_cls_discrete(variables: Any, ells: np.ndarray) -> np.ndarray:
    cls = np.zeros(len(ells), dtype=float)
    plane_weights = (
        variables.dchi
        / np.asarray(variables.chis) ** 2
        * np.asarray(variables.lensingweights) ** 2
    )
    for z_pl, chi_pl, pw in zip(variables.redshifts, variables.chis, plane_weights):
        k_ells = (ells + 0.5) / chi_pl
        pk_ells = variables.cosmo.get_nonlinear_pk(z_pl, ks=k_ells)
        cls += pw * pk_ells
    return cls


def _compute_one_cosmology(
    i: int,
    param_i: np.ndarray,
    kappa_bins: np.ndarray,
    var_sim_i: float,
    config: FullDVConfig,
) -> tuple[
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    np.ndarray,
    dict[str, np.ndarray] | None,
]:
    try:
        Om, sigma8, w0, h_100, ns, Ob = [float(x) for x in param_i]
        h = h_100 / 100.0
        Oc = Om - Ob

        variables = InitialiseVariables(
            h=h,
            Oc=Oc,
            Ob=Ob,
            w=w0,
            wa=0.0,
            sigma8=sigma8,
            dk=config.dk,
            kmin=config.kmin,
            kmax=config.kmax,
            ns=ns,
            nz_file=config.nz_file,
            zs=None,
            variability=False,
            theta1=config.theta1,
            nplanes=config.nplanes,
            numberofrealisations=1,
        )

        variance_theory = Variance(
            variables.cosmo,
            filter_type=config.filter_type,
            pk=variables.cosmo.pnl,
        )

        sigmasq_map_2c_th = float(
            np.sum(
                variables.dchi
                * np.asarray(variables.lensingweights) ** 2
                * np.array(
                    [
                        float(
                            variance_theory.get_sig_slice(
                                z,
                                chi * variables.theta1_radian,
                                chi * variables.theta2_radian,
                            )
                        )
                        for z, chi in zip(variables.redshifts, variables.chis)
                    ]
                )
            )
        )

        if float(var_sim_i) <= 0.0:
            raise ValueError(f"variance_sim[{i}] must be positive, got {var_sim_i}.")
        if not config.disable_recal:
            variables.recal_value = sigmasq_map_2c_th / float(var_sim_i)
        # else: leave variables.recal_value at its default 1.0 (set in InitialiseVariables)

        smallest_positive, largest_negative = find_critical_points_for_cosmo(
            variables,
            variance_theory,
            ngrid_critical=config.ngrid_critical,
            plot=False,
            min_z=config.min_z,
            max_z=config.max_z,
        )
        if smallest_positive is None or largest_negative is None:
            lambdas = np.linspace(
                config.lambda_fallback_min,
                config.lambda_fallback_max,
                config.nlambdas,
            )
        else:
            lambdas = np.linspace(largest_negative, smallest_positive, config.nlambdas)
        variables.lambdas = lambdas

        pdf_theory = computePDF(
            variables,
            variance_theory,
            kappa=np.asarray(kappa_bins, dtype=float),
            single_cell=False,
        )
        kappa_th = np.asarray(pdf_theory.kappa_values, dtype=float)
        p_th = np.asarray(pdf_theory.pdf_values, dtype=float)
        l1_th = np.abs(kappa_th) * p_th

        cls = _compute_cls_discrete(variables, np.asarray(config.ells, dtype=float))

        pk_payload = None
        if config.save_pk:
            redshifts = np.asarray(variables.redshifts, dtype=float)
            pnl_matrix = np.vstack(
                [_lookup_pk_slice(variables.cosmo.pnl, z) for z in redshifts]
            )
            pk_payload = {
                "k": np.asarray(variables.cosmo.k, dtype=float),
                "redshifts": redshifts,
                "pnl": pnl_matrix,
            }

        return (
            i,
            kappa_th,
            p_th,
            l1_th,
            sigmasq_map_2c_th,
            variables.recal_value,
            cls,
            pk_payload,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed processing cosmology index {i}.") from exc


def run_cosmogrid_fulldv(
    params: np.ndarray,
    kappa_sim: np.ndarray,
    l1_norms_sim: np.ndarray,
    variance_sim: np.ndarray,
    config: FullDVConfig,
    n_cosmo: int | None = None,
    n_jobs: int = 1,
    backend: str = "loky",
    verbose: int = 5,
) -> dict[str, Any]:
    available = min(
        int(params.shape[0]),
        int(kappa_sim.shape[0]),
        int(l1_norms_sim.shape[0]),
        int(variance_sim.shape[0]),
    )
    if available == 0:
        raise ValueError("No cosmologies available in inputs.")

    if n_cosmo is None:
        n_cosmo = available
    n_cosmo = int(min(max(1, n_cosmo), available))

    n_bins = int(kappa_sim.shape[1])
    n_ells = len(config.ells)

    out_params = np.zeros((n_cosmo, 6), dtype=float)
    out_kappa = np.zeros((n_cosmo, n_bins), dtype=float)
    out_pdf_theory = np.zeros((n_cosmo, n_bins), dtype=float)
    out_l1_theory = np.zeros((n_cosmo, n_bins), dtype=float)
    out_l1_sim = np.zeros((n_cosmo, n_bins), dtype=float)
    out_variance_ldt = np.zeros(n_cosmo, dtype=float)
    out_variance_sim = np.zeros(n_cosmo, dtype=float)
    out_recal = np.zeros(n_cosmo, dtype=float)
    out_cls = np.zeros((n_cosmo, n_ells), dtype=float)
    pk_payloads: list[dict[str, np.ndarray] | None] = [None] * n_cosmo

    indices = range(n_cosmo)
    if n_jobs == 1:
        results = [
            _compute_one_cosmology(
                i=i,
                param_i=params[i],
                kappa_bins=kappa_sim[i],
                var_sim_i=variance_sim[i],
                config=config,
            )
            for i in indices
        ]
    else:
        results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
            delayed(_compute_one_cosmology)(
                i=i,
                param_i=params[i],
                kappa_bins=kappa_sim[i],
                var_sim_i=variance_sim[i],
                config=config,
            )
            for i in indices
        )

    for idx, kappa_th, p_th, l1_th, sig2_ldt, recal, cl_kk, pk_payload in results:
        out_params[idx] = params[idx]
        out_kappa[idx] = kappa_th
        out_pdf_theory[idx] = p_th
        out_l1_theory[idx] = l1_th
        out_l1_sim[idx] = l1_norms_sim[idx]
        out_variance_ldt[idx] = sig2_ldt
        out_variance_sim[idx] = variance_sim[idx]
        out_recal[idx] = recal
        out_cls[idx] = cl_kk
        pk_payloads[idx] = pk_payload

    return {
        "params": out_params,
        "kappa_bins": out_kappa,
        "pdf_theory": out_pdf_theory,
        "l1_theory": out_l1_theory,
        "l1_sim": out_l1_sim,
        "variance_ldt": out_variance_ldt,
        "variance_sim": out_variance_sim,
        "recal_value": out_recal,
        "ells": np.asarray(config.ells, dtype=float),
        "cls": out_cls,
        "pk_payloads": pk_payloads,
        "n_cosmo": n_cosmo,
    }


def save_theory_outputs(
    output_file: Path,
    outputs: dict[str, Any],
    theta1: float,
    tomo_bin: int,
    filter_type: str,
    param_names: np.ndarray | None = None,
) -> None:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if param_names is None:
        param_names = PARAM_NAMES_DEFAULT

    np.savez_compressed(
        output_file,
        params=outputs["params"],
        param_names=np.asarray(param_names),
        kappa_bins=outputs["kappa_bins"],
        pdf_theory=outputs["pdf_theory"],
        l1_theory=outputs["l1_theory"],
        l1_sim=outputs["l1_sim"],
        variance_ldt=outputs["variance_ldt"],
        variance_sim=outputs["variance_sim"],
        recal_value=outputs["recal_value"],
        theta=float(theta1),
        tomo_bin=int(tomo_bin),
        filter_type=np.array(filter_type),
        n_cosmo=int(outputs["n_cosmo"]),
    )


def save_cls_outputs(
    output_file: Path,
    outputs: dict[str, Any],
    tomo_bin: int,
    param_names: np.ndarray | None = None,
) -> None:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if param_names is None:
        param_names = PARAM_NAMES_DEFAULT

    np.savez_compressed(
        output_file,
        params=outputs["params"],
        param_names=np.asarray(param_names),
        ells=outputs["ells"],
        cls=outputs["cls"],
        tomo_bin=int(tomo_bin),
        n_cosmo=int(outputs["n_cosmo"]),
    )


def save_pk_outputs(
    output_file: Path,
    outputs: dict[str, Any],
    tomo_bin: int,
) -> None:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    payloads = outputs["pk_payloads"]
    if not payloads or any(payload is None for payload in payloads):
        raise ValueError("No complete P(k,z) payloads available to save.")

    first = payloads[0]
    assert first is not None
    k0 = np.asarray(first["k"], dtype=float)
    zshape0 = np.asarray(first["redshifts"], dtype=float).shape
    pshape0 = np.asarray(first["pnl"], dtype=float).shape

    same_layout = True
    for payload in payloads[1:]:
        assert payload is not None
        kk = np.asarray(payload["k"], dtype=float)
        zz = np.asarray(payload["redshifts"], dtype=float)
        pp = np.asarray(payload["pnl"], dtype=float)
        if kk.shape != k0.shape or not np.allclose(kk, k0):
            same_layout = False
            break
        if zz.shape != zshape0 or pp.shape != pshape0:
            same_layout = False
            break

    if same_layout:
        redshift_grid = np.stack(
            [np.asarray(payload["redshifts"], dtype=float) for payload in payloads],
            axis=0,
        )
        pnl_cube = np.stack(
            [np.asarray(payload["pnl"], dtype=float) for payload in payloads], axis=0
        )
        np.savez_compressed(
            output_file,
            params=outputs["params"],
            k=k0,
            redshifts=redshift_grid,
            pnl=pnl_cube,
            tomo_bin=int(tomo_bin),
            n_cosmo=int(outputs["n_cosmo"]),
        )
    else:
        np.savez_compressed(
            output_file,
            params=outputs["params"],
            pk_records=np.array(payloads, dtype=object),
            tomo_bin=int(tomo_bin),
            n_cosmo=int(outputs["n_cosmo"]),
        )
