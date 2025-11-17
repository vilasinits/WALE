import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from .FilterFunctions import get_W2D_FL
from .CommonUtils import get_l1_from_pdf


def get_smoothed_app_pdf(mass_map, window_radius, binedges, filter_type, **kwargs):
    """
    Applies top-hat smoothing in Fourier space at two scales and returns the PDF of the difference map.

    The map is filtered with a top-hat window of radius R and 2R, then the difference is computed.

    Parameters:
        mass_map     : 2D numpy array.
        window_radius: The smoothing scale (R) in physical units.
        binedges     : Bin edges for the histogram.
        L            : Physical size of the map (default 505 MPC/h).

    Returns:
        tuple : (bin_edges, pdf_counts, difference_map)
    """
    if mass_map.ndim != 2:
        raise ValueError("mass_map must be 2D")

    # Pixel grid / shape
    shape = mass_map.shape  # (Ny, Nx)
    
    if kwargs.get("L") is not None:
        N = kwargs["L"]
    else:
        N = mass_map.shape[0]
        
    # Build the two Fourier-space windows
    W2D_1 = get_W2D_FL(window_radius, shape, filter_type,  **kwargs)
    W2D_2 = get_W2D_FL(window_radius * 2.0, shape, filter_type,  **kwargs)

    # Robust checks
    if W2D_1 is None or W2D_2 is None:
        raise RuntimeError(
            f"get_W2D_FL returned None for filter_type='{filter_type}'. "
            "Check the function signature/arguments and supported filter types."
        )
    if W2D_1.shape != shape or W2D_2.shape != shape:
        raise ValueError(
            f"Window shapes must match mass_map.shape={shape}, "
            f"got {W2D_1.shape=} and {W2D_2.shape=}."
        )

    # FFT of the input field
    field_ft = np.fft.fftshift(np.fft.fftn(mass_map))

    # Apply the windows in Fourier space (ensure dtype compatibility)
    W2D_1 = np.asarray(W2D_1, dtype=field_ft.dtype)
    W2D_2 = np.asarray(W2D_2, dtype=field_ft.dtype)

    smoothed_ft1 = field_ft * W2D_1
    smoothed_ft2 = field_ft * W2D_2

    # Back to real space
    smoothed1 = np.fft.ifftn(np.fft.ifftshift(smoothed_ft1)).real
    smoothed2 = np.fft.ifftn(np.fft.ifftshift(smoothed_ft2)).real
    
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # vmin = -0.015
    # vmax= 0.015
    # axs[0].imshow(smoothed1, cmap='viridis',vmin=vmin,vmax=vmax)
    # axs[0].set_title('Smoothed Map (R)')
    # axs[1].imshow(smoothed2, cmap='viridis',vmin=vmin, vmax = vmax)
    # axs[1].set_title('Smoothed Map (2R)')
    # axs[2].imshow(mass_map, cmap='viridis',vmin=vmin,vmax=vmax)
    # axs[2].set_title('Original Mass Map')
    # plt.tight_layout()
    # plt.show()
    
    difference_map = smoothed2 - smoothed1

    counts, _ = np.histogram(difference_map, bins=binedges, density=True)
    return binedges, counts, difference_map

def get_simulation_l1(
    cosmo_index_to_run,
    tomobin,
    edges,
    centers,
    snr,
    R_pixels=30,
    filter_type="tophat",
    plot=False,
):
    """
    Load simulation data for a specific cosmology and compute L1 norms and PDFs.
    Supports 20 realizations: 10 from each of two paths.

    Returns:
    - sim_l1_runs: Array of L1 norms for each simulation realization.
    - sim_pdf_runs: Array of PDF counts for each simulation realization.
    - avg_sim_l1: Average L1 norm across all realizations.
    - std_sim_l1: Standard deviation of L1 norms across realizations.
    - avg_sim_pdf: Average PDF counts across all realizations.
    - std_sim_pdf: Standard deviation of PDF counts across realizations.
    """
    num_realizations = 20
    sim_l1_runs_snr = np.zeros((num_realizations, len(snr)))
    sim_pdf_runs_snr = np.zeros((num_realizations, len(snr)))
    sim_sigmasq_runs = np.zeros((num_realizations, len(snr)))
    sim_l1_runs_kappa = np.zeros((num_realizations, len(centers)))
    sim_pdf_runs_kappa = np.zeros((num_realizations, len(centers)))
    simvar = []

    for i in range(1, num_realizations + 1):  # 1 to 20 inclusive
        los_cone_filename = f"GalCatalog_LOS_cone{((i-1)%10)+1}_bin{tomobin}.npy"

        if i <= 10:
            # First 10 realizations from first path
            if cosmo_index_to_run < 10:
                map_path = f"/feynman/work/dap/lcs/share/at/mass_maps/0{cosmo_index_to_run}_f/{los_cone_filename}"
            else:
                map_path = f"/feynman/work/dap/lcs/share/at/mass_maps/{cosmo_index_to_run}_f/{los_cone_filename}"
        else:
            # Next 10 realizations from second path (e.g., _f2 directory)
            if cosmo_index_to_run < 10:
                map_path = f"/feynman/work/dap/lcs/share/at/mass_maps/0{cosmo_index_to_run}_a/{los_cone_filename}"
            else:
                map_path = f"/feynman/work/dap/lcs/share/at/mass_maps/{cosmo_index_to_run}_a/{los_cone_filename}"

        try:
            mass_map_data = np.load(map_path)
            _, counts, diff_map = get_smoothed_app_pdf(
                mass_map_data, R_pixels, edges, filter_type
            )

            map_variance = np.var(diff_map)
            map_stdev = np.sqrt(map_variance)
            simvar.append(map_variance)

            kappa_over_sigma = centers / map_stdev
            l1_values = get_l1_from_pdf(counts, centers)

            sim_l1_spline = CubicSpline(kappa_over_sigma, l1_values, extrapolate=False)
            sim_pdf_spline = CubicSpline(kappa_over_sigma, counts, extrapolate=False)

            sim_l1_runs_kappa[i - 1] = l1_values
            sim_pdf_runs_kappa[i - 1] = counts
            sim_l1_runs_snr[i - 1] = sim_l1_spline(snr)
            sim_pdf_runs_snr[i - 1] = sim_pdf_spline(snr)
            sim_sigmasq_runs[i - 1] = map_stdev
        except FileNotFoundError:
            print(f"  Warning: File not found {map_path}")

    avg_sim_pdf_snr = np.nanmean(sim_pdf_runs_snr, axis=0)
    std_sim_pdf_snr = np.nanstd(sim_pdf_runs_snr, axis=0)
    avg_sim_l1_snr = np.nanmean(sim_l1_runs_snr, axis=0)
    std_sim_l1_snr = np.nanstd(sim_l1_runs_snr, axis=0)

    if plot:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(snr, avg_sim_l1_snr, label="Average L1 Norm", color="blue")
        plt.plot(snr, sim_l1_runs_snr.T, color="cornflowerblue", alpha=0.5)
        plt.title("L1 Norm")
        plt.xlabel("SNR")
        plt.ylabel("L1 Norm")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(snr, avg_sim_pdf_snr, label="Average PDF Counts", color="orange")
        plt.plot(snr, sim_pdf_runs_snr.T, color="gold", alpha=0.5)
        plt.title("PDF Counts")
        plt.xlabel("SNR")
        plt.ylabel("PDF Counts")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return (
        np.array(sim_l1_runs_snr),
        np.array(sim_pdf_runs_snr),
        np.array(sim_l1_runs_kappa),
        np.array(sim_pdf_runs_kappa),
        avg_sim_l1_snr,
        std_sim_l1_snr,
        avg_sim_pdf_snr,
        std_sim_pdf_snr,
        np.array(simvar),
    )
