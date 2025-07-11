import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from wale.FilterFunctions import get_W2D_FL
from wale.CommonUtils import get_l1_from_pdf

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
    if kwargs.get("L") is not None:
        N = kwargs["L"]
    else:
        N = mass_map.shape[0]
    if filter_type == "tophat":
        W2D_1 = get_W2D_FL(window_radius, mass_map.shape, "tophat", N)
        W2D_2 = get_W2D_FL(window_radius * 2, mass_map.shape, "tophat", N)
        # Fourier transform the input mass map.
        field_ft = np.fft.fftshift(np.fft.fftn(mass_map))
        # Apply the window functions in Fourier space.
        smoothed_ft1 = field_ft * W2D_1
        smoothed_ft2 = field_ft * W2D_2
        # Inverse Fourier transform to get back to real space.
        smoothed1 = np.fft.ifftn(np.fft.ifftshift(smoothed_ft1)).real
        smoothed2 = np.fft.ifftn(np.fft.ifftshift(smoothed_ft2)).real
        # Compute the difference map.
        difference_map = smoothed2 - smoothed1
    elif filter_type == "starlet":
        W2D_1 = get_W2D_FL(window_radius, mass_map.shape, "starlet", N)
        W2D_2 = get_W2D_FL(window_radius*2 , mass_map.shape, "starlet", N)
        # Fourier transform the input mass map.
        field_ft = np.fft.fftshift(np.fft.fftn(mass_map))
        # Apply the window functions in Fourier space.
        smoothed_ft1 = field_ft * W2D_1
        smoothed_ft2 = field_ft * W2D_2
        # Inverse Fourier transform to get back to real space.
        smoothed1 = np.fft.ifftn(np.fft.ifftshift(smoothed_ft1)).real
        smoothed2 = np.fft.ifftn(np.fft.ifftshift(smoothed_ft2)).real
        # Compute the difference map.
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
    Parameters:
    - cosmo_index_to_run: Index of the cosmology to run (0-9 for 10 different cosmologies).
    - R_pixels: Physical scale in pixels for smoothing.
    - filter_type: Type of filter to use for smoothing ('tophat' or 'gaussian').
    Returns:
    - sim_l1_runs: Array of L1 norms for each simulation realization.
    - sim_pdf_runs: Array of PDF counts for each simulation realization.
    - avg_sim_l1: Average L1 norm across all realizations.
    - std_sim_l1: Standard deviation of L1 norms across realizations.
    - avg_sim_pdf: Average PDF counts across all realizations.
    - std_sim_pdf: Standard deviation of PDF counts across realizations.
    """
    # Load simulation data for this cosmology
    sim_l1_runs = np.zeros((10, len(snr)))
    sim_pdf_runs = np.zeros((10, len(snr)))
    sim_sigmasq_runs = np.zeros((10, len(snr)))
    # ell_bins_runs = np.zeros((5, len(edges) - 1))
    # cls_runs = []
    simvar = []
    for i in range(1, 11):  # Loop over 10 simulation realizations
        los_cone_filename = f"GalCatalog_LOS_cone{i}_bin{tomobin}.npy"
        if cosmo_index_to_run < 10:
            map_path = f"/feynman/work/dap/lcs/share/at/mass_maps/0{cosmo_index_to_run}_a/{los_cone_filename}"
        else:
            map_path = f"/feynman/work/dap/lcs/share/at/mass_maps/{cosmo_index_to_run}_a/{los_cone_filename}"

        try:
            # print(f"  Loading map: {map_path}") # Optional: for debugging
            mass_map_data = np.load(map_path)
            _, counts, diff_map = get_smoothed_app_pdf(
                mass_map_data, R_pixels, edges, filter_type
            )  

            # Calculate L1 norm for this realization
            map_variance = np.var(diff_map)
            map_stdev = np.sqrt(map_variance)
            simvar.append(map_variance)

            kappa_over_sigma = centers / map_stdev
            l1_values = get_l1_from_pdf(counts, centers)
            # Interpolate L1 onto the common SNR grid
            sim_l1_spline = CubicSpline(kappa_over_sigma, l1_values, extrapolate=False)
            sim_pdf_spline = CubicSpline(kappa_over_sigma, counts, extrapolate=False)  
            sim_l1_runs[i - 1] = sim_l1_spline(snr)  # Store interpolated L1 for this run
            sim_pdf_runs[i - 1] = sim_pdf_spline(snr)  # Store interpolated PDF counts for this run
            sim_sigmasq_runs[i - 1] = map_stdev  # Store variance for this run

            # _, ell_bins, cls_values = calculate_Cls(mass_map_data, 10, 10, 1e3, 60)
            # print("the ell_bins shape: ", ell_bins.shape, "and :", ell_bins[0])
            # cls_runs.append(cls_values)
        except FileNotFoundError:
            print(f"  Warning: File not found {map_path}")

    # Average over simulation realizations
    avg_sim_pdf = np.nanmean(sim_pdf_runs, axis=0)
    std_sim_pdf = np.nanstd(sim_pdf_runs, axis=0)
    avg_sim_l1 = np.nanmean(sim_l1_runs, axis=0)
    std_sim_l1 = np.nanstd(sim_l1_runs, axis=0)

    if plot:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(snr, avg_sim_l1, label="Average L1 Norm", color="blue")
        plt.plot(
            snr, sim_l1_runs.T, color="cornflowerblue", alpha=0.5
        )  # Plot individual runs with transparency
        plt.title("L1 Norm")
        plt.xlabel("SNR")
        plt.ylabel("L1 Norm")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(snr, avg_sim_pdf, label="Average PDF Counts", color="orange")
        plt.plot(
            snr, sim_pdf_runs.T, color="gold", alpha=0.5
        )  # Plot individual runs with transparency
        plt.title("PDF Counts")
        plt.xlabel("SNR")
        plt.ylabel("PDF Counts")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return (
        np.array(sim_l1_runs),
        np.array(sim_pdf_runs),
        avg_sim_l1,
        std_sim_l1,
        avg_sim_pdf,
        std_sim_pdf,
        # ell_bins,
        # np.array(cls_runs),
        np.array(simvar)
    )
