
def E2_and_dlnE_dlnA(self,a, Om0, w0, wa):
        """
        Return (E^2(a), d ln E / d ln a) for a CPL w(a)=w0+wa(1-a) universe.
        """
        Ode0 = 1.0 - Om0
        # E^2(a)
        E2 = Om0 * a**(-3) + Ode0 * a**(-3*(1+w0+wa)) * np.exp(-3*wa*(1-a))
        # derivative dE2/da
        p = -3*(1 + w0 + wa)
        q =  3*wa
        # part from dark energy term f(a)=a^p exp(q a)
        fa = a**p * np.exp(q*(a-1))
        dfa_da = fa * (p/a + q)
        dE2_da = -3*Om0 * a**(-4) + Ode0 * dfa_da
        # then d ln E/d ln a = (a / (2 E2)) * (dE2/da)
        dlnE_dlnA = (a / (2*E2)) * dE2_da
        return E2, dlnE_dlnA

    def Dz_fz(self,z, Om0, w0=-1.0, wa=0.0):
        """
        Compute the linear growth factor D(z) (normalized to D(0)=1)
        and the growth rate f(z) = d ln D / d ln a for a CPL universe.
        
        Parameters
        ----------
        z  : float or array_like
            Redshift(s) at which to evaluate D and f.
        Om0: float
            Omega_matter today.
        w0 : float
            Dark energy EOS at z=0.
        wa : float
            Evolution of EOS.
        
        Returns
        -------
        D  : ndarray
            Growth factor(s), normalized so D(z=0)=1.
        f  : ndarray
            Growth rate(s), f = d ln D / d ln a.
        """
        # we solve in ln a from a_init to a=1
        a_init = 1e-4    # deep matter-domination
        lnA_init = np.log(a_init)
        lnA_final = 0.0  # a=1
        
        # the ODE system in y = [D, G] where G = dD/d ln a
        def dydlnA(lnA, y):
            a = np.exp(lnA)
            D, G = y
            E2, dlnE = self.E2_and_dlnE_dlnA(a, Om0, w0, wa)
            Omega_ma = Om0 * a**(-3) / E2
            dG_dlnA = - (2 + dlnE)*G + 1.5 * Omega_ma * D
            return [G, dG_dlnA]
        
        # initial conditions: in matter-dom D ≈ a => G = dD/dln a = D
        y0 = [a_init, a_init]
        
        # integrate from lnA_init → 0
        sol = solve_ivp(
            fun=dydlnA,
            t_span=(lnA_init, lnA_final),
            y0=y0,
            dense_output=True,
            atol=1e-8, rtol=1e-8
        )
        
        # build outputs
        z = np.atleast_1d(z)
        a_vals = 1.0 / (1.0 + z)
        lnA_vals = np.log(a_vals)
        Ds, Gs = sol.sol(lnA_vals)      # sol.sol returns [D, G]
        D  = Ds / Ds[z.size>0 and np.where(z==0)[0][0] if 0 in z else Ds[-1]]
        f  = Gs / Ds                   # f = d ln D / d ln a = G / D
        
        return D, f



cosmo = Cosmology_function(zs=1, h=0.68, Oc=0.24, Ob=0.043, w= -1.0, wa=0.0, sigma8=0.8, dk=0.009, kmin=1e-4, kmax=3)


redshifts = np.array([0.1, 0.25, 0.5, 1.0])
cov, pnlsamples, pnl = get_covariance(cosmo, z=redshifts, variability=True, numberofrealisations=100)
plt.imshow(cov[1], origin='lower', cmap='viridis', aspect='auto', vmin=0, vmax=5)
plt.colorbar(label='Covariance')
plt.title('Covariance Matrix')
plt.show()


for i, a in enumerate(redshifts):
    for sample in pnlsamples[i]:
        plt.loglog(cosmo.k, sample, color='gray', alpha=0.3)
    plt.loglog(cosmo.k, pnl[i], 'r-', lw=2, label='mean')
    plt.title(f'Realizations at z = {1/a-1:.2f}')
    plt.xlabel('k [h/Mpc]')
    plt.ylabel('P_nl(k)')
    plt.legend()




# Assuming `cov` is a (na, nk, nk) array and we pick the second redshift slice:
cov1 = cov[3]

# Build correlation matrix
diag = np.sqrt(np.diag(cov1))
corr = cov1 / np.outer(diag, diag)

# Build precision matrix (inverse of correlation)
prec = np.linalg.inv(corr)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

im0 = axes[0].imshow(corr, origin='lower', aspect='auto')
fig.colorbar(im0, ax=axes[0], label='Correlation')
axes[0].set_title('Correlation Matrix')

im1 = axes[1].imshow(prec, origin='lower', aspect='auto')
fig.colorbar(im1, ax=axes[1], label='Precision')
axes[1].set_title('Precision Matrix')

plt.tight_layout()
plt.show()




# --- assume you already have: ---
# ks          : array of k‐bins, shape (nk,)
# cov_tree    : tree‐level trispectrum matrix C_tree(k1,k2), shape (nk,nk)
# cov_resp    : response 1‐loop matrix      C_resp(k1,k2), shape (nk,nk)
# C_gauss_diag: Gaussian diag term         C_gauss(k_i,k_j)=δ_ij*2 P^2/Nmodes, shape (nk,)
# (so that C_full = cov_tree + cov_resp + np.diag(C_gauss_diag))

ks= cosmo.k
# pick the k2 values you want to slice at:
k2_vals = [0.05, 0.10, 0.20]   # in h/Mpc, adjust as desired
idx2    = [np.argmin(np.abs(ks - kv)) for kv in k2_vals]

fig, axes = plt.subplots(len(k2_vals), 1, figsize=(6, 4*len(k2_vals)), sharex=True)

for ax, k2, i2 in zip(axes, k2_vals, idx2):
    slice_full = cov[0][:, i2]    # C(k1, k2_fixed)
    
    ax.loglog(ks, slice_full, label='Full covariance')
    ax.axvline(k2, color='k', ls='--', alpha=0.5)
    ax.set_ylabel(r'$C(k_1,k_2)$')
    ax.set_title(f'$k_2 = {k2:.2f}\\,h/{{\\rm Mpc}}$')
    ax.grid(which='both', ls=':')
    ax.legend()

axes[-1].set_xlabel(r'$k_1\,[h/\mathrm{Mpc}]$')
plt.tight_layout()
plt.show()









# --- User parameters ---
tomo_bin = 5
nz_file = f"/feynman/work/dap/lcs/vt272285/final_codes/LDT_2cell_l1_norm/data/nz_arrays_bin{tomo_bin}.npy"

filter_type = 'tophat'

theta1 = 15
R_pixels = int(theta1 / 0.5)
edges = np.linspace(-0.06, 0.06, 401)
centers = 0.5 * (edges[:-1] + edges[1:])
snr = np.linspace(-6, 6, 800)

# Flag: include simulation results
simulation = False  # Set to False to disable simulation data

# Container for results
results = {}

def main():
    if simulation:
        fig, ax = plt.subplots(2, 1, figsize=(12, 5))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    for cosmo_index in range(0, 25):
        print(f"Processing cosmology index: {cosmo_index}")
        # --- Simulation block ---
        if simulation:
            sim_l1_runs, sim_pdf_runs, avg_sim_l1, std_sim_l1, avg_sim_pdf, std_sim_pdf, simvar = \
                get_simulation_l1(
                    cosmo_index,
                    tomo_bin,
                    edges,
                    centers,
                    snr,
                    R_pixels=R_pixels,
                    filter_type=filter_type,
                    plot=False,
                )
        else:
            sim_l1_runs = sim_pdf_runs = avg_sim_l1 = std_sim_l1 = avg_sim_pdf = std_sim_pdf = simvar = None

        # --- Cosmology parameters ---
        h = lh_df_new['h'].iloc[cosmo_index]
        Om = lh_df_new['Om'].iloc[cosmo_index]
        Ob = 0.043
        Oc = Om - Ob #lh_df_new['Oc'].iloc[cosmo_index]
       
        w = lh_df_new['w0'].iloc[cosmo_index]
        wa = 0.0
        sigma8 = lh_df_new['sig8'].iloc[cosmo_index]
        

        variables = InitialiseVariables(
            h=h, Oc=Oc, Ob=Ob, w=w, wa=wa, sigma8=sigma8,
            dk=0.005, kmin=1e-4, kmax=3,
            nz_file=nz_file, variability=False,
            theta1=theta1, nplanes=7
        )

        # Compute sigma^2 maps
        variance = Variance(variables.cosmo, filter_type=filter_type, pk=variables.cosmo.pnl)
        variables.sigmasq_map = np.sum(
            variables.dchi * (variables.lensingweights ** 2) * np.array([
                variance.get_sig_slice(z, chi * variables.theta1_radian, chi * variables.theta2_radian)
                for z, chi in zip(variables.redshifts, variables.chis)
            ])
        )

        # Theory (perturbation) sigma^2
        theory_sigmasq = compute_sigma_kappa_squared(
            theta1, variables.chis, variables.lensingweights,
            variables.redshifts, variables.cosmo.k, variables.cosmo.pnl,
            filter_type=filter_type, h=h
        )

        # Recalibration factor
        if simulation:
            variables.recal_value = variables.sigmasq_map / np.mean(simvar)
        else:
            variables.recal_value = variables.sigmasq_map / theory_sigmasq
            
        # variables.recal_value = variables.sigmasq_map / theory_sigmasq
        print(f"The ldt variance is: {variables.sigmasq_map}")
        print(f"The perturbation theory variance is: {theory_sigmasq}") 
        print(f"The recalibration factor is: {variables.recal_value}")
              
        # Critical points and lambdas
        smallest_positive, largest_negative = find_critical_points_for_cosmo(
            variables, variance, ngrid_critical=40, plot=False, min_z=0, max_z=3
        )
        if smallest_positive and largest_negative is not None:
            variables.lambdas = np.linspace(largest_negative, smallest_positive, 20)
        else:
            variables.lambdas = np.linspace(-650, 900, 20)

        # Compute PDF and L1
        computed_PDF = computePDF(variables, variance, plot_scgf=False)
        pdf_vals, kappa_vals = computed_PDF.pdf_values, computed_PDF.kappa_values
        norm_kappa = kappa_vals / np.sqrt(theory_sigmasq)
        prediction_pdf = CubicSpline(norm_kappa, pdf_vals)(snr)
        prediction_l1 = CubicSpline(norm_kappa, np.abs(kappa_vals) * pdf_vals)(snr)

        # Plot PDF
        # ax[0].plot(snr, prediction_pdf, label='Computed PDF',c=distinct_colours[cosmo_index])
        # if simulation:
        #     for run in sim_pdf_runs:
        #         ax[0].plot(snr, run, c='indianred', alpha=0.5)
        #     ax[0].errorbar(snr[::4], avg_sim_pdf[::4], yerr=std_sim_pdf[::4], c=dark_colours[cosmo_index], label='Simulation PDF', linewidth=2)
        # ax[0].set(xlabel='SNR', ylabel='PDF', xlim=(-5, 5))

        # Plot L1 Norm
        factor = 1
        if simulation:
            ax[0].plot(snr, prediction_l1 * factor, label='Computed L1 Norm',c=distinct_colours[cosmo_index])
        
            ax[0].errorbar(snr[::4], avg_sim_l1[::4] * factor, yerr=std_sim_l1[::4] * factor,
                          c=dark_colours[cosmo_index], label='Simulation L1 Norm', linewidth=2)
            mask = (snr > -2) & (snr < 2)
            residual = (prediction_l1 - avg_sim_l1) / avg_sim_l1
            ax[1].plot(snr[mask][::7], residual[mask][::7], label='Residuals', linestyle='--', c=distinct_colours[cosmo_index])
            ax[1].axhspan(-0.1, 0.1, alpha=0.3, label='±10%')
            ax[1].set_ylim(-0.5, 0.5)
            ax[1].axhline(0, color='black', linestyle='--', linewidth=0.5)
            ax[1].set(xlabel='SNR', ylabel='L1 Norm', xlim=(-2, 2))
            ax[0].set(xlabel='SNR', ylabel='L1 Norm', xlim=(-5, 5))
        else:
            ax.plot(snr, prediction_l1 * factor, label='Computed L1 Norm',c=distinct_colours[cosmo_index])
            ax.set(xlabel='SNR', ylabel='L1 Norm', xlim=(-5, 5))
        # Store results
        entry = {
            'cosmo_index': cosmo_index,
            'cosmology': {'h': h, 'Oc': Oc, 'Om': Om, 'w': w, 'wa': wa, 'sigma8': sigma8},
            'snr': snr,
            'prediction_l1': prediction_l1,
            'ldt_theory_sigma_sq': variables.sigmasq_map,
            'perturbation_theory_sigma_sq': theory_sigmasq,
        }
        if simulation:
            entry.update({
                'sim_pdf_runs': sim_pdf_runs,
                'sim_l1_runs': sim_l1_runs,
                'simvar': simvar,
            })
        results[cosmo_index] = entry
        pred_stats = get_moments(snr * (theory_sigmasq**0.5), prediction_pdf)
        if simulation:
            sim_stats = get_moments(snr * (np.mean(simvar)**0.5), avg_sim_pdf)
            mean_sim, variance_sim, s3_sim, kurtosis_sim, norm_sim = sim_stats
            print(f"simstats: mean: {mean_sim}, variance: {variance_sim}, s3: {s3_sim:.3f}, kurtosis: {kurtosis_sim:.3f}, norm: {norm_sim:.3f}")
        print(f"predstats: mean: {pred_stats[0]}, variance: {pred_stats[1]}, s3: {pred_stats[2]:.3f}, kurtosis: {pred_stats[3]:.3f}, norm: {pred_stats[4]:.3f}")
    plt.tight_layout()
    plt.show()
    return results

if __name__ == '__main__':
    key = f"filter{filter_type}_results_{theta1}_tomobin{tomo_bin}_simulation{simulation}"
          # or an existing dict
    results_fin[key] = main()

