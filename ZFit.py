import numpy as np
import uproot
import zfit
import mplhep as hep
import matplotlib.pyplot as plt
from pathlib import Path
import os
import boost_histogram as bh

# Import PDF classes directly to handle potential pathing issues in older zfit versions
from zfit.pdf import CrystalBall, Gauss, Exponential, SumPDF

# --- 0. Configuration & Constants ---
# Using a dictionary for configuration is a good practice in Python.
CONFIG = {
    "M_Lc": 2286.46,
    "m_range_min": 2146.0,
    "m_range_max": 2426.0,
    "n_bins": 100,
    "bdt_cut": 0.15,
    "data_dir": Path(__file__).parent.parent.parent / 'Final' / 'InputData',
    "figs_dir": Path("./figs_python"),
    "mc_file": "xgb_Lc2pemu_MC.root",
    "data_file": "xgb_Lc2pemu_DATA_osign_noBrem.root",
    "pmumu_file": "xgb_Lc2pmumu_DATA.root",
    "tree_name": "DecayTree"
}

# --- 2. Helper Functions for Plotting and Data Loading ---
def load_branch(file_name, branch_name):
    """Loads a single branch from a ROOT file."""
    full_path = CONFIG["data_dir"] / file_name
    with uproot.open(full_path) as f:
        return f[CONFIG["tree_name"]][branch_name].array(library="np")

def plot_fit(data, obs, model, result, filename, n_bins=100, title="Fit Result", total_yield=None):
    """Creates and saves a publication-quality plot of the fit result with pulls."""
    hep.style.use("CMS")
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)

    ax1 = fig.add_subplot(gs[0])
    plot_range = (obs.limits[0][0], obs.limits[1][0])
    bin_width = (plot_range[1] - plot_range[0]) / n_bins

    # --- Final Debugging and Conversion Snippet ---
    print(f"\n--- Debugging plot_fit for title: '{title}' ---")
    print(f"Initial 'data' object type: {type(data)}")

    # Step 1: Convert to a NumPy array, handling zfit.Data or other types.
    if hasattr(data, 'value'):
        data_np = data.value.numpy()
    else:
        data_np = np.asarray(data) # Forcefully convert to a numpy array

    print(f"Array type after conversion: {type(data_np)}")
    print(f"Shape of array BEFORE flattening: {data_np.shape}")

    # Step 2: Flatten the array to ensure it is 1-dimensional.
    data_flat = data_np.flatten()
    
    print(f"Shape of array AFTER flattening: {data_flat.shape}")
    print("-------------------------------------------------")

    # --- Histogramming ---
    '''counts, bin_edges = np.histogram(data_flat, bins=n_bins, range=plot_range)
    hep.histplot(counts, bin_edges, ax=ax1, yerr=True, histtype='errorbar', color='black', label='Data')
    x_plot = np.linspace(plot_range[0], plot_range[1], 500)'''

    try:
        # 1. Create a histogram object from the boost-histogram library
        hist = bh.Histogram(
            bh.axis.Regular(bins=n_bins, start=plot_range[0], stop=plot_range[1])
        )

        # 2. Fill the histogram with the data
        hist.fill(data_flat)

        # 3. Get the counts and bin edges from the histogram object
        counts = hist.view()
        bin_edges = hist.axes[0].edges

    except Exception as e:
        print(f"CRITICAL ERROR: Histogramming failed even with boost-histogram: {e}")
        # As a fallback, prevent the script from crashing
        counts, bin_edges = np.array([0]), np.array([0, 1])

    # Plot the histogram data
    hep.histplot(counts, bin_edges, ax=ax1, yerr=True, histtype='errorbar', color='black', label='Data')

    # --- Total Model Line ---
    # If model is already extended (S+B fit), use it. If not (MC fit), extend it now.
    if model.is_extended:
        y_total = model.ext_pdf(x_plot) * bin_width
        plot_model = model
    else:
        if total_yield is None:
            raise ValueError("total_yield must be provided for non-extended models.")
        extended_model = model.create_extended(zfit.Parameter("N_plot", total_yield))
        y_total = extended_model.ext_pdf(x_plot) * bin_width
        plot_model = extended_model

    ax1.plot(x_plot, y_total, label='Total Fit', color='royalblue', linewidth=3)

    # --- Component Lines ---
    colors = ['crimson', 'darkorange', 'limegreen', 'purple', 'cyan']
    styles = [':', '--', '-.', (0, (3, 5, 1, 5)), '-']

    # Case 1: Model is a sum of already extended PDFs (S+B fit)
    if all(p.is_extended for p in model.pdfs):
        for i, comp in enumerate(model.pdfs):
            y_comp = comp.ext_pdf(x_plot) * bin_width
            label = comp.name.replace('ext_', '')
            ax1.plot(x_plot, y_comp, label=label, color=colors[i % len(colors)], linestyle=styles[i % len(styles)], linewidth=2.5)

    # Case 2: Model is a sum of non-extended PDFs with fractions (MC signal fit)
    else:
        frac_params = model.fracs
        frac_values = [p.value() for p in frac_params]
        all_fracs = frac_values + [1.0 - sum(frac_values)] # Account for the last fraction

        for i, comp in enumerate(model.pdfs):
            # Use .pdf() and scale manually
            y_comp = comp.pdf(x_plot) * total_yield * all_fracs[i] * bin_width
            label = comp.name
            ax1.plot(x_plot, y_comp, label=label, color=colors[i % len(colors)], linestyle=styles[i % len(styles)], linewidth=2.5)

    ax1.legend(fontsize='large')
    ax1.set_ylabel(f"Events / ({bin_width:.1f} MeV)")
    ax1.set_title(title, fontsize=18)
    ax1.set_xlim(obs.limits)
    ax1.set_xticklabels([])

    # Pull plot
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    from zfit.visualization import plot_pulls
    # Use the extended model for pulls to get correct normalization
    plot_pulls(result, ax=ax2, n_bins=n_bins, model=plot_model) 
    ax2.set_xlabel(f"Mass [{obs.unit or 'MeV'}]")

    plt.tight_layout()

    save_path = CONFIG["figs_dir"] / f"{filename}.pdf"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

    ax1.set_yscale('log')
    ax1.set_ylim(bottom=0.5)
    save_path_log = CONFIG["figs_dir"] / f"{filename}_log.pdf"
    plt.savefig(save_path_log)
    print(f"Log scale plot saved to {save_path_log}")

    plt.close(fig)

# --- 3. Analysis Steps ---
def calculate_punzi_fom():
    """Calculates and plots the Punzi Figure of Merit."""
    print("\n--- Calculating Punzi FoM ---")
    bdt_sig = load_branch(CONFIG["mc_file"], "bdt_score")
    bdt_bkg = load_branch(CONFIG["data_file"], "bdt_score")

    bdt_bins = np.linspace(0, 1, 101)
    h_sig_total, _ = np.histogram(bdt_sig, bins=bdt_bins)
    h_bkg_total, _ = np.histogram(bdt_bkg, bins=bdt_bins)

    ns_all = h_sig_total.sum()
    
    effs = []
    nbkgs = []
    foms = []
    bdt_cuts = bdt_bins[:-1]

    for i in range(len(bdt_cuts)):
        ns_pass = h_sig_total[i:].sum()
        nb_pass = h_bkg_total[i:].sum()
        
        eff = ns_pass / ns_all if ns_all > 0 else 0
        # The factor 50 / mass_range is from the C++ code to scale the background
        bkg_ratio = 50.0 / (CONFIG["m_range_max"] - CONFIG["m_range_min"])
        nb = bkg_ratio * nb_pass
        
        fom = eff / (1.5 + np.sqrt(nb)) if nb > 0 else 0
        
        effs.append(eff)
        nbkgs.append(np.sqrt(nb))
        foms.append(fom)

    # Plotting
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(bdt_cuts, foms, label='Punzi FoM', color='red')
    ax.set_xlabel("BDT Cut")
    ax.set_ylabel("Punzi FoM = $\\epsilon / (1.5 + \\sqrt{N_b})$")
    ax.set_title("Punzi Figure of Merit Scan")
    ax.grid(True)
    ax.legend()
    plt.savefig(CONFIG["figs_dir"] / "fom_punzi.pdf")
    plt.close(fig)
    print("Punzi FoM plot saved.")
    
    # Find optimal cut
    optimal_idx = np.argmax(foms)
    optimal_cut = bdt_cuts[optimal_idx]
    print(f"Optimal BDT cut found at: {optimal_cut:.3f} (using pre-defined {CONFIG['bdt_cut']})")
    return CONFIG['bdt_cut'] # Use the pre-defined cut from the C++ script

def fit_signal_shape(mass_mc, obs):
    """Fits the MC signal shape with a CB+Gauss model."""
    print("\n--- Fitting Signal MC Shape ---")
    data_mc = zfit.Data.from_numpy(obs=obs, array=mass_mc)
    
    # Parameters for CB + Gauss
    m_lc = CONFIG["M_Lc"]
    cbmean = zfit.Parameter("cbmean", m_lc, m_lc - 50, m_lc + 50)
    cbsig = zfit.Parameter("cbsig", 8.0, 5.0, 12.0)
    cbalpha = zfit.Parameter("cbalpha", 0.3, -0.1, 20.0)
    cbn = zfit.Parameter("cbn", 3.0, 0.1, 30.0)
    
    gmean = zfit.Parameter("gmean", m_lc - 50, m_lc - 100, m_lc + 100)
    gsig = zfit.Parameter("gsig", 90.0, 10.0, 150.0)
    
    frac = zfit.Parameter("frac_sig", 0.8, 0.0, 1.0)
    
    # Model
    cb_pdf = zfit.pdf.CrystalBall(mu=cbmean, sigma=cbsig, alpha=cbalpha, n=cbn, obs=obs)
    gauss_pdf = zfit.pdf.Gauss(mu=gmean, sigma=gsig, obs=obs)
    model_sig = zfit.pdf.SumPDF([cb_pdf, gauss_pdf], fracs=frac)
    
    # Fit
    loss = zfit.loss.UnbinnedNLL(model=model_sig, data=data_mc)
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(loss)
    result.hesse()
    
    print("Signal Shape Fit Results:")
    print(result)
    
    # Plot
    # NEW Line 200
    plot_fit(mass_mc, obs, model_sig, result, "SignalShapeCB_Gauss", title="MC Signal Shape Fit", total_yield=len(mass_mc))
    
    return result.params

def main():
    """Main analysis workflow."""
    # --- Setup ---
    if not CONFIG["figs_dir"].exists():
        CONFIG["figs_dir"].mkdir()

    # --- Step 1: Punzi FoM and BDT Cut ---
    bdt_cut = calculate_punzi_fom()

    # --- Step 2: Load and cut data ---
    print("\n--- Loading and cutting data ---")
    mass_mc = load_branch(CONFIG["mc_file"], "Lc_MM")
    bdt_mc = load_branch(CONFIG["mc_file"], "xgb_prediction")
    mass_mc = mass_mc[bdt_mc > bdt_cut]

    mass_data = load_branch(CONFIG["data_file"], "Lc_MM")
    bdt_data = load_branch(CONFIG["data_file"], "xgb_prediction")
    mass_data = mass_data[bdt_data > bdt_cut]

    mass_pmumu = load_branch(CONFIG["pmumu_file"], "Lc_MM")
    bdt_pmumu = load_branch(CONFIG["pmumu_file"], "xgb_prediction")
    mass_pmumu = mass_pmumu[bdt_pmumu > bdt_cut]

    print(f"Events after cut: MC={len(mass_mc)}, Data={len(mass_data)}, pmumu={len(mass_pmumu)}")

    # --- Step 3: Define observable space ---
    obs = zfit.Space('mass', limits=(CONFIG["m_range_min"], CONFIG["m_range_max"]))

    # --- Step 4: Fit signal shape from MC and fix parameters ---
    sig_shape_params = fit_signal_shape(mass_mc, obs)

    # --- Step 5: Define the full S+B model for the final fit ---
    print("\n--- Defining Final S+B Model for Data ---")
    
    # --- Signal Component ---
    # Parameters are created again, but their values will be fixed from the MC fit.
    cbmean_s = zfit.Parameter("cbmean_s", sig_shape_params['cbmean']['value'])
    cbsig_s = zfit.Parameter("cbsig_s", sig_shape_params['cbsig']['value'])
    cbalpha_s = zfit.Parameter("cbalpha_s", sig_shape_params['cbalpha']['value'])
    cbn_s = zfit.Parameter("cbn_s", sig_shape_params['cbn']['value'])
    gmean_s = zfit.Parameter("gmean_s", sig_shape_params['gmean']['value'])
    gsig_s = zfit.Parameter("gsig_s", sig_shape_params['gsig']['value'])
    frac_s = zfit.Parameter("frac_s", sig_shape_params['frac_sig']['value'])
    
    # Fix the shape parameters to the values from the MC fit
    for p in [cbmean_s, cbsig_s, cbalpha_s, cbn_s, gmean_s, gsig_s, frac_s]:
        p.floating = False

    cb_s = zfit.pdf.CrystalBall(mu=cbmean_s, sigma=cbsig_s, alpha=cbalpha_s, n=cbn_s, obs=obs)
    gauss_s = zfit.pdf.Gauss(mu=gmean_s, sigma=gsig_s, obs=obs)
    model_s = zfit.pdf.Sum([cb_s, gauss_s], fracs=frac_s)
    n_sig = zfit.Parameter("N_sig", 150, -20, 1e6)
    ext_model_s = model_s.create_extended(n_sig)
    ext_model_s.set_norm_range(obs.limits)

    # --- Peaking Background Component (from pmumu, hard-coded from C++ for simplicity) ---
    # In a full analysis, one would fit the pmumu data first. Here we use the values from the C++ script.
    g_pcbmean = zfit.Parameter("g_pcbmean", 2.260e+03, floating=False)
    g_pcbsig = zfit.Parameter("g_pcbsig", 7.826e+00, floating=False)
    g_pcbalpha = zfit.Parameter("g_pcbalpha", 7.990e-01, floating=False)
    g_pcbn = zfit.Parameter("g_pcbn", 1.950e+01, floating=False)
    g_pgmean = zfit.Parameter("g_pgmean", 2.254e+03, floating=False)
    g_pgsig = zfit.Parameter("g_pgsig", 6.879e+01, floating=False)
    g_pfrac_val = zfit.Parameter("g_pfrac_val", 9.508e-01, floating=False)
    
    cb_bkg1 = zfit.pdf.CrystalBall(mu=g_pcbmean, sigma=g_pcbsig, alpha=g_pcbalpha, n=g_pcbn, obs=obs)
    gauss_bkg1 = zfit.pdf.Gauss(mu=g_pgmean, sigma=g_pgsig, obs=obs)
    model_bkg1 = zfit.pdf.Sum([cb_bkg1, gauss_bkg1], fracs=g_pfrac_val)
    n_bkg1 = zfit.Parameter("N_bkg_peak", 300, 0, 1e6)
    ext_model_bkg1 = model_bkg1.create_extended(n_bkg1)
    ext_model_bkg1.set_norm_range(obs.limits)

    # --- Combinatorial Background Component (Exponential) ---
    lambda_bkg2 = zfit.Parameter("lambda_bkg2", -0.001, -0.1, 0.1)
    model_bkg2 = zfit.pdf.Exponential(lambda_=lambda_bkg2, obs=obs)
    n_bkg2 = zfit.Parameter("N_bkg_comb", 5000, 0, 1e6)
    ext_model_bkg2 = model_bkg2.create_extended(n_bkg2)
    ext_model_bkg2.set_norm_range(obs.limits)

    # --- Full Model ---
    full_model = zfit.pdf.Sum([ext_model_s, ext_model_bkg1, ext_model_bkg2])
    
    # --- Step 6: Fit the data ---
    print("\n--- Performing Final Fit on Data ---")
    data_final = zfit.Data.from_numpy(obs=obs, array=mass_data)
    loss_sb = zfit.loss.ExtendedUnbinnedNLL(model=full_model, data=data_final)
    minimizer = zfit.minimize.Minuit()
    result_sb = minimizer.minimize(loss_sb)
    result_sb.hesse()
    
    print("S+B Fit Results:")
    print(result_sb)
    
    # --- Step 7: Calculate Significance ---
    print("\n--- Calculating Significance ---")
    # To get NLL_b, we fit again with N_sig fixed to 0
    n_sig.floating = False
    n_sig.set_value(0)
    result_b = minimizer.minimize(loss_sb)
    n_sig.floating = True # Set it back for plotting
    
    nll_sb = result_sb.fmin
    nll_b = result_b.fmin
    
    # The factor of 2 is for the likelihood ratio test (Wilks' theorem)
    delta_nll = nll_b - nll_sb
    significance = np.sqrt(2 * delta_nll) if delta_nll > 0 else 0
    
    print(f"NLL (S+B): {nll_sb:.2f}")
    print(f"NLL (B only): {nll_b:.2f}")
    print(f"Delta NLL: {delta_nll:.2f}")
    print(f"Significance = sqrt(2 * DeltaNLL) = {significance:.2f} sigma")

    # --- Step 8: Plot final result ---
    plot_fit(mass_data, obs, full_model, result_sb, "Lc2pemu_mass_fit", n_bins=CONFIG["n_bins"], title="Fit to $L_c \\to p e \\mu$ Candidates")


if __name__ == "__main__":
    main()

