from etc_utils import *
from EXOSIMS.OpticalSystem.MHRS import read_snr_results_from_file,read_snr_results_and_json_from_file


import matplotlib.pyplot as plt
import numpy as np
import os
import astropy.units as u

def plot_noise_histograms(noise_dict, tint,bins=50,detec_noise_dict = None):
    """
    Plot overlaid histograms of summed photon counts for four noise groups across all stars.

    Group 1: Astrophysical background = C_zodi + C_exozodi
    Group 2: Detector noise = C_dark + C_CIC + C_readnoise + C_background_leakage
    Group 3: Starlight leakage = C_local_starlight
    Group 4: Planet signal = C_planet

    Parameters
    ----------
    noise_dict : dict of str -> np.ndarray
        Dictionary with photon counts for different noise terms.
    bins : int
        Number of log-spaced bins for the x-axis (photon counts).
    """
    color_list = ["#ff9900", "#006699", "#6600ff", "pink", "grey"]

    tint_h = tint.to_value(u.h)

    required_keys = [
        "C_zodi", "C_exozodi",
        "C_dark", "C_CIC", "C_readnoise", "C_background_leakage",
        "C_local_starlight", "C_correlated_speckles","C_planet"
    ]
    for key in required_keys:
        if key not in noise_dict:
            raise KeyError(f"Missing key: {key}")

    arrays = {k: np.asarray(noise_dict[k]) for k in required_keys}
    N = len(arrays["C_zodi"])
    for key, arr in arrays.items():
        if len(arr) != N:
            raise ValueError(f"Inconsistent array length for key {key}")

    # Define groups
    group1 = arrays["C_zodi"] + arrays["C_exozodi"]
    group2 = arrays["C_dark"] + arrays["C_CIC"] + arrays["C_readnoise"] + arrays["C_background_leakage"]
    group3 = arrays["C_local_starlight"]
    group4 = arrays["C_planet"]
    group5 = arrays["C_correlated_speckles"]

    # Remove zeros to avoid log(0)
    all_data = np.concatenate([group1, group2, group3, group4, group5])/tint_h
    all_data = all_data[all_data > 0]
    min_val = np.min(all_data)
    max_val = np.max(all_data)
    bin_edges = np.logspace(np.log10(min_val), np.log10(max_val), bins)

    # Plot the photon counts
    plt.figure(figsize=(6, 5))

    plt.hist(group3[group3 > 0]/tint_h, bins=bin_edges, histtype='stepfilled', label="Starlight", linewidth=2,color=color_list[0],alpha=0.4)
    plt.hist(group4[group4 > 0]/tint_h, bins=bin_edges, histtype='stepfilled', linewidth=2, label='Planet (ie, "signal")',color=color_list[1],alpha=0.4)
    plt.hist(group1[group1 > 0]/tint_h, bins=bin_edges, histtype='step', linewidth=2, label="Background",color=color_list[2])
    # plt.hist(group1[group1 > 0]/tint_h, bins=bin_edges, histtype='stepfilled', linewidth=2,color=color_list[2],alpha=0.8)
    # plt.hist(group3[group3 > 0]/tint_h, bins=bin_edges, histtype='step', linewidth=2, label="Starlight",color=color_list[0])
    plt.hist(group5[group5 > 0]/tint_h, bins=bin_edges, histtype='step', linewidth=2, label="Correlated starlight",color=color_list[3])
    # plt.hist(group5[group5 > 0]/tint_h, bins=bin_edges, histtype='stepfilled', linewidth=2,color=color_list[3],alpha=0.8)

    if detec_noise_dict is not None:
        for x_val,key in zip(detec_noise_dict.values(),detec_noise_dict.keys()):
            x_val_rate = x_val/tint_h
            # Annotate with vertical line and text
            # plt.axvline(x=x_val_rate, color='black', linestyle='-', linewidth=1.5)
            plt.plot([x_val_rate, x_val_rate], [24, 30], color='black', linewidth=1.5, linestyle='-')
            plt.text(x_val_rate * 1.05, 27, key, rotation=90, va='center', color='black')
        plt.text(x_val_rate * 0.7, 26, "Detector noise", rotation=90, va='center', color='black')

    plt.xscale("log")
    plt.xlabel("# of Photons / hour")
    plt.ylabel("# of Stars")
    plt.ylim([0,30])
    plt.legend(loc='center right', fontsize=10, frameon=True)
    # plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Remove zeros to avoid log(0)
    all_data = np.concatenate([np.sqrt(group1), np.sqrt(group2), np.sqrt(group3), np.sqrt(group4), group5,group4])
    all_data = all_data[all_data > 0]
    min_val = np.min(all_data)
    max_val = np.max(all_data)
    bin_edges_std = np.logspace(np.log10(min_val), np.log10(max_val), bins)

    # Plot the standard deviations
    plt.figure(figsize=(6, 5))

    # plt.hist(group3[group3 > 0]/tint_h, bins=bin_edges, histtype='stepfilled', label="Starlight", linewidth=2,color=color_list[0],alpha=0.4)
    # plt.hist(group4[group4 > 0]/tint_h, bins=bin_edges, histtype='stepfilled', linewidth=2, label='Planet (ie, "signal")',color=color_list[1],alpha=0.4)
    plt.hist(group4[group4 > 0], bins=bin_edges_std, histtype='stepfilled', linewidth=2,label='Planet (ie, "signal")',color=color_list[1],alpha=0.4)
    plt.hist(np.sqrt(group3[group3 > 0]), bins=bin_edges_std, histtype='stepfilled', linewidth=2, label="$\sigma$ Starlight",color=color_list[0],alpha=0.4)
    plt.hist(np.sqrt(group1[group1 > 0]), bins=bin_edges_std, histtype='step', linewidth=2, linestyle="-", label="$\sigma$ Background",color=color_list[2])
    # The important thing here is that the correlated speckles is not photon noise, so no sqrt!
    plt.hist(group5[group5 > 0], bins=bin_edges_std, histtype='step', linewidth=2, linestyle="-", label="$\sigma$ Correlated starlight",color=color_list[3])
    plt.hist(np.sqrt(group4[group4 > 0]), bins=bin_edges_std, histtype='step', linewidth=2, linestyle="-", label="$\sigma$ Planet",color=color_list[1])


    if detec_noise_dict is not None:
        for x_val,key in zip(detec_noise_dict.values(),detec_noise_dict.keys()):
            x_val_sqrt = np.sqrt(x_val)
            # Annotate with vertical line and text
            plt.plot([x_val_sqrt, x_val_sqrt], [45, 60], color='black', linewidth=1.5, linestyle='-')
            plt.text(x_val_sqrt * 1.05, 55, key, rotation=90, va='center', color='black')
        plt.text(x_val_sqrt * 0.7, 51, "Detector noise", rotation=90, va='center', color='black')

    plt.xscale("log")
    plt.xlabel(r"# of Photons (T$_{int}$"+" = {0:.0f} h)".format(tint_h))
    plt.ylabel("# of Stars")
    plt.ylim([0,60])
    plt.legend(loc='upper right', fontsize=10, frameon=True)
    # plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    fig_dir = "/fast/jruffio/data/exosims/exosims_samples/figures"

    # R_list = [20,50,140,400,1000,3000,10000]
    R_list = [20,140,1000,10000]
    # override_local_starlight_flux_ratio_list = [1e-10,1e-12]
    override_local_starlight_flux_ratio = 1e-10
    # ppFact_Char_list = [1,0.1,0.01,0.001]
    ppFact_Char = 0.1

    output_filename0 = "/fast/jruffio/data/exosims/exosims_samples/20250621_output/20250621_MHRS_Romandetecnoise_SNR_outputs_paper.txt"
    # output_filename0 = "/fast/jruffio/data/exosims/exosims_samples/20250621_output/20250621_MHRS_Romandetecnoise_undersamp_SNR_outputs_paper.txt"
    # output_filename0 = "/fast/jruffio/data/exosims/exosims_samples/20250621_output/20250621_MHRS_10xbetterRoman_SNR_outputs_paper.txt"
    # output_filename0 = "/fast/jruffio/data/exosims/exosims_samples/20250621_output/20250621_MHRS_nodetecnoise_SNR_outputs_paper.txt"
    det_label = os.path.basename(output_filename0).split("_")[2]

    SNR_dict_list = []
    for R in R_list:
        output_filename = output_filename0.replace(".txt","_R{0}_starlight{1:.1e}_corr{2:.1e}.txt".format(R,override_local_starlight_flux_ratio,override_local_starlight_flux_ratio*ppFact_Char))
        print(output_filename)
        SNR_dict, config_dict = read_snr_results_and_json_from_file(output_filename)
        tint = config_dict["observingModes"][1]["intTime"] *u.h
        SNR_dict_list.append(SNR_dict)

    detec_noise_dict = {}
    for R_id,R in enumerate(R_list):
        total_detec_noise = SNR_dict_list[R_id]["C_dark"] + SNR_dict_list[R_id]["C_CIC"] + SNR_dict_list[R_id]["C_readnoise"] + SNR_dict_list[R_id]["C_background_leakage"]
        # print(np.nanmedian(total_detec_noise))
        if R >= 1000:
            mykey = "R={0:.0f}k".format(R/1000.)
        else:
            mykey = "R={0}".format(R)
        detec_noise_dict[mykey] = np.nanmedian(total_detec_noise)

    plot_noise_histograms(SNR_dict_list[0],tint,detec_noise_dict=detec_noise_dict)

    plt.figure(1)
    out_filename = os.path.join(fig_dir, "noise_budget_signal_histograms_{0}_starlight{1:.1e}_corr{2:.1e}.png".format(det_label,override_local_starlight_flux_ratio,override_local_starlight_flux_ratio*ppFact_Char))
    print("Saving " + out_filename)
    plt.savefig(out_filename, dpi=300)
    plt.savefig(out_filename.replace(".png", ".pdf"))
    plt.figure(2)
    out_filename = os.path.join(fig_dir, "noise_budget_stddev_histograms{0}_starlight{1:.1e}_corr{2:.1e}.png".format(det_label,override_local_starlight_flux_ratio,override_local_starlight_flux_ratio*ppFact_Char))
    print("Saving " + out_filename)
    plt.savefig(out_filename, dpi=300)
    plt.savefig(out_filename.replace(".png", ".pdf"))
    plt.show()




