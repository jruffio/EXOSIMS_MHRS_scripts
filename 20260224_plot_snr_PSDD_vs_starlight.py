from etc_utils import *
from EXOSIMS.OpticalSystem.MHRS import read_snr_results_from_file
import matplotlib.pyplot as plt
import numpy as np
from histogram_violin import histogram_violin
import astropy.units as u
import os
import math
import matplotlib.patheffects as PathEffects

def to_latex_sci(num, decimals=2):
    """Convert a number to a LaTeX scientific notation string for Matplotlib."""
    if num == 0:
        return r"$0$"

    # Extract mantissa and exponent from Python's own formatting to avoid float errors
    s = f"{num:.{decimals}e}"
    mantissa, exp = s.split("e")
    exp = int(exp)
    mantissa = float(mantissa)

    if mantissa == 1.0:
        return rf"$10^{{{exp}}}$"
    else:
        return rf"${mantissa:.{decimals}g} \times 10^{{{exp}}}$"


def plot_snr_violin_panels_3x3(SNR_dict_table, R_list, row_labels=None, col_labels=None, axis_labels=None,snr_key_group="SNR_O2_corr"):
    """
    Plot 3x3 grid of violin plots for selected SNR keys.
    """
    fontsize = 12
    logR = np.log10(R_list)

    N_rows = len(SNR_dict_table)//3

    fig, axes = plt.subplots(
        N_rows, 3, figsize=(12, 8),
        sharex=True, sharey=True,
        gridspec_kw={'wspace': 0, 'hspace': 0}
    )
    axes = axes.flatten()


    color_list = ["#006699","#ff9900", "#6600ff", "purple", "grey"]
    if isinstance(snr_key_group, str):
        snr_key_group = [snr_key_group]

    for idx, (ax, SNR_dict_list) in enumerate(zip(axes, SNR_dict_table)):
        for k, (snr_key, color) in enumerate(zip(snr_key_group, color_list)):
            data = []
            logR4data = []
            p1, p2, p3 = [], [], []

            for snr_dict, _logR in zip(SNR_dict_list, logR):
                snr_values = snr_dict[snr_key]
                _d = snr_values[~np.isnan(snr_values)]
                if len(_d) != 0:
                    data.append(_d)
                    logR4data.append(_logR)
                    p1.append(np.nanpercentile(_d, 25))
                    p2.append(np.nanpercentile(_d, 50))
                    p3.append(np.nanpercentile(_d, 75))

            if k == 0:
                # # Violin plot
                # parts = ax.violinplot(
                #     data, positions=logR4data,
                #     showmeans=False, showmedians=False,
                #     showextrema=False,
                #     # quantiles=[[0.1, 0.9]] * len(data),
                #     widths=0.3
                # )
                for pos, d in zip(logR4data, data):
                    histogram_violin(ax, d, position=pos, width=0.2, bins=300, color='#9eccf7')

            # Plot percentile lines
            if idx == 0 and k == 0:
                _label = '50th percentile'
            elif idx == 0 and k == 1:
                _label = 'High-pass filtered'
            ax.plot(logR4data, p2, color=color, lw=2, linestyle='-', marker='o', label=_label)
            _argmax = np.argmax(p2)
            plt.sca(ax)
            if k==0:
                txt = plt.text(logR4data[_argmax], p2[_argmax]+0.5, f"{p2[_argmax]:.1f}", color=color, ha='center', va='bottom', fontsize=fontsize)
            else:
                txt = plt.text(logR4data[_argmax], p2[_argmax]-0.5, f"{p2[_argmax]:.1f}", color=color, ha='center', va='top', fontsize=fontsize)
            txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])
            ax.plot(logR4data, p1, color=color, lw=1, linestyle='--', label='25th and 75th percentile' if idx == 0 and k == 0 else None)
            ax.plot(logR4data, p3, color=color, lw=1, linestyle='--', label= None)
            if axis_labels is not None:
                txt = ax.text(0.5, 0.97, axis_labels[idx], transform=ax.transAxes, ha='center', va='top', fontsize=fontsize)
                txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])

            # if k == 0 and idx % 3 == 2:
            #     all_values = np.concatenate(data)
            #     p85 = np.nanpercentile(all_values, 75)
            ax.set_ylim(0, 17)
            ax.set_yticks([0,5,10,15])

        if idx <3:
            # print(idx)
            plt.sca(ax)
            plt.title(col_labels[idx %3])
            # ax.text(0.5, 0.99, "\u2193"+col_labels[idx %3]+"\u2193", transform=ax.transAxes,
            #         ha='center', va='top', fontsize=12)

        if idx // 3 == N_rows-1:
            ax.set_xlabel("Spectral Resolution (R=$\lambda/\Delta\lambda$)", fontsize=fontsize)
            ax.set_xticks(logR)
            myxtickslabels = []
            for r in R_list:
                if r >= 1000:
                    myxtickslabels.append("{0:.0f}k".format(r/1000.))
                else:
                    myxtickslabels.append(str(r))
            ax.set_xticklabels(myxtickslabels, fontsize=fontsize)
        else:
            ax.tick_params(labelbottom=False)

        if idx % 3 == 0:
            if row_labels is not None:
                ax.set_ylabel(row_labels[idx//3]+"\nS/N", fontsize=fontsize)
            else:
                ax.set_ylabel("S/N", fontsize=fontsize)
            ax.tick_params(labelleft=True, labelsize=fontsize)
        else:
            ax.tick_params(labelleft=False)

        ax.tick_params(axis='y', labelsize=fontsize)
        # ax.grid(True)

        # Add legend to top-right panel only
        if idx == 0:
            ax.legend(loc='lower left',  # anchor point of the legend box itself
                    bbox_to_anchor=(0.02, 0.4),  # position in axes fraction coords
                    bbox_transform=ax.transAxes, # interpret coords as axes fractions
                    fontsize=fontsize, frameon=True)

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.08, top=0.92)

if __name__ == "__main__":
    fig_dir = "/fast/jruffio/data/exosims/exosims_samples/figures"

    R_list = [20,50,140,400,1000,3000,10000,30000]
    # R_list = [100,1000]

    contrast_floor = 1e-9
    contrast_floor_list = [1e-9,5e-10,1e-10]
    ppFact_Char_list = [0.1,0.01,0.001]
    # ppFact_Char = ppFact_Char_list[1]
    # output_filename0 = "/fast/jruffio/data/exosims/exosims_samples/20260224_output/20260224_MHRS_emccd_DC1e-4_SNR_outputs_paper"
    # output_filename0 = "/fast/jruffio/data/exosims/exosims_samples/20260224_output/20260224_MHRS_emccd_DC3e-5_SNR_outputs_paper"
    output_filename0 = "/fast/jruffio/data/exosims/exosims_samples/20260224_output/20260224_MHRS_emccd_DC3e-5_undersamp_SNR_outputs_paper"
    # output_filename0 = "/fast/jruffio/data/exosims/exosims_samples/20260224_output/20260224_MHRS_emccd_DC0_SNR_outputs_paper"
    row_labels = [to_latex_sci(contrast_floor)+" starlight" for contrast_floor in contrast_floor_list]
    SNR_dict_table = []
    axis_labels = []
    for row_label,contrast_floor in zip(row_labels,contrast_floor_list):
        col_labels = []
        for ppFact_Char in ppFact_Char_list:
            SNR_dict_list = []
            for R in R_list:
                print(output_filename0,contrast_floor,R)
                output_filename = output_filename0+ "_R{0}_starlight{1:.1e}_corr{2:.1e}.txt".format(R,contrast_floor,contrast_floor*ppFact_Char)
                try:
                    SNR_dict = read_snr_results_from_file(output_filename)
                    SNR_dict_list.append(SNR_dict)
                except:
                     Warning("missing file "+output_filename)
            SNR_dict_table.append(SNR_dict_list)
            col_labels.append(to_latex_sci(ppFact_Char)+" post.-proc. gain")
            axis_labels.append(to_latex_sci(contrast_floor*ppFact_Char)+" correlated")


    plot_snr_violin_panels_3x3(SNR_dict_table, R_list, row_labels, col_labels, axis_labels=axis_labels,snr_key_group=["SNR_H2O_corr", "SNR_H2O_uncorr_small_scale"])

    det_label4file = os.path.basename(output_filename0).split("_")[2]
    out_filename = os.path.join(fig_dir, "PSDD_O2_corr_DC3e-5_undersamp.png")
    print("Saving " + out_filename)
    plt.savefig(out_filename, dpi=300)
    plt.savefig(out_filename.replace(".png", ".pdf"))

    plot_snr_violin_panels_3x3(SNR_dict_table, R_list, row_labels, col_labels,  axis_labels=axis_labels,snr_key_group=["SNR_H2O_corr"])

    det_label4file = os.path.basename(output_filename0).split("_")[2]
    out_filename = os.path.join(fig_dir, "PSDD_O2_corr_noHPF_DC3e-5_undersamp.png")
    print("Saving " + out_filename)
    plt.savefig(out_filename, dpi=300)
    plt.savefig(out_filename.replace(".png", ".pdf"))

    # plot_snr_violin_panels_3x3(SNR_dict_table, R_list, detector_labels, col_labels, snr_key_group=["SNR_O2_ignore_corr"])
    #
    # det_label4file = os.path.basename(output_filename0).split("_")[2]
    # out_filename = os.path.join(fig_dir, "PSDD_ignore_corr_starlight{0:.1e}.png".format(override_local_starlight_flux_ratio))
    # print("Saving " + out_filename)
    # plt.savefig(out_filename, dpi=300)
    # plt.savefig(out_filename.replace(".png", ".pdf"))
    plt.show()



