from etc_utils import *
from EXOSIMS.OpticalSystem.MHRS import read_snr_results_from_file
import matplotlib.pyplot as plt
import numpy as np
from histogram_violin import histogram_violin
import astropy.units as u
import os

def plot_snr_violin_panels_3x3(SNR_dict_table, R_list, row_labels=None, col_labels=None, snr_key_group="SNR_O2_corr"):
    """
    Plot 3x3 grid of violin plots for selected SNR keys.
    """

    logR = np.log10(R_list)

    N_rows = len(SNR_dict_table)//3

    fig, axes = plt.subplots(
        N_rows, 3, figsize=(12, 12),
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
            ax.plot(logR4data, p1, color=color, lw=1, linestyle='--', label='25th and 75th percentile' if idx == 0 and k == 0 else None)
            ax.plot(logR4data, p3, color=color, lw=1, linestyle='--', label= None)

            # if k == 0 and idx % 3 == 2:
            #     all_values = np.concatenate(data)
            #     p85 = np.nanpercentile(all_values, 75)
            ax.set_ylim(0, 6.5)

        if idx <3:
            # print(idx)
            plt.sca(ax)
            plt.title(col_labels[idx %3])
            # ax.text(0.5, 0.99, "\u2193"+col_labels[idx %3]+"\u2193", transform=ax.transAxes,
            #         ha='center', va='top', fontsize=12)

        if idx // 3 == N_rows-1:
            ax.set_xlabel("Spectral Resolution (R=$\lambda/\Delta\lambda$)", fontsize=12)
            ax.set_xticks(logR)
            myxtickslabels = []
            for r in R_list:
                if r >= 1000:
                    myxtickslabels.append("{0:.0f}k".format(r/1000.))
                else:
                    myxtickslabels.append(str(r))
            ax.set_xticklabels(myxtickslabels, fontsize=12)
        else:
            ax.tick_params(labelbottom=False)

        if idx % 3 == 0:
            if row_labels is not None:
                ax.set_ylabel(row_labels[idx//3]+"\nS/N", fontsize=12)
            else:
                ax.set_ylabel("S/N", fontsize=12)
            ax.tick_params(labelleft=True, labelsize=12)
        else:
            ax.tick_params(labelleft=False)

        ax.tick_params(axis='y', labelsize=12)
        # ax.grid(True)

        # Add legend to top-right panel only
        if idx == 0:
            ax.legend(loc='upper left', fontsize=10, frameon=True)

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.08, top=0.92)

if __name__ == "__main__":
    fig_dir = "/fast/jruffio/data/exosims/exosims_samples/figures"

    R_list = [20,50,140,400,1000,3000,10000,30000]
    # R_list = [100,1000]

    override_local_starlight_flux_ratio_list = [1e-10,1e-12]
    override_local_starlight_flux_ratio = override_local_starlight_flux_ratio_list[0]
    ppFact_Char_list = [0.1,0.01,0.001]
    # ppFact_Char = ppFact_Char_list[1]
    output_filelist0 = []
    output_filelist0.append("/fast/jruffio/data/exosims/exosims_samples/20250621_output/20250621_MHRS_Romandetecnoise_SNR_outputs_paper")
    output_filelist0.append("/fast/jruffio/data/exosims/exosims_samples/20250621_output/20250621_MHRS_Romandetecnoise_undersamp_SNR_outputs_paper")
    output_filelist0.append("/fast/jruffio/data/exosims/exosims_samples/20250621_output/20250621_MHRS_10xbetterRoman_SNR_outputs_paper")
    output_filelist0.append("/fast/jruffio/data/exosims/exosims_samples/20250621_output/20250621_MHRS_nodetecnoise_SNR_outputs_paper")
    detector_labels = ['Roman-analog detector','Roman-analog\n& undersampled',"10x better detector","No detector noise"]
    SNR_dict_table = []
    for det_label,output_filename0 in zip(detector_labels,output_filelist0):
        col_labels = []
        for ppFact_Char in ppFact_Char_list:
            SNR_dict_list = []
            for R in R_list:
                print(output_filename0,override_local_starlight_flux_ratio,R)
                output_filename = output_filename0+ "_R{0}_starlight{1:.1e}_corr{2:.1e}.txt".format(R,override_local_starlight_flux_ratio,override_local_starlight_flux_ratio*ppFact_Char)
                try:
                    SNR_dict = read_snr_results_from_file(output_filename)
                    SNR_dict_list.append(SNR_dict)
                except:
                     Warning("missing file "+output_filename)
            SNR_dict_table.append(SNR_dict_list)
            label = "{0:.0e} starlight ; {1:.0e} correlated".format(override_local_starlight_flux_ratio,override_local_starlight_flux_ratio*ppFact_Char)
            col_labels.append(label)

    # plot_snr_violin_panels_3x3(SNR_dict_table, R_list, label_list, snr_key_group="SNR_O2_corr")
    plot_snr_violin_panels_3x3(SNR_dict_table, R_list, detector_labels, col_labels, snr_key_group=["SNR_O2_corr", "SNR_O2_uncorr_small_scale"])

    det_label4file = os.path.basename(output_filename0).split("_")[2]
    out_filename = os.path.join(fig_dir, "PSDD_corr_starlight{0:.1e}.png".format(override_local_starlight_flux_ratio))
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



