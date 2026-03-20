

from EXOSIMS.OpticalSystem.MHRS import write_snr_results_to_file, read_snr_results_from_file
import matplotlib.pyplot as plt
import numpy as np
import os
from histogram_violin import histogram_violin
import warnings


def plot_snr_violin_panels(SNR_dict_list, R_list,label=None,plot_hpf_snr = False, Ty=False):
    """
    Plot violin plots of SNR values for all stars across different spectral resolutions.
    Each panel may show the distribution of one or more SNR metrics, as specified in snr_keys.

    Parameters
    ----------
    SNR_dict_list : list of dict
        List of SNR result dictionaries, one per spectral resolution.

    R_list : list of int or float
        Spectral resolution values corresponding to each entry in SNR_dict_list.
    label : str, optional
        A label to add to the titles of the panels for context (e.g., detector type or noise level).
    plot_hpf_snr : bool, optional
        If True, include high-pass filtered SNR metrics in the plots. If False, only include the standard SNR metrics.
    Ty : bool, optional
        If True, swap labels of H2O and O2.
    """
    fontsize = 16
    if Ty:
        if plot_hpf_snr:
            snr_keys = [
                "SNR_all_avg_per_bin",
                ["SNR_all_corr","SNR_all_uncorr_small_scale"],
                ["SNR_H2O_corr", "SNR_H2O_uncorr_small_scale"],
                ["SNR_O2_corr","SNR_O2_uncorr_small_scale"]
            ]
        else:
            snr_keys = [
                "SNR_all_avg_per_bin",
                "SNR_all_corr",
                "SNR_H2O_corr",
                "SNR_O2_corr"
            ]
    else:
        if plot_hpf_snr:
            snr_keys = [
                "SNR_all_avg_per_bin",
                ["SNR_all_corr","SNR_all_uncorr_small_scale"],
                ["SNR_O2_corr", "SNR_O2_uncorr_small_scale"],
                ["SNR_H2O_corr","SNR_H2O_uncorr_small_scale"]
            ]
        else:
            snr_keys = [
                "SNR_all_avg_per_bin",
                "SNR_all_corr",
                "SNR_O2_corr",
                "SNR_H2O_corr"
            ]
    titles = [
        "Average S/N per bin\n",
        'Template matching S/N \nIncluding all molecules & broadband\n',
        "S/N of O2 only\n",
        "S/N of H2O only\n"
    ]
    snr_lims = [[0,25],[0,120],[0,20],[0,20]]
    # if label is not None:
    #     titles = [title+label for title in titles]

    logR = np.log10(R_list)
    R_list_xaxis = [R for R in R_list if R != 400]
    logR_xaxis = np.log10(R_list_xaxis)

    fig, axes = plt.subplots(1, len(snr_keys), figsize=(4 * len(snr_keys), 4.5))  # Adapt width
    if len(snr_keys) == 1:
        axes = [axes]  # Ensure axes is always iterable

    # List of colors used in the plots
    color_list = ["#006699","#ff9900", "#6600ff", "purple", "grey"]
    for idx, (ax, snr_key_group, title) in enumerate(zip(axes, snr_keys, titles)):
        # Ensure snr_key_group is always a list
        if isinstance(snr_key_group, str):
            snr_key_group = [snr_key_group]
        for k,(snr_key,color) in enumerate(zip(snr_key_group,color_list)):
            # Collect SNR values for each resolution, removing NaNs
            data = []
            logR4data = []
            p1, p2, p3 = [], [], []
            for snr_dict,_logR in zip(SNR_dict_list,logR):
                snr_values = snr_dict[snr_key]
                _d = snr_values[~np.isnan(snr_values)]
                if len(_d) != 0:
                    data.append(_d)
                    logR4data.append(_logR)
                    p1.append(np.nanpercentile(_d, 25))
                    p2.append(np.nanpercentile(_d, 50))
                    p3.append(np.nanpercentile(_d, 75))

            if k ==0:
                # Plot violin        rint()
                # parts = ax.violinplot(data, positions=logR4data, showmeans=False, showmedians=True, showextrema=False, widths=0.3,points=100)
                #                       quantiles=[[0.25,0.75]]*len(data))
                # for pc in parts['bodies']:
                #     pc.set_facecolor(color)
                #     # pc.set_edgecolor('black')
                #     pc.set_alpha(0.5)
                for pos, d in zip(logR4data, data):
                    histogram_violin(ax, d, position=pos, width=0.2, bins=300, color='#9eccf7')

            # Plot percentile lines
            if idx == 3 and k == 1:
                mylabel = 'High-pass filtered S/N'
            else:
                mylabel = '50th percentile'
            # print(idx,k,mylabel)
            # print('25th and 75th percentile' if idx == 3 and k == 0 else None)
            ax.plot(logR4data, p2, color=color, lw=2, linestyle='-', marker='o', label=mylabel)
            ax.plot(logR4data, p1, color=color, lw=1, linestyle='--', label='25th & 75th percentile' if idx == 3 and k == 0 else None)
            ax.plot(logR4data, p3, color=color, lw=1, linestyle='--', label= None)

            if k==0:
                # Set individual limits per panel
                all_values = np.concatenate(data)
                ymin, ymax = np.nanmin(all_values), np.nanmax(all_values)
                p95 = np.nanpercentile(all_values, 85)
                # ax.set_ylim(0, np.max([1,p95 * 1.1]))
                ax.set_ylim(snr_lims[idx][0], snr_lims[idx][1])

        # Axis formatting
        ax.set_title(title, fontsize=fontsize)
        # ax.text(0.5, 0.99, title, transform=ax.transAxes,
        #         fontsize=12, verticalalignment='top', horizontalalignment='center')
        # if label is not None:
        #     ax.text(0.5, 0.85, label, transform=ax.transAxes, color = "grey",
        #             fontsize=12, verticalalignment='top', horizontalalignment='center')
        ax.set_xlabel("Spectral Resolution (R=$\lambda/d\lambda$)",fontsize=fontsize)
        ax.set_xticks(logR_xaxis)
        myxtickslabels = []
        for r in R_list_xaxis:
            if r >= 1000:
                myxtickslabels.append("{0:.0f}k".format(r/1000.))
            else:
                myxtickslabels.append(str(r))
        ax.set_xticklabels(myxtickslabels, fontsize=fontsize)
        # ax.grid(True)

        if idx == 0:
            if label is not None:
                ax.set_ylabel(label+"\nS/N",fontsize=20)
            else:
                ax.set_ylabel("S/N",fontsize=20)
        elif idx == 3:
            ax.legend(loc='upper left', fontsize=fontsize, frameon=True)
        else:
            ax.tick_params(labelleft=True)  # Ensure y-axis tick labels are shown

        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
        plt.tight_layout()

if __name__ == "__main__":
    fig_dir = "/fast/jruffio/data/exosims/exosims_samples/figures"

    # R_list = [20,140,400,1000,3000,10000,30000]
    R_list = [20,50,140,400,1000,3000]
    # R_list = [100,1000]
    # contrast_floor_list = [1e-10,1e-12]
    contrast_floor = 1e-9
    # ppFact_Char_list = [0.1,0.01,0.001]
    ppFact_Char = 0.1
    output_filelist0 = []
    scriptfile_list = []
    output_filelist0 = []
    # _output_filename0 = "/fast/jruffio/data/exosims/exosims_samples/20260224_output/20260224_MHRS_emccd_DC1e-3_SNR_outputs_paper"
    # output_filelist0.append(_output_filename0)
    # _output_filename0 = "/fast/jruffio/data/exosims/exosims_samples/20260224_output/20260224_MHRS_emccd_DC1e-4_SNR_outputs_paper"
    # output_filelist0.append(_output_filename0)
    # _output_filename0 = "/fast/jruffio/data/exosims/exosims_samples/20260224_output/20260224_MHRS_emccd_DC3e-5_SNR_outputs_paper"
    # output_filelist0.append(_output_filename0)
    _output_filename0 = "/fast/jruffio/data/exosims/exosims_samples/20260224_output/20260224_MHRS_emccd_DC3e-5_undersamp_SNR_outputs_paper"
    output_filelist0.append(_output_filename0)
    # _output_filename0 = "/fast/jruffio/data/exosims/exosims_samples/20260224_output/20260224_MHRS_emccd_DC0_SNR_outputs_paper"
    # output_filelist0.append(_output_filename0)
    detector_labels = [r'$10^{-4}$ e-/s',r'$3x10^{-5}$ e-/s',r'$3x10^{-5}$ e-/s (undersamp.)',r"zero noise"] #,'Roman-analog & undersampled',"10x better detector","No detector noise"
    for det_label,output_filename0 in zip(detector_labels,output_filelist0):
        split_filename = os.path.basename(output_filename0).split("_")
        det_label4file = "_".join(split_filename[2:(len(split_filename)-3)])

        SNR_dict_list = []
        for R in R_list:
            output_filename = output_filename0+ "_R{0}_starlight{1:.1e}_corr{2:.1e}.txt".format(R,contrast_floor,contrast_floor*ppFact_Char)
            if 1:
                print("Reading "+output_filename)
                SNR_dict = read_snr_results_from_file(output_filename)
                print(SNR_dict)
                SNR_dict_list.append(SNR_dict)
                print("Done reading "+output_filename)
                # label = det_label+" ; {0:.0e} ; {1:.0e}".format(contrast_floor,contrast_floor*ppFact_Char)
            # except:
            #      Warning("missing file "+output_filename)
            label = det_label

        plot_snr_violin_panels(SNR_dict_list, R_list,label=label,plot_hpf_snr=False,Ty=True)

        out_filename = os.path.join(fig_dir, "SNRs_vs_R_{0}_starlight{1:.1e}_corr{2:.1e}.png".format(det_label4file,contrast_floor,contrast_floor*ppFact_Char))
        print("Saving " + out_filename)
        plt.savefig(out_filename, dpi=300)
        plt.savefig(out_filename.replace(".png", ".pdf"))
    plt.show()

