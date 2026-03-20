from EXOSIMS.OpticalSystem.MHRS import write_snr_results_to_file, read_snr_results_from_file
import matplotlib.pyplot as plt
import numpy as np
import os
from histogram_violin import histogram_violin
import warnings


def plot_snr_violin_panels(SNR_dict_list, R_list, label=None, plot_hpf_snr=False, Ty=False,
                           axes=None, show_xlabel=True,plotlegend=True):
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
        A label to add to the y-axis of the first panel for context (e.g., detector type or noise level).
    plot_hpf_snr : bool, optional
        If True, include high-pass filtered SNR metrics in the plots. If False, only include the standard SNR metrics.
    Ty : bool, optional
        If True, swap labels of H2O and O2.
    axes : array-like of Axes, optional
        Pre-existing axes to plot into. If None, a new figure and axes are created.
    show_xlabel : bool, optional
        If True, show x-axis tick labels and axis label. Set False for all but the bottom row.
    """
    fontsize = 12
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

    logR = np.log10(R_list)
    R_list_xaxis = R_list
    # R_list_xaxis = [R for R in R_list if R != 400]
    logR_xaxis = np.log10(R_list_xaxis)

    # Create figure only if axes not provided
    if axes is None:
        fig, axes = plt.subplots(1, len(snr_keys), figsize=(12, 4.5))
        if len(snr_keys) == 1:
            axes = [axes]
        created_fig = True
    else:
        created_fig = False

    color_list = ["#006699","#ff9900", "#6600ff", "purple", "grey"]
    for idx, (ax, snr_key_group, title) in enumerate(zip(axes, snr_keys, titles)):
        if isinstance(snr_key_group, str):
            snr_key_group = [snr_key_group]
        for k,(snr_key,color) in enumerate(zip(snr_key_group,color_list)):
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

            if k == 0:
                for pos, d in zip(logR4data, data):
                    histogram_violin(ax, d, position=pos, width=0.2, bins=300, color='#9eccf7')

            if idx == 3 and k == 1:
                mylabel = 'High-pass filtered S/N'
            else:
                mylabel = '50th percentile'
            ax.plot(logR4data, p2, color=color, lw=2, linestyle='-', marker='o', label=mylabel)
            ax.plot(logR4data, p1, color=color, lw=1, linestyle='--', label='25th & 75th perc.' if idx == 3 and k == 0 else None)
            ax.plot(logR4data, p3, color=color, lw=1, linestyle='--', label=None)

            if k == 0:
                ax.set_ylim(snr_lims[idx][0], snr_lims[idx][1])

        # Only show titles on the top row (when axes were provided externally, caller controls this)
        if created_fig:
            ax.set_title(title, fontsize=fontsize)

        ax.set_xticks(logR_xaxis)
        myxtickslabels = []
        for r in R_list_xaxis:
            if r >= 1000:
                myxtickslabels.append("{0:.0f}k".format(r/1000.))
            else:
                myxtickslabels.append(str(r))

        if show_xlabel:
            ax.set_xticklabels(myxtickslabels, fontsize=fontsize,rotation=90)
            if idx == 0:
                ax.set_xlabel("Spectral Resolution (R=$\lambda/d\lambda$)", fontsize=fontsize)
            else:
                ax.set_xlabel("R", fontsize=fontsize)
        else:
            ax.set_xticklabels([])

        if idx>=2:
            ax.set_yticks([0,5,10,15])
            ax.tick_params(axis='y', labelsize=fontsize)

        if idx == 0:
            if label is not None:
                ax.set_ylabel(label+"\nS/N", fontsize=14)
            else:
                ax.set_ylabel("S/N", fontsize=14)
        elif idx == 3:
            if plotlegend:
                ax.legend(loc='upper left', fontsize=fontsize, frameon=True,handlelength=1.2)
        else:
            ax.tick_params(labelleft=True)

        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)

    if created_fig:
        plt.tight_layout()


if __name__ == "__main__":
    fig_dir = "/fast/jruffio/data/exosims/exosims_samples/figures"

    R_list = [20,50,140,400,1000,3000,10000,30000]
    contrast_floor = 1e-10
    ppFact_Char = 0.001

    output_filelist0 = []
    _output_filename0 = "/fast/jruffio/data/exosims/exosims_samples/20260224_output/20260224_MHRS_emccd_DC1e-4_SNR_outputs_paper"
    output_filelist0.append(_output_filename0)
    _output_filename0 = "/fast/jruffio/data/exosims/exosims_samples/20260224_output/20260224_MHRS_emccd_DC3e-5_SNR_outputs_paper"
    output_filelist0.append(_output_filename0)
    _output_filename0 = "/fast/jruffio/data/exosims/exosims_samples/20260224_output/20260224_MHRS_emccd_DC3e-5_undersamp_SNR_outputs_paper"
    output_filelist0.append(_output_filename0)
    _output_filename0 = "/fast/jruffio/data/exosims/exosims_samples/20260224_output/20260224_MHRS_emccd_DC0_SNR_outputs_paper"
    output_filelist0.append(_output_filename0)
    detector_labels = [r'$10^{-4}$ e-/s', r'$3x10^{-5}$ e-/s', '(undersampled)\n'+r'$3x10^{-5}$ e-/s', r"zero noise"]

    n_rows = len(output_filelist0)
    n_cols = 4  # number of SNR panels

    # Column titles (only shown once at top)
    col_titles = [
        "Average S/N per bin",
        "Template matching S/N\nAll molecules & broadband",
        "S/N of O2 only",
        "S/N of H2O only"
    ]

    fig, all_axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3 * n_cols, 2 * n_rows),
        sharex=True
    )
    fig.subplots_adjust(hspace=0.05)

    # Add column titles to the top row
    for col_idx, title in enumerate(col_titles):
        all_axes[0, col_idx].set_title(title, fontsize=14)

    for row_idx, (det_label, output_filename0) in enumerate(zip(detector_labels, output_filelist0)):
        SNR_dict_list = []
        for R in R_list:
            output_filename = output_filename0 + "_R{0}_starlight{1:.1e}_corr{2:.1e}.txt".format(
                R, contrast_floor, contrast_floor * ppFact_Char)
            print("Reading " + output_filename)
            SNR_dict = read_snr_results_from_file(output_filename)
            print(SNR_dict)
            SNR_dict_list.append(SNR_dict)
            print("Done reading " + output_filename)

        is_bottom_row = (row_idx == n_rows - 1)

        if row_idx == 0:
            plotlegend = True  # Show legend only on the first panel of the first row
        else:
            plotlegend = False

        plot_snr_violin_panels(
            SNR_dict_list, R_list,
            label=det_label,
            plot_hpf_snr=False,
            Ty=True,
            axes=all_axes[row_idx],
            show_xlabel=is_bottom_row,
            plotlegend=plotlegend  # Legend only on the first panel of the last row
        )

    # Save combined figure using the first output filename's label components for naming
    out_filename = os.path.join(
        fig_dir,
        "SNRs_vs_R_combined.png".format(
            contrast_floor, contrast_floor * ppFact_Char)
    )
    print("Saving " + out_filename)
    plt.savefig(out_filename, dpi=300, bbox_inches='tight')
    plt.savefig(out_filename.replace(".png", ".pdf"), bbox_inches='tight')
    plt.show()

