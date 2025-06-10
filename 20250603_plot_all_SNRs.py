import os, time, shutil, datetime
import numpy as np
import yaml
from astropy.io import fits
import astropy.units as u
import EXOSIMS.MissionSim as ems
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from etc_utils import *

from EXOSIMS.util.deltaMag import deltaMag

from synphot import Observation
from synphot import SourceSpectrum
from astropy.modeling.models import Tabular1D

from EXOSIMS.OpticalSystem.MHRS import write_snr_results_to_file, read_snr_results_from_file


import matplotlib.pyplot as plt
import numpy as np
import json
import ternary

from histogram_violin import histogram_violin
def plot_snr_violin_panels(SNR_dict_list, R_list,label=None,plot_hpf_snr = False):
    """
    Plot violin plots of SNR values for all stars across different spectral resolutions.
    Each panel may show the distribution of one or more SNR metrics, as specified in snr_keys.

    Parameters
    ----------
    SNR_dict_list : list of dict
        List of SNR result dictionaries, one per spectral resolution.

    R_list : list of int or float
        Spectral resolution values corresponding to each entry in SNR_dict_list.
    """
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
    # if label is not None:
    #     titles = [title+label for title in titles]

    logR = np.log10(R_list)

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
            print(idx,k,mylabel)
            print('25th and 75th percentile' if idx == 3 and k == 0 else None)
            ax.plot(logR4data, p2, color=color, lw=2, linestyle='-', marker='o', label=mylabel)
            ax.plot(logR4data, p1, color=color, lw=1, linestyle='--', label='25th and 75th percentile' if idx == 3 and k == 0 else None)
            ax.plot(logR4data, p3, color=color, lw=1, linestyle='--', label= None)

            if k==0:
                # Set individual limits per panel
                all_values = np.concatenate(data)
                ymin, ymax = np.nanmin(all_values), np.nanmax(all_values)
                p95 = np.nanpercentile(all_values, 85)
                ax.set_ylim(0, np.max([1,p95 * 1.1]))

        # Axis formatting
        ax.set_title(title)
        # ax.text(0.5, 0.99, title, transform=ax.transAxes,
        #         fontsize=12, verticalalignment='top', horizontalalignment='center')
        # if label is not None:
        #     ax.text(0.5, 0.85, label, transform=ax.transAxes, color = "grey",
        #             fontsize=12, verticalalignment='top', horizontalalignment='center')
        ax.set_xlabel("Spectral Resolution (R=$\lambda/d\lambda$)")
        ax.set_xticks(logR)
        ax.set_xticklabels([str(r) for r in R_list], fontsize=12)
        ax.grid(True)

        if idx == 0:
            ax.set_ylabel(label+"\nS/N")
            # ax.set_ylabel("S/N")
        elif idx == 3:
            ax.legend(loc='upper left', fontsize=10, frameon=True)
        else:
            ax.tick_params(labelleft=True)  # Ensure y-axis tick labels are shown

        plt.tight_layout()



if __name__ == "__main__":
    fig_dir = "/exosims_samples/figures"

    R_list = [20,50,140,400,1000,3000,10000]
    # R_list = [100,1000]
    override_local_starlight_flux_ratio_list = [1e-10,1e-12]
    override_local_starlight_flux_ratio = override_local_starlight_flux_ratio_list[0]
    ppFact_Char_list = [1,0.1,0.01,0.001]
    ppFact_Char = ppFact_Char_list[-1]
    output_filelist0 = []
    output_filelist0.append("/fast/jruffio/data/exosims/exosims_samples/output/20250604_MHRS_Romandetecnoise_SNR_outputs_paper.txt")
    output_filelist0.append("/fast/jruffio/data/exosims/exosims_samples/output/20250604_MHRS_10xbetterRoman_SNR_outputs_paper.txt")
    output_filelist0.append("/fast/jruffio/data/exosims/exosims_samples/output/20250604_MHRS_nodetecnoise_SNR_outputs_paper.txt")
    detector_labels = ['Roman-analog detector',"10x better detector","No detector noise"]
    for det_label,output_filename0 in zip(detector_labels,output_filelist0):
        SNR_dict_list = []
        for R in R_list:
            output_filename = output_filename0.replace(".txt","_R{0}_starlight{1:.1e}_corr{2:.1e}.txt".format(R,override_local_starlight_flux_ratio,override_local_starlight_flux_ratio*ppFact_Char))
            SNR_dict = read_snr_results_from_file(output_filename)
            SNR_dict_list.append(SNR_dict)
            # label = det_label+" ; {0:.0e} ; {1:.0e}".format(override_local_starlight_flux_ratio,override_local_starlight_flux_ratio*ppFact_Char)
            label = det_label

        plot_snr_violin_panels(SNR_dict_list, R_list,label=label,plot_hpf_snr=False)

        det_label4file = os.path.basename(output_filename0).split("_")[2]
        out_filename = os.path.join(fig_dir, "SNRs_vs_R_{0}_starlight{1:.1e}_corr{2:.1e}.png".format(det_label4file,override_local_starlight_flux_ratio,override_local_starlight_flux_ratio*ppFact_Char))
        print("Saving " + out_filename)
        plt.savefig(out_filename, dpi=300)
        plt.savefig(out_filename.replace(".png", ".pdf"))
    plt.show()

