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

from histogram_violin import plot_snr_violin_panels
import warnings


if __name__ == "__main__":
    fig_dir = "/fast/jruffio/data/exosims/exosims_samples/figures"

    R_list = [20,50,140,400,1000,3000,10000,30000]
    # R_list = [100,1000]
    # override_local_starlight_flux_ratio_list = [1e-10,1e-12]
    override_local_starlight_flux_ratio = 1e-10
    # ppFact_Char_list = [0.1,0.01,0.001]
    ppFact_Char = 0.001
    output_filelist0 = []
    output_filelist0.append("/fast/jruffio/data/exosims/exosims_samples/20250621_output/20250621_MHRS_Romandetecnoise_SNR_outputs_paper")
    output_filelist0.append("/fast/jruffio/data/exosims/exosims_samples/20250621_output/20250621_MHRS_Romandetecnoise_undersamp_SNR_outputs_paper")
    output_filelist0.append("/fast/jruffio/data/exosims/exosims_samples/20250621_output/20250621_MHRS_10xbetterRoman_SNR_outputs_paper")
    output_filelist0.append("/fast/jruffio/data/exosims/exosims_samples/20250621_output/20250621_MHRS_nodetecnoise_SNR_outputs_paper")
    detector_labels = ['Roman-analog detector','Roman-analog & undersampled',"10x better detector","No detector noise"]
    for det_label,output_filename0 in zip(detector_labels,output_filelist0):
        split_filename = os.path.basename(output_filename0).split("_")
        det_label4file = "_".join(split_filename[2:(len(split_filename)-3)])

        SNR_dict_list = []
        for R in R_list:
            output_filename = output_filename0+ "_R{0}_starlight{1:.1e}_corr{2:.1e}.txt".format(R,override_local_starlight_flux_ratio,override_local_starlight_flux_ratio*ppFact_Char)
            try:
                SNR_dict = read_snr_results_from_file(output_filename)
                SNR_dict_list.append(SNR_dict)
                # label = det_label+" ; {0:.0e} ; {1:.0e}".format(override_local_starlight_flux_ratio,override_local_starlight_flux_ratio*ppFact_Char)
            except:
                 Warning("missing file "+output_filename)
            label = det_label

        plot_snr_violin_panels(SNR_dict_list, R_list,label=label,plot_hpf_snr=False)

        out_filename = os.path.join(fig_dir, "SNRs_vs_R_{0}_starlight{1:.1e}_corr{2:.1e}.png".format(det_label4file,override_local_starlight_flux_ratio,override_local_starlight_flux_ratio*ppFact_Char))
        print("Saving " + out_filename)
        plt.savefig(out_filename, dpi=300)
        plt.savefig(out_filename.replace(".png", ".pdf"))
    plt.show()

