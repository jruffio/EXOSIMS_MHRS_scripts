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

if __name__ == "__main__":
    fig_dir = "/fast/jruffio/data/exosims/exosims_samples/figures"

    # R_list = [20,50,140,400,1000,3000,10000]
    R_list = [1000]
    # override_local_starlight_flux_ratio_list = [1e-10,1e-12]
    override_local_starlight_flux_ratio_list = [1e-10]
    # ppFact_Char_list = [1,0.1,0.01]
    ppFact_Char_list = [0.1]

    n_EZ = 3  # nEZ is the number of "zodis" where 1 zodi is equivalent to the amount of dust in the solar system. So it's like a way to tune the amount of dust in a planetary system
    # pl_dist_ee_coefs =  [0.95,1.0,1.35,1.67]
    pl_dist_ee_coef =  1.0
    p = 0.2 # Max albedo of the planet
    Rp = 1 * u.earthRad # Planet Radius
    n_angles = 1#len(pl_dist_ee_coefs)
    # target_list = None
    # target_list = ["HIP 32439 A","HIP 77052 A","HIP 79672","HIP 26779","HIP 113283"]
    target_list = ["HIP 79672"]

    scriptfile = "/home/jruffio/code/EXOSIMS_MHRS_scripts/configs/20260224_exosims_genOutSpec_MHRS_emccd_DC3e-5.json"
    output_filename0 = "/fast/jruffio/data/exosims/exosims_samples/20260224_output/20260224_MHRS_emccd_DC3e-5_SNR_outputs_paper.txt"
    with open(scriptfile, "r") as ff:
        script = ff.read()
    exosims_pars_dict = json.loads(script)
    print(exosims_pars_dict)

    # Check that the instruments and observing modes check expectations:
    assert 'imager' in exosims_pars_dict['observingModes'][0]['instName'], "1st instrument in observingModes list is not a imager"
    assert 'spectro' in exosims_pars_dict['observingModes'][1]['instName'], "2nd instrument in observingModes list is not a spectrograph"
    assert 'imager' in exosims_pars_dict['scienceInstruments'][0]['name'], "1st instrument in scienceInstruments list is not a imager"
    assert 'spectro' in exosims_pars_dict['scienceInstruments'][1]['name'], "2nd instrument in scienceInstruments list is not a spectrograph"

    if target_list is not None:
        exosims_pars_dict['cherryPickStars'] = target_list
        print(exosims_pars_dict['cherryPickStars'])

    for override_local_starlight_flux_ratio in override_local_starlight_flux_ratio_list:
        for ppFact_Char in ppFact_Char_list:
            SNR_dict_list = []
            for R in R_list:
                output_filename = output_filename0.replace(".txt","_R{0}_starlight{1:.1e}_corr{2:.1e}.txt".format(R,override_local_starlight_flux_ratio,override_local_starlight_flux_ratio*ppFact_Char))

                exosims_pars_dict['scienceInstruments'][1]["Rs"] = R
                exosims_pars_dict['starlightSuppressionSystems'][0]["override_local_starlight_flux_ratio"] = override_local_starlight_flux_ratio
                exosims_pars_dict["ppFact_char"] = ppFact_Char
                print("Spectral resolution R: ",exosims_pars_dict['scienceInstruments'][1]["Rs"])
                print("coronagraph flux ratio: ",exosims_pars_dict['starlightSuppressionSystems'][0]["override_local_starlight_flux_ratio"])
                print("Post proc charac factor: ",exosims_pars_dict["ppFact_char"])

                sim = ems.MissionSim(**deepcopy(exosims_pars_dict))
                # sim = ems.MissionSim(scriptfile,use_core_thruput_for_ez=False)
                # sim.genOutSpec("/fast/jruffio/data/exosims/exosims_samples/20250528_exosims_genOutSpec_MHRS.json")

                sInds = np.array([np.where(sim.TargetList.Name == t)[0][0] for t in sim.TargetList.Name])
                eeid_au_TL = sim.TargetList.calc_EEID(sInds).to(u.au).value # in AU
                eeid_as_TL = eeid_au_TL / sim.TargetList.dist[sInds].to(u.pc).value # in as
                WA_au_TL = pl_dist_ee_coef * eeid_au_TL
                WA_as_TL = pl_dist_ee_coef * eeid_as_TL
                d_TL = WA_au_TL * u.au # Planet-star distance in units of AU
                # Planet phase function
                beta = np.pi/2 *u.rad
                phi = sim.SimulatedUniverse.PlanetPhysicalModel.calc_Phi(beta)
                dMags = np.array([deltaMag(p, Rp, d, phi) for d in d_TL])
                # print("dMags",dMags,10**(-dMags/2.5))

                fZ = sim.ZodiacalLight.fZ0

                ## Load the albedo spectral model
                R_pl_template = np.inf

                lmin, lmax = 650, 850
                R = 2e6
                wv0 = np.arange(lmin, lmax, lmin / R) * u.nm
                nc_reflectance_all = []
                nc_reflectance_o2 = []
                nc_reflectance_h2o = []
                for clouds in ["highcloud", "lowcloud", "clearsky"]:
                    fname = "/fast/jruffio/data/exosims/model_Ty/earth_maxres/earth_icrccm_hitran2020_" + clouds + "_50_100000cm-1_toa_R150000.rad"
                    nc_data = np.loadtxt(fname, comments='#')
                    # print("nc_data[:, 0].shape",clouds, nc_data[:, 0].shape)
                    nc_where_wvs = np.where((nc_data[:, 0] > lmin) * (nc_data[:, 0] < lmax))
                    nc_data = nc_data[nc_where_wvs[0], :]
                    nc_wavelength_nm = nc_data[:, 0] * u.nm  # column 1
                    # print(nc_data[:, 1].shape)
                    nc_reflectance_all.append(nc_data[:, 1])  # Reflectance spectrum

                    fname = "/fast/jruffio/data/exosims/model_Ty/earth_maxres/earth_icrccm_hitran2020_" + clouds + "_o2_50_100000cm-1_toa_R150000.rad"
                    nc_data = np.loadtxt(fname, comments='#')
                    # print("nc_data[:, 0].shape",clouds, nc_data[:, 0].shape)
                    nc_data = nc_data[nc_where_wvs[0], :]
                    nc_reflectance_o2.append(nc_data[:, 1])

                    fname = "/fast/jruffio/data/exosims/model_Ty/earth_maxres/earth_icrccm_hitran2020_" + clouds + "_h2o_50_100000cm-1_toa_R150000.rad"
                    nc_data = np.loadtxt(fname, comments='#')
                    # print("nc_data[:, 0].shape",clouds, nc_data[:, 0].shape)
                    nc_data = nc_data[nc_where_wvs[0], :]
                    nc_reflectance_h2o.append(nc_data[:, 1])  # Reflectance spectrum
                # exit()
                nc_reflectance_all = np.array(nc_reflectance_all)
                nc_reflectance_all = np.nansum(nc_reflectance_all * np.array([0.25, 0.25, 0.5])[:, None], axis=0)
                nc_reflectance_o2 = np.array(nc_reflectance_o2)
                nc_reflectance_o2 = np.nansum(nc_reflectance_o2 * np.array([0.25, 0.25, 0.5])[:, None], axis=0)
                nc_reflectance_h2o = np.array(nc_reflectance_h2o)
                nc_reflectance_h2o = np.nansum(nc_reflectance_h2o * np.array([0.25, 0.25, 0.5])[:, None], axis=0)

                envelop = np.nanmax(np.concatenate(
                    [nc_reflectance_all[None, :], nc_reflectance_o2[None, :], nc_reflectance_h2o[None, :]]), axis=0)
                envelop2, albedo_cont_subtracted, x_knots, y_fit = subtract_continuum_envelop(nc_wavelength_nm, envelop,
                                                                                              n_control=30, penalty=1)

                nc_reflectance_o2 = nc_reflectance_o2 - envelop2
                nc_reflectance_h2o = nc_reflectance_h2o - envelop2

                # plt.figure()
                # # plt.plot(nc_wavelength_nm,nc_reflectance_all,label="all")
                # plt.plot(nc_wavelength_nm,nc_reflectance_h2o,label="H2O")
                # plt.plot(nc_wavelength_nm,nc_reflectance_o2,label="O2")
                # # plt.plot(nc_wavelength_nm,envelop,label="envelop")
                # # plt.plot(nc_wavelength_nm,envelop2,label="envelop2")
                # plt.legend()
                # plt.show()

                pl_template = [nc_reflectance_all, nc_reflectance_o2, nc_reflectance_h2o]
                pl_template_name = ["all", "O2", "H2O"]

                mode = sim.OpticalSystem.observingModes[1]
                _JEZ0_TL = sim.TargetList.JEZ0[mode['hex']][sInds]
                JEZ_TL = _JEZ0_TL * n_EZ / eeid_au_TL ** 2

                # TL, sInds, fZ, JEZ, dMag, WA, mode, returnExtra=False, TK=None, pl_waves = None,
                #        pl_template = None, R_pl_template=None,pl_template_name=None,n_jobs=-1,broaden_pixel=True)
                out = sim.OpticalSystem.Cp_Cb_Csp_spec(sim.TargetList,sInds,[fZ.value] * len(sInds) * fZ.unit,
                    JEZ_TL,dMags,WA_as_TL * u.arcsec,mode,returnExtra=True,
                    R_pl_template=R_pl_template, pl_waves=nc_wavelength_nm, pl_template=pl_template,
                    pl_template_name=pl_template_name,
                    n_jobs=0, broaden_pixel=False)
                data_waves = out[0] # Wavelength sampling of the "data", ie the spectra below
                pl0_template_scaled_C_p_list = out[1]   # List of planet spectra (including PCeff * NCTE)
                _C_b_spec_list = out[2]  # List of white noise stddev spectra (including k_SZ, ENF2, k_det)
                star_template_scaled_C_sp_list = out[3] # List of residual starlight spectra, ie correlated noise (_C_sr * post processing factor * stability factor)

                C_extra = out[4] # The outputs in there do not typically include the photon counting detector stuff
                pl0_template_scaled_C_p0_list = C_extra["C_p0_spec"] # List of planet spectra (NOT including PCeff * NCTE)
                star_template_scaled_C_sr_list = C_extra["C_sr_spec"] # List of starlight spectra (before post-processing)
                _C_z_spec_list = C_extra["C_z_spec"] # List of Zodi spectra
                _C_ez_spec_list = C_extra["C_ez_spec"] # List of exzodi spectra
                _C_dc_spec_list = C_extra["C_dc_spec"] # List of dark current spectra
                _C_bl_spec_list = C_extra["C_bl_spec"]
                _C_star_spec_list = C_extra["C_star_spec"]
                _C_rn_spec_list = C_extra["C_rn_spec"] # List of read noise spectra
                _C_cc_spec_list = C_extra["C_cc_spec"] # List of clock-induced charge spectra
                Npix = C_extra["Npix_per_bin"]
                k_SZ = C_extra["k_SZ"]
                k_det = C_extra["k_det"]
                ENF2 = C_extra["ENF2"]
                lambda_center = C_extra["lambda_center"] # Center wavelength of the bandpass

                intTime = (mode["intTime"] * u.h << u.s)
                inv_cov0, cov_matrix0, corr_matrix0 = sim.OpticalSystem.compute_cov_matrices(data_waves, WA_as_TL[0] * u.arcsec,
                                                                                mode["syst"]["chromaticity_dwave_nm"],
                                                                                intTime * star_template_scaled_C_sp_list[0],
                                                                                np.sqrt(intTime * _C_b_spec_list[0]))

                print("Npix",Npix)
                print(data_waves[0],data_waves[-1])

                plt.figure(figsize=(12,4))
                plt.subplot(1,2,1)
                plt.plot(data_waves.to_value(u.nm)-data_waves[0].to_value(u.nm),cov_matrix0[0,:])
                plt.xlabel(r"$\Delta\lambda$ (nm)",fontsize=12)
                plt.ylabel("Covariance profile (# of photons)",fontsize=12)
                plt.xlim([-1,50])
                plt.subplot(1,2,2)
                plt.imshow(cov_matrix0)
                plt.xlabel("Column index",fontsize=12)
                plt.ylabel("Row index",fontsize=12)
                cbar = plt.colorbar()
                cbar.set_label('Covariance (# of photons)')
                plt.tight_layout()
                out_filename = os.path.join(fig_dir, "covariance.png")
                print("Saving " + out_filename)
                plt.savefig(out_filename, dpi=300)
                plt.savefig(out_filename.replace(".png", ".pdf"))
                plt.show()
