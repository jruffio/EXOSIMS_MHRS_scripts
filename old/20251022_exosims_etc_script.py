from etc_utils import *
from EXOSIMS.util.deltaMag import deltaMag

from EXOSIMS.OpticalSystem.MHRS import read_snr_results_from_file
from histogram_violin import plot_snr_violin_panels

import matplotlib.pyplot as plt
import numpy as np
import json
import astropy.units as u
import EXOSIMS.MissionSim as ems
from copy import deepcopy

if __name__ == "__main__":
    plot_results = False

    R_list = [20,50,140,400,1000,3000,10000,30000]
    # R_list = [20,50,140,400,1000]
    # R_list = [400]
    contrast_floor_list = [1e-10]
    ppFact_Char_list = [0.1,0.01,0.001]
    # ppFact_Char_list = [0.001]

    n_EZ = 3  # nEZ is the number of "zodis" where 1 zodi is equivalent to the amount of dust in the solar system. So it's like a way to tune the amount of dust in a planetary system
    # pl_dist_ee_coefs =  [0.95,1.0,1.35,1.67]
    pl_dist_ee_coef =  1.0
    p = 0.2 # Max albedo of the planet
    Rp = 1 * u.earthRad # Planet Radius
    n_angles = 1#len(pl_dist_ee_coefs)
    target_list = None
    # target_list = ["HIP 32439 A","HIP 77052 A","HIP 79672","HIP 26779","HIP 113283"]
    # target_list = ["HIP 79672"]

    scriptfile_list = []
    output_filename0_list = []
    scriptfile = "/home/jruffio/code/EXOSIMS_MHRS_scripts/configs/20251022_exosims_genOutSpec_MHRS_emccd.json"
    output_filename0 = "/fast/jruffio/data/exosims/exosims_samples/20251022_output/20251022_MHRS_emccd_SNR_outputs_paper.txt"
    scriptfile_list.append(scriptfile)
    output_filename0_list.append(output_filename0)
    scriptfile = "/home/jruffio/code/EXOSIMS_MHRS_scripts/configs/20251022_exosims_genOutSpec_MHRS_10xbetterEmccd.json"
    output_filename0 = "/fast/jruffio/data/exosims/exosims_samples/20251022_output/20251022_MHRS_10xbetterEmccd_SNR_outputs_paper.txt"
    scriptfile_list.append(scriptfile)
    output_filename0_list.append(output_filename0)
    scriptfile = "/home/jruffio/code/EXOSIMS_MHRS_scripts/configs/20251022_exosims_genOutSpec_MHRS_nodetecnoise.json"
    output_filename0 = "/fast/jruffio/data/exosims/exosims_samples/20251022_output/20251022_MHRS_nodetecnoise_SNR_outputs_paper.txt"
    scriptfile_list.append(scriptfile)
    output_filename0_list.append(output_filename0)
    scriptfile = "/home/jruffio/code/EXOSIMS_MHRS_scripts/configs/20251022_exosims_genOutSpec_MHRS_emccd_undersamp.json"
    output_filename0 = "/fast/jruffio/data/exosims/exosims_samples/20251022_output/20251022_MHRS_emccd_undersamp_SNR_outputs_paper.txt"
    scriptfile_list.append(scriptfile)
    output_filename0_list.append(output_filename0)

    for scriptfile, output_filename0 in zip(scriptfile_list, output_filename0_list):
        with open(scriptfile, "r") as ff:
            script = ff.read()
        exosims_pars_dict = json.loads(script)
        print(exosims_pars_dict)

        # output_filename0 = "/fast/jruffio/data/exosims/exosims_samples/20251022_output/20251022_test.txt"

        # Check that the instruments and observing modes check expectations:
        assert 'imager' in exosims_pars_dict['observingModes'][0]['instName'], "1st instrument in observingModes list is not a imager"
        assert 'spectro' in exosims_pars_dict['observingModes'][1]['instName'], "2nd instrument in observingModes list is not a spectrograph"
        assert 'imager' in exosims_pars_dict['scienceInstruments'][0]['name'], "1st instrument in scienceInstruments list is not a imager"
        assert 'spectro' in exosims_pars_dict['scienceInstruments'][1]['name'], "2nd instrument in scienceInstruments list is not a spectrograph"

        if target_list is not None:
            exosims_pars_dict['cherryPickStars'] = target_list
            print(exosims_pars_dict['cherryPickStars'])

        for contrast_floor in contrast_floor_list:
            for ppFact_Char in ppFact_Char_list:
                SNR_dict_list = []
                for R in R_list:
                    output_filename = output_filename0.replace(".txt","_R{0}_starlight{1:.1e}_corr{2:.1e}.txt".format(R,contrast_floor,contrast_floor*ppFact_Char))

                    exosims_pars_dict['scienceInstruments'][1]["Rs"] = R
                    exosims_pars_dict['starlightSuppressionSystems'][0]["contrast_floor"] = contrast_floor
                    exosims_pars_dict["ppFact_char"] = ppFact_Char
                    print("Spectral resolution R: ",exosims_pars_dict['scienceInstruments'][1]["Rs"])
                    print("coronagraph flux ratio: ",exosims_pars_dict['starlightSuppressionSystems'][0]["contrast_floor"])
                    print("Post proc charac factor: ",exosims_pars_dict["ppFact_char"])

                    # exosims_pars_dict['observingModes'][1]["lam"] = 950
                    # exosims_pars_dict['starlightSuppressionSystems'][0]["lam"] = 950

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
                    print("dMags",dMags,10**(-dMags/2.5))
                    # exit()
                    # shoudl I use sim.ZodiacalLight.fZ() here?
                    fZ = sim.ZodiacalLight.fZ0

                    ## Load the albedo spectral model
                    R_pl_template = np.inf

                    lmin,lmax = 650,850
                    R = 2e6
                    wv0 = np.arange(lmin,lmax,lmin/R) * u.nm
                    nc_reflectance_all = []
                    nc_reflectance_o2 = []
                    nc_reflectance_h2o = []
                    for clouds in ["highcloud", "lowcloud", "clearsky"]:
                        fname = "/fast/jruffio/data/exosims/model_Ty/earth_maxres/earth_icrccm_hitran2020_" + clouds + "_50_100000cm-1_toa_R150000.rad"
                        nc_data = np.loadtxt(fname, comments='#')
                        nc_where_wvs = np.where((nc_data[:, 0] > lmin) * (nc_data[:, 0] < lmax))
                        nc_data = nc_data[nc_where_wvs[0], :]
                        nc_wavelength_nm = nc_data[:, 0] * u.nm  # column 1
                        nc_reflectance_all.append(nc_data[:, 1])   # Reflectance spectrum

                        fname = "/fast/jruffio/data/exosims/model_Ty/earth_maxres/earth_icrccm_hitran2020_" + clouds + "_o2_50_100000cm-1_toa_R150000.rad"
                        nc_data = np.loadtxt(fname, comments='#')
                        nc_data = nc_data[nc_where_wvs[0], :]
                        nc_reflectance_o2.append(nc_data[:, 1])

                        fname = "/fast/jruffio/data/exosims/model_Ty/earth_maxres/earth_icrccm_hitran2020_" + clouds + "_h2o_50_100000cm-1_toa_R150000.rad"
                        nc_data = np.loadtxt(fname, comments='#')
                        nc_data = nc_data[nc_where_wvs[0], :]
                        nc_reflectance_h2o.append(nc_data[:, 1])   # Reflectance spectrum
                    nc_reflectance_all = np.array(nc_reflectance_all)
                    nc_reflectance_all = np.nansum(nc_reflectance_all*np.array([0.25,0.25,0.5])[:,None],axis=0)
                    nc_reflectance_o2 = np.array(nc_reflectance_o2)
                    nc_reflectance_o2 = np.nansum(nc_reflectance_o2*np.array([0.25,0.25,0.5])[:,None],axis=0)
                    nc_reflectance_h2o = np.array(nc_reflectance_h2o)
                    nc_reflectance_h2o = np.nansum(nc_reflectance_h2o*np.array([0.25,0.25,0.5])[:,None],axis=0)


                    envelop = np.nanmax(np.concatenate([nc_reflectance_all[None,:],nc_reflectance_o2[None,:],nc_reflectance_h2o[None,:]]),axis=0)
                    envelop2, albedo_cont_subtracted, x_knots, y_fit = subtract_continuum_envelop(nc_wavelength_nm, envelop, n_control=30, penalty=1)

                    nc_reflectance_o2 = nc_reflectance_o2-envelop2
                    nc_reflectance_h2o = nc_reflectance_h2o-envelop2

                    # plt.figure()
                    # plt.plot(nc_wavelength_nm,nc_reflectance_all,label="all")
                    # plt.plot(nc_wavelength_nm,nc_reflectance_h2o,label="H2O")
                    # plt.plot(nc_wavelength_nm,nc_reflectance_o2,label="O2")
                    # plt.plot(nc_wavelength_nm,envelop,label="envelop")
                    # plt.plot(nc_wavelength_nm,envelop2,label="envelop2")
                    # plt.legend()
                    # plt.show()

                    pl_template = [nc_reflectance_all, nc_reflectance_o2, nc_reflectance_h2o]
                    pl_template_name = ["all", "O2", "H2O"]

                    mode = sim.OpticalSystem.observingModes[1]
                    _JEZ0_TL = sim.TargetList.JEZ0[mode['hex']][sInds]
                    JEZ_TL = _JEZ0_TL * n_EZ / WA_au_TL ** 2

                    figs = None
                    # figs = [plt.figure() for fig_id in range(len(sInds))]

                    # import cProfile
                    #
                    # cProfile.run('my_function()')
                    SNR = sim.OpticalSystem.calc_snr(sim.TargetList,sInds,[fZ.value] * len(sInds) * fZ.unit,
                        JEZ_TL,dMags,WA_as_TL * u.arcsec,mode,
                        R_pl_template=R_pl_template, pl_waves=nc_wavelength_nm, pl_template=pl_template,
                        pl_template_name=pl_template_name,
                        figs=figs, n_jobs=0, broaden_pixel=False,output_filename=output_filename,config_json_path=scriptfile,
                    )

                    if plot_results:
                        SNR_dict = read_snr_results_from_file(output_filename)
                        # for key, val in zip(SNR_dict.keys(), SNR_dict.values()):
                        #     print("SNR {0}: {1}".format(key, val))
                        SNR_dict_list.append(SNR_dict)
                if plot_results:
                    plot_snr_violin_panels(SNR_dict_list, R_list,plot_hpf_snr=True)
    if plot_results:
        plt.show()


