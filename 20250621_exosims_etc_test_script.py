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
from etc_utils import subtract_continuum_envelop

if __name__ == "__main__":
    plot_results = True

    # R_list = [20,140,400,1000,3000,10000,30000]
    # R_list = [20,50,140,400,1000]
    R_list = [1000]
    override_local_starlight_flux_ratio_list = [1e-10]
    # ppFact_Char_list = [0.1,0.01,0.001]
    ppFact_Char_list = [0.001]

    n_EZ = 3  # nEZ is the number of "zodis" where 1 zodi is equivalent to the amount of dust in the solar system. So it's like a way to tune the amount of dust in a planetary system
    # pl_dist_ee_coefs =  [0.95,1.0,1.35,1.67]
    pl_dist_ee_coef =  1.0
    p = 0.2 # Max albedo of the planet
    Rp = 1 * u.earthRad # Planet Radius
    n_angles = 1#len(pl_dist_ee_coefs)
    target_list = None
    # target_list = ["HIP 32439 A","HIP 77052 A","HIP 79672","HIP 26779","HIP 113283"]
    # target_list = ["HIP 79672"]

    # scriptfile = "/home/jruffio/code/EXOSIMS_MHRS_scripts/configs/20250621_exosims_genOutSpec_MHRS_Romandetecnoise.json"
    # output_filename0 = "/fast/jruffio/data/exosims/exosims_samples/20250621_output/20250621_MHRS_Romandetecnoise_SNR_outputs_paper.txt"
    # scriptfile = "/home/jruffio/code/EXOSIMS_MHRS_scripts/configs/20250621_exosims_genOutSpec_MHRS_10xbetterRoman.json"
    # output_filename0 = "/fast/jruffio/data/exosims/exosims_samples/20250621_output/20250621_MHRS_10xbetterRoman_SNR_outputs_paper.txt"
    scriptfile = "/home/jruffio/code/EXOSIMS_MHRS_scripts/configs/20250621_exosims_genOutSpec_MHRS_nodetecnoise.json"
    output_filename0 = "/fast/jruffio/data/exosims/exosims_samples/20250621_output/20250621_MHRS_nodetecnoise_SNR_outputs_paper.txt"
    # scriptfile = "/home/jruffio/code/EXOSIMS_MHRS_scripts/configs/20250621_exosims_genOutSpec_MHRS_Romandetecnoise_undersamp.json"
    # output_filename0 = "/fast/jruffio/data/exosims/exosims_samples/20250621_output/20250621_MHRS_Romandetecnoise_undersamp_SNR_outputs_paper.txt"


    output_filename0 = "/fast/jruffio/data/exosims/exosims_samples/20250621_output/20251016_test.txt"
    with open(scriptfile, "r") as ff:
        script = ff.read()
    exosims_pars_dict = json.loads(script)
    print(exosims_pars_dict)

    # output_filename0 = "/fast/jruffio/data/exosims/exosims_samples/20250621_output/20250621_test.txt"

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
                # print("dMags",dMags,10**(-dMags/2.5))
                # exit()
                # shoudl I use sim.ZodiacalLight.fZ() here?
                fZ = sim.ZodiacalLight.fZ0

                ## Load the albedo spectral model
                R_pl_template = 400000
                # filename1_broad = '/fast/jruffio/data/exosims/model_Renyu/HighResSpec/EarthSpec/GeometricA_Earth_HighCloud_UltraRes_500-1500nm_R{0:.0f}.dat'.format(
                #     R_pl_template)
                # filename2_broad = '/fast/jruffio/data/exosims/model_Renyu/HighResSpec/EarthSpec/GeometricA_Earth_NoCloud_UltraRes_500-1500nm_R{0:.0f}.dat'.format(
                #     R_pl_template)
                # filename_broad = '/fast/jruffio/data/exosims/model_Renyu/HighResSpec/EarthSpec/GeometricA_Earth_NoCloud_UltraRes_500-1500nm_R{0:.0f}.dat'.format(R_pl_template)
                filename_broad = '/fast/jruffio/data/exosims/model_Renyu/HighResSpec/EarthSpec/GeometricA_Earth_HighCloud_UltraRes.dat'
                # data_broad = np.loadtxt(filename1_broad, dtype=float)
                # data_broad2 = np.loadtxt(filename2_broad, dtype=float)
                data_broad = np.loadtxt(filename_broad, dtype=float)
                where_wvs = np.where((data_broad[:,0]>200)*(data_broad[:,0]<1500))
                data_broad = data_broad[where_wvs[0],:]
                # index 0: Wavelength
                # index 1: All
                # index 2: H2O
                # index 3: CO2
                # index 4: N2O
                # index 5: CH4
                # index 6: O2
                # index 7: O3
                pl_waves = data_broad[:, 0] * u.nm  # or u.AA, u.nm, etc.
                data_broad /= np.nanmean(data_broad[:, 4])
                pl_all_template = data_broad[:, 1]  # unitless reflectance
                pl_H2O_template = data_broad[:, 2] - data_broad[:, 4]  # unitless reflectance
                pl_O2_template = data_broad[:, 6] - data_broad[:, 4]  # unitless reflectance
                pl_template = [pl_all_template, pl_H2O_template, pl_O2_template]
                pl_template_name = ["all", "H2O", "O2"]
                # pl_template_name = ["all","H2O","CO2","N2O","CH4","O2","O3"]
                # plt.plot(pl_waves,pl_all_template)
                # plt.plot(pl_waves,pl_H2O_template)
                # plt.plot(pl_waves,pl_O2_template)
                # plt.show()

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
                    R_pl_template=R_pl_template, pl_waves=pl_waves, pl_template=pl_template,
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
                plt.show()


