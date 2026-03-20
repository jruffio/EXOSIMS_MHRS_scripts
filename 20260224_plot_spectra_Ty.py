import numpy as np
import matplotlib.pyplot as plt
import os
from etc_utils import subtract_continuum_envelop,gaussian_broaden
import astropy.units as u
from EXOSIMS.OpticalSystem.MHRS import broaden_and_resample,broaden
import matplotlib.patheffects as PathEffects

if __name__ == "__main__":

    fontsize=16

    # lmin, lmax = 759,771
    # clouds = "highcloud"
    # fname = "/fast/jruffio/data/exosims/model_Ty/earth_maxres/earth_icrccm_hitran2020_" + clouds + "_o2_50_100000cm-1_toa.rad"
    # # fname = "/fast/jruffio/data/exosims/model_Ty/earth_maxres/earth_icrccm_hitran2020_" + clouds + "_h2o_50_100000cm-1_toa.rad"
    # nc_data = np.loadtxt(fname, comments='#')
    # nc_where_wvs = np.where((nc_data[:, 0] > lmin / 1000.) * (nc_data[:, 0] < lmax / 1000.))
    # nc_data = nc_data[nc_where_wvs[0], :]
    #
    # # Split into named columns
    # nc_wavelength_nm = nc_data[::-1, 0]*1000  * u.nm # column 1
    # nc_wavenumber_cm1 = nc_data[::-1, 1]  # column 2
    # nc_solar_flux_1au = nc_data[::-1, 2]  # column 3
    # nc_earth_flux_toa = nc_data[::-1, 3]  # column 4
    # nc_radiance_streams = nc_data[::-1, 4:]  # columns 5-8
    # nc_reflectance = nc_earth_flux_toa / nc_solar_flux_1au  # Reflectance spectrum
    # # plt.scatter(nc_wavelength_nm, nc_reflectance, label="Ty no cloud (Original)",s=30,c="red",marker="o")
    #
    # plt.plot(nc_wavelength_nm, nc_reflectance)
    # plt.show()

    fig_dir = "/fast/jruffio/data/exosims/exosims_samples/figures"

    ## Load the albedo spectral model
    R_pl_template = np.inf

    lmin, lmax = 600, 1000
    # lmin, lmax = 750*0.9,750*1.1
    lmin_zoom, lmax_zoom = 759,771
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
        target_value = 750
        idx = np.argmin(np.abs(nc_wavelength_nm.value - target_value))
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

    envelop = np.nanmax(
        np.concatenate([nc_reflectance_all[None, :], nc_reflectance_o2[None, :], nc_reflectance_h2o[None, :]]), axis=0)
    envelop2, albedo_cont_subtracted, x_knots, y_fit = subtract_continuum_envelop(nc_wavelength_nm, envelop,
                                                                                  n_control=30, penalty=1)

    # nc_reflectance_o2 = nc_reflectance_o2 - envelop2
    # nc_reflectance_h2o = nc_reflectance_h2o - envelop2

    # plt.figure()
    # plt.plot(nc_wavelength_nm,nc_reflectance_all,label="all")
    # # plt.plot(nc_wavelength_nm,nc_reflectance_h2o,label="H2O")
    # plt.plot(nc_wavelength_nm,nc_reflectance_o2,label="O2")
    # # plt.plot(nc_wavelength_nm,envelop,label="envelop")
    # # plt.plot(nc_wavelength_nm,envelop2,label="envelop2")
    # plt.legend()
    # plt.show()

    scaling_fac = 0.2/np.mean(nc_reflectance_all[np.where((nc_wavelength_nm.value > 750*0.9) * (nc_wavelength_nm.value < 750*1.1))])
    pl_template_list = [nc_reflectance_all*scaling_fac, nc_reflectance_h2o*scaling_fac, nc_reflectance_o2*scaling_fac]
    pl_template_name_list = ['All Molecules', 'O$_2$', 'H$_2$O']

    R_list = [10000, 1000, 140]

    # Color setup
    color_original = 'black'
    color_list = ["#006699", "#ff9900", "#6600ff"]

    # Create figure with no vertical spacing
    n_panels = len(pl_template_name_list)
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(12, 1.5 * n_panels),
        sharex=True,
        gridspec_kw={'hspace': 0}
    )

    broadened_dict = {}

    for i,(ax,pl_template,pl_template_name) in enumerate(zip(axes,pl_template_list,pl_template_name_list)):
        print(pl_template_name)
        broad_datasets = []
        # plt.figure()
        # plt.plot(nc_wavelength_nm,pl_template,label="pl_template")
        for R in R_list:
            R_vec = np.full_like(nc_wavelength_nm.value, R)
            data_broad = broaden(nc_wavelength_nm, pl_template, R_vec, kernel="gaussian",n_jobs=8)
            broad_datasets.append((R, data_broad))
            broadened_dict[pl_template_name+f"{R}"] = data_broad

        #     plt.plot(nc_wavelength_nm,data_broad,label=f"{R}")
        # plt.legend()
        # plt.show()


        ax.plot(nc_wavelength_nm, pl_template, color=color_original, label='Original', linewidth=0.5,alpha=0.5)
        # Plot each broadened version
        for j, (R, data_broad) in enumerate(broad_datasets):
            ax.plot(nc_wavelength_nm, data_broad, '--',
                    color=color_list[j], label=f'R = {R}', linewidth=1.2)

        ax.set_ylim(0.0, 0.24)

        # Annotation
        txt = ax.text(0.01, 0.01, pl_template_name, transform=ax.transAxes, fontsize=20, verticalalignment='bottom', horizontalalignment='left')
        txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])

        # Y-label and grid
        ax.set_yticks([0,0.1,0.2])
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.set_ylabel('Albedo',fontsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
        # ax.grid(True)

        ax.fill_between([lmin_zoom,lmax_zoom], [0,0], [1,1], color="pink", alpha=0.75)
        ax.fill_between([675, lmin_zoom], [0, 0], [1, 1], color="grey", alpha=0.15)
        ax.fill_between([lmax_zoom,825], [0, 0], [1, 1], color="grey", alpha=0.15)

    # X-label only for bottom panel
    axes[-1].set_xlabel('Wavelength (nm)',fontsize=fontsize)

    # Show legend only in the top panel
    axes[1].legend(loc='lower right', fontsize=12)


    plt.xlim(lmin,lmax)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.98,hspace=0)  # absolutely no space between panels

    out_filename = os.path.join(fig_dir, "Ty_spectra.png")
    print("Saving " + out_filename)
    plt.savefig(out_filename, dpi=300)
    plt.savefig(out_filename.replace(".png", ".pdf"))
    # plt.show()

    if 1: #plot zoomed-in version
        plt.figure(figsize=(12, 3))
        ax=plt.gca()


        ax.plot(nc_wavelength_nm, pl_template_list[1], color=color_original, label='Original', linewidth=0.5, alpha=0.5)
        # Plot each broadened version
        for j, R in enumerate(R_list):
            ax.plot(nc_wavelength_nm, broadened_dict[pl_template_name_list[1]+f"{R}"], '--',
                    color=color_list[j], label=f'R = {R}', linewidth=1.2)

        ax.set_ylim(0.0, 0.24)

        # Annotation
        txt = ax.text(0.01, 0.01, pl_template_name_list[1], transform=ax.transAxes, fontsize=20, verticalalignment='bottom',
                horizontalalignment='left')
        txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])

        # ax.grid(True)
        ax.fill_between([lmin_zoom, lmax_zoom], [0, 0], [1, 1], color="pink", alpha=0.15)

        # Only show x-axis for the middle plot
        ax.set_xlabel('Wavelength (nm)',fontsize=fontsize)
        ax.xaxis.set_ticks_position('bottom')
        ax.tick_params(axis='x', which='both', labelbottom=True, labelsize=fontsize)
        plt.sca(ax)
        # plt.yticks([])
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.set_ylabel('Albedo',fontsize=fontsize)

        plt.xlim(lmin_zoom, lmax_zoom)
        plt.tight_layout()
        # plt.subplots_adjust(bottom=0.15, top=0.98,hspace=0)

        # Show legend only in the top panel
        ax.legend(loc='lower right', fontsize=fontsize)

        out_filename = os.path.join(fig_dir, "Ty_spectra_zoom.png")
        print("Saving " + out_filename)
        plt.savefig(out_filename, dpi=300)
        plt.savefig(out_filename.replace(".png", ".pdf"))
    plt.show()

