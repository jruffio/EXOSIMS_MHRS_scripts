import numpy as np
import matplotlib.pyplot as plt
import os
from etc_utils import subtract_continuum_envelop
import astropy.units as u
from EXOSIMS.OpticalSystem.MHRS import broaden
import matplotlib.patheffects as PathEffects
from matplotlib.patches import ConnectionPatch

if __name__ == "__main__":

    fontsize = 16
    fig_dir = "/fast/jruffio/data/exosims/exosims_samples/figures"

    # lmin, lmax = 761, 765
    # lmin_zoom, lmax_zoom = 762, 764
    lmin, lmax = 600, 1000
    lmin_zoom, lmax_zoom = 759, 771
    R = 2e6

    # -----------------------------
    # Load spectra
    # -----------------------------
    nc_reflectance_all = []
    nc_reflectance_o2 = []
    nc_reflectance_h2o = []

    for clouds in ["highcloud", "lowcloud", "clearsky"]:

        fname = f"/fast/jruffio/data/exosims/model_Ty/earth_maxres/earth_icrccm_hitran2020_{clouds}_50_100000cm-1_toa_R150000.rad"
        nc_data = np.loadtxt(fname, comments='#')
        nc_where_wvs = np.where((nc_data[:, 0] > lmin) * (nc_data[:, 0] < lmax))
        nc_data = nc_data[nc_where_wvs[0], :]
        nc_wavelength_nm = nc_data[:, 0] * u.nm
        nc_reflectance_all.append(nc_data[:, 1])

        fname = f"/fast/jruffio/data/exosims/model_Ty/earth_maxres/earth_icrccm_hitran2020_{clouds}_h2o_50_100000cm-1_toa_R150000.rad"
        nc_data = np.loadtxt(fname, comments='#')
        nc_data = nc_data[nc_where_wvs[0], :]
        nc_reflectance_o2.append(nc_data[:, 1])

        fname = f"/fast/jruffio/data/exosims/model_Ty/earth_maxres/earth_icrccm_hitran2020_{clouds}_o2_50_100000cm-1_toa_R150000.rad"
        nc_data = np.loadtxt(fname, comments='#')
        nc_data = nc_data[nc_where_wvs[0], :]
        nc_reflectance_h2o.append(nc_data[:, 1])

    nc_reflectance_all = np.nansum(np.array(nc_reflectance_all) * np.array([0.25, 0.25, 0.5])[:, None], axis=0)
    nc_reflectance_o2  = np.nansum(np.array(nc_reflectance_o2)  * np.array([0.25, 0.25, 0.5])[:, None], axis=0)
    nc_reflectance_h2o = np.nansum(np.array(nc_reflectance_h2o) * np.array([0.25, 0.25, 0.5])[:, None], axis=0)

    envelop = np.nanmax(
        np.vstack([nc_reflectance_all, nc_reflectance_o2, nc_reflectance_h2o]),
        axis=0
    )

    subtract_continuum_envelop(nc_wavelength_nm, envelop, n_control=30, penalty=1)

    scaling_fac = 0.2/np.mean(
        nc_reflectance_all[(nc_wavelength_nm.value > 750*0.9) & (nc_wavelength_nm.value < 750*1.1)]
    )

    pl_template_list = [
        nc_reflectance_all*scaling_fac,
        nc_reflectance_o2*scaling_fac,
        nc_reflectance_h2o*scaling_fac
    ]

    pl_template_name_list = ['All Molecules', 'O$_2$', 'H$_2$O']
    R_list = [10000, 1000, 140]

    color_original = 'black'
    color_list = ["#006699", "#ff9900", "#6600ff"]

    # -----------------------------
    # Figure layout
    # -----------------------------
    fig = plt.figure(figsize=(12,8))

    gs_outer = fig.add_gridspec(2,1, height_ratios=[3,1.4], hspace=0.35)

    gs_top = gs_outer[0].subgridspec(3,1, hspace=0)

    axes = [fig.add_subplot(gs_top[i]) for i in range(3)]
    ax_zoom = fig.add_subplot(gs_outer[1])

    broadened_dict = {}

    # -----------------------------
    # Main panels
    # -----------------------------
    for ax,pl_template,name in zip(axes,pl_template_list,pl_template_name_list):
        print(name)

        broad_datasets = []

        for R in R_list:
            print(R)
            R_vec = np.full_like(nc_wavelength_nm.value, R)
            data_broad = broaden(nc_wavelength_nm, pl_template, R_vec,
                                 kernel="gaussian", n_jobs=8)
            broadened_dict[name+str(R)] = data_broad
            broad_datasets.append((R,data_broad))

        ax.plot(nc_wavelength_nm, pl_template,
                color=color_original, linewidth=0.5, alpha=0.5)

        for j,(R,data_broad) in enumerate(broad_datasets):
            ax.plot(nc_wavelength_nm, data_broad,'--',
                    color=color_list[j], linewidth=1.2)

        ax.set_ylim(0,0.24)
        ax.set_ylabel("Albedo",fontsize=fontsize)
        ax.set_yticks([0,0.1,0.2])
        ax.tick_params(axis='y',labelsize=fontsize)

        txt = ax.text(
            0.01,0.01,name,
            transform=ax.transAxes,
            fontsize=20,
            va='bottom',ha='left'
        )
        txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])

        ax.fill_between([lmin_zoom,lmax_zoom],[0,0],[1,1],color="pink",alpha=0.75)
        ax.fill_between([675, lmin_zoom], [0, 0], [1, 1], color="grey", alpha=0.15)
        ax.fill_between([lmax_zoom,825], [0, 0], [1, 1], color="grey", alpha=0.15)

        ax.set_xlim(lmin,lmax)

    # remove x ticks on top two panels
    for ax in axes[:-1]:
        ax.tick_params(axis='x', labelbottom=False)

    axes[-1].set_xlabel("Wavelength (nm)",fontsize=fontsize)
    axes[-1].tick_params(axis='x',labelsize=fontsize)

    axes[1].legend(['R = 150k','R = 10k','R = 1k','R = 140'],
                   loc='lower right',fontsize=12)

    # -----------------------------
    # Zoom panel
    # -----------------------------
    idx = 1  # O2

    ax_zoom.plot(nc_wavelength_nm,
                 pl_template_list[idx],
                 color=color_original,
                 linewidth=0.5,alpha=0.5)

    for j,R in enumerate(R_list):
        ax_zoom.plot(
            nc_wavelength_nm,
            broadened_dict[pl_template_name_list[idx]+str(R)],
            '--',
            color=color_list[j],
            linewidth=1.2
        )

    ax_zoom.set_xlim(lmin_zoom,lmax_zoom)
    ax_zoom.set_ylim(0,0.24)
    ax_zoom.set_ylabel("Albedo",fontsize=fontsize)
    ax_zoom.set_xlabel("Wavelength (nm)",fontsize=fontsize)
    ax_zoom.tick_params(labelsize=fontsize)

    # ax_zoom.fill_between([lmin_zoom,lmax_zoom],[0,0],[1,1],
    #                      color="pink",alpha=0.15)

    ax_zoom.legend(['R = 150k','R = 10k','R = 1k','R = 140'],loc='lower right',fontsize=fontsize)

    # -----------------------------
    # Connection lines
    # -----------------------------
    ax_source = axes[1]

    y_zoom_top = ax_zoom.get_ylim()[1]

    for x in [lmin_zoom,lmax_zoom]:

        con = ConnectionPatch(
            xyA=(x,0), coordsA=ax_source.transData,
            xyB=(x,y_zoom_top), coordsB=ax_zoom.transData,
            color="deeppink", linewidth=1.2
        )

        fig.add_artist(con)

    # -----------------------------
    # Save
    # -----------------------------
    out = os.path.join(fig_dir,"Ty_spectra_with_zoom.png")
    print("Saving",out)
    plt.savefig(out,dpi=300)
    plt.savefig(out.replace(".png",".pdf"))

    plt.show()

