import numpy as np
import matplotlib.pyplot as plt
import os
from etc_utils import subtract_continuum_envelop,gaussian_broaden

if __name__ == "__main__":

    lmin,lmax = 600,1000
    lmin_zoom, lmax_zoom = 759,765

    fig_dir = "/exosims_samples/figures"
    # File paths and resolutions
    base_dir1 = '/fast/jruffio/data/exosims/model_Renyu/HighResSpec/EarthSpec/'
    base_dir2 = '/fast/jruffio/data/exosims/model_Renyu/HighResSpec/JupiterSpec/'
    for base_dir,spec_label in [(base_dir1,"Earth_HighCloud"),(base_dir1,"Earth_NoCloud")]:
    # for base_dir,spec_label in [(base_dir1,"Earth_NoCloud")]:
        file_original = base_dir + 'GeometricA_'+spec_label+'_UltraRes.dat'
        R_list = [10000,1000,140]
        file_broad_template = base_dir + 'GeometricA_'+spec_label+'_UltraRes_500-1500nm_R{0:.0f}.dat'

        # Read original spectrum
        data_original = np.loadtxt(file_original, dtype=float)

        # Load all broadened versions
        broad_datasets = []
        for R in R_list:
            file_broad = file_broad_template.format(R)
            data_broad = np.loadtxt(file_broad, dtype=float)
            broad_datasets.append((R, data_broad))

        # Molecule indices and labels (in plotting order)
        plot_indices = [
            (1, 'All Molecules'),
            (6, 'O$_2$'),
            (2, 'H$_2$O'),
            # (3, 'CO$_2$'),
            # (4, 'N$_2$O'),
            # (5, 'CH$_4$'),
            # (7, 'O$_3$'),
        ]

        # Color setup
        color_original = 'black'
        color_list = ["#006699", "#ff9900", "#6600ff"]

        # Create figure with no vertical spacing
        n_panels = len(plot_indices)
        fig, axes = plt.subplots(
            n_panels, 1, figsize=(8, 1.5 * n_panels),
            sharex=True,
            gridspec_kw={'hspace': 0}
        )

        # Plot each molecule panel
        for i, (ax, (index, label)) in enumerate(zip(axes, plot_indices)):
            # Plot original
            y = data_original[:, index]
            ax.plot(data_original[:, 0], y, color=color_original, label='Original', linewidth=0.5,alpha=0.5)

            # Plot each broadened version
            for j, (R, data_broad) in enumerate(broad_datasets):
                ax.plot(data_broad[:, 0], data_broad[:, index], '--',
                        color=color_list[j], label=f'R = {R}', linewidth=1.2)

            # Adaptive y-limits
            all_y = [data_original[np.where((lmin<data_original[:, 0]) & (data_original[:, 0]<lmax))[0], index]] + [d[np.where((lmin<d[:, 0]) & (d[:, 0]<lmax))[0], index] for _, d in broad_datasets]
            y_min = min([np.nanmin(y) for y in all_y])
            y_max = max([np.nanmax(y) for y in all_y])
            # ax.set_ylim(y_min, y_max)
            ax.set_ylim(0.0, y_max*1.1)

            # Annotation
            ax.text(0.01, 0.01, label, transform=ax.transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='left')

            # Y-label and grid
            ax.set_ylabel('Albedo')
            ax.grid(True)

            ax.fill_between([lmin_zoom,lmax_zoom], [0,0], [1,1], color="pink", alpha=0.75)

        # X-label only for bottom panel
        axes[-1].set_xlabel('Wavelength [nm]')

        # Show legend only in the top panel
        axes[1].legend(loc='lower right', fontsize=10)


        plt.xlim(lmin,lmax)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, top=0.98,hspace=0)  # absolutely no space between panels

        out_filename = os.path.join(fig_dir, spec_label+"_spectra.png")
        print("Saving " + out_filename)
        plt.savefig(out_filename, dpi=300)
        plt.savefig(out_filename.replace(".png", ".pdf"))

        if 1:  # plot zoomed-in version
            n_panels = len(plot_indices)
            fig, axes = plt.subplots(
                n_panels, 1, figsize=(4, 1.5 * n_panels),
                sharex=True,
                gridspec_kw={'hspace': 0}
            )

            # Make all axes invisible except the middle one
            for i, ax in enumerate(axes):
                if i != 1:
                    ax.set_visible(False)

            # Middle axis (e.g. index 1)
            i, ax = 1, axes[1]
            index, label = plot_indices[1]

            # Plot original
            y = data_original[:, index]
            ax.plot(data_original[:, 0], y, color=color_original, label='Original', linewidth=0.5, alpha=0.5)

            # Plot each broadened version
            for j, (R, data_broad) in enumerate(broad_datasets):
                ax.plot(data_broad[:, 0], data_broad[:, index], '--',
                        color=color_list[j], label=f'R = {R}', linewidth=1.2)

            ax.set_ylim(0.0, y_max * 1.1)

            # Annotation
            ax.text(0.01, 0.01, label, transform=ax.transAxes, fontsize=12,
                    verticalalignment='bottom', horizontalalignment='left')

            ax.grid(True)
            ax.fill_between([lmin_zoom, lmax_zoom], [0, 0], [1, 1], color="pink", alpha=0.25)

            # Only show x-axis for the middle plot
            ax.set_xlabel('Wavelength [nm]')
            ax.xaxis.set_ticks_position('bottom')
            ax.tick_params(axis='x', which='both', labelbottom=True)
            plt.sca(ax)
            plt.yticks([])

            plt.xlim(lmin_zoom, lmax_zoom)
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15, top=0.98,hspace=0)

            out_filename = os.path.join(fig_dir, spec_label + "_spectra_zoom.png")
            print("Saving " + out_filename)
            plt.savefig(out_filename, dpi=300)
            plt.savefig(out_filename.replace(".png", ".pdf"))

        if 1: #plot zoomed-in version
            # Create figure with no vertical spacing
            n_panels = len(plot_indices)
            fig, axes = plt.subplots(
                n_panels, 1, figsize=(4, 1.5 * n_panels),
                sharex=True,
                gridspec_kw={'hspace': 0}
            )

            # Plot each molecule panel
            # for i, (ax, (index, label)) in enumerate(zip(axes, plot_indices)):
            if 1:
                i, ax =  1,axes[1]
                index, label = plot_indices[1]
                # Plot original
                y = data_original[:, index]
                ax.plot(data_original[:, 0], y, color=color_original, label='Original', linewidth=0.5,alpha=0.5)

                # Plot each broadened version
                for j, (R, data_broad) in enumerate(broad_datasets):
                    ax.plot(data_broad[:, 0], data_broad[:, index], '--',
                            color=color_list[j], label=f'R = {R}', linewidth=1.2)

                # ax.set_ylim(y_min, y_max)
                ax.set_ylim(0.0, y_max*1.1)

                # Annotation
                ax.text(0.01, 0.01, label, transform=ax.transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='left')

                # Y-label and grid
                # ax.set_ylabel('Albedo')
                ax.grid(True)
                ax.fill_between([lmin_zoom,lmax_zoom], [0,0], [1,1], color="pink", alpha=0.25)

                plt.sca(ax)
                plt.yticks([])

            # X-label only for bottom panel
            axes[1].set_xlabel('Wavelength [nm]')

            # # Show legend only in the top panel
            # axes[i].legend(loc='lower right', fontsize=10)

            plt.xlim(lmin_zoom,lmax_zoom)
            plt.tight_layout()
            plt.subplots_adjust(hspace=0)  # absolutely no space between panels

            out_filename = os.path.join(fig_dir, spec_label+"_spectra_zoom.png")
            print("Saving " + out_filename)
            plt.savefig(out_filename, dpi=300)
            plt.savefig(out_filename.replace(".png", ".pdf"))
    plt.show()

