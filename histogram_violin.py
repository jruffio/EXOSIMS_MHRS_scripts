import numpy as np
import matplotlib.pyplot as plt

def histogram_violin(ax, data, position, width=0.3, bins=20, color='C0', alpha=0.6):
    """
    Plot a histogram-based violin plot at a specific x-axis position.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to plot on.
    data : array-like
        1D array of data values.
    position : float
        The x-position at which to draw the violin.
    width : float
        Maximum half-width of the violin.
    bins : int
        Number of histogram bins.
    color : str
        Fill color for the violin.
    alpha : float
        Transparency level.
    """
    data = np.asarray(data)
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Normalize histogram to max width
    hist_widths = hist / hist.max() * width

    # Plot left side
    ax.fill_betweenx(bin_centers, position - hist_widths, position,
                     facecolor=color, alpha=alpha, edgecolor=None)

    # Plot right side
    ax.fill_betweenx(bin_centers, position, position + hist_widths,
                     facecolor=color, alpha=alpha, edgecolor=None)

    # # Optionally add median line
    # median = np.median(data)
    # ax.plot([position - width, position + width], [median, median],
    #         color='k', linestyle='-', linewidth=1)


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
                # ax.set_ylim(0, np.max([1,p95 * 1.1]))
                ax.set_ylim(0, np.max([1,10]))

        # Axis formatting
        ax.set_title(title)
        # ax.text(0.5, 0.99, title, transform=ax.transAxes,
        #         fontsize=12, verticalalignment='top', horizontalalignment='center')
        # if label is not None:
        #     ax.text(0.5, 0.85, label, transform=ax.transAxes, color = "grey",
        #             fontsize=12, verticalalignment='top', horizontalalignment='center')
        ax.set_xlabel("Spectral Resolution (R=$\lambda/d\lambda$)")
        ax.set_xticks(logR)
        myxtickslabels = []
        for r in R_list:
            if r >= 1000:
                myxtickslabels.append("{0:.0f}k".format(r/1000.))
            else:
                myxtickslabels.append(str(r))
        ax.set_xticklabels(myxtickslabels, fontsize=12)
        # ax.grid(True)

        if idx == 0:
            if label is not None:
                ax.set_ylabel(label+"\nS/N")
            else:
                ax.set_ylabel("S/N")
        elif idx == 3:
            ax.legend(loc='upper left', fontsize=10, frameon=True)
        else:
            ax.tick_params(labelleft=True)  # Ensure y-axis tick labels are shown

        plt.tight_layout()