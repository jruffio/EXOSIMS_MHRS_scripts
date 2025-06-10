import numpy as np

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
