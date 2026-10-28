"""Shared visual constants for the moment-bound figures.

Keeping the geometry and typography here prevents the c-to-1 diagnostic plots
from drifting away from the higher-central-charge production figures.
"""

PLOT_FIGSIZE = (10.0, 7.0)
PLOT_SUBPLOT = {
    "left": 0.12,
    "right": 0.98,
    "top": 0.90,
    "bottom": 0.40,
}

TITLE_SIZE = 17
AXIS_LABEL_SIZE = 16
TICK_LABEL_SIZE = 13
LEGEND_SIZE = 11

# One neutral color for both numerical bound edges.  The blue is reserved for
# the translucent allowed-band fill; model families keep their own colors.
C_BOUND = "#444444"
C_ALLOWED_BAND = "#0072B2"
C_MODEL = "#009E73"
C_LIOUVILLE = "#222222"


#: global marker size factor of the moment figures (every marker on the axes and in the legends)
MARKER_SCALE = 0.65
#: opacity of the data markers (the bound is drawn on top of them; legend keys stay opaque)
MARKER_ALPHA = 0.8
#: width of the numerical bound edges
BOUND_LINEWIDTH = 1.3


def scale_markers(fig, factor=MARKER_SCALE):
    """Scale every marker of ``fig`` -- data markers and legend keys -- by ``factor``.

    Applied once, just before saving, so the figures share one marker scale however each artist
    chose its own size.  Edge widths scale by ``sqrt(factor)`` so small markers keep a visible rim.
    """
    from matplotlib.legend import Legend
    from matplotlib.lines import Line2D

    edge = factor ** 0.5
    data = [line for ax in fig.axes for line in ax.get_lines()]
    keys = [h for legend in fig.findobj(Legend) for h in legend.legend_handles if isinstance(h, Line2D)]
    for line in data + keys:
        if line.get_marker() in (None, "None", "", " "):
            continue
        line.set_markersize(line.get_markersize() * factor)
        line.set_markeredgewidth(line.get_markeredgewidth() * edge)
    for line in data:
        if line.get_marker() not in (None, "None", "", " ") and line.get_linestyle() in ("None", "none", "", " "):
            line.set_alpha(MARKER_ALPHA if line.get_alpha() is None else line.get_alpha() * MARKER_ALPHA)


def set_plot_style(plt):
    """Apply the common typography to a Matplotlib module."""

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "legend.fontsize": LEGEND_SIZE,
    })


__all__ = [
    "PLOT_FIGSIZE", "PLOT_SUBPLOT", "TITLE_SIZE", "AXIS_LABEL_SIZE",
    "TICK_LABEL_SIZE", "LEGEND_SIZE", "C_BOUND", "C_ALLOWED_BAND",
    "C_MODEL", "C_LIOUVILLE", "MARKER_SCALE", "MARKER_ALPHA", "BOUND_LINEWIDTH", "scale_markers", "set_plot_style",
]
