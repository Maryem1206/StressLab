"""Shared Excel-like chart style for StressLab v2 figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Excel default theme colors
BLUE = "#4472C4"       # Baseline
ORANGE = "#ED7D31"     # Adverse
RED = "#C00000"        # Severe
LIGHT_BLUE = "#9DC3E6" # secondary series (e.g. default rate in dual axis)
NAVY = "#203864"       # primary dark series (e.g. EAD)
GRAY = "#A6A6A6"
GREEN = "#548235"
GRID = "#D9D9D9"

plt.rcParams.update({
    "font.family": "Calibri, Segoe UI, Arial, sans-serif",
    "font.size": 11,
    "axes.edgecolor": "#BFBFBF",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "legend.frameon": False,
    "xtick.color": "#404040",
    "ytick.color": "#404040",
    "axes.labelcolor": "#404040",
    "text.color": "#262626",
})


def style_axes(ax, pct_left=True, pct_right=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(pct_right)
    ax.spines["left"].set_visible(True)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", visible=True)
    ax.tick_params(axis="both", length=0)
    if pct_left:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))


def bottom_legend(fig, ax, ncol=3):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=ncol,
               bbox_to_anchor=(0.5, -0.02), frameon=False)


def caption(fig, text):
    fig.text(0.5, -0.11, text, ha="center", va="top", fontsize=8.5,
              color="#595959", wrap=True)
