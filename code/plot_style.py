import matplotlib.pyplot as plt
from pathlib import Path

FONT_SIZE_BASE = 14
FONT_SIZE_SMALL = 12
FONT_SIZE_LARGE = 18
DPI = 300


def init_plotting_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
        "font.size": FONT_SIZE_BASE,
        "axes.labelsize": FONT_SIZE_BASE,
        "axes.titlesize": FONT_SIZE_LARGE,
        "xtick.labelsize": FONT_SIZE_SMALL,
        "ytick.labelsize": FONT_SIZE_SMALL,
        "legend.fontsize": FONT_SIZE_SMALL,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.linewidth": 0.8,
        "lines.linewidth": 2.5,
        "patch.linewidth": 0.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })


def save_figure(fig, filepath, dpi=DPI):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure to {filepath}")


def add_panel_labels(fig, axes, labels=None, x_offset=-0.01, y_offset=0.12,
                     fontsize=None):
    labels = labels or [chr(ord("a") + i) for i in range(len(axes))]
    fontsize = fontsize or FONT_SIZE_LARGE + 4
    for ax, label in zip(axes, labels):
        bbox = ax.get_position()
        fig.text(bbox.x0 + x_offset, bbox.y1 + y_offset, label,
                 fontsize=fontsize, fontweight="bold", ha="right", va="bottom")
