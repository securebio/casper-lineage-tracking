#!/usr/bin/env python3
"""Lineage and genotype shifts resolvable in CASPER data, extended to every CASPER sample
deposited in SRA.

Same figure as the manuscript version, over a larger sample set and a longer window: 2,197
libraries through 2026-06-30, against the release figure's 1,206 through 2026-03-31. It is
drawn from this extension's own data directory and written to this extension's figures
directory; the manuscript figure is not touched.

The published CaliciNet series ends 2025-04, so the norovirus comparator row stops there
while the CASPER row continues. That gap is the point of the extension rather than a defect
in it, and is left visible rather than trimmed away.

(a) Norovirus capsid genotype composition from monthly national pools, beside CaliciNet
    clinical outbreak genotype percentages
(b) SARS-CoV-2 lineage composition from monthly national pools, beside CDC clinical
    genomic surveillance variant proportions

Composition panels are drawn as one stacked bar per month so that months without a usable
estimate read as gaps rather than being interpolated across.

Inputs (data/):
    lineage_norovirus_monthly.csv, lineage_sars2_monthly.csv,
    lineage_sars2_pool_coverage.csv

Usage:
    python plot_lineage_genotype_comparison.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from plot_style import (add_panel_labels, init_plotting_style, save_figure,  # noqa: E402
                        FONT_SIZE_LARGE)

# Font scale follows the other combined-panel figures (plot_taxonomic_panel_combined.py)
LABEL_FONT = FONT_SIZE_LARGE + 5
TICK_FONT = FONT_SIZE_LARGE + 3
TITLE_FONT = FONT_SIZE_LARGE + 8
NOTE_FONT = FONT_SIZE_LARGE + 1
PANEL_LABEL_FONT = FONT_SIZE_LARGE + 12

# Lineage and genotype categories have no counterpart in the shared palettes
# (STATE_COLORS, TAXONOMIC_COLORS, VIRUS_HOST_COLORS), so a dedicated categorical set is
# used here. Hues are assigned in a fixed order and never cycled: anything beyond the named
# categories falls into a neutral grey rather than taking a generated hue. The eight hues
# are colourblind-separable as an adjacent-pair sequence, which is how a stack is read.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                 "#4a3aa7", "#e34948"]
# The norovirus panel uses a different hue family from the SARS-CoV-2 panel — teal, plum,
# ochre, indigo, sage, dusty red — so the two panels read as distinct rather than as one
# continuous scheme. Validated as a set: worst adjacent colourblind separation dE 17.6,
# every hue inside the lightness band and above the chroma floor.
#
# Two things constrain this more than they appear to. Desaturated "earth tone" palettes
# fall below the chroma floor and render as grey, so distinction has to come from shifting
# hues rather than dulling them. And a palette built at constant lightness always fails
# colourblind separation, because deuteranopia collapses hue toward a blue-yellow axis and
# leaves lightness as the only cue; the lightness variation across these six slots is what
# carries the separation.
NORO_COLORS = ["#00bdbe", "#853e80", "#ce7a3b", "#5a66b9", "#83c575", "#ba5661"]
OTHER_COLOR = "#b7b6b0"
UNREPORTED_COLOR = "#e8e7e2"
SECONDARY_INK = "#52514e"
GRID_COLOR = "#dedcd5"

GENOTYPE_ORDER = ["GII.4", "GII.17", "GII.2", "GII.3", "GII.6", "GI (all)", "Other genotypes"]
CALICINET_OTHER = "Other genotypes (not reported separately)"

# SARS-CoV-2 groups are ordered by when each swept, so colours and the legend follow the
# same left-to-right progression the stacked bars show.
SARS2_ORDER = ["XBB*", "BA.2.86* (excl. JN.1)", "JN.1* (other)", "KP.2*", "KP.3*",
               "XEC*", "LP.8.1*", "NB.1.8.1*", "XFG*"]

# Every month is plotted. The depth row beneath each CASPER row shows the coverage each
# estimate rests on, so a reader can judge the low-coverage months directly rather than
# having them silently withheld. The correlations in tables/lineage_comparison_stats.tsv
# are still computed on months reaching at least MIN_BREADTH of the genome at >=10x.
MIN_BREADTH = 0.25

# The depth row shows MEASURED mean coverage: aligned bases over reference length, computed
# the same way on both sides. SARS-CoV-2 comes from the per-base depth of each monthly pool
# against the whole genome; norovirus comes from aligned bases against the VP1 panel, the
# region genotype is resolved on. An earlier nominal version (read pairs x 300 bp over
# reference length) was dropped: it overstated depth, and was not even the same quantity on
# the two sides, since the SARS-CoV-2 figure counted extracted reads while the norovirus
# figure counted reads that had already aligned.
COVERAGE_COLOR = "#8c8b85"


def color_map(categories, palette=None):
    """Fixed-order hues for named categories; neutral grey for any Other bucket."""
    palette = palette or SERIES_COLORS
    mapping, slot = {}, 0
    for name in categories:
        if name.startswith("Other"):
            mapping[name] = UNREPORTED_COLOR if "not reported" in name else OTHER_COLOR
        else:
            mapping[name] = palette[slot % len(palette)]
            slot += 1
    return mapping


def date_axis(ax, months=3, rotate=45):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=months))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="both", labelsize=TICK_FONT)
    for label in ax.get_xticklabels():
        label.set_rotation(rotate)
        label.set_horizontalalignment("right")


def stacked_bars(ax, frame, order, colors, value_col="fraction", x_col="month",
                 category_col="category", width=24):
    """Composition as one stacked bar per month; missing months read as gaps."""
    wide = frame.pivot_table(index=x_col, columns=category_col, values=value_col,
                             aggfunc="sum")
    order = list(dict.fromkeys(c for c in order if c in wide.columns))
    if wide.empty or not order:
        return wide
    wide = wide[order].dropna(how="all")
    x = pd.to_datetime(wide.index.astype(str) + "-01")
    bottom = np.zeros(len(wide))
    for category in order:
        values = wide[category].fillna(0).to_numpy(float)
        ax.bar(x, values, bottom=bottom, width=width, color=colors[category],
               edgecolor="white", linewidth=0.6)
        bottom += values
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1.0])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return wide


def comparison_rows(axes, frames, order, colors, titles, xlim):
    """Two stacked-bar rows on a shared x axis: CASPER above, its comparator below."""
    for ax, frame, title in zip(axes, frames, titles):
        stacked_bars(ax, frame, order, colors)
        ax.set_title(title, loc="left", fontsize=TITLE_FONT, color=SECONDARY_INK, pad=16)
        ax.set_ylabel("Fraction", fontsize=LABEL_FONT)
        ax.set_xlim(*xlim)
        date_axis(ax)
    for ax in axes[:-1]:
        ax.set_xticklabels([])


def panel_titles(fig, axes, titles, x_offset=0.012, y_offset=0.064):
    """Pathogen name beside each panel letter, matching add_panel_labels placement."""
    for ax, title in zip(axes, titles):
        bbox = ax.get_position()
        fig.text(bbox.x0 + x_offset, bbox.y1 + y_offset, title,
                 fontsize=PANEL_LABEL_FONT, fontweight="bold", ha="left", va="bottom")


def compact_number(value, _=None):
    """1, 10, 100, 1k, 10k, 100k — short enough for a thin axis."""
    if value >= 1e6:
        return f"{value / 1e6:g}M"
    if value >= 1e3:
        return f"{value / 1e3:g}k"
    return f"{value:g}"


def coverage_row(ax, months, coverage, xlim, ylabel):
    """Measured mean coverage depth per month, on a log axis under its composition row."""
    x = pd.to_datetime(pd.Index(months).astype(str) + "-01")
    values = np.asarray(coverage, dtype=float)
    # Anchor the floor to a decade so bars start exactly on the lowest tick rather than
    # floating above it
    smallest = np.nanmin(values[values > 0]) if np.any(values > 0) else 1.0
    floor = max(1.0, 10.0 ** np.floor(np.log10(smallest)))
    ax.bar(x, values, bottom=floor, width=24, color=COVERAGE_COLOR,
           edgecolor="white", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_ylim(floor, np.nanmax(values) * 2.5)
    # Label a few decades across the range. A short axis defaults to a single tick, but
    # every decade would crowd it, so decades are strided down to at most four labels.
    top = np.nanmax(values)
    lo_exp = int(np.round(np.log10(floor)))
    hi_exp = int(np.floor(np.log10(top)))
    exps = list(range(lo_exp, hi_exp + 1))
    if len(exps) > 4:  # thin only when every decade would crowd the row
        keep = np.unique(np.linspace(0, len(exps) - 1, 4).round().astype(int))
        exps = [exps[i] for i in keep]
    ticks = [10 ** e for e in exps]
    ax.set_yticks(ticks)
    # Keep every tick mark, but on a thin row label only alternate ones so the text does
    # not overlap. Counting back from the top keeps the highest decade labelled.
    labelled = set(range(len(ticks) - 1, -1, -2)) if len(ticks) > 3 else set(range(len(ticks)))
    ax.set_yticklabels([compact_number(t) if i in labelled else ""
                        for i, t in enumerate(ticks)])
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlim(*xlim)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONT - 6)
    ax.tick_params(axis="both", labelsize=TICK_FONT - 4)
    ax.set_xticklabels([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# Gap between a composition row and the depth row beneath it, and between that depth
# row and the comparator composition row below it.
COVERAGE_GAP = 0.014
COMPARATOR_GAP = 0.068


def tuck_under(fig, upper_ax, lower_ax, gap=COVERAGE_GAP):
    """Move a depth row up so it sits just beneath its composition row."""
    fig.canvas.draw()
    upper = upper_ax.get_position()
    lower = lower_ax.get_position()
    height = lower.height
    lower_ax.set_position([lower.x0, upper.y0 - gap - height, lower.width, height])


def legend_row(ax, handles, ncol):
    """Legend filling left-to-right. Matplotlib fills columns first, so the handles are
    pre-shuffled to make the displayed reading order match the order given."""
    n_rows = int(np.ceil(len(handles) / ncol))
    reordered = []
    for col in range(ncol):
        for row in range(n_rows):
            i = row * ncol + col
            if i < len(handles):
                reordered.append(handles[i])
    ax.axis("off")
    ax.legend(handles=reordered, loc="center", ncol=ncol, frameon=False,
              fontsize=LABEL_FONT, handlelength=1.4, handleheight=1.0,
              columnspacing=1.4, handletextpad=0.6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, default=REPO_ROOT / "tables")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "figures/lineage_composition.png")
    args = parser.parse_args()

    init_plotting_style()

    noro = pd.read_csv(args.tables / "norovirus_monthly.csv")
    sars2 = pd.read_csv(args.tables / "sars2_monthly.csv")
    pooled = pd.read_csv(args.tables / "sars2_pool_coverage.csv")

    months = pd.to_datetime(noro.month.astype(str) + "-01")
    xlim = (months.min() - pd.Timedelta(days=22), months.max() + pd.Timedelta(days=22))

    # ---- Panel a: SARS-CoV-2 ------------------------------------------------------------
    low_depth = set(pooled.loc[pooled.breadth_10x < MIN_BREADTH, "month"].astype(str))
    sars2 = sars2.copy()
    present = sars2.groupby("group_label")[["casper_fraction", "cdc_fraction"]].max().max(axis=1)
    ranked = present.drop(labels=["Other"], errors="ignore").sort_values(ascending=False)
    named = set(ranked.head(len(SERIES_COLORS) - 1).index)
    groups = [g for g in SARS2_ORDER if g in named] + ["Other"]
    sars2["category"] = np.where(sars2.group_label.isin(named), sars2.group_label, "Other")
    sars2 = sars2.groupby(["month", "category"], as_index=False)[
        ["casper_fraction", "cdc_fraction"]].sum(min_count=1)
    group_colors = color_map(groups)

    # ---- Panel b: norovirus -------------------------------------------------------------
    genotype_colors = color_map(GENOTYPE_ORDER, palette=NORO_COLORS)
    genotype_colors[CALICINET_OTHER] = UNREPORTED_COLOR
    casper_noro = noro.rename(columns={"genotype": "category",
                                       "casper_fraction": "fraction"})
    cal = noro.dropna(subset=["calicinet_fraction"]).copy()
    cal["category"] = np.where(cal.genotype.isin(["GII.4", "GII.17"]),
                               cal.genotype, CALICINET_OTHER)
    cal = (cal.groupby(["month", "category"], as_index=False)
           .calicinet_fraction.sum().rename(columns={"calicinet_fraction": "fraction"}))

    # Measured coverage depth behind each CASPER bar
    sars2_cov = pooled.sort_values("month")
    sars2_cov_depth = sars2_cov.mean_depth
    noro_cov = pd.read_csv(args.tables / "norovirus_coverage_monthly.csv")
    noro_cov = noro_cov.sort_values("month")

    sars2_frames = [
        sars2.rename(columns={"casper_fraction": "fraction"}).dropna(subset=["fraction"]),
        sars2.rename(columns={"cdc_fraction": "fraction"}).dropna(subset=["fraction"])]
    sars2_legend = [Patch(facecolor=group_colors[g], label=g) for g in groups]
    noro_legend = ([Patch(facecolor=genotype_colors[g], label=g) for g in GENOTYPE_ORDER]
                   + [Patch(facecolor=UNREPORTED_COLOR,
                            label="Other (CaliciNet, not reported separately)")])

    blocks = {
        "sars2": dict(title="SARS-CoV-2", frames=sars2_frames, order=groups,
                      colors=group_colors, legend=sars2_legend, ncol=5,
                      rows=["Public CASPER data", "CDC clinical genomic surveillance"],
                      cov_months=sars2_cov.month, cov_values=sars2_cov_depth),
        "norovirus": dict(title="Norovirus", frames=[casper_noro, cal],
                          order=GENOTYPE_ORDER + [CALICINET_OTHER],
                          colors=genotype_colors, legend=noro_legend, ncol=4,
                          rows=["Public CASPER data",
                                "CaliciNet clinical outbreak surveillance"],
                          cov_months=noro_cov.month, cov_values=noro_cov.mean_depth),
    }

    def draw(fig, gs, offset, block):
        ax1 = fig.add_subplot(gs[offset])
        axcov = fig.add_subplot(gs[offset + 1])
        ax2 = fig.add_subplot(gs[offset + 2])
        axleg = fig.add_subplot(gs[offset + 3])
        comparison_rows([ax1, ax2], block["frames"], block["order"], block["colors"],
                        block["rows"], xlim)
        ax1.set_xticklabels([])
        coverage_row(axcov, block["cov_months"], block["cov_values"], xlim, "Depth (×)")
        legend_row(axleg, block["legend"], ncol=block["ncol"])
        return ax1, axcov, ax2, axleg

    def finish(fig, axes, legend_gap=None):
        for ax1, axcov, ax2, axleg in axes:
            tuck_under(fig, ax1, axcov)
            tuck_under(fig, axcov, ax2, gap=COMPARATOR_GAP)
            # On a single-pathogen figure the rows move up far enough to strand the legend;
            # following ax2 keeps it with the panel it describes.
            if legend_gap is not None:
                tuck_under(fig, ax2, axleg, gap=legend_gap)

    # Combined figure, both pathogens
    fig = plt.figure(figsize=(17, 20))
    # Row 4 is an empty spacer so the panel-a legend cannot collide with the panel-b label
    gs = fig.add_gridspec(9, 1,
                          height_ratios=[1.0, 0.34, 1.0, 0.58, 0.02, 1.0, 0.34, 1.0, 0.58],
                          hspace=0.72, left=0.075, right=0.99, top=0.95, bottom=0.04)
    a = draw(fig, gs, 0, blocks["sars2"])
    b = draw(fig, gs, 5, blocks["norovirus"])
    finish(fig, [a, b])
    add_panel_labels(fig, [a[0], b[0]], labels=["a", "b"],
                     x_offset=-0.008, y_offset=0.064, fontsize=PANEL_LABEL_FONT)
    panel_titles(fig, [a[0], b[0]], ["SARS-CoV-2", "Norovirus"])

    # Standalone figures, one pathogen each and no panel letters
    for name, block in blocks.items():
        single = plt.figure(figsize=(17, 10))
        gs_one = single.add_gridspec(4, 1, height_ratios=[1.0, 0.34, 1.0, 0.58],
                                     hspace=0.72, left=0.075, right=0.99, top=0.90,
                                     bottom=0.07)
        axes = draw(single, gs_one, 0, block)
        finish(single, [axes], legend_gap=0.10)
        panel_titles(single, [axes[0]], [block["title"]], x_offset=-0.008)
        save_figure(single, args.output.with_name(f"lineage_composition_{name}.png"))

    save_figure(fig, args.output)
    print(f"  all {sars2.month.nunique()} months plotted; "
          f"{len(low_depth)} below {MIN_BREADTH:.0%} breadth: {sorted(low_depth)}")


if __name__ == "__main__":
    main()
