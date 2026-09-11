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

from plot_style import (init_plotting_style, save_figure,  # noqa: E402
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
MISSING_COLOR = "#9b9a92"
# Hue is pinned to the group, not to its rank, so showing or hiding a group never repaints
# the others. NB.1.8.1 takes the eighth hue, which nothing else was using.
SARS2_COLORS = {
    "XBB*": "#2a78d6",
    "JN.1* (other)": "#eb6834",
    "KP.2*": "#1baf7a",
    "KP.3*": "#eda100",
    "XEC*": "#e87ba4",
    "LP.8.1*": "#008300",
    "NB.1.8.1*": "#e34948",
    "XFG*": "#4a3aa7",
}

UNREPORTED_COLOR = "#e8e7e2"
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
COVERAGE_POINT_COLOR = "#55544d"


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
        ax.set_title(title, loc="left", fontsize=TITLE_FONT, color="black", pad=16)
        ax.set_ylabel("Fraction", fontsize=LABEL_FONT)
        ax.set_xlim(*xlim)
        date_axis(ax)
    for ax in axes[:-1]:
        ax.set_xticklabels([])


def panel_titles(fig, axes, titles, x_offset=0.012, y_offset=0.064):
    """Pathogen name above its panel."""
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
    # A line on a log axis reads by position; a bar would encode length from an arbitrary
    # baseline, since a log scale has no zero.
    ax.plot(x, values, "-o", color=COVERAGE_COLOR, linewidth=2.8, markersize=7,
            markerfacecolor=COVERAGE_POINT_COLOR, markeredgecolor="white",
            markeredgewidth=1.0, zorder=3, clip_on=False)
    ax.set_yscale("log")
    ax.set_ylim(floor, np.nanmax(values) * 2.5)
    # Label a few decades across the range. A short axis defaults to a single tick, but
    # every decade would crowd it, so decades are strided down to at most four labels.
    top = np.nanmax(values)
    lo_exp = int(np.round(np.log10(floor)))
    hi_exp = int(np.floor(np.log10(top)))
    exps = list(range(lo_exp, hi_exp + 1))
    if len(exps) > 4:  # thin only when every decade would crowd the row
        # Take every nth decade rather than sampling evenly across the list: sampling gave
        # ticks 1, 2 and 1 decades apart, so the middle tick sat visibly off-centre.
        exps = exps[::int(np.ceil(len(exps) / 4))]
    ticks = [10 ** e for e in exps]
    ax.set_yticks(ticks)
    # Keep every tick mark, but on a thin row label only alternate ones so the text does
    # not overlap. Counting back from the top keeps the highest decade labelled.
    labelled = set(range(len(ticks) - 1, -1, -2)) if len(ticks) > 3 else set(range(len(ticks)))
    ax.set_yticklabels([compact_number(t) if i in labelled else ""
                        for i, t in enumerate(ticks)])
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(*xlim)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONT - 6)
    ax.tick_params(axis="both", labelsize=TICK_FONT - 4, labelcolor="black")
    ax.set_xticklabels([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# Gap between a composition row and the depth row beneath it, and between that depth
# row and the comparator composition row below it.
# Wide enough to seat the "Coverage (×)" title between the composition row and
# the depth bars; a tighter gap ran the title into the bars above.
COVERAGE_GAP = 0.040
COMPARATOR_GAP = 0.092


def tuck_under(fig, upper_ax, lower_ax, gap=COVERAGE_GAP):
    """Move a depth row up so it sits just beneath its composition row."""
    fig.canvas.draw()
    upper = upper_ax.get_position()
    lower = lower_ax.get_position()
    height = lower.height
    lower_ax.set_position([lower.x0, upper.y0 - gap - height, lower.width, height])


def mark_missing(ax, months, xlim):
    """Short hatched stubs where the comparator published nothing, so a gap is not read
    as a genuine zero."""
    if months is None or not len(months):
        return
    import matplotlib as mpl
    x = pd.to_datetime(pd.Index(months).astype(str) + "-01")
    with mpl.rc_context({"hatch.linewidth": 1.1}):
        ax.bar(x, 0.075, width=24, facecolor="white", edgecolor=MISSING_COLOR,
               hatch="///", linewidth=0.9, zorder=2)
    ax.text(x.min() + (x.max() - x.min()) / 2, 0.13, "No CaliciNet data available",
            ha="center", va="bottom", fontsize=TICK_FONT, color="black", zorder=4)
    ax.set_xlim(*xlim)


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
    parser.add_argument("--figures", type=Path, default=REPO_ROOT / "figures")
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
    # Other carries its own grey rather than a palette slot, so all eight hues are
    # available to named groups.
    named = set(ranked.head(len(SERIES_COLORS)).index)
    groups = [g for g in SARS2_ORDER if g in named] + ["Other"]
    sars2["category"] = np.where(sars2.group_label.isin(named), sars2.group_label, "Other")
    sars2 = sars2.groupby(["month", "category"], as_index=False)[
        ["casper_fraction", "cdc_fraction"]].sum(min_count=1)
    group_colors = {g: SARS2_COLORS.get(g, OTHER_COLOR) for g in groups}
    group_colors["Other"] = OTHER_COLOR

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
    cal_missing = sorted(set(noro.month.astype(str)) - set(cal.month.astype(str)))

    # Measured coverage depth behind each CASPER bar
    sars2_cov = pooled.sort_values("month")
    sars2_cov_depth = sars2_cov.mean_depth
    noro_cov = pd.read_csv(args.tables / "norovirus_coverage_monthly.csv")
    noro_cov = noro_cov.sort_values("month")

    sars2_frames = [
        sars2.rename(columns={"casper_fraction": "fraction"}).dropna(subset=["fraction"]),
        sars2.rename(columns={"cdc_fraction": "fraction"}).dropna(subset=["fraction"])]
    # Other sits in the last column rather than trailing the second row; the named groups
    # keep their sweep order around it.
    named_order = [g for g in groups if g != "Other"]
    legend_order = named_order[:4] + ["Other"] + named_order[4:]
    sars2_legend = [Patch(facecolor=group_colors[g], label=g) for g in legend_order]
    # The two remainder categories stack in the last column so the pair reads together;
    # they are named for their source, since they are not the same quantity.
    genotypes = [g for g in GENOTYPE_ORDER if g != "Other genotypes"]
    noro_legend = ([Patch(facecolor=genotype_colors[g], label=g) for g in genotypes[:3]]
                   + [Patch(facecolor=genotype_colors["Other genotypes"],
                            label="Other genotypes (CASPER)")]
                   + [Patch(facecolor=genotype_colors[g], label=g) for g in genotypes[3:]]
                   + [Patch(facecolor=UNREPORTED_COLOR,
                            label="Other genotypes (CaliciNet, not reported separately)")])

    blocks = {
        "sars2": dict(title="SARS-CoV-2", frames=sars2_frames, order=groups,
                      colors=group_colors, legend=sars2_legend, ncol=5,
                      rows=["CASPER wastewater sequencing",
                            "CDC clinical genomic surveillance"],
                      cov_months=sars2_cov.month, cov_values=sars2_cov_depth),
        "norovirus": dict(title="Norovirus", frames=[casper_noro, cal],
                          order=GENOTYPE_ORDER + [CALICINET_OTHER],
                          colors=genotype_colors, legend=noro_legend, ncol=4,
                          rows=["CASPER wastewater sequencing",
                                "CaliciNet clinical outbreak surveillance"],
                          cov_months=noro_cov.month, cov_values=noro_cov.mean_depth,
                          missing=cal_missing),
    }

    def draw(fig, gs, offset, block):
        ax1 = fig.add_subplot(gs[offset])
        axcov = fig.add_subplot(gs[offset + 1])
        ax2 = fig.add_subplot(gs[offset + 2])
        axleg = fig.add_subplot(gs[offset + 3])
        comparison_rows([ax1, ax2], block["frames"], block["order"], block["colors"],
                        block["rows"], xlim)
        ax1.set_xticklabels([])
        mark_missing(ax2, block.get("missing"), xlim)
        # The depth row is thin, so its scale is named in a small title above it rather
        # than a rotated y label, which crowded the axis at this height.
        coverage_row(axcov, block["cov_months"], block["cov_values"], xlim, "")
        axcov.set_title("Coverage (×)", loc="left", fontsize=TICK_FONT - 2,
                        color="black", pad=2)
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

    # One figure per pathogen; no panel letters, so each stands alone.
    for name, block in blocks.items():
        single = plt.figure(figsize=(17, 10))
        gs_one = single.add_gridspec(4, 1, height_ratios=[1.0, 0.34, 1.0, 0.58],
                                     hspace=0.72, left=0.075, right=0.99, top=0.90,
                                     bottom=0.07)
        axes = draw(single, gs_one, 0, block)
        finish(single, [axes], legend_gap=0.125)
        panel_titles(single, [axes[0]], [block["title"]], x_offset=-0.008)
        save_figure(single, args.figures / f"lineage_composition_{name}.png")

    print(f"  all {sars2.month.nunique()} months plotted; "
          f"{len(low_depth)} below {MIN_BREADTH:.0%} breadth: {sorted(low_depth)}")


if __name__ == "__main__":
    main()
