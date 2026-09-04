"""Build monthly composition tables and agreement statistics.

Writes tables/ from data/. Run before plot.py.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from lineage_groups import DISPLAY_NAMES, collapse, load_membership  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "tables"

FOCAL_GENOTYPES = ["GII.4", "GII.17", "GII.2", "GII.3", "GII.6"]
VP1_BP = 1626
N_BOOTSTRAP = 2000


def norovirus_monthly():
    counts = pd.read_csv(DATA / "norovirus_genotype_counts.tsv", sep="\t")
    meta = pd.read_csv(DATA / "sample_metadata.csv")
    counts = counts.merge(meta[["sample", "date"]], on="sample")
    counts["month"] = pd.to_datetime(counts.date).dt.to_period("M").astype(str)

    assigned = counts[(counts.level == "genotype") & (counts.group != "ambiguous")].copy()
    assigned["genotype"] = np.select(
        [assigned.group.isin(FOCAL_GENOTYPES), assigned.group.str.startswith("GI.")],
        [assigned.group, "GI (all)"],
        default="Other genotypes",
    )
    monthly = assigned.groupby(["month", "genotype"], as_index=False).n_pairs.sum()
    totals = (monthly.groupby("month", as_index=False).n_pairs.sum()
              .rename(columns={"n_pairs": "total"}))
    monthly = monthly.merge(totals, on="month")
    monthly["casper_fraction"] = monthly.n_pairs / monthly.total

    cal = pd.read_csv(DATA / "calicinet_monthly.tsv", sep="\t", comment="#")
    cal["other_pct"] = 100 - cal.gii4_pct - cal.gii17_pct
    cal = cal.melt(id_vars="month", value_vars=["gii4_pct", "gii17_pct", "other_pct"],
                   var_name="genotype", value_name="pct")
    cal["genotype"] = cal.genotype.map({"gii4_pct": "GII.4", "gii17_pct": "GII.17",
                                        "other_pct": "Other genotypes"})
    cal["calicinet_fraction"] = cal.pct / 100.0

    out = monthly.merge(cal[["month", "genotype", "calicinet_fraction"]],
                        on=["month", "genotype"], how="left")
    return out.rename(columns={"n_pairs": "casper_read_pairs",
                               "total": "casper_read_pairs_total"})


def norovirus_coverage():
    cov = pd.read_csv(DATA / "norovirus_coverage.tsv", sep="\t")
    meta = pd.read_csv(DATA / "sample_metadata.csv")
    cov = cov.merge(meta[["sample", "date"]], on="sample")
    cov["month"] = pd.to_datetime(cov.date).dt.to_period("M").astype(str)
    monthly = cov.groupby("month", as_index=False).agg(
        aligned_bases=("aligned_bases", "sum"), aligned_reads=("aligned_reads", "sum"),
        n_samples=("sample", "nunique"))
    monthly["mean_depth"] = monthly.aligned_bases / VP1_BP
    return monthly


def sars2_monthly():
    demix = pd.read_csv(DATA / "sars2_demix_national.tsv", sep="\t")
    pooled = (demix[["pool", "n_libraries", "n_sites", "n_pairs", "mean_depth",
                     "breadth_1x", "breadth_10x", "resid"]]
              .drop_duplicates("pool").rename(columns={"pool": "month",
                                                       "n_libraries": "n_samples"}))
    membership = load_membership()
    casper = collapse(demix[demix.level == "lineage"], "lineage", "fraction", ["pool"],
                      membership=membership).rename(columns={"pool": "month",
                                                             "fraction": "casper_fraction"})

    cdc = pd.read_csv(DATA / "cdc_variant_proportions.csv")
    cdc = cdc[cdc.usa_or_hhsregion == "USA"].copy()
    cdc["month"] = pd.to_datetime(cdc.week_ending).dt.to_period("M").astype(str)
    cdc = (cdc.groupby(["month", "week_ending", "variant"], as_index=False).share.first()
           .groupby(["month", "variant"], as_index=False).share.mean())
    cdc = collapse(cdc, "variant", "share", ["month"], membership=membership)
    cdc = cdc.rename(columns={"fraction": "cdc_fraction"})[["month", "group", "cdc_fraction"]]

    out = casper[["month", "group", "casper_fraction"]].merge(cdc, on=["month", "group"],
                                                              how="outer")
    out = out[out.month.isin(casper.month.unique())]
    out = out.merge(pooled, on="month", how="left")
    out["group_label"] = out.group.map(DISPLAY_NAMES).fillna(out.group)
    return out.sort_values(["month", "group"]), pooled


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return np.nan
    return spearmanr(x[ok], y[ok]).statistic


def _bootstrap_ci(frame, a, b, seed=0):
    """Percentile CI for Spearman R, resampling whole months.

    Months rather than group-months: fractions within a month are compositional and sum to
    one, so group-months are not independent observations.
    """
    rng = np.random.default_rng(seed)
    months = frame.month.unique()
    if len(months) < 4:
        return (np.nan, np.nan)
    stats = []
    for _ in range(N_BOOTSTRAP):
        pick = rng.choice(months, size=len(months), replace=True)
        sample = pd.concat([frame[frame.month == m] for m in pick])
        r = _spearman(sample[a], sample[b])
        if np.isfinite(r):
            stats.append(r)
    return tuple(np.percentile(stats, [2.5, 97.5])) if len(stats) >= 20 else (np.nan, np.nan)


def _total_variation(frame, cat, a, b):
    """Per-month total variation distance: 0 identical, 1 disjoint."""
    rows = []
    for month, g in frame.groupby("month"):
        x = g[a].fillna(0).to_numpy(float)
        y = g[b].fillna(0).to_numpy(float)
        if x.sum() <= 0 or y.sum() <= 0:
            continue
        x, y = x / x.sum(), y / y.sum()
        rows.append({"month": month, "total_variation_distance": 0.5 * np.abs(x - y).sum()})
    return pd.DataFrame(rows)


def statistics(noro, sars2):
    rows = []
    # Correlations run per reported group; the composition distance runs over the whole
    # composition, which for norovirus includes the other-genotype category CaliciNet
    # reports as a remainder. Restricting the distance to GII.4 and GII.17 would
    # renormalise the two against each other and measure something else.
    noro_corr = noro[noro.genotype.isin(["GII.4", "GII.17"])]
    # A CASPER month with no CaliciNet counterpart is outside the published series, not a
    # zero, so norovirus months are kept on the CaliciNet column alone; the SARS-CoV-2
    # sources span the same months and are matched on both.
    for label, corr_frame, dist_frame, cat, a, b in [
        ("SARS-CoV-2: CASPER vs CDC clinical", sars2,
         sars2.dropna(subset=["casper_fraction", "cdc_fraction"]), "group_label",
         "casper_fraction", "cdc_fraction"),
        ("Norovirus: CASPER vs CaliciNet", noro_corr,
         noro.dropna(subset=["calicinet_fraction"]), "genotype",
         "casper_fraction", "calicinet_fraction"),
    ]:
        sub = corr_frame.dropna(subset=[a, b])
        for group, g in sub.groupby(cat):
            if len(g) < 4:
                continue
            lo, hi = _bootstrap_ci(g, a, b)
            rows.append({"comparison": label, "group": group, "n_months": len(g),
                         "spearman_r": _spearman(g[a], g[b]), "ci_low": lo, "ci_high": hi})
        tv = _total_variation(dist_frame, cat, a, b)
        if not tv.empty:
            rows.append({"comparison": label, "group": "ALL (composition distance)",
                         "n_months": len(tv),
                         "median_total_variation_distance": tv.total_variation_distance.median(),
                         "iqr_low": tv.total_variation_distance.quantile(0.25),
                         "iqr_high": tv.total_variation_distance.quantile(0.75)})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    noro = norovirus_monthly()
    noro.to_csv(OUT / "norovirus_monthly.csv", index=False)
    print(f"norovirus_monthly.csv       {len(noro):5d} rows, {noro.month.nunique()} months")

    cov = norovirus_coverage()
    cov.to_csv(OUT / "norovirus_coverage_monthly.csv", index=False)
    print(f"norovirus_coverage_monthly  {len(cov):5d} rows")

    sars2, pooled = sars2_monthly()
    sars2.to_csv(OUT / "sars2_monthly.csv", index=False)
    pooled.to_csv(OUT / "sars2_pool_coverage.csv", index=False)
    print(f"sars2_monthly.csv           {len(sars2):5d} rows, {sars2.month.nunique()} months")

    stats = statistics(noro, sars2)
    stats.to_csv(OUT / "comparison_statistics.tsv", sep="\t", index=False)
    show = stats.dropna(subset=["spearman_r"])
    print("\nSpearman R (one observation per month, 95% CI by resampling months):")
    for r in show.itertuples():
        ci = f"[{r.ci_low:.2f}, {r.ci_high:.2f}]" if np.isfinite(r.ci_low) else "n/a"
        print(f"  {r.comparison:36s} {r.group:22s} n={r.n_months:2d}  "
              f"{r.spearman_r:5.2f}  {ci}")
    print("\nComposition distance (0 = identical):")
    for r in stats[stats.group == "ALL (composition distance)"].itertuples():
        print(f"  {r.comparison:36s} median {r.median_total_variation_distance:.2f} "
              f"(IQR {r.iqr_low:.2f}-{r.iqr_high:.2f}) over {r.n_months} months")


if __name__ == "__main__":
    main()
