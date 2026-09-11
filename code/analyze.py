"""Build the monthly composition tables behind the figure.

Writes tables/ from data/. Run before plot.py.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from lineage_groups import DISPLAY_NAMES, collapse, load_membership  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "tables"

FOCAL_GENOTYPES = ["GII.4", "GII.17", "GII.2", "GII.3", "GII.6"]
VP1_BP = 1626


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
    # CDC windows are 4 weeks labelled by their end date; the midpoint centres them.
    cdc["month"] = (pd.to_datetime(cdc.week_ending)
                    - pd.Timedelta(days=14)).dt.to_period("M").astype(str)
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



if __name__ == "__main__":
    main()
