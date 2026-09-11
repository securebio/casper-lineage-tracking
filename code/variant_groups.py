"""Collapse SARS-CoV-2 variant names onto mutually exclusive comparison groups.

Variants are named as Pango lineages, so the lookups below work on lineage names
and the hierarchy shipped with Freyja."""

from pathlib import Path

import yaml

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_LINEAGES_YML = DATA_DIR / "pango_lineages.yml"

# Most specific first; a lineage joins the first group containing it.
COMPARISON_GROUPS = [
    "XFG",
    "NB.1.8.1",
    "LP.8.1",
    "XEC",
    "KP.3",
    "KP.2",
    "JN.1",
    "BA.2.86",
    "XBB",
]

DISPLAY_NAMES = {
    "XFG": "XFG*",
    "NB.1.8.1": "NB.1.8.1*",
    "LP.8.1": "LP.8.1*",
    "XEC": "XEC*",
    "KP.3": "KP.3*",
    "KP.2": "KP.2*",
    "JN.1": "JN.1* (other)",
    "BA.2.86": "BA.2.86* (excl. JN.1)",
    "XBB": "XBB*",
    "Other": "Other",
}


def load_membership(lineages_yml=DEFAULT_LINEAGES_YML, groups=None):
    groups = groups or COMPARISON_GROUPS
    records = yaml.safe_load(Path(lineages_yml).read_text())
    by_name = {r["name"]: set(r.get("children") or []) for r in records}
    return {g: set(by_name.get(g, set())) | {g} for g in groups}


def assign_group(lineage, membership, groups=None):
    groups = groups or COMPARISON_GROUPS
    if not lineage or not isinstance(lineage, str):
        return "Other"
    name = lineage.strip().rstrip("*")
    for group in groups:
        if name in membership[group]:
            return group
    return "Other"


def collapse(frame, lineage_col, weight_col, by, membership=None, groups=None):
    import pandas as pd

    membership = membership or load_membership(groups=groups)
    out = frame.copy()
    out["group"] = out[lineage_col].map(lambda x: assign_group(x, membership, groups))
    grouped = out.groupby(by + ["group"], as_index=False)[weight_col].sum()
    totals = (grouped.groupby(by, as_index=False)[weight_col].sum()
              .rename(columns={weight_col: "_total"}))
    grouped = grouped.merge(totals, on=by)
    grouped["fraction"] = grouped[weight_col] / grouped["_total"].replace(0, pd.NA)
    return grouped.drop(columns="_total")
