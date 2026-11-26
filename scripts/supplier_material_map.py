"""Supplier -> material map, with the BOM columns computed from the authoritative BOM.

Single source of truth for the map. Two kinds of content:
- INPUT (typed below): what each supplier does, which priced materials it is
  linked to, and how confident the link is. The BOM has no supplier names, so
  this link cannot be derived from data; it comes from public web descriptions
  and user-supplied company descriptions.
- DERIVED (never typed): whether each linked material is in the RM6 / AirSeT BOM
  and its share of BOM mass. These come from ``_bom_quantities()`` (the
  authoritative BOM loaded by ``ingestion.load_authoritative_bom``).

``supplier_product_risk.py`` imports ``MATERIAL_SUPPLIERS`` from here, so the
risk calculation and the CSV cannot disagree about who supplies what.

Usage:
    uv run --locked python scripts/supplier_material_map.py
"""

from __future__ import annotations

import pandas as pd
from build_challenge_pack import PRODUCTS, _bom_quantities

from schneider_electric_forecasting import config

# ASSUMPTION: the three sheet-metal fabricators are applied to every steel grade
# (carbon, stainless, galvanised) because none states which grade it handles.
STEEL = ["carbon_steel", "stainless_steel", "galvanised_steel"]
CAUTION_STEEL = "medium (processor; grade unknown)"
USER = "user-supplied company descriptions"

# ASSUMPTION: KUVAG -> epoxy is inferred from its product line, not from Schneider.
# Order matters: it fixes the supplier order in downstream tables.
SUPPLIERS: dict[str, dict] = {
    "Gindre Duchavany": {
        "what": "Semi-finished copper products; copper metallurgy (NAF 2444Z); Lyon HQ / Pont-de-Cheruy plant",
        "materials": ["copper"],
        "confidence": "high",
        "evidence": "https://www.xerfi.com/etudes-par-entreprise/ets-gindre-duchavany_955508817",
    },
    "KUVAG": {
        "what": "Electrical insulation: epoxy-resin insulators (KUVATEC) composite insulators and SMC/BMC; Austria",
        "materials": ["epoxy"],
        "confidence": "high on activity; supplier-to-Schneider link unconfirmed",
        "evidence": "https://railmarket.com/eu/profile/kuvag-gmbh-co-kg",
    },
    "Green Transfo": {
        "what": "Designs and makes power/distribution transformers (Cahors); wholesale of electrical equipment (registry)",
        "materials": [],
        "reason": "none (finished electrical equipment)",
        "confidence": "medium",
        "evidence": "https://www.societe.com/societe/green-transfo-910392844.html",
    },
    "MSS International (Manchester)": {
        "what": "Manufactures copper components (stamping, pressing, bending, forging) from in-house raw material stock; assembles them in-house; 27,000 m2 site (user-supplied)",
        "materials": ["copper"],
        "confidence": "medium-high (copper components; other metals possible)",
        "evidence": "user-supplied company description",
    },
    "Maintenance Travaux et Cablage": {
        "what": "Electronic components / electrical installation and engineering services (NAF codes conflict)",
        "materials": [],
        "reason": "none (services)",
        "confidence": "low",
        "evidence": "https://www.xerfi.com/etudes-par-entreprise/maintenance-travaux-et-cablage_340564798",
    },
    "Groupe Aim": {
        "what": "Flat-metal and tube transformation: cutting through assembly, incl. electrical component integration (user-supplied)",
        "materials": STEEL,
        "confidence": CAUTION_STEEL,
        "evidence": USER,
    },
    "Electris": {
        "what": "Custom copper and aluminium components for power systems; switchgear assembly (user-supplied)",
        "materials": ["copper", "aluminium"],
        "confidence": "medium-high",
        "evidence": USER,
    },
    "Iltom": {
        "what": "Sheet-metal processing (blanking, bending, welding), powder coating, stamping, machining (user-supplied)",
        "materials": STEEL,
        "confidence": CAUTION_STEEL,
        "evidence": USER,
    },
    "Rotzer Maschinenbau": {
        "what": "Sheet-metal processing (user-supplied)",
        "materials": STEEL,
        "confidence": CAUTION_STEEL,
        "evidence": USER,
    },
}

# material -> suppliers, in SUPPLIERS order (used by supplier_product_risk.py)
MATERIAL_SUPPLIERS: dict[str, list[str]] = {}
for _name, _s in SUPPLIERS.items():
    for _m in _s["materials"]:
        MATERIAL_SUPPLIERS.setdefault(_m, []).append(_name)


def _mass_share(bom: dict, product: str, material: str) -> float:
    """Percent of the product's total BOM mass that is this priced material."""
    qty = bom[product]["quantities"].get(material, 0.0)
    return 100 * qty / bom[product]["total_mass_kg"]


def _bom_text(bom: dict, product: str, materials: list[str]) -> str:
    parts = [(m, _mass_share(bom, product, m)) for m in materials]
    parts = [(m, p) for m, p in parts if p > 0]
    if not parts:
        return "no"
    if set(materials) == set(STEEL):
        return f"yes (steel lines, {sum(p for _, p in parts):.1f}% of mass)"
    if len(parts) == 1:
        return f"yes ({parts[0][1]:.1f}% of mass)"
    return "yes (" + ", ".join(f"{m} {p:.1f}%" for m, p in parts) + " of mass)"


def build() -> pd.DataFrame:
    bom = _bom_quantities()
    rows = []
    for name, s in SUPPLIERS.items():
        mats = s["materials"]
        row = {
            "supplier": name,
            "what_it_does_per_public_sources": s["what"],
            "mapped_material": "; ".join(mats) if mats else s["reason"],
            "confidence": s["confidence"],
            "status": "mapped" if mats else "unmapped",
            "evidence": s["evidence"],
        }
        for product in PRODUCTS:
            col = "in_rm6_bom" if product.startswith("RM6") else "in_airset_bom"
            row[col] = _bom_text(bom, product, mats) if mats else ""
        rows.append(row)
    cols = [
        "supplier",
        "what_it_does_per_public_sources",
        "mapped_material",
        "in_rm6_bom",
        "in_airset_bom",
        "confidence",
        "status",
        "evidence",
    ]
    return pd.DataFrame(rows)[cols]


def main() -> None:
    out = config.DATA_DIR / "submission"
    out.mkdir(parents=True, exist_ok=True)
    table = build()
    table.to_csv(out / "supplier_material_map.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print(
            table.drop(
                columns=["what_it_does_per_public_sources", "evidence"]
            ).to_string(index=False)
        )
    bom = _bom_quantities()
    in_bom = {k for p in PRODUCTS for k in bom[p]["quantities"]}
    missing = [m for m in MATERIAL_SUPPLIERS if m not in in_bom]
    if missing:
        print(f"WARNING: mapped materials absent from every BOM: {missing}")


if __name__ == "__main__":
    main()
