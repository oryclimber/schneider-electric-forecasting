"""Single command-line entry point for the Schneider challenge workflows.

Usage (from the repository root):

    uv run --locked python scripts/cli.py audit
    uv run --locked python scripts/cli.py phase 3
    uv run --locked python scripts/cli.py notebook --phase 3
    uv run --locked python scripts/cli.py pack
    uv run --locked python scripts/cli.py offer
    uv run --locked python scripts/cli.py supplier
    uv run --locked python scripts/cli.py competitor
    uv run --locked python scripts/cli.py report --report all
    uv run --locked python scripts/cli.py deck

The implementation modules live in ``scripts/commands``. Only
``consolidate_data.py`` remains a separate, manual utility because it mutates
source files and must not be part of a normal build.

Typical fresh run:

    uv run --locked python scripts/cli.py pack
    uv run --locked python scripts/cli.py supplier
    uv run --locked python scripts/cli.py offer
    uv run --locked python scripts/cli.py deck

``offer`` without supplier outputs still writes the price-only offer matrix and
competitor comparison; ``competitor`` additionally needs ``supplier`` to have
run, because the risk view joins supplier terms.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Running ``python scripts/cli.py`` puts scripts/ on sys.path, so the commands
# package is importable without installing the scripts directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from commands import (  # noqa: E402
    audit_data,
    build_challenge_pack,
    build_reports,
    build_strategy_deck,
    competitive_risk_view,
    competitor_pricing,
    offer_matrix,
    run_notebook,
    run_phase3,
    run_phase4,
    run_phase5,
    supplier_material_map,
    supplier_offer_terms,
    supplier_product_risk,
    supplier_risk,
)


def _phase(args):
    """Run one research phase headlessly."""
    {3: run_phase3.main, 4: run_phase4.main, 5: run_phase5.main}[args.number]()


def _notebook(args):
    """Execute the source notebook for a phase (2 = EDA)."""
    run_notebook.main(
        phase3=args.phase == 3,
        phase4=args.phase == 4,
        phase5=args.phase == 5,
    )


def _supplier(_args):
    """Run the supplier scorecard, material map, product risk and offer terms."""
    supplier_risk.main()
    supplier_material_map.main()
    supplier_product_risk.main()
    supplier_offer_terms.main()


def _offer(_args):
    """Build the price-only offer matrix and competitor comparison."""
    offer_matrix.main()
    competitor_pricing.main()


def _competitor(_args):
    """Build the competitor view; requires the supplier terms from ``supplier``."""
    competitor_pricing.main()
    competitive_risk_view.main()


def _report(args):
    """Rebuild the EDA, Phase 4 and/or Phase 5 reports."""
    build_reports.main(["--report", args.report])


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for all workflows."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("audit", help="inventory and hash local sources (read-only)")
    p.set_defaults(func=lambda _args: audit_data.main())

    p = sub.add_parser("phase", help="run a research phase headlessly")
    p.add_argument("number", type=int, choices=(3, 4, 5))
    p.set_defaults(func=_phase)

    p = sub.add_parser("notebook", help="execute the source notebook for a phase")
    p.add_argument("--phase", type=int, choices=(2, 3, 4, 5), default=2)
    p.set_defaults(func=_notebook)

    p = sub.add_parser("pack", help="build the September 2026 challenge pack")
    p.set_defaults(func=lambda _args: build_challenge_pack.main())

    p = sub.add_parser(
        "offer", help="build the offer matrix and competitor price comparison"
    )
    p.set_defaults(func=_offer)

    p = sub.add_parser(
        "supplier", help="build supplier scorecard, map, product risk and terms"
    )
    p.set_defaults(func=_supplier)

    p = sub.add_parser(
        "competitor", help="build the competitor risk view (run supplier first)"
    )
    p.set_defaults(func=_competitor)

    p = sub.add_parser("report", help="rebuild generated MyST/Markdown reports")
    p.add_argument(
        "--report",
        choices=("all", "eda", "phase4", "phase5"),
        default="all",
        help="which report set to rebuild (default: all)",
    )
    p.set_defaults(func=_report)

    p = sub.add_parser("deck", help="rebuild the self-contained HTML strategy deck")
    p.set_defaults(func=lambda _args: build_strategy_deck.build())

    return parser


def main(argv: list[str] | None = None) -> None:
    """Dispatch one CLI command."""
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except FileNotFoundError as error:
        raise SystemExit(
            f"Missing required artifact: {error}. "
            "Run the earlier command shown in the help text first."
        ) from error


if __name__ == "__main__":
    main()
