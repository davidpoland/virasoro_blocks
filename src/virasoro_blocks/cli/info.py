"""Implementation of ``virasoro_blocks info``."""

from __future__ import annotations

import argparse
import json
import sys

from ..requests import RequestValidationError, capabilities_for


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="virasoro_blocks info",
        allow_abbrev=False,   # a prefix like --delta silently meant --delta-ext
        description="Show block-family capabilities supported by the CLI.",
    )
    p.add_argument("--family", default=None, help="show one family (default all)")
    p.add_argument("--format", choices=["json", "text"], default="text", help="text (default) or json")
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        records = [record.as_dict() for record in capabilities_for(args.family)]
    except RequestValidationError as exc:
        parser.error(str(exc))
    if args.format == "json":
        text = json.dumps(records if len(records) != 1 else records[0], indent=2)
    else:
        lines = []
        for record in records:
            lines.append(f"{record['family']}:")
            lines.append(f"  operations: {', '.join(record['operations'])}")
            lines.append(f"  surfaces: {', '.join(record['surfaces'])}")
            lines.append(f"  bases: {', '.join(record['bases'])}")
            lines.append(f"  coordinates: {', '.join(record['coordinates'])}")
            lines.append(
                "  direct evaluation coordinates: "
                f"{', '.join(record['evaluate_coordinates'])}"
            )
            lines.append(
                "  point-table coordinates: "
                f"{', '.join(record['point_table_coordinates'])}"
            )
            lines.append(f"  output formats: {', '.join(record['output_formats'])}")
            lines.append(
                "  controls: "
                f"order={record['supports_order']}, "
                f"lambda={record['supports_lambda']}, "
                f"npoles={record['supports_npoles']}"
            )
        text = "\n".join(lines)
    sys.stdout.write(text + "\n")
    return 0
