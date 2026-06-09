from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.disagreement_utils import render_pytest_skeleton
from scripts.analysis.disagreement_utils import render_generation_report


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate pytest skeletons from suspicious disagreement cases.")
    parser.add_argument("input", type=Path, help="Suspicious case JSONL or a single trace JSON file.")
    parser.add_argument("--source-root", type=Path, default=ROOT / "datasets" / "eval", help="Optional dataset root used to enrich logic questions with full premises.")
    parser.add_argument("--output", type=Path, default=ROOT / "tests" / "generated" / "test_disagreement_regressions.py", help="Output pytest file.")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "disagreement_generation_report.md", help="Output markdown report summarizing accepted and skipped cases.")
    parser.add_argument("--stdout", action="store_true", help="Print the generated test module instead of writing it.")
    args = parser.parse_args()

    if args.input.suffix.lower() == ".jsonl":
        records = _read_jsonl(args.input)
    else:
        records = [json.loads(args.input.read_text(encoding="utf-8"))]
    rendered = render_pytest_skeleton(records, source_root=args.source_root)
    report = render_generation_report(records, source_root=args.source_root)
    if args.stdout:
        print(rendered, end="")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(f"wrote {len(records)} test skeleton(s) to {args.output}")
    print(f"wrote generation report to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
