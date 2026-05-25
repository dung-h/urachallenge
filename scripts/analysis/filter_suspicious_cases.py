from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.disagreement_utils import collect_suspicious_cases, iter_json_records


def _default_trace_paths() -> list[Path]:
    trace_root = ROOT / "outputs" / "traces"
    if trace_root.exists():
        return [trace_root]
    return []


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter suspicious solver/LLM trace cases for review.")
    parser.add_argument("paths", nargs="*", type=Path, help="Trace JSON/JSONL files or directories. Defaults to outputs/traces.")
    parser.add_argument("--min-score", type=int, default=20, help="Minimum suspicion score to include.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of cases to emit.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "disagreements" / "suspicious_cases.jsonl", help="Output JSONL path.")
    parser.add_argument("--print-summary", action="store_true", help="Print a short summary to stdout.")
    args = parser.parse_args()

    paths = args.paths or _default_trace_paths()
    cases = collect_suspicious_cases(paths, min_score=args.min_score)[: args.limit]
    records = [case.to_record() for case in cases]
    _write_jsonl(args.output, records)

    if args.print_summary:
        print(f"wrote {len(records)} suspicious case(s) to {args.output}")
        for case in cases:
            print(f"{case.request_id} score={case.score} task={case.task_type} answer={case.answer} reasons={','.join(case.reasons)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
