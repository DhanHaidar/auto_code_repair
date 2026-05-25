from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluator.tester import run_tests_with_status


def discover_programs(quixbugs_root: Path) -> List[Tuple[str, Path, Path, Path]]:
    """Discover programs that have matching tests and correct versions."""
    tests_dir = quixbugs_root / "python_testcases"
    buggy_dir = quixbugs_root / "python_programs"
    correct_dir = quixbugs_root / "correct_python_programs"

    programs: List[Tuple[str, Path, Path, Path]] = []
    for test_file in sorted(tests_dir.glob("test_*.py")):
        name = test_file.stem.replace("test_", "", 1)
        buggy_path = buggy_dir / f"{name}.py"
        correct_path = correct_dir / f"{name}.py"
        if not buggy_path.exists() or not correct_path.exists():
            continue
        programs.append((name, test_file, buggy_path, correct_path))
    return programs


def build_dataset(
    quixbugs_root: Path,
    limit: int = 0,
    max_output_chars: int = 4000,
) -> List[Dict[str, object]]:
    """Build dataset entries for all discovered programs."""
    entries: List[Dict[str, object]] = []
    programs = discover_programs(quixbugs_root)
    if limit > 0:
        programs = programs[:limit]

    for name, test_file, buggy_path, correct_path in programs:
        buggy_code = buggy_path.read_text(encoding="utf-8")
        correct_code = correct_path.read_text(encoding="utf-8")

        reward, status, output = run_tests_with_status(
            buggy_code,
            str(test_file),
            module_name=name,
            package_name="python_programs",
        )

        error_message = output.strip() or f"Status: {status}"
        if max_output_chars and len(error_message) > max_output_chars:
            error_message = error_message[:max_output_chars] + "\n...<truncated>"

        entries.append(
            {
                "buggy_code": buggy_code,
                "error_message": error_message,
                "correct_code": correct_code,
                "metadata": {
                    "program": name,
                    "test_file": str(test_file),
                    "status": status,
                    "reward": reward,
                },
            }
        )

    return entries


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for dataset generation."""
    parser = argparse.ArgumentParser(description="Build QuixBugs APE dataset")
    parser.add_argument(
        "--quixbugs-root",
        default="QuixBugs",
        help="Path to QuixBugs root directory",
    )
    parser.add_argument(
        "--output",
        default="ape_quixbugs_python_dataset.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of programs (0 = all)",
    )
    parser.add_argument(
        "--max-output-chars",
        type=int,
        default=4000,
        help="Max length of error_message output",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    quixbugs_root = Path(args.quixbugs_root).resolve()
    if not quixbugs_root.exists():
        raise SystemExit(f"QuixBugs root not found: {quixbugs_root}")

    entries = build_dataset(
        quixbugs_root=quixbugs_root,
        limit=args.limit,
        max_output_chars=args.max_output_chars,
    )

    output_path = Path(args.output).resolve()
    output_path.write_text(
        json.dumps(entries, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(f"Wrote {len(entries)} entries to {output_path}")


if __name__ == "__main__":
    main()
