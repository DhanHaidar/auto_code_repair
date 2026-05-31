from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repair_metrics import RepairMetricsEvaluator
_EQUIV_MODE = "hybrid"


def _load_dataset(dataset_path: Path) -> List[dict]:
    with dataset_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise SystemExit("Dataset file must contain a JSON list.")
    return payload


def _collect_candidates(
    dataset: List[dict],
    candidate_dir: Path,
    default_package_name: Optional[str],
) -> List[Tuple[str, str, str, str, Optional[str]]]:
    collected: List[Tuple[str, str, str, str, Optional[str]]] = []
    for entry in dataset:
        metadata = entry.get("metadata", {}) if isinstance(entry, dict) else {}
        program = metadata.get("program") or entry.get("program")
        if not program:
            continue

        candidate_path = candidate_dir / f"{program}.py"
        if not candidate_path.exists():
            continue

        test_file = metadata.get("test_file")
        if not test_file:
            continue

        package_name = metadata.get("package_name", default_package_name)
        if package_name == "":
            package_name = None

        collected.append(
            (
                program,
                candidate_path.read_text(encoding="utf-8"),
                entry["correct_code"],
                test_file,
                package_name,
            )
        )
    return collected


def _evaluate_case(
    program: str,
    candidate_code: str,
    reference_code: str,
    test_file: str,
    package_name: Optional[str],
) -> dict:
    evaluator = RepairMetricsEvaluator(
        tests_path=str(Path(test_file).resolve()),
        module_name=program,
        package_name=package_name,
        equivalence_mode=_EQUIV_MODE,
    )
    record = evaluator.evaluate(candidate_code, reference_code, name=program)
    return record.to_dict()


def _evaluate_cases_parallel(
    cases: List[Tuple[str, str, str, str, Optional[str]]],
    workers: Optional[int],
) -> List[dict]:
    if not cases:
        return []

    worker_count = workers if workers and workers > 0 else min(32, os.cpu_count() or 1, len(cases))
    if worker_count <= 1:
        return [
            _evaluate_case(program, candidate_code, reference_code, test_file, package_name)
            for program, candidate_code, reference_code, test_file, package_name in cases
        ]

    ordered_results: List[Optional[dict]] = [None] * len(cases)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_index = {
            executor.submit(
                _evaluate_case,
                program,
                candidate_code,
                reference_code,
                test_file,
                package_name,
            ): index
            for index, (program, candidate_code, reference_code, test_file, package_name) in enumerate(cases)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            ordered_results[index] = future.result()

    return [record for record in ordered_results if record is not None]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score plausible fix and correct fix for QuixBugs Python repairs",
    )
    parser.add_argument("--dataset", required=True, help="APE-style dataset JSON")
    parser.add_argument(
        "--candidate-dir",
        required=True,
        help="Directory containing candidate patches named after each program",
    )
    parser.add_argument(
        "--package-name",
        default="python_programs",
        help="Python package name for candidate modules; use an empty string for package-less modules",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to write a JSON summary",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel test workers; 0 picks an automatic value",
    )
    parser.add_argument(
        "--equivalence-mode",
        choices=["fast", "ast", "ast_canonical", "behavioral", "hybrid"],
        default="hybrid",
        help="Equivalence checking mode (default: hybrid)",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    candidate_dir = Path(args.candidate_dir).resolve()
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")
    if not candidate_dir.exists():
        raise SystemExit(f"Candidate directory not found: {candidate_dir}")

    dataset = _load_dataset(dataset_path)
    package_name = args.package_name
    if package_name == "":
        package_name = None
    global _EQUIV_MODE
    _EQUIV_MODE = args.equivalence_mode
    cases = _collect_candidates(dataset, candidate_dir, package_name)
    if not cases:
        raise SystemExit("No matching candidates were found.")

    records = _evaluate_cases_parallel(cases, args.workers)
    plausible_fix_count = sum(1 for record in records if record["plausible_fix"])
    correct_fix_count = sum(1 for record in records if record["correct_fix"])

    total = len(records)
    summary = {
        "total": total,
        "plausible_fix_count": plausible_fix_count,
        "correct_fix_count": correct_fix_count,
        "plausible_fix_rate": plausible_fix_count / total if total else 0.0,
        "correct_fix_rate": correct_fix_count / total if total else 0.0,
        "records": records,
    }

    print(f"total: {total}")
    print(f"plausible_fix: {plausible_fix_count}")
    print(f"correct_fix: {correct_fix_count}")
    print(f"plausible_fix_rate: {summary['plausible_fix_rate']:.2%}")
    print(f"correct_fix_rate: {summary['correct_fix_rate']:.2%}")

    if args.output.strip():
        output_path = Path(args.output).resolve()
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")


if __name__ == "__main__":
    main()