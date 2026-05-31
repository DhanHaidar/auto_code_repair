from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ape_prompt_optimizer import BugExample, ExampleSelector
from core.mcts import MCTS
from core.node import Node
from evaluator.tester import Tester


@dataclass
class BatchRepairCase:
    """A single repair task extracted from a dataset entry."""

    index: int
    program: str
    buggy_code: str
    error_message: str
    test_file: str
    package_name: Optional[str]


@dataclass
class BatchRepairRecord:
    """Result of one generated candidate patch."""

    index: int
    program: str
    candidate_path: str
    test_file: str
    package_name: Optional[str]
    status: str
    error_message: str
    generated_code: str
    best_reward: float
    iterations: int
    pass_count: int
    elapsed_sec: float
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BatchRepairSummary:
    """Aggregate output of a batch repair run."""

    total: int
    output_dir: str
    manifest_path: str
    generated_count: int
    error_count: int
    records: List[BatchRepairRecord]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "output_dir": self.output_dir,
            "manifest_path": self.manifest_path,
            "generated_count": self.generated_count,
            "error_count": self.error_count,
            "records": [record.to_dict() for record in self.records],
        }


class BatchRepairRunner:
    """Generate repair candidates for a whole QuixBugs dataset in one run."""

    def __init__(
        self,
        generator: Any,
        instruction_override: Optional[str] = None,
        default_package_name: Optional[str] = "python_programs",
        ape_examples: Optional[Sequence[BugExample]] = None,
        ape_few_shot_k: int = 0,
        mcts_iterations: int = 10,
        mcts_parallelism: int = 1,
        stop_on_pass: bool = True,
        max_refine_attempts: int = 1,
        workers: int = 0,
    ) -> None:
        self.generator = generator
        self.instruction_override = instruction_override.strip() if instruction_override else None
        self.default_package_name = default_package_name
        self.ape_examples = list(ape_examples or [])
        self.ape_few_shot_k = max(0, int(ape_few_shot_k))
        self.mcts_iterations = max(1, int(mcts_iterations))
        self.mcts_parallelism = max(1, int(mcts_parallelism))
        self.stop_on_pass = bool(stop_on_pass)
        self.max_refine_attempts = max(0, int(max_refine_attempts))
        self.workers = workers

    def run_from_json(
        self,
        dataset_path: str,
        output_dir: str,
    ) -> BatchRepairSummary:
        with open(dataset_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise SystemExit("Dataset file must contain a JSON list.")
        return self.run_entries(payload, output_dir)

    def run_entries(
        self,
        entries: Sequence[dict],
        output_dir: str,
    ) -> BatchRepairSummary:
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        cases = self._collect_cases(entries)
        records = self._repair_cases(cases, output_path)
        records.sort(key=lambda record: record.index)

        generated_count = len(records)
        error_count = sum(1 for record in records if record.status == "ERROR")
        manifest = BatchRepairSummary(
            total=len(cases),
            output_dir=str(output_path),
            manifest_path=str(output_path / "batch_results.json"),
            generated_count=generated_count,
            error_count=error_count,
            records=records,
        )

        manifest_path = output_path / "batch_results.json"
        manifest_path.write_text(
            json.dumps(manifest.to_dict(), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        return manifest

    def _collect_cases(self, entries: Sequence[dict]) -> List[BatchRepairCase]:
        cases: List[BatchRepairCase] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue

            metadata = entry.get("metadata", {}) if isinstance(entry.get("metadata", {}), dict) else {}
            program = metadata.get("program") or entry.get("program")
            buggy_code = entry.get("buggy_code", "")
            test_file = metadata.get("test_file", "")
            if not program or not buggy_code or not test_file:
                continue

            error_message = entry.get("error_message") or metadata.get("error_message") or ""
            package_name = metadata.get("package_name", self.default_package_name)
            if package_name == "":
                package_name = None

            cases.append(
                BatchRepairCase(
                    index=index,
                    program=str(program),
                    buggy_code=str(buggy_code),
                    error_message=str(error_message),
                    test_file=str(test_file),
                    package_name=package_name,
                )
            )
        return cases

    def _repair_cases(
        self,
        cases: Sequence[BatchRepairCase],
        output_path: Path,
    ) -> List[BatchRepairRecord]:
        if not cases:
            return []

        worker_count = self._resolve_worker_count(len(cases))
        if worker_count <= 1:
            return [self._repair_case(case, output_path) for case in cases]

        ordered_records: List[Optional[BatchRepairRecord]] = [None] * len(cases)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_index = {
                executor.submit(self._repair_case, case, output_path): idx
                for idx, case in enumerate(cases)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                ordered_records[idx] = future.result()

        return [record for record in ordered_records if record is not None]

    def _repair_case(self, case: BatchRepairCase, output_path: Path) -> BatchRepairRecord:
        candidate_path = output_path / f"{case.program}.py"
        status = "OK"
        error_message = ""
        generated_code = case.buggy_code
        best_reward = float("-inf")
        iterations = 0
        pass_count = 0
        elapsed_sec = 0.0

        tester = Tester(case.test_file, module_name=case.program, package_name=case.package_name)
        bug_description = case.error_message
        if not bug_description:
            _, _, initial_output = tester.run_with_details(case.buggy_code)
            if initial_output:
                bug_description = initial_output

        few_shot_examples = self._build_few_shot_examples(case, bug_description)

        def _generate(code: str) -> str:
            return self.generator.generate(
                code,
                bug_description=bug_description,
                instruction=self.instruction_override,
                few_shot_examples=few_shot_examples,
                error_message=bug_description,
            )

        def _refine(code: str, feedback: str) -> str:
            return self.generator.generate(
                code,
                bug_description=bug_description,
                feedback=feedback,
                instruction=self.instruction_override,
                few_shot_examples=few_shot_examples,
                error_message=bug_description,
            )

        def _reward(code: str) -> Tuple[int, str, str]:
            return tester.run_with_details(code)

        def _record(record: Dict[str, object]) -> None:
            nonlocal iterations, pass_count
            iterations += 1
            if record.get("status") == "PASS":
                pass_count += 1

        start_time = time.perf_counter()
        try:
            root = Node(state=case.buggy_code)
            mcts = MCTS(
                root,
                generator=_generate,
                reward_fn=_reward,
                record_fn=_record,
                refine_fn=_refine,
                parallelism=self.mcts_parallelism,
                stop_on_pass=self.stop_on_pass,
                max_refine_attempts=self.max_refine_attempts,
            )
            mcts.run(self.mcts_iterations)
            generated_code = mcts.best_patch or case.buggy_code
            best_reward = mcts.best_reward
            if best_reward >= 1.0:
                status = "PASS"
            elif best_reward > float("-inf"):
                status = "FAIL"
            else:
                status = "UNKNOWN"
        except Exception as exc:
            status = "ERROR"
            error_message = str(exc)
            generated_code = case.buggy_code
        finally:
            elapsed_sec = time.perf_counter() - start_time

        candidate_path.write_text(generated_code, encoding="utf-8")
        return BatchRepairRecord(
            index=case.index,
            program=case.program,
            candidate_path=str(candidate_path),
            test_file=case.test_file,
            package_name=case.package_name,
            status=status,
            error_message=error_message,
            generated_code=generated_code,
            best_reward=best_reward,
            iterations=iterations,
            pass_count=pass_count,
            elapsed_sec=elapsed_sec,
            source="mcts",
        )

    def _build_few_shot_examples(self, case: BatchRepairCase, bug_description: str) -> str:
        if not self.ape_examples or self.ape_few_shot_k <= 0:
            return ""

        selector = ExampleSelector(max_examples=self.ape_few_shot_k)
        query = BugExample(
            buggy_code=case.buggy_code,
            error_message=bug_description,
            correct_code="",
        )
        selected = selector.select_examples(query, self.ape_examples, k=self.ape_few_shot_k)
        return selector.format_examples(selected)

    def _resolve_worker_count(self, total_cases: int) -> int:
        if self.workers and self.workers > 0:
            return min(self.workers, total_cases)
        cpu_count = os.cpu_count() or 1
        return max(1, min(cpu_count, total_cases, 8))