from __future__ import annotations

import json
from pathlib import Path

from batch_repair import BatchRepairRunner


class FakeGenerator:
    def generate(self, buggy_code: str, **_: object) -> str:
        if "return a - b" in buggy_code:
            return buggy_code.replace("return a - b", "return a + b")
        if "return 0" in buggy_code:
            return buggy_code.replace("return 0", "return 1 if flag else 0")
        return buggy_code


def test_batch_runner_writes_all_candidates_and_manifest(tmp_path: Path) -> None:
    tests_a = tmp_path / "test_a.py"
    tests_a.write_text("from a import add\n\ndef test_add():\n    assert add(2, 3) == 5\n", encoding="utf-8")

    tests_b = tmp_path / "test_b.py"
    tests_b.write_text("from b import pick\n\ndef test_pick():\n    assert pick(True) == 1\n", encoding="utf-8")

    entries = [
        {
            "buggy_code": "def add(a, b):\n    return a - b\n",
            "error_message": "addition expected",
            "correct_code": "def add(a, b):\n    return a + b\n",
            "metadata": {"program": "a", "test_file": str(tests_a)},
        },
        {
            "buggy_code": "def pick(flag):\n    return 0\n",
            "error_message": "boolean fix expected",
            "correct_code": "def pick(flag):\n    return 1 if flag else 0\n",
            "metadata": {"program": "b", "test_file": str(tests_b)},
        },
    ]

    output_dir = tmp_path / "repairs"
    runner = BatchRepairRunner(FakeGenerator(), workers=2)
    summary = runner.run_entries(entries, str(output_dir))

    assert summary.total == 2
    assert summary.generated_count == 2
    assert summary.error_count == 0

    candidate_a = output_dir / "a.py"
    candidate_b = output_dir / "b.py"
    manifest = output_dir / "batch_results.json"

    assert candidate_a.exists()
    assert candidate_b.exists()
    assert manifest.exists()

    assert "return a + b" in candidate_a.read_text(encoding="utf-8")
    assert "return 1 if flag else 0" in candidate_b.read_text(encoding="utf-8")

    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_data["total"] == 2
    assert manifest_data["generated_count"] == 2
    assert manifest_data["error_count"] == 0