from __future__ import annotations

from textwrap import dedent
from pathlib import Path

from repair_metrics import RepairMetricsEvaluator


def test_plausible_and_correct_can_diverge(tmp_path: Path) -> None:
    tests_path = tmp_path / "test_candidate.py"
    tests_path.write_text(
        dedent(
            """
            from candidate import pick


            def test_pick_true_case():
                assert pick(True) == 1
            """
        ),
        encoding="utf-8",
    )

    candidate_code = dedent(
        """
        def pick(flag):
            return 1 if flag else 0
        """
    )
    reference_code = dedent(
        """
        def pick(flag):
            return 1 if flag else 2
        """
    )

    evaluator = RepairMetricsEvaluator(
        tests_path=str(tests_path),
        module_name="candidate",
    )

    result = evaluator.evaluate(candidate_code, reference_code, name="pick")

    assert result.plausible_fix is True
    assert result.correct_fix is False


def test_python_ast_equivalence_ignores_formatting_and_docstrings(tmp_path: Path) -> None:
    tests_path = tmp_path / "test_candidate.py"
    tests_path.write_text(
        dedent(
            """
            from candidate import add


            def test_add():
                assert add(2, 3) == 5
            """
        ),
        encoding="utf-8",
    )

    candidate_code = dedent(
        '''
        def add(a, b):
            """Add two values."""
            return a + b
        '''
    )
    reference_code = dedent(
        '''
        def add(a, b):
            return a + b
        '''
    )

    evaluator = RepairMetricsEvaluator(
        tests_path=str(tests_path),
        module_name="candidate",
    )

    result = evaluator.evaluate(candidate_code, reference_code, name="add")

    assert result.plausible_fix is True
    assert result.correct_fix is True
    assert result.equivalence == "python-ast"