from __future__ import annotations

import ast
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple
import builtins
import keyword
import random
import tempfile
import multiprocessing
import time
from multiprocessing import Queue

from evaluator.tester import run_tests_with_status


def normalize_whitespace(code: str) -> str:
    """Collapse whitespace so small formatting differences do not matter."""
    return re.sub(r"\s+", " ", code.strip())


def _strip_leading_docstring(body: List[ast.stmt]) -> List[ast.stmt]:
    if not body:
        return body
    first_stmt = body[0]
    if not isinstance(first_stmt, ast.Expr):
        return body
    value = getattr(first_stmt, "value", None)
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return body[1:]
    return body


class _PythonDocstringStripper(ast.NodeTransformer):
    """Remove leading docstrings before AST comparison."""

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self.generic_visit(node)
        node.body = _strip_leading_docstring(list(node.body))
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        node.body = _strip_leading_docstring(list(node.body))
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        self.generic_visit(node)
        node.body = _strip_leading_docstring(list(node.body))
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        self.generic_visit(node)
        node.body = _strip_leading_docstring(list(node.body))
        return node


def _python_ast_fingerprint(code: str) -> Optional[str]:
    try:
        parsed = ast.parse(code)
    except SyntaxError:
        return None

    cleaned = _PythonDocstringStripper().visit(parsed)
    ast.fix_missing_locations(cleaned)
    return ast.dump(cleaned, include_attributes=False)


class _AlphaRenamer(ast.NodeTransformer):
    """Rename identifiers deterministically to canonical tokens.

    This is conservative: it skips obvious builtins and keywords and does not
    rename attribute names to avoid changing external API references.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mapping: Dict[str, str] = {}
        self.counter = 0
        # builtins and keywords should not be renamed
        self._skip = set(dir(builtins)) | set(keyword.kwlist)

    def _canon(self, name: str) -> str:
        if name in ("self", "cls"):
            return name
        if name in self._skip:
            return name
        if name not in self.mapping:
            self.mapping[name] = f"v{self.counter}"
            self.counter += 1
        return self.mapping[name]

    def visit_Name(self, node: ast.Name) -> ast.AST:
        # Only replace plain names, not attributes
        new_id = self._canon(node.id)
        return ast.copy_location(ast.Name(id=new_id, ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.arg:
        if node.arg in ("self", "cls"):
            return node
        node.arg = self._canon(node.arg)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node = self.generic_visit(node)
        node.name = self._canon(node.name)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node = self.generic_visit(node)
        node.name = self._canon(node.name)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node = self.generic_visit(node)
        node.name = self._canon(node.name)
        return node


def _python_ast_canonical(code: str) -> Optional[str]:
    try:
        parsed = ast.parse(code)
    except SyntaxError:
        return None

    cleaned = _PythonDocstringStripper().visit(parsed)
    renamed = _AlphaRenamer().visit(cleaned)
    ast.fix_missing_locations(renamed)
    return ast.dump(renamed, include_attributes=False)


def _worker_for_subprocess(path: str, fname: str, q: Queue, a: tuple) -> None:
    try:
        ns = {}
        with open(path, "r", encoding="utf-8") as fh:
            code = fh.read()
        exec(compile(code, path, "exec"), ns)
        func = ns.get(fname)
        if not callable(func):
            q.put((False, "<no-func>"))
            return
        res = func(*a)
        q.put((True, repr(res)))
    except Exception as e:
        q.put((False, repr(e)))


def _call_function_in_subprocess(module_path: str, func_name: str, args: tuple, timeout: float = 2.0) -> Tuple[bool, str]:
    """Run a function from a module file in a subprocess and return (ok, repr).

    ok True means the call returned a value; False means exception or timeout.
    """

    q: Queue = multiprocessing.Manager().Queue()
    proc = multiprocessing.Process(target=_worker_for_subprocess, args=(module_path, func_name, q, args))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        return False, "<timeout>"
    try:
        ok, payload = q.get_nowait()
    except Exception:
        return False, "<no-result>"
    return ok, payload


def _behavioral_equivalence(candidate_code: str, reference_code: str, trials: int = 10) -> bool:
    """Heuristic differential testing between candidate and reference.

    It attempts to find top-level function names that appear in both modules
    and invoke them on a set of small random/sample inputs. If outputs match
    for a sufficient fraction, we treat them as behaviorally equivalent.
    """
    try:
        cand_ast = ast.parse(candidate_code)
        ref_ast = ast.parse(reference_code)
    except SyntaxError:
        return False

    cand_funcs = {n.name: n for n in cand_ast.body if isinstance(n, ast.FunctionDef)}
    ref_funcs = {n.name: n for n in ref_ast.body if isinstance(n, ast.FunctionDef)}
    common = set(cand_funcs.keys()) & set(ref_funcs.keys())
    if not common:
        return False

    # prepare temp files
    with tempfile.TemporaryDirectory() as td:
        cand_path = f"{td}/cand_mod.py"
        ref_path = f"{td}/ref_mod.py"
        open(cand_path, "w", encoding="utf-8").write(candidate_code)
        open(ref_path, "w", encoding="utf-8").write(reference_code)

        # sample values
        base_vals = [0, 1, -1, 2, [], [1], [1, 2], "", "a", None, True, False]

        for func_name in common:
            func_node = cand_funcs[func_name]
            argcount = len(func_node.args.args)
            if argcount > 3:
                # skip functions with many args (hard to fuzz generically)
                continue

            matches = 0
            runs = 0
            for _ in range(trials):
                args = tuple(random.choice(base_vals) for _ in range(argcount))
                ok1, out1 = _call_function_in_subprocess(cand_path, func_name, args)
                ok2, out2 = _call_function_in_subprocess(ref_path, func_name, args)
                runs += 1
                if ok1 and ok2 and out1 == out2:
                    matches += 1
                # if both raised similarly (repr equal) consider match as well
                elif (not ok1) and (not ok2) and out1 == out2:
                    matches += 1

            if runs and (matches / runs) >= 0.8:
                return True

    return False


def compare_code(
    candidate_code: str,
    reference_code: str,
    language: str = "python",
    mode: str = "hybrid",
) -> Tuple[bool, str]:
    """Compare two code snippets using a selectable equivalence heuristic.

    Modes:
    - fast: whitespace or exact AST fingerprint
    - ast_canonical: use AST canonicalization (alpha-renaming)
    - behavioral: run heuristic differential testing
    - hybrid: try ast_canonical then behavioral fallback
    """
    if normalize_whitespace(candidate_code) == normalize_whitespace(reference_code):
        return True, "whitespace"

    if language.lower() != "python":
        return False, "mismatch"

    mode = mode.lower()
    if mode in ("fast", "ast"):
        cand_fp = _python_ast_fingerprint(candidate_code)
        ref_fp = _python_ast_fingerprint(reference_code)
        if cand_fp is not None and cand_fp == ref_fp:
            return True, "python-ast"
        return False, "mismatch"

    if mode == "ast_canonical":
        cand_can = _python_ast_canonical(candidate_code)
        ref_can = _python_ast_canonical(reference_code)
        if cand_can is not None and cand_can == ref_can:
            return True, "ast-canonical"
        return False, "mismatch"

    if mode == "behavioral":
        ok = _behavioral_equivalence(candidate_code, reference_code)
        return (ok, "behavioral") if ok else (False, "mismatch")

    # hybrid: quick fingerprint first (keeps previous semantics), then
    # canonical AST, then behavioral fallback.
    cand_fp = _python_ast_fingerprint(candidate_code)
    ref_fp = _python_ast_fingerprint(reference_code)
    if cand_fp is not None and cand_fp == ref_fp:
        return True, "python-ast"

    cand_can = _python_ast_canonical(candidate_code)
    ref_can = _python_ast_canonical(reference_code)
    if cand_can is not None and cand_can == ref_can:
        return True, "ast-canonical"

    # behavioral fallback
    ok = _behavioral_equivalence(candidate_code, reference_code)
    return (ok, "behavioral") if ok else (False, "mismatch")


@dataclass
class RepairEvaluation:
    """Result for a single candidate repair."""

    name: str
    plausible_fix: bool
    correct_fix: bool
    tests_passed: bool
    test_status: str
    equivalence: str
    test_output: str
    candidate_code: str
    reference_code: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class RepairSummary:
    """Aggregate counts for a batch of repair candidates."""

    total: int
    plausible_fix_count: int
    correct_fix_count: int
    records: List[RepairEvaluation]

    @property
    def plausible_fix_rate(self) -> float:
        return self.plausible_fix_count / self.total if self.total else 0.0

    @property
    def correct_fix_rate(self) -> float:
        return self.correct_fix_count / self.total if self.total else 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "total": self.total,
            "plausible_fix_count": self.plausible_fix_count,
            "correct_fix_count": self.correct_fix_count,
            "plausible_fix_rate": self.plausible_fix_rate,
            "correct_fix_rate": self.correct_fix_rate,
            "records": [record.to_dict() for record in self.records],
        }


class RepairMetricsEvaluator:
    """Evaluate plausible and correct fixes for a single benchmark case."""

    def __init__(
        self,
        tests_path: str,
        module_name: str = "candidate",
        package_name: Optional[str] = None,
        language: str = "python",
        equivalence_mode: str = "hybrid",
    ) -> None:
        self.tests_path = tests_path
        self.module_name = module_name
        self.package_name = package_name
        self.language = language
        self.equivalence_mode = equivalence_mode

    def evaluate(
        self,
        candidate_code: str,
        reference_code: str,
        name: str = "candidate",
    ) -> RepairEvaluation:
        reward, test_status, test_output = run_tests_with_status(
            candidate_code,
            self.tests_path,
            self.module_name,
            self.package_name,
        )
        tests_passed = reward > 0 and test_status == "PASS"
        plausible_fix = tests_passed
        equivalent, equivalence = compare_code(
            candidate_code,
            reference_code,
            language=self.language,
            mode=self.equivalence_mode,
        )
        correct_fix = tests_passed and equivalent
        return RepairEvaluation(
            name=name,
            plausible_fix=plausible_fix,
            correct_fix=correct_fix,
            tests_passed=tests_passed,
            test_status=test_status,
            equivalence=equivalence if tests_passed else "not-checked",
            test_output=test_output,
            candidate_code=candidate_code,
            reference_code=reference_code,
        )

    def evaluate_batch(
        self,
        candidates: Sequence[Tuple[str, str, str]],
    ) -> RepairSummary:
        records: List[RepairEvaluation] = []
        plausible_fix_count = 0
        correct_fix_count = 0

        for name, candidate_code, reference_code in candidates:
            record = self.evaluate(candidate_code, reference_code, name=name)
            records.append(record)
            if record.plausible_fix:
                plausible_fix_count += 1
            if record.correct_fix:
                correct_fix_count += 1

        return RepairSummary(
            total=len(records),
            plausible_fix_count=plausible_fix_count,
            correct_fix_count=correct_fix_count,
            records=records,
        )