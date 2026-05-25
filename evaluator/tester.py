from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
from typing import Optional, Tuple

MAX_TEST_OUTPUT_CHARS = 8000


def run_tests_with_status(
    code: str,
    tests_path: str,
    module_name: str = "candidate",
    package_name: Optional[str] = None,
) -> Tuple[int, str, str]:
    """Run pytest and return (reward, status, output)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        if package_name:
            package_dir = os.path.join(temp_dir, package_name)
            os.makedirs(package_dir, exist_ok=True)
            init_path = os.path.join(package_dir, "__init__.py")
            with open(init_path, "w", encoding="utf-8"):
                pass
            module_path = os.path.join(package_dir, f"{module_name}.py")
        else:
            module_path = os.path.join(temp_dir, f"{module_name}.py")
        with open(module_path, "w", encoding="utf-8") as handle:
            handle.write(code)

        env = os.environ.copy()
        tests_root = (
            tests_path if os.path.isdir(tests_path) else os.path.dirname(tests_path)
        )
        if not tests_root:
            tests_root = os.getcwd()

        env["PYTHONPATH"] = os.pathsep.join(
            [temp_dir, tests_root, env.get("PYTHONPATH", "")]
        ).strip(os.pathsep)
        env["APR_PATCH_PATH"] = module_path

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    tests_path,
                    "-q",
                    "--rootdir",
                    temp_dir,
                    "--import-mode=importlib",
                ],
                cwd=temp_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except Exception as exc:
            return -1, "ERROR", f"Exception running pytest: {exc}"

        output = _combine_output(result.stdout, result.stderr)
        if result.returncode == 0:
            return 1, "PASS", output
        if result.returncode == 1:
            return 0, "FAIL", output
        return -1, "ERROR", output


def run_tests(
    code: str,
    tests_path: str,
    module_name: str = "candidate",
    package_name: Optional[str] = None,
) -> int:
    """Run pytest against tests_path and return a reward.

    Reward mapping:
    - PASS: +1
    - FAIL: 0
    - ERROR: -1
    """
    reward, _, _ = run_tests_with_status(code, tests_path, module_name, package_name)
    return reward


class Tester:
    def __init__(
        self,
        tests_path: str,
        module_name: str = "candidate",
        package_name: Optional[str] = None,
    ) -> None:
        self.tests_path = tests_path
        self.module_name = module_name
        self.package_name = package_name
        self.last_status = "UNKNOWN"
        self.last_output = ""
        self._lock = threading.Lock()

    def run(self, code: str) -> int:
        """Run tests and return a reward based on results."""
        reward, status, output = run_tests_with_status(
            code, self.tests_path, self.module_name, self.package_name
        )
        with self._lock:
            self.last_status = status
            self.last_output = output
        return reward

    def run_with_status(self, code: str) -> Tuple[int, str]:
        """Run tests and return (reward, status)."""
        reward, status, output = run_tests_with_status(
            code, self.tests_path, self.module_name, self.package_name
        )
        with self._lock:
            self.last_status = status
            self.last_output = output
        return reward, status

    def run_with_details(self, code: str) -> Tuple[int, str, str]:
        """Run tests and return (reward, status, output)."""
        reward, status, output = run_tests_with_status(
            code, self.tests_path, self.module_name, self.package_name
        )
        with self._lock:
            self.last_status = status
            self.last_output = output
        return reward, status, output


def _combine_output(stdout: str, stderr: str) -> str:
    parts = []
    if stdout.strip():
        parts.append(f"STDOUT:\n{stdout}")
    if stderr.strip():
        parts.append(f"STDERR:\n{stderr}")
    combined = "\n\n".join(parts).strip()
    if not combined:
        return ""
    if len(combined) <= MAX_TEST_OUTPUT_CHARS:
        return combined
    return combined[:MAX_TEST_OUTPUT_CHARS] + "\n...<truncated>"
