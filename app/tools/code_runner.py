"""
Code execution engine for the agentic network.

Runs generated code in a subprocess sandbox, captures stdout/stderr,
and returns structured results that the repair loop can act on.

Supports: Python, JavaScript/Node, Bash, TypeScript (via ts-node or npx tsx)
"""

import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Result of running a file or command."""
    success: bool
    returncode: int
    stdout: str
    stderr: str
    elapsed_s: float
    command: str
    file_path: str = ""

    @property
    def output(self) -> str:
        """Combined stdout + stderr for display."""
        parts = []
        if self.stdout.strip():
            parts.append(self.stdout.strip())
        if self.stderr.strip():
            parts.append(f"[stderr]\n{self.stderr.strip()}")
        return "\n".join(parts) if parts else "(no output)"

    @property
    def error_summary(self) -> str:
        """Short error description for the repair prompt."""
        if self.success:
            return ""
        lines = (self.stderr or self.stdout).strip().splitlines()
        # Return last 20 lines — that's where Python tracebacks end
        return "\n".join(lines[-20:])


@dataclass
class TestResult:
    """Result of running a test suite."""
    success: bool
    passed: int
    failed: int
    errors: int
    output: str
    framework: str  # pytest, unittest, jest, etc.
    elapsed_s: float
    failed_tests: List[str] = field(default_factory=list)


@dataclass
class BuildReport:
    """Aggregated report for one build-test-fix iteration."""
    iteration: int
    files_written: List[str]
    execution_results: List[ExecutionResult]
    test_result: Optional[TestResult]
    syntax_errors: List[Tuple[str, str]]   # (file_path, error_message)
    all_passed: bool
    summary: str


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

_EXT_RUNNER: Dict[str, List[str]] = {
    ".py":  [sys.executable],
    ".js":  ["node"],
    ".ts":  ["npx", "tsx"],
    ".sh":  ["bash"],
    ".rb":  ["ruby"],
    ".go":  ["go", "run"],
    ".rs":  [],   # needs cargo — skip direct execution
    ".java": [],  # needs javac — skip direct execution
}

_TEST_PATTERNS = {
    "pytest":    re.compile(r"(test_\w+\.py|_test\.py)$"),
    "unittest":  re.compile(r"test.*\.py$"),
    "jest":      re.compile(r"\.(test|spec)\.(js|ts)$"),
}


def _detect_runner(path: str) -> Optional[List[str]]:
    ext = Path(path).suffix.lower()
    return _EXT_RUNNER.get(ext)


def _is_test_file(path: str) -> bool:
    name = Path(path).name
    return any(p.search(name) for p in _TEST_PATTERNS.values())


# ---------------------------------------------------------------------------
# Code runner
# ---------------------------------------------------------------------------

class CodeRunner:
    """Execute code files and test suites in a subprocess sandbox."""

    TIMEOUT = 30  # seconds per execution

    def __init__(self, workspace_root: str = "."):
        self.workspace_root = str(Path(workspace_root).resolve())

    # ------------------------------------------------------------------
    # Syntax checking (fast, no execution)
    # ------------------------------------------------------------------

    def check_syntax(self, file_path: str) -> Optional[str]:
        """Return error string if file has syntax errors, else None."""
        full = os.path.join(self.workspace_root, file_path)
        if not os.path.exists(full):
            return f"File not found: {file_path}"

        ext = Path(file_path).suffix.lower()
        if ext == ".py":
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", full],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return result.stderr.strip()
        return None

    def check_syntax_all(self, file_paths: List[str]) -> List[Tuple[str, str]]:
        """Check syntax for all files. Returns list of (path, error) for failures."""
        errors = []
        for path in file_paths:
            err = self.check_syntax(path)
            if err:
                errors.append((path, err))
        return errors

    # ------------------------------------------------------------------
    # File execution
    # ------------------------------------------------------------------

    def run_file(self, file_path: str, args: List[str] = None) -> ExecutionResult:
        """Execute a single file and return the result."""
        full = os.path.join(self.workspace_root, file_path)
        if not os.path.exists(full):
            return ExecutionResult(
                success=False, returncode=-1,
                stdout="", stderr=f"File not found: {file_path}",
                elapsed_s=0.0, command="", file_path=file_path
            )

        runner = _detect_runner(file_path)
        if runner is None:
            return ExecutionResult(
                success=False, returncode=-1,
                stdout="", stderr=f"No runner for {Path(file_path).suffix} files",
                elapsed_s=0.0, command="", file_path=file_path
            )
        if not runner:
            return ExecutionResult(
                success=True, returncode=0,
                stdout=f"(skipped — {Path(file_path).suffix} requires build step)",
                stderr="", elapsed_s=0.0, command="", file_path=file_path
            )

        cmd = runner + [full] + (args or [])
        return self._run_cmd(cmd, file_path=file_path)

    # ------------------------------------------------------------------
    # Test execution
    # ------------------------------------------------------------------

    def run_tests(self, test_paths: List[str] = None) -> TestResult:
        """Run the test suite. Auto-discovers tests if test_paths is None."""
        t0 = time.perf_counter()

        # Auto-discover test files if not specified
        if test_paths is None:
            test_paths = self._discover_tests()

        if not test_paths:
            return TestResult(
                success=True, passed=0, failed=0, errors=0,
                output="No test files found.", framework="none",
                elapsed_s=0.0
            )

        # Prefer pytest if available
        if self._has_pytest():
            return self._run_pytest(test_paths, t0)
        else:
            return self._run_unittest(test_paths, t0)

    def _has_pytest(self) -> bool:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "--version"],
                capture_output=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def _run_pytest(self, test_paths: List[str], t0: float) -> TestResult:
        full_paths = [os.path.join(self.workspace_root, p) for p in test_paths]
        cmd = [sys.executable, "-m", "pytest", "--tb=short", "-q"] + full_paths
        result = self._run_cmd(cmd)

        elapsed = time.perf_counter() - t0
        output = result.output

        # Parse pytest summary line: "3 passed, 1 failed in 0.12s"
        passed = failed = errors = 0
        failed_tests: List[str] = []

        m = re.search(r"(\d+) passed", output)
        if m:
            passed = int(m.group(1))
        m = re.search(r"(\d+) failed", output)
        if m:
            failed = int(m.group(1))
        m = re.search(r"(\d+) error", output)
        if m:
            errors = int(m.group(1))

        # Extract failed test names
        for line in output.splitlines():
            if line.startswith("FAILED "):
                failed_tests.append(line[7:].split(" - ")[0].strip())

        return TestResult(
            success=(result.success and failed == 0 and errors == 0),
            passed=passed, failed=failed, errors=errors,
            output=output, framework="pytest",
            elapsed_s=elapsed, failed_tests=failed_tests
        )

    def _run_unittest(self, test_paths: List[str], t0: float) -> TestResult:
        full_paths = [os.path.join(self.workspace_root, p) for p in test_paths]
        cmd = [sys.executable, "-m", "unittest"] + full_paths
        result = self._run_cmd(cmd)

        elapsed = time.perf_counter() - t0
        output = result.output

        passed = failed = errors = 0
        m = re.search(r"Ran (\d+) test", output)
        total = int(m.group(1)) if m else 0
        m = re.search(r"failures=(\d+)", output)
        if m:
            failed = int(m.group(1))
        m = re.search(r"errors=(\d+)", output)
        if m:
            errors = int(m.group(1))
        passed = total - failed - errors

        return TestResult(
            success=result.success and failed == 0 and errors == 0,
            passed=passed, failed=failed, errors=errors,
            output=output, framework="unittest",
            elapsed_s=elapsed
        )

    def _discover_tests(self) -> List[str]:
        """Find test files in the workspace."""
        tests = []
        for root, dirs, files in os.walk(self.workspace_root):
            # Skip hidden dirs and common non-source dirs
            dirs[:] = [d for d in dirs if not d.startswith(".")
                       and d not in ("node_modules", "__pycache__", ".git", "venv", ".venv")]
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), self.workspace_root)
                if _is_test_file(rel):
                    tests.append(rel)
        return tests

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------

    def run_command(
        self,
        command,
        cwd: str = ".",
        shell: bool = False,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """Run a command in the workspace."""
        return self._run_cmd(
            command,
            shell=shell,
            cwd=cwd,
            env=env,
            timeout=timeout,
        )

    def _run_cmd(
        self,
        cmd,
        shell: bool = False,
        file_path: str = "",
        cwd: str = ".",
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        cmd_str = cmd if isinstance(cmd, str) else " ".join(str(c) for c in cmd)
        t0 = time.perf_counter()
        try:
            full_cwd = self._resolve_cwd(cwd)
            merged_env = os.environ.copy()
            if env:
                merged_env.update(env)
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout or self.TIMEOUT,
                cwd=full_cwd,
                shell=shell,
                env=merged_env,
            )
            elapsed = time.perf_counter() - t0
            return ExecutionResult(
                success=proc.returncode == 0,
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                elapsed_s=elapsed,
                command=cmd_str,
                file_path=file_path,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - t0
            return ExecutionResult(
                success=False, returncode=-1,
                stdout="", stderr=f"Timed out after {timeout or self.TIMEOUT}s",
                elapsed_s=elapsed, command=cmd_str, file_path=file_path
            )
        except FileNotFoundError as e:
            elapsed = time.perf_counter() - t0
            return ExecutionResult(
                success=False, returncode=-1,
                stdout="", stderr=f"Command not found: {e}",
                elapsed_s=elapsed, command=cmd_str, file_path=file_path
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            return ExecutionResult(
                success=False, returncode=-1,
                stdout="", stderr=str(e),
                elapsed_s=elapsed, command=cmd_str, file_path=file_path
            )

    def _resolve_cwd(self, cwd: str) -> str:
        """Resolve a command working directory inside the workspace."""
        raw_cwd = (cwd or ".").strip() or "."
        candidate = Path(raw_cwd)
        if candidate.is_absolute():
            raise ValueError(f"Absolute working directories are not allowed: {cwd!r}")

        root = Path(self.workspace_root).resolve()
        full = (root / candidate).resolve(strict=False)
        try:
            common = os.path.commonpath([
                os.path.normcase(str(root)),
                os.path.normcase(str(full)),
            ])
        except ValueError as exc:
            raise ValueError(f"Working directory escapes workspace: {cwd!r}") from exc

        if common != os.path.normcase(str(root)):
            raise ValueError(f"Working directory escapes workspace: {cwd!r}")
        return str(full)
