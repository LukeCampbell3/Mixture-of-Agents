"""
Codebase builder: recursive write -> run -> test -> fix loop.

Wraps the agent execution pipeline with:
  1. Write files (via tool calls and CodeExtractor fallback)
  2. Syntax check all written files
  3. Execute entry-point files to verify they run
  4. Run manifest-derived validation commands when applicable
  5. Run tests
  6. If failures exist, build a repair prompt and re-invoke the agent
  7. Repeat until all checks pass or max_iterations is reached
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.tools.code_runner import CodeRunner, ExecutionResult, TestResult
from app.tools.filesystem import (
    CodeExtractor,
    FileOperation,
    FilesystemExecutor,
    OperationResult,
    parse_tool_calls,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BuildConfig:
    """Configuration for a codebase build session."""

    workspace_root: str = "."
    max_iterations: int = 5
    tokens_per_iteration: int = 4000
    run_entry_points: bool = True
    run_tests: bool = True
    run_validation_commands: bool = True
    test_scope: str = "written"  # "written" or "workspace"
    auto_generate_tests: bool = True
    timeout_per_run: int = 30
    verbose: bool = True


@dataclass(frozen=True)
class ValidationCommand:
    """A build or test command derived from project manifests."""

    command: Tuple[str, ...]
    description: str
    cwd: str = "."
    env: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Build session
# ---------------------------------------------------------------------------


@dataclass
class IterationRecord:
    """Record of one build-test-fix iteration."""

    iteration: int
    files_written: List[str]
    syntax_errors: List[tuple]
    execution_results: List[ExecutionResult]
    command_results: List[ExecutionResult]
    test_result: Optional[TestResult]
    passed: bool
    repair_needed: bool
    repair_prompt: str = ""
    elapsed_s: float = 0.0


@dataclass
class BuildSession:
    """Complete record of a codebase build session."""

    task: str
    workspace_root: str
    iterations: List[IterationRecord] = field(default_factory=list)
    final_files: List[str] = field(default_factory=list)
    applied_tool_calls: List[FileOperation] = field(default_factory=list)
    success: bool = False
    total_elapsed_s: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Build session: {len(self.iterations)} iteration(s)",
            f"Final files: {', '.join(self.final_files) or 'none'}",
            f"Status: {'PASS' if self.success else 'FAIL'}",
            f"Total time: {self.total_elapsed_s:.1f}s",
        ]
        for rec in self.iterations:
            status = "PASS" if rec.passed else "FAIL"
            failing_commands = sum(1 for result in rec.command_results if not result.success)
            tests_summary = (
                "N/A"
                if rec.test_result is None
                else f"{rec.test_result.passed}P/{rec.test_result.failed}F"
            )
            lines.append(
                f"  Iter {rec.iteration}: {status}  "
                f"files={len(rec.files_written)}  "
                f"syntax_errors={len(rec.syntax_errors)}  "
                f"commands={failing_commands}F  "
                f"tests={tests_summary}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class CodebaseBuilder:
    """Orchestrates the recursive build loop."""

    def __init__(
        self,
        config: BuildConfig,
        agent_fn: Callable[[str, int], str],
    ):
        self.config = config
        self.agent_fn = agent_fn
        self.runner = CodeRunner(workspace_root=config.workspace_root)
        self.executor = FilesystemExecutor(workspace_root=config.workspace_root)
        self.extractor = CodeExtractor(workspace_root=config.workspace_root)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        task_text: str,
        initial_response: str,
        existing_tool_calls: List[FileOperation] = None,
        existing_tool_results: List[OperationResult] = None,
    ) -> BuildSession:
        """Run the full build loop starting from an initial agent response."""

        session = BuildSession(task=task_text, workspace_root=self.config.workspace_root)
        t_session = time.perf_counter()

        current_response = initial_response
        current_tool_calls = existing_tool_calls or []
        current_tool_results = existing_tool_results or []
        tracked_files: List[str] = []

        for iteration in range(1, self.config.max_iterations + 1):
            t_iter = time.perf_counter()
            if self.config.verbose:
                print(f"\n  [build] Iteration {iteration}/{self.config.max_iterations}")

            new_files_written, applied_ops = self._write_files(
                current_response,
                task_text,
                current_tool_calls,
                current_tool_results,
            )
            for path in new_files_written:
                if path not in tracked_files:
                    tracked_files.append(path)
            files_written = list(tracked_files)
            for op in applied_ops:
                if not self._has_matching_op(session.applied_tool_calls, op):
                    session.applied_tool_calls.append(op)

            if self.config.verbose and new_files_written:
                for path in new_files_written:
                    print(f"  [build]   wrote: {path}")

            py_files = [path for path in files_written if path.endswith(".py")]
            syntax_errors = self.runner.check_syntax_all(py_files)
            if self.config.verbose and syntax_errors:
                for path, err in syntax_errors:
                    print(f"  [build]   syntax error in {path}: {err[:80]}")

            execution_results: List[ExecutionResult] = []
            if self.config.run_entry_points and not syntax_errors:
                for entry_point in self._find_entry_points(files_written):
                    result = self.runner.run_file(entry_point)
                    execution_results.append(result)
                    if self.config.verbose:
                        status = "PASS" if result.success else "FAIL"
                        print(
                            f"  [build]   run {entry_point}: {status} "
                            f"({result.elapsed_s:.2f}s)"
                        )
                        if not result.success:
                            print(f"  [build]     {result.error_summary[:200]}")

            related_tests = self._related_test_paths(files_written)
            if self.config.auto_generate_tests and iteration == 1 and py_files and not related_tests:
                test_response = self._request_tests(task_text, files_written)
                test_files, test_ops = self._write_files(
                    test_response,
                    task_text + " tests",
                    [],
                    [],
                )
                if self.config.verbose and test_files:
                    for path in test_files:
                        print(f"  [build]   wrote test: {path}")
                for path in test_files:
                    if path not in tracked_files:
                        tracked_files.append(path)
                files_written = list(tracked_files)
                for op in test_ops:
                    if not self._has_matching_op(session.applied_tool_calls, op):
                        session.applied_tool_calls.append(op)
                related_tests = self._related_test_paths(files_written)

            command_results: List[ExecutionResult] = []
            if self.config.run_validation_commands and not syntax_errors:
                command_results = self._run_validation_commands(files_written)

            test_result: Optional[TestResult] = None
            if self.config.run_tests and not syntax_errors:
                test_paths = None if self.config.test_scope == "workspace" else related_tests
                test_result = self.runner.run_tests(test_paths=test_paths)
                if self.config.verbose:
                    if test_result.framework == "none":
                        print("  [build]   tests: no generated test files found")
                    else:
                        status = "PASS" if test_result.success else "FAIL"
                        print(
                            f"  [build]   tests ({test_result.framework}): {status}  "
                            f"{test_result.passed}P / {test_result.failed}F / {test_result.errors}E"
                        )

            passed = self._assess_iteration(
                syntax_errors,
                execution_results,
                command_results,
                test_result,
            )
            elapsed_iter = time.perf_counter() - t_iter

            record = IterationRecord(
                iteration=iteration,
                files_written=files_written,
                syntax_errors=syntax_errors,
                execution_results=execution_results,
                command_results=command_results,
                test_result=test_result,
                passed=passed,
                repair_needed=not passed,
                elapsed_s=elapsed_iter,
            )
            session.iterations.append(record)

            if passed:
                if self.config.verbose:
                    print(f"  [build] PASS all checks passed on iteration {iteration}")
                session.success = True
                break

            if iteration < self.config.max_iterations:
                repair_prompt = self._build_repair_prompt(
                    task_text,
                    files_written,
                    syntax_errors,
                    execution_results,
                    command_results,
                    test_result,
                )
                record.repair_prompt = repair_prompt

                if self.config.verbose:
                    print(f"  [build] Requesting repair (iteration {iteration + 1})...")

                current_response = self.agent_fn(
                    repair_prompt,
                    self.config.tokens_per_iteration,
                )
                current_tool_calls = []
                current_tool_results = []
            else:
                if self.config.verbose:
                    print("  [build] FAIL max iterations reached without full pass")

        all_files: set[str] = set()
        for record in session.iterations:
            all_files.update(record.files_written)
        session.final_files = sorted(all_files)
        session.total_elapsed_s = time.perf_counter() - t_session
        return session

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_files(
        self,
        response: str,
        task_text: str,
        existing_tool_calls: List[FileOperation],
        existing_tool_results: List[OperationResult],
    ) -> Tuple[List[str], List[FileOperation]]:
        """Apply tool calls and fallback extraction, then return written paths."""

        written: List[str] = []
        applied_ops: List[FileOperation] = []

        executed_signatures = {
            self._op_signature(result.op)
            for result in existing_tool_results or []
            if result.success
        }
        for result in existing_tool_results or []:
            if result.success and result.op.tool in {"write_file", "edit_file"}:
                if result.op.path not in written:
                    written.append(result.op.path)
                applied_ops.append(result.op)

        parsed_tool_calls = parse_tool_calls(response or "")
        combined_tool_calls = list(existing_tool_calls or [])
        for op in parsed_tool_calls:
            if not self._has_matching_op(combined_tool_calls, op):
                combined_tool_calls.append(op)

        for op in combined_tool_calls:
            if op.tool not in {"write_file", "edit_file", "delete_file", "mkdir"}:
                continue
            if self._op_signature(op) in executed_signatures:
                continue

            result = self.executor.execute(op)
            if result.success:
                applied_ops.append(op)
                executed_signatures.add(self._op_signature(op))
                if op.tool in {"write_file", "edit_file"} and op.path not in written:
                    written.append(op.path)
                if op.tool == "delete_file" and op.path in written:
                    written.remove(op.path)

        extra_results = self.extractor.extract_and_write(
            response=response,
            task_text=task_text,
            existing_tool_calls=combined_tool_calls,
        )
        for result in extra_results:
            if result.success:
                applied_ops.append(result.op)
                if result.op.path not in written:
                    written.append(result.op.path)

        return written, applied_ops

    def _find_entry_points(self, files: List[str]) -> List[str]:
        """Find files that are safe to execute as entry points."""

        entry_points: List[str] = []
        test_re = _is_test_file_re()

        for path in files:
            if test_re(path):
                continue
            ext = Path(path).suffix.lower()
            if ext not in (".py", ".js", ".sh"):
                continue

            full = os.path.join(self.config.workspace_root, path)
            if not os.path.exists(full):
                continue

            if ext == ".py":
                try:
                    with open(full, encoding="utf-8", errors="replace") as handle:
                        content = handle.read()
                except Exception:
                    continue
                if "__name__" in content or "def main" in content:
                    entry_points.append(path)
                elif len(content.splitlines()) < 50:
                    entry_points.append(path)
            else:
                entry_points.append(path)

        return entry_points

    def _run_validation_commands(self, files_written: List[str]) -> List[ExecutionResult]:
        """Run build/test commands inferred from project manifests."""

        results: List[ExecutionResult] = []
        for spec in self._infer_validation_commands(files_written):
            result = self.runner.run_command(
                list(spec.command),
                cwd=spec.cwd,
                env=spec.env,
                timeout=self.config.timeout_per_run,
            )
            results.append(result)
            if self.config.verbose:
                status = "PASS" if result.success else "FAIL"
                print(
                    f"  [build]   cmd {spec.description}: {status} "
                    f"({result.elapsed_s:.2f}s)"
                )
                if not result.success:
                    print(f"  [build]     {result.error_summary[:200]}")
        return results

    def _infer_validation_commands(self, files_written: List[str]) -> List[ValidationCommand]:
        """Infer validation commands from nearby manifests for written files."""

        commands: List[ValidationCommand] = []
        seen: set[Tuple[Tuple[str, ...], str]] = set()

        for project_dir in self._find_manifest_dirs(files_written, "package.json"):
            package_json = Path(self.config.workspace_root) / project_dir / "package.json"
            try:
                package_data = json.loads(package_json.read_text(encoding="utf-8"))
            except Exception:
                continue
            scripts = package_data.get("scripts", {})
            for name, description in (
                ("typecheck", "npm run typecheck"),
                ("build", "npm run build"),
                ("test", "npm test"),
            ):
                if name not in scripts:
                    continue
                spec = ValidationCommand(
                    command=("npm", "run", name) if name != "test" else ("npm", "test"),
                    description=description,
                    cwd=project_dir,
                    env={"CI": "true"},
                )
                key = (spec.command, spec.cwd)
                if key not in seen:
                    seen.add(key)
                    commands.append(spec)

        for project_dir in self._find_manifest_dirs(files_written, "Cargo.toml"):
            spec = ValidationCommand(
                command=("cargo", "test"),
                description="cargo test",
                cwd=project_dir,
            )
            key = (spec.command, spec.cwd)
            if key not in seen:
                seen.add(key)
                commands.append(spec)

        for project_dir in self._find_manifest_dirs(files_written, "go.mod"):
            spec = ValidationCommand(
                command=("go", "test", "./..."),
                description="go test ./...",
                cwd=project_dir,
            )
            key = (spec.command, spec.cwd)
            if key not in seen:
                seen.add(key)
                commands.append(spec)

        for project_dir in self._find_manifest_dirs(files_written, "pom.xml"):
            spec = ValidationCommand(
                command=("mvn", "test"),
                description="mvn test",
                cwd=project_dir,
            )
            key = (spec.command, spec.cwd)
            if key not in seen:
                seen.add(key)
                commands.append(spec)

        return commands

    def _find_manifest_dirs(self, files_written: List[str], manifest_name: str) -> List[str]:
        """Find nearest directories containing the requested manifest."""

        root = Path(self.config.workspace_root).resolve()
        found: set[str] = set()

        for rel_path in files_written:
            candidate = (root / rel_path).resolve(strict=False)
            current = candidate.parent if candidate.suffix else candidate
            while True:
                manifest = current / manifest_name
                if manifest.exists():
                    found.add(os.path.relpath(current, root))
                    break
                if current == root:
                    break
                current = current.parent

        return sorted(found)

    def _assess_iteration(
        self,
        syntax_errors: List[tuple],
        exec_results: List[ExecutionResult],
        command_results: List[ExecutionResult],
        test_result: Optional[TestResult],
    ) -> bool:
        """Return True if this iteration is considered passing."""

        if syntax_errors:
            return False
        if any(not result.success for result in exec_results):
            return False
        if any(not result.success for result in command_results):
            return False
        if test_result is not None and not test_result.success:
            if test_result.framework != "none":
                return False
        return True

    def _related_test_paths(self, files_written: List[str]) -> List[str]:
        """Return tests that were generated for this build session."""

        if self.config.test_scope == "workspace":
            return self.runner._discover_tests()
        return [path for path in files_written if _is_test_file_re()(path)]

    def _build_repair_prompt(
        self,
        task_text: str,
        files_written: List[str],
        syntax_errors: List[tuple],
        exec_results: List[ExecutionResult],
        command_results: List[ExecutionResult],
        test_result: Optional[TestResult],
    ) -> str:
        """Build a focused repair prompt from the failure evidence."""

        lines = [
            "The following code was generated for this task:",
            f"TASK: {task_text}",
            "",
            f"FILES WRITTEN: {', '.join(files_written) or 'none'}",
            "",
            "FAILURES DETECTED - fix all of them:",
            "",
        ]

        if syntax_errors:
            lines.append("## Syntax Errors")
            for path, err in syntax_errors:
                lines.append(f"File: {path}")
                lines.append(f"Error: {err}")
                lines.append("")

        for result in exec_results:
            if not result.success:
                lines.append(f"## Runtime Error in {result.file_path}")
                lines.append(f"Command: {result.command}")
                lines.append("Error output:")
                lines.append(result.error_summary)
                lines.append("")

        for result in command_results:
            if not result.success:
                lines.append("## Validation Command Failure")
                lines.append(f"Command: {result.command}")
                lines.append("Error output:")
                lines.append(result.error_summary)
                lines.append("")

        if test_result and not test_result.success and test_result.framework != "none":
            lines.append(f"## Test Failures ({test_result.framework})")
            lines.append(
                f"Passed: {test_result.passed}  "
                f"Failed: {test_result.failed}  Errors: {test_result.errors}"
            )
            if test_result.failed_tests:
                lines.append(f"Failed tests: {', '.join(test_result.failed_tests)}")
            lines.append("Test output (last 40 lines):")
            lines.extend(test_result.output.splitlines()[-40:])
            lines.append("")

        lines.append("## Current File Contents")
        for path in files_written[:5]:
            full = os.path.join(self.config.workspace_root, path)
            if not os.path.exists(full):
                continue
            try:
                with open(full, encoding="utf-8", errors="replace") as handle:
                    content = handle.read()
            except Exception:
                continue
            lines.append(f"### {path}")
            lines.append("```" + Path(path).suffix.lstrip("."))
            lines.append(content[:3000])
            if len(content) > 3000:
                lines.append(f"... ({len(content) - 3000} more chars)")
            lines.append("```")
            lines.append("")

        lines.extend(
            [
                "## Instructions",
                "Fix ALL the errors above. Rewrite the complete corrected files.",
                "Use write_file tool calls for each file you change.",
                "Do not explain - just emit the corrected code.",
            ]
        )
        return "\n".join(lines)

    def _request_tests(self, task_text: str, source_files: List[str]) -> str:
        """Ask the agent to generate tests for the written files."""

        file_contents = []
        for path in source_files[:3]:
            full = os.path.join(self.config.workspace_root, path)
            if not os.path.exists(full):
                continue
            try:
                with open(full, encoding="utf-8", errors="replace") as handle:
                    content = handle.read()
            except Exception:
                continue
            file_contents.append(f"### {path}\n```python\n{content[:2000]}\n```")

        prompt = "\n".join(
            [
                f"Write pytest tests for the following code (task: {task_text}).",
                "",
                "Requirements:",
                "- Use pytest",
                "- Test the main functions/classes",
                "- Include edge cases",
                "- Save as test_<module>.py",
                "- Use write_file tool calls",
                "",
                "Source files:",
                "\n".join(file_contents),
            ]
        )
        return self.agent_fn(prompt, self.config.tokens_per_iteration)

    @staticmethod
    def _op_signature(op: FileOperation) -> Tuple[str, str, str, str, str]:
        return (
            op.tool,
            op.path,
            op.content,
            op.old_str,
            op.new_str,
        )

    def _has_matching_op(self, ops: List[FileOperation], candidate: FileOperation) -> bool:
        signature = self._op_signature(candidate)
        return any(self._op_signature(op) == signature for op in ops)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _is_test_file_re():
    """Return a function that checks if a path is a test file."""

    import re

    pattern = re.compile(r"(test_\w+\.py|_test\.py|\.(test|spec)\.(js|ts))$")
    return lambda path: bool(pattern.search(Path(path).name))
