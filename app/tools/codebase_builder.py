"""
Codebase builder — recursive write → run → test → fix loop.

Wraps the agent execution pipeline with:
  1. Write files (via CodeExtractor + tool_calls)
  2. Syntax check all written files
  3. Execute entry-point files to verify they run
  4. Run test suite
  5. If failures exist, build a repair prompt and re-invoke the agent
  6. Repeat until all tests pass OR max_iterations reached

Token budget: uses a large per-iteration budget (default 4000 tokens)
so the agent can produce complete modules, not just snippets.
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable

from app.tools.code_runner import CodeRunner, BuildReport, ExecutionResult, TestResult
from app.tools.filesystem import FilesystemExecutor, FileOperation, CodeExtractor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BuildConfig:
    """Configuration for a codebase build session."""
    workspace_root: str = "."
    max_iterations: int = 5
    tokens_per_iteration: int = 4000   # large budget for complete modules
    run_entry_points: bool = True       # execute main files after writing
    run_tests: bool = True              # run test suite after each iteration
    test_scope: str = "written"         # "written" or "workspace"
    auto_generate_tests: bool = True    # ask agent to write tests if none exist
    timeout_per_run: int = 30           # seconds per file execution
    verbose: bool = True


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
    success: bool = False
    total_elapsed_s: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Build session: {len(self.iterations)} iteration(s)",
            f"Final files: {', '.join(self.final_files) or 'none'}",
            f"Status: {'✓ PASSED' if self.success else '✗ FAILED'}",
            f"Total time: {self.total_elapsed_s:.1f}s",
        ]
        for rec in self.iterations:
            status = "✓" if rec.passed else "✗"
            lines.append(
                f"  Iter {rec.iteration}: {status}  "
                f"files={len(rec.files_written)}  "
                f"syntax_errors={len(rec.syntax_errors)}  "
                f"tests={'N/A' if rec.test_result is None else f'{rec.test_result.passed}P/{rec.test_result.failed}F'}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class CodebaseBuilder:
    """
    Orchestrates the recursive build loop.

    Usage:
        builder = CodebaseBuilder(config, agent_fn)
        session = builder.build(task_text, initial_response)
    """

    def __init__(
        self,
        config: BuildConfig,
        agent_fn: Callable[[str, int], str],
    ):
        """
        Args:
            config:    Build configuration.
            agent_fn:  Callable(prompt, max_tokens) -> response_text.
                       This is the LLM call — pass a lambda that calls
                       llm_client.generate().
        """
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
    ) -> BuildSession:
        """
        Run the full build loop starting from an initial agent response.

        Args:
            task_text:          The original user request.
            initial_response:   The first agent response (may contain code blocks).
            existing_tool_calls: Tool calls already executed by the agent.

        Returns:
            BuildSession with complete history.
        """
        session = BuildSession(task=task_text, workspace_root=self.config.workspace_root)
        t_session = time.perf_counter()

        current_response = initial_response
        current_tool_calls = existing_tool_calls or []

        for iteration in range(1, self.config.max_iterations + 1):
            t_iter = time.perf_counter()
            if self.config.verbose:
                print(f"\n  [build] Iteration {iteration}/{self.config.max_iterations}")

            # ── Step 1: Write files ──────────────────────────────────────────
            files_written = self._write_files(
                current_response, task_text, current_tool_calls
            )

            if self.config.verbose and files_written:
                for f in files_written:
                    print(f"  [build]   wrote: {f}")

            # ── Step 2: Syntax check ─────────────────────────────────────────
            py_files = [f for f in files_written if f.endswith(".py")]
            syntax_errors = self.runner.check_syntax_all(py_files)

            if self.config.verbose and syntax_errors:
                for path, err in syntax_errors:
                    print(f"  [build]   syntax error in {path}: {err[:80]}")

            # ── Step 3: Execute entry points ─────────────────────────────────
            exec_results: List[ExecutionResult] = []
            if self.config.run_entry_points and not syntax_errors:
                entry_points = self._find_entry_points(files_written)
                for ep in entry_points:
                    r = self.runner.run_file(ep)
                    exec_results.append(r)
                    if self.config.verbose:
                        status = "✓" if r.success else "✗"
                        print(f"  [build]   run {ep}: {status}  ({r.elapsed_s:.2f}s)")
                        if not r.success:
                            print(f"  [build]     {r.error_summary[:200]}")

            # ── Step 4: Generate tests if none exist ─────────────────────────
            related_tests = self._related_test_paths(files_written)
            if self.config.auto_generate_tests and iteration == 1:
                existing_tests = related_tests
                if not existing_tests and py_files:
                    test_response = self._request_tests(task_text, files_written)
                    test_files = self._write_files(test_response, task_text + " tests", [])
                    if self.config.verbose and test_files:
                        for f in test_files:
                            print(f"  [build]   wrote test: {f}")
                    files_written.extend(
                        path for path in test_files if path not in files_written
                    )
                    related_tests = self._related_test_paths(files_written)

            # ── Step 5: Run tests ────────────────────────────────────────────
            test_result: Optional[TestResult] = None
            if self.config.run_tests and not syntax_errors:
                if self.config.test_scope == "workspace":
                    test_paths = None
                else:
                    test_paths = related_tests
                test_result = self.runner.run_tests(test_paths=test_paths)
                if self.config.verbose:
                    if test_result.framework == "none":
                        print(f"  [build]   tests: no generated test files found")
                    else:
                        status = "✓" if test_result.success else "✗"
                        print(
                            f"  [build]   tests ({test_result.framework}): {status}  "
                            f"{test_result.passed}P / {test_result.failed}F / {test_result.errors}E"
                        )

            # ── Step 6: Assess pass/fail ─────────────────────────────────────
            passed = self._assess_iteration(syntax_errors, exec_results, test_result)
            elapsed_iter = time.perf_counter() - t_iter

            # Record iteration
            rec = IterationRecord(
                iteration=iteration,
                files_written=files_written,
                syntax_errors=syntax_errors,
                execution_results=exec_results,
                test_result=test_result,
                passed=passed,
                repair_needed=not passed,
                elapsed_s=elapsed_iter,
            )
            session.iterations.append(rec)

            if passed:
                if self.config.verbose:
                    print(f"  [build] ✓ All checks passed on iteration {iteration}")
                session.success = True
                break

            # ── Step 7: Build repair prompt and re-invoke agent ──────────────
            if iteration < self.config.max_iterations:
                repair_prompt = self._build_repair_prompt(
                    task_text, files_written, syntax_errors, exec_results, test_result
                )
                rec.repair_prompt = repair_prompt

                if self.config.verbose:
                    print(f"  [build] Requesting repair (iteration {iteration + 1})...")

                current_response = self.agent_fn(
                    repair_prompt, self.config.tokens_per_iteration
                )
                current_tool_calls = []  # fresh tool calls for repair
            else:
                if self.config.verbose:
                    print(f"  [build] ✗ Max iterations reached without full pass")

        # Collect all files written across all iterations
        all_files: set = set()
        for rec in session.iterations:
            all_files.update(rec.files_written)
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
    ) -> List[str]:
        """Write files from response. Returns list of written file paths."""
        written: List[str] = []

        # Already-executed tool calls (from streaming)
        for op in existing_tool_calls:
            if op.tool == "write_file" and op.path not in written:
                written.append(op.path)

        # Fallback: extract markdown code blocks
        extra = self.extractor.extract_and_write(
            response=response,
            task_text=task_text,
            existing_tool_calls=existing_tool_calls,
        )
        for r in extra:
            if r.success and r.op.path not in written:
                written.append(r.op.path)

        return written

    def _find_entry_points(self, files: List[str]) -> List[str]:
        """Find files that are safe to execute as entry points.

        Skips test files and files that import from packages not yet installed.
        Prefers files with `if __name__ == '__main__':` or `main()` calls.
        """
        entry_points = []
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

            # For Python: only run if it has a main guard or is a simple script
            if ext == ".py":
                try:
                    content = open(full, encoding="utf-8", errors="replace").read()
                    if '__name__' in content or 'def main' in content:
                        entry_points.append(path)
                    elif len(content.splitlines()) < 50:
                        # Short scripts are safe to run
                        entry_points.append(path)
                except Exception:
                    pass
            else:
                entry_points.append(path)

        return entry_points

    def _assess_iteration(
        self,
        syntax_errors: List[tuple],
        exec_results: List[ExecutionResult],
        test_result: Optional[TestResult],
    ) -> bool:
        """Return True if this iteration is considered passing."""
        if syntax_errors:
            return False
        if any(not r.success for r in exec_results):
            return False
        if test_result is not None and not test_result.success:
            # Allow pass if no tests were found (test_result.framework == "none")
            if test_result.framework != "none":
                return False
        return True

    def _related_test_paths(self, files_written: List[str]) -> List[str]:
        """Return tests that were generated for this build session.

        The builder should not treat unrelated repository test failures as
        failures of a standalone generated snippet.
        """

        if self.config.test_scope == "workspace":
            return self.runner._discover_tests()
        return [
            path
            for path in files_written
            if _is_test_file_re()(path)
        ]

    def _build_repair_prompt(
        self,
        task_text: str,
        files_written: List[str],
        syntax_errors: List[tuple],
        exec_results: List[ExecutionResult],
        test_result: Optional[TestResult],
    ) -> str:
        """Build a focused repair prompt from the failure evidence."""
        lines = [
            f"The following code was generated for this task:",
            f"TASK: {task_text}",
            "",
            f"FILES WRITTEN: {', '.join(files_written) or 'none'}",
            "",
            "FAILURES DETECTED — fix all of them:",
            "",
        ]

        if syntax_errors:
            lines.append("## Syntax Errors")
            for path, err in syntax_errors:
                lines.append(f"File: {path}")
                lines.append(f"Error: {err}")
                lines.append("")

        for r in exec_results:
            if not r.success:
                lines.append(f"## Runtime Error in {r.file_path}")
                lines.append(f"Command: {r.command}")
                lines.append(f"Error output:")
                lines.append(r.error_summary)
                lines.append("")

        if test_result and not test_result.success and test_result.framework != "none":
            lines.append(f"## Test Failures ({test_result.framework})")
            lines.append(f"Passed: {test_result.passed}  Failed: {test_result.failed}  Errors: {test_result.errors}")
            if test_result.failed_tests:
                lines.append(f"Failed tests: {', '.join(test_result.failed_tests)}")
            lines.append("Test output (last 40 lines):")
            output_lines = test_result.output.splitlines()
            lines.extend(output_lines[-40:])
            lines.append("")

        # Read current file contents for context
        lines.append("## Current File Contents")
        for path in files_written[:5]:  # limit to 5 files to keep prompt manageable
            full = os.path.join(self.config.workspace_root, path)
            if os.path.exists(full):
                try:
                    content = open(full, encoding="utf-8", errors="replace").read()
                    lines.append(f"### {path}")
                    lines.append("```" + Path(path).suffix.lstrip("."))
                    lines.append(content[:3000])  # cap at 3000 chars per file
                    if len(content) > 3000:
                        lines.append(f"... ({len(content) - 3000} more chars)")
                    lines.append("```")
                    lines.append("")
                except Exception:
                    pass

        lines.extend([
            "## Instructions",
            "Fix ALL the errors above. Rewrite the complete corrected files.",
            "Use write_file tool calls for each file you change.",
            "Do not explain — just emit the corrected code.",
        ])

        return "\n".join(lines)

    def _request_tests(self, task_text: str, source_files: List[str]) -> str:
        """Ask the agent to generate tests for the written files."""
        file_contents = []
        for path in source_files[:3]:
            full = os.path.join(self.config.workspace_root, path)
            if os.path.exists(full):
                try:
                    content = open(full, encoding="utf-8", errors="replace").read()
                    file_contents.append(f"### {path}\n```python\n{content[:2000]}\n```")
                except Exception:
                    pass

        prompt = "\n".join([
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
        ])

        return self.agent_fn(prompt, self.config.tokens_per_iteration)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _is_test_file_re():
    """Return a function that checks if a path is a test file."""
    import re
    pattern = re.compile(r"(test_\w+\.py|_test\.py|\.(test|spec)\.(js|ts))$")
    return lambda path: bool(pattern.search(Path(path).name))
