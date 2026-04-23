"""Regression tests for the build and file-operation pipeline."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from app.agents.code_primary import CodePrimaryAgent
from app.schemas.task_frame import TaskFrame, TaskType
from app.tools.code_runner import CodeRunner
from app.tools.codebase_builder import BuildConfig, CodebaseBuilder
from app.tools.filesystem import FileOperation, FilesystemExecutor


class StubLLM:
    """Minimal stub LLM for agent tests."""

    def __init__(self, response: str):
        self.response = response

    def generate(self, prompt, max_tokens=800, temperature=0.7):
        return self.response


class BuildPipelineTests(unittest.TestCase):
    def test_codebase_builder_executes_tool_calls_from_response(self):
        """Tool calls embedded in the response should be applied before validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = CodebaseBuilder(
                config=BuildConfig(
                    workspace_root=tmpdir,
                    max_iterations=1,
                    run_entry_points=False,
                    run_tests=False,
                    run_validation_commands=False,
                    auto_generate_tests=False,
                    verbose=False,
                ),
                agent_fn=lambda prompt, max_tok: "",
            )

            response = (
                '<tool_call>{"tool": "write_file", "path": "hello.py", '
                '"content": "print(\\"ok\\")\\n"}</tool_call>'
            )
            session = builder.build(
                task_text="write hello",
                initial_response=response,
                existing_tool_calls=[],
                existing_tool_results=[],
            )

            self.assertEqual(
                Path(tmpdir, "hello.py").read_text(encoding="utf-8"),
                'print("ok")\n',
            )
            self.assertEqual(session.final_files, ["hello.py"])

    def test_code_primary_ignores_tool_calls_for_explanatory_prompt(self):
        """Explanatory prompts should not create durable file operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            llm = StubLLM(
                '<tool_call>{"tool": "write_file", "path": "linked_list.py", '
                '"content": "class Node:\\n    pass\\n"}</tool_call>\n'
                "```python\nclass Node:\n    pass\n```"
            )
            agent = CodePrimaryAgent(
                agent_id="code_primary",
                name="Code Primary",
                description="Primary coding agent",
                llm_client=llm,
                tools=["repo_tool"],
            )
            task_frame = TaskFrame(
                task_id="task-1",
                normalized_request="How would you implement a doubly linked list in python?",
                task_type=TaskType.CODING_STABLE,
            )

            result = agent.execute(
                {
                    "task_frame": task_frame,
                    "shared_context": "",
                    "workspace_root": tmpdir,
                }
            )

            self.assertEqual(result["tool_calls"], [])
            self.assertFalse(Path(tmpdir, "linked_list.py").exists())

    def test_filesystem_executor_rejects_escape_and_strict_batch_edit(self):
        """Path validation and batch edit behavior should fail safely."""
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = FilesystemExecutor(workspace_root=tmpdir)

            with self.assertRaises(ValueError):
                executor.resolve("../escape.txt")

            with self.assertRaises(ValueError):
                executor.resolve(str(Path(tmpdir) / "absolute.txt"))

            target = Path(tmpdir, "note.txt")
            target.write_text("hello\n", encoding="utf-8")

            results = executor.execute_all(
                [
                    FileOperation(
                        tool="edit_file",
                        path="note.txt",
                        old_str="missing",
                        new_str="updated",
                    )
                ],
                batch_mode=True,
            )

            self.assertFalse(results[0].success)
            self.assertEqual(target.read_text(encoding="utf-8"), "hello\n")

    def test_builder_infers_package_commands(self):
        """Manifest-driven command inference should pick up build/test scripts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = Path(tmpdir, "src")
            src_dir.mkdir()
            Path(src_dir, "app.ts").write_text(
                "export const ok = true;\n",
                encoding="utf-8",
            )
            Path(tmpdir, "package.json").write_text(
                '{"scripts":{"build":"tsc -p .","test":"vitest run","typecheck":"tsc --noEmit"}}',
                encoding="utf-8",
            )

            builder = CodebaseBuilder(
                config=BuildConfig(
                    workspace_root=tmpdir,
                    max_iterations=1,
                    run_entry_points=False,
                    run_tests=False,
                    auto_generate_tests=False,
                    verbose=False,
                ),
                agent_fn=lambda prompt, max_tok: "",
            )

            commands = builder._infer_validation_commands(["src/app.ts"])
            inferred = {tuple(spec.command) for spec in commands}

            self.assertIn(("npm", "run", "typecheck"), inferred)
            self.assertIn(("npm", "run", "build"), inferred)
            self.assertIn(("npm", "test"), inferred)

    def test_generated_tests_persist_across_repair_iterations(self):
        """Generated tests should continue running during later repair iterations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            def agent_fn(prompt, max_tok):
                if "tests for the following code" in prompt:
                    payload = {
                        "tool": "write_file",
                        "path": "test_bad.py",
                        "content": (
                            "import unittest\n\n"
                            "from bad import add\n\n"
                            "class AddTests(unittest.TestCase):\n"
                            "    def test_add(self):\n"
                            "        self.assertEqual(add(1, 2), 4)\n"
                        ),
                    }
                    return f'<tool_call>{json.dumps(payload)}</tool_call>'
                payload = {
                    "tool": "write_file",
                    "path": "bad.py",
                    "content": "def add(a, b):\n    return a + b\n",
                }
                return f'<tool_call>{json.dumps(payload)}</tool_call>'

            builder = CodebaseBuilder(
                config=BuildConfig(
                    workspace_root=tmpdir,
                    max_iterations=2,
                    run_entry_points=False,
                    run_tests=True,
                    run_validation_commands=False,
                    auto_generate_tests=True,
                    verbose=False,
                ),
                agent_fn=agent_fn,
            )

            session = builder.build(
                task_text="create add function",
                initial_response=agent_fn("initial", 0),
                existing_tool_calls=[],
                existing_tool_results=[],
            )

            self.assertEqual(len(session.iterations), 2)
            self.assertIsNotNone(session.iterations[0].test_result)
            self.assertIsNotNone(session.iterations[1].test_result)
            self.assertNotEqual(session.iterations[1].test_result.framework, "none")
            self.assertFalse(session.success)

    def test_request_tests_uses_unittest_when_pytest_is_unavailable(self):
        """Generated test prompts should match the framework the runner can execute."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir, "model.py")
            source.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

            builder = CodebaseBuilder(
                config=BuildConfig(
                    workspace_root=tmpdir,
                    max_iterations=1,
                    run_entry_points=False,
                    run_tests=False,
                    auto_generate_tests=True,
                    verbose=False,
                ),
                agent_fn=lambda prompt, max_tok: prompt,
            )

            with patch.object(builder.runner, "_has_pytest", return_value=False):
                prompt = builder._request_tests("create add", ["model.py"])

            self.assertIn("Write unittest tests", prompt)
            self.assertIn("Use unittest from the Python standard library", prompt)
            self.assertNotIn("Use pytest", prompt)

    def test_code_runner_pytest_detection_checks_return_code(self):
        """A missing pytest install should not be treated as available."""
        runner = CodeRunner(workspace_root=".")
        with patch("subprocess.run", return_value=Mock(returncode=1)):
            self.assertFalse(runner._has_pytest())


if __name__ == "__main__":
    unittest.main()
