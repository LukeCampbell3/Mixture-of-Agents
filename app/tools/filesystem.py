"""
Filesystem tool executor for the agentic network.

Agents emit tool calls in their response using a structured block:
    <tool_call>
    {"tool": "write_file", "path": "src/foo.py", "content": "..."}
    </tool_call>

This module parses those blocks and executes the operations safely.
"""

import os
import re
import json
import difflib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class FileOperation:
    """A single file operation parsed from an agent response."""

    VALID_TOOLS = {"write_file", "edit_file", "delete_file", "mkdir", "read_file"}

    def __init__(self, tool: str, path: str, content: str = "",
                 old_str: str = "", new_str: str = "", description: str = ""):
        self.tool = tool
        self.path = path
        self.content = content      # for write_file
        self.old_str = old_str      # for edit_file (patch)
        self.new_str = new_str      # for edit_file (patch)
        self.description = description

    def __repr__(self):
        return f"FileOperation({self.tool}, {self.path!r})"


class OperationResult:
    def __init__(self, op: FileOperation, success: bool,
                 message: str, diff: str = ""):
        self.op = op
        self.success = success
        self.message = message
        self.diff = diff


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(
    r'<tool_call>\s*(.*?)\s*</tool_call>',
    re.DOTALL | re.IGNORECASE
)

# Also accept markdown-fenced JSON blocks labelled tool_call
_FENCED_RE = re.compile(
    r'```(?:tool_call|json)?\s*\n(\{.*?\})\s*\n```',
    re.DOTALL | re.IGNORECASE
)


def parse_tool_calls(response_text: str) -> List[FileOperation]:
    """Extract all tool calls from an agent response string."""
    ops: List[FileOperation] = []

    # Try XML-style blocks first
    for m in _TOOL_CALL_RE.finditer(response_text):
        op = _parse_json_block(m.group(1))
        if op:
            ops.append(op)

    # Fall back to fenced blocks if nothing found
    if not ops:
        for m in _FENCED_RE.finditer(response_text):
            op = _parse_json_block(m.group(1))
            if op:
                ops.append(op)

    return ops


def _parse_json_block(text: str) -> Optional[FileOperation]:
    """Parse a single JSON tool call block."""
    try:
        data = json.loads(text.strip())
    except json.JSONDecodeError:
        return None

    tool = data.get("tool", "")
    if tool not in FileOperation.VALID_TOOLS:
        return None

    path = data.get("path", "").strip()
    if not path:
        return None

    return FileOperation(
        tool=tool,
        path=path,
        content=data.get("content", ""),
        old_str=data.get("old_str", ""),
        new_str=data.get("new_str", ""),
        description=data.get("description", ""),
    )


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class FilesystemExecutor:
    """Execute file operations safely within a workspace root."""

    def __init__(self, workspace_root: str = "."):
        self.workspace_root = os.path.abspath(workspace_root)

    def resolve(self, path: str) -> str:
        """Resolve a relative path to an absolute path inside the workspace."""
        full = os.path.abspath(os.path.join(self.workspace_root, path))
        # Safety: must stay inside workspace
        if not full.startswith(self.workspace_root):
            raise ValueError(f"Path escapes workspace: {path!r}")
        return full

    def preview(self, op: FileOperation) -> str:
        """Return a human-readable diff/preview of what the operation will do."""
        try:
            full = self.resolve(op.path)
        except ValueError as e:
            return f"[ERROR] {e}"

        if op.tool == "write_file":
            if os.path.exists(full):
                old = open(full, encoding="utf-8", errors="replace").read()
                return _unified_diff(old, op.content, op.path)
            else:
                lines = op.content.splitlines(keepends=True)
                diff = f"--- /dev/null\n+++ {op.path}\n"
                diff += "".join(f"+{l}" for l in lines[:60])
                if len(lines) > 60:
                    diff += f"\n... (+{len(lines)-60} more lines)"
                return diff

        elif op.tool == "edit_file":
            if not os.path.exists(full):
                return f"[ERROR] File does not exist: {op.path}"
            old = open(full, encoding="utf-8", errors="replace").read()
            if op.old_str not in old:
                return f"[ERROR] old_str not found in {op.path}"
            new_content = old.replace(op.old_str, op.new_str, 1)
            return _unified_diff(old, new_content, op.path)

        elif op.tool == "delete_file":
            return f"[DELETE] {op.path}"

        elif op.tool == "mkdir":
            return f"[MKDIR] {op.path}"

        elif op.tool == "read_file":
            if os.path.exists(full):
                return f"[READ] {op.path}"
            return f"[ERROR] File not found: {op.path}"

        return f"[{op.tool}] {op.path}"

    def execute(self, op: FileOperation) -> OperationResult:
        """Execute a single file operation. Returns result with success/error."""
        try:
            full = self.resolve(op.path)
        except ValueError as e:
            return OperationResult(op, False, str(e))

        try:
            if op.tool == "write_file":
                return self._write(op, full)
            elif op.tool == "edit_file":
                return self._edit(op, full)
            elif op.tool == "delete_file":
                return self._delete(op, full)
            elif op.tool == "mkdir":
                return self._mkdir(op, full)
            elif op.tool == "read_file":
                return self._read(op, full)
            else:
                return OperationResult(op, False, f"Unknown tool: {op.tool}")
        except Exception as e:
            return OperationResult(op, False, f"Exception: {e}")

    def execute_all(self, ops: List[FileOperation]) -> List[OperationResult]:
        return [self.execute(op) for op in ops]

    # ------------------------------------------------------------------
    # Individual operations
    # ------------------------------------------------------------------

    def _write(self, op: FileOperation, full: str) -> OperationResult:
        old = ""
        if os.path.exists(full):
            old = open(full, encoding="utf-8", errors="replace").read()
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(op.content)
        diff = _unified_diff(old, op.content, op.path)
        verb = "Updated" if old else "Created"
        return OperationResult(op, True, f"{verb}: {op.path}", diff)

    def _edit(self, op: FileOperation, full: str) -> OperationResult:
        if not os.path.exists(full):
            return OperationResult(op, False, f"File not found: {op.path}")
        old = open(full, encoding="utf-8", errors="replace").read()
        if op.old_str not in old:
            return OperationResult(op, False,
                f"old_str not found in {op.path}. No changes made.")
        new_content = old.replace(op.old_str, op.new_str, 1)
        diff = _unified_diff(old, new_content, op.path)
        with open(full, "w", encoding="utf-8") as f:
            f.write(new_content)
        return OperationResult(op, True, f"Edited: {op.path}", diff)

    def _delete(self, op: FileOperation, full: str) -> OperationResult:
        if not os.path.exists(full):
            return OperationResult(op, False, f"File not found: {op.path}")
        os.remove(full)
        return OperationResult(op, True, f"Deleted: {op.path}")

    def _mkdir(self, op: FileOperation, full: str) -> OperationResult:
        os.makedirs(full, exist_ok=True)
        return OperationResult(op, True, f"Directory created: {op.path}")

    def _read(self, op: FileOperation, full: str) -> OperationResult:
        if not os.path.exists(full):
            return OperationResult(op, False, f"File not found: {op.path}")
        content = open(full, encoding="utf-8", errors="replace").read()
        return OperationResult(op, True, f"Read: {op.path}",
                               diff=content[:2000])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unified_diff(old: str, new: str, path: str) -> str:
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = list(difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{path}", tofile=f"b/{path}",
        lineterm=""
    ))
    if not diff:
        return "(no changes)"
    # Cap at 80 lines for display
    if len(diff) > 80:
        diff = diff[:80] + [f"... ({len(diff)-80} more lines)"]
    return "\n".join(diff)
