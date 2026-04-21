"""
Filesystem tool executor for the agentic network.

Agents emit tool calls in their response using a structured block:
    <tool_call>
    {"tool": "write_file", "path": "src/foo.py", "content": "..."}
    </tool_call>

This module parses those blocks and executes the operations safely.
Streaming execution is supported: call stream_execute() to parse and
run operations as they arrive rather than waiting for the full response.
"""

import os
import re
import json
import difflib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Iterator


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


def stream_parse_tool_calls(token_iterator: Iterator[str]) -> Iterator[FileOperation]:
    """
    Parse tool calls from a streaming token iterator.

    Yields FileOperation objects as soon as each complete <tool_call>...</tool_call>
    block is received, without waiting for the full response.

    Single-threaded: consumes the iterator directly, buffering only what's
    needed to detect the open/close tags.

    Usage:
        for op in stream_parse_tool_calls(ollama_token_stream):
            executor.execute(op)
    """
    OPEN  = "<tool_call>"
    CLOSE = "</tool_call>"
    buf = ""

    for token in token_iterator:
        buf += token

        # Process as many complete blocks as are present in the buffer
        while True:
            lo = buf.lower().find(OPEN)
            if lo == -1:
                # No open tag yet — keep only a tail in case the tag spans tokens
                buf = buf[-(len(OPEN) - 1):]
                break

            lc = buf.lower().find(CLOSE, lo + len(OPEN))
            if lc == -1:
                # Open tag found but close tag not yet arrived — keep from open tag
                buf = buf[lo:]
                break

            # Complete block found
            inner = buf[lo + len(OPEN): lc].strip()
            buf = buf[lc + len(CLOSE):]   # advance past the close tag

            op = _parse_json_block(inner)
            if op:
                yield op


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
                # Only read file if it's small (< 100KB) to avoid slow previews
                file_size = os.path.getsize(full)
                if file_size > 100 * 1024:
                    return f"[WRITE] {op.path} ({file_size // 1024} KB - skipping large file preview)"
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
            # Only read file if it's small (< 100KB)
            file_size = os.path.getsize(full)
            if file_size > 100 * 1024:
                return f"[EDIT] {op.path} ({file_size // 1024} KB - skipping large file preview)"
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

    def execute_all(self, ops: List[FileOperation], batch_mode: bool = False) -> List[OperationResult]:
        """Execute multiple file operations.
        
        Args:
            ops: List of operations to execute
            batch_mode: If True, skip individual result collection for speed
        """
        if batch_mode:
            # Fast path - just execute without collecting detailed results
            results = []
            for op in ops:
                try:
                    full = self.resolve(op.path)
                    if op.tool == "write_file":
                        self._write_fast(op, full)
                    elif op.tool == "edit_file":
                        self._edit_fast(op, full)
                    elif op.tool == "delete_file":
                        self._delete_fast(op, full)
                    elif op.tool == "mkdir":
                        self._mkdir_fast(op, full)
                    results.append(OperationResult(op, True, f"Executed: {op.path}"))
                except Exception as e:
                    results.append(OperationResult(op, False, f"Exception: {e}"))
            return results
        else:
            return [self.execute(op) for op in ops]

    # Fast execution methods (no diff generation)
    def _write_fast(self, op: FileOperation, full: str) -> None:
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(op.content)

    def _edit_fast(self, op: FileOperation, full: str) -> None:
        with open(full, "r", encoding="utf-8") as f:
            old = f.read()
        new_content = old.replace(op.old_str, op.new_str, 1)
        with open(full, "w", encoding="utf-8") as f:
            f.write(new_content)

    def _delete_fast(self, op: FileOperation, full: str) -> None:
        os.remove(full)

    def _mkdir_fast(self, op: FileOperation, full: str) -> None:
        os.makedirs(full, exist_ok=True)

    # ------------------------------------------------------------------
    # Individual operations
    # ------------------------------------------------------------------

    def _write(self, op: FileOperation, full: str) -> OperationResult:
        # Only read existing file if it's small (< 100KB) for diff generation
        old = ""
        if os.path.exists(full):
            file_size = os.path.getsize(full)
            if file_size <= 100 * 1024:
                old = open(full, encoding="utf-8", errors="replace").read()
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(op.content)
        # Only generate diff if we have old content
        diff = ""
        if old:
            diff = _unified_diff(old, op.content, op.path)
        verb = "Updated" if old else "Created"
        return OperationResult(op, True, f"{verb}: {op.path}", diff)

    def _edit(self, op: FileOperation, full: str) -> OperationResult:
        if not os.path.exists(full):
            return OperationResult(op, False, f"File not found: {op.path}")
        # Only read file if it's small (< 100KB)
        file_size = os.path.getsize(full)
        if file_size > 100 * 1024:
            return OperationResult(op, False, f"File too large for edit: {op.path} ({file_size // 1024} KB)")
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
        file_size = os.path.getsize(full)
        if file_size > 100 * 1024:
            return OperationResult(op, True, f"Read: {op.path} ({file_size // 1024} KB - large file, content truncated)",
                                   diff=f"[File too large to display: {file_size // 1024} KB]")
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
