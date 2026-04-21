"""
File reader with image support, fuzzy filename resolution, and binary detection.

Handles:
- Text files (source code, markdown, config, etc.)
- Images (PNG, JPG, GIF, WebP, BMP, SVG) — base64 encoded + metadata
- Binary files — detected and skipped gracefully
- Fuzzy filename matching — "main" resolves to "src/main.py"
- Glob patterns — "*.py" expands to all Python files
- Directory scanning with .gitignore awareness
"""

import base64
import fnmatch
import mimetypes
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FileEntry:
    """A single file loaded into context."""
    path: str           # relative path from workspace root
    abs_path: str       # absolute path
    content: str        # text content (empty for images)
    image_b64: str      # base64 image data (empty for text)
    mime_type: str      # e.g. "text/x-python", "image/png"
    size_bytes: int
    is_image: bool
    is_binary: bool
    truncated: bool = False
    error: str = ""

    @property
    def display_name(self) -> str:
        return self.path

    @property
    def context_block(self) -> str:
        """Format for injection into an LLM prompt."""
        if self.error:
            return f"[{self.path}] ERROR: {self.error}\n"
        if self.is_binary and not self.is_image:
            return f"[{self.path}] (binary file, {self.size_bytes} bytes — content not shown)\n"
        if self.is_image:
            return (
                f"[{self.path}] IMAGE ({self.mime_type}, {self.size_bytes} bytes)\n"
                f"Base64: {self.image_b64[:80]}...\n"
            )
        lang = _ext_to_lang(self.path)
        trunc = " [truncated]" if self.truncated else ""
        return f"### {self.path}{trunc}\n```{lang}\n{self.content}\n```\n"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".tiff", ".tif"}
TEXT_EXTS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs",
    ".html", ".htm", ".css", ".scss", ".sass", ".less",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".md", ".rst", ".txt", ".csv", ".tsv",
    ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd",
    ".sql", ".graphql", ".gql",
    ".rs", ".go", ".java", ".kt", ".swift", ".rb", ".php",
    ".c", ".cpp", ".cc", ".h", ".hpp",
    ".cs", ".fs", ".fsx",
    ".r", ".R", ".jl", ".lua", ".pl", ".pm",
    ".xml", ".svg", ".env", ".gitignore", ".dockerignore",
    ".makefile", ".mk", "", ".lock",
}

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    ".env", "dist", "build", ".next", ".nuxt", "target", "out",
    ".idea", ".vscode", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "coverage", ".coverage", "htmlcov", "eggs", ".eggs",
}

MAX_TEXT_BYTES = 100_000   # 100 KB per file
MAX_IMAGE_BYTES = 5_000_000  # 5 MB per image
MAX_TOTAL_BYTES = 500_000  # 500 KB total context


def _ext_to_lang(path: str) -> str:
    ext = Path(path).suffix.lower()
    return {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".jsx": "jsx", ".tsx": "tsx", ".html": "html", ".css": "css",
        ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
        ".sh": "bash", ".bash": "bash", ".sql": "sql", ".rs": "rust",
        ".go": "go", ".java": "java", ".kt": "kotlin", ".rb": "ruby",
        ".c": "c", ".cpp": "cpp", ".cs": "csharp", ".md": "markdown",
        ".xml": "xml", ".svg": "xml", ".r": "r", ".lua": "lua",
    }.get(ext, "")


def _is_binary(data: bytes) -> bool:
    """Heuristic: file is binary if it contains null bytes in first 8KB."""
    return b"\x00" in data[:8192]


def _load_gitignore(workspace_root: str) -> List[str]:
    patterns = []
    for name in (".gitignore", ".dockerignore"):
        p = os.path.join(workspace_root, name)
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            patterns.append(line)
            except Exception:
                pass
    return patterns


def _matches_ignore(rel_path: str, patterns: List[str]) -> bool:
    name = os.path.basename(rel_path)
    for pat in patterns:
        if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(rel_path, pat):
            return True
    return False


# ---------------------------------------------------------------------------
# Core reader
# ---------------------------------------------------------------------------

class FileReader:
    """
    Load files into context with fuzzy resolution, image support,
    and total-size budgeting.
    """

    def __init__(self, workspace_root: str = "."):
        self.workspace_root = os.path.abspath(workspace_root)
        self._ignore = _load_gitignore(self.workspace_root)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self, path_or_name: str) -> FileEntry:
        """
        Read a single file. Resolves:
        - Absolute paths
        - Relative paths from workspace root
        - Bare filenames (fuzzy match against workspace)
        - Glob patterns (returns first match)
        """
        resolved = self.resolve(path_or_name)
        if resolved is None:
            return FileEntry(
                path=path_or_name, abs_path="", content="", image_b64="",
                mime_type="", size_bytes=0, is_image=False, is_binary=False,
                error=f"File not found: {path_or_name!r}",
            )
        return self._load_file(resolved)

    def read_many(self, paths: List[str]) -> List[FileEntry]:
        """Read multiple files."""
        return [self.read(p) for p in paths]

    def scan_directory(
        self,
        path: str = ".",
        max_files: int = 50,
        extensions: Optional[List[str]] = None,
    ) -> List[FileEntry]:
        """
        Scan a directory and return all readable files up to max_files.
        Respects .gitignore, skips binary files and large files.
        """
        root = os.path.abspath(os.path.join(self.workspace_root, path))
        if not os.path.isdir(root):
            # Maybe it's a file
            if os.path.isfile(root):
                return [self._load_file(root)]
            return []

        entries: List[FileEntry] = []
        total_bytes = 0

        for dirpath, dirnames, filenames in os.walk(root):
            # Prune skip dirs in-place
            dirnames[:] = [
                d for d in sorted(dirnames)
                if d not in SKIP_DIRS and not d.startswith(".")
            ]

            rel_dir = os.path.relpath(dirpath, self.workspace_root)

            for fname in sorted(filenames):
                if len(entries) >= max_files:
                    break
                if total_bytes >= MAX_TOTAL_BYTES:
                    break

                rel_path = os.path.normpath(os.path.join(rel_dir, fname))
                if _matches_ignore(rel_path, self._ignore):
                    continue

                ext = Path(fname).suffix.lower()
                if extensions and ext not in extensions:
                    continue

                abs_path = os.path.join(dirpath, fname)
                entry = self._load_file(abs_path, rel_path)
                if not entry.error and not entry.is_binary:
                    entries.append(entry)
                    total_bytes += entry.size_bytes

        return entries

    def resolve(self, name: str) -> Optional[str]:
        """
        Resolve a name to an absolute path.

        Resolution order:
        1. Absolute path that exists
        2. Relative to cwd
        3. Relative to workspace root
        4. Glob expansion (first match)
        5. Fuzzy: basename match anywhere in workspace
        6. Fuzzy: partial name match (stem contains name)
        """
        # 1. Absolute
        if os.path.isabs(name) and os.path.exists(name):
            return name

        # 2. Relative to cwd
        cwd_path = os.path.abspath(os.path.join(os.getcwd(), name))
        if os.path.exists(cwd_path):
            return cwd_path

        # 3. Relative to workspace root
        ws_path = os.path.abspath(os.path.join(self.workspace_root, name))
        if os.path.exists(ws_path):
            return ws_path

        # 4. Glob
        import glob
        for pattern in (
            os.path.join(os.getcwd(), name),
            os.path.join(self.workspace_root, name),
            os.path.join(self.workspace_root, "**", name),
        ):
            matches = glob.glob(pattern, recursive=True)
            if matches:
                return os.path.abspath(matches[0])

        # 5 & 6. Fuzzy search
        return self._fuzzy_resolve(name)

    def fuzzy_candidates(self, name: str, limit: int = 5) -> List[str]:
        """Return up to `limit` candidate paths that fuzzy-match `name`."""
        name_lower = name.lower()
        stem = Path(name).stem.lower()
        ext = Path(name).suffix.lower()

        exact: List[str] = []
        partial: List[str] = []

        for dirpath, dirnames, filenames in os.walk(self.workspace_root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
            for fname in filenames:
                fname_lower = fname.lower()
                fname_stem = Path(fname).stem.lower()
                rel = os.path.relpath(os.path.join(dirpath, fname), self.workspace_root)

                # Exact basename match
                if fname_lower == name_lower:
                    exact.append(rel)
                # Stem match with optional extension
                elif fname_stem == stem and (not ext or Path(fname).suffix.lower() == ext):
                    exact.append(rel)
                # Partial: name is contained in filename
                elif name_lower in fname_lower or fname_stem.startswith(stem):
                    partial.append(rel)

        results = exact + partial
        return results[:limit]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fuzzy_resolve(self, name: str) -> Optional[str]:
        candidates = self.fuzzy_candidates(name, limit=1)
        if candidates:
            return os.path.join(self.workspace_root, candidates[0])
        return None

    def _load_file(self, abs_path: str, rel_path: str = None) -> FileEntry:
        if rel_path is None:
            try:
                rel_path = os.path.relpath(abs_path, self.workspace_root)
            except ValueError:
                rel_path = abs_path

        ext = Path(abs_path).suffix.lower()
        mime = mimetypes.guess_type(abs_path)[0] or "application/octet-stream"

        try:
            size = os.path.getsize(abs_path)
        except OSError as e:
            return FileEntry(
                path=rel_path, abs_path=abs_path, content="", image_b64="",
                mime_type=mime, size_bytes=0, is_image=False, is_binary=False,
                error=str(e),
            )

        # ── Image ────────────────────────────────────────────────────────────
        if ext in IMAGE_EXTS:
            if size > MAX_IMAGE_BYTES:
                return FileEntry(
                    path=rel_path, abs_path=abs_path, content="", image_b64="",
                    mime_type=mime, size_bytes=size, is_image=True, is_binary=True,
                    error=f"Image too large ({size // 1024} KB > {MAX_IMAGE_BYTES // 1024} KB limit)",
                )
            try:
                with open(abs_path, "rb") as f:
                    data = f.read()
                b64 = base64.b64encode(data).decode("ascii")
                return FileEntry(
                    path=rel_path, abs_path=abs_path, content="", image_b64=b64,
                    mime_type=mime, size_bytes=size, is_image=True, is_binary=True,
                )
            except Exception as e:
                return FileEntry(
                    path=rel_path, abs_path=abs_path, content="", image_b64="",
                    mime_type=mime, size_bytes=size, is_image=True, is_binary=True,
                    error=str(e),
                )

        # ── Text ─────────────────────────────────────────────────────────────
        try:
            with open(abs_path, "rb") as f:
                raw = f.read(min(size, MAX_TEXT_BYTES + 1))

            if _is_binary(raw):
                return FileEntry(
                    path=rel_path, abs_path=abs_path, content="", image_b64="",
                    mime_type=mime, size_bytes=size, is_image=False, is_binary=True,
                )

            truncated = len(raw) > MAX_TEXT_BYTES
            text = raw[:MAX_TEXT_BYTES].decode("utf-8", errors="replace")
            return FileEntry(
                path=rel_path, abs_path=abs_path, content=text, image_b64="",
                mime_type=mime, size_bytes=size, is_image=False, is_binary=False,
                truncated=truncated,
            )
        except Exception as e:
            return FileEntry(
                path=rel_path, abs_path=abs_path, content="", image_b64="",
                mime_type=mime, size_bytes=size, is_image=False, is_binary=False,
                error=str(e),
            )


# ---------------------------------------------------------------------------
# Context builder — assembles FileEntry list into a prompt-ready string
# ---------------------------------------------------------------------------

def build_context_block(entries: List[FileEntry], title: str = "FILE CONTEXT") -> str:
    """Format a list of FileEntry objects into a single context block for LLM injection."""
    if not entries:
        return ""

    lines = [f"=== {title} ===\n"]
    total = 0
    for e in entries:
        if e.error:
            lines.append(f"[{e.path}] ⚠ {e.error}\n")
            continue
        if e.is_binary and not e.is_image:
            lines.append(f"[{e.path}] (binary, {e.size_bytes} bytes)\n")
            continue
        if e.is_image:
            lines.append(
                f"[{e.path}] IMAGE {e.mime_type} {e.size_bytes} bytes\n"
                f"  data:image/{e.mime_type.split('/')[-1]};base64,{e.image_b64[:60]}...\n"
            )
            continue
        lines.append(e.context_block)
        total += e.size_bytes

    lines.append(f"\n=== END {title} ({len(entries)} file(s), ~{total // 1024} KB) ===\n")
    return "\n".join(lines)
