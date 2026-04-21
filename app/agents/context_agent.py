"""
Context Builder Agent — scans the workspace, resolves files, and builds
a rich context block that gets injected into every subsequent agent prompt.

Responsibilities:
1. Scan the current directory for relevant files
2. Fuzzy-resolve filenames mentioned in user input
3. Read text files and images
4. Summarize large codebases (file tree + key file contents)
5. Detect the project type (Python, Node, Rust, etc.)
6. Return a structured context block ready for LLM injection
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from app.tools.file_reader import FileReader, FileEntry, build_context_block, SKIP_DIRS


# ---------------------------------------------------------------------------
# Project detection
# ---------------------------------------------------------------------------

_PROJECT_MARKERS: Dict[str, List[str]] = {
    "python":     ["requirements.txt", "setup.py", "pyproject.toml", "Pipfile"],
    "node":       ["package.json", "yarn.lock", "pnpm-lock.yaml"],
    "rust":       ["Cargo.toml"],
    "go":         ["go.mod"],
    "java":       ["pom.xml", "build.gradle"],
    "dotnet":     ["*.csproj", "*.sln"],
    "ruby":       ["Gemfile"],
    "php":        ["composer.json"],
    "docker":     ["Dockerfile", "docker-compose.yml", "docker-compose.yaml"],
    "terraform":  ["main.tf", "*.tf"],
}

# Files that are always useful to include if present
_PRIORITY_FILES = [
    "README.md", "README.rst", "README.txt",
    "requirements.txt", "pyproject.toml", "setup.py",
    "package.json", "Cargo.toml", "go.mod",
    "Makefile", "Dockerfile", "docker-compose.yml",
    ".env.example",
]


def detect_project_type(workspace_root: str) -> List[str]:
    """Return list of detected project types."""
    detected = []
    for ptype, markers in _PROJECT_MARKERS.items():
        for marker in markers:
            if "*" in marker:
                import glob
                if glob.glob(os.path.join(workspace_root, marker)):
                    detected.append(ptype)
                    break
            elif os.path.exists(os.path.join(workspace_root, marker)):
                detected.append(ptype)
                break
    return detected


def build_file_tree(workspace_root: str, max_depth: int = 3) -> str:
    """Build a compact file tree string."""
    lines = []

    def _walk(path: str, prefix: str, depth: int):
        if depth > max_depth:
            return
        try:
            entries = sorted(os.scandir(path), key=lambda e: (not e.is_dir(), e.name.lower()))
        except PermissionError:
            return

        visible = [e for e in entries if not e.name.startswith(".") and e.name not in SKIP_DIRS]
        for i, entry in enumerate(visible):
            is_last = i == len(visible) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
            if entry.is_dir():
                extension = "    " if is_last else "│   "
                _walk(entry.path, prefix + extension, depth + 1)

    lines.append(os.path.basename(workspace_root) + "/")
    _walk(workspace_root, "", 1)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Context result
# ---------------------------------------------------------------------------

@dataclass
class ContextResult:
    """Result of a context-building operation."""
    workspace_root: str
    project_types: List[str]
    file_tree: str
    entries: List[FileEntry]
    context_block: str          # ready for LLM injection
    resolved_files: List[str]   # files explicitly resolved from user input
    fuzzy_matches: Dict[str, str]  # {requested_name: resolved_path}
    ambiguous: Dict[str, List[str]]  # {name: [candidates]} when multiple matches
    total_files: int
    total_bytes: int
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Context: {self.total_files} file(s), ~{self.total_bytes // 1024} KB",
            f"Project: {', '.join(self.project_types) or 'unknown'}",
        ]
        if self.fuzzy_matches:
            for req, res in self.fuzzy_matches.items():
                lines.append(f"  Resolved: {req!r} → {res}")
        if self.ambiguous:
            for name, candidates in self.ambiguous.items():
                lines.append(f"  Ambiguous: {name!r} — {', '.join(candidates[:3])}")
        if self.warnings:
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Context Builder Agent
# ---------------------------------------------------------------------------

class ContextBuilderAgent:
    """
    Builds workspace context for injection into agent prompts.

    Usage:
        agent = ContextBuilderAgent(workspace_root=".")
        result = agent.build(user_input="fix the bug in main.py")
        # result.context_block is ready to inject into the LLM prompt
    """

    def __init__(self, workspace_root: str = "."):
        self.workspace_root = os.path.abspath(workspace_root)
        self.reader = FileReader(workspace_root=self.workspace_root)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        user_input: str = "",
        explicit_files: Optional[List[str]] = None,
        scan_dir: bool = True,
        max_scan_files: int = 20,
        include_tree: bool = True,
    ) -> ContextResult:
        """
        Build context for a user request.

        Args:
            user_input:      The user's message — scanned for @file references
                             and bare filenames.
            explicit_files:  Files explicitly requested (e.g. from /context cmd).
            scan_dir:        Whether to scan the current directory for context.
            max_scan_files:  Max files to include from directory scan.
            include_tree:    Whether to include the file tree.

        Returns:
            ContextResult with context_block ready for LLM injection.
        """
        entries: List[FileEntry] = []
        resolved_files: List[str] = []
        fuzzy_matches: Dict[str, str] = {}
        ambiguous: Dict[str, List[str]] = {}
        warnings: List[str] = []

        # ── Step 1: Extract file references from user input ──────────────────
        mentioned = self._extract_file_refs(user_input)

        # ── Step 2: Resolve explicit + mentioned files ───────────────────────
        all_requested = list(explicit_files or []) + mentioned
        for name in all_requested:
            resolved = self.reader.resolve(name)
            if resolved:
                rel = os.path.relpath(resolved, self.workspace_root)
                if name != rel and name != resolved:
                    fuzzy_matches[name] = rel
                entry = self.reader._load_file(resolved)
                if not any(e.abs_path == entry.abs_path for e in entries):
                    entries.append(entry)
                    resolved_files.append(rel)
            else:
                # Check for multiple candidates
                candidates = self.reader.fuzzy_candidates(name, limit=5)
                if candidates:
                    ambiguous[name] = candidates
                    warnings.append(
                        f"Ambiguous: {name!r} matches {len(candidates)} files — "
                        f"using {candidates[0]!r}"
                    )
                    # Auto-pick the first candidate
                    abs_c = os.path.join(self.workspace_root, candidates[0])
                    entry = self.reader._load_file(abs_c, candidates[0])
                    if not any(e.abs_path == entry.abs_path for e in entries):
                        entries.append(entry)
                        resolved_files.append(candidates[0])
                        fuzzy_matches[name] = candidates[0]
                else:
                    warnings.append(f"Not found: {name!r}")

        # ── Step 3: Priority files (README, requirements, etc.) ──────────────
        for fname in _PRIORITY_FILES:
            abs_p = os.path.join(self.workspace_root, fname)
            if os.path.exists(abs_p):
                if not any(e.abs_path == abs_p for e in entries):
                    entry = self.reader._load_file(abs_p, fname)
                    if not entry.error and not entry.is_binary:
                        entries.append(entry)

        # ── Step 4: Directory scan ───────────────────────────────────────────
        if scan_dir:
            scan_root = os.getcwd()
            scanned = self.reader.scan_directory(
                path=os.path.relpath(scan_root, self.workspace_root),
                max_files=max_scan_files,
            )
            for e in scanned:
                if not any(ex.abs_path == e.abs_path for ex in entries):
                    entries.append(e)

        # ── Step 5: Project detection ────────────────────────────────────────
        project_types = detect_project_type(self.workspace_root)

        # ── Step 6: File tree ────────────────────────────────────────────────
        file_tree = build_file_tree(self.workspace_root) if include_tree else ""

        # ── Step 7: Assemble context block ───────────────────────────────────
        parts = []

        if project_types:
            parts.append(f"PROJECT TYPE: {', '.join(project_types)}\n")

        if file_tree:
            parts.append(f"FILE TREE:\n```\n{file_tree}\n```\n")

        if entries:
            parts.append(build_context_block(entries, title="WORKSPACE FILES"))

        context_block = "\n".join(parts)

        total_bytes = sum(e.size_bytes for e in entries)

        return ContextResult(
            workspace_root=self.workspace_root,
            project_types=project_types,
            file_tree=file_tree,
            entries=entries,
            context_block=context_block,
            resolved_files=resolved_files,
            fuzzy_matches=fuzzy_matches,
            ambiguous=ambiguous,
            total_files=len(entries),
            total_bytes=total_bytes,
            warnings=warnings,
        )

    def resolve_file(self, name: str) -> Tuple[Optional[FileEntry], List[str]]:
        """
        Resolve a single filename to a FileEntry.

        Returns (entry, candidates):
        - If found: (entry, [])
        - If ambiguous: (entry_for_first_match, all_candidates)
        - If not found: (None, [])
        """
        resolved = self.reader.resolve(name)
        if resolved:
            entry = self.reader._load_file(resolved)
            return entry, []

        candidates = self.reader.fuzzy_candidates(name, limit=5)
        if candidates:
            abs_c = os.path.join(self.workspace_root, candidates[0])
            entry = self.reader._load_file(abs_c, candidates[0])
            return entry, candidates

        return None, []

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extract_file_refs(self, text: str) -> List[str]:
        """
        Extract file references from user input.

        Patterns detected:
        - @filename or @path/to/file
        - "file.py", 'file.py' (quoted)
        - bare filenames with known extensions at word boundaries
        - paths starting with ./ or ../
        """
        refs = []
        seen = set()

        def _add(name: str):
            name = name.strip("\"'`")
            if name and name not in seen:
                seen.add(name)
                refs.append(name)

        # @filename syntax
        for m in re.finditer(r"@([\w./\-]+)", text):
            _add(m.group(1))

        # Quoted paths
        for m in re.finditer(r'["\']([^"\']+\.[a-zA-Z0-9]+)["\']', text):
            _add(m.group(1))

        # Relative paths
        for m in re.finditer(r"(?:^|\s)(\.{1,2}/[\w./\-]+)", text):
            _add(m.group(1).strip())

        # Bare filenames with known extensions
        ext_pattern = (
            r"\b([\w\-]+\."
            r"(?:py|js|ts|jsx|tsx|html|css|json|yaml|yml|toml|md|txt|sh|"
            r"sql|rs|go|java|kt|rb|c|cpp|h|cs|env|cfg|conf|ini|lock|svg|png|jpg|jpeg|gif))"
            r"\b"
        )
        for m in re.finditer(ext_pattern, text, re.IGNORECASE):
            _add(m.group(1))

        return refs


import re  # needed for _extract_file_refs
