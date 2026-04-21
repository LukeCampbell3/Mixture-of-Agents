"""
Knowledge Enricher — fetches real documentation and injects it into
agent prompts before execution.

This is the core intelligence upgrade: instead of agents relying purely
on training data (which may be stale or incomplete), the enricher:

1. Analyses the task to identify what external knowledge is relevant
2. Fetches that knowledge (docs, package metadata, GitHub READMEs)
3. Injects it as grounded context into the agent's prompt

The enricher runs BEFORE the primary agent, so every agent response
is grounded in current, accurate information.

Design principles:
- Fast: parallel fetches, 8s timeout, in-memory cache
- Selective: only fetches what's relevant to the task (max 5 sources)
- Transparent: logs what was fetched so users can see the grounding
- Graceful: fetch failures are silently skipped, never block execution
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from app.tools.web_fetcher import WebFetcher, FetchResult, build_knowledge_queries


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EnrichmentResult:
    """Knowledge fetched for a single task."""
    task_text: str
    sources: List[FetchResult] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def has_content(self) -> bool:
        return any(s.ok and s.content for s in self.sources)

    def as_context_block(self) -> str:
        """Format all fetched knowledge as a prompt-ready context block."""
        if not self.has_content:
            return ""

        good = [s for s in self.sources if s.ok and s.content]
        if not good:
            return ""

        lines = [
            "=== RETRIEVED KNOWLEDGE (current documentation) ===",
            f"Fetched {len(good)} source(s) in {self.elapsed_s:.1f}s\n",
        ]
        for src in good:
            lines.append(src.as_context_block())
        lines.append("=== END RETRIEVED KNOWLEDGE ===\n")
        return "\n".join(lines)

    def summary(self) -> str:
        good = [s for s in self.sources if s.ok]
        failed = [s for s in self.sources if not s.ok]
        parts = [f"Knowledge: {len(good)} source(s) fetched"]
        if failed:
            parts.append(f"{len(failed)} failed")
        for s in good:
            parts.append(f"  ✓ {s.title[:60]} ({s.source})")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Knowledge Enricher
# ---------------------------------------------------------------------------

class KnowledgeEnricher:
    """
    Pre-execution knowledge fetcher.

    Usage:
        enricher = KnowledgeEnricher()
        result = enricher.enrich(task_text)
        # Inject result.as_context_block() into the agent prompt
    """

    def __init__(self, fetcher: Optional[WebFetcher] = None):
        self._fetcher = fetcher or WebFetcher()

    def enrich(self, task_text: str, verbose: bool = False) -> EnrichmentResult:
        """
        Fetch relevant knowledge for a task.

        Args:
            task_text: The user's request / task description.
            verbose:   Print fetch progress to stdout.

        Returns:
            EnrichmentResult with fetched content ready for injection.
        """
        t0 = time.perf_counter()
        queries = build_knowledge_queries(task_text)

        if not queries:
            return EnrichmentResult(task_text=task_text, elapsed_s=0.0)

        if verbose:
            print(f"  [knowledge] Fetching {len(queries)} source(s)...")

        sources: List[FetchResult] = []

        # Fetch all queries in parallel
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futures = {}
            for kind, query in queries:
                if kind == "url":
                    futures[ex.submit(self._fetcher.fetch, query)] = query
                elif kind == "pypi":
                    futures[ex.submit(self._fetcher.fetch_pypi, query)] = query
                elif kind == "npm":
                    futures[ex.submit(self._fetcher.fetch_npm, query)] = query
                elif kind == "github":
                    owner, repo = query.split("/", 1)
                    futures[ex.submit(self._fetcher.fetch_github_readme, owner, repo)] = query

            for fut, query in futures.items():
                try:
                    result = fut.result(timeout=10)
                    sources.append(result)
                    if verbose:
                        status = "✓" if result.ok else "✗"
                        print(f"  [knowledge]   {status} {result.title[:50]} ({result.source})")
                except Exception as e:
                    if verbose:
                        print(f"  [knowledge]   ✗ {query}: {e}")

        elapsed = time.perf_counter() - t0
        return EnrichmentResult(
            task_text=task_text,
            sources=sources,
            elapsed_s=elapsed,
        )

    def enrich_for_domain(self, domain: str, task_text: str) -> EnrichmentResult:
        """
        Fetch knowledge specifically for a specialist domain.
        Used when a specialist agent is spawned.
        """
        # Build domain-specific queries
        domain_urls: Dict[str, List[str]] = {
            "ml_engineering":  [
                "https://pytorch.org/docs/stable/index.html",
                "https://huggingface.co/docs/transformers/index",
            ],
            "database":        ["https://docs.sqlalchemy.org/en/20/"],
            "api":             ["https://fastapi.tiangolo.com/"],
            "devops":          ["https://docs.docker.com/get-started/"],
            "security":        ["https://owasp.org/www-project-top-ten/"],
            "frontend":        ["https://react.dev/learn"],
            "testing":         ["https://docs.pytest.org/en/stable/"],
            "data_analysis":   ["https://pandas.pydata.org/docs/user_guide/index.html"],
        }

        urls = domain_urls.get(domain, [])
        if not urls:
            return self.enrich(task_text)

        sources = []
        for url in urls[:2]:
            result = self._fetcher.fetch(url)
            sources.append(result)

        return EnrichmentResult(
            task_text=task_text,
            sources=sources,
            elapsed_s=0.0,
        )
