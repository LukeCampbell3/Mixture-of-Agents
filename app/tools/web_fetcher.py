"""
Web fetcher — retrieves real content from URLs and search-like queries.

Used by the KnowledgeEnricher to ground agent responses in current,
accurate documentation rather than relying solely on training data.

Capabilities:
- Fetch a URL and extract clean text (strips HTML boilerplate)
- Search PyPI, npm, crates.io for package metadata
- Fetch GitHub README / file content via raw URLs
- Fetch official docs pages (Python docs, MDN, etc.)
- Cache results in-memory for the session to avoid duplicate fetches
"""

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, urljoin


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FetchResult:
    url: str
    title: str
    content: str        # cleaned text, max ~4000 chars
    source: str         # "web", "pypi", "npm", "github", etc.
    fetched_at: float = field(default_factory=time.time)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def snippet(self) -> str:
        """First 500 chars for display."""
        return self.content[:500] + ("..." if len(self.content) > 500 else "")

    def as_context_block(self) -> str:
        if self.error:
            return f"[{self.source}] {self.url} — ERROR: {self.error}\n"
        return (
            f"### {self.title}\n"
            f"Source: {self.url}\n"
            f"```\n{self.content[:3000]}\n```\n"
        )


# ---------------------------------------------------------------------------
# HTML cleaner
# ---------------------------------------------------------------------------

def _clean_html(html: str, max_chars: int = 4000) -> Tuple[str, str]:
    """Extract title and clean text from HTML. Returns (title, text)."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        # Title
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header",
                          "aside", "advertisement", "noscript", "iframe"]):
            tag.decompose()

        # Prefer main content areas
        main = (soup.find("main") or soup.find("article") or
                soup.find(id=re.compile(r"content|main|article", re.I)) or
                soup.find(class_=re.compile(r"content|main|article|body", re.I)) or
                soup.body or soup)

        text = main.get_text(separator="\n", strip=True)

        # Collapse whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        return title, text[:max_chars]
    except Exception as e:
        # Fallback: strip all tags with regex
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return "", text[:max_chars]


# ---------------------------------------------------------------------------
# Web fetcher
# ---------------------------------------------------------------------------

class WebFetcher:
    """
    Fetch web content for agent knowledge enrichment.

    All results are cached in-memory for the session lifetime.
    """

    TIMEOUT = 8  # seconds
    MAX_CONTENT = 4000  # chars per page

    def __init__(self):
        self._cache: Dict[str, FetchResult] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self, url: str) -> FetchResult:
        """Fetch a URL and return cleaned content."""
        cache_key = hashlib.md5(url.encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = self._fetch_url(url)
        self._cache[cache_key] = result
        return result

    def fetch_many(self, urls: List[str]) -> List[FetchResult]:
        """Fetch multiple URLs concurrently."""
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            results = list(ex.map(self.fetch, urls))
        return results

    def fetch_pypi(self, package: str) -> FetchResult:
        """Fetch PyPI package metadata."""
        url = f"https://pypi.org/pypi/{package}/json"
        cache_key = f"pypi:{package}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            import requests
            r = requests.get(url, timeout=self.TIMEOUT)
            r.raise_for_status()
            data = r.json()
            info = data.get("info", {})
            latest = info.get("version", "?")
            summary = info.get("summary", "")
            home = info.get("home_page", "") or info.get("project_url", "")
            requires = info.get("requires_python", "")
            deps = info.get("requires_dist", []) or []

            content = (
                f"Package: {package}\n"
                f"Latest version: {latest}\n"
                f"Summary: {summary}\n"
                f"Python requires: {requires}\n"
                f"Homepage: {home}\n"
                f"Dependencies: {', '.join(deps[:10])}\n"
            )
            result = FetchResult(
                url=url, title=f"{package} {latest} — PyPI",
                content=content, source="pypi"
            )
        except Exception as e:
            result = FetchResult(url=url, title=package, content="",
                                 source="pypi", error=str(e))

        self._cache[cache_key] = result
        return result

    def fetch_npm(self, package: str) -> FetchResult:
        """Fetch npm package metadata."""
        url = f"https://registry.npmjs.org/{package}/latest"
        cache_key = f"npm:{package}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            import requests
            r = requests.get(url, timeout=self.TIMEOUT)
            r.raise_for_status()
            data = r.json()
            version = data.get("version", "?")
            desc = data.get("description", "")
            deps = list(data.get("dependencies", {}).keys())[:10]
            content = (
                f"Package: {package}\n"
                f"Latest version: {version}\n"
                f"Description: {desc}\n"
                f"Dependencies: {', '.join(deps)}\n"
            )
            result = FetchResult(
                url=f"https://www.npmjs.com/package/{package}",
                title=f"{package} {version} — npm",
                content=content, source="npm"
            )
        except Exception as e:
            result = FetchResult(url=url, title=package, content="",
                                 source="npm", error=str(e))

        self._cache[cache_key] = result
        return result

    def fetch_github_readme(self, owner: str, repo: str) -> FetchResult:
        """Fetch a GitHub repo's README."""
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/README.md"
        cache_key = f"github:{owner}/{repo}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            import requests
            r = requests.get(url, timeout=self.TIMEOUT)
            if r.status_code == 404:
                url = url.replace("/main/", "/master/")
                r = requests.get(url, timeout=self.TIMEOUT)
            r.raise_for_status()
            content = r.text[:self.MAX_CONTENT]
            result = FetchResult(
                url=f"https://github.com/{owner}/{repo}",
                title=f"{owner}/{repo} README",
                content=content, source="github"
            )
        except Exception as e:
            result = FetchResult(url=url, title=f"{owner}/{repo}", content="",
                                 source="github", error=str(e))

        self._cache[cache_key] = result
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch_url(self, url: str) -> FetchResult:
        try:
            import requests
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (compatible; MoA-CLI/1.0; "
                    "+https://github.com/mixture-of-agents)"
                ),
                "Accept": "text/html,application/xhtml+xml,text/plain",
            }
            r = requests.get(url, timeout=self.TIMEOUT, headers=headers,
                             allow_redirects=True)
            r.raise_for_status()

            content_type = r.headers.get("content-type", "")
            if "html" in content_type:
                title, text = _clean_html(r.text, self.MAX_CONTENT)
            else:
                title = urlparse(url).path.split("/")[-1]
                text = r.text[:self.MAX_CONTENT]

            return FetchResult(
                url=url, title=title or url,
                content=text, source="web"
            )
        except Exception as e:
            return FetchResult(url=url, title=url, content="",
                               source="web", error=str(e))


# ---------------------------------------------------------------------------
# Knowledge query builder
# ---------------------------------------------------------------------------

# Maps domain keywords to documentation URLs worth fetching
_DOMAIN_DOCS: Dict[str, List[str]] = {
    "pytorch":      ["https://pytorch.org/docs/stable/index.html"],
    "tensorflow":   ["https://www.tensorflow.org/api_docs/python/tf"],
    "fastapi":      ["https://fastapi.tiangolo.com/"],
    "flask":        ["https://flask.palletsprojects.com/en/latest/"],
    "django":       ["https://docs.djangoproject.com/en/stable/"],
    "pandas":       ["https://pandas.pydata.org/docs/user_guide/index.html"],
    "numpy":        ["https://numpy.org/doc/stable/reference/index.html"],
    "sqlalchemy":   ["https://docs.sqlalchemy.org/en/20/"],
    "pydantic":     ["https://docs.pydantic.dev/latest/"],
    "asyncio":      ["https://docs.python.org/3/library/asyncio.html"],
    "react":        ["https://react.dev/learn"],
    "nextjs":       ["https://nextjs.org/docs"],
    "typescript":   ["https://www.typescriptlang.org/docs/handbook/intro.html"],
    "rust":         ["https://doc.rust-lang.org/book/"],
    "docker":       ["https://docs.docker.com/get-started/"],
    "kubernetes":   ["https://kubernetes.io/docs/concepts/"],
    "graphql":      ["https://graphql.org/learn/"],
    "openai":       ["https://platform.openai.com/docs/overview"],
    "anthropic":    ["https://docs.anthropic.com/en/docs/welcome"],
}

_PYPI_PACKAGES = {
    "requests", "httpx", "aiohttp", "fastapi", "flask", "django",
    "sqlalchemy", "pydantic", "celery", "redis", "pymongo",
    "pandas", "numpy", "scipy", "matplotlib", "seaborn",
    "pytorch", "torch", "tensorflow", "keras", "transformers",
    "pytest", "black", "ruff", "mypy", "poetry",
}

_NPM_PACKAGES = {
    "react", "vue", "angular", "svelte", "next", "nuxt",
    "express", "fastify", "axios", "lodash", "typescript",
    "webpack", "vite", "esbuild", "jest", "vitest",
}


def build_knowledge_queries(task_text: str) -> List[Tuple[str, str]]:
    """
    Given a task description, return a list of (type, query) tuples
    representing knowledge to fetch.

    Returns e.g.:
        [("pypi", "fastapi"), ("url", "https://fastapi.tiangolo.com/"), ...]
    """
    text_lower = task_text.lower()
    queries: List[Tuple[str, str]] = []
    seen: set = set()

    def _add(kind: str, val: str):
        key = f"{kind}:{val}"
        if key not in seen:
            seen.add(key)
            queries.append((kind, val))

    # Check domain docs
    for keyword, urls in _DOMAIN_DOCS.items():
        if keyword in text_lower:
            for url in urls[:1]:  # one URL per domain
                _add("url", url)

    # Check PyPI packages
    for pkg in _PYPI_PACKAGES:
        if pkg in text_lower:
            _add("pypi", pkg)

    # Check npm packages
    for pkg in _NPM_PACKAGES:
        if pkg in text_lower:
            _add("npm", pkg)

    # GitHub patterns: "owner/repo"
    for m in re.finditer(r"\b([a-zA-Z0-9_-]+)/([a-zA-Z0-9_.-]+)\b", task_text):
        owner, repo = m.group(1), m.group(2)
        if owner.lower() not in ("http", "https", "www") and len(repo) > 2:
            _add("github", f"{owner}/{repo}")

    return queries[:5]  # cap at 5 fetches per task
