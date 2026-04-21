"""Web research agent — fetches real documentation and validates facts."""

import re
from app.agents.base_agent import BaseAgent
from app.tools.web_fetcher import WebFetcher, build_knowledge_queries


class WebResearchAgent(BaseAgent):
    """
    Research agent that actually fetches current documentation.

    Unlike a pure LLM agent, this agent:
    1. Identifies what URLs/packages are relevant to the task
    2. Fetches real content from those sources
    3. Synthesises the fetched content with LLM reasoning
    4. Returns grounded, cited responses
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fetcher = WebFetcher()

    def get_system_prompt(self) -> str:
        return """You are a research agent with access to current documentation.

RULES:
- Base your answers on the RETRIEVED KNOWLEDGE provided in context
- Always cite the source URL when referencing fetched content
- If retrieved content contradicts your training data, prefer the retrieved content
- Note version numbers and API changes explicitly
- Flag anything that may have changed since the retrieved content was fetched

When retrieved knowledge is available:
1. Lead with what the documentation says
2. Add your synthesis and explanation
3. Note any gaps or caveats
4. Provide working code examples that match the documented API

When no retrieved knowledge is available:
- Be explicit that you're relying on training data
- Note the knowledge cutoff risk
- Recommend the user verify against official docs
"""

    def execute(self, task_context):
        """Execute with real web fetching before LLM generation."""
        task_frame = task_context.get("task_frame")
        task_text = task_frame.normalized_request if task_frame else ""

        # Fetch relevant knowledge
        queries = build_knowledge_queries(task_text)
        fetched_blocks = []

        if queries:
            print(f"  [research] Fetching {len(queries)} source(s)...")
            for kind, query in queries[:3]:  # cap at 3 for research sub-tasks
                try:
                    if kind == "url":
                        result = self._fetcher.fetch(query)
                    elif kind == "pypi":
                        result = self._fetcher.fetch_pypi(query)
                    elif kind == "npm":
                        result = self._fetcher.fetch_npm(query)
                    elif kind == "github":
                        owner, repo = query.split("/", 1)
                        result = self._fetcher.fetch_github_readme(owner, repo)
                    else:
                        continue

                    if result.ok and result.content:
                        fetched_blocks.append(result.as_context_block())
                        print(f"  [research]   ✓ {result.title[:50]}")
                except Exception:
                    pass

        # Inject fetched content into shared context
        if fetched_blocks:
            existing = task_context.get("shared_context", "")
            knowledge_block = (
                "\n=== RETRIEVED DOCUMENTATION ===\n"
                + "\n".join(fetched_blocks)
                + "\n=== END RETRIEVED DOCUMENTATION ===\n"
            )
            task_context = dict(task_context)
            task_context["shared_context"] = knowledge_block + "\n\n" + existing

        # Delegate to base execute (which builds prompt + calls LLM)
        return super().execute(task_context)
