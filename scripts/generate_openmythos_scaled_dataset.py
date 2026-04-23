"""Generate a larger staged-refinement dataset for OpenMythos training.

The Docker proof image ships only a handful of teacher pairs. This generator
creates a deterministic instruction-code corpus with explicit loop-stage
targets so recurrent-depth training is not asked to invent refinement behavior
from final answers alone.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path("data/openmythos_scaled_refinement_dataset.jsonl")


TASK_FAMILIES = [
    {
        "category": "code_generation",
        "difficulty": "easy",
        "names": ["factorial", "fibonacci", "is_palindrome", "flatten_list"],
        "prompts": [
            "Write a Python function named {name} with type hints.",
            "Create a concise implementation of {name} in Python.",
            "Implement {name} and include a short example.",
        ],
    },
    {
        "category": "debugging",
        "difficulty": "medium",
        "names": ["sorted_copy", "safe_get", "parse_int", "divide"],
        "prompts": [
            "Fix a buggy Python helper named {name} and explain the issue.",
            "Repair {name} so it handles edge cases safely.",
            "Debug {name}; return complete corrected code.",
        ],
    },
    {
        "category": "refactoring",
        "difficulty": "medium",
        "names": ["normalize_name", "chunked", "dedupe_preserve_order", "load_json"],
        "prompts": [
            "Refactor repeated logic into a helper named {name}.",
            "Clean up a small Python routine by extracting {name}.",
            "Improve readability and testability for {name}.",
        ],
    },
    {
        "category": "test_writing",
        "difficulty": "medium",
        "names": ["add", "slugify", "clamp", "moving_average"],
        "prompts": [
            "Write pytest coverage for a function named {name}.",
            "Create edge-case tests for {name}.",
            "Generate a focused pytest module for {name}.",
        ],
    },
    {
        "category": "algorithm",
        "difficulty": "hard",
        "names": ["lru_cache", "binary_search", "topological_sort", "dijkstra"],
        "prompts": [
            "Implement {name} in Python with complexity notes and tests.",
            "Build a robust {name} implementation with edge-case handling.",
            "Write production-quality {name} code and explain trade-offs.",
        ],
    },
    {
        "category": "api",
        "difficulty": "hard",
        "names": ["rate_limiter", "request_validator", "retry_client", "task_queue"],
        "prompts": [
            "Design and implement a small {name} component in Python.",
            "Create a resilient {name} helper with tests and error handling.",
            "Build {name} for agentic coding workflows.",
        ],
    },
]


FINAL_TEMPLATES = {
    "factorial": """```python
def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("n must be non-negative")
    result = 1
    for value in range(2, n + 1):
        result *= value
    return result
```""",
    "fibonacci": """```python
def fibonacci(n: int) -> int:
    if n < 0:
        raise ValueError("n must be non-negative")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
```""",
    "is_palindrome": """```python
def is_palindrome(value: str) -> bool:
    cleaned = "".join(ch.lower() for ch in value if ch.isalnum())
    return cleaned == cleaned[::-1]
```""",
    "flatten_list": """```python
def flatten_list(items: list[list[object]]) -> list[object]:
    return [value for group in items for value in group]
```""",
    "sorted_copy": """```python
def sorted_copy(nums: list[int]) -> list[int]:
    return sorted(nums)
```
`list.sort()` mutates and returns `None`; `sorted()` returns a new list.""",
    "safe_get": """```python
def safe_get(mapping: dict[str, object], key: str, default: object = None) -> object:
    return mapping.get(key, default)
```""",
    "parse_int": """```python
def parse_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
```""",
    "divide": """```python
def divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        raise ZeroDivisionError("denominator must not be zero")
    return numerator / denominator
```""",
    "normalize_name": """```python
def normalize_name(value: str) -> str:
    return " ".join(value.strip().lower().split())
```""",
    "chunked": """```python
def chunked(items: list[object], size: int) -> list[list[object]]:
    if size <= 0:
        raise ValueError("size must be positive")
    return [items[index:index + size] for index in range(0, len(items), size)]
```""",
    "dedupe_preserve_order": """```python
def dedupe_preserve_order(items: list[object]) -> list[object]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
```""",
    "load_json": """```python
import json
from pathlib import Path

def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
```""",
    "add": """```python
def test_add_returns_sum():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
```""",
    "slugify": """```python
def test_slugify_normalizes_text():
    assert slugify("Hello, World!") == "hello-world"
    assert slugify("  many   spaces ") == "many-spaces"
```""",
    "clamp": """```python
def test_clamp_bounds_values():
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(11, 0, 10) == 10
```""",
    "moving_average": """```python
def test_moving_average_window():
    assert moving_average([1, 2, 3, 4], window=2) == [1.5, 2.5, 3.5]
```""",
    "lru_cache": """```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._data: OrderedDict[str, object] = OrderedDict()

    def get(self, key: str) -> object | None:
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: str, value: object) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self.capacity:
            self._data.popitem(last=False)
```""",
    "binary_search": """```python
def binary_search(items: list[int], target: int) -> int:
    low, high = 0, len(items) - 1
    while low <= high:
        mid = (low + high) // 2
        if items[mid] == target:
            return mid
        if items[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
```""",
    "topological_sort": """```python
from collections import deque

def topological_sort(graph: dict[str, list[str]]) -> list[str]:
    indegree = {node: 0 for node in graph}
    for neighbors in graph.values():
        for neighbor in neighbors:
            indegree[neighbor] = indegree.get(neighbor, 0) + 1
    queue = deque(node for node, degree in indegree.items() if degree == 0)
    result = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph.get(node, []):
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)
    if len(result) != len(indegree):
        raise ValueError("cycle detected")
    return result
```""",
    "dijkstra": """```python
import heapq

def dijkstra(graph: dict[str, list[tuple[str, float]]], start: str) -> dict[str, float]:
    distances = {start: 0.0}
    queue = [(0.0, start)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance > distances[node]:
            continue
        for neighbor, weight in graph.get(node, []):
            candidate = distance + weight
            if candidate < distances.get(neighbor, float("inf")):
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return distances
```""",
    "rate_limiter": """```python
import time

class TokenBucket:
    def __init__(self, rate: float, capacity: int):
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(capacity)
        self.updated_at = time.monotonic()

    def allow(self, cost: float = 1.0) -> bool:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated_at) * self.rate)
        self.updated_at = now
        if self.tokens < cost:
            return False
        self.tokens -= cost
        return True
```""",
    "request_validator": """```python
def validate_request(payload: dict[str, object]) -> dict[str, object]:
    if not isinstance(payload.get("id"), str):
        raise ValueError("id is required")
    if not isinstance(payload.get("items"), list):
        raise ValueError("items must be a list")
    return payload
```""",
    "retry_client": """```python
import time

def retry_call(fn, attempts: int = 3, delay: float = 0.1):
    last_error = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(delay * (2 ** attempt))
    raise last_error
```""",
    "task_queue": """```python
from collections import deque

class TaskQueue:
    def __init__(self):
        self._items = deque()

    def push(self, item: object) -> None:
        self._items.append(item)

    def pop(self) -> object:
        if not self._items:
            raise IndexError("queue is empty")
        return self._items.popleft()
```""",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate scaled OpenMythos staged data.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--examples", type=int, default=1200)
    parser.add_argument("--holdout-ratio", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    examples = build_examples(args.examples, args.holdout_ratio, rng)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in examples:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = summarize(examples)
    print(json.dumps({"output": str(args.output), **summary}, indent=2, sort_keys=True))
    return 0


def build_examples(
    total: int,
    holdout_ratio: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    rows = []
    holdout_interval = max(2, round(1 / holdout_ratio))
    for index in range(total):
        family = TASK_FAMILIES[index % len(TASK_FAMILIES)]
        family_position = index // len(TASK_FAMILIES)
        name = family["names"][(index // len(TASK_FAMILIES)) % len(family["names"])]
        prompt_template = rng.choice(family["prompts"])
        style = rng.choice(
            [
                "Prefer clear names.",
                "Keep the implementation dependency-light.",
                "Include a brief reasoning note.",
                "Mention the main edge case.",
                "Return only practical code and concise explanation.",
            ]
        )
        prompt = f"{prompt_template.format(name=name)} {style}"
        final_answer = final_for(name)
        split = "holdout" if family_position % holdout_interval == 0 else "train"
        rows.append(
            {
                "task_id": f"scaled_{index:05d}_{name}",
                "prompt": prompt,
                "final_answer": final_answer,
                "response": final_answer,
                "category": family["category"],
                "difficulty": family["difficulty"],
                "split": split,
                "stages": stage_targets(name, final_answer),
                "metadata": {
                    "source": "openmythos_scaled_synthetic_v1",
                    "template_family": family["category"],
                    "function_name": name,
                },
            }
        )
    return rows


def final_for(name: str) -> str:
    return FINAL_TEMPLATES[name]


def stage_targets(name: str, final_answer: str) -> list[dict[str, Any]]:
    return [
        {
            "loop": 1,
            "label": "rough_draft",
            "target": f"Plan a compact implementation for `{name}` and identify the main edge case.",
        },
        {
            "loop": 2,
            "label": "valid_code",
            "target": final_answer.split("\n```", 1)[0] + "\n```",
        },
        {
            "loop": 3,
            "label": "correct_logic",
            "target": final_answer,
        },
        {
            "loop": 4,
            "label": "polished_final",
            "target": final_answer + "\n\nThe implementation keeps behavior explicit and handles the key edge case.",
        },
    ]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "examples": len(rows),
        "split_counts": {},
        "category_counts": {},
        "difficulty_counts": {},
        "staged_examples": sum(1 for row in rows if row.get("stages")),
    }
    for key in ["split", "category", "difficulty"]:
        bucket: dict[str, int] = {}
        for row in rows:
            value = row[key]
            bucket[value] = bucket.get(value, 0) + 1
        summary[f"{key}_counts"] = bucket
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
