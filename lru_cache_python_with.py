from typing import Dict, Optional, Tuple
import threading
import openai

class LRUCache:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache = {}
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        with self.lock:
            if key in self.cache:
                return self.cache[key]
            else:
                return None

    def put(self, key: str, value: str):
        with self.lock:
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache, key=self.cache.get)
                del self.cache[oldest_key]
            self.cache[key] = value
