"""Storage for agent registry."""

import json
import logging
import shutil
from pathlib import Path
from typing import Optional
from app.schemas.registry import AgentRegistry

logger = logging.getLogger(__name__)


class RegistryStore:
    """Persistent storage for agent registry."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.registry_path = self.data_dir / "agent_registry.json"
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def save_registry(self, registry: AgentRegistry) -> None:
        """Save agent registry to disk atomically.
        
        Writes to a temp file first, then renames to avoid
        partial writes on crash.
        """
        tmp_path = self.registry_path.with_suffix(".tmp")
        try:
            with open(tmp_path, 'w') as f:
                json.dump(registry.model_dump(), f, indent=2)
            # Atomic rename (same filesystem)
            shutil.move(str(tmp_path), str(self.registry_path))
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
    
    def load_registry(self) -> Optional[AgentRegistry]:
        """Load agent registry from disk.
        
        Returns None only when no registry file exists.
        Raises on corruption so the caller can decide how to recover.
        """
        if not self.registry_path.exists():
            return None
        
        try:
            with open(self.registry_path, 'r') as f:
                data = json.load(f)
            return AgentRegistry(**data)
        except Exception as e:
            logger.error("Failed to load registry from %s: %s", self.registry_path, e)
            raise
