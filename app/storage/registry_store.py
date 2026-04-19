"""Storage for agent registry."""

import json
import os
from pathlib import Path
from typing import Optional
from app.schemas.registry import AgentRegistry


class RegistryStore:
    """Persistent storage for agent registry."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.registry_path = self.data_dir / "agent_registry.json"
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def save_registry(self, registry: AgentRegistry) -> None:
        """Save agent registry to disk."""
        with open(self.registry_path, 'w') as f:
            json.dump(registry.model_dump(), f, indent=2)
    
    def load_registry(self) -> Optional[AgentRegistry]:
        """Load agent registry from disk."""
        if not self.registry_path.exists():
            return None
        
        try:
            with open(self.registry_path, 'r') as f:
                data = json.load(f)
            return AgentRegistry(**data)
        except Exception as e:
            print(f"Error loading registry: {e}")
            return None
