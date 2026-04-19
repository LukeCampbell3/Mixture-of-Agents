"""Storage for execution artifacts."""

import json
from pathlib import Path
from typing import Dict, Any
from app.schemas.run_state import RunState


class ArtifactStore:
    """Persistent storage for execution artifacts."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.episodes_dir = self.data_dir / "episodes"
        self.reports_dir = self.data_dir / "reports"
        
        # Create directories
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def save_run_state(self, run_state: RunState) -> None:
        """Save run state to disk."""
        episode_file = self.episodes_dir / f"{run_state.task_id}.json"
        with open(episode_file, 'w', encoding='utf-8') as f:
            json.dump(run_state.model_dump(), f, indent=2, ensure_ascii=False)
    
    def save_context(self, task_id: str, context: str) -> None:
        """Save shared context to disk."""
        context_file = self.episodes_dir / f"{task_id}_context.md"
        with open(context_file, 'w', encoding='utf-8') as f:
            f.write(context)
    
    def save_analysis(self, task_id: str, analysis: Dict[str, Any]) -> None:
        """Save analysis data to disk."""
        analysis_file = self.episodes_dir / f"{task_id}_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    def load_run_state(self, task_id: str) -> RunState:
        """Load run state from disk."""
        episode_file = self.episodes_dir / f"{task_id}.json"
        with open(episode_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return RunState(**data)
