"""User manager for user-aware control and personalization."""

from typing import Dict, Any, Optional
from pathlib import Path
import json
from app.schemas.user_profile import UserProfile, AgentPreference
from app.schemas.task_frame import TaskType


class UserManager:
    """Manage user profiles and preferences."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.profiles_dir = self.data_dir / "user_profiles"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_user_id = "default"
        self.current_profile: Optional[UserProfile] = None
    
    def get_profile(self, user_id: str = "default") -> UserProfile:
        """Get or create user profile."""
        if self.current_profile and self.current_profile.user_id == user_id:
            return self.current_profile
        
        profile_path = self.profiles_dir / f"{user_id}.json"
        
        if profile_path.exists():
            with open(profile_path, 'r') as f:
                data = json.load(f)
            profile = UserProfile(**data)
        else:
            profile = UserProfile(user_id=user_id)
            self.save_profile(profile)
        
        self.current_profile = profile
        self.current_user_id = user_id
        return profile
    
    def save_profile(self, profile: UserProfile) -> None:
        """Save user profile to disk."""
        profile_path = self.profiles_dir / f"{profile.user_id}.json"
        with open(profile_path, 'w') as f:
            json.dump(profile.model_dump(), f, indent=2)
    
    def update_from_task(
        self,
        user_id: str,
        task_type: TaskType,
        agent_outputs: Dict[str, Any],
        quality_scores: Dict[str, float],
        cost_metrics: Dict[str, float]
    ) -> None:
        """Update user profile from task execution."""
        profile = self.get_profile(user_id)
        
        # Update task category stats
        category = task_type.value
        profile.update_task_stats(category, quality_scores, cost_metrics)
        
        # Save updated profile
        self.save_profile(profile)
    
    def get_user_adjusted_activation_threshold(
        self,
        user_id: str,
        agent_id: str,
        task_type: TaskType,
        base_threshold: float = 0.3
    ) -> float:
        """Get user-adjusted activation threshold for an agent."""
        profile = self.get_profile(user_id)
        
        # Check if agent is pinned or disabled
        pref = profile.get_agent_preference(agent_id)
        if pref.disabled:
            return 1.0  # Effectively disable
        if pref.pinned:
            return 0.0  # Always activate
        
        # Check custom threshold
        if pref.custom_activation_threshold is not None:
            return pref.custom_activation_threshold
        
        # Adjust based on historical performance for this category
        category = task_type.value
        quality_gain = profile.get_agent_quality_for_category(agent_id, category)
        
        # Lower threshold for agents that perform well for this user/category
        if quality_gain > 0.7:
            return base_threshold * 0.7  # 30% lower threshold
        elif quality_gain > 0.5:
            return base_threshold
        else:
            return base_threshold * 1.3  # 30% higher threshold
    
    def get_pinned_agents(
        self,
        user_id: str,
        task_type: Optional[TaskType] = None
    ) -> list[str]:
        """Get list of pinned agents for user."""
        profile = self.get_profile(user_id)
        
        pinned = []
        for agent_id, pref in profile.agent_preferences.items():
            if pref.pinned:
                # Check if pinned for specific categories
                if task_type and pref.preferred_for_categories:
                    if task_type.value in pref.preferred_for_categories:
                        pinned.append(agent_id)
                else:
                    pinned.append(agent_id)
        
        return pinned
    
    def get_disabled_agents(self, user_id: str) -> list[str]:
        """Get list of disabled agents for user."""
        profile = self.get_profile(user_id)
        
        return [
            agent_id
            for agent_id, pref in profile.agent_preferences.items()
            if pref.disabled
        ]
    
    def get_budget_mode(self, user_id: str) -> str:
        """Get user's preferred budget mode."""
        profile = self.get_profile(user_id)
        return profile.budget_preference.default_mode
    
    def get_max_agents(self, user_id: str, default: int) -> int:
        """Get user's max agents preference."""
        profile = self.get_profile(user_id)
        return profile.budget_preference.max_agents_override or default
    
    def get_validation_thoroughness(self, user_id: str) -> str:
        """Get user's validation thoroughness preference."""
        profile = self.get_profile(user_id)
        return profile.validation_thoroughness
    
    def get_task_distribution_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary of user's task distribution."""
        profile = self.get_profile(user_id)
        
        return {
            "user_id": user_id,
            "total_tasks": profile.total_tasks,
            "category_distribution": profile.get_category_distribution(),
            "pinned_agents": [
                agent_id for agent_id, pref in profile.agent_preferences.items()
                if pref.pinned
            ],
            "disabled_agents": [
                agent_id for agent_id, pref in profile.agent_preferences.items()
                if pref.disabled
            ],
            "budget_mode": profile.budget_preference.default_mode
        }
