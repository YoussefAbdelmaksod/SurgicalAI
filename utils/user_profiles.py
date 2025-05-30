"""
User profile management for SurgicalAI.

This module implements a profile management system to store surgeon preferences,
experience levels, and historical performance data for personalized guidance.
"""

import os
import json
import logging
import datetime
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class UserProfile:
    """
    Represents a surgeon's profile with experience level, preferences, and history.
    """
    
    # Experience level definitions
    EXPERIENCE_LEVELS = {
        "novice": 1,
        "junior": 2, 
        "intermediate": 3,
        "senior": 4,
        "expert": 5
    }
    
    def __init__(self, user_id: str, name: str = "", 
                 experience_level: str = "intermediate",
                 profile_path: Optional[str] = None):
        """
        Initialize a user profile.
        
        Args:
            user_id: Unique identifier for the user
            name: User's name
            experience_level: Experience level of the surgeon
            profile_path: Path to load/save profile data
        """
        self.user_id = user_id
        self.name = name
        self.created_at = datetime.datetime.now().isoformat()
        self.last_updated = self.created_at
        
        # Set experience level
        if experience_level in self.EXPERIENCE_LEVELS:
            self.experience_level = experience_level
        else:
            logger.warning(f"Invalid experience level: {experience_level}. Setting to 'intermediate'.")
            self.experience_level = "intermediate"
        
        # Experience level as numeric value (1-5)
        self.experience_value = self.EXPERIENCE_LEVELS[self.experience_level]
        
        # Guidance preferences
        self.preferences = {
            "voice_guidance": True,
            "guidance_detail_level": "standard",  # minimal, standard, detailed
            "voice_gender": "female",
            "voice_rate": 150,
            "critical_warnings_only": False,
            "show_phase_recommendations": True,
            "show_tool_recommendations": True
        }
        
        # Performance history
        self.performance_history = []
        
        # Personalization data
        self.personalization = {
            "common_mistakes": {},  # phase -> list of common mistakes
            "phase_durations": {},  # phase -> average duration
            "tool_preferences": {},  # phase -> preferred tools
            "learning_focus": []    # areas to focus guidance on
        }
        
        # Profile path
        self.profile_path = profile_path
        
        # Load profile if path is provided
        if profile_path and os.path.exists(profile_path):
            self.load()
    
    def update_experience_level(self, new_level: str) -> bool:
        """
        Update the surgeon's experience level.
        
        Args:
            new_level: New experience level
            
        Returns:
            bool: True if update was successful
        """
        if new_level in self.EXPERIENCE_LEVELS:
            self.experience_level = new_level
            self.experience_value = self.EXPERIENCE_LEVELS[new_level]
            self.last_updated = datetime.datetime.now().isoformat()
            return True
        return False
    
    def update_preference(self, key: str, value: Any) -> bool:
        """
        Update a user preference.
        
        Args:
            key: Preference key
            value: Preference value
            
        Returns:
            bool: True if update was successful
        """
        if key in self.preferences:
            self.preferences[key] = value
            self.last_updated = datetime.datetime.now().isoformat()
            return True
        return False
    
    def add_performance_record(self, 
                               procedure_type: str,
                               performance_metrics: Dict[str, Any],
                               mistakes: List[Dict[str, Any]],
                               phase_durations: Dict[str, float],
                               tool_usage: Dict[str, Any]) -> None:
        """
        Add a performance record to history and update personalization data.
        
        Args:
            procedure_type: Type of surgical procedure
            performance_metrics: Performance metrics
            mistakes: List of mistakes made
            phase_durations: Duration of each surgical phase
            tool_usage: Tool usage statistics
        """
        # Create performance record
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "procedure_type": procedure_type,
            "performance_metrics": performance_metrics,
            "mistake_count": len(mistakes),
            "phase_durations": phase_durations,
        }
        
        # Add to history
        self.performance_history.append(record)
        
        # Update personalization data
        self._update_personalization(procedure_type, mistakes, phase_durations, tool_usage)
        
        # Update last updated timestamp
        self.last_updated = datetime.datetime.now().isoformat()
    
    def _update_personalization(self, 
                                procedure_type: str,
                                mistakes: List[Dict[str, Any]],
                                phase_durations: Dict[str, float],
                                tool_usage: Dict[str, Any]) -> None:
        """
        Update personalization data based on latest performance.
        
        Args:
            procedure_type: Type of surgical procedure
            mistakes: List of mistakes made
            phase_durations: Duration of each surgical phase
            tool_usage: Tool usage statistics
        """
        # Update common mistakes
        mistake_by_phase = {}
        for mistake in mistakes:
            phase = mistake.get("phase", "unknown")
            if phase not in mistake_by_phase:
                mistake_by_phase[phase] = []
            mistake_by_phase[phase].append(mistake)
        
        # Update or initialize common mistakes
        for phase, phase_mistakes in mistake_by_phase.items():
            if phase not in self.personalization["common_mistakes"]:
                self.personalization["common_mistakes"][phase] = []
            
            # Add new unique mistakes
            for mistake in phase_mistakes:
                mistake_type = mistake.get("type", "unknown")
                if not any(m.get("type") == mistake_type for m in self.personalization["common_mistakes"][phase]):
                    self.personalization["common_mistakes"][phase].append({
                        "type": mistake_type,
                        "count": 1,
                        "description": mistake.get("description", "")
                    })
                else:
                    # Increment count for existing mistake type
                    for m in self.personalization["common_mistakes"][phase]:
                        if m.get("type") == mistake_type:
                            m["count"] = m.get("count", 0) + 1
        
        # Update phase durations (running average)
        for phase, duration in phase_durations.items():
            if phase not in self.personalization["phase_durations"]:
                self.personalization["phase_durations"][phase] = duration
            else:
                # Simple running average
                history_count = len([r for r in self.performance_history if phase in r.get("phase_durations", {})])
                if history_count > 0:
                    current_avg = self.personalization["phase_durations"][phase]
                    self.personalization["phase_durations"][phase] = (current_avg * history_count + duration) / (history_count + 1)
        
        # Update tool preferences based on successful usage
        for phase, tools in tool_usage.items():
            if phase not in self.personalization["tool_preferences"]:
                self.personalization["tool_preferences"][phase] = tools
            else:
                # Merge tool preferences with existing data
                for tool, usage in tools.items():
                    if tool not in self.personalization["tool_preferences"][phase]:
                        self.personalization["tool_preferences"][phase][tool] = usage
                    else:
                        # Update usage statistics
                        self.personalization["tool_preferences"][phase][tool] = (
                            self.personalization["tool_preferences"][phase][tool] + usage
                        ) / 2
        
        # Update learning focus areas based on mistakes
        # Focus on phases with most mistakes
        if mistake_by_phase:
            worst_phase = max(mistake_by_phase.items(), key=lambda x: len(x[1]))
            if worst_phase[0] not in self.personalization["learning_focus"]:
                self.personalization["learning_focus"].append(worst_phase[0])
                # Keep only 3 most recent focus areas
                if len(self.personalization["learning_focus"]) > 3:
                    self.personalization["learning_focus"].pop(0)
    
    def get_personalized_guidance(self, phase: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get personalized guidance based on the user's profile.
        
        Args:
            phase: Current surgical phase
            context: Additional context information
            
        Returns:
            Dict with personalized guidance information
        """
        guidance = {
            "detail_level": self.preferences["guidance_detail_level"],
            "focus_areas": self.personalization["learning_focus"],
            "common_mistakes": self.personalization["common_mistakes"].get(phase, []),
            "experience_level": self.experience_level,
            "experience_value": self.experience_value
        }
        
        # Add phase-specific guidance
        if phase in self.personalization["common_mistakes"]:
            guidance["phase_warnings"] = [
                m["description"] for m in self.personalization["common_mistakes"][phase]
                if m.get("count", 0) > 1  # Only warn about repeated mistakes
            ]
        
        # Add recommended tools based on experience level and preferences
        if phase in self.personalization["tool_preferences"]:
            tool_prefs = self.personalization["tool_preferences"][phase]
            # Sort tools by preference
            recommended_tools = sorted(tool_prefs.items(), key=lambda x: x[1], reverse=True)
            guidance["recommended_tools"] = [tool for tool, _ in recommended_tools]
        
        return guidance
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert profile to dictionary.
        
        Returns:
            Dict representation of the profile
        """
        return {
            "user_id": self.user_id,
            "name": self.name,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "experience_level": self.experience_level,
            "preferences": self.preferences,
            "performance_history": self.performance_history,
            "personalization": self.personalization
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Load profile from dictionary.
        
        Args:
            data: Dictionary containing profile data
        """
        self.user_id = data.get("user_id", self.user_id)
        self.name = data.get("name", self.name)
        self.created_at = data.get("created_at", self.created_at)
        self.last_updated = data.get("last_updated", self.last_updated)
        self.experience_level = data.get("experience_level", self.experience_level)
        
        if self.experience_level in self.EXPERIENCE_LEVELS:
            self.experience_value = self.EXPERIENCE_LEVELS[self.experience_level]
        
        self.preferences = data.get("preferences", self.preferences)
        self.performance_history = data.get("performance_history", self.performance_history)
        self.personalization = data.get("personalization", self.personalization)
    
    def save(self, path: Optional[str] = None) -> bool:
        """
        Save profile to file.
        
        Args:
            path: Path to save profile (uses self.profile_path if None)
            
        Returns:
            bool: True if save was successful
        """
        save_path = path or self.profile_path
        if not save_path:
            logger.error("No profile path specified for saving")
            return False
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save profile
            with open(save_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            
            logger.info(f"Profile saved to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save profile: {e}")
            return False
    
    def load(self, path: Optional[str] = None) -> bool:
        """
        Load profile from file.
        
        Args:
            path: Path to load profile from (uses self.profile_path if None)
            
        Returns:
            bool: True if load was successful
        """
        load_path = path or self.profile_path
        if not load_path:
            logger.error("No profile path specified for loading")
            return False
        
        try:
            # Load profile
            with open(load_path, 'r') as f:
                data = json.load(f)
            
            # Update profile
            self.from_dict(data)
            
            logger.info(f"Profile loaded from {load_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load profile: {e}")
            return False


class ProfileManager:
    """
    Manages multiple user profiles.
    """
    
    def __init__(self, profiles_dir: str = "data/profiles"):
        """
        Initialize profile manager.
        
        Args:
            profiles_dir: Directory to store profiles
        """
        self.profiles_dir = profiles_dir
        self.profiles = {}
        
        # Create profiles directory if it doesn't exist
        os.makedirs(self.profiles_dir, exist_ok=True)
        
        # Load existing profiles
        self._load_profiles()
    
    def _load_profiles(self) -> None:
        """Load all profiles from the profiles directory."""
        try:
            for filename in os.listdir(self.profiles_dir):
                if filename.endswith('.json'):
                    user_id = filename.split('.')[0]
                    profile_path = os.path.join(self.profiles_dir, filename)
                    
                    # Create profile instance
                    profile = UserProfile(user_id=user_id, profile_path=profile_path)
                    if profile.load():
                        self.profiles[user_id] = profile
        except Exception as e:
            logger.error(f"Failed to load profiles: {e}")
    
    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Get a user profile by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            UserProfile or None if not found
        """
        return self.profiles.get(user_id)
    
    def create_profile(self, user_id: str, name: str = "", 
                       experience_level: str = "intermediate") -> UserProfile:
        """
        Create a new user profile.
        
        Args:
            user_id: User ID
            name: User name
            experience_level: Experience level
            
        Returns:
            Newly created UserProfile
        """
        # Create profile path
        profile_path = os.path.join(self.profiles_dir, f"{user_id}.json")
        
        # Create profile
        profile = UserProfile(
            user_id=user_id,
            name=name,
            experience_level=experience_level,
            profile_path=profile_path
        )
        
        # Save profile
        profile.save()
        
        # Add to profiles
        self.profiles[user_id] = profile
        
        return profile
    
    def update_profile(self, profile: UserProfile) -> bool:
        """
        Update a profile in the manager.
        
        Args:
            profile: Updated profile
            
        Returns:
            bool: True if update was successful
        """
        # Check if profile exists
        if profile.user_id not in self.profiles:
            logger.warning(f"Profile {profile.user_id} does not exist")
            return False
        
        # Update profile
        self.profiles[profile.user_id] = profile
        
        # Save profile
        return profile.save()
    
    def delete_profile(self, user_id: str) -> bool:
        """
        Delete a profile.
        
        Args:
            user_id: User ID
            
        Returns:
            bool: True if deletion was successful
        """
        # Check if profile exists
        if user_id not in self.profiles:
            logger.warning(f"Profile {user_id} does not exist")
            return False
        
        # Get profile path
        profile = self.profiles[user_id]
        profile_path = profile.profile_path
        
        # Remove from profiles
        del self.profiles[user_id]
        
        # Delete file
        if profile_path and os.path.exists(profile_path):
            try:
                os.remove(profile_path)
                return True
            except Exception as e:
                logger.error(f"Failed to delete profile file: {e}")
                return False
        
        return True
    
    def get_all_profiles(self) -> Dict[str, UserProfile]:
        """
        Get all profiles.
        
        Returns:
            Dict of user_id -> UserProfile
        """
        return self.profiles 