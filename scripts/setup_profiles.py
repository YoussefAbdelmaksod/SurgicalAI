#!/usr/bin/env python
"""
Setup script for SurgicalAI user profiles.

This script creates and initializes user profiles for the SurgicalAI system,
allowing for personalized guidance during surgical procedures.
"""

import os
import sys
import argparse
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from utils.user_profiles import ProfileManager, UserProfile
from utils.helpers import setup_logging, load_config

def create_default_profile(profile_manager, config):
    """
    Create default user profile.
    
    Args:
        profile_manager: Profile manager instance
        config: Configuration dictionary
        
    Returns:
        Created user profile
    """
    # Get default profile settings from config
    default_id = config.get('app', {}).get('default_user_id', 'default')
    default_name = config.get('app', {}).get('default_user_name', 'Default User')
    default_experience = config.get('app', {}).get('default_experience_level', 'intermediate')
    
    # Check if profile already exists
    existing_profile = profile_manager.get_profile(default_id)
    if existing_profile:
        logger.info(f"Default profile already exists: {default_id}")
        return existing_profile
        
    # Create default profile
    logger.info(f"Creating default profile: {default_id}")
    profile = profile_manager.create_profile(
        user_id=default_id,
        name=default_name,
        experience_level=default_experience
    )
    
    # Set default preferences
    profile.update_preference('voice_guidance', True)
    profile.update_preference('guidance_detail_level', 'standard')
    profile.update_preference('voice_gender', 'female')
    profile.update_preference('critical_warnings_only', False)
    
    # Save profile
    profile_manager.update_profile(profile)
    
    logger.info(f"Default profile created successfully: {default_id}")
    return profile
    
def create_sample_profiles(profile_manager):
    """
    Create sample profiles for different experience levels.
    
    Args:
        profile_manager: Profile manager instance
        
    Returns:
        List of created profiles
    """
    profiles = []
    
    # Sample profiles
    sample_profiles = [
        {
            'user_id': 'novice_surgeon',
            'name': 'Novice Surgeon',
            'experience_level': 'novice',
            'preferences': {
                'guidance_detail_level': 'detailed',
                'voice_rate': 140,
                'critical_warnings_only': False,
                'show_phase_recommendations': True,
                'show_tool_recommendations': True
            }
        },
        {
            'user_id': 'resident_surgeon',
            'name': 'Resident Surgeon',
            'experience_level': 'junior',
            'preferences': {
                'guidance_detail_level': 'standard',
                'voice_rate': 150,
                'critical_warnings_only': False,
                'show_phase_recommendations': True,
                'show_tool_recommendations': True
            }
        },
        {
            'user_id': 'attending_surgeon',
            'name': 'Attending Surgeon',
            'experience_level': 'senior',
            'preferences': {
                'guidance_detail_level': 'minimal',
                'voice_rate': 170,
                'critical_warnings_only': True,
                'show_phase_recommendations': False,
                'show_tool_recommendations': False
            }
        }
    ]
    
    # Create each sample profile
    for profile_data in sample_profiles:
        # Check if profile already exists
        existing_profile = profile_manager.get_profile(profile_data['user_id'])
        if existing_profile:
            logger.info(f"Sample profile already exists: {profile_data['user_id']}")
            profiles.append(existing_profile)
            continue
            
        # Create profile
        logger.info(f"Creating sample profile: {profile_data['user_id']}")
        profile = profile_manager.create_profile(
            user_id=profile_data['user_id'],
            name=profile_data['name'],
            experience_level=profile_data['experience_level']
        )
        
        # Set preferences
        for key, value in profile_data.get('preferences', {}).items():
            profile.update_preference(key, value)
            
        # Save profile
        profile_manager.update_profile(profile)
        
        profiles.append(profile)
        logger.info(f"Sample profile created successfully: {profile_data['user_id']}")
        
    return profiles
    
def setup_profiles(config_path, create_samples=False, force_reset=False):
    """
    Setup user profiles for SurgicalAI.
    
    Args:
        config_path: Path to configuration file
        create_samples: Whether to create sample profiles
        force_reset: Whether to force reset existing profiles
        
    Returns:
        bool: True if setup was successful
    """
    # Load configuration
    config = load_config(config_path)
    
    # Get profiles directory
    profiles_dir = config.get('paths', {}).get('profiles_dir', 'data/profiles')
    
    # Handle force reset
    if force_reset and os.path.exists(profiles_dir):
        import shutil
        logger.warning(f"Force reset requested. Removing profiles directory: {profiles_dir}")
        shutil.rmtree(profiles_dir)
    
    # Create profiles directory if it doesn't exist
    os.makedirs(profiles_dir, exist_ok=True)
    
    # Initialize profile manager
    profile_manager = ProfileManager(profiles_dir=profiles_dir)
    
    # Create default profile
    default_profile = create_default_profile(profile_manager, config)
    
    # Create sample profiles if requested
    if create_samples:
        sample_profiles = create_sample_profiles(profile_manager)
    
    return True
    
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Setup SurgicalAI user profiles')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--samples', action='store_true',
                        help='Create sample profiles for different experience levels')
    parser.add_argument('--reset', action='store_true',
                        help='Force reset existing profiles')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    return parser.parse_args()
    
if __name__ == '__main__':
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(log_level)
    
    logger.info("Setting up SurgicalAI user profiles...")
    
    # Setup profiles
    success = setup_profiles(
        config_path=args.config,
        create_samples=args.samples,
        force_reset=args.reset
    )
    
    if success:
        logger.info("Profile setup completed successfully")
    else:
        logger.error("Profile setup failed")
        sys.exit(1)
    
    sys.exit(0) 