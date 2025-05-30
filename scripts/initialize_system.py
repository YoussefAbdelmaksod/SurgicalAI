#!/usr/bin/env python
"""
Initialization script for SurgicalAI system.

This script initializes and tests all components of the SurgicalAI system,
ensuring that everything is properly set up for surgical video analysis.
"""

import os
import sys
import argparse
import logging
import yaml
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from utils.helpers import setup_logging, load_config, get_device
from app.main import SurgicalAISystem
from utils.user_profiles import ProfileManager
from models.voice_assistant import VoiceAssistant
from models.gpt_guidance import SurgicalGPTGuidance

def create_directories(config):
    """
    Create necessary directories for the system.
    
    Args:
        config: Configuration dictionary
    """
    # Create directory structure
    dirs_to_create = [
        'data/profiles',
        'models/weights/phase',
        'models/weights/tool',
        'models/weights/mistake',
        'models/weights/guidance',
        'logs',
        'output'
    ]
    
    # Create additional directories from config
    if 'paths' in config:
        for key, path in config['paths'].items():
            if isinstance(path, str) and not path.endswith('.json') and not path.endswith('.yaml') and not path.endswith('.yml'):
                dirs_to_create.append(path)
    
    # Create directories
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Directory created/verified: {dir_path}")

def initialize_user_profiles():
    """
    Initialize user profiles for the system.
    
    Returns:
        ProfileManager instance
    """
    # Set up profile manager
    profiles_dir = 'data/profiles'
    os.makedirs(profiles_dir, exist_ok=True)
    
    profile_manager = ProfileManager(profiles_dir=profiles_dir)
    
    # Create default profile if it doesn't exist
    default_profile = profile_manager.get_profile('default')
    if not default_profile:
        logger.info("Creating default user profile")
        default_profile = profile_manager.create_profile(
            user_id='default',
            name='Default User',
            experience_level='intermediate'
        )
    else:
        logger.info("Default user profile already exists")
    
    return profile_manager

def test_voice_assistant(profile_manager):
    """
    Test the voice assistant functionality.
    
    Args:
        profile_manager: ProfileManager instance
        
    Returns:
        bool: True if test was successful
    """
    try:
        # Get default profile
        default_profile = profile_manager.get_profile('default')
        
        # Create voice assistant
        logger.info("Testing voice assistant...")
        voice_assistant = VoiceAssistant(
            feedback_level="standard",
            critical_warnings_only=False,
            enable_voice_commands=False,
            user_profile=default_profile
        )
        
        # Test phase guidance
        logger.info("Testing phase guidance...")
        voice_assistant.provide_phase_guidance("Calot's Triangle Dissection", is_transition=True)
        time.sleep(3)  # Wait for speech to complete
        
        # Test mistake warning
        logger.info("Testing mistake warning...")
        mistake_info = {
            "type": "anatomical_risk",
            "description": "Dissection too close to common bile duct",
            "risk_level": 0.75
        }
        voice_assistant.warn_about_mistake(mistake_info)
        time.sleep(3)  # Wait for speech to complete
        
        # Test tool guidance
        logger.info("Testing tool guidance...")
        current_tools = ["grasper", "hook"]
        recommended_tools = ["grasper", "clipper", "scissors"]
        voice_assistant.provide_tool_guidance(current_tools, recommended_tools)
        time.sleep(3)  # Wait for speech to complete
        
        # Clean up
        voice_assistant.cleanup()
        
        logger.info("Voice assistant test completed successfully")
        return True
    except Exception as e:
        logger.error(f"Voice assistant test failed: {e}")
        return False

def test_gpt_guidance(config, profile_manager):
    """
    Test the GPT guidance functionality.
    
    Args:
        config: Configuration dictionary
        profile_manager: ProfileManager instance
        
    Returns:
        bool: True if test was successful
    """
    try:
        # Get default profile
        default_profile = profile_manager.get_profile('default')
        
        # Get procedure knowledge path
        procedure_knowledge_path = config.get('paths', {}).get('procedure_knowledge', 'data/procedure_knowledge.json')
        
        # Create GPT guidance system
        logger.info("Testing GPT guidance system...")
        guidance_system = SurgicalGPTGuidance(
            model_name="gpt2",
            procedure_knowledge_path=procedure_knowledge_path
        )
        
        # Test personalized guidance
        logger.info("Testing personalized guidance...")
        context = {
            "phase": "calot_triangle_dissection",
            "detected_tools": ["grasper", "hook"],
            "anatomical_structures": ["gallbladder", "cystic_duct"],
            "cvs_achieved": False,
            "previous_actions": ["Initial dissection started"]
        }
        
        guidance = guidance_system.get_personalized_guidance(context, default_profile)
        logger.info(f"Guidance: {guidance['primary_guidance']}")
        
        # Test phase guidance
        logger.info("Testing phase guidance...")
        phase_guidance = guidance_system.get_phase_guidance(
            phase_name="clipping_and_cutting",
            is_transition=True,
            user_profile=default_profile
        )
        logger.info(f"Phase guidance: {phase_guidance}")
        
        # Test mistake guidance
        logger.info("Testing mistake guidance...")
        mistake_info = {
            "type": "anatomical_risk",
            "description": "Dissection too close to common bile duct",
            "risk_level": 0.75
        }
        mistake_guidance = guidance_system.get_mistake_guidance(
            mistake_info=mistake_info,
            context=context,
            user_profile=default_profile
        )
        logger.info(f"Mistake guidance: {mistake_guidance}")
        
        logger.info("GPT guidance test completed successfully")
        return True
    except Exception as e:
        logger.error(f"GPT guidance test failed: {e}")
        return False

def test_full_system(config):
    """
    Test the full SurgicalAI system.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        bool: True if test was successful
    """
    try:
        # Create SurgicalAI system
        logger.info("Initializing SurgicalAI system...")
        system = SurgicalAISystem(
            config_path='config/default_config.yaml',
            use_ensemble=True,
            use_gpt=True
        )
        
        # Test model initialization
        if not system.models_initialized:
            logger.warning("SurgicalAI system models not fully initialized")
        else:
            logger.info("SurgicalAI system models initialized successfully")
        
        # Test user profile integration
        logger.info("Testing user profile integration...")
        profile_manager = ProfileManager(profiles_dir='data/profiles')
        default_profile = profile_manager.get_profile('default')
        
        if default_profile:
            system.set_user_profile(user_profile=default_profile)
            logger.info("User profile integrated successfully")
        else:
            logger.warning("Could not load default user profile")
        
        logger.info("SurgicalAI system initialization completed successfully")
        return True
    except Exception as e:
        logger.error(f"SurgicalAI system initialization failed: {e}")
        return False

def initialize_system(config_path, skip_voice=False, skip_gpt=False, skip_full=False):
    """
    Initialize and test the SurgicalAI system.
    
    Args:
        config_path: Path to configuration file
        skip_voice: Whether to skip voice assistant test
        skip_gpt: Whether to skip GPT guidance test
        skip_full: Whether to skip full system test
        
    Returns:
        bool: True if initialization was successful
    """
    # Load configuration
    config = load_config(config_path)
    
    # Create directories
    create_directories(config)
    
    # Initialize user profiles
    profile_manager = initialize_user_profiles()
    
    # Run tests
    success = True
    
    # Test voice assistant
    if not skip_voice:
        voice_success = test_voice_assistant(profile_manager)
        success = success and voice_success
    
    # Test GPT guidance
    if not skip_gpt:
        gpt_success = test_gpt_guidance(config, profile_manager)
        success = success and gpt_success
    
    # Test full system
    if not skip_full:
        system_success = test_full_system(config)
        success = success and system_success
    
    return success

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Initialize SurgicalAI system')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--skip-voice', action='store_true',
                        help='Skip voice assistant test')
    parser.add_argument('--skip-gpt', action='store_true',
                        help='Skip GPT guidance test')
    parser.add_argument('--skip-full', action='store_true',
                        help='Skip full system test')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    return parser.parse_args()

if __name__ == '__main__':
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(log_level)
    
    logger.info("Initializing SurgicalAI system...")
    
    # Initialize system
    success = initialize_system(
        config_path=args.config,
        skip_voice=args.skip_voice,
        skip_gpt=args.skip_gpt,
        skip_full=args.skip_full
    )
    
    if success:
        logger.info("SurgicalAI system initialization completed successfully")
    else:
        logger.error("SurgicalAI system initialization failed")
        sys.exit(1)
    
    sys.exit(0) 