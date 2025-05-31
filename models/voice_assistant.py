"""
Voice assistant module for SurgicalAI.

This module implements real-time voice guidance and feedback during surgical procedures,
with context-aware warnings and instructions.
"""

import torch
import torch.nn as nn
import logging
import os
import threading
import queue
import time
import random
from typing import List, Dict, Optional, Union, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Try importing TTS libraries
try:
    import pyttsx3
    TTS_ENGINE_AVAILABLE = True
except ImportError:
    logger.warning("pyttsx3 not installed. Using fallback TTS implementation.")
    TTS_ENGINE_AVAILABLE = False

# Try importing STT libraries for voice command recognition
try:
    import speech_recognition as sr
    STT_AVAILABLE = True
except ImportError:
    logger.warning("speech_recognition not installed. Voice command recognition will not be available.")
    STT_AVAILABLE = False


class TTSEngine:
    """Text-to-Speech engine for voice feedback."""
    
    def __init__(self, voice_id=None, rate=175, volume=0.9):
        """
        Initialize TTS engine.
        
        Args:
            voice_id: Voice ID to use (None for default)
            rate: Speech rate (words per minute)
            volume: Volume level (0.0 to 1.0)
        """
        self.voice_id = voice_id
        self.rate = rate
        self.volume = volume
        self.engine = None
        
        # Message queue for asynchronous speech generation
        self.message_queue = queue.Queue()
        self.is_running = False
        self.thread = None
        
        # Initialize TTS engine
        self._initialize_engine()
        
    def _initialize_engine(self):
        """Initialize the appropriate TTS engine based on availability."""
        if TTS_ENGINE_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                
                # Configure engine properties
                self.engine.setProperty('rate', self.rate)
                self.engine.setProperty('volume', self.volume)
                
                # Set voice if specified
                if self.voice_id is not None:
                    self.engine.setProperty('voice', self.voice_id)
                
                logger.info("TTS engine initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize TTS engine: {str(e)}")
                self.engine = None
        else:
            logger.warning("Using fallback TTS implementation.")
            self.engine = None
            
    def get_available_voices(self):
        """Get list of available voices."""
        if self.engine is not None:
            voices = self.engine.getProperty('voices')
            return [(voice.id, voice.name) for voice in voices]
        return []
            
    def speak(self, text, blocking=False, priority=False):
        """
        Convert text to speech.
        
        Args:
            text: Text to speak
            blocking: Whether to block until speech is complete
            priority: Whether this is a high-priority message
        """
        if not text:
            return
            
        if blocking:
            self._speak_now(text)
        else:
            # Add to queue for asynchronous processing
            if priority:
                # For priority messages, clear the queue first
                with self.message_queue.mutex:
                    self.message_queue.queue.clear()
            
            self.message_queue.put(text)
            
            # Start processing thread if not already running
            if not self.is_running:
                self._start_processing_thread()
                
    def _speak_now(self, text):
        """Speak text immediately."""
        if self.engine is not None:
            self.engine.say(text)
            self.engine.runAndWait()
        else:
            # Fallback: just log the text
            logger.info(f"TTS Fallback: {text}")
            
    def _start_processing_thread(self):
        """Start background thread for processing speech queue."""
        if self.thread is None or not self.thread.is_alive():
            self.is_running = True
            self.thread = threading.Thread(target=self._process_queue)
            self.thread.daemon = True  # Thread will exit when main program exits
            self.thread.start()
            
    def _process_queue(self):
        """Process message queue in background thread."""
        while self.is_running:
            try:
                # Get message from queue with timeout to allow thread to exit
                text = self.message_queue.get(timeout=0.5)
                
                # Speak the message
                self._speak_now(text)
                
                # Mark task as done
                self.message_queue.task_done()
            except queue.Empty:
                # No messages in queue, continue waiting
                pass
            except Exception as e:
                logger.error(f"Error in TTS processing thread: {str(e)}")
        
        logger.debug("TTS processing thread exiting.")
            
    def stop(self):
        """Stop TTS processing."""
        self.is_running = False
            
        # Clear the queue
        with self.message_queue.mutex:
            self.message_queue.queue.clear()
        
    def cleanup(self):
        """Clean up resources."""
        self.stop()
        
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
            
        self.engine = None


class STTEngine:
    """Speech-to-Text engine for voice command recognition."""
    
    def __init__(self, language="en-US", timeout=5, phrase_time_limit=5):
        """
        Initialize STT engine.
        
        Args:
            language: Language code
            timeout: Recognition timeout in seconds
            phrase_time_limit: Max phrase duration in seconds
        """
        self.language = language
        self.timeout = timeout
        self.phrase_time_limit = phrase_time_limit
        
        # Initialize recognizer
        if STT_AVAILABLE:
            self.recognizer = sr.Recognizer()
            logger.info("STT engine initialized successfully.")
        else:
            self.recognizer = None
            logger.warning("STT engine not available.")
            
    def listen_for_command(self, commands_list=None, blocking=True):
        """
        Listen for voice command.
        
        Args:
            commands_list: List of expected commands for improved recognition
            blocking: Whether to block until command is recognized
            
        Returns:
            Recognized command text or None if not recognized
        """
        if not STT_AVAILABLE or self.recognizer is None:
            logger.error("STT engine not available.")
            return None
            
        # Non-blocking mode not yet implemented
        if not blocking:
            logger.warning("Non-blocking STT not implemented yet.")
            return None
            
        try:
            # Initialize microphone
            with sr.Microphone() as source:
                logger.info("Listening for command...")
                
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source)
                
                # Listen for command
                audio = self.recognizer.listen(
                    source, 
                    timeout=self.timeout, 
                    phrase_time_limit=self.phrase_time_limit
                )
                
                # Recognize command
                if commands_list:
                    # If we have a list of expected commands, use a more restrictive model
                    command = self.recognizer.recognize_sphinx(
                        audio,
                        language=self.language,
                        keyword_entries=[(cmd, 0.5) for cmd in commands_list]
                    )
                else:
                    # Otherwise use Google's more accurate but internet-dependent model
                    command = self.recognizer.recognize_google(
                        audio,
                        language=self.language
                    )
                
                logger.info(f"Recognized command: {command}")
                return command
        except sr.WaitTimeoutError:
            logger.warning("Command recognition timed out.")
        except sr.UnknownValueError:
            logger.warning("Command not recognized.")
        except sr.RequestError as e:
            logger.error(f"Recognition service error: {str(e)}")
        except Exception as e:
            logger.error(f"Error in command recognition: {str(e)}")
            
        return None


class VoiceAssistant:
    """
    Voice assistant for surgical guidance and feedback.
    
    Provides real-time voice notifications, warnings, and instructions
    based on surgical context.
    """
    
    def __init__(self, voice_id=None, feedback_level="full", 
                 critical_warnings_only=False, enable_voice_commands=False,
                 user_profile=None):
        """
        Initialize voice assistant.
        
        Args:
            voice_id: Voice ID to use (None for default)
            feedback_level: Level of feedback detail ("minimal", "standard", "full")
            critical_warnings_only: Whether to only provide critical warnings
            enable_voice_commands: Whether to enable voice command recognition
            user_profile: Optional user profile for personalized guidance
        """
        self.feedback_level = feedback_level
        self.critical_warnings_only = critical_warnings_only
        self.enable_voice_commands = enable_voice_commands
        self.user_profile = user_profile
        
        # Initialize TTS engine
        self.tts_engine = TTSEngine(voice_id=voice_id)
        
        # Initialize STT engine if voice commands enabled
        self.stt_engine = None
        if enable_voice_commands and STT_AVAILABLE:
            self.stt_engine = STTEngine()
            
        # Message priority levels
        self.priority_levels = {
            "critical": 0,  # Highest priority
            "warning": 1,
            "instruction": 2,
            "information": 3,
            "feedback": 4    # Lowest priority
        }
        
        # Cooldown timers to prevent message spam
        self.last_message_time = {}
        self.cooldown_periods = {
            "critical": 0,      # No cooldown for critical messages
            "warning": 5,       # 5 seconds cooldown
            "instruction": 8,   # 8 seconds cooldown
            "information": 15,  # 15 seconds cooldown
            "feedback": 20      # 20 seconds cooldown
        }
        
        # Message history
        self.message_history = []
        self.max_history = 100
        
        # Current surgical context
        self.current_context = {
            "phase": None,
            "detected_tools": [],
            "active_warnings": []
        }
        
        # Voice command handlers
        self.voice_commands = {
            "repeat": self._handle_repeat_command,
            "stop": self._handle_stop_command,
            "help": self._handle_help_command,
            "status": self._handle_status_command
        }
        
        # Response templates
        self.responses = {
            "help_text": "Available commands: repeat, stop, help, status. You can ask for guidance or information about the current phase.",
            "no_context": "I don't have enough context to provide guidance.",
            "welcome": "Surgical AI assistant is ready. Voice guidance is active."
        }
        
        # Personalized guidance phrases based on experience level
        self.experience_level_phrases = {
            "novice": {
                "instruction_prefix": ["Remember to ", "Make sure to ", "It's important to ", "Don't forget to "],
                "warning_prefix": ["Be careful with ", "Watch out for ", "Pay close attention to ", "Be cautious of "],
                "instruction_detail": "detailed",
                "pace": "slower"
            },
            "junior": {
                "instruction_prefix": ["Remember to ", "You should ", "Please ", "Make sure you "],
                "warning_prefix": ["Be careful with ", "Watch out for ", "Pay attention to "],
                "instruction_detail": "standard",
                "pace": "standard"
            },
            "intermediate": {
                "instruction_prefix": ["Please ", "Now ", ""],
                "warning_prefix": ["Watch for ", "Be aware of ", "Note the "],
                "instruction_detail": "standard",
                "pace": "standard"
            },
            "senior": {
                "instruction_prefix": ["", "Consider ", "You may want to "],
                "warning_prefix": ["Note ", "Be aware of ", "Watch for "],
                "instruction_detail": "concise",
                "pace": "faster"
            },
            "expert": {
                "instruction_prefix": ["", "Consider "],
                "warning_prefix": ["Note ", "Be aware of "],
                "instruction_detail": "minimal",
                "pace": "faster"
            }
        }
        
        # Welcome message
        self.speak(self.responses["welcome"], priority_level="information")
    
    def speak(self, message, priority_level="information", context=None, blocking=False):
        """
        Speak a message with the given priority level.
        
        Args:
            message: Message text to speak
            priority_level: Priority level ("critical", "warning", "instruction", "information", "feedback")
            context: Additional context for message processing
            blocking: Whether to block until speech is complete
        """
        # Check if this message should be suppressed based on feedback level
        if self.critical_warnings_only and priority_level not in ["critical", "warning"]:
            return
            
        if self.feedback_level == "minimal" and priority_level in ["information", "feedback"]:
            return
            
        # Check cooldown period
        now = time.time()
        if priority_level in self.last_message_time:
            elapsed = now - self.last_message_time[priority_level]
            if elapsed < self.cooldown_periods[priority_level]:
                # Message is in cooldown period, suppress unless critical
                if priority_level != "critical":
                    return
        
        # Update last message time
        self.last_message_time[priority_level] = now
        
        # Add to message history
        self.message_history.append((message, priority_level))
        if len(self.message_history) > self.max_history:
            self.message_history.pop(0)
            
        # Determine if this is a high-priority message
        is_priority = priority_level in ["critical", "warning"]
        
        # Speak the message
        self.tts_engine.speak(message, blocking=blocking, priority=is_priority)
        
    def update_context(self, context_data):
        """
        Update the current surgical context.
        
        Args:
            context_data: Dictionary with context data
        """
        # Update context
        for key, value in context_data.items():
            self.current_context[key] = value
        
    def provide_personalized_guidance(self, message, priority_level="instruction", context=None):
        """
        Provide guidance personalized to the surgeon's experience level.
        
        Args:
            message: Base message text
            priority_level: Priority level
            context: Additional context information
        """
        if not self.user_profile:
            # No user profile, provide standard guidance
            self.speak(message, priority_level=priority_level, context=context)
            return
            
        # Get experience level
        experience_level = self.user_profile.experience_level
        if experience_level not in self.experience_level_phrases:
            experience_level = "intermediate"  # Default if invalid
            
        # Get personalization phrases
        phrases = self.experience_level_phrases[experience_level]
        
        # Personalize message based on priority level and experience
        if priority_level == "instruction":
            prefix = random.choice(phrases["instruction_prefix"])
            personalized_message = f"{prefix}{message[0].lower()}{message[1:]}" if message else ""
        elif priority_level == "warning":
            prefix = random.choice(phrases["warning_prefix"])
            personalized_message = f"{prefix}{message[0].lower()}{message[1:]}" if message else ""
        else:
            personalized_message = message
            
        # Adjust TTS engine rate based on experience level
        original_rate = self.tts_engine.rate
        if phrases["pace"] == "slower":
            self.tts_engine.rate = max(120, original_rate - 30)
        elif phrases["pace"] == "faster":
            self.tts_engine.rate = min(220, original_rate + 20)
            
        # Speak personalized message
        self.speak(personalized_message, priority_level=priority_level, context=context)
        
        # Restore original rate
        self.tts_engine.rate = original_rate
        
    def provide_phase_guidance(self, phase_name, is_transition=False):
        """
        Provide guidance for the current surgical phase.
        
        Args:
            phase_name: Name of the current surgical phase
            is_transition: Whether this is a transition to a new phase
        """
        if is_transition:
            # Get personalized phase guidance if user profile is available
            if self.user_profile:
                # Get personalized guidance
                context = {"is_transition": True}
                phase_guidance = self.user_profile.get_personalized_guidance(phase_name, context)
                
                # Basic announcement
                self.provide_personalized_guidance(
                    f"Entering {phase_name} phase.", 
                    priority_level="instruction"
                )
                
                # Add personalized warnings if available
                if "phase_warnings" in phase_guidance and phase_guidance["phase_warnings"]:
                    # Pick most relevant warning
                    warning = phase_guidance["phase_warnings"][0]
                    self.provide_personalized_guidance(
                        f"Pay attention to {warning}",
                        priority_level="warning"
                    )
                
                # Add tool recommendations if available and enabled
                if "recommended_tools" in phase_guidance and self.user_profile.preferences["show_tool_recommendations"]:
                    tools = phase_guidance["recommended_tools"][:2]  # Top 2 tools
                    if tools:
                        tools_str = " and ".join(tools)
                        self.provide_personalized_guidance(
                            f"Recommended tools: {tools_str}",
                            priority_level="information"
                        )
            else:
                # Standard non-personalized guidance
                self.speak(f"Entering {phase_name} phase.", priority_level="instruction")
            
            # Provide additional guidance based on the phase
            if phase_name == "Preparation":
                self.speak("Please ensure all tools are properly sterilized and ready.", 
                           priority_level="instruction")
            elif phase_name == "Calot's Triangle Dissection":
                self.speak("Critical phase. Identify cystic duct and artery.", 
                           priority_level="instruction")
            elif phase_name == "Clipping and Cutting":
                self.speak("Ensure correct placement of clips before cutting.", 
                           priority_level="warning")
            elif phase_name == "Gallbladder Dissection":
                self.speak("Maintain gentle upward traction during dissection.", 
                           priority_level="instruction")
            elif phase_name == "Gallbladder Extraction":
                self.speak("Verify complete removal and inspect for bleeding.", 
                           priority_level="instruction")
            elif phase_name == "Cleaning and Coagulation":
                self.speak("Check for bile leakage and ensure hemostasis.", 
                           priority_level="instruction")
            elif phase_name == "Closing":
                self.speak("Verify no instruments or sponges remain in the cavity.", 
                           priority_level="warning")
    
    def warn_about_mistake(self, mistake_info):
        """
        Provide voice warning about a detected mistake.
        
        Args:
            mistake_info: Dictionary with mistake information
        """
        mistake_type = mistake_info.get("type", "unknown")
        description = mistake_info.get("description", "")
        risk_level = mistake_info.get("risk_level", 0.0)
        
        # Determine priority level based on risk level
        if risk_level >= 0.8:
            priority_level = "critical"
            prefix = "Critical error: "
        elif risk_level >= 0.5:
            priority_level = "warning"
            prefix = "Warning: "
        else:
            priority_level = "information"
            prefix = "Note: "
            
        # Create warning message
        if description:
            warning_message = f"{prefix}{description}"
        else:
            warning_message = f"{prefix}Potential error detected of type {mistake_type}"
            
        # Use personalized guidance if available
        if self.user_profile:
            self.provide_personalized_guidance(
                warning_message, 
                priority_level=priority_level
            )
        else:
            self.speak(warning_message, priority_level=priority_level)
    
    def provide_tool_guidance(self, current_tools, recommended_tools):
        """
        Provide guidance on tool usage.
        
        Args:
            current_tools: List of currently detected tools
            recommended_tools: List of recommended tools for the current phase
        """
        # Check if any recommended tools are not being used
        missing_tools = [tool for tool in recommended_tools if tool not in current_tools]
        
        if missing_tools and self.feedback_level != "minimal":
            missing_tool = missing_tools[0]  # Just mention the first missing tool
            
            # Use personalized guidance if available
            if self.user_profile:
                self.provide_personalized_guidance(
                    f"Consider using {missing_tool} for this step.",
                    priority_level="information"
                )
            else:
                self.speak(f"Consider using {missing_tool} for this step.", 
                           priority_level="information")
    
    def notify_critical_view_achieved(self, is_achieved, missing_criteria=None):
        """
        Notify about critical view of safety status.
        
        Args:
            is_achieved: Whether critical view of safety is achieved
            missing_criteria: List of missing criteria if not achieved
        """
        if is_achieved:
            # Use personalized guidance if available
            if self.user_profile:
                self.provide_personalized_guidance(
                    "Critical view of safety achieved. Safe to proceed.",
                    priority_level="instruction"
                )
            else:
                self.speak("Critical view of safety achieved. Safe to proceed.", 
                           priority_level="instruction")
        elif missing_criteria and self.feedback_level != "minimal":
            criteria_str = ", ".join(missing_criteria)
            
            # Use personalized guidance if available
            if self.user_profile:
                self.provide_personalized_guidance(
                    f"Critical view not yet achieved. Missing: {criteria_str}",
                    priority_level="warning"
                )
            else:
                self.speak(f"Critical view not yet achieved. Missing: {criteria_str}", 
                           priority_level="warning")
        
    def acknowledge_correct_action(self):
        """Provide positive feedback for correct action."""
        if self.feedback_level == "full":
            messages = [
                "Correct technique.",
                "Good progress.",
                "Well executed.",
                "Proper approach."
            ]
            message = random.choice(messages)
            
            # Use personalized guidance if available
            if self.user_profile:
                self.provide_personalized_guidance(message, priority_level="feedback")
            else:
                self.speak(message, priority_level="feedback")
            
    def listen_for_command(self):
        """
        Listen for voice command and handle it.
        
        Returns:
            True if command was recognized and handled, False otherwise
        """
        if not self.enable_voice_commands or self.stt_engine is None:
            return False
            
        # Get list of supported commands
        commands = list(self.voice_commands.keys())
        
        # Listen for command
        command = self.stt_engine.listen_for_command(commands_list=commands)
        
        if command and command in self.voice_commands:
            # Call appropriate handler
            self.voice_commands[command]()
            return True
            
        return False
        
    def _handle_repeat_command(self):
        """Handle 'repeat' command by repeating last message."""
        if self.message_history:
            last_message, priority = self.message_history[-1]
            self.speak(f"Repeating: {last_message}", priority_level=priority)
        else:
            self.speak("No messages to repeat.", priority_level="information")
            
    def _handle_stop_command(self):
        """Handle 'stop' command by stopping all audio."""
        self.tts_engine.stop()
        self.speak("Voice guidance paused.", priority_level="information", blocking=True)
            
    def _handle_help_command(self):
        """Handle 'help' command by providing help information."""
        self.speak(self.responses["help_text"], priority_level="information")
            
    def _handle_status_command(self):
        """Handle 'status' command by providing system status."""
        self.speak("System is operational.", priority_level="information")
        
    def cleanup(self):
        """Clean up resources."""
        if self.tts_engine:
            self.tts_engine.cleanup() 