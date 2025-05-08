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
from typing import List, Dict, Optional, Union, Any

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
        """Process speech queue in background thread."""
        try:
            while self.is_running:
                try:
                    # Get next message from queue with timeout
                    text = self.message_queue.get(timeout=0.5)
                    self._speak_now(text)
                    self.message_queue.task_done()
                except queue.Empty:
                    # Queue is empty, check if we should exit
                    if self.message_queue.empty():
                        # Exit if queue remains empty
                        self.is_running = False
                except Exception as e:
                    logger.error(f"Error in TTS processing: {str(e)}")
        finally:
            self.is_running = False
            
    def stop(self):
        """Stop all speech and clear queue."""
        if self.engine is not None:
            self.engine.stop()
            
        # Clear the message queue
        with self.message_queue.mutex:
            self.message_queue.queue.clear()
            
        self.is_running = False
        
    def cleanup(self):
        """Clean up resources."""
        self.stop()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)


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
            with sr.Microphone() as source:
                logger.info("Listening for command...")
                
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen for audio
                audio = self.recognizer.listen(
                    source, 
                    timeout=self.timeout, 
                    phrase_time_limit=self.phrase_time_limit
                )
                
                logger.info("Audio captured, recognizing...")
                
                # Recognize speech using Google Speech Recognition
                text = self.recognizer.recognize_google(audio, language=self.language)
                
                logger.info(f"Recognized: {text}")
                
                # If commands list provided, find best match
                if commands_list:
                    text = self._find_best_command_match(text, commands_list)
                    
                return text
                
        except sr.WaitTimeoutError:
            logger.info("No speech detected within timeout period.")
        except sr.UnknownValueError:
            logger.info("Speech recognition could not understand audio.")
        except sr.RequestError as e:
            logger.error(f"Could not request results; {str(e)}")
        except Exception as e:
            logger.error(f"Error in speech recognition: {str(e)}")
            
        return None
        
    def _find_best_command_match(self, text, commands_list):
        """Find best matching command from recognized text."""
        text = text.lower()
        
        # First check for exact matches
        for command in commands_list:
            if command.lower() == text:
                return command
                
        # Then check for commands contained in the text
        for command in commands_list:
            if command.lower() in text:
                return command
                
        # If no match, return original text
        return text


class VoiceAssistant:
    """
    Voice assistant for surgical guidance and feedback.
    
    Provides real-time voice notifications, warnings, and instructions
    based on surgical context.
    """
    
    def __init__(self, voice_id=None, feedback_level="full", 
                 critical_warnings_only=False, enable_voice_commands=False):
        """
        Initialize voice assistant.
        
        Args:
            voice_id: Voice ID to use (None for default)
            feedback_level: Level of feedback detail ("minimal", "standard", "full")
            critical_warnings_only: Whether to only provide critical warnings
            enable_voice_commands: Whether to enable voice command recognition
        """
        self.feedback_level = feedback_level
        self.critical_warnings_only = critical_warnings_only
        self.enable_voice_commands = enable_voice_commands
        
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
        
        # Voice commands dictionary
        self.voice_commands = {
            "repeat": self._handle_repeat_command,
            "stop": self._handle_stop_command,
            "help": self._handle_help_command,
            "status": self._handle_status_command
        }
        
        # Store last few messages for repeat functionality
        self.message_history = []
        self.max_history = 5
        
        # Standard responses dictionary
        self.responses = self._load_responses()
        
    def _load_responses(self):
        """Load standard responses from configuration."""
        # In a real implementation, this would load from a config file
        return {
            "greeting": "Surgical AI assistant initialized and ready.",
            "help_text": "Available commands: repeat, stop, help, and status.",
            "acknowledgment": "Understood.",
            "error": "I'm sorry, I couldn't process that request.",
            "warnings": {
                "artery_nearby": "Careful! Artery nearby.",
                "incorrect_clipping": "Incorrect clipping attempt.",
                "wrong_tool": "Wrong tool for this phase.",
                "excessive_force": "Excessive force detected.",
                "anatomical_anomaly": "Anatomical anomaly detected."
            }
        }
    
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
        
    def provide_phase_guidance(self, phase_name, is_transition=False):
        """
        Provide guidance for the current surgical phase.
        
        Args:
            phase_name: Name of the current surgical phase
            is_transition: Whether this is a transition to a new phase
        """
        if is_transition:
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
    
    def warn_wrong_tool(self, current_tool, required_tool, phase_name):
        """
        Warn about wrong tool usage.
        
        Args:
            current_tool: Currently detected tool
            required_tool: Required tool for current phase/action
            phase_name: Current surgical phase
        """
        message = f"Wrong tool. {current_tool} detected, but {required_tool} is needed for {phase_name}."
        self.speak(message, priority_level="warning")
    
    def warn_risk_situation(self, risk_type, risk_level, details=None):
        """
        Warn about risky situation.
        
        Args:
            risk_type: Type of risk ("anatomical", "technique", "tool")
            risk_level: Risk level (1-5)
            details: Specific details about the risk
        """
        if risk_level >= 4:  # High risk
            priority = "critical"
        elif risk_level >= 2:  # Medium risk
            priority = "warning"
        else:  # Low risk
            priority = "information"
            
        if risk_type == "anatomical":
            message = f"Caution! {details if details else 'Important anatomical structure'} nearby."
        elif risk_type == "technique":
            message = f"Technique risk: {details if details else 'Adjust approach'}."
        elif risk_type == "tool":
            message = f"Tool risk: {details if details else 'Check tool position'}."
        else:
            message = f"Risk detected: {details if details else 'Exercise caution'}."
            
        self.speak(message, priority_level=priority)
    
    def instruct_correction(self, mistake_type, correction):
        """
        Provide correction instruction.
        
        Args:
            mistake_type: Type of mistake detected
            correction: Correction instructions
        """
        message = f"Correction needed: {correction}"
        self.speak(message, priority_level="instruction")
        
    def acknowledge_correct_action(self):
        """Provide positive feedback for correct action."""
        if self.feedback_level == "full":
            messages = [
                "Correct technique.",
                "Good progress.",
                "Well executed.",
                "Proper approach."
            ]
            import random
            message = random.choice(messages)
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