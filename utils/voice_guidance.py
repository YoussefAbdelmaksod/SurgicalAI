"""
Voice guidance module for SurgicalAI.

This module implements real-time voice guidance for surgical procedures,
providing audio feedback and warnings to the surgeon during operations.
"""

import threading
import time
import queue
import logging
import os
import tempfile
import platform

logger = logging.getLogger(__name__)

class VoiceGuidanceSystem:
    """
    Real-time voice guidance system for surgical procedures.
    """
    
    def __init__(self, tts_engine='auto', voice_gender='female', 
                 rate=150, volume=0.9, enable_logging=True):
        """
        Initialize the voice guidance system.
        
        Args:
            tts_engine (str): Text-to-speech engine to use ('pyttsx3', 'gtts', 'auto')
            voice_gender (str): Preferred voice gender ('male', 'female')
            rate (int): Speech rate (words per minute)
            volume (float): Volume level (0.0 to 1.0)
            enable_logging (bool): Whether to log voice messages
        """
        self.voice_gender = voice_gender
        self.rate = rate
        self.volume = volume
        self.enable_logging = enable_logging
        
        # Message queue
        self.message_queue = queue.Queue()
        
        # Initialize TTS engine
        self.tts_available = False
        self.tts_engine_name = None
        self._setup_tts_engine(tts_engine)
        
        # Current message being spoken
        self.current_message = None
        
        # Thread for processing speech queue
        self.speech_thread = None
        self.running = False
        
        # Message history
        self.message_history = []
        
        # Start the speech thread if TTS is available
        if self.tts_available:
            self.running = True
            self.speech_thread = threading.Thread(target=self._process_queue, daemon=True)
            self.speech_thread.start()
            logger.info(f"Voice guidance system initialized with {self.tts_engine_name}")
        else:
            logger.warning("No TTS engine available. Voice guidance will be disabled.")
    
    def _setup_tts_engine(self, engine_name):
        """Set up the text-to-speech engine."""
        if engine_name == 'auto' or engine_name == 'pyttsx3':
            try:
                import pyttsx3
                self.tts_engine = pyttsx3.init()
                
                # Configure voice properties
                self.tts_engine.setProperty('rate', self.rate)
                self.tts_engine.setProperty('volume', self.volume)
                
                # Try to set preferred voice gender
                voices = self.tts_engine.getProperty('voices')
                for voice in voices:
                    if self.voice_gender in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
                
                self.tts_available = True
                self.tts_engine_name = 'pyttsx3'
                return
            except ImportError:
                logger.warning("pyttsx3 not installed.")
            except Exception as e:
                logger.error(f"Failed to initialize pyttsx3: {e}")
        
        if engine_name == 'auto' or engine_name == 'gtts':
            try:
                from gtts import gTTS
                import pygame
                
                # Initialize pygame mixer
                pygame.mixer.init()
                
                # Test if it works
                test_tts = gTTS("System ready", lang='en')
                
                self.tts_available = True
                self.tts_engine_name = 'gtts'
                return
            except ImportError:
                logger.warning("gTTS or pygame not installed.")
            except Exception as e:
                logger.error(f"Failed to initialize gTTS: {e}")
        
        # Try system-specific TTS options as last resort
        if engine_name == 'auto':
            try:
                system = platform.system()
                if system == 'Darwin':  # macOS
                    # Use macOS say command
                    import subprocess
                    subprocess.run(['say', 'Test'], check=True)
                    self.tts_available = True
                    self.tts_engine_name = 'macos_say'
                    return
                elif system == 'Windows':
                    # Use Windows SAPI
                    import win32com.client
                    self.tts_engine = win32com.client.Dispatch("SAPI.SpVoice")
                    self.tts_available = True
                    self.tts_engine_name = 'windows_sapi'
                    return
            except Exception as e:
                logger.error(f"Failed to initialize system TTS: {e}")
    
    def _process_queue(self):
        """Process messages in the queue."""
        while self.running:
            try:
                # Get message from queue (with timeout to allow checking self.running)
                try:
                    message_data = self.message_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Extract message details
                message = message_data['text']
                priority = message_data['priority']
                
                # Set as current message
                self.current_message = message
                
                # Speak the message
                self._speak(message)
                
                # Add to history
                self.message_history.append({
                    'text': message,
                    'priority': priority,
                    'timestamp': time.time()
                })
                
                # Log if enabled
                if self.enable_logging:
                    logger.info(f"Voice guidance ({priority}): {message}")
                
                # Mark as done
                self.message_queue.task_done()
                self.current_message = None
                
                # Small delay between messages
                time.sleep(0.3)
            except Exception as e:
                logger.error(f"Error in speech thread: {e}")
                time.sleep(1)  # Avoid tight loop on error
    
    def _speak(self, text):
        """Speak the given text using the selected TTS engine."""
        if not self.tts_available or not text:
            return
        
        try:
            if self.tts_engine_name == 'pyttsx3':
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            elif self.tts_engine_name == 'gtts':
                from gtts import gTTS
                import pygame
                import tempfile
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                    temp_filename = fp.name
                
                # Generate speech
                speech = gTTS(text=text, lang='en', slow=False)
                speech.save(temp_filename)
                
                # Play the speech
                pygame.mixer.music.load(temp_filename)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                
                # Clean up temporary file
                try:
                    os.unlink(temp_filename)
                except:
                    pass
            elif self.tts_engine_name == 'macos_say':
                import subprocess
                subprocess.run(['say', text])
            elif self.tts_engine_name == 'windows_sapi':
                self.tts_engine.Speak(text)
        except Exception as e:
            logger.error(f"Error in speech synthesis: {e}")
    
    def speak(self, text, priority='normal'):
        """
        Add a message to the speech queue.
        
        Args:
            text (str): Text to speak
            priority (str): Priority level ('high', 'normal', 'low')
        """
        if not self.tts_available or not text:
            return False
        
        # Clear queue for high priority messages
        if priority == 'high' and not self.message_queue.empty():
            # Clear the queue while preserving thread safety
            while not self.message_queue.empty():
                try:
                    self.message_queue.get_nowait()
                    self.message_queue.task_done()
                except queue.Empty:
                    break
        
        # Add message to queue
        self.message_queue.put({
            'text': text,
            'priority': priority,
            'timestamp': time.time()
        })
        
        return True
    
    def stop(self):
        """Stop the voice guidance system."""
        self.running = False
        if self.speech_thread:
            self.speech_thread.join(timeout=1.0)
        
        # Clear the queue
        while not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
                self.message_queue.task_done()
            except queue.Empty:
                break
    
    def is_speaking(self):
        """Check if the system is currently speaking."""
        return self.current_message is not None
    
    def get_message_history(self, limit=None):
        """
        Get the history of spoken messages.
        
        Args:
            limit (int, optional): Maximum number of messages to return
            
        Returns:
            list: List of message dictionaries
        """
        if limit:
            return self.message_history[-limit:]
        return self.message_history.copy() 