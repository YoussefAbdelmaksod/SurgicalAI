"""
GPT-based personalized guidance system for SurgicalAI.

This module implements a lightweight GPT-based guidance system for providing
contextual and personalized surgical guidance based on real-time observations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import json
import numpy as np
from typing import List, Dict, Optional, Union, Any, Tuple

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    GPT2Tokenizer,
    GPT2LMHeadModel
)

logger = logging.getLogger(__name__)

class SurgicalGPTGuidance(nn.Module):
    """
    GPT-based guidance system for surgical procedures.
    
    Uses a fine-tuned language model to generate contextual guidance
    based on the current surgical context, detected tools, phases, and user profile.
    """
    
    def __init__(self, 
                 model_name="gpt2",
                 weights_path=None,
                 procedure_knowledge_path=None,
                 guidance_templates_path=None,
                 device=None):
        """
        Initialize the GPT guidance system.
        
        Args:
            model_name: Base model name (e.g., "gpt2", "distilgpt2")
            weights_path: Path to fine-tuned model weights (if available)
            procedure_knowledge_path: Path to procedure knowledge JSON
            guidance_templates_path: Path to guidance templates
            device: Device to run the model on (None for auto-detection)
        """
        super().__init__()
        
        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            # Fallback to GPT2
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Initialize model
        try:
            if weights_path and os.path.exists(weights_path):
                logger.info(f"Loading fine-tuned model from {weights_path}")
                self.model = AutoModelForCausalLM.from_pretrained(weights_path)
            else:
                logger.info(f"Loading base model: {model_name}")
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fallback to GPT2
            logger.info("Falling back to GPT2 base model")
            self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Load procedure knowledge
        self.procedure_knowledge = {}
        if procedure_knowledge_path and os.path.exists(procedure_knowledge_path):
            try:
                with open(procedure_knowledge_path, 'r') as f:
                    self.procedure_knowledge = json.load(f)
                logger.info(f"Loaded procedure knowledge from {procedure_knowledge_path}")
            except Exception as e:
                logger.error(f"Failed to load procedure knowledge: {e}")
        
        # Load guidance templates
        self.guidance_templates = {
            "phase_transition": "You are now entering the {phase_name} phase. {phase_description}",
            "tool_recommendation": "For this step, consider using {tool_name}. {tool_purpose}",
            "mistake_warning": "{warning_prefix}Potential mistake detected: {mistake_description}",
            "safety_guidance": "To ensure safety: {safety_tip}",
            "critical_view": "Critical view of safety: {cvs_status}"
        }
        
        if guidance_templates_path and os.path.exists(guidance_templates_path):
            try:
                with open(guidance_templates_path, 'r') as f:
                    templates = json.load(f)
                    self.guidance_templates.update(templates)
                logger.info(f"Loaded guidance templates from {guidance_templates_path}")
            except Exception as e:
                logger.error(f"Failed to load guidance templates: {e}")
                
        # Experience level-specific prefixes
        self.experience_prefixes = {
            "novice": "Step-by-step instructions: ",
            "junior": "Guidance for this phase: ",
            "intermediate": "Surgical guidance: ",
            "senior": "Note: ",
            "expert": ""
        }
        
        # Prompt template for guidance generation
        self.base_prompt_template = """
Surgical Context:
- Phase: {phase}
- Detected tools: {tools}
- Anatomical structures visible: {anatomical_structures}
- Critical View of Safety achieved: {cvs_achieved}

Previous actions:
{previous_actions}

Surgeon experience level: {experience_level}

Common mistakes in this phase:
{common_mistakes}

Provide concise, specific guidance for the surgeon on how to proceed safely:
"""
        
        # Set model to evaluation mode
        self.model.eval()
        
    def format_prompt(self, 
                     phase: str, 
                     tools: List[str], 
                     anatomical_structures: List[str],
                     cvs_achieved: bool,
                     previous_actions: List[str],
                     experience_level: str = "intermediate",
                     common_mistakes: List[str] = None) -> str:
        """
        Format the prompt for guidance generation.
        
        Args:
            phase: Current surgical phase
            tools: List of detected tools
            anatomical_structures: List of visible anatomical structures
            cvs_achieved: Whether Critical View of Safety is achieved
            previous_actions: List of previous actions
            experience_level: Surgeon experience level
            common_mistakes: List of common mistakes in this phase
            
        Returns:
            Formatted prompt string
        """
        # Format tools
        tools_str = ", ".join(tools) if tools else "None"
        
        # Format anatomical structures
        structures_str = ", ".join(anatomical_structures) if anatomical_structures else "None"
        
        # Format previous actions (limit to last 3)
        actions_str = "\n".join([f"- {action}" for action in previous_actions[-3:]]) if previous_actions else "None"
        
        # Format common mistakes
        if not common_mistakes:
            common_mistakes = []
            phase_key = phase.lower().replace(" ", "_")
            if phase_key in self.procedure_knowledge.get("common_mistakes", {}):
                common_mistakes = self.procedure_knowledge["common_mistakes"][phase_key]
        
        mistakes_str = "\n".join([f"- {mistake}" for mistake in common_mistakes]) if common_mistakes else "None"
        
        # Format prompt
        prompt = self.base_prompt_template.format(
            phase=phase,
            tools=tools_str,
            anatomical_structures=structures_str,
            cvs_achieved="Yes" if cvs_achieved else "No",
            previous_actions=actions_str,
            experience_level=experience_level,
            common_mistakes=mistakes_str
        )
        
        # Add experience-specific prefix
        if experience_level in self.experience_prefixes:
            prompt += "\n" + self.experience_prefixes[experience_level]
        
        return prompt
    
    def generate_guidance(self, 
                          prompt: str, 
                          max_length: int = 100,
                          temperature: float = 0.7,
                          top_p: float = 0.9,
                          top_k: int = 50,
                          num_return_sequences: int = 1) -> List[str]:
        """
        Generate guidance based on the prompt.
        
        Args:
            prompt: Formatted prompt
            max_length: Maximum length of the generated text
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            num_return_sequences: Number of guidance options to generate
            
        Returns:
            List of generated guidance texts
        """
        # Encode the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Set generation parameters
        gen_kwargs = {
            "max_length": inputs["input_ids"].shape[1] + max_length,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "num_return_sequences": num_return_sequences,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Generate guidance
        with torch.no_grad():
            try:
                outputs = self.model.generate(**inputs, **gen_kwargs)
                
                # Extract the generated text
                guidance_texts = []
                for output in outputs:
                    # Get only the newly generated tokens
                    new_tokens = output[inputs["input_ids"].shape[1]:]
                    # Decode to text
                    text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    # Clean up the text
                    text = text.strip()
                    guidance_texts.append(text)
                
                return guidance_texts
            except Exception as e:
                logger.error(f"Error generating guidance: {e}")
                return ["Error generating guidance. Please proceed with caution."]
    
    def get_personalized_guidance(self, 
                                  context: Dict[str, Any],
                                  user_profile: Any = None) -> Dict[str, Any]:
        """
        Get personalized guidance based on the current context and user profile.
        
        Args:
            context: Dictionary with surgical context
            user_profile: User profile object (optional)
            
        Returns:
            Dictionary with guidance information
        """
        # Extract context information
        phase = context.get("phase", "unknown")
        tools = context.get("detected_tools", [])
        anatomical_structures = context.get("anatomical_structures", [])
        cvs_achieved = context.get("cvs_achieved", False)
        previous_actions = context.get("previous_actions", [])
        
        # Set default experience level
        experience_level = "intermediate"
        common_mistakes = []
        
        # Get personalization data from user profile if available
        if user_profile:
            experience_level = user_profile.experience_level
            
            # Get common mistakes from user profile
            if phase in user_profile.personalization.get("common_mistakes", {}):
                common_mistakes = [
                    m["description"] for m in user_profile.personalization["common_mistakes"][phase]
                ]
        
        # Format prompt
        prompt = self.format_prompt(
            phase=phase,
            tools=tools,
            anatomical_structures=anatomical_structures,
            cvs_achieved=cvs_achieved,
            previous_actions=previous_actions,
            experience_level=experience_level,
            common_mistakes=common_mistakes
        )
        
        # Generate guidance
        guidance_texts = self.generate_guidance(
            prompt=prompt,
            # Adjust parameters based on experience level
            temperature=0.5 if experience_level in ["novice", "junior"] else 0.7,
            num_return_sequences=2
        )
        
        # Construct response
        response = {
            "primary_guidance": guidance_texts[0] if guidance_texts else "",
            "alternative_guidance": guidance_texts[1:] if len(guidance_texts) > 1 else [],
            "phase": phase,
            "experience_level": experience_level
        }
        
        return response
    
    def get_phase_guidance(self, 
                          phase_name: str, 
                          is_transition: bool = True,
                          user_profile: Any = None) -> str:
        """
        Get guidance for a specific surgical phase.
        
        Args:
            phase_name: Name of the surgical phase
            is_transition: Whether this is a transition to the phase
            user_profile: User profile for personalization
            
        Returns:
            Phase guidance text
        """
        # Get phase description from procedure knowledge
        phase_key = phase_name.lower().replace(" ", "_")
        phase_description = ""
        
        if phase_key in self.procedure_knowledge.get("phases", {}):
            phase_description = self.procedure_knowledge["phases"][phase_key].get("description", "")
            
            # If this is a transition, add key points
            if is_transition and "key_points" in self.procedure_knowledge["phases"][phase_key]:
                key_points = self.procedure_knowledge["phases"][phase_key]["key_points"]
                if key_points:
                    phase_description += " Key points: " + "; ".join(key_points) + "."
        
        # Create context for guidance
        context = {
            "phase": phase_name,
            "detected_tools": [],
            "anatomical_structures": [],
            "cvs_achieved": False,
            "previous_actions": [],
            "is_transition": is_transition
        }
        
        # Get personalized guidance
        guidance = self.get_personalized_guidance(context, user_profile)
        
        return guidance["primary_guidance"]
    
    def get_mistake_guidance(self,
                            mistake_info: Dict[str, Any],
                            context: Dict[str, Any],
                            user_profile: Any = None) -> str:
        """
        Get guidance for handling a detected mistake.
        
        Args:
            mistake_info: Information about the detected mistake
            context: Current surgical context
            user_profile: User profile for personalization
            
        Returns:
            Mistake guidance text
        """
        # Extract mistake information
        mistake_type = mistake_info.get("type", "unknown")
        description = mistake_info.get("description", "")
        risk_level = mistake_info.get("risk_level", 0.0)
        
        # Set warning prefix based on risk level
        if risk_level >= 0.8:
            warning_prefix = "CRITICAL: "
        elif risk_level >= 0.5:
            warning_prefix = "WARNING: "
        else:
            warning_prefix = "Note: "
            
        # Add mistake to context's previous actions
        previous_actions = context.get("previous_actions", []).copy()
        previous_actions.append(f"Mistake detected: {description} (risk level: {risk_level:.1f})")
        
        # Update context
        updated_context = context.copy()
        updated_context["previous_actions"] = previous_actions
        
        # Get personalized guidance
        guidance = self.get_personalized_guidance(updated_context, user_profile)
        
        # Format final guidance
        if risk_level >= 0.8:
            # For critical mistakes, prefix the guidance
            return f"{warning_prefix}{description}. {guidance['primary_guidance']}"
        else:
            return guidance["primary_guidance"]
    
    def get_tool_guidance(self, 
                         current_tools: List[str],
                         recommended_tools: List[str],
                         phase_name: str,
                         user_profile: Any = None) -> str:
        """
        Get guidance for tool usage.
        
        Args:
            current_tools: Currently detected tools
            recommended_tools: Recommended tools for this phase
            phase_name: Current surgical phase
            user_profile: User profile for personalization
            
        Returns:
            Tool guidance text
        """
        # Identify missing recommended tools
        missing_tools = [tool for tool in recommended_tools if tool not in current_tools]
        
        if not missing_tools:
            return ""
            
        # Get tool information from procedure knowledge
        tool_info = {}
        if "tools" in self.procedure_knowledge:
            tool_info = self.procedure_knowledge["tools"]
            
        # Create context for guidance
        context = {
            "phase": phase_name,
            "detected_tools": current_tools,
            "recommended_tools": recommended_tools,
            "missing_tools": missing_tools,
            "anatomical_structures": [],
            "cvs_achieved": False,
            "previous_actions": [f"Current tools: {', '.join(current_tools)}"]
        }
        
        # Add information about missing tools
        for tool in missing_tools:
            if tool in tool_info:
                purpose = tool_info[tool].get("purpose", "")
                context["previous_actions"].append(f"Missing recommended tool: {tool} - {purpose}")
            else:
                context["previous_actions"].append(f"Missing recommended tool: {tool}")
        
        # Get personalized guidance
        guidance = self.get_personalized_guidance(context, user_profile)
        
        return guidance["primary_guidance"]
    
    def save(self, path: str) -> bool:
        """
        Save the model and tokenizer.
        
        Args:
            path: Directory path to save to
            
        Returns:
            True if save was successful
        """
        try:
            os.makedirs(path, exist_ok=True)
            
            # Save model
            self.model.save_pretrained(path)
            
            # Save tokenizer
            self.tokenizer.save_pretrained(path)
            
            # Save templates
            with open(os.path.join(path, 'guidance_templates.json'), 'w') as f:
                json.dump(self.guidance_templates, f, indent=2)
                
            logger.info(f"Model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load(self, path: str) -> bool:
        """
        Load the model and tokenizer.
        
        Args:
            path: Directory path to load from
            
        Returns:
            True if load was successful
        """
        try:
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(path)
            self.model = self.model.to(self.device)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Load templates
            templates_path = os.path.join(path, 'guidance_templates.json')
            if os.path.exists(templates_path):
                with open(templates_path, 'r') as f:
                    self.guidance_templates = json.load(f)
            
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False 