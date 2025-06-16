"""
Integration of Chain of Thought reasoning with the RAG pipeline.

This module contains the logic for integrating reasoning capabilities with 
the RAG pipeline, including reasoning prompt generation, parsing, and output formatting.
"""
from typing import Any, Dict, List, Optional

from rag_engine.core.reasoning import (
    OutputMode, 
    ReasoningEngine, 
    ReasoningMode,
    ReasoningResult, 
    VerbosityLevel
)


class RAGReasoningIntegrator:
    """Integrates Chain of Thought reasoning with the RAG pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config
        self.enabled = config.get("enabled", False)
        
        if not self.enabled:
            return
            
        self.reasoning_engine = ReasoningEngine(config)
    
    def enhance_prompt(self, query: str, rag_results: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance the RAG prompt with reasoning guidance."""
        if not self.enabled:
            return rag_results
        
        # Get retrieved documents
        documents = rag_results.get("retrieved_documents", [])
        
        # Generate a reasoning prompt
        reasoning_prompt = self.reasoning_engine.create_reasoning_prompt(query, documents)
        
        # Modify the prompting section of RAG results
        enhanced_results = {**rag_results}
        if "prompting" not in enhanced_results:
            enhanced_results["prompting"] = {}
            
        # Determine which template to use
        if self.reasoning_engine.mode == ReasoningMode.STEP_BY_STEP:
            template_key = "cot_reasoning"
        elif self.reasoning_engine.mode == ReasoningMode.TREE_OF_THOUGHT:
            template_key = "tot_reasoning"
        else:
            template_key = "scratchpad_reasoning"
            
        # Check if a template exists for this reasoning mode
        templates = enhanced_results["prompting"].get("templates", {})
        if template_key in templates:
            # Template exists, make sure it's selected
            enhanced_results["prompting"]["selected_template"] = template_key
        else:
            # No template, modify the system prompt to include reasoning instructions
            system_prompt = enhanced_results["prompting"].get("system_prompt", "")
            
            # Add reasoning instructions
            if system_prompt:
                system_prompt += "\n\n"
                
            system_prompt += "Please use careful reasoning to answer the question based on the provided documents. "
            system_prompt += "Think step by step and explain your thought process clearly."
            
            enhanced_results["prompting"]["system_prompt"] = system_prompt
        
        return enhanced_results
    
    def process_response(self, llm_response: str, rag_results: Dict[str, Any]) -> Dict[str, Any]:
        """Process the LLM's response to extract and format reasoning."""
        if not self.enabled:
            return {"response": llm_response}
        
        # Parse the LLM's reasoning
        reasoning_result = self.reasoning_engine.parse_llm_reasoning(llm_response)
        
        # Determine output based on configuration
        output_mode = OutputMode(self.config.get("output_mode", OutputMode.FULL_PROCESS))
        verbosity = VerbosityLevel(self.config.get("verbosity", VerbosityLevel.MEDIUM))
        format = self.config.get("format", "markdown")
        
        # Format the reasoning output
        formatted_output = reasoning_result.format_output(
            verbosity=verbosity,
            output_mode=output_mode,
            format=format
        )
        
        # Create the response
        response = {
            "response": formatted_output if output_mode == OutputMode.FULL_PROCESS else reasoning_result.final_answer,
        }
        
        # Include the structured reasoning if configured
        if self.config.get("structured_format", False):
            response["structured_reasoning"] = reasoning_result.to_dict()
            
        # Include the raw LLM response if in detailed mode
        if verbosity == VerbosityLevel.DETAILED:
            response["raw_llm_response"] = llm_response
            
        # If expandable output is requested, include both versions
        if output_mode == OutputMode.EXPANDABLE:
            response["response"] = reasoning_result.final_answer
            response["reasoning_details"] = reasoning_result.format_output(
                verbosity=verbosity,
                output_mode=OutputMode.FULL_PROCESS,
                format=format
            )
        
        return response


# Factory function to build a reasoning integrator from config
def build_reasoning_system(config: Dict[str, Any]) -> RAGReasoningIntegrator:
    """Build a reasoning system from configuration."""
    return RAGReasoningIntegrator(config)
