"""
Chain of Thought (CoT) reasoning module for RAG Engine.

This module provides reasoning capabilities that make AI responses more transparent,
explainable and trustworthy by exposing the model's intermediate reasoning steps.
"""
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class ReasoningMode(str, Enum):
    """Available reasoning modes."""
    STEP_BY_STEP = "step_by_step"
    TREE_OF_THOUGHT = "tree_of_thought"
    SCRATCHPAD = "scratchpad"


class VerbosityLevel(str, Enum):
    """Controls how detailed the reasoning should be."""
    MINIMAL = "minimal"
    MEDIUM = "medium"
    DETAILED = "detailed"


class OutputMode(str, Enum):
    """How reasoning should be presented to the user."""
    FULL_PROCESS = "full_process"  # Show full reasoning process
    ANSWER_ONLY = "answer_only"    # Show only final answer
    EXPANDABLE = "expandable"      # Show answer with option to expand reasoning


class ReasoningStep(BaseModel):
    """A single step in the reasoning process."""
    name: str
    content: str
    # Optional fields for tracking step execution
    confidence: Optional[float] = None
    documents_used: Optional[List[str]] = None
    elapsed_time_ms: Optional[int] = None


class StepByStepReasoning(BaseModel):
    """Container for step-by-step reasoning process."""
    steps: List[ReasoningStep] = Field(default_factory=list)
    final_answer: str = ""
    
    def add_step(self, name: str, content: str, **kwargs) -> None:
        """Add a reasoning step."""
        self.steps.append(
            ReasoningStep(
                name=name,
                content=content,
                **kwargs
            )
        )
    
    def format_output(self, verbosity: VerbosityLevel = VerbosityLevel.MEDIUM, 
                     format: str = "markdown") -> str:
        """Format the reasoning steps for output."""
        output = ""
        
        if format == "markdown":
            for i, step in enumerate(self.steps):
                if verbosity == VerbosityLevel.MINIMAL and i < len(self.steps) - 1:
                    # In minimal mode, only include the last step before the answer
                    continue
                    
                output += f"### Step {i+1}: {step.name}\n\n"
                output += f"{step.content}\n\n"
                
                # Include document references for medium and detailed verbosity
                if verbosity != VerbosityLevel.MINIMAL and step.documents_used:
                    output += "**References:**\n"
                    for doc in step.documents_used:
                        output += f"- {doc}\n"
                    output += "\n"
            
            output += f"### Answer\n\n{self.final_answer}\n"
            
        elif format == "plain":
            for i, step in enumerate(self.steps):
                if verbosity == VerbosityLevel.MINIMAL and i < len(self.steps) - 1:
                    continue
                
                output += f"Step {i+1}: {step.name}\n"
                output += f"{step.content}\n\n"
                
            output += f"Answer: {self.final_answer}\n"
            
        return output
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for structured output."""
        return {
            "reasoning_steps": [
                {
                    "name": step.name,
                    "content": step.content,
                    **({"confidence": step.confidence} if step.confidence is not None else {}),
                    **({"documents_used": step.documents_used} if step.documents_used else {})
                }
                for step in self.steps
            ],
            "final_answer": self.final_answer
        }


class ThoughtNode(BaseModel):
    """A node in a tree of thought."""
    content: str
    children: List["ThoughtNode"] = Field(default_factory=list)
    score: Optional[float] = None
    is_selected: bool = False


class TreeOfThoughtReasoning(BaseModel):
    """Container for tree of thought reasoning process."""
    root: Optional[ThoughtNode] = None
    final_answer: str = ""
    
    def add_root(self, content: str) -> None:
        """Initialize the root thought."""
        self.root = ThoughtNode(content=content)
    
    def add_thought(self, parent_content: str, content: str, score: Optional[float] = None) -> None:
        """Add a thought to the tree."""
        if not self.root:
            self.add_root(parent_content)
        
        # Find the parent node
        def find_node(node: ThoughtNode, target: str) -> Optional[ThoughtNode]:
            if node.content == target:
                return node
            
            for child in node.children:
                result = find_node(child, target)
                if result:
                    return result
            
            return None
        
        parent_node = find_node(self.root, parent_content)
        if parent_node:
            parent_node.children.append(ThoughtNode(content=content, score=score))
    
    def select_path(self, contents: List[str]) -> None:
        """Mark a specific path through the tree as selected."""
        # Reset all selection flags
        def reset_selection(node: ThoughtNode):
            node.is_selected = False
            for child in node.children:
                reset_selection(child)
        
        if self.root:
            reset_selection(self.root)
        
        # Mark the new path
        current_node = self.root
        if not current_node or not contents:
            return
        
        current_node.is_selected = True
        for content in contents:
            found = False
            for child in current_node.children:
                if child.content == content:
                    child.is_selected = True
                    current_node = child
                    found = True
                    break
            
            if not found:
                break
    
    def format_output(self, verbosity: VerbosityLevel = VerbosityLevel.MEDIUM, 
                     format: str = "markdown") -> str:
        """Format the tree of thought for output."""
        if not self.root:
            return ""
        
        output = ""
        
        if format == "markdown":
            output += "# Reasoning Process\n\n"
            
            # Function to recursively format the tree
            def format_node(node: ThoughtNode, depth: int) -> str:
                # Skip non-selected paths in minimal verbosity
                if verbosity == VerbosityLevel.MINIMAL and not node.is_selected:
                    return ""
                
                result = "#" * (depth + 2) + " " + ("Selected: " if node.is_selected else "") + "\n\n"
                result += node.content + "\n\n"
                
                if node.score is not None and verbosity == VerbosityLevel.DETAILED:
                    result += f"*Confidence: {node.score:.2f}*\n\n"
                
                for child in node.children:
                    child_result = format_node(child, depth + 1)
                    if child_result:  # Only append non-empty results
                        result += child_result
                
                return result
            
            output += format_node(self.root, 0)
            output += f"## Final Answer\n\n{self.final_answer}\n"
            
        elif format == "plain":
            # Similar implementation for plain text format
            pass
            
        return output


class ScratchpadReasoning(BaseModel):
    """Container for free-form scratchpad reasoning."""
    content: str = ""
    final_answer: str = ""
    
    def add_content(self, new_content: str) -> None:
        """Add content to the scratchpad."""
        if self.content:
            self.content += "\n\n" + new_content
        else:
            self.content = new_content
    
    def format_output(self, verbosity: VerbosityLevel = VerbosityLevel.MEDIUM, 
                     format: str = "markdown") -> str:
        """Format the scratchpad for output."""
        if format == "markdown":
            if verbosity == VerbosityLevel.MINIMAL:
                return f"### Answer\n\n{self.final_answer}\n"
            else:
                return f"### Scratchpad\n\n{self.content}\n\n### Answer\n\n{self.final_answer}\n"
        elif format == "plain":
            if verbosity == VerbosityLevel.MINIMAL:
                return f"Answer: {self.final_answer}\n"
            else:
                return f"Scratchpad:\n{self.content}\n\nAnswer: {self.final_answer}\n"
        
        return ""


class ReasoningResult(BaseModel):
    """The result of a reasoning process."""
    mode: ReasoningMode
    step_by_step: Optional[StepByStepReasoning] = None
    tree_of_thought: Optional[TreeOfThoughtReasoning] = None
    scratchpad: Optional[ScratchpadReasoning] = None
    final_answer: str = ""
    
    def format_output(self, verbosity: VerbosityLevel = VerbosityLevel.MEDIUM,
                     output_mode: OutputMode = OutputMode.FULL_PROCESS,
                     format: str = "markdown") -> str:
        """Format the reasoning result for output."""
        if output_mode == OutputMode.ANSWER_ONLY:
            if format == "markdown":
                return f"### Answer\n\n{self.final_answer}\n"
            else:
                return f"Answer: {self.final_answer}\n"
        
        if self.mode == ReasoningMode.STEP_BY_STEP and self.step_by_step:
            return self.step_by_step.format_output(verbosity, format)
        elif self.mode == ReasoningMode.TREE_OF_THOUGHT and self.tree_of_thought:
            return self.tree_of_thought.format_output(verbosity, format)
        elif self.mode == ReasoningMode.SCRATCHPAD and self.scratchpad:
            return self.scratchpad.format_output(verbosity, format)
        else:
            return f"Reasoning not available. Answer: {self.final_answer}"
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for structured output."""
        if self.mode == ReasoningMode.STEP_BY_STEP and self.step_by_step:
            result = self.step_by_step.to_dict()
        elif self.mode == ReasoningMode.TREE_OF_THOUGHT and self.tree_of_thought:
            # Implement TreeOfThought serialization
            result = {"final_answer": self.final_answer}
        elif self.mode == ReasoningMode.SCRATCHPAD and self.scratchpad:
            result = {
                "scratchpad": self.scratchpad.content,
                "final_answer": self.final_answer
            }
        else:
            result = {"final_answer": self.final_answer}
        
        result["reasoning_mode"] = self.mode
        return result


class ReasoningEngine:
    """Engine for managing reasoning processes."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the reasoning engine with configuration."""
        self.config = config
        self.mode = ReasoningMode(config.get("mode", ReasoningMode.STEP_BY_STEP))
        self.verbosity = VerbosityLevel(config.get("verbosity", VerbosityLevel.MEDIUM))
        self.output_mode = OutputMode(config.get("output_mode", OutputMode.FULL_PROCESS))
        self.include_in_output = config.get("include_in_output", True)
        self.structured_format = config.get("structured_format", False)
    
    def create_reasoning_prompt(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Create a prompt that guides the LLM through the reasoning process."""
        if self.mode == ReasoningMode.STEP_BY_STEP:
            steps = self.config.get("steps", [
                "analyze_question",
                "extract_key_information",
                "evaluate_evidence",
                "formulate_answer",
                "verify_answer"
            ])
            
            prompt = f"Question: {query}\n\n"
            prompt += "Please think through this step by step:\n\n"
            
            for i, step in enumerate(steps):
                prompt += f"{i+1}. {step.replace('_', ' ').title()}: "
                if i < len(steps) - 1:
                    prompt += "[Your reasoning for this step]\n\n"
            
            prompt += "\nBased on the above reasoning, provide your final answer."
            return prompt
            
        elif self.mode == ReasoningMode.TREE_OF_THOUGHT:
            max_branches = self.config.get("max_branches", 3)
            max_depth = self.config.get("max_depth", 2)
            
            prompt = f"Question: {query}\n\n"
            prompt += f"Consider {max_branches} different approaches to answer this question. "
            prompt += f"For each approach, reason for {max_depth} steps, then provide a conclusion.\n\n"
            prompt += "Finally, select the best approach and provide your final answer."
            return prompt
            
        elif self.mode == ReasoningMode.SCRATCHPAD:
            prompt = f"Question: {query}\n\n"
            prompt += "Use this scratchpad to work through your thinking step by step.\n"
            prompt += "Write out your thoughts, analyze the evidence, and then provide your final answer."
            return prompt
            
        else:
            # Default fallback
            prompt = f"Question: {query}\n\n"
            prompt += "Reason step by step before providing your final answer."
            return prompt
    
    def parse_llm_reasoning(self, llm_output: str) -> ReasoningResult:
        """Parse the LLM's output to extract reasoning steps and final answer."""
        # This parsing would be much more sophisticated in a real implementation
        # and would depend on how the LLM structures its output
        
        result = ReasoningResult(mode=self.mode)
        
        if self.mode == ReasoningMode.STEP_BY_STEP:
            # Basic parsing logic for step-by-step reasoning
            step_by_step = StepByStepReasoning()
            
            # Very simplistic parsing - in reality would be more robust
            parts = llm_output.split("\n\n")
            
            for part in parts[:-1]:  # All but the last part are considered steps
                if ":" in part:
                    name, content = part.split(":", 1)
                    step_by_step.add_step(name.strip(), content.strip())
            
            # Assume the last part is the final answer
            step_by_step.final_answer = parts[-1].strip()
            result.step_by_step = step_by_step
            result.final_answer = step_by_step.final_answer
            
        elif self.mode == ReasoningMode.SCRATCHPAD:
            # Simple parsing for scratchpad
            parts = llm_output.split("\n\nFinal Answer:")
            
            scratchpad = ScratchpadReasoning()
            if len(parts) > 1:
                scratchpad.content = parts[0].strip()
                scratchpad.final_answer = parts[1].strip()
            else:
                # No clear delineation, treat it all as the answer
                scratchpad.final_answer = llm_output.strip()
            
            result.scratchpad = scratchpad
            result.final_answer = scratchpad.final_answer
            
        # Note: Tree of Thought parsing would be more complex and is not fully implemented here
            
        return result
