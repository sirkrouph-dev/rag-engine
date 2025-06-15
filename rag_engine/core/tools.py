"""
Core interfaces and base classes for RAG Engine tools.

Tools are specialized components that extend LLM capabilities beyond text generation.
They allow the language model to perform actions, access external information,
or interact with external systems.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel


class ToolResult(BaseModel):
    """The result of a tool execution."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BaseTool(ABC):
    """Base interface for all tools."""
    
    def __init__(self, name: str, description: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.description = description
        self.config = config or {}
    
    @abstractmethod
    def execute(self, args: Dict[str, Any]) -> ToolResult:
        """Execute the tool with the provided arguments."""
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for this tool."""
        # Default implementation - override for more specific schemas
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }


class ToolRegistry:
    """Manages registered tools and their selection."""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool) -> None:
        """Register a tool with the registry."""
        self.tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def get_all(self) -> Dict[str, BaseTool]:
        """Get all registered tools."""
        return self.tools
    
    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get JSON schemas for all tools."""
        return [tool.get_schema() for tool in self.tools.values()]


class WebSearchTool(BaseTool):
    """Tool for performing web searches."""
    
    def execute(self, args: Dict[str, Any]) -> ToolResult:
        """Execute a web search."""
        query = args.get("query")
        if not query:
            return ToolResult(success=False, error="Query is required")
        
        try:
            # Implement actual search logic here
            # This would involve calling the configured search provider API
            provider = self.config.get("provider", "default")
            api_key = self.config.get("api_key")
            
            # Placeholder for actual implementation
            results = [
                {"title": f"Demo result for {query}", "url": "https://example.com", "snippet": "This is a demo result."}
            ]
            
            return ToolResult(success=True, data={"results": results})
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class CalculatorTool(BaseTool):
    """Tool for performing mathematical calculations."""
    
    def execute(self, args: Dict[str, Any]) -> ToolResult:
        """Execute a mathematical calculation."""
        expression = args.get("expression")
        if not expression:
            return ToolResult(success=False, error="Expression is required")
        
        try:
            # Use a safe evaluation approach - this is just a simple example
            # In a real implementation, you'd want more sophisticated parsing and evaluation
            result = eval(expression, {"__builtins__": {}}, {})
            return ToolResult(success=True, data={"result": result})
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class ToolSelector:
    """Selects appropriate tools based on query context."""
    
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
    
    def select_tools(self, query: str, context: Optional[Dict[str, Any]] = None) -> List[BaseTool]:
        """Select appropriate tools for the given query."""
        # This is where you would implement tool selection logic
        # For now, return all tools as a simple implementation
        return list(self.registry.tools.values())


# Factory function to create tools from config
def create_tool_from_config(tool_config: Dict[str, Any]) -> Optional[BaseTool]:
    """Create a tool instance from a configuration dictionary."""
    tool_type = tool_config.get("type")
    name = tool_config.get("name")
    description = tool_config.get("description", "")
    config = tool_config.get("config", {})
    
    if not tool_type or not name:
        return None
    
    # Map tool types to their classes
    tool_classes = {
        "web_search": WebSearchTool,
        "calculator": CalculatorTool,
        # Add more tool types here
    }
    
    tool_class = tool_classes.get(tool_type)
    if not tool_class:
        return None
    
    return tool_class(name=name, description=description, config=config)
