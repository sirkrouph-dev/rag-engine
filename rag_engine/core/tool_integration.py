"""
Tool integration module for RAG Engine.

This module contains the logic for integrating tools with the RAG pipeline,
including tool selection, execution, and result incorporation.
"""
from typing import Any, Dict, List, Optional

from rag_engine.core.tools import BaseTool, ToolRegistry, ToolResult, ToolSelector


class ToolRunner:
    """Responsible for executing tools and managing their results."""
    
    def __init__(self, registry: ToolRegistry, selector: ToolSelector):
        self.registry = registry
        self.selector = selector
    
    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Optional[ToolResult]:
        """Execute a specific tool by name."""
        tool = self.registry.get(tool_name)
        if not tool:
            return None
        
        return tool.execute(args)
    
    def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a query using appropriate tools."""
        # Select tools that might be relevant for this query
        selected_tools = self.selector.select_tools(query, context)
        
        # In an actual implementation, you might:
        # 1. Analyze the query to determine which tools to use
        # 2. Extract the necessary parameters for each tool
        # 3. Execute the tools in the appropriate order
        # 4. Incorporate the results into the response
        
        # This is a simplified version
        results = {
            "query": query,
            "tool_results": []
        }
        
        # For demo purposes, we'll execute the calculator tool if it's present
        # and the query contains math-related keywords
        calculator = next((tool for tool in selected_tools 
                          if tool.name.lower() in ["calculator", "calc", "math"]), None)
        
        if calculator and any(kw in query.lower() for kw in ["calculate", "compute", "math", "+"]):
            # Extract a simple expression (this would be more sophisticated in reality)
            import re
            expression_match = re.search(r'(\d+[\+\-\*\/\^]\d+)', query)
            if expression_match:
                expression = expression_match.group(1)
                tool_result = calculator.execute({"expression": expression})
                results["tool_results"].append({
                    "tool": calculator.name,
                    "success": tool_result.success,
                    "result": tool_result.data["result"] if tool_result.success else None,
                    "error": tool_result.error if not tool_result.success else None
                })
        
        # Similarly for web search
        search_tool = next((tool for tool in selected_tools 
                           if tool.name.lower() in ["search", "web_search"]), None)
        
        if search_tool and any(kw in query.lower() for kw in ["search", "find", "look up"]):
            tool_result = search_tool.execute({"query": query})
            results["tool_results"].append({
                "tool": search_tool.name,
                "success": tool_result.success,
                "result": tool_result.data["results"] if tool_result.success else None,
                "error": tool_result.error if not tool_result.success else None
            })
        
        return results


class RAGToolIntegrator:
    """Integrates tools with the RAG pipeline."""
    
    def __init__(self, tool_runner: ToolRunner):
        self.tool_runner = tool_runner
    
    def enhance_response(self, query: str, rag_results: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance RAG results with tool-generated information."""
        # Process the query with tools
        tool_results = self.tool_runner.process_query(query, context=rag_results)
        
        # Merge RAG results with tool results
        enhanced_results = {**rag_results}
        enhanced_results["tool_results"] = tool_results.get("tool_results", [])
        
        # For successful tool executions, we might want to modify the prompt to include the tool results
        if enhanced_results.get("prompting") and tool_results.get("tool_results"):
            successful_results = [r for r in tool_results["tool_results"] if r["success"]]
            if successful_results:
                # Add tool results to the context for the LLM
                tools_context = "Tool results:\n"
                for result in successful_results:
                    tools_context += f"- {result['tool']}: {result['result']}\n"
                
                # Append to system prompt
                system_prompt = enhanced_results["prompting"].get("system_prompt", "")
                enhanced_results["prompting"]["system_prompt"] = system_prompt + "\n\n" + tools_context
        
        return enhanced_results


# Factory function to build a complete tools system from config
def build_tools_system(tools_config: List[Dict[str, Any]]) -> RAGToolIntegrator:
    """Build a tools system from a configuration."""
    from rag_engine.core.tools import create_tool_from_config
    
    # Create the tool registry
    registry = ToolRegistry()
    
    # Create and register tools from the configuration
    for tool_config in tools_config:
        tool = create_tool_from_config(tool_config)
        if tool:
            registry.register(tool)
    
    # Create the selector and runner
    selector = ToolSelector(registry)
    runner = ToolRunner(registry, selector)
    
    # Return the integrator
    return RAGToolIntegrator(runner)
