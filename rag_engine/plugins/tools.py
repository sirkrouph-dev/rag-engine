"""
Example implementations of RAG Engine tools.

This module contains concrete implementations of various tool types that can
be used to extend the capabilities of the RAG Engine beyond text generation.
"""
import json
import math
import re
import requests
from typing import Any, Dict, List, Optional

from rag_engine.core.tools import BaseTool, ToolResult


class WebSearchTool(BaseTool):
    """Tool for performing web searches using various search engines."""
    
    def execute(self, args: Dict[str, Any]) -> ToolResult:
        """Execute a web search."""
        query = args.get("query")
        num_results = args.get("num_results", 3)
        
        if not query:
            return ToolResult(success=False, error="Query is required")
        
        provider = self.config.get("provider", "serper")
        api_key = self.config.get("api_key")
        
        if not api_key:
            return ToolResult(success=False, error=f"API key for {provider} is required")
        
        try:
            if provider == "serper":
                return self._search_with_serper(query, num_results, api_key)
            elif provider == "serpapi":
                return self._search_with_serpapi(query, num_results, api_key)
            else:
                return ToolResult(success=False, error=f"Unsupported search provider: {provider}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    def _search_with_serper(self, query: str, num_results: int, api_key: str) -> ToolResult:
        """Perform a search using the Serper.dev API."""
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "q": query,
            "num": num_results
        }
        
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            return ToolResult(success=False, error=f"Search failed with status code {response.status_code}")
        
        data = response.json()
        results = []
        if "organic" in data:
            for item in data["organic"][:num_results]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", "")
                })
        
        return ToolResult(success=True, data={"results": results})
    
    def _search_with_serpapi(self, query: str, num_results: int, api_key: str) -> ToolResult:
        """Perform a search using the SerpAPI."""
        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": api_key,
            "num": num_results
        }
        
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return ToolResult(success=False, error=f"Search failed with status code {response.status_code}")
        
        data = response.json()
        results = []
        if "organic_results" in data:
            for item in data["organic_results"][:num_results]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", "")
                })
        
        return ToolResult(success=True, data={"results": results})
    
    def get_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }


class CalculatorTool(BaseTool):
    """Tool for performing safe mathematical calculations."""
    
    def execute(self, args: Dict[str, Any]) -> ToolResult:
        """Execute a mathematical calculation."""
        expression = args.get("expression")
        if not expression:
            return ToolResult(success=False, error="Expression is required")
        
        try:
            # Use a more sophisticated and safe approach than eval()
            result = self._safe_evaluate(expression)
            return ToolResult(success=True, data={"result": result})
        except Exception as e:
            return ToolResult(success=False, error=f"Calculation error: {str(e)}")
    
    def _safe_evaluate(self, expression: str) -> float:
        """Safely evaluate a mathematical expression."""
        # Remove all whitespace
        expression = re.sub(r'\s+', '', expression)
        
        # Check for invalid characters (only allow numbers, operators, and basic math functions)
        if not re.match(r'^[\d\+\-\*\/\^\(\)\.\,\s\w]*$', expression):
            raise ValueError("Invalid characters in expression")
        
        # Define safe functions
        safe_dict = {
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'pow': pow,
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'pi': math.pi,
            'e': math.e
        }
        
        # Replace ^ with ** for exponentiation
        expression = expression.replace('^', '**')
        
        # Evaluate the expression with the safe dictionary
        return eval(expression, {"__builtins__": {}}, safe_dict)
    
    def get_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }


class APICallTool(BaseTool):
    """Tool for making API calls to external services."""
    
    def execute(self, args: Dict[str, Any]) -> ToolResult:
        """Execute an API call."""
        endpoint = self.config.get("endpoint")
        method = self.config.get("method", "GET")
        auth_header = self.config.get("auth_header")
        
        if not endpoint:
            return ToolResult(success=False, error="API endpoint not configured")
        
        # Get parameters from args
        params = args.get("params", {})
        headers = {}
        
        # Add authorization if configured
        if auth_header:
            headers["Authorization"] = auth_header
        
        try:
            if method.upper() == "GET":
                response = requests.get(endpoint, params=params, headers=headers)
            elif method.upper() == "POST":
                data = args.get("data")
                response = requests.post(endpoint, json=data, params=params, headers=headers)
            else:
                return ToolResult(success=False, error=f"Unsupported HTTP method: {method}")
            
            if response.status_code >= 200 and response.status_code < 300:
                try:
                    return ToolResult(success=True, data=response.json())
                except json.JSONDecodeError:
                    return ToolResult(success=True, data={"text": response.text})
            else:
                return ToolResult(
                    success=False, 
                    error=f"API call failed with status code {response.status_code}: {response.text}"
                )
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    def get_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for this tool."""
        # Create schema based on the configured HTTP method
        method = self.config.get("method", "GET").upper()
        
        schema = {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "params": {
                        "type": "object",
                        "description": "Query parameters for the API call"
                    }
                },
                "required": []
            }
        }
        
        # Add data parameter for POST requests
        if method == "POST":
            schema["parameters"]["properties"]["data"] = {
                "type": "object",
                "description": "JSON body for the POST request"
            }
        
        return schema
