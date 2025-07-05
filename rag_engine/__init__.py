"""
RAG Engine - Modular Retrieval-Augmented Generation Framework
============================================================

A modular, AI-powered framework for building advanced RAG pipelines with
zero-bloat installation, intelligent stack selection, and user-friendly automation.

Key Features:
- 🤖 AI-powered setup and ongoing assistance
- ⚡ Instant demo and quick start options
- 🏗️ Modular, plugin-based architecture
- 🎨 Modern Vue.js frontend with dark mode
- 🚀 Multi-framework API support (FastAPI, etc.)
"""

__version__ = "0.1.0-experimental"
__author__ = "RAG Engine Development Team"
__license__ = "MIT"

# Core components
from .core.orchestration import ComponentRegistry, BaseOrchestrator, DefaultOrchestrator, OrchestratorFactory

# Configuration
try:
    from .config.loader import ConfigLoader
except ImportError:
    pass

# Interfaces
try:
    from .interfaces.cli import app as cli_app
except ImportError:
    pass

__all__ = [
    "__version__",
    "__author__", 
    "__license__",
    "ComponentRegistry",
    "BaseOrchestrator", 
    "DefaultOrchestrator",
    "OrchestratorFactory",
]
