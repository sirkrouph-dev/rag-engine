# RAG Engine

A modular Retrieval-Augmented Generation (RAG) framework with configuration-as-code, plugin support, CLI, API, and UI.

## Features
- Modular pipeline (loader, chunker, embedder, vectorstore, retriever, prompting, LLM)
- Configurable via YAML or JSON
- Plugin system for custom components
- CLI (Typer), API (FastAPI), UI (Streamlit/Gradio)
- Pydantic-based config validation

## Quickstart
1. Install dependencies: `pip install -r requirements.txt`
2. Run CLI: `python -m rag_engine`

See `rag_engine_design.md` for architecture and details.
