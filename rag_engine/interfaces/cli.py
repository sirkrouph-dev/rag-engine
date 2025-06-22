"""
RAG Engine CLI interface with multi-framework API support.
"""
import typer
import os
from typing import Optional
from rag_engine.core.pipeline import Pipeline
from rag_engine.config.loader import load_config


app = typer.Typer(help="RAG Engine CLI - Modular Retrieval-Augmented Generation Framework")


@app.command()
def build(config: str = typer.Option(..., '--config', '-c', help='Path to config file')):
    """Build vector database from configuration."""
    try:
        if not os.path.exists(config):
            typer.echo(f"❌ Config file not found: {config}", err=True)
            raise typer.Exit(1)
        
        typer.echo(f"🚀 Building RAG pipeline with config: {config}")
        
        # Load configuration and build pipeline
        rag_config = load_config(config)
        pipeline = Pipeline(rag_config)
        pipeline.build()
        
        typer.echo("✅ Pipeline build completed successfully!")
        
    except Exception as e:
        typer.echo(f"❌ Build failed: {str(e)}", err=True)
        raise typer.Exit(1)


@app.command()
def chat(config: str = typer.Option(..., '--config', '-c', help='Path to config file')):
    """Start interactive chat with your data."""
    try:
        if not os.path.exists(config):
            typer.echo(f"❌ Config file not found: {config}", err=True)
            raise typer.Exit(1)
        
        typer.echo(f"🚀 Starting chat with config: {config}")
        
        # Load configuration and start chat
        rag_config = load_config(config)
        pipeline = Pipeline(rag_config)
        pipeline.chat()  # This starts the interactive chat loop
        
    except KeyboardInterrupt:
        typer.echo("\n👋 Chat session ended.")
    except Exception as e:
        typer.echo(f"❌ Chat failed: {str(e)}", err=True)
        raise typer.Exit(1)


@app.command()
def init(
    name: str = typer.Option("my-rag-project", '--name', '-n', help='Project name'),
    template: str = typer.Option("basic", '--template', '-t', help='Project template (basic, advanced)')
):
    """Initialize a new RAG project."""
    try:
        project_dir = os.path.join(os.getcwd(), name)
        
        if os.path.exists(project_dir):
            typer.echo(f"❌ Directory already exists: {project_dir}", err=True)
            raise typer.Exit(1)
        
        typer.echo(f"🚀 Initializing new RAG project: {name}")
        typer.echo(f"📁 Creating project directory: {project_dir}")
        
        # Create project structure
        os.makedirs(project_dir)
        os.makedirs(os.path.join(project_dir, "documents"))
        os.makedirs(os.path.join(project_dir, "configs"))
        os.makedirs(os.path.join(project_dir, "vector_store"))
        
        # Create example config
        config_content = _get_example_config(template)
        config_path = os.path.join(project_dir, "configs", "config.yml")
        
        with open(config_path, "w") as f:
            f.write(config_content)
        
        # Create README
        readme_content = _get_project_readme(name)
        readme_path = os.path.join(project_dir, "README.md")
        
        with open(readme_path, "w") as f:
            f.write(readme_content)
        
        typer.echo("✅ Project initialized successfully!")
        typer.echo(f"📖 Next steps:")
        typer.echo(f"   1. cd {name}")
        typer.echo(f"   2. Add documents to the ./documents/ folder")
        typer.echo(f"   3. Edit ./configs/config.yml with your settings")
        typer.echo(f"   4. Run: rag-engine build --config configs/config.yml")
        typer.echo(f"   5. Run: rag-engine chat --config configs/config.yml")
        
    except Exception as e:
        typer.echo(f"❌ Initialization failed: {str(e)}", err=True)
        raise typer.Exit(1)


@app.command()
def serve(
    config: str = typer.Option(..., '--config', '-c', help='Path to config file'),
    framework: str = typer.Option("fastapi", '--framework', '-f', help='API framework (fastapi, flask, django)'),
    host: str = typer.Option("0.0.0.0", '--host', help='Host to bind to'),
    port: int = typer.Option(8000, '--port', help='Port to bind to'),
    reload: bool = typer.Option(False, '--reload', help='Enable auto-reload for development'),
    ui: str = typer.Option(None, '--ui', help='Serve web UI (streamlit, gradio)'),
    ui_port: int = typer.Option(8501, '--ui-port', help='Port for UI server')
):
    """Serve the RAG Engine API and/or UI."""
    try:
        if not os.path.exists(config):
            typer.echo(f"❌ Config file not found: {config}", err=True)
            raise typer.Exit(1)
        
        # Start UI if requested
        if ui:
            typer.echo(f"🎨 Starting {ui.upper()} UI server on port {ui_port}...")
            
            if ui.lower() == "streamlit":
                try:
                    import subprocess
                    import sys
                    
                    # Create a temporary Streamlit app file
                    temp_app_content = f"""
import sys
sys.path.append('{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}')
from rag_engine.interfaces.ui import run_streamlit_ui
run_streamlit_ui('{config}')
"""
                    temp_app_path = "/tmp/streamlit_app.py"
                    with open(temp_app_path, "w") as f:
                        f.write(temp_app_content)
                    
                    # Run Streamlit
                    subprocess.run([
                        sys.executable, "-m", "streamlit", "run", 
                        temp_app_path, 
                        "--server.port", str(ui_port),
                        "--server.address", host
                    ])
                    return
                    
                except ImportError:
                    typer.echo("❌ Streamlit not installed. Run: pip install streamlit", err=True)
                    raise typer.Exit(1)
            
            elif ui.lower() == "gradio":
                try:
                    from rag_engine.interfaces.ui import run_gradio_ui
                    run_gradio_ui(config, server_name=host, server_port=ui_port)
                    return
                    
                except ImportError:
                    typer.echo("❌ Gradio not installed. Run: pip install gradio", err=True)
                    raise typer.Exit(1)
            
            else:
                typer.echo(f"❌ Unsupported UI framework: {ui}. Use 'streamlit' or 'gradio'", err=True)
                raise typer.Exit(1)
        
        # Start API server
        typer.echo(f"🚀 Starting RAG Engine API server...")
        typer.echo(f"⚙️  Framework: {framework}")
        typer.echo(f"🔧 Config: {config}")
        typer.echo(f"🌐 Host: {host}:{port}")
        
        # Import the API factory
        from rag_engine.interfaces.base_api import APIModelFactory
        
        # Import framework implementations to register them
        try:
            from rag_engine.interfaces.api import FastAPIServer  # FastAPI
        except ImportError:
            typer.echo("⚠️  FastAPI not available")
        
        try:
            from rag_engine.interfaces.flask_api import FlaskServer  # Flask
        except ImportError:
            typer.echo("⚠️  Flask not available")
        
        try:
            from rag_engine.interfaces.django_api import DjangoServer  # Django
        except ImportError:
            typer.echo("⚠️  Django not available")
        
        # List available frameworks
        available_frameworks = APIModelFactory.list_frameworks()
        typer.echo(f"📋 Available frameworks: {', '.join(available_frameworks)}")
        
        if framework not in available_frameworks:
            typer.echo(f"❌ Framework '{framework}' not available. Use one of: {', '.join(available_frameworks)}", err=True)
            raise typer.Exit(1)
        
        # Create and start the server
        server = APIModelFactory.create_server(framework, config_path=config)
        server.start_server(host=host, port=port, reload=reload)
        
    except KeyboardInterrupt:
        typer.echo("\n👋 Server stopped.")
    except Exception as e:
        typer.echo(f"❌ Server failed: {str(e)}", err=True)
        raise typer.Exit(1)


def _get_example_config(template: str) -> str:
    """Get example configuration based on template."""
    if template == "advanced":
        return """# Advanced RAG Engine Configuration
documents:
  - type: pdf
    path: ./documents/sample.pdf
  - type: txt
    path: ./documents/sample.txt

chunking:
  method: recursive
  max_tokens: 512
  overlap: 50

embedding:
  provider: openai
  model: text-embedding-3-small
  api_key: ${OPENAI_API_KEY}

vectorstore:
  provider: chroma
  persist_directory: ./vector_store

retrieval:
  top_k: 5

prompting:
  system_prompt: >
    You are a helpful AI assistant. Answer questions based on the provided context.
    Be accurate, concise, and cite sources when possible.

llm:
  provider: openai
  model: gpt-4
  temperature: 0.3
  api_key: ${OPENAI_API_KEY}

output:
  method: console
"""
    else:
        return """# Basic RAG Engine Configuration
documents:
  - type: txt
    path: ./documents/sample.txt

chunking:
  method: fixed
  max_tokens: 256
  overlap: 20

embedding:
  provider: huggingface
  model: sentence-transformers/all-MiniLM-L6-v2

vectorstore:
  provider: chroma
  persist_directory: ./vector_store

retrieval:
  top_k: 3

prompting:
  system_prompt: "You are a helpful assistant."

llm:
  provider: openai
  model: gpt-3.5-turbo
  temperature: 0.7
  api_key: ${OPENAI_API_KEY}

output:
  method: console
"""


def _get_project_readme(name: str) -> str:
    """Get project README content."""
    return f"""# {name}

A RAG (Retrieval-Augmented Generation) project powered by RAG Engine.

## Quick Start

1. **Add Documents**: Place your documents in the `./documents/` folder
2. **Configure**: Edit `./configs/config.yml` with your settings and API keys
3. **Build**: Run `rag-engine build --config configs/config.yml`
4. **Chat**: Run `rag-engine chat --config configs/config.yml`

## API Server

Start the API server:
```bash
rag-engine serve --config configs/config.yml --framework fastapi
```

Available frameworks: `fastapi`, `flask`, `django`

## Project Structure

```
{name}/
├── documents/          # Place your documents here
├── configs/           # Configuration files
│   └── config.yml    # Main configuration
├── vector_store/     # Vector database storage
└── README.md         # This file
```

## Environment Variables

Set these environment variables or add them to a `.env` file:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## Learn More

- [RAG Engine Documentation](https://github.com/your-repo/rag-engine)
- [Configuration Guide](https://github.com/your-repo/rag-engine/docs/config)
- [API Reference](https://github.com/your-repo/rag-engine/docs/api)
"""
