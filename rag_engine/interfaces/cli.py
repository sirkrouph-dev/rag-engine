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
    orchestrator: str = typer.Option("default", '--orchestrator', '-o', help='Orchestrator type (default, hybrid, multimodal)'),
    host: str = typer.Option("0.0.0.0", '--host', help='Host to bind to'),
    port: int = typer.Option(8000, '--port', help='Port to bind to'),
    workers: int = typer.Option(1, '--workers', '-w', help='Number of worker processes (production scaling)'),
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
          # Import the enhanced API factory
        from rag_engine.interfaces.enhanced_base_api import enhanced_factory
        from rag_engine.interfaces.enhanced_base_api import APICustomization, AuthMethod, RateLimitType
          # Import and register enhanced framework implementations
        try:
            from rag_engine.interfaces.fastapi_enhanced import FastAPIEnhanced
            enhanced_factory.register_framework("fastapi", FastAPIEnhanced)
            typer.echo("✅ FastAPI Enhanced registered")
        except ImportError:
            typer.echo("⚠️  FastAPI Enhanced not available")
        
        try:
            from rag_engine.interfaces.flask_enhanced import FlaskEnhanced
            enhanced_factory.register_framework("flask", FlaskEnhanced)
            typer.echo("✅ Flask Enhanced registered")
        except ImportError:
            typer.echo("⚠️  Flask Enhanced not available")
        
        try:
            from rag_engine.interfaces.django_enhanced import DjangoEnhanced
            enhanced_factory.register_framework("django", DjangoEnhanced)
            typer.echo("✅ Django Enhanced registered")
        except ImportError:
            typer.echo("⚠️  Django Enhanced not available")
        
        # Register example custom servers
        try:
            from rag_engine.interfaces.custom_servers import register_example_servers
            register_example_servers()
            custom_servers = enhanced_factory.list_custom_servers()
            if custom_servers:
                typer.echo(f"✅ Custom servers registered: {', '.join(custom_servers)}")
        except ImportError:
            typer.echo("⚠️  Custom servers not available")
        
        # List available frameworks
        available_frameworks = enhanced_factory.list_frameworks()
        builtin_frameworks = enhanced_factory.list_builtin_frameworks()
        custom_servers = enhanced_factory.list_custom_servers()
        
        typer.echo(f"📋 Built-in frameworks: {', '.join(builtin_frameworks)}")
        if custom_servers:
            typer.echo(f"🔧 Custom servers: {', '.join(custom_servers)}")
        typer.echo(f"📋 Total available: {', '.join(available_frameworks)}")
        
        if framework not in available_frameworks:
            typer.echo(f"❌ Framework '{framework}' not available.", err=True)
            
            # Show detailed help for custom servers
            if framework.startswith('custom'):
                typer.echo("\n💡 To use a custom server:", err=True)
                typer.echo("1. Create a custom server class", err=True)
                typer.echo("2. Register it with enhanced_factory.register_custom_server()", err=True)
                typer.echo("3. Use the registered name as framework", err=True)
                typer.echo("\nSee ENHANCED_API_GUIDE.md for examples", err=True)
            
            raise typer.Exit(1)
        
        # Show framework info if it's a custom server
        framework_info = enhanced_factory.get_framework_info(framework)
        if framework_info and framework_info.get('type') == 'custom':
            typer.echo(f"🔧 Using custom server: {framework_info.get('description', 'Custom server')}")
          # Create API customization configuration
        api_config = APICustomization(
            host=host,
            port=port,
            workers=workers,
            debug=reload,
            reload=reload,
            enable_docs=True,
            enable_metrics=True,
            enable_health_checks=True,
            enable_rate_limiting=True,
            enable_compression=True,
            enable_request_logging=True,
            requests_per_minute=100,
            cors_origins=["*"],
            auth_method=AuthMethod.NONE,  # Can be configured via environment
            custom_headers={
                "X-Powered-By": "RAG-Engine-Enhanced",
                "X-Framework": framework.upper()
            }
        )
        
        # Validate orchestrator type
        from rag_engine.core.orchestration import OrchestratorFactory
        
        # Load alternative orchestrators
        try:
            import rag_engine.core.alternative_orchestrators
            typer.echo("✅ Alternative orchestrators loaded")
        except ImportError:
            typer.echo("⚠️  Alternative orchestrators not available")
        
        available_orchestrators = OrchestratorFactory.list_orchestrators()
        typer.echo(f"🧠 Available orchestrators: {', '.join(available_orchestrators)}")
        
        if orchestrator not in available_orchestrators:
            typer.echo(f"❌ Orchestrator '{orchestrator}' not available.", err=True)
            typer.echo(f"Available options: {', '.join(available_orchestrators)}", err=True)
            raise typer.Exit(1)
        
        typer.echo(f"🧠 Using orchestrator: {orchestrator}")
        
        if orchestrator not in available_orchestrators:
            typer.echo(f"❌ Orchestrator '{orchestrator}' not available.", err=True)
            typer.echo(f"Available options: {', '.join(available_orchestrators)}", err=True)
            raise typer.Exit(1)
        
        typer.echo(f"🧠 Using orchestrator: {orchestrator}")
        
        # Load RAG configuration
        rag_config = load_config(config) if config else None
        
        # Create and start the enhanced server
        server = enhanced_factory.create_server(
            framework, 
            config=rag_config, 
            api_config=api_config,
            orchestrator_type=orchestrator
        )
        
        # Production scaling info
        if workers > 1:
            typer.echo(f"🏭 Production mode: Using {workers} workers for scalability")
        elif reload:
            typer.echo("🛠️  Development mode: Auto-reload enabled")
        
        server.start_server(host=host, port=port, workers=workers, reload=reload)
        
    except KeyboardInterrupt:
        typer.echo("\n👋 Server stopped.")
    except Exception as e:
        typer.echo(f"❌ Server failed: {str(e)}", err=True)
        raise typer.Exit(1)


@app.command()
def custom_server(
    action: str = typer.Argument(..., help="Action: list, create, validate, register"),
    name: str = typer.Option(None, '--name', '-n', help='Server name for create/register actions'),
    framework: str = typer.Option("custom", '--framework', '-f', help='Framework type for template'),
    file_path: str = typer.Option(None, '--file', help='Path to custom server file for register/validate')
):
    """Manage custom server implementations."""
    
    if action == "list":
        from rag_engine.interfaces.enhanced_base_api import enhanced_factory
        from rag_engine.interfaces.custom_servers import register_example_servers
        
        # Register example servers to show what's available
        try:
            register_example_servers()
        except:
            pass
        
        builtin = enhanced_factory.list_builtin_frameworks()
        custom = enhanced_factory.list_custom_servers()
        
        typer.echo("📋 Available Servers:")
        typer.echo(f"  Built-in: {', '.join(builtin) if builtin else 'None'}")
        typer.echo(f"  Custom: {', '.join(custom) if custom else 'None'}")
        
        if custom:
            typer.echo("\n🔧 Custom Server Details:")
            for server_name in custom:
                info = enhanced_factory.get_framework_info(server_name)
                typer.echo(f"  • {server_name}: {info.get('description', 'No description')}")
    
    elif action == "create":
        if not name:
            typer.echo("❌ Name is required for create action", err=True)
            raise typer.Exit(1)
        
        from rag_engine.interfaces.custom_servers import create_custom_server_template        
        template = create_custom_server_template(name, framework)
        
        filename = f"{name.lower()}_server.py"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(template)
        
        typer.echo(f"✅ Created custom server template: {filename}")
        typer.echo(f"📝 Edit the file to implement your {framework} server")
        typer.echo(f"🚀 Then use: rag-engine serve --framework {name.lower()}")
    
    elif action == "validate":
        if not file_path:
            typer.echo("❌ File path is required for validate action", err=True)
            raise typer.Exit(1)
        
        try:
            import importlib.util
            import sys
            
            # Load the module
            spec = importlib.util.spec_from_file_location("custom_server", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find server classes
            from rag_engine.interfaces.custom_servers import validate_custom_server_implementation
            
            server_classes = []
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    hasattr(attr, 'create_app') and 
                    hasattr(attr, 'start_server')):
                    server_classes.append(attr)
            
            if not server_classes:
                typer.echo("❌ No custom server classes found in file", err=True)
                raise typer.Exit(1)
            
            # Validate each server class
            all_valid = True
            for server_class in server_classes:
                issues = validate_custom_server_implementation(server_class)
                if issues:
                    typer.echo(f"❌ {server_class.__name__} has issues:")
                    for issue in issues:
                        typer.echo(f"  • {issue}")
                    all_valid = False
                else:
                    typer.echo(f"✅ {server_class.__name__} is valid")
            
            if all_valid:
                typer.echo("🎉 All server implementations are valid!")
            else:
                raise typer.Exit(1)
                
        except Exception as e:
            typer.echo(f"❌ Validation failed: {e}", err=True)
            raise typer.Exit(1)
    
    elif action == "register":
        if not name or not file_path:
            typer.echo("❌ Both name and file path are required for register action", err=True)
            raise typer.Exit(1)
        
        try:
            import importlib.util
            
            # Load the module
            spec = importlib.util.spec_from_file_location("custom_server", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Register with factory (this would typically be done in the module itself)
            typer.echo(f"✅ Custom server module loaded from {file_path}")
            typer.echo(f"🔧 Server should register itself with enhanced_factory.register_custom_server()")
            typer.echo(f"🚀 Now you can use: rag-engine serve --framework {name}")
            
        except Exception as e:
            typer.echo(f"❌ Registration failed: {e}", err=True)
            raise typer.Exit(1)
    
    else:
        typer.echo(f"❌ Unknown action: {action}", err=True)
        typer.echo("Available actions: list, create, validate, register")
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
