"""
RAG Engine CLI interface with multi-framework API support.
"""
import typer
import os
from typing import Optional
from rag_engine.core.pipeline import Pipeline
from rag_engine.config.loader import load_config


app = typer.Typer(
    help="RAG Engine CLI - Modular Retrieval-Augmented Generation Framework",
    no_args_is_help=True,
    rich_markup_mode=None  # Disable rich formatting to avoid compatibility issues
)


@app.command()
def build(config: Optional[str] = typer.Option(None, '--config', '-c', help='Path to config file (auto-detects if not provided)')):
    """Build vector database from configuration."""
    try:
        # Smart config detection if not provided
        if config is None:
            possible_configs = [
                "config.json",
                "config.yml", 
                "config.yaml",
                "examples/configs/demo_local_config.json",
                "configs/config.json"
            ]
            
            for possible_config in possible_configs:
                if os.path.exists(possible_config):
                    config = possible_config
                    typer.echo(f"🔍 Auto-detected config: {config}")
                    break
            
            if config is None:
                typer.echo("❌ No config file found. Please provide one with --config or create config.json", err=True)
                typer.echo("💡 Try copying: cp examples/configs/demo_local_config.json config.json", err=True)
                raise typer.Exit(1)
        
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
def chat(config: Optional[str] = typer.Option(None, '--config', '-c', help='Path to config file (auto-detects if not provided)')):
    """Start interactive chat with your data."""
    try:
        # Smart config detection if not provided
        if config is None:
            possible_configs = [
                "config.json",
                "config.yml", 
                "config.yaml",
                "examples/configs/demo_local_config.json",
                "configs/config.json"
            ]
            
            for possible_config in possible_configs:
                if os.path.exists(possible_config):
                    config = possible_config
                    typer.echo(f"🔍 Auto-detected config: {config}")
                    break
            
            if config is None:
                typer.echo("❌ No config file found. Please provide one with --config or create config.json", err=True)
                typer.echo("💡 Try copying: cp examples/configs/demo_local_config.json config.json", err=True)
                raise typer.Exit(1)
        
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
    config: Optional[str] = typer.Option(None, '--config', '-c', help='Path to config file (defaults to config.json or examples/configs/demo_local_config.json)'),
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
        # Smart config detection if not provided
        if config is None:
            possible_configs = [
                "config.json",
                "config.yml", 
                "config.yaml",
                "examples/configs/demo_local_config.json",
                "configs/config.json",
                "configs/config.yml"
            ]
            
            for possible_config in possible_configs:
                if os.path.exists(possible_config):
                    config = possible_config
                    typer.echo(f"🔍 Auto-detected config: {config}")
                    break
            
            if config is None:
                typer.echo("❌ No config file found. Please provide one with --config or create config.json", err=True)
                typer.echo("💡 Try copying: cp examples/configs/demo_local_config.json config.json", err=True)
                raise typer.Exit(1)
        
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



@app.command()
def ask(
    question: str = typer.Argument(..., help='Question to ask the RAG Engine assistant'),
    model: str = typer.Option("phi3.5:latest", '--model', '-m', help='LLM model to use for assistance')
):
    """Ask the RAG Engine AI assistant for help and guidance."""
    try:
        # Import here to avoid dependency issues if ollama is not installed
        try:
            import ollama
        except ImportError:
            typer.echo("❌ AI assistant not available. Please install ollama-python:")
            typer.echo("   pip install ollama-python")
            raise typer.Exit(1)
        
        typer.echo(f"🤖 Asking RAG Engine assistant: {question}")
        
        # System prompt for the assistant
        system_prompt = """You are a helpful RAG Engine assistant specialized in package bloat management and optimization. You provide support for users of the RAG Engine framework.

STACK INFORMATION:
- DEMO: Quick demos (~200MB) - Minimal deps: ollama-python, sentence-transformers, faiss-cpu, typer, rich
- LOCAL: Local development (~500MB) - Adds: transformers, torch, multiple vector stores, advanced chunking  
- CLOUD: Production APIs (~100MB) - Cloud APIs only: openai, anthropic, requests, minimal local processing
- MINI: Embedded systems (~50MB) - Ultra minimal: only core logic, no UI, basic text processing
- FULL: Everything (~1GB) - All features: research models, advanced chunking, multiple frameworks
- RESEARCH: Academic (~1GB+) - Cutting-edge: experimental models, specialized libraries

BLOAT MANAGEMENT STRATEGIES:
1. **Tiered Requirements**: Use requirements-{stack}.txt files for different use cases
2. **Optional Dependencies**: Install only what's needed with pip extras: pip install rag-engine[demo]
3. **Lazy Imports**: Import heavy libraries only when needed at runtime  
4. **Runtime Detection**: Auto-detect available packages and gracefully fallback
5. **Plugin Architecture**: Load components dynamically based on user needs
6. **Dependency Analysis**: Help users understand what each package does and if they need it

PACKAGE OPTIMIZATION:
- Suggest lighter alternatives (e.g., sentence-transformers vs full transformers)
- Identify unused dependencies in current setup
- Recommend stack switches for user needs
- Help with dependency conflicts and version pinning
- Guide on Docker vs local installs for bloat reduction

You can help with:
- Configuration questions and stack recommendations
- Package bloat analysis and optimization suggestions  
- Troubleshooting setup issues and dependency conflicts
- Best practices for different use cases and environments
- Model recommendations and performance optimization
- Feature explanations and upgrade/downgrade paths
- Analyzing requirements.txt files and suggesting improvements
- Docker vs local installation trade-offs

Be concise, helpful, and provide actionable advice. Use emojis appropriately. Always consider the user's specific needs and suggest the minimal viable setup."""

        # Get response from local LLM
        client = ollama.Client()
        
        # Check if model is available
        try:
            response = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ]
            )
            
            answer = response['message']['content']
            typer.echo(f"\n🤖 Assistant: {answer}")
            
        except ollama.ResponseError as e:
            if "not found" in str(e):
                typer.echo(f"❌ Model {model} not found. Pulling it now...")
                client.pull(model)
                # Retry the question
                response = client.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question}
                    ]
                )
                answer = response['message']['content']
                typer.echo(f"\n🤖 Assistant: {answer}")
            else:
                raise e
        
    except Exception as e:
        typer.echo(f"❌ Assistant error: {str(e)}", err=True)
        typer.echo("💡 Tip: Make sure Ollama is running: ollama serve")
        raise typer.Exit(1)


@app.command()
def analyze_bloat(
    requirements_file: Optional[str] = typer.Option(None, '--file', '-f', help='Requirements file to analyze (defaults to requirements.txt)'),
    stack: Optional[str] = typer.Option(None, '--stack', '-s', help='Compare against specific stack (demo, local, cloud, mini, full, research)')
):
    """Analyze package bloat and suggest optimizations using AI assistant."""
    try:
        # Import here to avoid dependency issues if ollama is not installed
        try:
            import ollama
        except ImportError:
            typer.echo("❌ AI assistant not available. Please install ollama-python:")
            typer.echo("   pip install ollama-python")
            raise typer.Exit(1)
        
        # Determine requirements file
        if requirements_file is None:
            possible_files = [
                "requirements.txt",
                "requirements-demo.txt", 
                "requirements-local.txt",
                "requirements-cloud.txt",
                "requirements-mini.txt",
                "requirements-full.txt",
                "requirements-research.txt"
            ]
            
            for possible_file in possible_files:
                if os.path.exists(possible_file):
                    requirements_file = possible_file
                    typer.echo(f"🔍 Analyzing: {requirements_file}")
                    break
            
            if requirements_file is None:
                typer.echo("❌ No requirements file found", err=True)
                raise typer.Exit(1)
        
        if not os.path.exists(requirements_file):
            typer.echo(f"❌ Requirements file not found: {requirements_file}", err=True)
            raise typer.Exit(1)
        
        # Read requirements file
        with open(requirements_file, 'r') as f:
            requirements_content = f.read()
        
        # Build analysis prompt
        analysis_prompt = f"""Please analyze this requirements file for package bloat and suggest optimizations:

FILE: {requirements_file}
CONTENT:
{requirements_content}

"""
        
        if stack:
            analysis_prompt += f"TARGET STACK: {stack.upper()}\n"
            analysis_prompt += "Please suggest how to optimize this for the target stack.\n"
        
        analysis_prompt += """
Please provide:
1. 📊 Bloat analysis (heavy packages, unused deps)
2. 🎯 Optimization suggestions (lighter alternatives)
3. 📦 Stack recommendations (which stack fits best)
4. ⚡ Quick wins (easy reductions)
5. 🔧 Specific commands to optimize

Be practical and actionable."""

        typer.echo(f"🤖 Analyzing package bloat in {requirements_file}...")
        
        # Get AI analysis
        client = ollama.Client()
        try:
            response = client.chat(
                model="phi3.5:latest",
                messages=[
                    {"role": "user", "content": analysis_prompt}
                ]
            )
            
            analysis = response['message']['content']
            typer.echo(f"\n🤖 Bloat Analysis:\n{analysis}")
            
        except Exception as e:
            typer.echo(f"❌ Analysis failed: {str(e)}", err=True)
            typer.echo("💡 Tip: Make sure Ollama is running and phi3.5:latest is installed")
        
    except Exception as e:
        typer.echo(f"❌ Bloat analysis failed: {str(e)}", err=True)
        raise typer.Exit(1)


@app.command()
def optimize_stack(
    current_stack: str = typer.Argument(..., help='Current stack (demo, local, cloud, mini, full, research)'),
    target_stack: str = typer.Argument(..., help='Target stack to optimize for'),
    interactive: bool = typer.Option(True, '--interactive/--no-interactive', help='Interactive optimization with AI guidance')
):
    """Optimize dependencies for a different stack using AI guidance."""
    try:
        if interactive:
            # Use AI assistant for interactive optimization
            try:
                import ollama
                client = ollama.Client()
            except ImportError:
                typer.echo("❌ AI assistant not available. Please install ollama-python")
                raise typer.Exit(1)
            
            optimization_prompt = f"""I want to optimize my RAG Engine setup from {current_stack.upper()} stack to {target_stack.upper()} stack.

Please guide me through:
1. 📋 What packages can I remove?
2. 📦 What packages do I need to add?
3. ⚙️ What configuration changes are needed?
4. 🚀 Step-by-step migration commands
5. ⚠️ What functionality will change?

Current stack: {current_stack.upper()}
Target stack: {target_stack.upper()}

Please provide specific pip commands and file changes needed."""

            typer.echo(f"🤖 Optimizing from {current_stack.upper()} → {target_stack.upper()}...")
            
            try:
                response = client.chat(
                    model="phi3.5:latest", 
                    messages=[
                        {"role": "user", "content": optimization_prompt}
                    ]
                )
                
                guidance = response['message']['content']
                typer.echo(f"\n🤖 Optimization Guide:\n{guidance}")
                
                # Ask if user wants to proceed
                if typer.confirm("\n💡 Would you like to apply these optimizations automatically?"):
                    typer.echo("🔧 Auto-optimization not implemented yet. Please follow the manual steps above.")
                    typer.echo("💡 Tip: Use 'rag-engine ask' for follow-up questions!")
                
            except Exception as e:
                typer.echo(f"❌ Optimization guidance failed: {str(e)}", err=True)
        else:
            # Non-interactive mode
            typer.echo(f"🔧 Non-interactive optimization from {current_stack} to {target_stack}")
            typer.echo("💡 This would automatically optimize your setup (not implemented yet)")
            typer.echo("💡 Use --interactive for AI-guided optimization")
            
    except Exception as e:
        typer.echo(f"❌ Stack optimization failed: {str(e)}", err=True)
        raise typer.Exit(1)


@app.command()
def dependency_audit(
    show_sizes: bool = typer.Option(True, '--sizes/--no-sizes', help='Show estimated package sizes'),
    unused_only: bool = typer.Option(False, '--unused-only', help='Only show potentially unused packages')
):
    """Audit current dependencies with AI-powered analysis."""
    try:
        typer.echo("🔍 Auditing current Python dependencies...")
        
        # Get currently installed packages
        import subprocess
        import sys
        
        result = subprocess.run([sys.executable, "-m", "pip", "list", "--format=freeze"], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            typer.echo("❌ Failed to get package list", err=True)
            raise typer.Exit(1)
        
        installed_packages = result.stdout
        
        # Use AI to analyze dependencies
        try:
            import ollama
            client = ollama.Client()
            
            audit_prompt = f"""Please audit these installed Python packages for a RAG Engine project:

INSTALLED PACKAGES:
{installed_packages}

Please analyze:
1. 📊 Large packages (>50MB) that might be bloat
2. 🔍 Packages that seem unused for RAG functionality  
3. 🎯 Lighter alternatives for heavy packages
4. 📦 Which stack (DEMO/LOCAL/CLOUD/MINI/FULL) this setup resembles
5. 💡 Specific recommendations to reduce bloat

Focus on RAG-related packages: transformers, torch, faiss, sentence-transformers, ollama, openai, etc.
Ignore common Python packages like pip, setuptools unless they're problematic."""

            typer.echo("🤖 AI is analyzing your dependencies...")
            
            response = client.chat(
                model="phi3.5:latest",
                messages=[
                    {"role": "user", "content": audit_prompt}
                ]
            )
            
            audit_results = response['message']['content']
            typer.echo(f"\n🤖 Dependency Audit Results:\n{audit_results}")
            
        except ImportError:
            typer.echo("⚠️ AI analysis not available (ollama-python not installed)")
            typer.echo("📋 Raw package list:")
            typer.echo(installed_packages)
            
        except Exception as e:
            typer.echo(f"⚠️ AI analysis failed: {str(e)}")
            typer.echo("📋 Raw package list:")
            typer.echo(installed_packages)
        
    except Exception as e:
        typer.echo(f"❌ Dependency audit failed: {str(e)}", err=True)
        raise typer.Exit(1)


def _get_example_config(template: str) -> str:
    """Generate example config content based on template."""
    if template == "advanced":
        return """# Advanced RAG Engine Configuration
documents:
  - type: file
    path: "./documents"
  - type: url
    path: "https://example.com/docs"

chunking:
  method: recursive
  max_tokens: 1000
  overlap: 100

embedding:
  provider: openai
  model: text-embedding-ada-002
  api_key: ${OPENAI_API_KEY}

vectorstore:
  provider: pinecone
  persist_directory: "./vector_store"

retrieval:
  top_k: 5

prompting:
  system_prompt: "You are an expert assistant that provides detailed answers based on the provided context."

llm:
  provider: openai
  model: gpt-4
  temperature: 0.2

output:
  method: text
"""
    else:  # basic template
        return """# Basic RAG Engine Configuration
documents:
  - type: file
    path: "./documents"

chunking:
  method: recursive
  max_tokens: 500
  overlap: 50

embedding:
  provider: sentence_transformers
  model: all-MiniLM-L6-v2

vectorstore:
  provider: faiss
  persist_directory: "./vector_store"

retrieval:
  top_k: 3

prompting:
  system_prompt: "You are a helpful assistant that answers questions based on the provided context."

llm:
  provider: ollama
  model: llama3.2:1b
  temperature: 0.7

output:
  method: text
"""


def _get_project_readme(name: str) -> str:
    """Generate README content for new project."""
    return f"""# {name}

A RAG (Retrieval-Augmented Generation) project powered by RAG Engine.

## Quick Start

1. **Add your documents:**
   ```bash
   # Copy your documents to the documents/ folder
   cp /path/to/your/docs/* documents/
   ```

2. **Configure the system:**
   ```bash
   # Edit the configuration file
   nano configs/config.yml
   ```

3. **Build the knowledge base:**
   ```bash
   rag-engine build --config configs/config.yml
   ```

4. **Start chatting:**
   ```bash
   rag-engine chat --config configs/config.yml
   ```

## API Server

Start the API server:
```bash
rag-engine serve --config configs/config.yml
```

The API will be available at `http://localhost:8000`

## Configuration

Edit `configs/config.yml` to customize:
- Document sources
- Embedding models
- Vector storage
- LLM providers
- Retrieval settings

## Need Help?

Use the AI assistant:
```bash
rag-engine ask "How do I add more documents?"
rag-engine ask "Which embedding model should I use?"
rag-engine ask "How do I optimize for speed?"
```

## Package Management

Analyze and optimize your dependencies:
```bash
# Analyze current package bloat
rag-engine analyze-bloat

# Audit dependencies
rag-engine dependency-audit

# Optimize for different stack
rag-engine optimize-stack current_stack target_stack
```
"""


# ...existing code...
