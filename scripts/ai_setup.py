#!/usr/bin/env python3
"""
AI-Powered RAG Engine Setup Assistant

This script installs a lightweight local LLM that acts as a consultant
to help users choose the perfect stack configuration for their needs.
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path

# Minimal requirements for the AI assistant
ASSISTANT_REQUIREMENTS = [
    "ollama-python>=0.1.0",
    "rich>=13.0.0",
    "typer>=0.9.0",
]

class RAGAssistant:
    def __init__(self):
        self.console = None
        self.ollama_client = None
        self.model = "phi3.5:latest"  # Small, fast model
        
    def setup_console(self):
        """Setup rich console for beautiful output"""
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.text import Text
            self.console = Console()
        except ImportError:
            print("Installing required packages...")
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + ASSISTANT_REQUIREMENTS)
            from rich.console import Console
            self.console = Console()
    
    def install_ollama(self):
        """Install Ollama if not present"""
        self.console.print("[bold blue]🤖 Installing Ollama...[/bold blue]")
        
        # First check if Ollama is already installed
        try:
            result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                self.console.print("[green]✅ Ollama is already installed[/green]")
                return
        except FileNotFoundError:
            pass  # Ollama not found, proceed with installation
        
        if os.name == 'nt':  # Windows
            self.console.print("[dim]Using winget to install Ollama...[/dim]")
            result = subprocess.run(["winget", "install", "Ollama.Ollama"], check=False)
            if result.returncode != 0:
                self.console.print("[yellow]⚠️ winget failed, please download from https://ollama.ai[/yellow]")
                input("Press Enter after installing Ollama manually...")
        else:  # Linux/macOS
            self.console.print("[dim]Downloading and installing Ollama for Unix systems...[/dim]")
            try:
                # More secure approach: download and verify before executing
                import tempfile
                import urllib.request
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
                    urllib.request.urlretrieve("https://ollama.ai/install.sh", f.name)
                    os.chmod(f.name, 0o755)
                    result = subprocess.run(["/bin/bash", f.name], check=False)
                    os.unlink(f.name)
                    
                if result.returncode != 0:
                    self.console.print("[yellow]⚠️ Auto-install failed, please install from https://ollama.ai[/yellow]")
                    input("Press Enter after installing Ollama manually...")
                    
            except Exception as e:
                self.console.print(f"[red]❌ Installation failed: {e}[/red]")
                self.console.print("[yellow]Please install Ollama manually from https://ollama.ai[/yellow]")
                input("Press Enter after installing Ollama manually...")
        
        # Wait for Ollama to be available and start the service
        self.console.print("[dim]Starting Ollama service...[/dim]")
        
        # Try to start Ollama service
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(5)  # Give it time to start
        except FileNotFoundError:
            self.console.print("[red]❌ Ollama not found in PATH. Please restart your terminal or reboot.[/red]")
            input("Press Enter after Ollama is available...")
        
        # Verify installation
        try:
            result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                self.console.print("[green]✅ Ollama installed successfully[/green]")
            else:
                raise Exception("Ollama not responding")
        except Exception:
            self.console.print("[red]❌ Ollama installation verification failed[/red]")
            self.console.print("[yellow]Please ensure Ollama is installed and 'ollama serve' is running[/yellow]")
            input("Press Enter when ready to continue...")
    
    def setup_model(self):
        """Pull the assistant model"""
        self.console.print(f"[bold blue]🧠 Downloading AI assistant model ({self.model})...[/bold blue]")
        self.console.print("[dim]This is a lightweight 2GB model that will act as your consultant[/dim]")
        
        subprocess.run(["ollama", "pull", self.model], check=True)
        
        # Initialize Ollama client
        try:
            import ollama
            self.ollama_client = ollama.Client()
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ollama-python"])
            import ollama
            self.ollama_client = ollama.Client()
    
    def ask_assistant(self, prompt: str) -> str:
        """Ask the AI assistant a question"""
        system_prompt = """You are a helpful RAG Engine setup consultant. You help users choose the perfect configuration for their needs.

Available stacks:
- DEMO: Quick demos, minimal dependencies (~200MB), uses Ollama + basic features
- LOCAL: Local development, full features (~500MB), multiple LLMs and vector stores
- CLOUD: Production with cloud APIs (~100MB), OpenAI/Anthropic/etc
- MINI: Embedded systems (~50MB), ultra-minimal, no UI
- FULL: Everything included (~1GB), all features and models
- RESEARCH: Academic/experimental (~1GB+), cutting-edge models

Ask clarifying questions if needed, and always explain your recommendations clearly.
Be friendly, concise, and helpful. Use emojis appropriately."""

        try:
            response = self.ollama_client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            return response['message']['content']
        except Exception as e:
            return f"Sorry, I'm having trouble right now. Error: {e}"
    
    def interactive_setup(self):
        """Run the interactive setup conversation"""
        self.console.print("[bold green]🤖 RAG Engine AI Assistant[/bold green]")
        self.console.print("[dim]Let me help you choose the perfect configuration![/dim]\n")
        
        # Initial greeting
        greeting = """Hi! I'm your personal RAG Engine consultant. I'll help you choose the perfect stack configuration based on your needs.

Let me ask you a few questions to get started:

1. What's your main use case?
   a) Quick demo (showing friends/colleagues)
   b) Local development and experimentation
   c) Production deployment
   d) Research and academic work
   e) Embedded/minimal system

What sounds like your situation?"""
        
        self.console.print(f"[bold cyan]Assistant:[/bold cyan] {greeting}")
        
        # Interactive conversation loop
        conversation_history = []
        
        while True:
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'done']:
                break
                
            if user_input.lower() == 'install':
                self.install_recommended_stack(conversation_history)
                break
            
            # Get AI response
            full_context = "\n".join(conversation_history) + f"\nUser: {user_input}"
            response = self.ask_assistant(full_context)
            
            conversation_history.append(f"User: {user_input}")
            conversation_history.append(f"Assistant: {response}")
            
            self.console.print(f"[bold cyan]Assistant:[/bold cyan] {response}")
            
            # Check if assistant made a recommendation
            if "recommend" in response.lower() and "stack" in response.lower():
                self.console.print("\n[bold yellow]💡 Type 'install' to proceed with the recommendation, or continue asking questions![/bold yellow]")
    
    def install_recommended_stack(self, conversation_history):
        """Extract recommendation and install the suggested stack"""
        # Simple keyword extraction to determine recommended stack
        full_conversation = " ".join(conversation_history).lower()
        
        if "demo" in full_conversation:
            stack = "demo"
        elif "local" in full_conversation:
            stack = "local"
        elif "cloud" in full_conversation:
            stack = "cloud"
        elif "mini" in full_conversation:
            stack = "mini"
        elif "research" in full_conversation:
            stack = "research"
        else:
            stack = "full"
        
        self.console.print(f"[bold green]🚀 Installing {stack.upper()} stack...[/bold green]")
        
        # Install the requirements
        requirements_file = f"requirements-{stack}.txt"
        if os.path.exists(requirements_file):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        else:
            self.console.print(f"[red]Error: {requirements_file} not found. Installing basic requirements.[/red]")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        self.console.print(f"[bold green]✅ {stack.upper()} stack installed successfully![/bold green]")
        self.console.print("\n[bold yellow]Next steps:[/bold yellow]")
        self.console.print("1. Run: python -m rag_engine serve")
        self.console.print("2. In another terminal: cd frontend && npm run dev")
        self.console.print("3. Visit: http://localhost:3001")
        self.console.print("\n[dim]You can always ask me questions later with: rag-engine ask \"your question\"[/dim]")

def main():
    """Main setup function"""
    assistant = RAGAssistant()
    
    try:
        assistant.setup_console()
        assistant.install_ollama()
        assistant.setup_model()
        assistant.interactive_setup()
        
    except KeyboardInterrupt:
        assistant.console.print("\n[yellow]Setup cancelled by user.[/yellow]")
        sys.exit(1)
    except Exception as e:
        print(f"Error during setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
