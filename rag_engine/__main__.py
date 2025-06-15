import typer
from rag_engine.core.pipeline import Pipeline
from rag_engine.config.loader import load_config

app = typer.Typer()

@app.command()
def init():
    """Scaffold a new RAG project."""
    typer.echo("Project initialized.")

@app.command()
def build(config: str):
    """Build vector DB from config file."""
    cfg = load_config(config)
    pipeline = Pipeline(cfg)
    pipeline.build()
    typer.echo("Vector DB built.")

@app.command()
def chat(config: str):
    """Chat with your data."""
    cfg = load_config(config)
    pipeline = Pipeline(cfg)
    pipeline.chat()

@app.command()
def serve(api: bool = False, ui: bool = False):
    """Serve API and/or UI."""
    if api:
        typer.echo("Starting FastAPI server...")
    if ui:
        typer.echo("Starting Streamlit/Gradio UI...")

if __name__ == "__main__":
    app()
