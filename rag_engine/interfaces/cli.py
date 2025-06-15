import typer

app = typer.Typer(help="RAG Engine CLI")

@app.command()
def build(config: str = typer.Option(None, '--config', '-c', help='Path to config file', required=True)):
    """Build vector DB from config."""
    typer.echo(f"Building vector DB with config: {config}")
    # TODO: Add build logic here

@app.command()
def chat(config: str = typer.Option(None, '--config', '-c', help='Path to config file', required=True)):
    """Chat with your data using the specified config."""
    typer.echo(f"Starting chat with config: {config}")
    # TODO: Add chat logic here

@app.command()
def init():
    """Scaffold a new RAG project."""
    typer.echo("Initializing new RAG project...")
    # TODO: Add init logic here

@app.command()
def serve(api: bool = typer.Option(False, '--api', help='Serve API'),
          ui: bool = typer.Option(False, '--ui', help='Serve UI')):
    """Serve API and/or UI."""
    if api:
        typer.echo("Serving API...")
        # TODO: Add API server logic
    if ui:
        typer.echo("Serving UI...")
        # TODO: Add UI server logic
    if not api and not ui:
        typer.echo("Specify --api and/or --ui to serve.")
