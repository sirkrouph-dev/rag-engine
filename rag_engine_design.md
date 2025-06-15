# RAG Engine — Modular Retrieval-Augmented Generation Framework

## 🚀 Vision

**RAG Engine** is a plug-n-play, deeply configurable framework to build Retrieval-Augmented Generation (RAG) pipelines using configuration-as-code. It allows developers to customize and orchestrate every step of the RAG stack — from data loading to prompting — with multiple interfaces (CLI, API, UI).

---

## 🧩 Design Principles

| Principle           | Description                                                                     |
| ------------------- | ------------------------------------------------------------------------------- |
| **Modularity**      | Each pipeline stage (loader, chunker, embedder, vectorstore, etc.) is pluggable |
| **Config-as-Code**  | Full YAML-based configuration with Pydantic schema validation                   |
| **Multi-Interface** | CLI, FastAPI (REST), and Streamlit/Gradio interface support                     |
| **Extensibility**   | User-defined modules and plugin system                                          |
| **Ease of Use**     | Minimal CLI commands and smart defaults                                         |

---

## 📦 Project Structure

```plaintext
rag-engine/
├── rag_engine/
│   ├── core/                # Core logic and base interfaces
│   │   ├── base.py          # Abstract classes/interfaces
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── vectorstore.py
│   │   ├── retriever.py
│   │   ├── llm.py
│   │   ├── prompting.py
│   │   ├── pipeline.py
│   │   └── registry.py      # Dynamic component loader
│   ├── plugins/             # Optional user-defined modules
│   ├── interfaces/
│   │   ├── cli.py           # Typer-based CLI
│   │   ├── api.py           # FastAPI REST server
│   │   └── ui.py            # Streamlit or Gradio interface
│   ├── config/
│   │   ├── schema.py        # Pydantic config models
│   │   └── loader.py        # Config parser and validator
│   └── __main__.py          # CLI entry point
├── configs/
│   └── example_config.yml
├── examples/
│   └── quickstart.md
├── tests/
├── README.md
└── pyproject.toml / setup.py
```

---

## 🔧 Configuration (YAML Example)

```yaml
documents:
  - type: pdf
    path: ./docs/tech_guide.pdf

chunking:
  method: recursive
  max_tokens: 512
  overlap: 50

embedding:
  model: openai
  api_key: ${OPENAI_API_KEY}

vectorstore:
  provider: chroma
  persist_directory: ./vector_store

retrieval:
  top_k: 4

prompting:
  system_prompt: >
    You are a technical assistant. Answer clearly and concisely.

llm:
  provider: openai
  model: gpt-4
  temperature: 0.3

output:
  method: console
```

---

## 🧠 Core Components

- `Loader`: PDF, Markdown, Web, Notion, etc.
- `Chunker`: Sentence, recursive, fixed-size
- `Embedder`: OpenAI, HuggingFace, InstructorXL
- `Vectorstore`: Chroma, FAISS, Pinecone, Weaviate
- `Retriever`: Top-K, MMR, filters
- `Prompting`: Template engine (Jinja2, etc.)
- `LLM`: OpenAI, Anthropic, Local LLMs (Ollama)
- `Pipeline`: Glue logic
- `Evaluation`: Response quality grading (optional)

---

## 🧪 CLI Commands

```bash
# Scaffold new RAG project
rag-engine init

# Build vector DB from config
rag-engine build --config configs/example_config.yml

# Chat with your data
rag-engine chat --config configs/example_config.yml

# Serve API and/or UI
rag-engine serve --api
rag-engine serve --ui
```

---

## 🧑‍💻 Stretch Goals (Future Enhancements)

- ✅ Versioned pipelines
- ✅ Plugin support for user-defined chunkers/retrievers
- ✅ LLM cost estimation
- ✅ CI test runner for prompts
- ✅ Eval mode using GPT-based graders
- ✅ Docker image for full stack
- ✅ Web dashboard for monitoring retrieval + chat

---

## 🛠 Tech Stack

| Layer         | Tool                                   |
| ------------- | -------------------------------------- |
| CLI           | [`Typer`](https://typer.tiangolo.com/) |
| Config        | `YAML` + `Pydantic`                    |
| LLM/Embedding | OpenAI, Claude, Ollama                 |
| Vector DB     | Chroma, FAISS, Pinecone                |
| API           | FastAPI                                |
| UI (optional) | Streamlit or Gradio                    |
| Docs          | Markdown / Sphinx / MkDocs             |

---

## ✅ Next Steps

1. Setup virtual environment & install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Run CLI to scaffold example:

   ```bash
   rag-engine init
   rag-engine build --config configs/example_config.yml
   ```

3. Start experimenting:

   ```bash
   rag-engine chat
   ```

---

## 🧪 Example Prompt Engineering Flow

1. Write prompt templates in `prompts/`
2. Reference them in YAML config
3. Evaluate with sample questions via `rag-engine evaluate`
4. Log results to CSV or visual dashboard

---

## 🤝 Contribute

- All modules follow an interface pattern.
- To add a new `embedder`, simply implement the `BaseEmbedder` class in `plugins/`.
- Contributions welcome via PRs or plugins!

---

## 📜 License

MIT (you can change this)

