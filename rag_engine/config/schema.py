from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class DocumentConfig(BaseModel):
    type: str
    path: str

class ChunkingConfig(BaseModel):
    method: str
    max_tokens: int
    overlap: int

class EmbeddingConfig(BaseModel):
    provider: str = "openai"
    model: str
    api_key: Optional[str] = None

class VectorStoreConfig(BaseModel):
    provider: str
    persist_directory: str

class RetrievalConfig(BaseModel):
    top_k: int

class PromptingConfig(BaseModel):
    system_prompt: str

class LLMConfig(BaseModel):
    provider: str
    model: str
    temperature: float

class OutputConfig(BaseModel):
    method: str

class RAGConfig(BaseModel):
    documents: List[DocumentConfig]
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    vectorstore: VectorStoreConfig
    retrieval: RetrievalConfig
    prompting: PromptingConfig
    llm: LLMConfig
    output: OutputConfig
