# Core interfaces and base classes for RAG Engine

class BaseLoader:
    def load(self, config):
        raise NotImplementedError

class BaseChunker:
    def chunk(self, documents, config):
        raise NotImplementedError

class BaseEmbedder:
    def embed(self, chunks, config):
        raise NotImplementedError

class BaseVectorStore:
    def add(self, embeddings, config):
        raise NotImplementedError
    def query(self, query_embedding, config):
        raise NotImplementedError

class BaseRetriever:
    def retrieve(self, vectorstore, query, config):
        raise NotImplementedError

class BaseLLM:
    def generate(self, prompt, config):
        raise NotImplementedError

class BasePrompting:
    def format(self, context, config):
        raise NotImplementedError

class BasePipeline:
    def run(self, config):
        raise NotImplementedError
