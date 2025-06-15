import os
from typing import List, Dict, Any
from rag_engine.core.base import BaseLoader

class TxtLoader(BaseLoader):
    def load(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        path = config["path"]
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return [{"type": "txt", "content": text, "path": path}]

class PdfLoader(BaseLoader):
    def load(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        import PyPDF2
        path = config["path"]
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return [{"type": "pdf", "content": text, "path": path}]

class DocxLoader(BaseLoader):
    def load(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        import docx
        path = config["path"]
        doc = docx.Document(path)
        text = "\n".join([p.text for p in doc.paragraphs])
        return [{"type": "docx", "content": text, "path": path}]

class HtmlLoader(BaseLoader):
    def load(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        from bs4 import BeautifulSoup
        path = config["path"]
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
        soup = BeautifulSoup(html, "html.parser")
        # Semantic chunking by elements (e.g., h1, h2, p, li)
        elements = []
        for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]):
            text = tag.get_text(strip=True)
            if text:
                elements.append({"type": tag.name, "content": text, "path": path})
        return elements

# Loader registry for dynamic selection
LOADER_REGISTRY = {
    "txt": TxtLoader(),
    "pdf": PdfLoader(),
    "docx": DocxLoader(),
    "html": HtmlLoader(),
}

def load_documents(doc_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    docs = []
    for doc_cfg in doc_configs:
        doc_type = doc_cfg["type"].lower()
        loader = LOADER_REGISTRY.get(doc_type)
        if not loader:
            raise ValueError(f"No loader for type: {doc_type}")
        docs.extend(loader.load(doc_cfg))
    return docs
