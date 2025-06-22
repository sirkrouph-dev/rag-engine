# Document Loaders

Document loaders are responsible for reading and parsing various file formats into a standardized text format that can be processed by the RAG pipeline.

## Available Loaders

### TextLoader
Loads plain text files (.txt, .md, .rst).

**Configuration:**
```json
{
  "loader": {
    "type": "text",
    "config": {
      "encoding": "utf-8",
      "chunk_overlap": 0
    }
  }
}
```

**Supported Formats:**
- `.txt` - Plain text files
- `.md` - Markdown files
- `.rst` - reStructuredText files

### PDFLoader
Extracts text from PDF documents using PyPDF2 or pdfplumber.

**Configuration:**
```json
{
  "loader": {
    "type": "pdf",
    "config": {
      "use_pdfplumber": true,
      "extract_images": false,
      "preserve_layout": true
    }
  }
}
```

**Features:**
- Text extraction with layout preservation
- Optional image extraction
- Multiple PDF parsing backends
- Metadata extraction (title, author, creation date)

### DocxLoader
Loads Microsoft Word documents (.docx).

**Configuration:**
```json
{
  "loader": {
    "type": "docx",
    "config": {
      "extract_tables": true,
      "extract_images": false,
      "preserve_formatting": false
    }
  }
}
```

**Features:**
- Text and table extraction
- Header and footer processing
- Style and formatting preservation options

### HTMLLoader
Parses HTML documents and web pages.

**Configuration:**
```json
{
  "loader": {
    "type": "html",
    "config": {
      "remove_scripts": true,
      "remove_styles": true,
      "extract_links": false,
      "parse_tables": true
    }
  }
}
```

**Features:**
- Clean HTML text extraction
- Table parsing
- Link extraction
- Script and style removal

## Usage Examples

### Basic Usage
```python
from rag_engine.core.loader import TextLoader

loader = TextLoader({"encoding": "utf-8"})
documents = loader.load("path/to/documents/")
```

### Batch Loading
```python
from rag_engine.core.orchestration import ComponentRegistry

registry = ComponentRegistry()
loader = registry.get_component("loader", "pdf", config)
documents = loader.load_batch([
    "doc1.pdf",
    "doc2.pdf",
    "doc3.pdf"
])
```

### With Metadata
```python
documents = loader.load("document.pdf")
for doc in documents:
    print(f"Title: {doc.metadata.get('title', 'Unknown')}")
    print(f"Content: {doc.content[:100]}...")
```

## Creating Custom Loaders

To create a custom loader, inherit from `BaseLoader`:

```python
from rag_engine.core.base import BaseLoader
from typing import List, Dict, Any

class CustomLoader(BaseLoader):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Initialize your loader
    
    def load(self, source: str) -> List[Dict[str, Any]]:
        """Load documents from source."""
        documents = []
        # Implement loading logic
        return documents
    
    def load_batch(self, sources: List[str]) -> List[Dict[str, Any]]:
        """Load multiple documents."""
        all_documents = []
        for source in sources:
            all_documents.extend(self.load(source))
        return all_documents

# Register the loader
from rag_engine.core.component_registry import ComponentRegistry
ComponentRegistry.register_component("loader", "custom", CustomLoader)
```

## Configuration Options

### Common Options
All loaders support these common configuration options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `encoding` | str | "utf-8" | Text encoding |
| `max_file_size` | int | 100MB | Maximum file size to process |
| `ignore_errors` | bool | false | Continue on file errors |
| `metadata_fields` | List[str] | ["title", "source"] | Metadata to extract |

### Error Handling
```json
{
  "loader": {
    "type": "pdf",
    "config": {
      "ignore_errors": true,
      "error_handling": "skip",
      "fallback_loader": "text"
    }
  }
}
```

## Performance Considerations

### Memory Usage
- Large files are processed in chunks
- Use streaming for very large documents
- Configure `max_file_size` appropriately

### Parallel Processing
```python
loader_config = {
    "parallel": true,
    "max_workers": 4,
    "batch_size": 10
}
```

### Caching
```python
loader_config = {
    "cache_enabled": true,
    "cache_directory": "./cache/documents",
    "cache_ttl": 3600  # 1 hour
}
```

## Troubleshooting

### Common Issues

**1. Encoding Errors**
```json
{
  "loader": {
    "config": {
      "encoding": "utf-8-sig",
      "ignore_encoding_errors": true
    }
  }
}
```

**2. Large File Handling**
```json
{
  "loader": {
    "config": {
      "streaming": true,
      "chunk_size": 1024,
      "max_memory_usage": "1GB"
    }
  }
}
```

**3. PDF Parsing Issues**
```json
{
  "loader": {
    "type": "pdf",
    "config": {
      "fallback_parser": "pdfminer",
      "ocr_enabled": true,
      "ocr_language": "eng"
    }
  }
}
```

## Dependencies

Loaders require specific dependencies based on format:

```bash
# PDF support
pip install pypdf2 pdfplumber

# DOCX support  
pip install python-docx

# HTML support
pip install beautifulsoup4 lxml

# OCR support (optional)
pip install pytesseract
```
