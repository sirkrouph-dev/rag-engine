import json
import yaml
from .schema import RAGConfig

def load_config(path: str):
    if path.endswith('.json'):
        with open(path, 'r') as f:
            data = json.load(f)
    elif path.endswith('.yml') or path.endswith('.yaml'):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError('Unsupported config format')
    return RAGConfig(**data)
