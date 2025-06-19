import json
import yaml
import os
import re
from .schema import RAGConfig

def substitute_env_vars(data):
    """Recursively substitute environment variables in config data."""
    if isinstance(data, dict):
        return {k: substitute_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [substitute_env_vars(item) for item in data]
    elif isinstance(data, str):
        # Replace ${VAR_NAME} with environment variable values
        def replace_env_var(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))  # Keep original if not found
        
        return re.sub(r'\$\{([^}]+)\}', replace_env_var, data)
    else:
        return data

def load_config(path: str):
    if path.endswith('.json'):
        with open(path, 'r') as f:
            data = json.load(f)
    elif path.endswith('.yml') or path.endswith('.yaml'):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError('Unsupported config format')
    
    # Substitute environment variables
    data = substitute_env_vars(data)
    
    return RAGConfig(**data)
