
from dataclasses import dataclass
from typing import Any, Callable, Dict

@dataclass
class ModelEntry:
    name: str
    run_fn: Callable[[Any, dict], dict]
    schema: dict

_REGISTRY: Dict[str, ModelEntry] = {}

def register_model(name: str, schema: dict):
    def _wrap(fn):
        if name in _REGISTRY:
            raise ValueError(f"Model '{name}' already registered")
        _REGISTRY[name] = ModelEntry(name=name, run_fn=fn, schema=schema)
        return fn
    return _wrap

def list_models():
    return sorted(_REGISTRY.keys())

def get_schema(name: str):
    return _REGISTRY[name].schema

def run_model(name: str, data, cfg: dict):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list_models()}")
    return _REGISTRY[name].run_fn(data, cfg)
