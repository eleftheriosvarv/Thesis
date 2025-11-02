"""
Model registry (dynamic import version)
---------------------------------------
Keeps Prophet and LSTM as lightweight skeletons.
"""

from typing import Dict, Any, Callable

_REGISTRY: Dict[str, Callable[[Any, Dict[str, Any]], Dict[str, Any]]] = {}

def register(name: str, fn: Callable):
    """Manual register (if needed)."""
    _REGISTRY[name] = fn

def list_models():
    """List available registered models."""
    return list(_REGISTRY.keys())

def get_schema(name: str):
    """Get model schema dynamically."""
    mod = __import__(f"forecast_lab.models.{name}", fromlist=["*"])
    return getattr(mod, "get_schema")()

def run_model(name: str, data, cfg):
    """Dynamically import and execute model.run()"""
    try:
        mod = __import__(f"forecast_lab.models.{name}", fromlist=["*"])
    except Exception as e:
        raise KeyError(f"Unknown model '{name}'") from e
    return getattr(mod, "run")(data, cfg)

# No heavy imports here — Prophet and LSTM are loaded only when called.
