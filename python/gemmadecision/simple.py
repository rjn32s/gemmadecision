"""Lazy process-local default engine for the simplest Python entry points."""
import os
import threading

_engine = None
_lock = threading.Lock()


def get_engine():
    global _engine
    with _lock:
        if _engine is None:
            from .engine import DecisionEngine
            _engine = DecisionEngine.from_pretrained(
                model_path=os.getenv("GEMMADECISION_MODEL_PATH") or None,
            )
        return _engine
