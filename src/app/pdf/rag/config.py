# config.py
try:
    from pydantic import ConfigDict
    MODEL_CONFIG = {"model_config": ConfigDict(arbitrary_types_allowed=True)}
except Exception:
    MODEL_CONFIG = {"Config": type("Config", (), {"arbitrary_types_allowed": True})}
