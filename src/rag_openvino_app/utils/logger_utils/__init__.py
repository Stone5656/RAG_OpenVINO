# ner_openvino/utils/logger_utils/__init__.py
from .logger_injector import with_logger

__all__ = ["with_logger"]

# 任意: パッケージ属性アクセスで with_logger 以外を弾く（PEP 562）
def __getattr__(name: str):
    if name in __all__:
        return globals()[name]
    raise AttributeError(
        f"'logger_utils' public API is only: {', '.join(__all__)}. "
        f"Got: {name!r}"
    )
