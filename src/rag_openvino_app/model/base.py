# -*- coding: utf-8 -*-
"""
Base model manager (抽象基底クラス)。

全てのモデルマネージャは以下のインターフェースを満たします:
- generate(prompt: str, **kwargs) -> str : 同期生成
将来的に stream() や batch_generate() を追加可能。

注意:
- 依存を薄く保つため、このモジュールは外部フレームワークに依存しません。
"""

from __future__ import annotations
from abc import ABC, abstractmethod


class BaseModelManager(ABC):
    """LLM への統一インターフェース。"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: object) -> str:
        """同期的にテキストを生成して返します。"""
        raise NotImplementedError

    # 将来の拡張例:
    # def stream(self, prompt: str, **kwargs: object) -> Iterable[str]: ...
    # def batch_generate(self, prompts: list[str], **kwargs) -> list[str]: ...
