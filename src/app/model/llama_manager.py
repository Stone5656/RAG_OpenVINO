# -*- coding: utf-8 -*-
"""
LlamaCpp モデル管理モジュール

- 役割:
    * モデルのロード・GPU設定の統一
    * パス検証とエラーハンドリング
    * モデルごとの初期化パラメータ管理
- 根拠:
    * 既存 app.py では LlamaCpp を直に生成し、GPU設定やcontext長などを定義していた（:contentReference[oaicite:1]{index=1}）。
      これを共通化・抽象化して再利用可能にする。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from langchain_community.llms import LlamaCpp


@dataclass
class LlamaConfig:
    """LlamaCpp モデル初期化設定"""
    n_gpu_layers: int = -1
    n_batch: int = 512
    n_ctx: int = 4096
    f16_kv: bool = True
    verbose: bool = False


def load_llama_model(model_path: Path, config: Optional[LlamaConfig] = None) -> LlamaCpp:
    """
    LlamaCpp モデルをロードする。

    Raises:
        FileNotFoundError: モデルファイルが存在しない場合。
        RuntimeError: LlamaCpp 初期化時のエラー。
    """
    if not model_path.exists():
        raise FileNotFoundError(f"モデルファイルが存在しません: {model_path}")

    cfg = config or LlamaConfig()
    try:
        llm = LlamaCpp(
            model_path=str(model_path),
            n_gpu_layers=cfg.n_gpu_layers,
            n_batch=cfg.n_batch,
            n_ctx=cfg.n_ctx,
            f16_kv=cfg.f16_kv,
            verbose=cfg.verbose,
        )
        return llm
    except Exception as e:
        raise RuntimeError(f"LlamaCpp モデル初期化に失敗しました: {e}") from e
