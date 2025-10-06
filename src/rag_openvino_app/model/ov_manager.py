# -*- coding: utf-8 -*-
"""
OpenVINO モデルマネージャ。

責務:
- OpenVINO Runtime でモデルをロードして推論を実行
- generate() により同期生成を提供（スケルトン）
- 各ステップを DEBUG ログで可視化
"""
from __future__ import annotations
import time
from pathlib import Path

from rag_openvino_app.utils.logger_utils import with_logger
from rag_openvino_app.utils.model_resolver import resolve_ir_path
from .base import BaseModelManager

try:
    from openvino.runtime import Core  # type: ignore
    _OV_AVAILABLE = True
except Exception as e:
    _OV_AVAILABLE = False
    _OV_IMPORT_ERR = e


class OVManager(BaseModelManager):
    """OpenVINO による LLM 推論マネージャ（IR 自動解決版）。"""

    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def __init__(self, config: dict[str, object], *, logger=None):
        if not _OV_AVAILABLE:
            raise RuntimeError(f"OpenVINO が利用できません: {_OV_IMPORT_ERR}")

        # 1) 入力（ローカルパス or HF リポID）を IR(.xml) の絶対パスに解決
        raw_id = str(config.get("model_id", "")).strip()
        if not raw_id:
            raise ValueError("OVManager: model_id が空です。サイドバー『モデルID / IRパス』に IR の .xml か HF のリポジトリIDを指定してください。")

        xml_path: Path = resolve_ir_path(raw_id)
        # IR の場合は .bin も必要
        if xml_path.suffix.lower() == ".xml":
            bin_path = xml_path.with_suffix(".bin")
            if not bin_path.exists():
                raise FileNotFoundError(
                    f"IR の .bin が見つかりません: {bin_path}\n"
                    "ヒント: .xml と同じフォルダに .bin が必要です。"
                )

        device = str(config.get("device", "CPU"))

        self.config = {
            "model_id": str(xml_path),
            "device": device,
            "max_new_tokens": int(config.get("max_new_tokens", 512)),
            "temperature": float(config.get("temperature", 0.2)),
        }
        logger.debug("OVManager 初期化: %s", self.config)

        self.core = Core()
        # 2) 解決済みの .xml で compile
        self.compiled = self.core.compile_model(self.config["model_id"], self.config["device"])
        self.infer_req = self.compiled.create_infer_request()
        logger.debug("OVManager: モデルを %s にコンパイルしました。", self.config["device"])

    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def generate(self, prompt: str, *, logger=None, **kwargs: object) -> str:
        """OpenVINO で推論を実行（スケルトン）。"""
        t0 = time.time()
        max_new_tokens = int(kwargs.get("max_new_tokens", self.config["max_new_tokens"]))
        temperature = float(kwargs.get("temperature", self.config["temperature"]))

        logger.debug(
            "OVManager.generate: プロンプト長=%d 生成長=%d 温度=%.2f",
            len(prompt), max_new_tokens, temperature
        )

        # 実際は tokenizer.encode → 推論ステップ → decode（省略）
        output = f"[OpenVINO 出力] ({self.config['model_id']}) → {prompt[:60]}..."

        logger.debug("OVManager.generate: 完了 (処理時間 %.3f 秒)", time.time() - t0)
        return output
