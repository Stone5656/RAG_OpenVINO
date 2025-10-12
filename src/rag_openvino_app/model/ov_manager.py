# -*- coding: utf-8 -*-
"""
OpenVINO モデルマネージャ（遅延初期化 & 安全化版）

変更点:
- model_id 未設定でも allow_deferred_init=True なら例外を投げず「未準備」状態で待機
- 初回 generate()/update_model_config() 時に一度だけロード（スレッド安全）
- UI 向けに get_status() を提供（状態/理由/推奨アクション）
- 例外を専用クラスに分離（UI 側で分岐しやすく）
"""
from __future__ import annotations
from pathlib import Path
import os
import time
import threading
from typing import Optional, Dict, Any

from rag_openvino_app.utils.logger_utils import with_logger
from rag_openvino_app.utils.model_resolver import resolve_ir_path
from .base import BaseModelManager

# ===== 依存の存在チェック =====
try:
    from optimum.intel.openvino import OVModelForCausalLM
    from transformers import AutoTokenizer
    _OPTIMUM_OK = True
    _OPTIMUM_ERR = None
except Exception as e:
    _OPTIMUM_OK = False
    _OPTIMUM_ERR = e

try:
    from openvino import Core  # なくても致命ではない
    _OV_AVAILABLE = True
    _OV_IMPORT_ERR = None
except Exception as e:
    _OV_AVAILABLE = False
    _OV_IMPORT_ERR = e


# ===== 専用の例外 =====
class OVModelConfigError(ValueError):
    """設定不備（例: model_id 未指定など）"""
    code = "E_OVM_CONFIG"

class OVModelNotReady(RuntimeError):
    """未初期化のため実行不能（UI から案内できる）"""
    code = "E_OVM_NOT_READY"

class OVModelLoadError(RuntimeError):
    """モデル/トークナイザのロード失敗"""
    code = "E_OVM_LOAD"


class OVManager(BaseModelManager):
    """
    OpenVINO CausalLM の実行器。
    - model_id: HF リポ ID もしくは IR(.xml) のパス
    - device: "AUTO:GPU,CPU" など
    - allow_deferred_init: True で model_id 未指定でも初期化を成功させ、「未準備」で待機
    """

    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def __init__(self, config: dict, *, logger=None):
        # 依存チェック
        if not _OPTIMUM_OK:
            raise OVModelLoadError(f"optimum/transformers が見つかりません: {_OPTIMUM_ERR}")
        if not _OV_AVAILABLE:
            raise OVModelLoadError(f"OpenVINO が利用できません: {_OV_IMPORT_ERR}")

        self._lock = threading.RLock()
        self._loaded = False
        self._last_error: Optional[str] = None

        # ユーザ設定
        self._allow_deferred = bool(config.get("allow_deferred_init", False))
        raw_id = str(config.get("model_id", "")).strip()
        device = str(config.get("device", "AUTO:GPU,CPU"))
        temperature = float(config.get("temperature", 0.2))
        max_new_tokens = int(config.get("max_new_tokens", 512))
        trust_remote_code = bool(config.get("trust_remote_code", True))
        ov_config: Dict[str, Any] = dict(config.get("ov_config", {}))  # 追加の OpenVINO 設定用

        # 保存（model/tokenizer は遅延ロード）
        self.config = {
            "model_id": raw_id,  # ★ ここではまだ生値
            "device": device,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "trust_remote_code": trust_remote_code,
            "ov_config": ov_config,
        }
        self.tokenizer = None
        self.model = None

        # 厳格モード: 初期化中に未設定なら即エラー
        if not raw_id and not self._allow_deferred:
            raise OVModelConfigError(
                "OVManager: model_id が空です。IR(.xml) か HF のモデルIDを指定してください。"
            )

        # 遅延モード: ここでは何もしない（UI で後から設定→ load）
        if raw_id:
            # eager でロードしたい場合はここで _ensure_loaded() を呼ぶ選択も可
            logger.debug("OVManager: 初期化（model_id=%s, deferred=%s）", raw_id, self._allow_deferred)
        else:
            logger.info("OVManager: model_id 未設定（deferred=%s）。UIから設定待ちの未準備状態。",
                        self._allow_deferred)

    # ====== 内部: ロード一回だけ実行 ======
    def _resolve_sources(self, raw_id: str, *, logger) -> tuple[str, str]:
        """
        resolve_ir_path(raw_id) -> .xml パスを前提に、model_source / tok_source を決定
        """
        try:
            resolved = resolve_ir_path(raw_id)
        except Exception as e:
            raise OVModelLoadError(f"IR/リポジトリ解決に失敗しました: {e}") from e

        trust_remote_code = bool(self.config.get("trust_remote_code", True))
        if resolved.suffix.lower() == ".xml":
            local_dir = resolved.parent
            hf_tok = os.getenv("TOKENIZER_ID")  # ローカルIRで tokenizer を別指定したい場合
            model_source = str(local_dir)
            tok_source = hf_tok or str(local_dir)
            logger.debug("OVManager: ローカル IR を使用します: %s", model_source)
            return model_source, tok_source
        else:
            # 保険: repo を直接使う経路も維持
            logger.debug("OVManager: HF リポジトリを使用します: %s", raw_id)
            return raw_id, raw_id

    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def _ensure_loaded(self, *, logger=None) -> None:
        """二重チェック+ロックで一度だけロード"""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return

            raw_id = str(self.config.get("model_id", "")).strip()
            if not raw_id:
                self._last_error = "model_id 未設定"
                raise OVModelNotReady(
                    "OVManager: model_id が未設定です。update_model_config({'model_id': ...}) で設定してください。"
                )

            model_source, tok_source = self._resolve_sources(raw_id, logger=logger)

            try:
                # Tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tok_source,
                    use_fast=True,
                    trust_remote_code=bool(self.config.get("trust_remote_code", True)),
                )
                if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                # Model
                self.model = OVModelForCausalLM.from_pretrained(
                    model_source,
                    device=self.config["device"],
                    use_cache=True,
                    trust_remote_code=bool(self.config.get("trust_remote_code", True)),
                    ov_config=self.config.get("ov_config") or None,  # 任意の最適化オプション
                )

                maxlen = int(getattr(self.tokenizer, "model_max_length", 1024))
                if maxlen > 10**6:
                    maxlen = 4096
                self.config["model_max_length"] = maxlen

                self._loaded = True
                self._last_error = None
                logger.info("OVManager: モデル/トークナイザをロードしました。device=%s", self.config["device"])
                logger.debug("OVManager: 設定 %s", self.config)
            except Exception as e:
                self._last_error = str(e)
                # 後で UI が案内しやすいように情報を詰めて返す
                raise OVModelLoadError(f"モデル/トークナイザのロードに失敗しました: {e}") from e

    # ====== 外部: 設定更新（UI から呼ぶ想定） ======
    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def update_model_config(self, new: Dict[str, Any], *, logger=None) -> None:
        """
        例: update_model_config({"model_id": "...", "device": "AUTO:GPU,CPU"})
        - model_id を変更したら、再ロード（差し替え）する
        """
        with self._lock:
            # 反映
            for k, v in (new or {}).items():
                self.config[k] = v
            # 既存のモデルを捨てて再ロード（明示）
            self._loaded = False
            self.model = None
            self.tokenizer = None
            logger.debug("OVManager: 設定更新 -> 再ロード準備 %s", self.config)

        # ここでロードしておく（失敗なら例外を返す → UIで専用復旧）
        self._ensure_loaded()

    # ====== 外部: 状態取得（UI でバッジ表示などに） ======
    def get_status(self) -> Dict[str, Any]:
        return {
            "loaded": self._loaded,
            "model_id": self.config.get("model_id"),
            "device": self.config.get("device"),
            "last_error": self._last_error,
            "allow_deferred_init": self._allow_deferred,
        }

    # ====== 推論 ======
    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def generate(self, prompt: str, *, logger=None, **kwargs) -> str:
        """
        実際に生成を行う（入力プロンプトのエコーは剥がして返す）
        """
        t0 = time.time()

        # 遅延ロード（必要ならここで一度だけ）
        self._ensure_loaded()

        # ---- 生成ハイパラ（kwargs優先 / 既定は config）----
        max_new_tokens = int(kwargs.get("max_new_tokens", self.config.get("max_new_tokens", 512)))
        temperature     = float(kwargs.get("temperature",     self.config.get("temperature", 0.2)))
        top_p           = float(kwargs.get("top_p",           self.config.get("top_p", 0.9)))
        top_k           = int(kwargs.get("top_k",             self.config.get("top_k", 50)))
        repetition_penalty   = float(kwargs.get("repetition_penalty",   self.config.get("repetition_penalty", 1.15)))
        no_repeat_ngram_size = int(kwargs.get("no_repeat_ngram_size",   self.config.get("no_repeat_ngram_size", 6)))
        do_sample = bool(kwargs.get("do_sample", temperature > 0.0))

        # ---- トークナイズ ----
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if "attention_mask" not in inputs:
            from torch import ones_like
            inputs["attention_mask"] = ones_like(inputs["input_ids"])

        input_len = int(inputs["input_ids"].shape[1])

        # ---- 生成 ----
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # ---- 「生成分のみ」をデコード ----
        try:
            gen_ids = outputs[:, input_len:]
        except Exception:
            seq = outputs.sequences if hasattr(outputs, "sequences") else outputs
            gen_ids = seq[:, input_len:]

        text = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

        # 先頭にプロンプトが混入するモデルへの保険
        def _norm(s: str) -> str:
            return " ".join(s.replace("\u3000", " ").split())
        try:
            if _norm(text).startswith(_norm(prompt)):
                text = text[len(prompt):].lstrip()
        except Exception:
            pass

        elapsed = time.time() - t0
        total_len = int(gen_ids.shape[1] + input_len) if hasattr(gen_ids, "shape") else None
        logger.debug(
            "OVManager.generate: 完了（%.3f 秒, max_new_tokens=%d, temp=%.2f, "
            "rep=%.2f, ngram=%d, in_tokens=%d, gen_tokens≈%s, total_tokens≈%s）",
            elapsed, max_new_tokens, temperature, repetition_penalty, no_repeat_ngram_size,
            input_len, getattr(gen_ids, 'shape', ['?','?'])[-1], total_len
        )
        return text
