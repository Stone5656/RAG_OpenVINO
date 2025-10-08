# -*- coding: utf-8 -*-
"""
OpenVINO モデルマネージャ。

責務:
- OpenVINO Runtime でモデルをロードして推論を実行
- generate() により同期生成を提供（スケルトン）
- 各ステップを DEBUG ログで可視化
"""
from __future__ import annotations
from pathlib import Path
import os
import time

from rag_openvino_app.utils.logger_utils import with_logger
from rag_openvino_app.utils.model_resolver import resolve_ir_path
from .base import BaseModelManager

try:
    from optimum.intel.openvino import OVModelForCausalLM   # 実生成はこちらで
    from transformers import AutoTokenizer
    _OPTIMUM_OK = True
except Exception as e:
    _OPTIMUM_OK = False
    _OPTIMUM_ERR = e

try:
    from openvino.runtime import Core  # XML存在チェック等に使えるが、必須ではない
    _OV_AVAILABLE = True
except Exception as e:
    _OV_AVAILABLE = False
    _OV_IMPORT_ERR = e


class OVManager(BaseModelManager):
    """
    OpenVINO CausalLM の実行器。
    - model_id: HF リポ ID でも OK（例: OpenVINO/Qwen3-4B-int8-ov）
                ローカルの .xml でも OK（resolve_ir_path が親ディレクトリを返す）
    - device: "AUTO:GPU,CPU" など
    """

    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def __init__(self, config: dict, *, logger=None):
        if not _OPTIMUM_OK:
            raise RuntimeError(f"optimum/transformers が見つかりません: {_OPTIMUM_ERR}")
        if not _OV_AVAILABLE:
            raise RuntimeError(f"OpenVINO が利用できません: {_OV_IMPORT_ERR}")

        raw_id = str(config.get("model_id", "")).strip()
        if not raw_id:
            raise ValueError("OVManager: model_id が空です。IR(.xml) か HF のモデルIDを指定してください。")

        device = str(config.get("device", "AUTO:GPU,CPU"))
        temperature = float(config.get("temperature", 0.2))
        max_new_tokens = int(config.get("max_new_tokens", 512))

        # 1) 入力を解決
        #    - HF repo: そのまま from_pretrained が可能
        #    - ローカル .xml: 親ディレクトリを from_pretrained に渡す（tokenizer は TOKENIZER_ID から取る）
        resolved = resolve_ir_path(raw_id)  # 返り値は .xml の絶対パス or HF から落としたローカルパス配下の .xml
        if resolved.suffix.lower() == ".xml":
            local_dir = resolved.parent
            hf_repo_for_tokenizer: str | None = os.getenv("TOKENIZER_ID")  # ローカルIR用に別途指定できるように
            model_source = str(local_dir)
            tok_source = hf_repo_for_tokenizer or str(local_dir)
            logger.debug("OVManager: ローカル IR を使用します: %s", model_source)
        else:
            # resolve_ir_path は原則 .xml を返す仕様だが、保険で repo をそのまま使えるように
            model_source = raw_id
            tok_source = raw_id
            logger.debug("OVManager: HF リポジトリを使用します: %s", model_source)

        self.config = {
            "model_source": model_source,
            "tok_source": tok_source,
            "device": device,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }

        # 2) Tokenizer / OVModel をロード
        #    - trust_remote_code は Qwen 系などで必要になることがある
        self.tokenizer = AutoTokenizer.from_pretrained(
            tok_source,
            use_fast=True,
            trust_remote_code=True,
        )
        # pad_token 未設定モデル対策（gpt2等）
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # device_map は OpenVINO では compile 時に決まるため不要。
        self.model = OVModelForCausalLM.from_pretrained(
            model_source,
            device=device,
            use_cache=True,
            trust_remote_code=True,
        )

        maxlen = int(getattr(self.tokenizer, "model_max_length", 1024))
        if maxlen > 10**6:
            maxlen = 4096
        logger.info("OVManager: モデルの最大入力長（context window）=%d tokens", maxlen)
        self.config["model_max_length"] = maxlen

        logger.info("OVManager: モデル/トークナイザをロードしました。device=%s", device)
        logger.debug("OVManager: 設定 %s", self.config)

    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def generate(self, prompt: str, *, logger=None, **kwargs) -> str:
        """
        実際に生成を行う。
        kwargs: temperature, max_new_tokens, top_p, top_k, repetition_penalty, no_repeat_ngram_size など任意
        重要: 出力は「生成分のみ」（= 入力プロンプトのエコーを除去）を返す。
        """
        import time
        t0 = time.time()

        # ---- 生成ハイパラ（kwargs優先 / 既定は config）----
        max_new_tokens = int(kwargs.get("max_new_tokens", self.config.get("max_new_tokens", 512)))
        temperature     = float(kwargs.get("temperature",     self.config.get("temperature", 0.2)))
        top_p           = float(kwargs.get("top_p",           self.config.get("top_p", 0.9)))
        top_k           = int(kwargs.get("top_k",             self.config.get("top_k", 50)))
        repetition_penalty   = float(kwargs.get("repetition_penalty",   self.config.get("repetition_penalty", 1.15)))
        no_repeat_ngram_size = int(kwargs.get("no_repeat_ngram_size",   self.config.get("no_repeat_ngram_size", 6)))
        do_sample = bool(kwargs.get("do_sample", temperature > 0.0))

        # ---- トークナイズ（注意: 入力長を後で使う）----
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if "attention_mask" not in inputs:
            from torch import ones_like
            inputs["attention_mask"] = ones_like(inputs["input_ids"])

        input_len = int(inputs["input_ids"].shape[1])

        # ---- 生成（入力+生成の連結トークン列が返る想定）----
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
            # return_dict_in_generate=False の場合、Tensor で返る（既定）
        )

        # ---- 「生成分のみ」を取り出してデコード ----
        # outputs: [batch=1, input_len + gen_len]
        try:
            gen_ids = outputs[:, input_len:]              # ★ ここが肝（入力ぶんをスライス）
        except Exception:
            # 万一 generate の戻りが辞書の場合にも対応
            seq = outputs.sequences if hasattr(outputs, "sequences") else outputs
            gen_ids = seq[:, input_len:]
        text = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

        # ---- 念のための保険（正規化して前方一致でプロンプトを剥がす）----
        # 生成器によっては tokenization の都合で数トークンだけ先頭に残ることがあるため
        def _norm(s: str) -> str:
            return " ".join(s.replace("\u3000", " ").split())
        try:
            if _norm(text).startswith(_norm(prompt)):
                text = text[len(prompt):].lstrip()
        except Exception:
            pass

        elapsed = time.time() - t0
        # 追加のデバッグ情報（入力長・生成長・総長）
        total_len = int(gen_ids.shape[1] + input_len) if hasattr(gen_ids, "shape") else None
        logger.debug(
            "OVManager.generate: 生成完了（%.3f 秒, max_new_tokens=%d, temp=%.2f, "
            "rep=%.2f, ngram=%d, in_tokens=%d, gen_tokens≈%s, total_tokens≈%s）",
            elapsed, max_new_tokens, temperature, repetition_penalty, no_repeat_ngram_size,
            input_len, getattr(gen_ids, 'shape', ['?','?'])[-1], total_len
        )
        return text
