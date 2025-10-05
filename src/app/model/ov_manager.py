# -*- coding: utf-8 -*-
"""
OpenVINO (Optimum Intel) モデル管理モジュール

- 役割:
    * Hugging Face 上の OpenVINO IR モデル(ov) or Transformersモデルを OVModelForCausalLM でロード
    * GPU/CPU デバイス切替
- 参考:
    * OVModelForCausalLM の読み込みと export 指定, device 切替は Optimum Intel / OpenVINO 公式を踏襲
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer, pipeline, GenerationConfig
from huggingface_hub.utils import HFValidationError
import re
from pathlib import Path
from optimum.intel import OVModelForCausalLM


@dataclass
class OVConfig:
    device: str = "GPU"       # "GPU" or "CPU" など
    export: bool = False      # True にすると Transformers モデルからその場で IR 生成
    trust_remote_code: bool = True
    max_new_tokens: int = 512
    do_sample: bool = True
    local_files_only: bool = True
    temperature: float = 0.7
    top_p: float = 0.95


class OVLLM:
    """OpenVINO LLM + HF pipeline wrapper"""

    def __init__(self, model_id_or_path: str | Path, cfg: Optional[OVConfig] = None,
                 cache_dir: str | Path | None = None, persist_dir: str | Path | None = None):
        self.cfg = cfg or OVConfig()
        mid = str(model_id_or_path)

        # 1) OVModel 読み込み（Hub の ov モデル or ローカル IR）
        #    既に IR 済みモデルなら export=False、Transformers から直接なら export=True
        self.model = OVModelForCausalLM.from_pretrained(
            mid,
            export=self.cfg.export,
            device=self.cfg.device,
            # trust_remote_code=self.cfg.trust_remote_code,
            cache_dir=str(cache_dir) if cache_dir else None,
        )

        # 2) Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            mid, trust_remote_code=self.cfg.trust_remote_code, cache_dir=str(cache_dir) if cache_dir else None
        )
        # ★ GPT系の安全設定：左パディング＋pad_token の補完
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # 3) 生成設定をモデル側に明示（毎回の generate で参照される）
        gen_cfg = GenerationConfig.from_model_config(self.model.config)
        gen_cfg.max_new_tokens = int(self.cfg.max_new_tokens)
        gen_cfg.eos_token_id = self.tokenizer.eos_token_id
        gen_cfg.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config = gen_cfg

        # ★ サンプリング系の正規化（ここで一本化）
        wants_sampling = bool(self.cfg.do_sample) or any([
            self.cfg.temperature not in (None, 0.0),
            self.cfg.top_p not in (None, 1.0),
        ])
        do_sample = bool(wants_sampling)

        kw = dict(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            return_full_text=False,
        )
        if do_sample:
            # サンプリングする場合だけ注入
            kw["temperature"] = float(self.cfg.temperature)
            kw["top_p"] = float(self.cfg.top_p)

        self.pipe = pipeline(**kw)

        # ★ HFのrepo_idでロードした場合にローカルへ保存
        if persist_dir:
            p = Path(persist_dir)
            p.mkdir(parents=True, exist_ok=True)
            # OpenVINO IR 一式とトークナイザを保存
            self.model.save_pretrained(str(p))
            self.tokenizer.save_pretrained(str(p))

    def as_hf_pipeline(self):
        return self.pipe
