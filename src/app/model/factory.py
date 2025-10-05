# -*- coding: utf-8 -*-
"""
モデル・ファクトリ

- 目的:
    * Windows のローカルパス/相対パス/HF モデルIDのいずれでも安全に判定
    * .gguf は LlamaCpp、OpenVINO IR は OVModelForCausalLM、HFのOVモデルIDも可
- 方針:
    * pathlib.Path で存在確認・拡張子判定
    * IR は `openvino_model.xml` の存在で検出
    * 誤って .gguf を OV に渡さない（= HF repo_id 誤解を防止）
- 依存:
    * langchain_community.llms.LlamaCpp
    * transformers.pipeline
    * optimum.intel.openvino.OVModelForCausalLM
    * langchain_community.llms.HuggingFacePipeline
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, GenerationConfig
from optimum.intel.openvino import OVModelForCausalLM
from app.model.ov_manager import OVLLM, OVConfig


@dataclass
class LlamaCppConfig:
    n_ctx: int = 2048
    n_gpu_layers: int = 0          # Intel GPU では基本 0
    n_batch: int = 512
    f16_kv: bool = True
    verbose: bool = False


# ------------------------ ユーティリティ ------------------------
def _is_local(path_or_id: str | Path) -> bool:
    try:
        return Path(str(path_or_id)).exists()
    except Exception:
        return False


def _looks_like_hf_id(s: str) -> bool:
    # "owner/repo" 形式で、Windowsドライブ記号やバックスラッシュを含まない
    return ("/" in s) and ("\\" not in s) and (":" not in s)


def _detect_kind(p: Path) -> Literal["gguf", "ov_dir", "unknown"]:
    if p.is_file() and p.suffix.lower() == ".gguf":
        return "gguf"
    if p.is_dir() and (p / "openvino_model.xml").exists():
        return "ov_dir"
    # _hub/.../openvino_model.xml などを見つけたい場合は再帰でもOK
    if p.is_dir():
        cand = list(p.rglob("openvino_model.xml"))
        if cand:
            return "ov_dir"
    return "unknown"

# --------------------------- 本体 ---------------------------
def load_model(
    model_type: Literal["auto", "ov", "llamacpp"],
    model_id_or_path: str | Path,
    **kwargs: Any,
):
    """
    model_type:
        - "auto"     : 中身を見て自動判定
        - "ov"       : OpenVINO 優先（.gguf を渡したら自動で LlamaCpp に切替）
        - "llamacpp" : GGUF を明示
    kwargs:
        - OV 用: device, max_new_tokens, do_sample, temperature, top_p, export, local_files_only
        - Llama 用: n_ctx, n_gpu_layers, n_batch, f16_kv, verbose
    """
    # --- ローカル or HF ID を判定 ---
    ref = str(model_id_or_path)
    is_local = _is_local(model_id_or_path)
    path = Path(ref) if is_local else None

    # --- 自動判定（明示指定を上書きしない） ---
    prefer = model_type
    if model_type == "auto":
        if is_local:
            kind = _detect_kind(path)  # type: ignore[arg-type]
            prefer = "llamacpp" if kind == "gguf" else ("ov" if kind == "ov_dir" else "ov")
        else:
            # HF ID の場合：OpenVINO org の IR を前提に "ov" を既定
            prefer = "ov"

    # --- LlamaCpp (.gguf) ---
    if prefer == "llamacpp" or (is_local and _detect_kind(path) == "gguf"):  # type: ignore[arg-type]
        if not is_local:
            raise ValueError("llamacpp を使う場合は .gguf のローカルパスを指定してください。")
        cfg = LlamaCppConfig(
            n_ctx=int(kwargs.get("n_ctx", 2048)),
            n_gpu_layers=int(kwargs.get("n_gpu_layers", 0)),
            n_batch=int(kwargs.get("n_batch", 512)),
            f16_kv=bool(kwargs.get("f16_kv", True)),
            verbose=bool(kwargs.get("verbose", False)),
        )
        return LlamaCpp(
            model_path=str(path),
            n_ctx=cfg.n_ctx,
            n_gpu_layers=cfg.n_gpu_layers,
            n_batch=cfg.n_batch,
            f16_kv=cfg.f16_kv,
            verbose=cfg.verbose,
        )

    # --- OpenVINO (IR ディレクトリ または HF の OV モデルID) ---
    if prefer == "ov":
        cfg = OVConfig(
            device=str(kwargs.get("device", "AUTO:GPU,CPU")),
            max_new_tokens=int(kwargs.get("max_new_tokens", 256)),
            do_sample=bool(kwargs.get("do_sample", False)),
            temperature=float(kwargs.get("temperature", 0.7)),
            top_p=float(kwargs.get("top_p", 0.95)),
            export=bool(kwargs.get("export", False)),
            local_files_only=bool(kwargs.get("local_files_only", True)),
        )

        # 1) ローカル IR ディレクトリ（openvino_model.xml がある）
        if is_local and _detect_kind(path) == "ov_dir":  # type: ignore[arg-type]
            ov = OVModelForCausalLM.from_pretrained(
                str(path),
                device=cfg.device,
            )
        else:
            # 2) HF の OpenVINO モデルID（owner/repo）
            if not _looks_like_hf_id(ref):
                # ここで repo_id 検証に行かないように防御
                raise ValueError(
                    f"OV モデルに渡された参照が不正です: {ref}\n"
                    "・ローカルIRなら openvino_model.xml を含む**フォルダ**を指定\n"
                    "・HFのOVモデルなら 'owner/repo' 形式のモデルIDを指定"
                )
            ov = OVModelForCausalLM.from_pretrained(
                ref,
                device=cfg.device,
                export=cfg.export,           # 既にIRなら False、変換したいなら True
            )

        # HF pipeline を組み、LangChain ラッパで返す
        tok = AutoTokenizer.from_pretrained(ov.config.name_or_path, trust_remote_code=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token or tok.unk_token
        tok.padding_side = "left"

        gen = GenerationConfig.from_model_config(ov.config)
        gen.max_new_tokens = cfg.max_new_tokens
        gen.eos_token_id = tok.eos_token_id
        gen.pad_token_id = tok.pad_token_id
        ov.generation_config = gen

        ovllm = OVLLM(ref if not is_local else str(path), cfg=OVConfig(
            device=cfg.device,
            export=cfg.export,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
        ))
        return HuggingFacePipeline(pipeline=ovllm.as_hf_pipeline())

    raise ValueError(f"未対応のモデルタイプ: {model_type}")
