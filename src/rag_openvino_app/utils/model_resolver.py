# -*- coding: utf-8 -*-
"""
HF モデルID or ローカルパス → OpenVINO IR(.xml/.bin) への解決ユーティリティ（保存先を models/ に固定）

改良点:
- HF ID を指定された場合は必ず MODELS_DIR/<sanitized_repo_id>/ 配下に実体を保存
- OV_HF_CACHE_DIR はあくまでキャッシュ（symlink 不使用）
- allow_patterns で IR(.xml/.bin) と tokenizer 資材を確実に取得
- 取得後に .xml/.bin の存在を検証し、詳細ログを出力
"""
from __future__ import annotations
from pathlib import Path
import os
import glob
import re

from rag_openvino_app.utils.logger_utils import with_logger
from rag_openvino_app.constants.paths import MODELS_DIR, OV_HF_CACHE_DIR  # ★ 追加

try:
    from huggingface_hub import snapshot_download
    _HF_OK = True
except Exception:
    _HF_OK = False


def _find_first_xml(dir_path: Path) -> Path | None:
    # よくある IR 名を優先
    candidates: list[str] = [
        "openvino_model.xml",
        "ov_model.xml",
        "*model.xml",
        "*.xml",
    ]
    for pat in candidates:
        hits = sorted(dir_path.glob(pat))
        if hits:
            return hits[0]
    # 念のため深い階層も探索
    deep_hits = glob.glob(str(dir_path / "**" / "*.xml"), recursive=True)
    if deep_hits:
        return Path(sorted(deep_hits)[0])
    return None


def _sanitize_repo_id(repo_id: str) -> str:
    # "OpenVINO/Qwen3-4B-int8-ov" -> "OpenVINO__Qwen3-4B-int8-ov"
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", repo_id.strip())


@with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
def resolve_ir_path(model_id_or_path: str, *, logger=None) -> Path:
    """
    入力が HF リポジトリIDでもローカルパスでも受け付け、OpenVINO IR(.xml) の絶対パスを返す。
    - 直接 .xml/.onnx 指定 → そのまま返す
    - 既存ディレクトリ指定 → 配下から .xml を探索
    - HF リポID       指定 → MODELS_DIR/<repo_id_sanitized>/ に実体を保存してから .xml を返す
    """
    raw_path = Path(model_id_or_path)

    # 1) 直接ファイル指定（.xml / .onnx）
    if raw_path.suffix.lower() in {".xml", ".onnx"}:
        xml = raw_path.expanduser().resolve()
        if not xml.exists():
            raise FileNotFoundError(f"指定ファイルが見つかりません: {xml}")
        logger.info("ModelResolver: 直接ファイル指定 -> %s", xml)
        return xml

    # 2) 既存ディレクトリ指定
    if raw_path.exists() and raw_path.is_dir():
        xml = _find_first_xml(raw_path)
        if xml and xml.exists():
            logger.info("ModelResolver: 既存ディレクトリから IR を検出 -> %s", xml)
            return xml
        raise FileNotFoundError(f"ディレクトリ内に IR(.xml) が見つかりません: {raw_path}")

    # 3) HF リポジトリIDとして解釈
    if not _HF_OK:
        raise RuntimeError(
            "huggingface_hub が見つかりません。`uv add huggingface_hub` するか、"
            "transformers/optimum と一緒にインストールしてください。"
        )

    repo_id = model_id_or_path.strip()
    sanitized = _sanitize_repo_id(repo_id)
    target_dir = (MODELS_DIR / sanitized).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = os.getenv("OV_HF_CACHE_DIR", str(OV_HF_CACHE_DIR))
    logger.info(
        "ModelResolver: HF から取得します repo_id=%s → %s (cache=%s)",
        repo_id, target_dir, cache_dir
    )

    # IR と tokenizer 資材を取得（シンボリックリンクは使わない）
    local_dir = snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        cache_dir=str(cache_dir),
        resume_download=True,
        allow_patterns=[
            "*.xml", "*.bin",                         # OpenVINO IR
            "config.json", "generation_config.json",  # config
            "tokenizer.*", "vocab.*", "*.model",      # tokenizer 各種
            "special_tokens_map.json", "tokenizer_config.json",
        ],
    )
    local_dir = Path(local_dir)

    xml = _find_first_xml(local_dir)
    if not (xml and xml.exists()):
        raise FileNotFoundError(
            f"IR(.xml) を検出できませんでした: {local_dir}\n"
            f"モデルID: {repo_id}\n"
            "リポジトリ配下に .xml と .bin が存在するか確認してください。"
        )

    # .bin 検証（同名であることが多い）
    bin_path = xml.with_suffix(".bin")
    if not bin_path.exists():
        logger.warning("ModelResolver: IR の .bin が見つかりません: %s", bin_path)

    logger.info("ModelResolver: IR を検出しました -> %s", xml)
    return xml
