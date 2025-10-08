# -*- coding: utf-8 -*-
"""
models/ 配下の OpenVINO IR を探索し、UI で扱いやすい「ID だけ」の形に整形して返すユーティリティ。
- cache/.locks 等は探索しない
- MODELS_DIR ルートは除去
- 末尾の "openvino_model.xml" / ".xml" は削除
- HF 由来の "_hub/OpenVINO__Qwen… → OpenVINO/Qwen…" に復元
"""
from __future__ import annotations
from pathlib import Path
import re

from rag_openvino_app.constants.paths import MODELS_DIR
from rag_openvino_app.utils.logger_utils import with_logger

_EXCLUDE_PARTS = {"hf_cache", ".cache", ".locks"}

def _looks_excluded(path: Path) -> bool:
    """キャッシュ/ロック系のパスを除外判定"""
    for part in path.parts:
        if part in _EXCLUDE_PARTS or part.endswith(".locks"):
            return True
    return False

def _restore_repo_id_from_rel(parts: tuple[str, ...]) -> str:
    """
    HF 由来の階層から repo_id を復元する。
    - _hub/<sanitized>/models--<org>--<name>/snapshots/.../openvino_model.xml
    - _hub__<sanitized>__models--...（フラット）にも対応
    - <sanitized> は "__" → "/" に戻す
    - ローカル手置きは先頭ディレクトリ名を ID と見なす（必要に応じ変更可）
    """
    flat = "/".join(parts)
    m = re.match(r"^_hub__([^_][\s\S]+?)__models--", flat)
    if m:
        return m.group(1).replace("__", "/")
    if len(parts) >= 2 and parts[0] == "_hub":
        return parts[1].replace("__", "/")
    return parts[0].replace("\\", "/")

def _strip_suffixes(rel_path_str: str) -> str:
    """末尾の 'openvino_model.xml' / '.xml' を取り除く"""
    if rel_path_str.endswith("openvino_model.xml"):
        rel_path_str = rel_path_str[:-len("openvino_model.xml")].rstrip("/\\")
    elif rel_path_str.endswith(".xml"):
        rel_path_str = rel_path_str[:-len(".xml")].rstrip("/\\")
    return rel_path_str

@with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
def discover_model_ids(*, logger=None) -> list[str]:
    """
    models/ 配下の IR (.xml) を探索し、UI 用の「ID だけ」のリストを返す。
    戻り値例:
      - "_hub/OpenVINO__Qwen3-4B-int4-ov/.../openvino_model.xml" -> "OpenVINO/Qwen3-4B-int4-ov"
      - "my_llama/model.xml" -> "my_llama"
    """
    if not MODELS_DIR.exists():
        logger.info("モデル探索: MODELS_DIR が存在しません: %s", MODELS_DIR)
        return []

    ids: set[str] = set()
    for xml in MODELS_DIR.glob("**/*.xml"):
        if _looks_excluded(xml):
            continue
        rel = xml.relative_to(MODELS_DIR)
        parts = rel.parts
        repo_id = _restore_repo_id_from_rel(parts)
        repo_id = _strip_suffixes(repo_id).replace("\\", "/")
        if repo_id:
            ids.add(repo_id)

    out = sorted(ids)
    logger.debug("モデル探索: %d 件検出 -> %s", len(out), out)
    return out
