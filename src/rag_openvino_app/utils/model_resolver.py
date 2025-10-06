# -*- coding: utf-8 -*-
"""
HF モデルID or ローカルパス → OpenVINO IR(.xml/.bin) への解決ユーティリティ
- 入力が .xml / .onnx / 実在ディレクトリ ならそのまま使う
- それ以外は Hugging Face Hub から snapshot_download してローカルに展開
- 展開先から .xml（IR）を探索し、最適な1件を返す
- LOGPATH/LOG_LEVEL は with_logger が処理

環境変数:
- OV_HF_CACHE_DIR: ダウンロード先（未指定なら HF 既定キャッシュ or ./models/cache 相当を使用）
"""

from __future__ import annotations
from pathlib import Path
import os
import glob

from rag_openvino_app.utils.logger_utils import with_logger

try:
    from huggingface_hub import snapshot_download  # transformers 由来の hub が入っていれば OK
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
        hits = sorted(Path(dir_path).glob(pat))
        if hits:
            return hits[0]
    # 念のため深い階層も探索
    deep_hits = glob.glob(str(dir_path / "**" / "*.xml"), recursive=True)
    if deep_hits:
        return Path(sorted(deep_hits)[0])
    return None


@with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
def resolve_ir_path(model_id_or_path: str, *, logger=None) -> Path:
    """
    入力が HuggingFace のリポジトリID でも、ローカルパスでも受け付け、
    OpenVINO IR (.xml) の絶対パスを返す。

    戻り値:
        Path: 例) /abs/path/to/.../openvino_model.xml

    例外:
        FileNotFoundError: 解決できない
        RuntimeError: HF からの取得が必要だが huggingface_hub が未インストール
    """
    raw = Path(model_id_or_path)
    # 1) 直接 xml/onnx が指定された場合
    if raw.suffix.lower() in {".xml", ".onnx"}:
        xml = raw.expanduser().resolve()
        if not xml.exists():
            raise FileNotFoundError(f"指定ファイルが見つかりません: {xml}")
        logger.debug("ModelResolver: 直接ファイル指定を検出 -> %s", xml)
        return xml

    # 2) 既存ディレクトリが指定された場合 → 中の .xml を探す
    if raw.exists() and raw.is_dir():
        xml = _find_first_xml(raw)
        if xml and xml.exists():
            logger.debug("ModelResolver: 既存ディレクトリから IR を検出 -> %s", xml)
            return xml
        raise FileNotFoundError(f"ディレクトリ内に .xml が見つかりません: {raw}")

    # 3) HF リポジトリIDとして解釈（例: OpenVINO/open_llama_3b_v2-int8-ov）
    if not _HF_OK:
        raise RuntimeError(
            "huggingface_hub が見つかりません。`uv add huggingface_hub` するか、"
            "transformers/optimum と一緒にインストールしてください。"
        )

    cache_dir = os.getenv("OV_HF_CACHE_DIR")  # 任意: 明示キャッシュ先
    logger.debug("ModelResolver: HF からダウンロード開始 (repo_id=%s, cache_dir=%s)", model_id_or_path, cache_dir or "<HF default>")
    local_dir = snapshot_download(repo_id=model_id_or_path, local_dir=cache_dir)  # login が必要な場合は事前に `huggingface-cli login`
    local_dir = Path(local_dir)

    xml = _find_first_xml(local_dir)
    if xml and xml.exists():
        logger.debug("ModelResolver: ダウンロード後に IR を検出 -> %s", xml)
        return xml

    raise FileNotFoundError(
        f"HuggingFace から取得しましたが IR(.xml) を見つけられませんでした: {local_dir}\n"
        f"モデルID: {model_id_or_path}\n"
        "リポジトリ直下または配下に .xml と .bin があることを確認してください。"
    )
