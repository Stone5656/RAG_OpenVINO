# -*- coding: utf-8 -*-
"""
アプリ全体で使うパスやモデルIDの定数定義（.env で上書き可）
"""
from __future__ import annotations
import os
from pathlib import Path

# ルート推定（このファイルから2階層上 = src/ の親をルートとみなす）
ROOT = Path(__file__).resolve().parents[3]

# .env で上書きできるよう環境変数優先（なければデフォルト）
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models")).resolve()
PDF_DIR    = Path(os.getenv("PDF_DIR", ROOT / "uploaded_docs")).resolve()

# OpenVINO の IR を自動取得する場合のキャッシュ
OV_HF_CACHE_DIR = Path(os.getenv("OV_HF_CACHE_DIR", MODELS_DIR / "hf_cache")).resolve()

# 日本語対応の埋め込みモデル（Hugging Face）
# 例: intfloat/multilingual-e5-small は日/英など多言語で高コスパ
JA_EMBEDDING_MODEL_ID = os.getenv("JA_EMBEDDING_MODEL_ID", "intfloat/multilingual-e5-small")

# チャンク分割の既定（必要なら .env で調整）
CHUNK_SIZE   = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# ディレクトリは存在しなければ作成
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)
OV_HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / "indexes")).resolve()
VDB_BASENAME = os.getenv("VDB_BASENAME", "default_index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

