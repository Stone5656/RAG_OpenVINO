# -*- coding: utf-8 -*-
"""
RAG パイプラインの集中設定。

- 本ファイルは「コードに散らばりがちな定数」を 1 箇所に集約するためのものです。
- 実運用では、環境変数や .env / YAML から読み込む層を追加しても良いです。

主な設定:
- 検索件数や再ランク後の上位件数
- MMR の λ（関連性 vs 多様性のバランス）
- 冗長除去や圧縮の閾値
- タイムアウト、ログ詳細度など
"""

# === Retrieval (初段) ===
TOP_K: int = 12            # 類似検索の候補件数
USE_MMR: bool = True       # 取得直後に MMR を適用するか
MMR_LAMBDA: float = 0.4    # 0(多様性重視)〜1(関連性重視)

# === Rerank (二段) ===
RERANK_TOP_N: int = 6      # 最終的に使う件数 (k -> N)
RERANKER_NAME: str | None = "BAAI/bge-reranker-base"  # None なら簡易スコア

# === Compression (三段/任意) ===
REDUNDANT_SIM_THRESHOLD: float = 0.92  # 近接重複の類似度閾値（高いほど厳しめ）
MAX_CONTEXT_CHARS: int = 8500          # LLM に渡す総コンテキスト文字上限（例）
SNIPPET_MARGIN_CHARS: int = 64         # 抽出の余白

# === 実装向けユーティリティ ===
DEFAULT_TIMEOUT_SECS: float = 30.0
