# -*- coding: utf-8 -*-
"""
テキスト分割器

責務:
- 長文テキストをチャンクへ分割し、埋め込み・検索に適した粒度へ整形
- まずは文字数ベースのシンプル分割に対応（必要に応じて文/段落単位へ拡張）

API:
- split_text(text, chunk_size=1000, overlap=150, method="char")
  - "char": 文字数ベース。トークン見積もりを行わない軽量モード。

注意:
- RAG の品質は「適切な分割」に強く依存します。まずは単純な分割で開始し、必要なら文区切り・見出し分割・トークン見積もりに差し替えてください。
"""
from __future__ import annotations
import re
from rag_openvino_app.utils.logger_utils import with_logger


def _split_by_chars(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        return [text]
    out: list[str] = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        out.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return out


def _merge_short_sentences(sentences: list[str], chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    buf: list[str] = []
    cur = 0
    for s in sentences:
        if cur + len(s) <= chunk_size or not buf:
            buf.append(s)
            cur += len(s)
        else:
            chunks.append("".join(buf))
            joined = "".join(buf)
            keep = joined[max(0, len(joined) - overlap):]
            buf = [keep, s]
            cur = len(keep) + len(s)
    if buf:
        chunks.append("".join(buf))
    return chunks


@with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
def split_text(
    text: str,
    *,
    chunk_size: int = 1000,
    overlap: int = 150,
    method: str = "char",
    logger=None,
) -> list[str]:
    """
    テキストをチャンクに分割する。

    Returns
    -------
    list[str] : チャンク列（順序保持）
    """
    text = text or ""
    if not text.strip():
        logger.debug("Splitter: 空テキストのため分割をスキップしました。")
        return []

    logger.debug("Splitter: 分割開始（方法=%s, サイズ=%d, オーバーラップ=%d）", method, chunk_size, overlap)

    if method == "char":
        chunks = _split_by_chars(text, chunk_size, overlap)
    elif method == "sentence":
        sents = re.split(r"(?<=[。！？\!\?])\s+|\n+", text)
        chunks = _merge_short_sentences(sents, chunk_size, overlap)
    else:
        logger.debug("Splitter: 未知の method=%s → char にフォールバックします。", method)
        chunks = _split_by_chars(text, chunk_size, overlap)

    total_chars = sum(len(c) for c in chunks)
    logger.debug("Splitter: 分割完了（%d チャンク, 総文字数 %d）", len(chunks), total_chars)
    return chunks

