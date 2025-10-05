# -*- coding: utf-8 -*-
"""
会話管理モジュール（保存・読込）

- 役割:
    * 会話（モデル名、選択PDF、メッセージ列）の保存/読込を行う。
- 設計方針:
    * 既存の app.py 内の save/load を pathlib ベースに移植しつつ、UI と分離する。
    * UI はファイル名/パス等の引数を渡し、結果を session_state に反映する責務を最小限にする。
- 根拠:
    * 既存の保存・読込 UI と処理フロー（保存 UI, save_conversation, load_conversation）:contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6}
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import json
from typing import Iterable

import streamlit as st

from .session_state import ensure_defaults, snapshot


@dataclass
class ConversationPayload:
    model: str
    pdfs: list[str]
    messages: list[dict]


def save_conversation(conversations_dir: Path, filename_prefix: str | None = None) -> Path | None:
    """
    現在の会話を JSON として保存する。

    既存仕様の要点：
    - qa_chain が無い場合は保存しない（警告を出す）:contentReference[oaicite:7]{index=7}
    - メッセージが無い場合も保存しない（警告）:contentReference[oaicite:8]{index=8}
    - 保存構造: {"model", "pdfs", "messages"} を書き出す :contentReference[oaicite:9]{index=9}
    - ファイル名は prefix か timestamp ベース :contentReference[oaicite:10]{index=10}
    """
    ensure_defaults()
    if not st.session_state.get("qa_chain"):
        st.warning("保存する会話がありません。PDFを読み込んで会話を開始してください。")
        return None
    if not st.session_state.get("messages"):
        st.warning("メッセージがありません。")
        return None

    payload = ConversationPayload(
        model=st.session_state["selected_model"],
        pdfs=list(st.session_state["selected_pdfs"]),
        messages=list(st.session_state["messages"]),
    )

    if filename_prefix:
        filename = f"{filename_prefix}.json"
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{ts}.json"

    conversations_dir.mkdir(parents=True, exist_ok=True)
    outpath = conversations_dir / filename
    outpath.write_text(json.dumps(payload.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
    st.success(f"会話を `{filename}` に保存しました。")
    return outpath


def iter_conversations(conversations_dir: Path) -> list[Path]:
    """保存済み会話の JSON ファイル一覧を新しい順で返す（UI のセレクタ用）。"""
    conversations_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(conversations_dir.glob("*.json"), key=lambda p: p.name, reverse=True)
    return files


def load_conversation(conversations_dir: Path, selected_file: Path) -> None:
    """
    会話 JSON を読み込んで session_state を復元する。

    既存仕様の要点：
    - JSON から model/pdf/messages を復元し、RAG などのリソースは別途再ロードする流れ（load_resources を呼んだ後、messages を復元）:contentReference[oaicite:11]{index=11}
    - ここではファイルの読み込みと session_state への一時反映のみに留める（RAG 再構築はアプリ本体側コールバックで実施）
    """
    ensure_defaults()
    path = selected_file if selected_file.is_absolute() else (conversations_dir / selected_file)
    data = json.loads(path.read_text(encoding="utf-8"))

    # 一時格納：モデル名と PDF 名は UI/本体側で load_resources に渡して再構築
    st.session_state["selected_model"] = data.get("model", "")
    st.session_state["selected_pdfs"] = list(data.get("pdfs", []))
    st.session_state["messages"] = list(data.get("messages", []))
