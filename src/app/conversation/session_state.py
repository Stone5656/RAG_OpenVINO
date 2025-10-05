# -*- coding: utf-8 -*-
"""
会話用セッション状態ユーティリティ

- 役割:
    * Streamlit の session_state を初期化・取得・操作する関数群
- 設計:
    * 直接 st.session_state に触れる箇所を集約して、UI/ロジックの結合度を下げます。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, TypedDict, Literal

import streamlit as st


Role = Literal["user", "assistant", "system"]

class ChatMessage(TypedDict):
    role: Role
    content: str


@dataclass
class ConversationState:
    selected_model: str = ""
    selected_pdfs: list[str] = field(default_factory=list)
    messages: list[ChatMessage] = field(default_factory=list)
    # ここでは型名のみ（LLM 依存を避ける）。実体はアプリ側で注入。
    qa_chain: object | None = None


def ensure_defaults() -> None:
    """必要なキーが無ければデフォルト値を設定する。"""
    st.session_state.setdefault("selected_model", "")
    st.session_state.setdefault("selected_pdfs", [])
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("qa_chain", None)


def snapshot() -> ConversationState:
    """現在の session_state から ConversationState を生成。"""
    return ConversationState(
        selected_model=st.session_state.get("selected_model", ""),
        selected_pdfs=list(st.session_state.get("selected_pdfs", [])),
        messages=list(st.session_state.get("messages", [])),
        qa_chain=st.session_state.get("qa_chain"),
    )


def reset_for_new_conversation() -> None:
    """
    新規会話開始時のリセット。
    - 既存実装の「新しい会話を開始」操作に対応（messages/qa_chain/選択状態のリセット）。:contentReference[oaicite:4]{index=4}
    """
    st.session_state["messages"] = []
    st.session_state["qa_chain"] = None
    st.session_state["selected_model"] = ""
    st.session_state["selected_pdfs"] = []
