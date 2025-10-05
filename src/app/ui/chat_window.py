# -*- coding: utf-8 -*-
"""
チャット画面 UI モジュール

- 役割:
    * チャット履歴の描画
    * ユーザー入力（チャット欄）
    * アシスタント出力のプレースホルダ（ストリーミング表示を想定）
- 設計方針:
    * UI はステートフルに見せるが、外部ステート（session_state など）との結合は最小化
    * 生成テキストの逐次表示用にコールバック/ハンドラを注入可能な形を用意
- 依存:
    * Streamlit（st）
    * pathlib.Path（os 非依存）
- 型注釈:
    * `from __future__ import annotations` を採用
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Literal, TypedDict

import streamlit as st


Role = Literal["user", "assistant", "system"]


class ChatMessage(TypedDict):
    role: Role
    content: str


@dataclass
class StreamlitTokenSink:
    """
    逐次生成テキストを描画するための簡易ハンドラ。
    - `write(token)` を何度も呼ぶと、1つのメッセージとして更新し続けます。
    - 完了時に `finalize()` を呼んでカーソル風の記号を消します。
    """
    _placeholder: st.delta_generator.DeltaGenerator
    _buffer: str = ""

    def write(self, token: str) -> None:
        self._buffer += token
        self._placeholder.markdown(self._buffer + "▌")

    def finalize(self) -> None:
        self._placeholder.markdown(self._buffer)


def render_header(model_name: str, pdf_names: Iterable[str]) -> None:
    """画面上部に現在ステータス（モデル名・参照PDF）を表示する。"""
    pdf_label = "`, `".join(pdf_names)
    st.markdown(f"**🧠 モデル:** `{model_name}` | **📚 参照PDF:** `{pdf_label}`")
    st.divider()


def render_chat_history(messages: Iterable[ChatMessage]) -> None:
    """既存のチャット履歴を吹き出しで描画する。"""
    for m in messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])


def render_user_input(prompt_label: str, on_submit: Callable[[str, StreamlitTokenSink], None]) -> None:
    """
    チャット入力欄を表示し、送信時に on_submit(prompt, sink) を呼ぶ。
    - on_submit は LLM 呼び出しとストリーミング表示を担当
    """
    if prompt := st.chat_input(prompt_label):
        # User bubble
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant bubble (streaming sink)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            sink = StreamlitTokenSink(placeholder)
            with st.spinner("🤖 回答を生成中..."):
                on_submit(prompt, sink)
            sink.finalize()
