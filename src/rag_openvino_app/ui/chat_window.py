# -*- coding: utf-8 -*-
"""
Streamlit チャット画面

責務:
- 会話履歴の表示
- クエリ入力欄と送信ボタン
- ConversationManager のパイプライン実行
- 進行中の状態（スピナー）と日本語ログ

設計:
- UI 層は ConversationManager に *依存するだけ*（中身は知らない）
- 履歴の保管は呼び出し元（SessionState）に委任
"""
from __future__ import annotations

import streamlit as st

from rag_openvino_app.utils.logger_utils import with_logger
from rag_openvino_app.conversation.manager import ConversationManager
from rag_openvino_app.conversation.session_state import SessionState


def _render_history(history: list[dict[str, object]]) -> None:
    """会話履歴を時系列に表示。"""
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        else:
            with st.chat_message("assistant"):
                st.markdown(content)


@with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
def render_chat(
    manager: ConversationManager,
    session: SessionState,
    *,
    temperature: float = 0.2,
    logger=None,
) -> None:
    """
    チャット UI を描画し、ユーザ入力を受け取ってパイプラインを実行する。

    Parameters
    ----------
    manager : ConversationManager
        RAG+LLM を実行する統合マネージャ
    session : SessionState
        会話履歴の保管先
    temperature : float
        生成の多様性
    """
    st.title("RAG OpenVINO チャット")
    _render_history(session.get_history())

    prompt = st.chat_input("質問を入力してください")
    if prompt is None:
        return
    prompt = prompt.strip()
    if not prompt:
        return

    logger.debug("UI: ユーザ入力を受信しました。 '%s'", prompt[:80])
    session.add_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("検索と推論を実行中…"):
            try:
                result = manager.run_pipeline(prompt, temperature=temperature)
                answer = (result or {}).get("answer", "") or "（応答なし：生成テキストが空でした）"
                elapsed = float((result or {}).get("elapsed", 0.0))

                # 応答を確実に描画
                st.markdown(answer)
                # 処理時間の表示
                st.caption(f"⏱ {elapsed:.2f}s")

                # 参照コンテキストを確認できるように
                ctxs = (result or {}).get("contexts", [])
                if ctxs:
                    with st.expander("参照コンテキスト（RAG）"):
                        for i, c in enumerate(ctxs, 1):
                            st.markdown(f"**[{i}]** {c.get('meta', {}).get('source', '')}")
                            st.write(c.get("text", ""))
                session.add_message("assistant", answer)
                logger.debug("UI: 応答を履歴に追加しました。")
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
                logger.exception("UI: 実行中にエラーが発生しました。")
