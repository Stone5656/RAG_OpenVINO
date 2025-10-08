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
import streamlit as st
from rag_openvino_app.utils.logger_utils import with_logger
from rag_openvino_app.conversation.manager import ConversationManager
from rag_openvino_app.conversation.session_state import SessionState

# --- NEW: 重複文の簡易除去（同一行/段落の連続繰り返しを抑止） ---
def _dedupe_text(text: str, max_repeat: int = 1) -> str:
    lines = [l.strip() for l in text.splitlines()]
    out = []
    rep = 0
    prev = None
    for l in lines:
        if l and l == prev:
            rep += 1
            if rep <= max_repeat:
                out.append(l)
        else:
            rep = 0
            out.append(l)
        prev = l
    return "\n".join(out).strip()


def _render_history(history: list[dict]) -> None:
    for m in history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])


@with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
def render_chat(
    manager: ConversationManager,
    session: SessionState,
    *,
    temperature: float = 0.2,
    logger=None,
) -> None:
    st.title("RAG OpenVINO チャット")

    # 1) 先に履歴だけ描画
    _render_history(session.get_history())

    # 2) いま推論中かどうかのフラグ
    if "infer_running" not in st.session_state:
        st.session_state.infer_running = False

    # 3) 入力受付
    prompt = st.chat_input("質問を入力してください")
    if not prompt:
        return

    if st.session_state.infer_running:
        # 直前のリクエスト処理中。多重実行を防止
        logger.debug("UI: 推論中のため入力をスキップしました。")
        return

    # 4) ここから1ターン分を同期的に処理（同フレームでは描画しない）
    try:
        st.session_state.infer_running = True
        prompt = prompt.strip()
        if not prompt:
            st.session_state.infer_running = False
            return

        logger.debug("UI: ユーザ入力を受信 '%s'", prompt[:80])

        # 履歴にユーザ発話を先に入れる
        session.add_message("user", prompt)

        # 推論
        with st.spinner("検索と推論を実行中…"):
            result = manager.run_pipeline(prompt, temperature=temperature)

        answer = (result or {}).get("answer", "") or "（応答なし：生成テキストが空でした）"
        elapsed = float((result or {}).get("elapsed", 0.0))
        # NEW: 応答の簡易デデュープ
        answer = _dedupe_text(answer, max_repeat=0)

        # 履歴に応答を入れる（このターンでは画面に直接描画しない）
        session.add_message("assistant", answer)
        logger.debug("UI: 応答を履歴に追加（%.2fs）", elapsed)

    except Exception as e:
        # エラーも履歴に1件入れておくと再現調査しやすい
        err = f"エラーが発生しました: {e}"
        session.add_message("assistant", err)
        logger.exception("UI: 実行中にエラーが発生しました。")
    finally:
        st.session_state.infer_running = False
        # 5) 履歴に入れ終えたら rerun して、次フレームで「履歴だけ」描画
        st.rerun()
