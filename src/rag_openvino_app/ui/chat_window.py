# -*- coding: utf-8 -*-
"""
Streamlit チャット画面（例外を止め、Chatウィンドウ内で復旧UIを表示）
"""
import streamlit as st
from rag_openvino_app.utils.logger_utils import with_logger
from rag_openvino_app.conversation.manager import ConversationManager
from rag_openvino_app.conversation.session_state import SessionState

def _dedupe_text(text: str, max_repeat: int = 1) -> str:
    lines = [l.strip() for l in text.splitlines()]
    out, rep, prev = [], 0, None
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

def _render_fix_in_chat(manager: ConversationManager) -> bool:
    """
    Chatウィンドウ内にモデル設定フォームを描画し、
    保存が成功したら True を返す（同フレーム再試行のトリガに使う）
    """
    saved = False
    with st.chat_message("assistant"):
        st.error(
            "OpenVINO の **model_id** が未設定です。\n\n"
            "下のフォームで **Model ID / IR(.xml) パス** を入力して保存してください。"
        )
        with st.container(border=True):
            default_id = st.session_state.get("ov_model_id", "")
            new_id = st.text_input("Model ID / IR(.xml) パス", value=default_id, key="model_id_input")
            col1, col2 = st.columns(2)
            with col1:
                do_save = st.button("保存", use_container_width=True)
            with col2:
                do_save_and_retry = st.button("保存して再試行", use_container_width=True)

            if do_save or do_save_and_retry:
                try:
                    st.session_state.ov_model_id = new_id.strip()
                    manager.update_model_config({"model_id": st.session_state.ov_model_id})
                    st.success("モデル設定を保存しました。")
                    saved = True if do_save_and_retry else False
                except Exception as e:
                    st.error(f"設定の反映に失敗しました: {e}")
    return saved  # True のときは同フレーム内での再試行を呼び出し側で実施

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

    if manager is None:
        with st.chat_message("assistant"):
            st.info("モデルが未選択です。サイドバーの『モデル設定』で Model ID を選択してください。")
        return

    # 2) 実行状態フラグ
    infer_running = st.session_state.get("infer_running", False)

    # 3) 入力受付
    prompt = st.chat_input("質問を入力してください")
    if not prompt:
        return

    if infer_running:
        logger.debug("UI: 推論中のため入力をスキップしました。")
        return

    # 4) ここから1ターン同期処理
    st.session_state.infer_running = True
    success = False           # 応答確定できたら True（成功時のみ rerun）
    retried_in_frame = False  # 例外復旧UIからの「保存して再試行」で使う

    try:
        prompt = prompt.strip()
        if not prompt:
            return

        logger.debug("UI: ユーザ入力を受信 '%s'", prompt[:80])
        session.add_message("user", prompt)

        with st.spinner("検索と推論を実行中…"):
            result = manager.run_pipeline(prompt, temperature=temperature)

        answer = (result or {}).get("answer", "") or "（応答なし：生成テキストが空でした）"
        elapsed = float((result or {}).get("elapsed", 0.0))
        answer = _dedupe_text(answer, max_repeat=0)

        session.add_message("assistant", answer)
        logger.debug("UI: 応答を履歴に追加（%.2fs）", elapsed)
        success = True

    except Exception as e:
        msg = str(e)
        logger.exception("UI: 実行中にエラーが発生しました。")
        # 共通：履歴にも残す（調査用）
        session.add_message("assistant", f"エラーが発生しました: {msg}")

        # --- 専用ハンドリング：model_id 未設定 ---
        if "OVManager: model_id が空です" in msg:
            # Chatウィンドウ内で設定フォームを出す
            want_retry = _render_fix_in_chat(manager)
            if want_retry:
                retried_in_frame = True
                # そのまま同フレームで再試行（spinner 付き）
                try:
                    with st.spinner("設定を反映し、再実行中…"):
                        result = manager.run_pipeline(prompt, temperature=temperature)
                    answer = (result or {}).get("answer", "") or "（応答なし：生成テキストが空でした）"
                    elapsed = float((result or {}).get("elapsed", 0.0))
                    answer = _dedupe_text(answer, max_repeat=0)
                    session.add_message("assistant", answer)
                    logger.debug("UI: 再試行で応答を履歴に追加（%.2fs）", elapsed)
                    success = True
                except Exception as e2:
                    session.add_message("assistant", f"再試行でもエラーが発生しました: {e2}")
                    logger.exception("UI: 再試行中にエラーが発生しました。")
            else:
                # 保存のみ or 何もしない → このフレームでは止める（rerun しない）
                pass
        else:
            # その他の例外はチャットに例外詳細を表示（開発時）
            with st.chat_message("assistant"):
                st.exception(e)

    finally:
        st.session_state.infer_running = False
        # 成功時のみ rerun（履歴描画反映のため）
        # → 例外時は rerun しないので「止められない」問題を回避
        if success:
            st.rerun()
        else:
            if retried_in_frame:
                with st.chat_message("assistant"):
                    st.info("設定を保存しました。必要ならもう一度メッセージを送ってください。")
            return
