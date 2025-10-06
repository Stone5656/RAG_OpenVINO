# -*- coding: utf-8 -*-
"""
Streamlit サイドバー

責務:
- モデルと RAG の主要パラメータを入力
- 変更値を即時に返す（呼び出し側で manager 再生成などを行う）
- 日本語 DEBUG ログ

注意:
- ここでは値の妥当性チェックを最小限に留め、単純な UI を提供
"""
from __future__ import annotations
import streamlit as st
from rag_openvino_app.utils.logger_utils import with_logger


@with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
def render_sidebar(
    *,
    default_model_type: str = "ov",
    default_device: str = "CPU",
    default_temperature: float = 0.2,
    default_max_new_tokens: int = 512,
    default_top_k: int = 12,
    default_top_n: int = 6,
    default_mmr_lambda: float = 0.4,
    logger=None,
) -> tuple[dict[str, object], dict[str, object]]:
    """
    サイドバーを描画し、モデル・RAG 設定を返す。

    Returns
    -------
    model_cfg : dict
        get_model_manager に渡す設定
    rag_cfg : dict
        retriever/reranker/compressor 構築時に利用する設定
    """
    with st.sidebar:
        st.header("設定")

        st.subheader("モデル設定")
        model_type = st.selectbox("バックエンド", ["ov", "llama", "auto"], index=["ov", "llama", "auto"].index(default_model_type))
        device = st.selectbox("デバイス", ["CPU", "GPU"], index=["CPU", "GPU"].index(default_device))
        model_id = st.text_input("モデルID / IRパス", value="" if model_type != "llama" else "meta-llama/Llama-3-8b-instruct")
        temperature = st.slider("温度", min_value=0.0, max_value=1.0, value=default_temperature, step=0.05)
        max_new_tokens = st.number_input("生成最大トークン", min_value=1, max_value=4096, value=default_max_new_tokens, step=32)

        st.subheader("RAG 設定")
        top_k = st.number_input("Top-k（初段検索）", min_value=1, max_value=100, value=default_top_k, step=1)
        top_n = st.number_input("上位N（再ランク後）", min_value=1, max_value=top_k, value=default_top_n, step=1)
        mmr_lambda = st.slider("MMR λ（関連性↔多様性）", min_value=0.0, max_value=1.0, value=default_mmr_lambda, step=0.05)

        st.caption("※ 変更は即時反映。必要に応じて manager を再構築してください。")

    model_cfg = {
        "type": model_type,
        "device": device,
        "model_id": model_id,
        "temperature": float(temperature),
        "max_new_tokens": int(max_new_tokens),
    }
    rag_cfg = {
        "top_k": int(top_k),
        "top_n": int(top_n),
        "mmr_lambda": float(mmr_lambda),
    }

    logger.debug("Sidebar: model_cfg=%s", model_cfg)
    logger.debug("Sidebar: rag_cfg=%s", rag_cfg)
    return model_cfg, rag_cfg
