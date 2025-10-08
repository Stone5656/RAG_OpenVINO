# -*- coding: utf-8 -*-
"""
Streamlit サイドバー
- セクション単位（モデル/RAG/インデックス）で関数分割
- 探索ロジックは utils.model_discovery へ分離
"""
from __future__ import annotations
import streamlit as st
from typing import Dict, Tuple

from rag_openvino_app.utils.logger_utils import with_logger
from rag_openvino_app.utils.model_discovery import discover_model_ids
from rag_openvino_app.constants.paths import INDEX_DIR, VDB_BASENAME


# ---------------------------
# セクション: モデル設定
# ---------------------------
@with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
def render_model_section(
    *,
    key_prefix: str,
    default_model_type: str = "ov",
    default_device: str = "AUTO:GPU,CPU",
    default_temperature: float = 0.2,
    default_max_new_tokens: int = 1024,
    logger=None,
) -> Dict[str, object]:
    st.subheader("モデル設定")

    model_type = st.selectbox(
        "バックエンド",
        ["ov", "llama", "auto"],
        index=["ov", "llama", "auto"].index(default_model_type),
        key=f"{key_prefix}_model_type",
    )

    device = st.text_input(
        "デバイス（例: AUTO:GPU,CPU / CPU / GPU）",
        value=default_device,
        key=f"{key_prefix}_device",
    )

    # models/ の IR から ID を抽出
    discovered = discover_model_ids()
    model_choices = ["（models/ から選択）"] + discovered
    model_sel = st.selectbox(
        "モデルID / IR パス（models/ 探索結果）",
        options=model_choices,
        index=0,
        key=f"{key_prefix}_model_from_dir",
    )

    # 手入力（プリセットは出さない）
    custom_model_id = st.text_input(
        "モデルID / IR パス（手入力・空なら上の選択を使用）",
        value="",
        key=f"{key_prefix}_model_custom",
    )
    model_id = custom_model_id.strip() or ("" if model_sel == "（models/ から選択）" else model_sel)

    temperature = st.slider(
        "温度", 0.0, 1.0, default_temperature, 0.05, key=f"{key_prefix}_temp"
    )
    max_new_tokens = st.number_input(
        "生成最大トークン", 1, 8192, default_max_new_tokens, 32, key=f"{key_prefix}_max_tokens"
    )

    if not discovered:
        st.info("models/ 配下に IR(.xml) が見つかりません。HF のモデルIDを上の手入力欄に入れると自動取得します。")

    cfg = {
        "type": model_type,
        "device": device,
        "model_id": model_id,
        "temperature": float(temperature),
        "max_new_tokens": int(max_new_tokens),
    }
    logger.debug("Sidebar(Model)=%s", cfg)
    return cfg


# ---------------------------
# セクション: RAG 設定
# ---------------------------
@with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
def render_rag_section(
    *,
    key_prefix: str,
    default_top_k: int = 12,
    default_top_n: int = 6,
    default_mmr_lambda: float = 0.4,
    logger=None,
) -> Dict[str, object]:
    st.subheader("RAG 設定")

    top_k = st.number_input("Top-k（初段検索）", 1, 100, default_top_k, 1, key=f"{key_prefix}_topk")
    top_n = st.number_input("上位N（再ランク後）", 1, int(top_k), default_top_n, 1, key=f"{key_prefix}_topn")
    mmr_lambda = st.slider("MMR λ（関連性↔多様性）", 0.0, 1.0, default_mmr_lambda, 0.05, key=f"{key_prefix}_mmr")

    cfg = {
        "top_k": int(top_k),
        "top_n": int(top_n),
        "mmr_lambda": float(mmr_lambda),
    }
    logger.debug("Sidebar(RAG)=%s", cfg)
    return cfg


# ---------------------------
# セクション: ベクトルインデックス設定
# ---------------------------
@with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
def render_index_section(
    *,
    key_prefix: str,
    default_index_dir: str = str(INDEX_DIR),
    default_index_name: str = VDB_BASENAME,
    logger=None,
) -> Dict[str, object]:
    st.subheader("ベクトルインデックス")

    index_dir = st.text_input("保存ディレクトリ", value=default_index_dir, key=f"{key_prefix}_index_dir")
    index_name = st.text_input("ベース名（拡張子不要）", value=default_index_name, key=f"{key_prefix}_index_name")

    cfg = {
        "index_dir": index_dir,
        "index_name": index_name,
    }
    logger.debug("Sidebar(Index)=%s", cfg)
    return cfg


# ---------------------------
# エントリーポイント: サイドバー全体
# ---------------------------
@with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
def render_sidebar(
    *,
    key_prefix: str,  # ← 呼び出し側で必ずユニークに（例: "main_sidebar"）
    default_model_type: str = "ov",
    default_device: str = "AUTO:GPU,CPU",
    default_temperature: float = 0.2,
    default_max_new_tokens: int = 1024,
    default_top_k: int = 12,
    default_top_n: int = 6,
    default_mmr_lambda: float = 0.4,
    logger=None,
) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, object]]:
    with st.sidebar:
        st.header("設定")

        model_cfg = render_model_section(
            key_prefix=f"{key_prefix}_model",
            default_model_type=default_model_type,
            default_device=default_device,
            default_temperature=default_temperature,
            default_max_new_tokens=default_max_new_tokens,
        )

        rag_cfg = render_rag_section(
            key_prefix=f'{key_prefix}_rag',
            default_top_k=default_top_k,
            default_top_n=default_top_n,
            default_mmr_lambda=default_mmr_lambda,
        )

        index_cfg = render_index_section(
            key_prefix=f'{key_prefix}_index',
        )

        st.caption("※ 変更は即時反映。必要に応じて manager を再構築してください。")

    logger.debug("Sidebar: model=%s, rag=%s, index=%s", model_cfg, rag_cfg, index_cfg)
    return model_cfg, rag_cfg, index_cfg
