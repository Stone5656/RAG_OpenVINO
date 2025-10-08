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
from pathlib import Path
from rag_openvino_app.utils.logger_utils import with_logger
from rag_openvino_app.constants.paths import MODELS_DIR, INDEX_DIR, VDB_BASENAME
import re

_EXCLUDE_PARTS = {"hf_cache", ".cache", ".locks"}

def _looks_excluded(path: Path) -> bool:
    """キャッシュ/ロック系のパスを除外判定"""
    for part in path.parts:
        if part in _EXCLUDE_PARTS or part.endswith(".locks"):
            return True
    return False

def _restore_repo_id_from_rel(parts: tuple[str, ...]) -> str:
    """
    HF 由来の階層から repo_id を復元する。
    - _hub/<sanitized>/models--<org>--<name>/snapshots/<rev>/model_cache/openvino_model.xml
    - _hub__<sanitized>__models--...  のようなフラット名にも対応
    - <sanitized> は "__" → "/" に戻す
    """
    # 1) フラット化されたパターン: "_hub__{san}__models--..."
    flat = "/".join(parts)
    m = re.match(r"^_hub__([^_][\s\S]+?)__models--", flat)
    if m:
        return m.group(1).replace("__", "/")

    # 2) ふつうの階層: "_hub/{sanitized}/models--..."
    if len(parts) >= 2 and parts[0] == "_hub":
        return parts[1].replace("__", "/")  # 例: OpenVINO__Qwen3-4B-int4-ov → OpenVINO/Qwen3-4B-int4-ov

    # 3) 先頭ディレクトリ名から推測（手置きのローカルIRなど）
    #    例: qwen2_1.5B/openvino_model.xml → "qwen2_1.5B"
    return parts[0].replace("\\", "/")

def _strip_suffixes(rel_path_str: str) -> str:
    """
    "xxx/openvino_model.xml" や ".xml" を取り除く。
    返り値は「ID”だけ”」。
    """
    if rel_path_str.endswith("openvino_model.xml"):
        rel_path_str = rel_path_str[: -len("openvino_model.xml")].rstrip("/\\")
    elif rel_path_str.endswith(".xml"):
        rel_path_str = rel_path_str[: -len(".xml")].rstrip("/\\")
    return rel_path_str

def _discover_models() -> list[str]:
    """
    models/ 配下の IR (.xml) を探索し、UI 用の「ID だけ」のリストを返す。
    - cache/locks は探索しない
    - MODELS_DIR ルートは取り除く
    - openvino_model.xml / .xml はサフィックス削除
    - HF 由来は __ を / に復元して repo 形式に整形
    """
    if not MODELS_DIR.exists():
        return []

    ids: set[str] = set()

    for xml in MODELS_DIR.glob("**/*.xml"):
        if _looks_excluded(xml):
            continue

        # MODELS_DIR 相対パスのパーツ
        rel = xml.relative_to(MODELS_DIR)
        parts = rel.parts  # tuple
        # まずは repo_id 推定（__ → / 復元もここで）
        repo_id = _restore_repo_id_from_rel(parts)
        # サフィックス除去（openvino_model.xml / .xml）
        repo_id = _strip_suffixes(repo_id)
        # Windows 対応で / に揃える
        repo_id = repo_id.replace("\\", "/")

        if repo_id:  # 空防止
            ids.add(repo_id)

    return sorted(ids)


@with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
def render_sidebar(
    *,
    key_prefix: str,                     # ★ 呼び出し側で必ずユニークにする（例: "main_sidebar"）
    default_model_type: str = "ov",
    default_device: str = "AUTO:GPU,CPU",
    default_temperature: float = 0.2,
    default_max_new_tokens: int = 1024,
    default_top_k: int = 12,
    default_top_n: int = 6,
    default_mmr_lambda: float = 0.4,
    logger=None,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    with st.sidebar:
        st.header("設定")

        # === モデル設定 ===
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

        discovered = _discover_models()
        # models/ に IR がない場合も selectbox は出す（空1件＋手入力でカバー）
        model_choices = ["（models/ から選択）"] + discovered
        model_sel = st.selectbox(
            "モデルID / IR パス（models/ 探索結果）",
            options=model_choices,
            index=0,
            key=f"{key_prefix}_model_from_dir",
        )

        # HF ID やローカル任意パスの手入力（※プリセットは無し）
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

        # === RAG 設定 ===
        st.subheader("RAG 設定")
        top_k = st.number_input("Top-k（初段検索）", 1, 100, default_top_k, 1, key=f"{key_prefix}_topk")
        top_n = st.number_input("上位N（再ランク後）", 1, int(top_k), default_top_n, 1, key=f"{key_prefix}_topn")
        mmr_lambda = st.slider("MMR λ（関連性↔多様性）", 0.0, 1.0, default_mmr_lambda, 0.05, key=f"{key_prefix}_mmr")

        # === ベクトルインデックス設定 ===
        st.subheader("ベクトルインデックス")
        index_dir = st.text_input("保存ディレクトリ", value=str(INDEX_DIR), key=f"{key_prefix}_index_dir")
        index_name = st.text_input("ベース名（拡張子不要）", value=VDB_BASENAME, key=f"{key_prefix}_index_name")

        # ヘルプ
        if not discovered:
            st.info("models/ 配下に IR(.xml) が見つかりません。HF のモデルIDを上の手入力欄に入れると自動取得します。")

    model_cfg = {
        "type": model_type,
        "device": device,
        "model_id": model_id,  # 手入力があればそれ、無ければ models/ 選択
        "temperature": float(temperature),
        "max_new_tokens": int(max_new_tokens),
    }
    rag_cfg = {
        "top_k": int(top_k),
        "top_n": int(top_n),
        "mmr_lambda": float(mmr_lambda),
    }
    index_cfg = {
        "index_dir": index_dir,
        "index_name": index_name,
    }

    logger.debug("Sidebar(model_cfg)=%s", model_cfg)
    logger.debug("Sidebar(rag_cfg)=%s", rag_cfg)
    logger.debug("Sidebar(index_cfg)=%s", index_cfg)
    return model_cfg, rag_cfg, index_cfg
