# -*- coding: utf-8 -*-
"""
サイドバー UI モジュール

- 役割:
    * モデル選択（OpenVINO IR / GGUF）、PDF アップロード/選択、読込トリガー
    * 会話の開始・保存・読込の UI
- 設計方針:
    * UI は入力を集めて与えられたコールバックを呼ぶだけ（ロジックは本体へ委譲）
- 依存:
    * Streamlit（st）
    * pathlib.Path
- 型注釈:
    * `from __future__ import annotations` を採用
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import streamlit as st


def inject_base_style(css_path: Path) -> None:
    """共通 CSS を読み込んで `<style>` として注入する。"""
    if css_path.is_file():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


# ========= 再帰検出ユーティリティ =========

# Hugging Face のローカルキャッシュ等でノイズになりやすいセグメント
_IGNORED_SEGMENTS = {"snapshots", "refs", "blobs", ".cache"}

def _in_ignored(p: Path) -> bool:
    """パスの途中に無視すべきセグメントが含まれているか。"""
    parts = {s.lower() for s in p.parts}
    return any(seg in parts for seg in _IGNORED_SEGMENTS)

# ---------- 追加: 再帰的に OpenVINO IR を検出するユーティリティ ----------
def _is_openvino_ir_dir(dir: Path) -> bool:
    """
    OpenVINO IR ディレクトリ判定:
    - `openvino_model.xml` と同名の `.bin` が存在する か、
    - 任意の `*.xml` と対応する `*.bin` が同ディレクトリに存在する
    """
    if not dir.is_dir():
        return False

    xml = dir / "openvino_model.xml"
    bin_ = dir / "openvino_model.bin"
    if xml.exists() and bin_.exists():
        return True

    # 互換: 任意の *.xml / *.bin ペア
    for x in dir.glob("*.xml"):
        b = x.with_suffix(".bin")
        if b.exists():
            return True
    return False


def _safe_relposix(path: Path, base: Path) -> str:
    """base からの相対パスを POSIX 風スラッシュで返す（Windows対応）。"""
    try:
        rel = path.relative_to(base)
    except ValueError:
        rel = path
    return rel.as_posix()

def _list_ov_local_candidates(models_dir: Path) -> list[str]:
    """
    models_dir 以下を再帰探索し、OpenVINO IR の「親ディレクトリ」を列挙。
    - `snapshots/`, `refs/`, `blobs/`, `.cache/` を含むパスは除外
    - 表示は models_dir からの相対パス
    """
    results: set[str] = set()

    # 典型ファイル名から優先的に
    for xml in models_dir.rglob("openvino_model.xml"):
        parent = xml.parent
        if _in_ignored(parent):
            continue
        if _is_openvino_ir_dir(parent):
            results.add(_safe_relposix(parent, models_dir))

    # 念のため任意 *.xml でも拾う（上と重複は set で抑制）
    for xml in models_dir.rglob("*.xml"):
        parent = xml.parent
        if _in_ignored(parent):
            continue
        if _is_openvino_ir_dir(parent):
            results.add(_safe_relposix(parent, models_dir))

    return sorted(results)

def _list_gguf_candidates(models_dir: Path) -> list[str]:
    """
    models_dir 以下から *.gguf を再帰列挙。
    - `_IGNORED_SEGMENTS` を含むパスは除外
    - 表示は models_dir からの相対パス
    """
    out: list[str] = []
    for f in models_dir.rglob("*.gguf"):
        if _in_ignored(f):
            continue
        out.append(_safe_relposix(f, models_dir))
    return sorted(set(out))

# =========================================


def draw_sidebar(
    *,
    models_dir: Path,
    uploaded_docs_dir: Path,
    conversations_dir: Path,
    on_load_resources: Callable[[str, Sequence[str]], None],
    on_new_conversation: Callable[[], None],
    on_save_conversation: Callable[[str], None],
    on_load_conversation: Callable[[Path], None],
) -> None:
    """サイドバーを描画し、ユーザー操作に応じてコールバックを呼び出す。"""
    with st.sidebar:
        st.header("🧠 PocketAI Mentor")

        # --- セットアップ ---
        with st.expander("⚙️ セットアップ", expanded=True):

            # 1) モデル選択（バックエンド切替）
            st.subheader("1. モデルを指定（OpenVINO / GGUF）")

            backend = st.radio(
                "推論バックエンド",
                options=("OpenVINO (Intel GPU)", "LlamaCpp (GGUF)"),
                index=0,
                horizontal=True,
            )
            # main 側で参照できるように残しておく（シグネチャ変更を避けるため）
            st.session_state["backend"] = "ov" if backend.startswith("OpenVINO") else "llama"

            selected_model_name = ""

            if st.session_state["backend"] == "ov":
                # OpenVINO: ローカルIRの再帰列挙 + HF ID 直入力
                ov_local = _list_ov_local_candidates(models_dir)
                selected_local = st.selectbox(
                    "ローカルのOVモデル（models/配下）",
                    options=["(未選択)"] + ov_local,
                    index=0,
                )
                custom_model_id = st.text_input(
                    "または Hugging Face の OV モデルID（例: helenai/gpt2-ov）",
                    "",
                    help="HFのIDを指定すると、最初のロード時に models/_hub/ 以下へ保存されます（本体側実装依存）。",
                )
                selected_model_name = custom_model_id.strip() or (selected_local if selected_local != "(未選択)" else "")

            else:
                # LlamaCpp: *.gguf の再帰列挙
                gguf_local = _list_gguf_candidates(models_dir)
                selected_model_name = st.selectbox(
                    "ローカルの GGUF モデル（models/配下）",
                    options=["(未選択)"] + gguf_local,
                    index=0,
                )
                if selected_model_name == "(未選択)":
                    selected_model_name = ""

            if not selected_model_name:
                st.info("モデルを選択するか、IDを入力してください。")

            # 2) PDF アップロード
            uploaded = st.file_uploader(
                "2. 新しいPDFをアップロード",
                type=["pdf"],
                accept_multiple_files=True,
            )
            if uploaded:
                for up in uploaded:
                    dest = uploaded_docs_dir / up.name
                    if not dest.exists():
                        dest.write_bytes(up.getvalue())
                        st.success(f"「{up.name}」を保存しました。")
                    else:
                        st.info(f"「{up.name}」は既に存在します。")

            st.divider()

            # 3) 参照 PDF 選択
            stored_pdf_files = sorted(uploaded_docs_dir.glob("*.pdf"))
            stored_pdf_names = [p.name for p in stored_pdf_files]
            if not stored_pdf_names:
                st.info("AI に参照させる PDF をアップロードしてください。")
            selected_pdf_names = st.multiselect(
                "3. 参照するPDFを選択",
                options=stored_pdf_names,
                default=st.session_state.get("selected_pdfs", []),
            )

            # 4) 読み込みトリガー
            if st.button("読み込む", type="primary", use_container_width=True):
                if not selected_model_name:
                    st.warning("モデルを選択（またはIDを入力）してください。")
                elif not selected_pdf_names:
                    st.warning("参照する PDF を 1 つ以上選択してください。")
                else:
                    on_load_resources(selected_model_name, selected_pdf_names)

        # --- 会話の管理 ---
        with st.expander("📁 会話の管理", expanded=False):

            if st.button("新しい会話を開始", use_container_width=True):
                on_new_conversation()
                st.rerun()

            # 保存 UI
            save_name = st.text_input("会話の保存名（拡張子不要）")
            if st.button("現在の会話を保存", use_container_width=True):
                on_save_conversation(save_name)

            # 読込 UI
            conv_files = sorted(conversations_dir.glob("*.json"), reverse=True)
            if conv_files:
                st.divider()
                selected_conv = st.selectbox(
                    "保存した会話を読み込む",
                    options=[p.name for p in conv_files],
                )
                if st.button("この会話を読み込む", use_container_width=True):
                    on_load_conversation(conversations_dir / selected_conv)


def _resolve_select_index(options: Sequence[str], current: str | None) -> int:
    """現在値が options にあればその index、なければ 0。"""
    try:
        if current is None:
            return 0
        return options.index(current)
    except ValueError:
        return 0
