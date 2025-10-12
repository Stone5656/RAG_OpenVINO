# -*- coding: utf-8 -*-
from __future__ import annotations
import shutil
from pathlib import Path
import streamlit as st
from rag_openvino_app.utils.logger_utils import with_logger

# 保存先を docs/ に変更
PDF_SAVE_DIR = Path("docs")

@with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
def render_uploader(*, logger=None) -> list[Path]:
    """
    Streamlit 上で PDF をアップロードし、docs/ に保存する。
    保存後のパス一覧を返す。
    """
    st.subheader("📄 PDF アップロード")
    files = st.file_uploader(
        "PDF をアップロードしてください（複数可）",
        type=["pdf"],
        accept_multiple_files=True,
    )
    saved: list[Path] = []

    if st.button("アップロードして docs/ に追加"):
        if not files:
            st.warning("ファイルが選択されていません。")
            return saved

        # docs フォルダを自動生成
        PDF_SAVE_DIR.mkdir(parents=True, exist_ok=True)

        for f in files:
            dst = PDF_SAVE_DIR / f.name
            with open(dst, "wb") as out:
                shutil.copyfileobj(f, out)
            saved.append(dst)

        st.success(f"{len(saved)} 件の PDF を {PDF_SAVE_DIR} に保存しました。")
        logger.info("PDF を保存: %s", [p.name for p in saved])

    return saved
