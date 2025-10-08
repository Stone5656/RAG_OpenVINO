# -*- coding: utf-8 -*-
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from glob import glob
import os

from rag_openvino_app.ui.sidebar import render_sidebar
from rag_openvino_app.ui.chat_window import render_chat
from rag_openvino_app.ui.uploader import render_uploader  # ★ 追加
from rag_openvino_app.conversation.manager import ConversationManager
from rag_openvino_app.conversation.session_state import SessionState

from rag_openvino_app.rag.retriever import Retriever, RetrieverConfig
from rag_openvino_app.rag.reranker import Reranker, RerankerConfig
from rag_openvino_app.rag.compressor import Compressor, CompressorConfig

from rag_openvino_app.pdf.vectorstore_manager import InMemoryVectorStore
from rag_openvino_app.pdf.loader import load_pdf
from rag_openvino_app.pdf.splitter import split_text
from rag_openvino_app.pdf.embedding_manager import PDFEmbeddingManager

from rag_openvino_app.rag.ja_embedding import SentenceTransformerEmb
from rag_openvino_app.constants.paths import (
    JA_EMBEDDING_MODEL_ID, PDF_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
    INDEX_DIR, VDB_BASENAME
)
from rag_openvino_app.utils.logger_utils import with_logger

load_dotenv()
vdb = InMemoryVectorStore()

@with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
def load_index(index_dir: str | Path = INDEX_DIR, index_name: str = VDB_BASENAME, *, logger=None):
    """
    ベクトルインデックスを読み込む。

    Parameters
    ----------
    index_dir : str | Path, optional
        読み込み先ディレクトリ（既定: constants.paths.INDEX_DIR）
    index_name : str, optional
        ベース名（拡張子不要、既定: constants.paths.VDB_BASENAME）
    """
    base = Path(index_dir) / index_name
    npz = base.with_suffix(".npz")

    if npz.exists():
        vdb.load(base)
        logger.info("ベクトルインデックスを読み込みました: %s", base)
    else:
        logger.warning("インデックスが見つかりませんでした: %s", base)


@with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
def save_index(index_dir: str | Path = INDEX_DIR, index_name: str = VDB_BASENAME, *, logger=None):
    """
    ベクトルインデックスを保存する。

    Parameters
    ----------
    index_dir : str | Path, optional
        保存先ディレクトリ（既定: constants.paths.INDEX_DIR）
    index_name : str, optional
        ベース名（拡張子不要、既定: constants.paths.VDB_BASENAME）
    """
    base = Path(index_dir) / index_name
    base.parent.mkdir(parents=True, exist_ok=True)

    vdb.save(base)
    logger.info("ベクトルインデックスを保存しました: %s", base)

@with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
def build_index_from_pdfs(pdf_paths: list[Path], *, logger=None):
    emb = SentenceTransformerEmb(JA_EMBEDDING_MODEL_ID)
    pdf_embed_mgr = PDFEmbeddingManager(emb)
    all_docs = []
    for p in pdf_paths:
        text, meta = load_pdf(p)
        if not text.strip():
            continue
        chunks = split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, method="sentence")
        if not chunks:
            continue
        docs = pdf_embed_mgr.build_embeddings(chunks, base_meta={"source": str(p)})
        all_docs.extend(docs)
    if all_docs:
        vdb.add_documents(all_docs)
        logger.info("PDF から %d チャンクを追加しました。総件数=%d", len(all_docs), len(vdb.texts))

class VDBAdapter:
    def similarity_search(self, query_vec, k: int):
        return vdb.similarity_search(query_vec, k=k)

class EmbAdapter:
    def __init__(self, model_id: str):
        self._emb = SentenceTransformerEmb(model_id)
    def embed(self, texts):
        return self._emb.embed(texts)

def make_manager(model_cfg, rag_cfg):
    retriever = Retriever(EmbAdapter(JA_EMBEDDING_MODEL_ID), VDBAdapter(), RetrieverConfig(
        k=rag_cfg["top_k"], use_mmr=True, mmr_lambda=rag_cfg["mmr_lambda"]
    ))
    reranker = Reranker(RerankerConfig(top_n=rag_cfg["top_n"]))
    compressor = Compressor(CompressorConfig())
    return ConversationManager(retriever, reranker, compressor, model_cfg)

def main():
    st.set_page_config(page_title="RAG OpenVINO UI", layout="wide")

    model_cfg, rag_cfg, index_cfg = render_sidebar(
        key_prefix="main_sidebar",
        default_device="GPU"
    )
    index_dir = index_cfg["index_dir"]
    index_name = index_cfg["index_name"]

    # 1) 既存インデックスをロード
    load_index(index_dir, index_name)

    # 2) PDF アップロード UI（追加があれば即インデックス→保存）
    newly = render_uploader()
    if newly:
        build_index_from_pdfs(newly)
        save_index(index_dir, index_name)

    # 4) 会話UI
    if "session" not in st.session_state:
        st.session_state.session = SessionState()
        st.session_state.manager = make_manager(model_cfg, rag_cfg)
    else:
        st.session_state.manager = make_manager(model_cfg, rag_cfg)

    render_chat(
        manager=st.session_state.manager,
        session=st.session_state.session,
        temperature=model_cfg.get("temperature", 0.2),
    )

if __name__ == "__main__":
    main()
