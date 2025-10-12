# -*- coding: utf-8 -*-
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import json, hashlib

from rag_openvino_app.ui.sidebar import render_sidebar
from rag_openvino_app.ui.chat_window import render_chat
from rag_openvino_app.ui.uploader import render_uploader
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
    JA_EMBEDDING_MODEL_ID, CHUNK_SIZE, CHUNK_OVERLAP,
    INDEX_DIR, VDB_BASENAME
)
from rag_openvino_app.utils.logger_utils import with_logger
from rag_openvino_app.model import get_model_manager  # ★ LLM直作成に使う

load_dotenv()

# ベクタDBはプロセス内シングルトンでOK（Streamlitの再実行でもモジュールキャッシュされる）
vdb = InMemoryVectorStore()

# ===== キャッシュ資源 =====
@st.cache_resource
def get_embedder() -> SentenceTransformerEmb:
    return SentenceTransformerEmb(JA_EMBEDDING_MODEL_ID)

# ===== VDB I/O =====
@with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
def load_index(index_dir: str | Path = INDEX_DIR, index_name: str = VDB_BASENAME, *, logger=None):
    base = Path(index_dir) / index_name
    npz = base.with_suffix(".npz")
    if npz.exists():
        vdb.load(base)
        logger.info("ベクトルインデックスを読み込みました: %s", base)
    else:
        logger.warning("インデックスが見つかりませんでした: %s", base)

@with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
def save_index(index_dir: str | Path = INDEX_DIR, index_name: str = VDB_BASENAME, *, logger=None):
    base = Path(index_dir) / index_name
    base.parent.mkdir(parents=True, exist_ok=True)
    vdb.save(base)
    logger.info("ベクトルインデックスを保存しました: %s", base)

@with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
def build_index_from_pdfs(pdf_paths: list[Path], *, logger=None):
    emb = get_embedder()  # ★ キャッシュされた埋め込み器
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

# ===== RAG Adapter =====
class VDBAdapter:
    def similarity_search(self, query_vec, k: int):
        return vdb.similarity_search(query_vec, k=k)

class EmbAdapter:
    def __init__(self):
        self._emb = get_embedder()
    def embed(self, texts):
        return self._emb.embed(texts)

# ===== 署名作成（変更検知） =====
def _llm_sig(model_cfg: dict) -> str:
    key = {
        "type": model_cfg.get("type"),
        "model_id": model_cfg.get("model_id"),
        "device": model_cfg.get("device"),
        "max_new_tokens": model_cfg.get("max_new_tokens"),
        "temperature": model_cfg.get("temperature"),
    }
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()

def _rag_sig(rag_cfg: dict) -> str:
    key = {
        "top_k": rag_cfg.get("top_k"),
        "top_n": rag_cfg.get("top_n"),
        "mmr_lambda": rag_cfg.get("mmr_lambda"),
    }
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()

# ===== Manager 構築（LLMは外部注入） =====
def _make_manager(model_cfg, rag_cfg, llm):
    retriever = Retriever(EmbAdapter(), VDBAdapter(), RetrieverConfig(
        k=rag_cfg["top_k"], use_mmr=True, mmr_lambda=rag_cfg["mmr_lambda"]
    ))
    reranker = Reranker(RerankerConfig(top_n=rag_cfg["top_n"]))
    compressor = Compressor(CompressorConfig())
    # ★ ConversationManager に llm を注入（内部での毎回作成を回避）
    return ConversationManager(retriever, reranker, compressor, model_cfg, llm=llm)

def main():
    st.set_page_config(page_title="RAG OpenVINO UI", layout="wide")

    model_cfg, rag_cfg, index_cfg = render_sidebar(
        key_prefix="main_sidebar",
        default_device="GPU"
    )
    index_dir = index_cfg["index_dir"]
    index_name = index_cfg["index_name"]

    # 1) 既存インデックスをロード（存在すれば）
    load_index(index_dir, index_name)

    # 2) PDF アップロード UI（追加があれば即インデックス→保存）
    newly = render_uploader()
    if newly:
        build_index_from_pdfs(newly)
        save_index(index_dir, index_name)

    # 3) セッション初期化
    if "session" not in st.session_state:
        st.session_state.session = SessionState()
    # セッションに LLM と Manager の署名を持っておく
    llm_sig_now = _llm_sig(model_cfg)
    rag_sig_now = _rag_sig(rag_cfg)

    # 4) LLM の初期化は「モデル選択が確定した瞬間だけ」
    llm = st.session_state.get("llm")
    if model_cfg.get("model_id"):  # 空なら作らない
        if st.session_state.get("llm_sig") != llm_sig_now or llm is None:
            # ★ ここだけが LLM 作成のトリガ
            llm = get_model_manager(model_cfg)
            st.session_state.llm = llm
            st.session_state.llm_sig = llm_sig_now

    # 5) ConversationManager は軽量再構築（ただし LLM は再利用）
    manager = st.session_state.get("manager")
    need_rebuild = (
        manager is None or
        st.session_state.get("rag_sig") != rag_sig_now or
        st.session_state.get("llm_sig") != llm_sig_now  # LLM が変わったら再構築
    )
    if need_rebuild:
        manager = _make_manager(model_cfg, rag_cfg, llm)
        st.session_state.manager = manager
        st.session_state.rag_sig = rag_sig_now

    # 6) 会話UI（LLM未構築＝モデル未選択なら manager は None のまま → UIで誘導）
    render_chat(
        manager=manager,
        session=st.session_state.session,
        temperature=model_cfg.get("temperature", 0.2),
    )

if __name__ == "__main__":
    main()
