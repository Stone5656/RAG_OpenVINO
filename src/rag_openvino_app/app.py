import streamlit as st
import numpy as np
from dotenv import load_dotenv
import os

from rag_openvino_app.ui.sidebar import render_sidebar
from rag_openvino_app.ui.chat_window import render_chat
from rag_openvino_app.conversation.manager import ConversationManager
from rag_openvino_app.conversation.session_state import SessionState

from rag_openvino_app.rag.retriever import Retriever, RetrieverConfig
from rag_openvino_app.rag.reranker import Reranker, RerankerConfig
from rag_openvino_app.rag.compressor import Compressor, CompressorConfig

from rag_openvino_app.pdf.vectorstore_manager import InMemoryVectorStore


# .env を読み込む
load_dotenv()

# 読み込み確認
print(f"[ENV] LOG_FILE_PATH = {os.getenv('LOG_FILE_PATH')}")
print(f"[ENV] LOG_LEVEL = {os.getenv('LOG_LEVEL')}")


# --- ダミーの埋め込み器（実運用では実モデルに置換） ---
class DummyEmb:
    def embed(self, texts):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((len(texts), 8))
        x /= (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        return x

# --- ベクトルストア（起動時に空） ---
vdb = InMemoryVectorStore()
embedder = DummyEmb()

# VectorStore 互換のラッパを用意（Retriever の Protocol に合わせる）
class VDBAdapter:
    def similarity_search(self, query_vec, k: int):
        return vdb.similarity_search(query_vec, k=k)

# Retriever 互換の埋め込み器
class EmbAdapter:
    def embed(self, texts):
        return embedder.embed(texts)

def make_manager(model_cfg, rag_cfg):
    retriever = Retriever(EmbAdapter(), VDBAdapter(), RetrieverConfig(
        k=rag_cfg["top_k"], use_mmr=True, mmr_lambda=rag_cfg["mmr_lambda"]
    ))
    reranker = Reranker(RerankerConfig(top_n=rag_cfg["top_n"]))
    compressor = Compressor(CompressorConfig())
    return ConversationManager(retriever, reranker, compressor, model_cfg)

def main():
    st.set_page_config(page_title="RAG OpenVINO UI", layout="wide")
    model_cfg, rag_cfg = render_sidebar()

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
