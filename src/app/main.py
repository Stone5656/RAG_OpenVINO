# -*- coding: utf-8 -*-
"""
アプリ統合エントリポイント（Streamlit）

- 役割:
    * 分離済みの UI / PDF / モデル / 会話層を結線し、RAG型QA UI を提供する
- 方針:
    * pathlib.Path でパス管理
    * 型注釈は from __future__ import annotations
    * UI は ui/ モジュール、PDFは pdf/、モデルは model/、会話は conversation/ に委譲
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
from transformers import AutoTokenizer
import streamlit as st

# === 層の import ===
from app.pdf.rag.retrieval_chain_builder import build_retrieval_qa_chain
from ui.sidebar import draw_sidebar, inject_base_style
from ui.chat_window import (
    render_header,
    render_chat_history,
    render_user_input,
    ChatMessage,
    StreamlitTokenSink,
)

from conversation.session_state import ensure_defaults
from conversation.manager import save_conversation, iter_conversations, load_conversation

from model.factory import load_model
from pdf.embedding_manager import load_embeddings

# === 定数 ===
ROOT = Path(__file__).resolve().parent / "../../"
MODELS_DIR = ROOT / "models"
UPLOADED_DOCS_DIR = ROOT / "uploaded_docs"
CONVERSATIONS_DIR = ROOT / "conversations"
VECTOR_CACHE_BASE = ROOT / "faiss_index_cache"
STYLE = ROOT / "ui" / "style.css"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"  # 既存設計の継承（Embedding名）


def _mkdirs() -> None:
    """必要ディレクトリの作成"""
    for d in (MODELS_DIR, UPLOADED_DOCS_DIR, CONVERSATIONS_DIR, VECTOR_CACHE_BASE):
        d.mkdir(parents=True, exist_ok=True)


# ====== Streamlit 初期化 ======
st.set_page_config(page_title="PocketAI Mentor", page_icon="🧠", layout="wide")
inject_base_style(STYLE)
_mkdirs()
ensure_defaults()


# ====== コールバック: リソース読込 ======
def on_load_resources(model_ref: str, pdf_names: list[str]) -> None:
    """
    モデルとPDFを読み込み、RAGチェーンを作成して session_state に保存。
    - 既存 app.py の load_resources() 相当だが、pathlib化＆層分離版。
    """
    # 1) 参照をローカルorIDに正規化
    ref = model_ref.strip()
    cand = (MODELS_DIR / ref)
    is_local = cand.exists() or Path(ref).exists()
    model_path = cand.resolve() if cand.exists() else Path(ref).resolve() if Path(ref).exists() else None
    model_id_or_path = model_path if model_path else ref
    embeddings = load_embeddings(EMBEDDING_MODEL)

    pdf_paths = [UPLOADED_DOCS_DIR / n for n in pdf_names]

    # 2) HF repo_id の場合はローカル保存先を決める（例：models/_hub/<owner>__<repo>）
    def _sanitize(repo_id: str) -> str:
        return repo_id.replace("/", "__")
    persist_dir = None if is_local else (MODELS_DIR / "_hub" / _sanitize(ref))
    cache_dir = persist_dir  # そのままcache_dirも同じ場所に

    # 3) 参照の実体からバックエンドを自動選択
    def _looks_like_hf_id(s: str) -> bool:
        return ("/" in s) and ("\\" not in s) and (":" not in s)
    prefer = "ov"
    tok_ref = None
    if isinstance(model_id_or_path, Path):
        if model_id_or_path.is_file() and model_id_or_path.suffix.lower() == ".gguf":
            prefer = "llamacpp"
        elif model_id_or_path.is_dir():
            # openvino_model.xml を含むフォルダか？（再帰で検査）
            ov_xml = list(model_id_or_path.rglob("openvino_model.xml"))
            if ov_xml:
                model_id_or_path = ov_xml[0].parent  # IR フォルダの直下を指す
                prefer = "ov"
            else:
                # ディレクトリだがIRが無い→ 変換していない LLM フォルダ等。ここでは扱わない。
                raise ValueError("OpenVINO の IR ではありません（openvino_model.xml が見つかりません）。")
    else:
        # 文字列：HF の OV モデルIDならそのまま OV。Windows 絶対パスは \ や : を含むので弾く
        if not _looks_like_hf_id(model_id_or_path):
            raise ValueError(f"モデル参照が不正です: {model_id_or_path}")

    # 4) ロード（prefer に応じて切替）
    if prefer == "llamacpp":
        llm = load_model(
            "llamacpp",
            model_id_or_path,
            n_ctx=2048, n_gpu_layers=0, n_batch=512, f16_kv=True,
        )
        # トークン分割用トークナイザは別途指定（GGUFは直接は取れないため）
        tok_ref = "gpt2"  # ※日本語用に合わせたい場合は該当HF IDに差し替え
    else:
        llm = load_model(
            "ov",
            model_id_or_path,
            device="GPU",
            export=False,
            max_new_tokens=256,
            persist_dir=persist_dir,
            cache_dir=cache_dir,
        )
        tok_ref = str(model_id_or_path) if isinstance(model_id_or_path, Path) else model_id_or_path


    # 3) RAG チェーン
    chain = build_retrieval_qa_chain(
        llm=llm,
        embeddings=embeddings,
        pdf_paths=pdf_paths,
        vector_cache_base=VECTOR_CACHE_BASE,
        chunk_size=700,
        chunk_overlap=80,
        k=3,
        tokenizer_id_or_path=tok_ref,
    )

    # 4) セッション反映（既存はメッセージ初期化していたが、読込系では保持したい場合もあるので上書きしない）
    st.session_state["qa_chain"] = chain
    st.session_state["selected_model"] = str(model_id_or_path)
    st.session_state["selected_pdfs"] = pdf_names


# ====== コールバック: 新規会話 ======
def on_new_conversation() -> None:
    st.session_state["messages"] = []
    st.session_state["qa_chain"] = None
    st.session_state["selected_model"] = ""
    st.session_state["selected_pdfs"] = []

# ====== コールバック: 会話保存 ======
def on_save_conversation_cb(save_name: str) -> None:
    save_conversation(CONVERSATIONS_DIR, save_name or None)

# ====== コールバック: 会話読込 ======
def on_load_conversation_cb(path: Path) -> None:
    """
    1) JSON を読み込み session_state に model/pdf/messages を反映
    2) RAG を再構築（既存フローの「読込→load_resources→messages復元」を再現）
    """
    load_conversation(CONVERSATIONS_DIR, path)
    model_name = st.session_state.get("selected_model", "")
    pdf_names = list(st.session_state.get("selected_pdfs", []))
    if model_name and pdf_names:
        # 再構築（ここでは messages は維持）
        on_load_resources(model_name, pdf_names)
        st.rerun()

# ====== コールバック: チャット送信 ======
def on_submit(prompt: str, sink: StreamlitTokenSink) -> None:
    # ★ ユーザー質問をトークンで強制短縮（例：200 tokens）
    tok = st.session_state.get("tokenizer", None)
    if tok is not None:
        ids = tok.encode(prompt, add_special_tokens=False)
        if len(ids) > 200:
            head = tok.decode(ids[:140])
            tail = tok.decode(ids[-60:])
            prompt = head + "\n…（中略）…\n" + tail
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # QA チェーン実行（ストリーミングAPIが無い想定なので、一括→sink.write）
    chain = st.session_state.get("qa_chain")
    if not chain:
        sink.write("（まだPDF/モデルが読み込まれていません）")
        return

    # LangChain RetrievalQA.invoke() を使う（既存の呼び出し方を踏襲）
    result = chain.invoke(prompt)
    answer = (result.get("result") or "").strip() or "結果を取得できませんでした。"
    sink.write(answer)

    # 履歴にアシスタント発話を反映
    st.session_state["messages"].append({"role": "assistant", "content": answer})

# ====== UI（サイドバー） ======
draw_sidebar(
    models_dir=MODELS_DIR,
    uploaded_docs_dir=UPLOADED_DOCS_DIR,
    conversations_dir=CONVERSATIONS_DIR,
    on_load_resources=on_load_resources,
    on_new_conversation=on_new_conversation,
    on_save_conversation=on_save_conversation_cb,
    on_load_conversation=on_load_conversation_cb,
)

# ====== メイン（チャット） ======
selected_model = st.session_state.get("selected_model")
selected_pdfs = st.session_state.get("selected_pdfs")

if selected_model and selected_pdfs:
    render_header(selected_model, selected_pdfs)
    render_chat_history(st.session_state.get("messages", []))  # 過去ログ
    render_user_input("PDFに関する質問を入力してください", on_submit)
else:
    st.header("ようこそ！ PocketAI Mentorへ")
    st.markdown(
        "左のサイドバーで **モデル** と **PDF** をセットアップし、`読み込む` を押してください。"
    )
