# retrieval_chain_builder.py
from pathlib import Path
from typing import Iterable

from app.pdf.loader import load_pdfs
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings

from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.pdf.rag.budgeted_retriever import BudgetedRetriever
from app.pdf.rag.prompts import COMBINE_PROMPT, DEFAULT_QA_PROMPT, MAP_PROMPT
from app.pdf.vectorstore_manager import build_or_load_faiss, cache_dir_for


def build_retrieval_qa_chain(
    *,
    llm: object,  # LangChain 互換 LLM（OpenVINO/HF Pipeline/LlamaCpp 等）
    embeddings: HuggingFaceEmbeddings,
    pdf_paths: Iterable[Path],
    vector_cache_base: Path,
    # 分割
    chunk_size: int = 700,
    chunk_overlap: int = 80,
    # 取得/再ランク
    k: int = 3,                    # LLM に最終投入するドキュメント件数
    mmr_fetch_k: int = 40,         # まず広く拾う件数（MMR）
    mmr_lambda: float = 0.3,       # 多様性寄り
    use_cross_encoder: bool = True,
    reranker_model_name: str = "BAAI/bge-reranker-base",
    # 連結
    use_map_reduce: bool = True,   # True: map_reduce / False: stuff
    tokenizer_id_or_path: str | Path | None = None
) -> RetrievalQA:
    """
    取得→圧縮→map_reduce で、入力長を増やさず“長文をはける”構成にする。
    """

    # 1) 読み込み
    docs = load_pdfs(pdf_paths)

    # 2) 分割（小さめチャンクでリコールを確保）
    if tokenizer_id_or_path is not None:
        tok = AutoTokenizer.from_pretrained(str(tokenizer_id_or_path), trust_remote_code=True)
        # ★ モデルと同じトークナイザで“トークン数”として分割（日本語の膨張を考慮）
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tok, chunk_size=350, chunk_overlap=40  # ← まずは堅めに
        )
        chunks = splitter.split_documents(docs)
    else:
        # フォールバック（将来削除推奨）
        from .splitter import split_documents
        chunks = split_documents(docs, chunk_size=500, chunk_overlap=60)  # ← 少し厳しめ

    # 3) ベクトルストア（キャッシュ）
    cache_dir = cache_dir_for(pdf_paths, vector_cache_base)
    db = build_or_load_faiss(documents=chunks, embeddings=embeddings, cache_dir=cache_dir)

    # 4) Retriever（MMR で冗長回避 & 多様性確保）
    #    まずは広く fetch し、その後で絞る
    base_retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": max(k * 2, 6),
            "fetch_k": min(mmr_fetch_k, 32),
            "lambda_mult": mmr_lambda,
        },
    )

    retriever = base_retriever

    # ★ 最終安全弁：合計入力を物理的に 800 tokens 以内に制限
    if tokenizer_id_or_path is not None:
        retriever = BudgetedRetriever(base=retriever, tok=tok, budget_tokens=600)

    # 4.5) （任意）クロスエンコーダ再ランク ＋ 文脈圧縮
    #      top_n=k に絞り、LLM への最終投入を制限
    if use_cross_encoder:
        try:
            from langchain.retrievers import ContextualCompressionRetriever
            from langchain.retrievers.document_compressors import CrossEncoderReranker
            from langchain_community.cross_encoders import HuggingFaceCrossEncoder

            rerank_model = HuggingFaceCrossEncoder(model_name=reranker_model_name)
            compressor = CrossEncoderReranker(model=rerank_model, top_n=k)
            # ★ ここで "base_retriever" に包装済み retriever を渡す（上書き回避）
            retriever = ContextualCompressionRetriever(
                base_retriever=retriever,
                base_compressor=compressor,
            )
        except Exception as e:
            # 依存未導入・モデルDL不可などの場合は素の MMR にフォールバック
            # （ログだけ出して継続）
            import warnings
            warnings.warn(f"CrossEncoderReranker を無効化します: {e}")
            retriever = base_retriever

    # 5) RetrievalQA chain（map_reduce で分割要約→縮約）
    if use_map_reduce:
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="map_reduce",
            return_source_documents=False,
            chain_type_kwargs={
                "question_prompt": MAP_PROMPT,   # map 側は既定で {context}
                "combine_prompt": COMBINE_PROMPT, # combine 側は {summaries}
                "token_max": 800,                 # ★ 中間が長いと自動で「折り畳み」てから結合
            },
        )
    else:
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": DEFAULT_QA_PROMPT, "document_variable_name": "context"},
    )

    return chain
