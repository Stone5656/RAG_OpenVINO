# prompts.py
from langchain.prompts import PromptTemplate


# 各チャンクに対する“部分回答/要約”用（map フェーズ）
MAP_PROMPT = PromptTemplate(
    template=(
        "### 指示:\n"
        "与えられたコンテキストだけに基づき、ユーザーの質問に答えるための要点を短く日本語で抽出してください。\n"
        "不明な点は「不明」と記してください。推測はしないでください。\n\n"
        "### コンテキスト:\n{context}\n\n"
        "### 質問:\n{question}\n\n"
        "### 要点（箇条書きで可）:\n"
    ),
    input_variables=["context", "question"],
)

# map の結果を統合する“縮約”用（reduce フェーズ）
COMBINE_PROMPT = PromptTemplate(
    template=(
        "### 指示:\n"
        "以下は複数コンテキストから抽出された要点です。重複を除き、矛盾があれば明記し、"
        "ユーザーの質問に対する最終回答を日本語で簡潔にまとめてください。\n"
        "根拠となる情報がなければ「分かりません」と述べてください。\n\n"
        "### 要点一覧:\n{summaries}\n\n"
        "### 質問:\n{question}\n\n"
        "### 最終回答:\n"
    ),
    # ← map_reduce の reduce 側は既定で {summaries} を受け取る
    input_variables=["summaries", "question"],
)

# “stuff” 用のプレーン QA プロンプト（フォールバック）
DEFAULT_QA_PROMPT = PromptTemplate(
    template=(
        "### 指示:\n"
        "以下のコンテキストだけに基づいて日本語で回答してください。"
        "根拠が無い場合は「分かりません」と答えてください。\n\n"
        "### コンテキスト:\n{context}\n\n"
        "### 質問:\n{question}\n\n"
        "### 回答:\n"
    ),
    input_variables=["context", "question"],
)