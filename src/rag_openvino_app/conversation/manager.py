# -*- coding: utf-8 -*-
"""
Conversation Manager

責務:
- RAG 全体のパイプライン（検索→再ランク→要約→生成）を統合
- LLM と Retriever をシームレスに接続
- ストリームリットや FastAPI からの入出力に対応可能

ログ出力:
- 各ステップ開始/完了時に DEBUG 出力
- 処理時間・抽出件数・入力長を含む詳細ログ
"""
from __future__ import annotations
import time
import re
from rag_openvino_app.rag.prompts import SYSTEM_PROMPT, build_user_prompt
from rag_openvino_app.utils.logger_utils import with_logger
from rag_openvino_app.rag.retriever import Retriever
from rag_openvino_app.rag.reranker import Reranker
from rag_openvino_app.rag.compressor import Compressor
from rag_openvino_app.model import get_model_manager

PROMPT_HEADER_RE = re.compile(r"^以[下後]の文脈に基づいて質問に答えてください。.*?--- 質問 ---\s*.*?$", re.S)

class ConversationManager:
    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        compressor: Compressor,
        model_cfg: dict,
        *,
        llm=None,     # ★ 追加: 事前構築LLMを受け取れるように
        logger=None
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.compressor = compressor
        # 既存互換: llm が渡されなければ従来どおり factory で作る
        self.llm = llm if llm is not None else get_model_manager(model_cfg)
        logger.debug("ConversationManager 初期化完了。モデル=%s（prebuilt=%s）",
                     model_cfg.get("type"), llm is not None)

    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def run_pipeline(self, query: str, *, max_chunks: int = 8, temperature: float = 0.2, logger=None) -> dict:
        t0 = time.time()
        logger.debug("Conversation: パイプライン開始。query='%s'", query)

        retrieved = self.retriever.retrieve(query, top_k=max_chunks)
        logger.debug("Conversation: 検索完了（%d 件）", len(retrieved))
        reranked = self.reranker.rerank(query, retrieved)
        logger.debug("Conversation: 再ランク完了（上位 %d 件）", len(reranked))
        compressed = self.compressor.compress(reranked)
        logger.debug("Conversation: 厳選完了（%d 件）", len(compressed))

        prompt = self._build_prompt(query, compressed)
        raw = self.llm.generate(prompt, temperature=temperature)

        answer = self._strip_prompt_echo(raw, prompt)
        answer = self._postprocess_bullets(answer, max_lines=75)

        elapsed = time.time() - t0
        logger.debug("Conversation: 完了（%.3f 秒）", elapsed)
        return {"answer": answer, "contexts": compressed, "elapsed": elapsed}

    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def _build_prompt(self, query: str, contexts: list[str] | list[dict], *, logger=None) -> str:
        user = build_user_prompt(query, contexts)
        prompt = f"{SYSTEM_PROMPT}\n\n{user}\n"
        logger.debug("Conversation: プロンプト生成完了（chars=%d）", len(prompt))
        return prompt

    def _strip_prompt_echo(self, text: str, prompt: str) -> str:
        if not text:
            return ""
        s = text
        if "【回答】" in s:
            return s.split("【回答】", 1)[-1].strip()
        def _norm(u: str) -> str:
            return " ".join(u.replace("\u3000", " ").split())
        try:
            if _norm(s).startswith(_norm(prompt)):
                s = s[len(prompt):]
        except Exception:
            pass
        PAT_HEAD = r"(?:以下のコンテキスト.*?\[質問\]\s*.*?$)"
        s = re.sub(PAT_HEAD, "", s, flags=re.S).strip()
        s = re.sub(r"^\[OpenVINO 出力\]\s*\([^)]+\)\s*→\s*", "", s).strip()
        return s

    @staticmethod
    def _postprocess_bullets(answer: str, max_lines: int = 6) -> str:
        if not answer:
            return ""
        seen, out = set(), []
        for raw in answer.splitlines():
            line = raw.strip()
            if not line:
                continue
            if "この回答は文脈に基づいていますか" in line:
                continue
            if line not in seen:
                seen.add(line)
                out.append(line if line.startswith("- ") else f"- {line}")
        return "\n".join(out[:max_lines]).strip()
