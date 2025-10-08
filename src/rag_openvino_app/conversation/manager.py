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
    def __init__(self, retriever: Retriever, reranker: Reranker, compressor: Compressor, model_cfg: dict, *, logger=None):
        self.retriever = retriever
        self.reranker = reranker
        self.compressor = compressor
        self.llm = get_model_manager(model_cfg)
        logger.debug("ConversationManager 初期化完了。モデル=%s", model_cfg.get("type"))

    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def run_pipeline(self, query: str, *, max_chunks: int = 8, temperature: float = 0.2, logger=None) -> dict:
        t0 = time.time()
        logger.debug("Conversation: パイプライン開始。query='%s'", query)

        retrieved = self.retriever.retrieve(query, top_k=max_chunks)
        logger.debug("Conversation: 検索完了（%d 件）", len(retrieved))
        reranked = self.reranker.rerank(query, retrieved)
        logger.debug("Conversation: 再ランク完了（上位 %d 件）", len(reranked))
        compressed = self.compressor.compress(reranked)
        logger.debug("Conversation: 圧縮完了（%d 件）", len(compressed))

        prompt = self._build_prompt(query, compressed)
        raw = self.llm.generate(prompt, temperature=temperature)

        # 既存のテンプレ除去
        answer = self._strip_prompt_echo(raw, prompt)
        # ★ 最小の網（必要に応じてオン）
        answer = self._postprocess_bullets(answer, max_lines=6)

        elapsed = time.time() - t0
        logger.debug("Conversation: 完了（%.3f 秒）", elapsed)
        return {"answer": answer, "contexts": compressed, "elapsed": elapsed}

    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def _build_prompt(self, query: str, contexts: list[str] | list[dict], *, logger=None) -> str:
        """
        System + User を連結して 1 本のプロンプトにする。
        """
        user = build_user_prompt(query, contexts)   # ← ここで {context}/{question} を展開
        prompt = f"{SYSTEM_PROMPT}\n\n{user}\n"
        # デバッグ用に長さだけ出す（本文は出さない）
        logger.debug("Conversation: プロンプト生成完了（chars=%d）", len(prompt))
        return prompt

    def _strip_prompt_echo(self, text: str, prompt: str) -> str:
        if not text:
            return ""
        s = text

        # 1) まず「【回答】」以降があればそれだけ返す
        if "【回答】" in s:
            return s.split("【回答】", 1)[-1].strip()

        # 2) 入力プロンプトの“ほぼ前方一致”を許容して剥がす
        def _norm(u: str) -> str:
            return " ".join(u.replace("\u3000", " ").split())

        try:
            if _norm(s).startswith(_norm(prompt)):
                s = s[len(prompt):]
        except Exception:
            pass

        # 3) それでもテンプレ断片が先頭にあれば大きめに落とす（保険）
        #    [コンテキスト開始]〜[質問]などのヘッダを丸ごとカット
        PAT_HEAD = r"(?:以下のコンテキスト.*?\[質問\]\s*.*?$)"
        s = re.sub(PAT_HEAD, "", s, flags=re.S).strip()

        # 4) OpenVINO ラベル等の接頭辞を剥がす（既存処理）
        s = re.sub(r"^\[OpenVINO 出力\]\s*\([^)]+\)\s*→\s*", "", s).strip()

        return s

    @staticmethod
    def _postprocess_bullets(answer: str, max_lines: int = 6) -> str:
        if not answer:
            return ""
        # 行の重複を削る + 先頭を「- 」に統一 + 上限行数でカット
        seen = set()
        out = []
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


