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
from rag_openvino_app.rag.prompts import SYSTEM_PROMPT
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
        answer = self._strip_prompt_echo(raw)
        # ★ 最小の網（必要に応じてオン）
        answer = self._postprocess_bullets(answer, max_lines=6)

        elapsed = time.time() - t0
        logger.debug("Conversation: 完了（%.3f 秒）", elapsed)
        return {"answer": answer, "contexts": compressed, "elapsed": elapsed}

    @with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
    def _build_prompt(self, query: str, contexts: list[str] | list[dict], *, logger=None) -> str:
        # OpenVINO 側が system/user を明確に切れない場合は、先頭に System、続けて User を連結
        user = build_user_prompt(query, contexts)
        prompt = f"{SYSTEM_PROMPT}\n\n{user}\n"
        return prompt

    def _strip_prompt_echo(self, text: str) -> str:
        if not text:
            return ""
        # 1) 「--- 回答 ---」以降を優先
        if "--- 回答 ---" in text:
            return text.split("--- 回答 ---", 1)[-1].strip()
        # 2) 冒頭のテンプレヘッダを正規表現で削除
        s = PROMPT_HEADER_RE.sub("", text).strip()
        # 3) ラベルや接頭辞の削除（デモ用のダミーラベルを剥がす）
        s = re.sub(r"^\[OpenVINO 出力\]\s*\([^)]+\)\s*→\s*", "", s).strip()
        return s
    
    # conversation/manager.py の _strip_prompt_echo() の直後に追加関数を置いて使う
    def _postprocess_bullets(answer: str, max_lines: int = 6) -> str:
        """
        - 先頭の「【回答】」以降だけを残す
        - 同一行の重複を除去（順序保持）
        - 行数を最大 max_lines にカット
        - 末尾の評価系・ノイズ行をヒューリスティックに除去
        """
        if not answer:
            return ""

        # 1) 「【回答】」以降
        if "【回答】" in answer:
            answer = answer.split("【回答】", 1)[-1].strip()

        # 2) 行ごとの重複除去（順序保持）
        seen = set()
        lines_out = []
        for raw in answer.splitlines():
            line = raw.strip()
            if not line:
                continue
            # 評価系ノイズの簡易フィルタ
            if "この回答は文脈に基づいていますか" in line:
                continue
            if line not in seen:
                seen.add(line)
                lines_out.append(line)

        # 3) 先頭が「- 」でなければ整形（プロンプト逸脱の保険）
        normed = []
        for l in lines_out:
            if not l.startswith("- "):
                normed.append(f"- {l}")
            else:
                normed.append(l)

        # 4) 上限行数でカット
        normed = normed[:max_lines]
        return "\n".join(normed).strip()

