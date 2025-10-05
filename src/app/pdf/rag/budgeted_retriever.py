# budgeted_retriever.py
from typing import List, Optional, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from app.pdf.rag.config import MODEL_CONFIG

class BudgetedRetriever(BaseRetriever):
    # --- Pydantic の「フィールド宣言」(必須) ---
    base: BaseRetriever
    tok: Any  # transformers.PreTrainedTokenizerBase でもOK
    budget_tokens: int = 800

    # v2/v1 互換のために model_config か Config を付与
    locals().update(MODEL_CONFIG)

    # 同期版
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        docs = self.base.invoke(
            query, config={"callbacks": run_manager.get_child() if run_manager else None}
        )
        return self._trim_to_budget(docs)

    # 非同期版（必要なら）
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        docs = await self.base.invoke(
            query, config={"callbacks": run_manager.get_child() if run_manager else None}
        )
        return self._trim_to_budget(docs)

    # 予算で切り詰める共通ロジック
    def _trim_to_budget(self, docs: List[Document]) -> List[Document]:
        out: List[Document] = []
        used = 0
        for d in docs:
            ids = self.tok.encode(d.page_content, add_special_tokens=False)
            L = len(ids)
            if used + L > self.budget_tokens:
                remain = self.budget_tokens - used
                if remain > 0:
                    trimmed = self.tok.decode(ids[:remain])
                    out.append(Document(page_content=trimmed, metadata=d.metadata))
                break
            out.append(d)
            used += L
        return out
