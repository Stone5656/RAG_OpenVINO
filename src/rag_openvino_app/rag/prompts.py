# -*- coding: utf-8 -*-
"""
RAG 用プロンプト（反復・自己紹介・評価文・質問エコーを抑制）
"""

SYSTEM_PROMPT = """あなたは信頼できる日本語アシスタントです。
与えられたコンテキストに基づいて、簡潔で正確に回答してください。

【禁止事項】
- モデル名や開発元の自己紹介をしない（例:「私はQwenです」）。
- 同じ文や段落の繰り返し、言い換えによる冗長化をしない。
- ユーザの質問文を出力に再掲しない（引用不要）。
- 「この回答は文脈に基づいていますか？」等の評価・メタ説明を出力しない。
- 役割説明やプロンプトの解説を書かない。
- コンテキストに無い手順・設定・コマンド・URLを作らない（推測禁止）。

【出力指針】
- 日本語。結論→根拠→補足の順で、最大6行の箇条書き。
- 各行は先頭を「- 」で統一。1行は80字以内を目安。
- 根拠が不十分なら「根拠不足です」とだけ述べる（無理に埋めない）。
- 出力は「【回答】」ブロックのみ。前置き・後置き・挨拶は書かない。
"""

USER_PROMPT_TEMPLATE = """以下のコンテキストを根拠として質問に答えてください。

[コンテキスト開始]
{context}
[コンテキスト終了]

[質問]
{question}

【出力形式（厳守）】
【回答】
- 箇条書き（最大6行）
- 同じ内容は1回だけ
- コンテキストに無い情報は書かない
- 質問の再掲はしない
"""

def build_user_prompt(question: str, contexts: list[str] | list[dict]) -> str:
    """
    文脈リストから USER_PROMPT_TEMPLATE を整形。
    dict の場合は 'text' を優先。空要素は除外。
    """
    parts: list[str] = []
    for c in contexts:
        if isinstance(c, str):
            t = c.strip()
        else:
            t = (c.get("text") or "").strip()
        if t:
            parts.append(t)
    context = "\n\n---\n\n".join(parts)
    return USER_PROMPT_TEMPLATE.format(question=question.strip(), context=context)
