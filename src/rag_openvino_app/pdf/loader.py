# -*- coding: utf-8 -*-
"""
PDF ローダー

責務:
- PDF からプレーンテキストを抽出し、ページ単位のメタ情報を付与する
- 外部依存に過度に縛られないよう、まずは PyPDF2 を使用（未導入なら警告ログ）

戻り値:
- str: 結合テキスト（ページ区切りに改行）
- dict: メタ情報 {"pages": int, "source": str}

注意:
- 図表やレイアウトを保持しないテキスト抽出です。正確性要件が高い場合は pdfplumber / layoutparser 等の導入を検討してください。
"""
from __future__ import annotations
from pathlib import Path
from rag_openvino_app.utils.logger_utils import with_logger

try:
    import PyPDF2  # type: ignore
    _PYPDF_AVAILABLE = True
except Exception as e:
    _PYPDF_AVAILABLE = False
    _PYPDF_IMPORT_ERR = e


@with_logger("RAG-OpenVINO-APP", env_log_path="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
def load_pdf(path: str | Path, *, logger=None) -> tuple[str, dict]:
    """
    PDF を読み込み、テキストとメタ情報を返す。

    Parameters
    ----------
    path : str | Path
        PDF ファイルパス

    Returns
    -------
    text : str
        全ページのテキスト（改行で連結）
    meta : dict
        {"pages": int, "source": str}
    """
    path = Path(path)
    if not path.exists():
        logger.warning("PDF ローダー: ファイルが見つかりません: %s", path)
        return "", {"pages": 0, "source": str(path)}

    if not _PYPDF_AVAILABLE:
        logger.warning("PDF ローダー: PyPDF2 が未導入のため、内容を読み込めません（%s）", _PYPDF_IMPORT_ERR)
        return "", {"pages": 0, "source": str(path)}

    logger.debug("PDF ローダー: 読み込み開始 -> %s", path)
    texts: list[str] = []
    with path.open("rb") as f:
        reader = PyPDF2.PdfReader(f)
        pages = len(reader.pages)
        for i in range(pages):
            try:
                page = reader.pages[i]
                txt = page.extract_text() or ""
                texts.append(txt.strip())
            except Exception as e:
                logger.debug("PDF ローダー: ページ %d の抽出に失敗（%s）", i, e)
                texts.append("")

    joined = "\n\n".join(texts)
    logger.debug("PDF ローダー: 読み込み完了（%d ページ, 文字数 %d）", len(texts), len(joined))
    return joined, {"pages": len(texts), "source": str(path)}
