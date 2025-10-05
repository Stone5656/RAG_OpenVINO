import os
import time

# --- 必要なライブラリ ---
try:
    from langchain_community.llms import LlamaCpp
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
except ImportError:
    print("必要なライブラリがインストールされていません。")
    print("仮想環境を有効化して、pip install ... を実行してください。")
    exit()


# --- 1. 設定 ---
# 参照sしたいPDFファイルへのパス
PDF_PATH = "./docs/無題のドキュメント.pdf"
# ベクターストア（データベース）を保存する場所
VECTORSTORE_PATH = "faiss_index"
# ダウンロードしたGGUFモデルへのパス
MODEL_PATH = "./models/ELYZA-japanese-Llama-2-7b-fast-instruct-q2_K.gguf"
# 日本語に強い埋め込みモデル
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"


# --- 2. PDFからベクターストアを作成する関数 ---
def create_vectorstore():
    """PDFを読み込み、FAISSベクターストアを作成して保存する"""
    if not os.path.exists(PDF_PATH):
        print(f"エラー: PDFファイルが見つかりません: {PDF_PATH}")
        return False
        
    print("--- ステップ1: PDFからベクターストアを構築します ---")
    
    print("PDFをロード中...")
    loader = PyMuPDFLoader(PDF_PATH)
    documents = loader.load()
    
    print("テキストをチャンクに分割中...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    print("埋め込みモデルをロード中...（初回は時間がかかります）")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    print("テキストをベクトル化してインデックスを作成中...")
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(VECTORSTORE_PATH)
    
    print(f"--- ベクターストアの構築完了！ '{VECTORSTORE_PATH}'に保存しました ---")
    return True

# --- 3. メインの実行部分 ---
def main():
    """メインの実行処理"""
    # ベクターストアが存在しない場合のみ作成する
    if not os.path.exists(VECTORSTORE_PATH):
        if not create_vectorstore():
            return # PDFがない場合は終了
            
    # --- モデルとRAGパイプラインの準備 ---
    print("\n--- ステップ2: モデルとRAGパイプラインを準備します ---")
    
    print("埋め込みモデルをロード中...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    print("既存のベクターストアをロード中...")
    db = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 4}) 

    print("LLMをロード中...（PCの性能によっては数分かかります）")
    if not os.path.exists(MODEL_PATH):
        print(f"エラー: モデルファイルが見つかりません: {MODEL_PATH}")
        return
        
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_gpu_layers=0, # GPUを使わない場合は0、NVIDIA GPUは-1、AMD Radeonは-1（要・専用インストール）
        n_batch=512,
        n_ctx=8192,
        f16_kv=True,
        verbose=False,
    )
    print("--- モデルのロード完了！ ---")

    # プロンプトのテンプレートを定義
    template = """
### Instruction:
以下の「コンテキスト情報」を注意深く読み、その情報だけに基づいて「ユーザーの質問」に日本語で回答してください。コンテキスト情報に答えがない場合は、「分かりません」と回答してください。

### Context:
{context}

### Question:
{question}

### Response:
"""

    # プロンプトテンプレートを作成
    prompt = PromptTemplate(
        template=template, input_variables=["context", "question"]
    )
    
    # RAGチェーンを作成 (chain_type_kwargsでカスタムプロンプトを指定)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    print("\n準備が完了しました。PDFに関する質問を入力してください。")
    print("終了するには 'exit' または 'quit' と入力してください。")

    # --- 質問応答ループ ---
    while True:
        query = input("\n[質問] > ")
        if query.lower() in ['exit', 'quit']:
            break
        if not query.strip():
            continue

        start_time = time.time()
        print("🤖 回答を生成中...")
        
        result = qa_chain.invoke(query)
        
        end_time = time.time()

        print("\n--- 回答 ---")
        print(result['result'].strip())
        print("-----------")
        print(f"（生成時間: {end_time - start_time:.2f}秒）")
        
        # --- 参照ソース ---
        print("\n--- 参照ソース ---")
        for doc in result['source_documents']:
            content_preview = doc.page_content[:150].strip().replace('\n', ' ')
            page_number = doc.metadata.get('page', 'N/A')
            print(f"📄 ページ: {page_number}, 内容: {content_preview}...")
        print("------------------")

if __name__ == "__main__":
    main()