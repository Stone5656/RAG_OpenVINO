import streamlit as st
import os
import glob
import json
import tempfile
from datetime import datetime
import hashlib
from langchain_community.llms import LlamaCpp
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

# --- 1. 基本設定 & CSS ---
# ページの基本的な設定（タイトル、アイコン、レイアウト）を行います。
st.set_page_config(
    page_title="PocketAI Mentor",
    page_icon="🧠",
    layout="wide"
)

# --- UI改善のためのカスタムCSS ---
# Streamlitの標準デザインを上書きし、よりモダンな見た目にします。
st.markdown("""
<style>
    /* チャットメッセージのスタイル調整 */
    .stChatMessage {
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    /* ボタンのスタイル */
    .stButton>button {
        border-radius: 8px;
        border: 1px solid transparent;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        border-color: #4A90E2;
        color: #4A90E2;
    }
    /* プライマリボタン（読み込むボタン）のスタイル */
    .stButton>button[kind="primary"] {
        background-color: #4A90E2;
    }
    /* サイドバーのスタイル調整 (ダークテーマ) */
    [data-testid="stSidebar"] {
        background-color: #1E1E1E; /* 黒に近いダークグレー */
    }
    /* サイドバー内の全要素のテキスト色を明るい色に設定 */
    [data-testid="stSidebar"] * {
        color: #FAFAFA; /* 可読性の高いオフホワイト */
    }
    /* 参照ソース表示用のコンテナ */
    .source-container {
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        background-color: #FAFAFA;
    }
</style>
""", unsafe_allow_html=True)


# --- 定数定義 ---
# アプリケーションで利用するフォルダパスやモデル名を定義します。
MODELS_DIR = "./models/"
UPLOADED_DOCS_DIR = "./uploaded_docs/"
CONVERSATIONS_DIR = "./conversations/"
VECTORSTORE_DIR = "./faiss_index_cache/"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# --- ディレクトリの自動作成 ---
# 必要なフォルダが存在しない場合に、自動的に作成します。
for dir_path in [MODELS_DIR, UPLOADED_DOCS_DIR, CONVERSATIONS_DIR, VECTORSTORE_DIR]:
    os.makedirs(dir_path, exist_ok=True)


# --- 2. Streamlitコールバックハンドラ ---
# LLMからの回答をリアルタイムでUIに表示（ストリーミング）するためのクラスです。
class StreamlitCallbackHandler(BaseCallbackHandler):
    """LLMからのトークンをリアルタイムでUIに描画するためのコールバックハンドラ"""
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """新しいトークンが生成されるたびに呼び出されるメソッド"""
        # 処理内容: トークンを連結し、UIに表示します。
        self.text += token
        self.placeholder.markdown(self.text + "▌") # カーソル風の記号を追加

    def on_llm_end(self, response, **kwargs) -> None:
        """LLMの応答が完了したときに呼び出されるメソッド"""
        # 処理内容: 最終的なテキストでプレースホルダを更新します。
        self.placeholder.markdown(self.text)


# --- 3. コア機能の関数 ---

@st.cache_resource
def load_embeddings(model_name: str):
    """
    埋め込みモデルをロードし、Streamlitのキャッシュ機能で再利用します。
    処理内容: 指定されたモデル名のHuggingFace埋め込みモデルをロードします。
    """
    with st.spinner("Embeddingモデルをロード中です..."):
        return HuggingFaceEmbeddings(model_name=model_name)

def get_vectorstore_path(pdf_files: list) -> str:
    """
    選択されたPDFファイルの組み合わせから、一意のキャッシュファイルパスを生成します。
    処理内容: ファイル名をソートして結合し、MD5ハッシュを計算してファイル名を決定します。
    これにより、同じPDFの組み合わせの場合は同じキャッシュを利用できます。
    """
    sorted_files = sorted([os.path.basename(f) for f in pdf_files])
    identifier = "".join(sorted_files)
    file_hash = hashlib.md5(identifier.encode()).hexdigest()
    return os.path.join(VECTORSTORE_DIR, f"{file_hash}.faiss")


def create_rag_chain(llm, embeddings, pdf_files: list):
    """
    選択されたPDFからベクトルDBを構築（または読込）し、RAGチェーンを作成します。
    処理内容:
    1. 選択されたPDFの組み合わせに対するキャッシュパスを生成します。
    2. キャッシュが存在すればそれを読み込み、なければPDFを読み込んでベクトルDBを新規作成・保存します。
    3. LLMとベクトルDBを組み合わせたRetrievalQAチェーンを返します。
    """
    vectorstore_path = get_vectorstore_path(pdf_files)

    if not os.path.exists(vectorstore_path):
        with st.spinner("PDFからベクトルデータベースを構築中..."):
            all_documents = []
            for pdf_file in pdf_files:
                loader = PyMuPDFLoader(pdf_file)
                all_documents.extend(loader.load())

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.split_documents(all_documents)
            db = FAISS.from_documents(texts, embeddings)
            db.save_local(vectorstore_path)
            st.success(f"ベクトルDBを「{vectorstore_path}」に保存しました。")
    else:
        with st.spinner(f"キャッシュされたベクトルDBを読み込み中..."):
            db = FAISS.load_local(
                vectorstore_path, embeddings, allow_dangerous_deserialization=True
            )

    retriever = db.as_retriever(search_kwargs={"k": 4})
    
    template = """
### 指示:
以下の「コンテキスト情報」を注意深く読み、その情報だけに基づいて「ユーザーの質問」に日本語で回答してください。
コンテキスト情報に答えがない場合は、「分かりません」と回答してください。あなたの知識は使わないでください。

### コンテキスト情報:
{context}

### ユーザーの質問:
{question}

### 回答:
"""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

# --- 4. UI描画の関数 ---

def draw_sidebar():
    """サイドバーのUI要素を描画する関数"""
    with st.sidebar:
        st.header("🧠 PocketAI Mentor")
        
        with st.expander("⚙️ セットアップ", expanded=True):
            # AIモデル選択
            model_files = glob.glob(os.path.join(MODELS_DIR, "*.gguf"))
            if not model_files:
                st.error(f"`{MODELS_DIR}`にGGUFモデルを配置してください。")
                st.stop()
            model_filenames = [os.path.basename(f) for f in model_files]
            
            # --- セッション状態からモデルのインデックスを取得 ---
            try:
                model_index = model_filenames.index(st.session_state.get("selected_model", model_filenames[0]))
            except ValueError:
                model_index = 0
            selected_model_name = st.selectbox("1. AIモデルを選択", model_filenames, index=model_index)
            
            # PDFアップロード機能
            uploaded_files = st.file_uploader(
                "2. 新しいPDFをアップロード",
                type="pdf",
                accept_multiple_files=True
            )
            
            # アップロードされたファイルを保存する処理
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    filepath = os.path.join(UPLOADED_DOCS_DIR, uploaded_file.name)
                    if not os.path.exists(filepath):
                        with open(filepath, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        st.success(f"「{uploaded_file.name}」を保存しました。")
                    else:
                        st.info(f"「{uploaded_file.name}」は既に存在します。")
            
            st.divider()

            # 保存されたPDFを選択する機能
            stored_pdf_files = glob.glob(os.path.join(UPLOADED_DOCS_DIR, "*.pdf"))
            stored_pdf_filenames = sorted([os.path.basename(f) for f in stored_pdf_files])

            if not stored_pdf_filenames:
                 st.info("AIに参照させるPDFをアップロードしてください。")
            
            selected_pdf_names = st.multiselect(
                "3. 参照するPDFを選択",
                stored_pdf_filenames,
                default=st.session_state.get("selected_pdfs", [])
            )

            # 読込実行ボタン
            if st.button("読み込む", type="primary", use_container_width=True):
                if not selected_pdf_names:
                    st.warning("参照するPDFを1つ以上選択してください。")
                else:
                    load_resources(selected_model_name, selected_pdf_names)
        
        # --- 会話の管理 ---
        with st.expander("📁 会話の管理"):
            if st.button("新しい会話を開始", use_container_width=True):
                st.session_state.messages = []
                st.session_state.qa_chain = None
                st.session_state.selected_model = ""
                st.session_state.selected_pdfs = []
                st.rerun()

            # 会話保存用のUI
            save_name = st.text_input("会話の保存名（拡張子不要）")
            if st.button("現在の会話を保存", use_container_width=True):
                save_conversation(save_name)

            load_conversation()

def save_conversation(filename_prefix: str):
    """現在の会話をコンテキスト情報と共にJSONファイルとして保存する関数"""
    if not st.session_state.get("qa_chain"):
        st.warning("保存する会話がありません。PDFを読み込んで会話を開始してください。")
        return

    if not st.session_state.messages:
        st.warning("メッセージがありません。")
        return

    # 保存するデータ構造を作成
    conversation_data = {
        "model": st.session_state.selected_model,
        "pdfs": st.session_state.selected_pdfs,
        "messages": st.session_state.messages
    }

    # ファイル名を決定
    if filename_prefix:
        filename = f"{filename_prefix}.json"
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{ts}.json"
    
    filepath = os.path.join(CONVERSATIONS_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(conversation_data, f, ensure_ascii=False, indent=2)
    st.success(f"会話を `{filename}` に保存しました。")


def load_conversation():
    """保存された会話を読み込むためのUIを描画し、コンテキスト復元も行う関数"""
    conv_files = sorted(glob.glob(os.path.join(CONVERSATIONS_DIR, "*.json")), reverse=True)
    conv_filenames = [os.path.basename(f) for f in conv_files]
    
    if conv_filenames:
        st.divider()
        selected_conv = st.selectbox("保存した会話を読み込む", conv_filenames)
        if st.button("この会話を読み込む", use_container_width=True):
            filepath = os.path.join(CONVERSATIONS_DIR, selected_conv)
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            model_to_load = data.get("model")
            pdfs_to_load = data.get("pdfs", [])
            messages_to_load = data.get("messages", [])

            # 復元するモデルとPDFが存在するかチェック
            model_path = os.path.join(MODELS_DIR, model_to_load)
            pdf_paths_exist = all(os.path.exists(os.path.join(UPLOADED_DOCS_DIR, p)) for p in pdfs_to_load)

            if not os.path.exists(model_path):
                st.error(f"モデル '{model_to_load}' が見つかりません。")
                return
            if not pdf_paths_exist:
                st.error("保存時のPDFの一部またはすべてが見つかりません。")
                return

            # リソースを再読み込みしてコンテキストを復元
            load_resources(model_to_load, pdfs_to_load)
            # メッセージを復元
            st.session_state.messages = messages_to_load
            st.rerun()

def load_resources(model_name, pdf_names):
    """
    選択されたモデルとPDFを読み込み、RAGチェーンを構築する関数
    処理内容: LLMと埋め込みモデルをロードし、RAGチェーンを作成してセッション状態に保存します。
    """
    with st.spinner("モデルとPDFを読み込んでいます..."):
        model_path = os.path.join(MODELS_DIR, model_name)
        
        # 保存されたPDFのパスを構築
        pdf_paths = [os.path.join(UPLOADED_DOCS_DIR, name) for name in pdf_names]
        
        # RAGチェーンの構築
        # 処理内容: LlamaCppモデルを初期化します。
        llm = LlamaCpp(
            model_path=model_path,
            # ★★ GPU設定 ★★
            # この数値を変更することで、モデルの計算処理をGPUにオフロード（委譲）します。
            # -1 を指定すると、サポートされている全てのレイヤーがGPUで実行されます。
            # GPUのVRAM容量が足りない場合は、この数値を調整（減らす）必要があります。
            # GPUで実行するには、対応するライブラリのインストールが必須です。
            n_gpu_layers=-1,
            n_batch=512,
            n_ctx=4096,
            f16_kv=True,
            verbose=False,
        )
        embeddings = load_embeddings(EMBEDDING_MODEL)
        st.session_state.qa_chain = create_rag_chain(llm, embeddings, pdf_paths)
        
        # セッション状態の更新
        st.session_state.selected_model = model_name
        st.session_state.selected_pdfs = pdf_names
        st.session_state.messages = []
        
    st.success("読み込みが完了しました。")

def draw_welcome_message():
    """アプリケーションの初期画面を描画する関数"""
    st.header("ようこそ！ PocketAI Mentorへ")
    st.markdown("""
    このアプリケーションは、あなたが指定したPDFファイルの内容について、AIと対話形式で質問できるツールです。

    **利用方法:**
    1.  左のサイドバーにある **`⚙️ セットアップ`** を開きます。
    2.  使用するAIモデルを選択します。
    3.  **「新しいPDFをアップロード」** エリアにPDFをドラッグ＆ドロップして、AIに参照させたいファイルを保管します。
    4.  保管されたPDFの中から、**参照したいPDFを選択**します。(複数可)
    5.  **`読み込む`** ボタンを押して、AIの準備が完了するのを待ちます。
    6.  準備が完了するとチャット画面が表示されるので、質問を入力してください。

    全ての処理はあなたのPC内で完結するため、機密情報を含むPDFも安全に扱うことができます。
    """)

# --- 5. メイン処理 ---

# --- セッション状態の初期化 ---
# アプリケーション全体で利用する変数をst.session_stateに初期化します。
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = ""
if "selected_pdfs" not in st.session_state:
    st.session_state.selected_pdfs = []

# --- サイドバーの描画 ---
draw_sidebar()

# --- メイン画面の描画 ---
# RAGチェーンが準備できているかで、表示する画面を切り替えます。
if st.session_state.qa_chain:
    # --- 現在のステータス表示 ---
    # 選択中のモデルとPDFをヘッダーに表示します。
    st.markdown(
        f"**🧠 モデル:** `{st.session_state.selected_model}` "
        f"| **📚 参照PDF:** `{'`, `'.join(st.session_state.selected_pdfs)}`"
    )
    st.divider()

    # --- チャット履歴の表示 ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- ユーザーからの質問入力 ---
    if prompt := st.chat_input("PDFに関する質問を入力してください"):
        # ユーザーの質問を履歴に追加して表示
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- AIの回答を生成して表示 ---
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            handler = StreamlitCallbackHandler(message_placeholder)
            
            with st.spinner("🤖 回答を生成中..."):
                result = st.session_state.qa_chain.invoke(
                    prompt, 
                    config={"callbacks": [handler]}
                )
            
            response = result.get('result', "結果を取得できませんでした。").strip()
            st.session_state.messages.append({"role": "assistant", "content": response})

            # --- 参照ソースの表示 ---
            # 回答後に、参照したPDFの箇所を分かりやすく表示します。
            with st.expander("参照ソースを表示"):
                for doc in result.get('source_documents', []):
                    with st.container(border=True):
                        page_num = doc.metadata.get('page', 'N/A')
                        source_file = doc.metadata.get('source', 'N/A')
                        st.markdown(
                            f"**ファイル:** `{os.path.basename(source_file)}` | **ページ:** `{page_num + 1}`"
                        )
                        # 引用箇所を抜粋して表示
                        st.caption(doc.page_content[:200].replace('\n', ' ') + "...")

else:
    # --- 初期画面の表示 ---
    draw_welcome_message()

