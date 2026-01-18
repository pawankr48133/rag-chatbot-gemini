# =========================================================
#                GEMINI RAG CHATBOT (FULL FILE)
# =========================================================

import os
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import streamlit as st

# ===================== LangChain Gemini =====================
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

# ===================== LangChain Core ======================
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings


# ===================== Loaders ==============================
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    DirectoryLoader,
)

# ===================== Text Splitter ========================
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ===================== Vector Store =========================
from langchain_community.vectorstores import Chroma


# =========================================================
#                    CONFIG
# =========================================================

st.set_page_config(page_title="Gemini RAG Chatbot", layout="wide")
st.title("🤖 Gemini RAG Chatbot")

TMP_DIR = Path("data/tmp")
VECTOR_DIR = Path("data/vectorstores")
TMP_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
#                    SIDEBAR
# =========================================================

st.sidebar.header("🔐 Gemini Configuration")

st.session_state.google_api_key = st.sidebar.text_input(
    "Gemini API Key (Free)",
    type="password",
    placeholder="Paste Gemini API key here",
)

st.session_state.temperature = st.sidebar.slider(
    "Temperature", 0.0, 1.0, 0.3
)


# =========================================================
#                    GEMINI MODELS
# =========================================================

def get_llm():
    if not st.session_state.google_api_key:
        raise ValueError("Gemini API key missing")

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # ✅ CORRECT NAME
        google_api_key=st.session_state.google_api_key,
        temperature=st.session_state.temperature,
        convert_system_message_to_human=True,
    )



def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# =========================================================
#                    DOCUMENT PIPELINE
# =========================================================

def load_documents():
    loaders = [
        DirectoryLoader(TMP_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(TMP_DIR, glob="**/*.txt", loader_cls=TextLoader),
        DirectoryLoader(TMP_DIR, glob="**/*.csv", loader_cls=CSVLoader),
        DirectoryLoader(TMP_DIR, glob="**/*.docx", loader_cls=Docx2txtLoader),
    ]
    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    return chunks[:100]   # 🔥 critical for Gemini free tier



def create_vectorstore(chunks, name):
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=str(VECTOR_DIR / name),
    )


# =========================================================
#                    LCEL RAG CHAIN
# =========================================================

def create_rag_chain(retriever):

    prompt = ChatPromptTemplate.from_template(
        """Answer the question using only the context below.

<context>
{context}
</context>

Question: {question}
"""
    )

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | get_llm()
        | StrOutputParser()
    )

    return chain


# =========================================================
#                    UI - VECTORSTORE
# =========================================================

st.subheader("📂 Upload Documents")

uploaded_files = st.file_uploader(
    "Upload PDF, TXT, CSV, DOCX files",
    accept_multiple_files=True,
    type=["pdf", "txt", "csv", "docx"],
)

vectorstore_name = st.text_input("Vectorstore name")

if st.button("Create Vectorstore"):
    if not st.session_state.google_api_key:
        st.error("Please enter Gemini API key")
        st.stop()

    if not uploaded_files or not vectorstore_name:
        st.error("Upload files and enter vectorstore name")
        st.stop()

    # Save uploaded files
    for file in uploaded_files:
        with open(TMP_DIR / file.name, "wb") as f:
            f.write(file.read())

    with st.spinner("Processing documents..."):
        docs = load_documents()
        chunks = split_documents(docs)
        vectordb = create_vectorstore(chunks, vectorstore_name)

        st.session_state.retriever = vectordb.as_retriever(
            search_kwargs={"k": 6}
        )
        st.session_state.chain = create_rag_chain(
            st.session_state.retriever
        )

    st.success("Vectorstore created successfully ✅")


# =========================================================
#                    CHAT UI
# =========================================================

st.divider()
st.subheader("💬 Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

prompt = st.chat_input("Ask something about your documents...")

if prompt:
    if "chain" not in st.session_state:
        st.warning("Create a vectorstore first")
        st.stop()

    with st.spinner("Thinking..."):
        answer = st.session_state.chain.invoke(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": answer})

    st.chat_message("user").write(prompt)
    st.chat_message("assistant").write(answer)
