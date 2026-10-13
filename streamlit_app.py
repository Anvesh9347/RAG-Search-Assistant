import os
import tempfile
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv(Path(__file__).resolve().parent / ".env")

st.set_page_config(page_title="RAG Search", page_icon="🧠", layout="wide")


def _set_api_key() -> None:
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    elif "OPENAI_API_KEY" in os.environ:
        st.sidebar.caption("Loaded from .env")


def _load_document_from_url(url: str) -> List[Document]:
    loader = WebBaseLoader(url)
    return loader.load()


def _load_document_from_file(file) -> List[Document]:
    suffix = Path(file.name).suffix.lower()
    if suffix == ".pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.getvalue())
            temp_path = temp_file.name
        try:
            return PyPDFLoader(temp_path).load()
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    if suffix in {".txt", ".md"}:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(file.getvalue())
            temp_path = temp_file.name
        try:
            return TextLoader(temp_path, encoding="utf-8").load()
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    raise ValueError(f"Unsupported file type: {file.name}")


def _load_default_data_sources() -> List[Document]:
    docs: List[Document] = []
    data_dir = Path(__file__).resolve().parent / "data"
    if not data_dir.exists():
        return docs

    url_file = data_dir / "url.txt"
    if url_file.exists():
        for line in url_file.read_text(encoding="utf-8").splitlines():
            url = line.strip()
            if url:
                try:
                    docs.extend(_load_document_from_url(url))
                except Exception:
                    continue

    for pdf_path in sorted(data_dir.glob("*.pdf")):
        try:
            docs.extend(PyPDFLoader(str(pdf_path)).load())
        except Exception:
            continue

    for text_path in sorted(data_dir.glob("*.txt")):
        if text_path.name == "url.txt":
            continue
        try:
            docs.extend(TextLoader(str(text_path), encoding="utf-8").load())
        except Exception:
            continue

    return docs


@st.cache_resource
def build_knowledge_base(uploaded_files, urls: str):
    docs: List[Document] = _load_default_data_sources()

    for uploaded_file in uploaded_files or []:
        docs.extend(_load_document_from_file(uploaded_file))

    for raw_url in [line.strip() for line in urls.splitlines() if line.strip()]:
        docs.extend(_load_document_from_url(raw_url))

    if not docs and not uploaded_files and not urls.strip():
        raise ValueError("Please upload a file or add a URL first.")

    if not docs:
        raise ValueError("No readable documents were found.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return vectorstore, llm


def ask_question(vectorstore, llm, question: str):
    docs = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = f"""Answer the question using only the provided context.

Context:
{context}

Question:
{question}
"""
    response = llm.invoke(prompt)
    return response.content, docs


def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "rag_data" not in st.session_state:
        st.session_state.rag_data = None


def main():
    init_session_state()
    st.title("RAG Search Assistant")
    st.caption("Ask questions about your uploaded files or website sources.")

    _set_api_key()

    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
        st.warning("Enter your OpenAI API key in the sidebar to continue.")
        return

    with st.sidebar:
        st.header("Knowledge Base")
        uploaded_files = st.file_uploader(
            "Upload PDFs or text files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )
        urls = st.text_area(
            "Website URLs (one per line)",
            placeholder="https://example.com\nhttps://docs.example.com",
            height=180,
        )
        build_button = st.button("Build Knowledge Base", use_container_width=True)

    if build_button:
        try:
            st.session_state.rag_data = build_knowledge_base(uploaded_files, urls)
            st.success("Knowledge base built successfully.")
        except Exception as exc:
            st.error(f"Failed to build knowledge base: {exc}")
            st.session_state.rag_data = None

    if st.session_state.rag_data is not None:
        vectorstore, llm = st.session_state.rag_data
        question = st.chat_input("Ask a question about your documents")

        if question:
            with st.spinner("Thinking..."):
                answer, sources = ask_question(vectorstore, llm, question)

            st.session_state.history.append({"role": "user", "content": question})
            st.session_state.history.append({"role": "assistant", "content": answer})

            for entry in st.session_state.history:
                if entry["role"] == "user":
                    st.chat_message("user").write(entry["content"])
                else:
                    st.chat_message("assistant").write(entry["content"])

            if sources:
                with st.expander("View sources"):
                    for idx, doc in enumerate(sources, start=1):
                        st.markdown(f"### Source {idx}")
                        st.write(doc.page_content[:1200])
                        if getattr(doc, "metadata", None):
                            st.caption(doc.metadata)


if __name__ == "__main__":
    main()

