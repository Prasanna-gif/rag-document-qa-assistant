import os
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()
APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "data" / "rag_assistant.db"
EMBED_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

st.set_page_config(page_title="RAG Document Assistant", page_icon="📚", layout="wide")
st.title("📚 RAG Document Question-Answering Assistant")
st.caption("Upload documents, retrieve relevant passages, and ask questions grounded in your files.")

def db_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    with db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                uploaded_at TEXT NOT NULL,
                chunk_count INTEGER NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id INTEGER NOT NULL,
                rating TEXT NOT NULL,
                comment TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(query_id) REFERENCES queries(id)
            )
        """)
        conn.commit()

@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

def extract_text(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        pages = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"[Page {page_num}]\\n{text}")
        return "\\n\\n".join(pages)
    if name.endswith((".txt", ".md")):
        raw = uploaded_file.getvalue()
        return raw.decode("utf-8", errors="replace")
    raise ValueError("Supported file types are PDF, TXT, and Markdown (.md).")

def chunk_text(text, chunk_size=900, overlap=150):
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks

def add_documents(uploaded_files):
    embedder = load_embedder()
    new_chunks = []
    names_and_counts = []
    for uploaded in uploaded_files:
        text = extract_text(uploaded)
        chunks = chunk_text(text)
        if not chunks:
            st.warning(f"No extractable text found in {uploaded.name}. Scanned PDFs may need OCR.")
            continue
        names_and_counts.append((uploaded.name, len(chunks)))
        for index, chunk in enumerate(chunks, start=1):
            new_chunks.append({
                "filename": uploaded.name,
                "chunk_index": index,
                "text": chunk,
            })

    if not new_chunks:
        return

    texts = [item["text"] for item in new_chunks]
    vectors = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    vectors = np.asarray(vectors, dtype=np.float32)

    if "chunks" not in st.session_state:
        st.session_state.chunks = []
        st.session_state.vectors = np.empty((0, vectors.shape[1]), dtype=np.float32)

    # If a model dimension changes between sessions, reset the in-memory index safely.
    if st.session_state.vectors.shape[1] != vectors.shape[1]:
        st.session_state.chunks = []
        st.session_state.vectors = np.empty((0, vectors.shape[1]), dtype=np.float32)

    st.session_state.chunks.extend(new_chunks)
    st.session_state.vectors = np.vstack([st.session_state.vectors, vectors])

    with db_connection() as conn:
        for filename, count in names_and_counts:
            conn.execute(
                "INSERT INTO documents(filename, uploaded_at, chunk_count) VALUES (?, ?, ?)",
                (filename, datetime.now().isoformat(timespec="seconds"), count)
            )
        conn.commit()

def retrieve(question, top_k=4):
    if not st.session_state.get("chunks"):
        return []
    embedder = load_embedder()
    query_vector = embedder.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]
    scores = st.session_state.vectors @ np.asarray(query_vector, dtype=np.float32)
    top_indices = np.argsort(scores)[::-1][:min(top_k, len(scores))]
    return [
        {**st.session_state.chunks[int(i)], "score": float(scores[int(i)])}
        for i in top_indices
    ]

def generate_answer(question, passages):
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to your .env file and restart the app."
        )

    client = genai.Client(api_key=api_key)

    context = "\n\n".join(
        f"[Source {i}: {p['filename']} - chunk {p['chunk_index']}]\n{p['text']}"
        for i, p in enumerate(passages, start=1)
    )

    prompt = f"""
You are a helpful document question-answering assistant.
Answer the user's question using only the supplied document passages.
If the passages do not contain the answer, clearly say that the information
is not available in the provided documents.
Cite relevant sources using [Source 1], [Source 2], etc.

DOCUMENT PASSAGES:
{context}

USER QUESTION:
{question}
"""

    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        contents=prompt,
        config={"temperature": 0.1},
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response. Please try again.")

    return response.text

init_db()
if "chunks" not in st.session_state:
    st.session_state.chunks = []
    st.session_state.vectors = np.empty((0, 0), dtype=np.float32)

with st.sidebar:
    st.header("1. Add documents")
    uploaded_files = st.file_uploader(
        "Choose PDF, TXT, or Markdown files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if st.button("Process documents", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Upload at least one document first.")
        else:
            try:
                with st.spinner("Extracting text and creating embeddings…"):
                    add_documents(uploaded_files)
                st.success(f"Processed {len(uploaded_files)} uploaded file(s).")
            except Exception as exc:
                st.error(f"Could not process documents: {exc}")

    st.divider()
    st.header("2. Retrieval settings")
    top_k = st.slider("Passages to retrieve", min_value=1, max_value=8, value=4)
    st.caption("Embeddings and retrieved chunks are kept in this app session. Re-upload documents after restarting the app.")

left, right = st.columns([3, 2])
with left:
    st.subheader("Ask your documents")
    question = st.text_area("Question", placeholder="What are the key points in the document?", height=100)
    ask = st.button("Find answer", type="primary")
    if ask:
        if not question.strip():
            st.warning("Enter a question.")
        elif not st.session_state.chunks:
            st.warning("Upload and process documents first.")
        else:
            try:
                passages = retrieve(question.strip(), top_k=top_k)
                answer = generate_answer(question.strip(), passages)
                sources_text = " | ".join(f"{p['filename']} (chunk {p['chunk_index']})" for p in passages)
                with db_connection() as conn:
                    cursor = conn.execute(
                        "INSERT INTO queries(question, answer, sources, created_at) VALUES (?, ?, ?, ?)",
                        (question.strip(), answer, sources_text, datetime.now().isoformat(timespec="seconds"))
                    )
                    query_id = cursor.lastrowid
                    conn.commit()
                st.session_state.last_result = {
                    "query_id": query_id,
                    "question": question.strip(),
                    "answer": answer,
                    "passages": passages,
                }
            except Exception as exc:
                st.error(f"Could not generate an answer: {exc}")

    result = st.session_state.get("last_result")
    if result:
        st.markdown("### Answer")
        st.write(result["answer"])
        st.markdown("### Retrieved sources")
        for i, passage in enumerate(result["passages"], start=1):
            with st.expander(f"[Source {i}] {passage['filename']} — chunk {passage['chunk_index']} (similarity {passage['score']:.3f})"):
                st.write(passage["text"])
        st.markdown("### Was this answer helpful?")
        c1, c2 = st.columns(2)
        rating = None
        if c1.button("👍 Helpful", key=f"helpful_{result['query_id']}"):
            rating = "helpful"
        if c2.button("👎 Not helpful", key=f"not_helpful_{result['query_id']}"):
            rating = "not_helpful"
        comment = st.text_input("Optional feedback", key=f"comment_{result['query_id']}")
        if rating:
            with db_connection() as conn:
                conn.execute(
                    "INSERT INTO feedback(query_id, rating, comment, created_at) VALUES (?, ?, ?, ?)",
                    (result["query_id"], rating, comment.strip(), datetime.now().isoformat(timespec="seconds"))
                )
                conn.commit()
            st.success("Feedback saved to SQLite.")

with right:
    st.subheader("Session overview")
    st.metric("Document chunks indexed", len(st.session_state.chunks))
    if st.session_state.chunks:
        unique_files = sorted({item["filename"] for item in st.session_state.chunks})
        st.write("**Indexed files**")
        for filename in unique_files:
            st.write(f"• {filename}")
    with db_connection() as conn:
        docs = conn.execute(
            "SELECT filename, uploaded_at, chunk_count FROM documents ORDER BY id DESC LIMIT 10"
        ).fetchall()
        query_count = conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
        feedback_count = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    st.metric("Saved questions", query_count)
    st.metric("Feedback entries", feedback_count)
    if docs:
        st.write("**Recent document uploads (SQLite)**")
        for filename, uploaded_at, count in docs:
            st.caption(f"{filename} · {count} chunks · {uploaded_at}")
    st.info("Privacy note: uploaded text is sent to the configured LLM provider when you ask a question. Do not upload confidential files unless your organization's policies permit it.")
