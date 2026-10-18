# RAG Document Question-Answering Assistant

A portfolio project demonstrating Retrieval-Augmented Generation (RAG): users upload PDF/TXT/Markdown files, the app chunks the text, creates semantic embeddings, retrieves relevant passages, and asks an LLM to answer from those passages. SQLite stores document-upload metadata, question/answer history, and feedback.

## Features

- PDF, TXT, and Markdown upload
- Text extraction and overlapping chunking
- Semantic embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Cosine-similarity retrieval using NumPy
- Context-grounded answers through the OpenAI API
- Inline source labels and expandable retrieved passages
- SQLite persistence for uploads, query history, and feedback
- Simple Streamlit interface

## Architecture

1. **Ingestion:** Extract text from uploaded files.
2. **Chunking:** Split extracted text into overlapping chunks.
3. **Embedding:** Convert chunks into normalized vectors.
4. **Retrieval:** Embed the user's question and rank chunks by cosine similarity.
5. **Generation:** Send the question and top passages to the LLM with instructions to answer only from context.
6. **SQL logging:** Store document metadata, question/answer history, and feedback in SQLite.

## Requirements

- Python 3.10 or newer
- An OpenAI API key for answer generation
- Internet access the first time the sentence-transformers embedding model is downloaded

## Run locally

### Windows PowerShell

```powershell
cd rag_document_qa_assistant
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Add your API key to `.env`, save it, then run:

```powershell
streamlit run app.py
```

### macOS / Linux

```bash
cd rag_document_qa_assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your API key
streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`.

## How to test

1. Upload a text-based PDF or `.txt` document.
2. Click **Process documents**.
3. Ask a question whose answer is explicitly in the document.
4. Check that the answer cites source labels and that relevant passages appear under **Retrieved sources**.
5. Ask something not covered by the document; the assistant should say it could not find enough information.
6. Click helpful/not helpful and verify that a feedback row is saved in SQLite.

Scanned PDFs may not contain extractable text and would require OCR, which is not included in this starter version.

## SQL schema

The app initializes these SQLite tables automatically:

- `documents(id, filename, uploaded_at, chunk_count)`
- `queries(id, question, answer, sources, created_at)`
- `feedback(id, query_id, rating, comment, created_at)`

To inspect the database after running the app, use a SQLite browser or Python's `sqlite3` module. The database is stored at `data/rag_assistant.db` and is excluded from Git.

## Important implementation notes

- The embedding index is kept in Streamlit session state. Re-upload and process documents after restarting the app.
- SQLite stores metadata and conversation logs; the actual document chunks and vectors are not persisted in this starter version.
- OpenAI API usage may incur charges. Keep the API key private and never commit `.env`.
- Do not upload confidential or regulated documents unless permitted by your organization's policy.
- This is a learning/portfolio starter project, not a production-ready multi-user service.

## Resume bullet (use only after you have run and tested it)

**RAG-Based Document Question-Answering Assistant | Python, Streamlit, Sentence Transformers, OpenAI API, SQLite**
- Built a document Q&A application that extracts and chunks PDF/TXT content, creates semantic embeddings, retrieves relevant passages, and generates context-grounded LLM responses with source references.
- Used SQLite to store document metadata, query history, and user feedback; developed a Streamlit UI for document ingestion and answer review.

## Interview explanation

**What is RAG?** RAG combines retrieval with generation. Rather than asking an LLM to answer from its general knowledge alone, the app first retrieves relevant passages from user-provided documents and includes them as context for the model.

**Why chunking?** Documents can be too long to pass to a model in one request. Chunks make retrieval more targeted; overlap helps preserve context across chunk boundaries.

**Why embeddings?** Embeddings represent text as vectors. Similarity between the question vector and chunk vectors helps find semantically relevant text even when wording differs.

**Why SQL?** SQLite provides structured storage for uploaded-file metadata, question history, and feedback, allowing basic auditing and analysis.

**Limitations / next improvements:** persist vectors and chunks, add document deletion and duplicate detection, support OCR, add tests, improve retrieval with a reranker, add authentication, and evaluate answer quality against a test set.
