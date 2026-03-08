# RAG Lambda App

A simple **RAG** (Retrieval-Augmented Generation) API. Upload a **PDF**, then ask questions and get answers from your document using **AWS Bedrock** and **LangChain**.

## What it does

- **Upload PDFs** – Each user has their own session. Your file is turned into a **vector store** (Chroma) so the app can search it.
- **Chat** – Send a question; the app finds relevant chunks from your PDF and uses an **LLM** (Bedrock) to answer.
- **Sessions** – Data is stored **per user/session**. If you don’t use the app for **10 minutes**, your session (and your uploaded data) is removed.

## Tech stack

- **Flask** – Web API
- **LangChain** – RAG pipeline, prompts, chat history
- **AWS Bedrock** – Embeddings and chat model (e.g. Titan, Nova)
- **Chroma** – Vector database (stored on disk per session)
- **PyPDF** – PDF loading

## Setup

1. **Clone** the repo and create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure** – Copy `.env.example` to `.env` and set:
   - `secret_key` – Flask secret
   - AWS credentials (for Bedrock) – e.g. via `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_REGION`
   - Optional: `persist_directory_db`, `BEDROCK_EMBEDDING_MODEL_ID`, `BEDROCK_CHAT_MODEL_ID`, `PORT`

3. **Run** the app:
   ```bash
   python app.py
   ```
   By default it runs on port 3000.

## API

| Endpoint   | Method | Description |
|-----------|--------|-------------|
| `/upload` | POST   | Upload a PDF. Send the file as form field `pdf`. Stored only for your session. |
| `/chat`   | GET/POST | Send JSON `{"question": "your question"}`. Returns `{"answer": "...", "session_id": "..."}`. |

Sessions are identified by cookies (Flask session). Each session has its own vector store and chat history.

## Project structure

- `app.py` – Flask routes, session handling, 10‑minute cleanup
- `core/retreival.py` – RAG logic: PDF → chunks → Chroma, multi-query retrieval, Bedrock chat
- `Ingestion/Document_processing.py` – Batch ingestion from a folder of PDFs (optional)
- `requirements.txt` – Python dependencies
- `.env` – Local config (not committed)

## Keywords

RAG, retrieval-augmented generation, LangChain, AWS Bedrock, Chroma, vector store, embeddings, PDF upload, document QA, Flask API, session-based, LLM, Amazon Titan, Nova.
