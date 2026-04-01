"""
Flask application for RAG-based chat API.

Exposes /chat (RAG Q&A) and /upload (PDF upload stored per session).
Each user/session has a separate vector store; sessions inactive for 10 min are removed.
"""

import os
import uuid
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv
from flask import Flask, jsonify, request, session
from langchain.memory import ChatMessageHistory

from core.retreival import (
    LLM_response_text,
    save_pdf_to_vectorstore,
    get_vectorstore_for_session,
    delete_session_store,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)
app = Flask(__name__)
app.secret_key = os.getenv("secret_key")

# Chat history per session_id.
store = {}

# Per-session vector store and last-active time: { session_id: [vectorstore, datetime] }
# Sessions inactive for 10 minutes are removed (in-memory and persisted Chroma).
DB_store = {}
SESSION_TIMEOUT_MINUTES = 1

UPLOAD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")


def delete_session_upload_files(session_id):
    """Delete any temp upload files in uploads/ and delete the vector store for this session (filenames start with session_id_)."""
    if not os.path.isdir(UPLOAD_PATH):
        return
    prefix = f"{session_id}_"
    for name in os.listdir(UPLOAD_PATH):
        if name.startswith(prefix):
            try:

                os.remove(os.path.join(UPLOAD_PATH, name))
                logging.info(f"Deleted upload file {name}")
            except OSError:
                logging.error(f"Error deleting upload file {name}")

 

def cleanup_inactive_sessions():
    """Remove sessions inactive for more than SESSION_TIMEOUT_MINUTES from DB_store, disk, and temp uploads."""
    cutoff = datetime.now() - timedelta(minutes=SESSION_TIMEOUT_MINUTES)
    to_remove = [sid for sid, (_, last) in DB_store.items() if last < cutoff]
    for sid in to_remove:
        del DB_store[sid]
        delete_session_store(sid)
        delete_session_upload_files(sid)
        logging.info(f"Deleted session {sid}")

@app.before_request
def create_session():
    """Ensure each request has a session_id; create one if missing. Clean up inactive sessions."""
    # if "session_id" not in session:
    #     session["session_id"] = str(uuid.uuid4())
    logging.info(f"Creating session id")
    cleanup_inactive_sessions()


def get_session_history(session_id):
    """
    Return the ChatMessageHistory for the given session_id, creating     it if needed.

    Args:
        session_id: Unique identifier for the conversation session.

    Returns:
        ChatMessageHistory instance for this session.
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

@app.route("/upload", methods=["POST"])
def upload():
    """
    Upload a PDF file and store it in this session's vector store (separate per user).
    """
    session["session_id"]  = request.form.get("session_id") 
    logging.info(f"Uploading PDF for session {session['session_id']}")
    if not session["session_id"]:
        return jsonify({"error": "No session ID provided {}".format(session["session_id"])}), 400
    session_id = session["session_id"]
    pdf = request.files.get("pdf")
    if not pdf:
        logging.info(f"No PDF file provided for session {session['session_id']}")
        return jsonify({"error": "No PDF file provided"}), 400
    # Save to a temp path unique to this session to avoid collisions
    safe_name = pdf.filename or "upload.pdf"
    os.makedirs(UPLOAD_PATH, exist_ok=True)
    file_path = os.path.join(UPLOAD_PATH, f"{session_id}_{safe_name}")
    try:
        pdf.save(file_path)
        vectorstore = save_pdf_to_vectorstore(file_path, session_id)
        DB_store[session_id] = [vectorstore, datetime.now()]
        logging.info(f"PDF uploaded successfully for session {session['session_id']}")
        return jsonify({"message": "PDF uploaded successfully", "session_id": session_id}), 200
    finally:
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass

@app.route("/chat", methods=["GET", "POST"])
def chat():
    """
    Handle a chat request: run the user question through RAG and return the answer.
    Uses this session's vector store if available (from upload or restored from disk).
    """
    
    json_data = request.get_json(silent=True) or {}
    session["session_id"]  = json_data.get("session_id")
    if not session["session_id"]:
        logging.info(f"No session ID provided {session['session_id']}")
        return jsonify({"error": "No session ID provided {}".format(session["session_id"])}), 400
    session_id = session["session_id"]
    # Restore session's vector store from disk if not in memory (e.g. after restart)
    if session_id not in DB_store:
        vs = get_vectorstore_for_session(session_id)
        if vs is not None:
            DB_store[session_id] = [vs, datetime.now()]
    # Update last-active when we have a store; otherwise RAG has no context
    if session_id in DB_store:
        DB_store[session_id][1] = datetime.now()

    json_data = request.get_json(silent=True) or {}
    question = json_data.get("question")
    logging.info(f"Question for session {session['session_id']} =====> {question}")
    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400
    response = LLM_response_text(question, session_id, get_session_history, DB_store=DB_store)
    logging.info(f"Chat response for session {session['session_id']} {response}")

    return jsonify({"answer": response, "session_id": session_id})


if __name__ == "__main__":
    app.run(port=int(os.getenv("PORT", "3000")))
