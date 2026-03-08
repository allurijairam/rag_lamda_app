"""
Flask application for RAG-based chat API.

Exposes /chat (RAG Q&A) and /upload (PDF upload stored in memory for 10 minutes).
Maintains session history and optional per-session temporary vector store.
"""

import os
import uuid

from dotenv import load_dotenv
from flask import Flask, jsonify, request, session
from langchain.memory import ChatMessageHistory

from core.retreival import LLM_response_text

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("secret_key")

# In-memory store of ChatMessageHistory per session_id (same process only).
store = {}


@app.before_request
def create_session():
    """Ensure each request has a session_id; create one if missing."""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())


def get_session_history(session_id):
    """
    Return the ChatMessageHistory for the given session_id, creating it if needed.

    Args:
        session_id: Unique identifier for the conversation session.

    Returns:
        ChatMessageHistory instance for this session.
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


@app.route("/chat", methods=["GET"])
def chat():
    """
    Handle a chat request: run the user question through RAG and return the answer.

    Expects JSON body with "question" key. Returns JSON with "answer" and "session_id".
    """
    session_id = session["session_id"]
    question = request.json.get("question")
    response = LLM_response_text(question, session_id, get_session_history)
    return jsonify({"answer": response, "session_id": session_id})


if __name__ == "__main__":
    app.run(port=int(os.getenv("PORT", "3000")))
