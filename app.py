from flask import Flask,request,jsonify,session
import uuid
from dotenv import load_dotenv
import os
from core.retreival import LLM_response_text
from langchain.memory import ChatMessageHistory

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("secret_key")

@app.before_request
def create_session():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())



store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# _______________________________________________________________________________________
@app.route("/chat",methods=['GET'])
def chat():
    session_id = session["session_id"]
    question = request.json.get("question")
    # print(question)
    response = LLM_response_text(question,session_id,get_session_history)
    # print(response)
    return jsonify({"answer":response,"session_id":session_id})
if __name__ == "__main__":
    app.run(port=int(os.getenv("PORT", "3000")))