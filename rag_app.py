import os
from flask import Flask, jsonify, render_template, request
from flask_sock import Sock
from flask_cors import CORS
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

import rag_pdf_session as chat
from db_models import SessionLocal, save_chat, get_chat_history, update_rating

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()
app = Flask(__name__)
app.static_folder = 'static'
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

# ------------------------------------------
# INITIALIZE ON STARTUP
# ------------------------------------------
sock = Sock(app)
CORS(app)

# ------------------------------------------
# ROUTES
# ------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get", methods=['POST'])
def get_bot_response():
    db = SessionLocal()

    try:
        data = request.get_json()
        sessionID = data.get("sessionID")
        userText = data.get("msg")
        userTime = datetime.now(timezone(timedelta(hours=7))).replace(tzinfo=None)

        # Proses pertanyaan ke chatbot
        out = chat.answer(userText, sessionID)
        botTime = datetime.now(timezone(timedelta(hours=7))).replace(tzinfo=None)

        msgID = save_chat(db, sessionID, userText, userTime, out['answer'], botTime)
        db.commit()

        out["msgID"] = msgID

        return jsonify(out)

    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500

    finally:
        db.close()


@app.route("/rating", methods=["POST"])
def give_rating():
    db = SessionLocal()

    try:
        data = request.json
        msgID = data.get("msgID")
        rating = data.get("rating")

        update_rating(db, msgID, rating)

        db.commit()

        return jsonify({"message": "Rating berhasil disimpan"})

    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500

    finally:
        db.close()

# ------------------------------------------
# MAIN ENTRY POINT
# ------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
