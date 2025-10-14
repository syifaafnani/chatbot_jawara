import os
from flask import Flask, jsonify, render_template, request, redirect, url_for, session
from flask_sock import Sock
from flask_cors import CORS
import mysql.connector
from datetime import datetime
from dotenv import load_dotenv

import rag_pdf_session as chat

load_dotenv()
app = Flask(__name__)
app.static_folder = 'static'
app.config['SECRET_KEY'] = 'secret much'

db = mysql.connector.connect(
	host = os.getenv("DB_HOST"),
	user = os.getenv("DB_USER"),
	password = os.getenv("DB_PASS"),
	database = os.getenv("DB_NAME")
)

cursor = db.cursor(dictionary=True)
sock = Sock(app)
CORS(app)

@app.route("/")
def home():
	return render_template("index.html")

@app.route("/get", methods=['POST'])
def get_bot_response():
	data = request.get_json()
	sessionID = data.get("sessionID")
	userText = data.get("msg")
	userTime = datetime.now()

	out = chat.answer(userText, sessionID)
	botTime = datetime.now()

	# Simpan ke database
	cursor.execute(
		"""
		INSERT INTO conversations (session_id, user_msg, user_time, bot_msg, bot_time) 
		VALUES (%s, %s, %s, %s, %s)
		""",
		(sessionID, userText, userTime, out['answer'], botTime)
	)
	db.commit()

	msgID = cursor.lastrowid
	out["msgID"] = msgID

	return jsonify(out)

@app.route("/rating", methods=["POST"])
def save_rating():
    data = request.json
    msgID = data.get("msgID")
    rating = data.get("rating")

    cursor.execute(
        "UPDATE conversations SET rating=%s WHERE id=%s", (rating, msgID)
    )
    db.commit()

    return jsonify({"status": "ok"})

if __name__ == "__main__":
	app.run(host="0.0.0.0",port=5000)
