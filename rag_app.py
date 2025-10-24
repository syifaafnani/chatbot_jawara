import os
from flask import Flask, jsonify, render_template, request
from flask_sock import Sock
from flask_cors import CORS
from sshtunnel import SSHTunnelForwarder
from mysql.connector import pooling, Error
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

import rag_pdf_session as chat

load_dotenv()
app = Flask(__name__)
app.static_folder = 'static'
app.config['SECRET_KEY'] = 'secret much'

# -------------------------------
# SSH Tunnel (dibuka sekali saja)
# -------------------------------
tunnel = SSHTunnelForwarder(
    (os.getenv('SSH_HOST'), 22),
    ssh_username=os.getenv('SSH_USER'),
    ssh_password=os.getenv('SSH_PASS'),
    remote_bind_address=(os.getenv('DB_HOST'), 3306)
)
tunnel.start()

# ------------------------------------------
# MySQL Connection Pool (reusable connection)
# ------------------------------------------
try:
    dbconfig = {
        "host": os.getenv('DB_HOST'),
        "port": tunnel.local_bind_port,
        "user": os.getenv('DB_USER'),
        "password": os.getenv('DB_PASS'),
        "database": os.getenv('DB_NAME')
    }

    pool = pooling.MySQLConnectionPool(
        pool_name="mypool",
        pool_size=5,
        pool_reset_session=True,
        **dbconfig
    )

except Error as e:
    print("❌ Error while creating connection pool:", e)
    exit(1)

sock = Sock(app)
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get", methods=['POST'])
def get_bot_response():
    conn = None
    cursor = None
    try:
        data = request.get_json()
        sessionID = data.get("sessionID")
        userText = data.get("msg")
        userTime = datetime.now(timezone(timedelta(hours=7))).replace(tzinfo=None)

        # ambil koneksi dari pool
        conn = pool.get_connection()
        cursor = conn.cursor()

        out = chat.answer(userText, sessionID)
        botTime = datetime.now(timezone(timedelta(hours=7))).replace(tzinfo=None)

        # simpan ke database
        cursor.execute(
            """
            INSERT INTO conversations (session_id, user_msg, user_time, bot_msg, bot_time) 
            VALUES (%s, %s, %s, %s, %s)
            """,
            (sessionID, userText, userTime, out['answer'], botTime)
        )
        conn.commit()

        msgID = cursor.lastrowid
        out["msgID"] = msgID

        return jsonify(out)

    except Error as e:
        print("❌ error /get:", e)
        return jsonify({"error": str(e)}), 500

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()  # kembali ke pool, tidak benar-benar menutup koneksi


@app.route("/rating", methods=["POST"])
def save_rating():
    conn = None
    cursor = None
    try:
        data = request.json
        msgID = data.get("msgID")
        rating = data.get("rating")

        conn = pool.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE conversations SET rating=%s WHERE id=%s", (rating, msgID)
        )
        conn.commit()

        return jsonify({"status": "ok"})

    except Error as e:
        print("❌ error /rating:", e)
        return jsonify({"error": str(e)}), 500

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
