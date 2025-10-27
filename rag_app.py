import os
from flask import Flask, jsonify, render_template, request
from flask_sock import Sock
from flask_cors import CORS
from sshtunnel import SSHTunnelForwarder
from mysql.connector import pooling, Error
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import time

import rag_pdf_session as chat

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()
app = Flask(__name__)
app.static_folder = 'static'
app.config['SECRET_KEY'] = 'secret much'

# ------------------------------------------
# GLOBAL VARIABLES
# ------------------------------------------
tunnel = None
pool = None


# ------------------------------------------
# FUNCTION: Create or restart SSH Tunnel
# ------------------------------------------
def start_ssh_tunnel():
    global tunnel
    try:
        if tunnel is None or not tunnel.is_active:
            print("🔁 Starting new SSH tunnel...")
            tunnel = SSHTunnelForwarder(
                (os.getenv("SSH_HOST"), 22),
                ssh_username=os.getenv("SSH_USER"),
                ssh_password=os.getenv("SSH_PASS"),
                remote_bind_address=(os.getenv("DB_HOST"), 3306)
            )
            tunnel.start()
            print(f"✅ SSH tunnel active on local port {tunnel.local_bind_port}")
        else:
            print("🟢 SSH tunnel already active")
    except Exception as e:
        print("❌ Failed to start SSH tunnel:", e)
        tunnel = None


# ------------------------------------------
# FUNCTION: Create MySQL Pool
# ------------------------------------------
def create_pool():
    global pool
    try:
        if tunnel is None or not tunnel.is_active:
            start_ssh_tunnel()

        dbconfig = {
            "host": "127.0.0.1",
            "port": tunnel.local_bind_port,
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASS"),
            "database": os.getenv("DB_NAME"),
        }

        pool = pooling.MySQLConnectionPool(
            pool_name="mypool",
            pool_size=5,
            pool_reset_session=True,
            **dbconfig
        )
        print("✅ MySQL connection pool created")
    except Error as e:
        print("❌ Error creating connection pool:", e)
        pool = None


# ------------------------------------------
# FUNCTION: Get a valid MySQL connection
# ------------------------------------------
def get_connection():
    global pool
    try:
        if tunnel is None or not tunnel.is_active:
            start_ssh_tunnel()
            create_pool()

        conn = pool.get_connection()
        if not conn.is_connected():
            print("⚠️ Connection not active, reconnecting...")
            conn.reconnect(attempts=3, delay=2)
        return conn

    except Error as e:
        print("❌ Error getting connection:", e)
        time.sleep(2)
        # Retry sekali lagi
        try:
            start_ssh_tunnel()
            create_pool()
            conn = pool.get_connection()
            return conn
        except Exception as e2:
            print("❌ Second connection attempt failed:", e2)
            return None


# ------------------------------------------
# INITIALIZE ON STARTUP
# ------------------------------------------
start_ssh_tunnel()
create_pool()

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
    conn = None
    cursor = None
    try:
        data = request.get_json()
        sessionID = data.get("sessionID")
        userText = data.get("msg")
        userTime = datetime.now(timezone(timedelta(hours=7))).replace(tzinfo=None)

        conn = get_connection()
        if conn is None:
            return jsonify({"error": "Database connection unavailable"}), 500

        cursor = conn.cursor()

        # Proses pertanyaan ke chatbot
        out = chat.answer(userText, sessionID)
        botTime = datetime.now(timezone(timedelta(hours=7))).replace(tzinfo=None)

        # Simpan percakapan
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
            conn.close()


@app.route("/rating", methods=["POST"])
def save_rating():
    conn = None
    cursor = None
    try:
        data = request.json
        msgID = data.get("msgID")
        rating = data.get("rating")

        conn = get_connection()
        if conn is None:
            return jsonify({"error": "Database connection unavailable"}), 500

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


# ------------------------------------------
# MAIN ENTRY POINT
# ------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
