from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, scoped_session
import os
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DB_URL")

engine = create_engine(
    DB_URL,
    pool_size=5,        # koneksi utama
    max_overflow=10,    # tambahan saat spike
    pool_timeout=30,
    pool_recycle=1800,
    echo=False          # ubah True kalau mau debug SQL
)

SessionLocal = scoped_session(sessionmaker(bind=engine))

# =========================
# DB FUNCTIONS
# =========================

def save_chat(db, session_id, user_msg, user_time, bot_msg, bot_time):
    result = db.execute(
        text("""
            INSERT INTO chatbot (session_id, user_msg, user_time, bot_msg, bot_time)
            VALUES (:session_id, :user_msg, :user_time, :bot_msg, :bot_time)
            RETURNING id
        """),
    {
        "session_id": session_id,
        "user_msg": user_msg,
        "user_time": user_time,
        "bot_msg": bot_msg,
        "bot_time": bot_time
    })
    return result.scalar()

def get_chat_history(db, session_id, limit=5):
    result = db.execute(
        text("""
            SELECT user_msg, bot_msg
            FROM chatbot
            WHERE session_id = :session_id
            ORDER BY bot_time DESC
            LIMIT :limit
        """),
        {
            "session_id": session_id,
            "limit": limit
        }
    )
    return result.fetchall()

def update_rating(db, chat_id, rating):
    db.execute(
        text("""
            UPDATE chatbot
            SET rating = :rating
            WHERE id = :id
        """),
        {
            "rating": rating,
            "id": chat_id
        }
    )