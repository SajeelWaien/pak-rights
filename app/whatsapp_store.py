import sqlite3

from pathlib import Path
from uuid import uuid4

DB_PATH = Path("./storage/whatsapp.sqlite3")


def init_whatsapp_db():

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS whatsapp_sessions (
                phone_number_id TEXT NOT NULL,
                wa_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                PRIMARY KEY (phone_number_id, wa_id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS processed_messages (
                message_id TEXT PRIMARY KEY
            )
        """)


def get_or_create_conversation(phone_number_id, wa_id):

    with sqlite3.connect(DB_PATH) as conn:

        conn.execute(
            """
            INSERT OR IGNORE INTO whatsapp_sessions
            (phone_number_id, wa_id, conversation_id)
            VALUES (?, ?, ?)
            """,
            (phone_number_id, wa_id, str(uuid4()))
        )

        row = conn.execute(
            """
            SELECT conversation_id
            FROM whatsapp_sessions
            WHERE phone_number_id = ? AND wa_id = ?
            """,
            (phone_number_id, wa_id)
        ).fetchone()

        return row[0]


def create_new_conversation(phone_number_id, wa_id):

    new_id = str(uuid4())

    with sqlite3.connect(DB_PATH) as conn:

        conn.execute(
            """
            INSERT INTO whatsapp_sessions
            (phone_number_id, wa_id, conversation_id)
            VALUES (?, ?, ?)
            ON CONFLICT(phone_number_id, wa_id)
            DO UPDATE SET conversation_id = excluded.conversation_id
            """,
            (phone_number_id, wa_id, new_id)
        )

    return new_id


def claim_message(message_id):

    with sqlite3.connect(DB_PATH) as conn:

        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO processed_messages
            (message_id) VALUES (?)
            """,
            (message_id,)
        )

        return cursor.rowcount == 1