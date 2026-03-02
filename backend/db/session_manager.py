import sqlite3
import json
import os
from typing import List, Dict, Any

class SessionManager:
    def __init__(self, sessions_dir: str):
        self.sessions_dir = sessions_dir
        os.makedirs(self.sessions_dir, exist_ok=True)

    def _get_db_path(self, session_id: str) -> str:
        return os.path.join(self.sessions_dir, f"{session_id}.db")

    def _ensure_tables(self, session_id: str):
        """Creates tables if they don't exist yet."""
        conn = sqlite3.connect(self._get_db_path(session_id))
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS chat_history (id INTEGER PRIMARY KEY, role TEXT, content TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS game_state (key TEXT PRIMARY KEY, value TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS investigators (name TEXT PRIMARY KEY, data TEXT)''')
        conn.commit()
        conn.close()

    def init_session(self, session_id: str, investigators: List[Dict], language: str):
        self._ensure_tables(session_id)
        conn = sqlite3.connect(self._get_db_path(session_id))
        c = conn.cursor()
        
        # Clear existing data if restarting same session ID
        c.execute('DELETE FROM chat_history')
        c.execute('DELETE FROM game_state')
        c.execute('DELETE FROM investigators')

    def add_message(self, session_id: str, role: str, content: str):
        self._ensure_tables(session_id)
        conn = sqlite3.connect(self._get_db_path(session_id))
        c = conn.cursor()
        c.execute("INSERT INTO chat_history (role, content) VALUES (?, ?)", (role, content))
        conn.commit()
        conn.close()

    def get_history(self, session_id: str, limit: int = 6) -> str:
        self._ensure_tables(session_id)
        conn = sqlite3.connect(self._get_db_path(session_id))
        c = conn.cursor()
        c.execute("SELECT role, content FROM chat_history ORDER BY id DESC LIMIT ?", (limit,))
        rows = c.fetchall()
        conn.close()
        
        # Reverse to chronological order and format
        history = ""
        for role, content in reversed(rows):
            history += f"{role.upper()}: {content}\n\n"
        return history

    def get_language(self, session_id: str) -> str:
        self._ensure_tables(session_id)
        conn = sqlite3.connect(self._get_db_path(session_id))
        c = conn.cursor()
        c.execute("SELECT value FROM game_state WHERE key='language'")
        row = c.fetchone()
        conn.close()
        return row[0] if row else "en"