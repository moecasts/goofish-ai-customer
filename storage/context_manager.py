"""对话历史与商品缓存管理（SQLite）。"""

import json
import sqlite3


class ContextManager:
    def __init__(self, db_path: str = "data/chat_history.db", max_history: int = 100):
        self.db_path = db_path
        self.max_history = max_history
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        cur = self.conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_chat_id ON messages (chat_id);
            CREATE INDEX IF NOT EXISTS idx_timestamp ON messages (timestamp);

            CREATE TABLE IF NOT EXISTS chat_bargain_counts (
                chat_id TEXT PRIMARY KEY,
                count INTEGER DEFAULT 0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS items (
                item_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                price REAL,
                description TEXT,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.commit()

    def add_message(
        self, chat_id: str, user_id: str, item_id: str, role: str, content: str
    ):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO messages (chat_id, user_id, item_id, role, content) VALUES (?, ?, ?, ?, ?)",
            (chat_id, user_id, item_id, role, content),
        )
        self.conn.commit()
        self._cleanup(chat_id)

    def _cleanup(self, chat_id: str):
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM messages WHERE chat_id = ?", (chat_id,))
        count = cur.fetchone()[0]
        if count > self.max_history:
            excess = count - self.max_history
            cur.execute(
                "DELETE FROM messages WHERE id IN ("
                "  SELECT id FROM messages WHERE chat_id = ? ORDER BY timestamp ASC LIMIT ?"
                ")",
                (chat_id, excess),
            )
            self.conn.commit()

    def get_context(self, chat_id: str) -> list[dict]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY timestamp ASC",
            (chat_id,),
        )
        return [
            {"role": row["role"], "content": row["content"]} for row in cur.fetchall()
        ]

    def get_bargain_count(self, chat_id: str) -> int:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT count FROM chat_bargain_counts WHERE chat_id = ?", (chat_id,)
        )
        row = cur.fetchone()
        return row["count"] if row else 0

    def set_bargain_count(self, chat_id: str, count: int):
        """直接设置议价次数（用于从 LangGraph 同步状态）。"""
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO chat_bargain_counts (chat_id, count, last_updated) VALUES (?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(chat_id) DO UPDATE SET count = ?, last_updated = CURRENT_TIMESTAMP",
            (chat_id, count, count),
        )
        self.conn.commit()

    def increment_bargain_count(self, chat_id: str):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO chat_bargain_counts (chat_id, count, last_updated) VALUES (?, 1, CURRENT_TIMESTAMP) "
            "ON CONFLICT(chat_id) DO UPDATE SET count = count + 1, last_updated = CURRENT_TIMESTAMP",
            (chat_id,),
        )
        self.conn.commit()

    def save_item(self, item_id: str, data: dict, price: float, description: str):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO items (item_id, data, price, description) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(item_id) DO UPDATE SET data = ?, price = ?, description = ?, last_updated = CURRENT_TIMESTAMP",
            (
                item_id,
                json.dumps(data, ensure_ascii=False),
                price,
                description,
                json.dumps(data, ensure_ascii=False),
                price,
                description,
            ),
        )
        self.conn.commit()

    def get_item(self, item_id: str) -> dict | None:
        cur = self.conn.cursor()
        cur.execute("SELECT data FROM items WHERE item_id = ?", (item_id,))
        row = cur.fetchone()
        return json.loads(row["data"]) if row else None

    def close(self):
        self.conn.close()
