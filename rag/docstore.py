import sqlite3
import json
from typing import Union, Dict
from langchain_core.documents import Document
from langchain_community.docstore.base import Docstore

class SQLiteDocstore(Docstore):
    """A simple SQLite-backed docstore."""

    def __init__(self, db_path: str = "docstore.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()

    def add(self, texts: Dict[str, Document]) -> None:
        """Add documents to the store."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for k, v in texts.items():
                cursor.execute(
                    "INSERT OR REPLACE INTO documents (id, content, metadata) VALUES (?, ?, ?)",
                    (k, v.page_content, json.dumps(v.metadata))
                )
            conn.commit()

    def search(self, search: str) -> Union[Document, str]:
        """Search for a document by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT content, metadata FROM documents WHERE id = ?", (search,))
            row = cursor.fetchone()
            if row:
                content, metadata_json = row
                metadata = json.loads(metadata_json)
                return Document(page_content=content, metadata=metadata)
            else:
                return f"ID {search} not found."
