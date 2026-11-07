"""SQLite storage: documents, chunks, embeddings and an FTS5 keyword index."""
import re
import sqlite3
import threading
from pathlib import Path

import numpy as np

from config.settings import DATABASE_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    original_filename TEXT NOT NULL,
    stored_filename TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    file_extension TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    ingested_at TEXT NOT NULL,
    page_count INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    document_name TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (chunk_id, model)
);
"""

FTS_SCHEMA = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5("
    "text, chunk_id UNINDEXED, tokenize='porter unicode61')"
)

DOC_COLUMNS = (
    "document_id", "original_filename", "stored_filename", "stored_path",
    "file_extension", "file_size_bytes", "sha256", "ingested_at",
)
CHUNK_COLUMNS = (
    "chunk_id", "document_id", "document_name", "page_number",
    "chunk_index", "text", "created_at",
)


def tokenize(text):
    return re.findall(r"[A-Za-z0-9]+", text.lower())


class Database:
    def __init__(self, path=DATABASE_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.fts_enabled = self._init_fts()
        self.conn.commit()

    def _init_fts(self):
        try:
            self.conn.execute(FTS_SCHEMA)
            return True
        except sqlite3.OperationalError:
            return False  # SQLite build without FTS5: fall back to LIKE search

    def close(self):
        with self._lock:
            self.conn.close()

    # ---- documents ------------------------------------------------------
    def get_document_by_sha(self, sha256):
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM documents WHERE sha256 = ?", (sha256,)
            ).fetchone()
        return dict(row) if row else None

    def add_document(self, record):
        with self._lock, self.conn:
            self.conn.execute(
                f"INSERT INTO documents ({','.join(DOC_COLUMNS)}) "
                f"VALUES ({','.join('?' * len(DOC_COLUMNS))})",
                [record[c] for c in DOC_COLUMNS],
            )

    def update_document_counts(self, document_id, page_count, chunk_count):
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE documents SET page_count=?, chunk_count=? WHERE document_id=?",
                (page_count, chunk_count, document_id),
            )

    def list_documents(self):
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM documents ORDER BY ingested_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_document(self, document_id):
        with self._lock, self.conn:
            ids = [r[0] for r in self.conn.execute(
                "SELECT chunk_id FROM chunks WHERE document_id=?", (document_id,))]
            if self.fts_enabled:
                self.conn.executemany(
                    "DELETE FROM chunks_fts WHERE chunk_id = ?", [(i,) for i in ids])
            self.conn.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
            self.conn.execute("DELETE FROM documents WHERE document_id=?", (document_id,))

    # ---- chunks ---------------------------------------------------------
    def add_chunks(self, chunks):
        """Insert chunks that are not stored yet. Returns the newly added ones."""
        added = []
        with self._lock, self.conn:
            for chunk in chunks:
                exists = self.conn.execute(
                    "SELECT 1 FROM chunks WHERE chunk_id=?", (chunk["chunk_id"],)
                ).fetchone()
                if exists:
                    continue
                self.conn.execute(
                    f"INSERT INTO chunks ({','.join(CHUNK_COLUMNS)}) "
                    f"VALUES ({','.join('?' * len(CHUNK_COLUMNS))})",
                    [chunk[c] for c in CHUNK_COLUMNS],
                )
                if self.fts_enabled:
                    self.conn.execute(
                        "INSERT INTO chunks_fts (text, chunk_id) VALUES (?, ?)",
                        (chunk["text"], chunk["chunk_id"]),
                    )
                added.append(chunk)
        return added

    def load_chunks(self, document_id=None):
        query = "SELECT * FROM chunks"
        params = ()
        if document_id:
            query += " WHERE document_id=?"
            params = (document_id,)
        query += " ORDER BY document_id, chunk_index"
        with self._lock:
            return [dict(r) for r in self.conn.execute(query, params)]

    def get_chunks(self, chunk_ids):
        if not chunk_ids:
            return {}
        with self._lock:
            marks = ",".join("?" * len(chunk_ids))
            rows = self.conn.execute(
                f"SELECT * FROM chunks WHERE chunk_id IN ({marks})", list(chunk_ids)
            ).fetchall()
        return {r["chunk_id"]: dict(r) for r in rows}

    def count_chunks(self):
        with self._lock:
            return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    # ---- keyword search -------------------------------------------------
    def keyword_search(self, query, limit=5):
        terms = tokenize(query)
        if not terms:
            return []
        if self.fts_enabled:
            match = " OR ".join(f'"{t}"' for t in dict.fromkeys(terms))
            with self._lock:
                rows = self.conn.execute(
                    "SELECT c.*, bm25(chunks_fts) AS rank FROM chunks_fts "
                    "JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id "
                    "WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
                    (match, limit),
                ).fetchall()
            # bm25() is lower-is-better and negative; expose a positive score.
            return [{**dict(r), "score": -float(r["rank"])} for r in rows]
        return self._like_search(terms, limit)

    def _like_search(self, terms, limit):
        unique = list(dict.fromkeys(terms))
        with self._lock:
            rows = [dict(r) for r in self.conn.execute("SELECT * FROM chunks")]
        scored = []
        for row in rows:
            lowered = row["text"].lower()
            score = sum(lowered.count(t) for t in unique)
            if score:
                scored.append({**row, "score": float(score)})
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:limit]

    # ---- embeddings -----------------------------------------------------
    def chunks_missing_embeddings(self, model):
        with self._lock:
            rows = self.conn.execute(
                "SELECT c.chunk_id, c.text FROM chunks c "
                "LEFT JOIN embeddings e ON e.chunk_id = c.chunk_id AND e.model = ? "
                "WHERE e.chunk_id IS NULL",
                (model,),
            ).fetchall()
        return [dict(r) for r in rows]

    def save_embeddings(self, model, chunk_ids, matrix):
        matrix = np.asarray(matrix, dtype=np.float32)
        with self._lock, self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO embeddings (chunk_id, model, dim, vector) "
                "VALUES (?, ?, ?, ?)",
                [(cid, model, matrix.shape[1], matrix[i].tobytes())
                 for i, cid in enumerate(chunk_ids)],
            )

    def load_embeddings(self, model):
        with self._lock:
            rows = self.conn.execute(
                "SELECT chunk_id, dim, vector FROM embeddings WHERE model=?", (model,)
            ).fetchall()
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32)
        ids = [r["chunk_id"] for r in rows]
        matrix = np.vstack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
        return ids, matrix

    def count_embeddings(self, model):
        with self._lock:
            return self.conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE model=?", (model,)
            ).fetchone()[0]
