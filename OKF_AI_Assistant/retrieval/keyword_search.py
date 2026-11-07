class KeywordSearch:
    """BM25 keyword search backed by SQLite FTS5 (LIKE fallback if FTS5 is absent).

    The FTS index is maintained by Database.add_chunks, so it is always current.
    """

    def __init__(self, database):
        self.database = database

    def search(self, query, limit=5):
        return self.database.keyword_search(query, limit=limit)
