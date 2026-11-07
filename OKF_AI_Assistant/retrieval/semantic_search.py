import numpy as np

from config.settings import MIN_SEMANTIC_SCORE


class SemanticSearch:
    """Embedding search with incremental indexing (only new chunks are embedded)."""

    def __init__(self, database, embedder, min_score=MIN_SEMANTIC_SCORE, batch_size=64):
        self.database = database
        self.embedder = embedder
        self.min_score = min_score
        self.batch_size = batch_size
        self._cache = None  # (count, ids, matrix)

    def index_new_chunks(self, progress=None):
        """Embed chunks that have no vector for the current model. Returns count."""
        missing = self.database.chunks_missing_embeddings(self.embedder.name)
        done = 0
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start:start + self.batch_size]
            vectors = self.embedder.encode([c["text"] for c in batch])
            self.database.save_embeddings(
                self.embedder.name, [c["chunk_id"] for c in batch], vectors)
            done += len(batch)
            if progress:
                progress(done, len(missing))
        if done:
            self._cache = None
        return done

    def _matrix(self):
        count = self.database.count_embeddings(self.embedder.name)
        if self._cache is None or self._cache[0] != count:
            ids, matrix = self.database.load_embeddings(self.embedder.name)
            self._cache = (count, ids, matrix)
        return self._cache[1], self._cache[2]

    def search(self, query, limit=5):
        ids, matrix = self._matrix()
        if not ids or not query.strip():
            return []
        query_vector = self.embedder.encode([query])[0]
        scores = matrix @ query_vector  # vectors are L2-normalised -> cosine
        order = np.argsort(-scores)[:limit]
        order = [i for i in order if scores[i] >= self.min_score]
        chunks = self.database.get_chunks([ids[i] for i in order])
        return [
            {**chunks[ids[i]], "score": float(scores[i])}
            for i in order if ids[i] in chunks
        ]
