from config.settings import RRF_K


class HybridRetriever:
    """Reciprocal Rank Fusion of keyword (BM25) and semantic (cosine) results."""

    def __init__(self, keyword_search, semantic_search, rrf_k=RRF_K):
        self.keyword_search = keyword_search
        self.semantic_search = semantic_search
        self.rrf_k = rrf_k

    def retrieve(self, query, limit=5):
        pool = max(limit * 3, 10)
        keyword_results = self.keyword_search.search(query, limit=pool)
        try:
            semantic_results = self.semantic_search.search(query, limit=pool)
        except Exception:
            semantic_results = []  # keyword search still works if embeddings fail

        fused, chunks = {}, {}
        for results in (keyword_results, semantic_results):
            for rank, result in enumerate(results):
                cid = result["chunk_id"]
                chunks.setdefault(cid, result)
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        ranked = sorted(fused, key=fused.get, reverse=True)[:limit]
        return [{**chunks[cid], "score": fused[cid]} for cid in ranked]
