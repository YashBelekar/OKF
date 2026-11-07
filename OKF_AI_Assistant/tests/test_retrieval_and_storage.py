import pytest

from retrieval.embedder import HashingEmbedder
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.keyword_search import KeywordSearch
from retrieval.semantic_search import SemanticSearch
from storage.database import Database


def make_chunk(cid, text, doc="d1", page=1, index=0):
    return {"chunk_id": cid, "document_id": doc, "document_name": f"{doc}.txt",
            "page_number": page, "chunk_index": index, "text": text, "created_at": "now"}


def make_doc(doc_id="d1"):
    return {"document_id": doc_id, "original_filename": f"{doc_id}.txt",
            "stored_filename": f"{doc_id}.txt", "stored_path": f"/x/{doc_id}.txt",
            "file_extension": ".txt", "file_size_bytes": 10, "sha256": doc_id * 8,
            "ingested_at": "now"}


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "t.db")
    database.add_document(make_doc("d1"))
    database.add_chunks([
        make_chunk("c1", "Solar panels convert sunlight into electricity.", index=0),
        make_chunk("c2", "Wind turbines use moving air to generate power.", index=1),
        make_chunk("c3", "The quarterly budget was approved by the finance team.", index=2),
    ])
    yield database
    database.close()


def test_fts5_keyword_search_ranks_exact_term(db):
    assert db.fts_enabled
    results = KeywordSearch(db).search("turbines")
    assert [r["chunk_id"] for r in results] == ["c2"]
    assert results[0]["score"] > 0


def test_keyword_search_stems_and_ignores_punctuation(db):
    assert KeywordSearch(db).search("panel?!")[0]["chunk_id"] == "c1"
    assert KeywordSearch(db).search("???") == []


def test_keyword_query_cannot_break_fts_syntax(db):
    assert KeywordSearch(db).search('"; DROP TABLE chunks; --') == []
    assert db.count_chunks() == 3


def test_add_chunks_is_idempotent(db):
    assert db.add_chunks([make_chunk("c1", "duplicate")]) == []
    assert db.count_chunks() == 3


def test_semantic_index_is_incremental(db):
    semantic = SemanticSearch(db, HashingEmbedder(), min_score=0.0)
    assert semantic.index_new_chunks() == 3
    assert semantic.index_new_chunks() == 0
    db.add_chunks([make_chunk("c4", "Hydro dams store water.", index=3)])
    assert semantic.index_new_chunks() == 1
    assert db.count_embeddings("hashing-384") == 4


def test_semantic_search_returns_best_first(db):
    semantic = SemanticSearch(db, HashingEmbedder(), min_score=0.0)
    semantic.index_new_chunks()
    results = semantic.search("sunlight electricity solar", limit=3)
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["score"] >= results[1]["score"]


def test_semantic_min_score_filters_irrelevant(db):
    semantic = SemanticSearch(db, HashingEmbedder(), min_score=0.5)
    semantic.index_new_chunks()
    assert semantic.search("zebra giraffe safari") == []


def test_hybrid_fuses_both_sources(db):
    semantic = SemanticSearch(db, HashingEmbedder(), min_score=0.0)
    semantic.index_new_chunks()
    retriever = HybridRetriever(KeywordSearch(db), semantic)
    results = retriever.retrieve("how do solar panels produce electricity", limit=2)
    assert results[0]["chunk_id"] == "c1"
    assert len({r["chunk_id"] for r in results}) == len(results)


def test_hybrid_survives_semantic_failure(db):
    class Broken:
        def search(self, *a, **k):
            raise RuntimeError("model missing")

    results = HybridRetriever(KeywordSearch(db), Broken()).retrieve("turbines")
    assert results[0]["chunk_id"] == "c2"


def test_delete_document_removes_chunks_and_fts(db):
    db.delete_document("d1")
    assert db.count_chunks() == 0
    assert KeywordSearch(db).search("solar") == []
