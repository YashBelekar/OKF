from ingestion.text_processor import TextProcessor


def test_clean_text_removes_extra_whitespace():
    assert TextProcessor().clean_text("Hello world\n\n\n\nNext") == "Hello world\n\nNext"


def test_create_chunks_returns_chunks():
    record = {"document_id": "test123", "original_filename": "test.txt"}
    pages = [{"page_number": 1, "text": "This is a test document about solar panels."}]
    chunks = TextProcessor().create_chunks(record, pages)
    assert len(chunks) == 1
    assert chunks[0]["document_id"] == "test123"
    assert chunks[0]["chunk_id"] == "test123_0"


def test_long_text_is_split_with_overlap_and_terminates():
    processor = TextProcessor(chunk_size=200, overlap=40)
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    pieces = processor.split_text(text)
    assert len(pieces) > 5
    assert all(len(p) <= 200 for p in pieces)
    assert "Sentence number 199" in pieces[-1]


def test_unbroken_text_still_terminates():
    processor = TextProcessor(chunk_size=100, overlap=20)
    pieces = processor.split_text("x" * 1000)
    assert len(pieces) >= 10


def test_page_numbers_are_preserved_and_empty_pages_skipped():
    record = {"document_id": "d", "original_filename": "f.pdf"}
    pages = [{"page_number": 1, "text": "alpha"}, {"page_number": 2, "text": "  "},
             {"page_number": 3, "text": "gamma"}]
    chunks = TextProcessor().create_chunks(record, pages)
    assert [c["page_number"] for c in chunks] == [1, 3]


def test_invalid_overlap_rejected():
    import pytest
    with pytest.raises(ValueError):
        TextProcessor(chunk_size=100, overlap=100)
