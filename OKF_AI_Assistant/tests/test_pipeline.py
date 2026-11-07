import pytest

from ai_engine.answer_generator import AnswerGenerator
from ai_engine.llm_client import strip_thinking
from ai_engine.prompt_builder import INSUFFICIENT_INFO, PromptBuilder
from pipeline import KnowledgeAssistant
from retrieval.embedder import HashingEmbedder
from tests.helpers import FakeLLM

SOLAR = b"Solar panels convert sunlight into electricity using photovoltaic cells.\n\nThey are installed on rooftops."
WIND = b"Wind turbines generate electricity from moving air. Offshore farms are common in Denmark."


@pytest.fixture
def assistant(tmp_path):
    instance = KnowledgeAssistant(root=tmp_path, llm_client=FakeLLM(), embedder=HashingEmbedder())
    yield instance
    instance.database.close()


def test_end_to_end_ingest_and_ask(assistant):
    result = assistant.ingest_bytes("solar.txt", SOLAR)
    assert result["status"] == "ok" and result["chunks"] == 1
    assert result["concepts"] == 1 and result["entities"] == 1
    assert assistant.index_embeddings() == 1

    answer = assistant.ask("How do solar panels make electricity?")
    assert answer["error"] is None and "[Evidence 1]" in answer["answer"]
    assert answer["sources"][0]["document"] == "solar.txt"
    assert answer["sources"][0]["page"] == 1 and answer["sources"][0]["cited"]


def test_okf_files_created_and_valid(assistant):
    assistant.ingest_bytes("solar.txt", SOLAR)
    kb = assistant.kb_dir
    assert (kb / "concepts" / "solar-energy.md").exists()
    assert (kb / "entities" / "acme-corp.md").exists()
    assert (kb / "index.md").exists() and (kb / "log.md").exists()
    report = assistant.validate()
    assert report["errors"] == [] and report["warnings"] == []


def test_duplicate_upload_is_skipped(assistant):
    assert assistant.ingest_bytes("solar.txt", SOLAR)["status"] == "ok"
    assert assistant.ingest_bytes("copy.txt", SOLAR)["status"] == "duplicate"
    assert assistant.stats()["documents"] == 1


def test_incremental_indexing_only_embeds_new_chunks(assistant):
    assistant.ingest_bytes("solar.txt", SOLAR)
    assert assistant.index_embeddings() == 1
    assistant.ingest_bytes("wind.txt", WIND)
    assert assistant.index_embeddings() == 1
    assert assistant.stats()["embedded_chunks"] == 2


def test_multi_document_retrieval_cites_both(assistant):
    assistant.ingest_bytes("solar.txt", SOLAR)
    assistant.ingest_bytes("wind.txt", WIND)
    assistant.index_embeddings()
    docs = {c["document_name"] for c in assistant.retrieve("how is electricity generated", limit=5)}
    assert docs == {"solar.txt", "wind.txt"}


def test_unrelated_question_gets_insufficient_info(assistant):
    assistant.ingest_bytes("solar.txt", SOLAR)
    assistant.index_embeddings()
    answer = assistant.ask("zebra giraffe safari")
    assert answer["answer"] == INSUFFICIENT_INFO and answer["sources"] == []
    assert assistant.llm.prompts == [p for p in assistant.llm.prompts if "zebra" not in p]


def test_bad_uploads_report_errors_without_crashing(assistant):
    assert assistant.ingest_bytes("x.exe", b"abc")["status"] == "error"
    assert assistant.ingest_bytes("broken.docx", b"not a docx")["status"] == "error"
    assert assistant.stats()["documents"] == 0
    assert not list(assistant.raw_dir.glob("*broken*"))  # failed file cleaned up


def test_ingestion_works_when_llm_is_down(tmp_path):
    assistant = KnowledgeAssistant(root=tmp_path, llm_client=FakeLLM(fail=True),
                                   embedder=HashingEmbedder())
    result = assistant.ingest_bytes("solar.txt", SOLAR)
    assert result["status"] == "ok" and result["concepts"] == 0
    assert any("Ollama" in w for w in result["warnings"])
    assistant.index_embeddings()
    assert assistant.retrieve("solar panels")  # search still works without the LLM
    assistant.database.close()


def test_llm_failure_is_reported_not_raised(assistant):
    assistant.ingest_bytes("solar.txt", SOLAR)
    assistant.index_embeddings()
    assistant.answerer.llm_client.fail = True
    answer = assistant.ask("solar panels")
    assert "Ollama" in answer["error"] and answer["sources"]


def test_prompt_contains_evidence_and_grounding_rules():
    builder = PromptBuilder()
    context = builder.build_context([{"document_name": "a.pdf", "page_number": 4, "text": "Fact."}])
    prompt = builder.build_user_prompt("Q?", context)
    assert "[Evidence 1]" in prompt and "a.pdf" in prompt and "Page: 4" in prompt
    assert "only the evidence" in prompt and INSUFFICIENT_INFO in prompt


def test_uncited_answer_marks_sources_as_not_cited():
    generator = AnswerGenerator(FakeLLM(answer="No markers here."))
    chunk = {"document_name": "a", "page_number": 1, "chunk_id": "c", "text": "t"}
    assert generator.answer("q", [chunk])["sources"][0]["cited"] is False


def test_strip_thinking_blocks():
    assert strip_thinking("<think>reasoning</think>Answer") == "Answer"
    assert strip_thinking("reasoning</think> Answer") == "Answer"
    assert strip_thinking("Plain") == "Plain"
