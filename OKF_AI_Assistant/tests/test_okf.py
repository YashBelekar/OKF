import yaml

from knowledge_engine.concept_extractor import ConceptExtractor
from knowledge_engine.okf_generator import OKFGenerator, create_slug
from knowledge_engine.validator import OKFValidator
from tests.helpers import FakeLLM

DOC = {"document_id": "abc123def4567890", "original_filename": "Solar Report.pdf",
       "stored_filename": "abc123def4567890_Solar_Report.pdf", "sha256": "f" * 64,
       "ingested_at": "2026-01-01T00:00:00+00:00"}


def frontmatter(path):
    text = path.read_text(encoding="utf-8")
    return yaml.safe_load(text.split("---")[1])


def build(tmp_path):
    gen = OKFGenerator(tmp_path)
    gen.create_reference(DOC)
    return gen


def test_generated_bundle_validates_cleanly(tmp_path):
    gen = build(tmp_path)
    gen.create_concept("Solar Energy", "Energy from the sun.", DOC, 3, "abc_0", tags=["energy"])
    gen.create_entity("Acme Corp", "A company.", DOC, 1, "abc_1")
    gen.update_index()
    gen.append_log("test")
    report = OKFValidator().validate_bundle(tmp_path)
    assert report["errors"] == [] and report["warnings"] == []
    assert report["files_checked"] == 3


def test_concept_frontmatter_has_provenance(tmp_path):
    gen = build(tmp_path)
    path, created = gen.create_concept("Solar Energy", "Energy.", DOC, 3, "abc_0")
    meta = frontmatter(path)
    assert created and meta["type"] == "Concept"
    assert meta["sources"][0]["page"] == 3 and meta["sources"][0]["chunk_id"] == "abc_0"


def test_duplicate_concept_merges_sources_instead_of_overwriting(tmp_path):
    gen = build(tmp_path)
    gen.create_concept("Solar Energy", "Energy.", DOC, 3, "abc_0", tags=["a"])
    path, changed = gen.create_concept("Solar Energy", "Other.", DOC, 7, "abc_5", tags=["b"])
    meta = frontmatter(path)
    assert changed and len(meta["sources"]) == 2 and meta["tags"] == ["a", "b"]
    assert "page 7" in path.read_text(encoding="utf-8")
    _, changed_again = gen.create_concept("Solar Energy", "Other.", DOC, 7, "abc_5")
    assert not changed_again
    assert OKFValidator().validate_bundle(tmp_path)["errors"] == []


def test_validator_flags_missing_type_and_frontmatter_and_broken_links(tmp_path):
    gen = build(tmp_path)
    (tmp_path / "concepts" / "notype.md").write_text("---\ntitle: x\n---\n\nbody", encoding="utf-8")
    (tmp_path / "concepts" / "plain.md").write_text("# no frontmatter", encoding="utf-8")
    (tmp_path / "concepts" / "links.md").write_text(
        "---\ntype: Concept\ntitle: L\n---\n\n[gone](missing.md) [web](https://x.org)", encoding="utf-8")
    report = OKFValidator().validate_bundle(tmp_path)
    joined = " ".join(report["errors"])
    assert "notype.md" in joined and "plain.md" in joined
    assert any("broken link 'missing.md'" in w for w in report["warnings"])
    assert not any("x.org" in w for w in report["warnings"])


def test_slug_is_filesystem_safe():
    assert create_slug("C++ / Memory: Safety!") == "c-memory-safety"
    assert create_slug("???") == "untitled"


def test_extractor_parses_object_array_and_wrapped_json():
    obj = '{"concepts":[{"title":"A","description":"d","tags":["t"]}],"entities":[{"title":"B"}]}'
    kinds = [k for k, _ in ConceptExtractor.parse(obj)]
    assert kinds == ["Concept", "Entity"]
    assert ConceptExtractor.parse('Sure!\n[{"title":"Z","description":"zz"}]')[0][1]["title"] == "Z"
    assert ConceptExtractor.parse("not json at all") == []
    assert ConceptExtractor.parse('{"concepts":[{"title":""},"junk"]}') == []


def test_extractor_stops_when_llm_unavailable():
    extractor = ConceptExtractor(FakeLLM(fail=True))
    chunks = [{"text": "t", "page_number": 1, "chunk_id": f"c{i}"} for i in range(3)]
    assert extractor.extract(chunks) == []
    assert len(extractor.errors) == 1  # gave up after the first failure


def test_extractor_samples_when_too_many_chunks():
    llm = FakeLLM()
    extractor = ConceptExtractor(llm, max_chunks=5)
    chunks = [{"text": "t", "page_number": 1, "chunk_id": f"c{i}"} for i in range(50)]
    extractor.extract(chunks)
    assert len(llm.prompts) == 5
