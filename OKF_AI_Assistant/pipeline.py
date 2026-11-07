"""High-level orchestration used by the Streamlit app, tests and evaluation."""
from pathlib import Path

from ai_engine.answer_generator import AnswerGenerator
from ai_engine.llm_client import LLMClient
from config import settings
from ingestion.document_loader import DocumentLoader
from ingestion.text_extractor import ExtractionError, TextExtractor
from ingestion.text_processor import TextProcessor
from knowledge_engine.concept_extractor import ConceptExtractor
from knowledge_engine.okf_generator import OKFGenerator
from knowledge_engine.validator import OKFValidator
from retrieval.embedder import get_embedder
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.keyword_search import KeywordSearch
from retrieval.semantic_search import SemanticSearch
from storage.database import Database


class KnowledgeAssistant:
    def __init__(self, root=None, llm_client=None, embedder=None, extract_concepts=True):
        root = Path(root) if root else settings.PROJECT_ROOT
        self.root = root
        self.raw_dir = root / "raw_documents"
        self.kb_dir = root / "knowledge_base"

        self.database = Database(root / "data" / "okf_assistant.db")
        self.loader = DocumentLoader(self.raw_dir)
        self.extractor = TextExtractor()
        self.processor = TextProcessor()
        self.okf = OKFGenerator(self.kb_dir)
        self.llm = llm_client or LLMClient()

        if embedder is None:
            embedder, self.embedder_is_fallback = get_embedder()
        else:
            self.embedder_is_fallback = False
        self.embedder = embedder

        self.concepts = ConceptExtractor(self.llm)
        self.extract_concepts = extract_concepts
        self.keyword = KeywordSearch(self.database)
        self.semantic = SemanticSearch(self.database, embedder)
        self.retriever = HybridRetriever(self.keyword, self.semantic)
        self.answerer = AnswerGenerator(self.llm)

    # ---- ingestion ------------------------------------------------------
    def ingest_bytes(self, name, content):
        """Ingest one file. Returns a result dict; never raises for bad files."""
        result = {"name": name, "status": "ok", "chunks": 0, "concepts": 0,
                  "entities": 0, "warnings": []}
        try:
            record = self.loader.prepare(name, content)
        except ValueError as error:
            return {**result, "status": "error", "message": str(error)}

        if self.database.get_document_by_sha(record["sha256"]):
            return {**result, "status": "duplicate",
                    "message": "This exact file is already in the knowledge base."}

        record, path = self.loader.load_bytes(name, content)
        try:
            pages = self.extractor.extract(path)
            result["warnings"] = list(self.extractor.warnings)
            if not pages:
                raise ExtractionError("No readable text found in the document.")
            chunks = self.processor.create_chunks(record, pages)
            self.database.add_document(record)
            self.database.add_chunks(chunks)
            self.database.update_document_counts(record["document_id"], len(pages), len(chunks))
        except Exception as error:
            path.unlink(missing_ok=True)
            return {**result, "status": "error", "message": str(error)}

        result["chunks"] = len(chunks)
        self.okf.create_reference(record)

        if self.extract_concepts:
            items = self.concepts.extract(chunks)
            result["warnings"].extend(self.concepts.errors[:1])
            for item in items:
                create = self.okf.create_entity if item["kind"] == "Entity" else self.okf.create_concept
                create(item["title"], item["description"], record, item["source_page"],
                       item["source_chunk_id"], tags=item["tags"])
                key = "entities" if item["kind"] == "Entity" else "concepts"
                result[key] += 1

        self.okf.update_index()
        self.okf.append_log(
            f"Ingested `{record['original_filename']}`: {len(pages)} page(s), "
            f"{len(chunks)} chunk(s), {result['concepts']} concept(s), "
            f"{result['entities']} entit(ies)."
        )
        result["document_id"] = record["document_id"]
        return result

    def index_embeddings(self, progress=None):
        return self.semantic.index_new_chunks(progress)

    # ---- question answering --------------------------------------------
    def retrieve(self, question, limit=settings.RETRIEVAL_LIMIT):
        return self.retriever.retrieve(question, limit=limit)

    def ask(self, question, limit=settings.RETRIEVAL_LIMIT):
        chunks = self.retrieve(question, limit)
        result = self.answerer.answer(question, chunks)
        result["retrieved"] = len(chunks)
        return result

    # ---- maintenance ----------------------------------------------------
    def validate(self):
        return OKFValidator().validate_bundle(self.kb_dir)

    def stats(self):
        docs = self.database.list_documents()
        return {
            "documents": len(docs),
            "chunks": self.database.count_chunks(),
            "embedded_chunks": self.database.count_embeddings(self.embedder.name),
            "concepts": len(list((self.kb_dir / "concepts").glob("*.md"))),
            "entities": len(list((self.kb_dir / "entities").glob("*.md"))),
            "fts5": self.database.fts_enabled,
        }
