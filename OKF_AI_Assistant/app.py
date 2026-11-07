import streamlit as st

from config import settings
from pipeline import KnowledgeAssistant

st.set_page_config(page_title="OKF AI Knowledge Assistant", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Starting the knowledge engine...")
def get_assistant():
    return KnowledgeAssistant()


assistant = get_assistant()

st.title("OKF AI Knowledge Assistant")
st.caption("Offline document intelligence, OKF knowledge management and locally grounded question answering.")

# ---- sidebar: system status -------------------------------------------------
with st.sidebar:
    st.header("System status")
    ollama_up = assistant.llm.is_available()
    models = assistant.llm.list_models() if ollama_up else []
    if not ollama_up:
        st.error("Ollama is not running. Start the Ollama app or run `ollama serve`.")
    elif not any(m.split(":")[0] == assistant.llm.model.split(":")[0]
                 and (":" not in assistant.llm.model or m == assistant.llm.model) for m in models):
        st.warning(f"Model `{assistant.llm.model}` is not installed. Run `ollama pull {assistant.llm.model}`.")
    else:
        st.success(f"Ollama ready · `{assistant.llm.model}`")

    if assistant.embedder_is_fallback:
        st.warning("sentence-transformers is not installed: using a basic keyword-hash embedder. "
                   "Install it for true semantic search.")
    else:
        st.info(f"Embeddings: `{assistant.embedder.name}`")

    stats = assistant.stats()
    st.metric("Documents", stats["documents"])
    st.metric("Chunks", f"{stats['embedded_chunks']} / {stats['chunks']} embedded")
    st.metric("Concepts · Entities", f"{stats['concepts']} · {stats['entities']}")
    if not stats["fts5"]:
        st.warning("SQLite FTS5 unavailable; using slower fallback keyword search.")

tab_upload, tab_ask, tab_kb, tab_docs = st.tabs(
    ["Upload documents", "Ask questions", "Knowledge base", "Documents"]
)

# ---- upload -----------------------------------------------------------------
with tab_upload:
    st.header("Upload documents")
    extract = st.checkbox(
        "Extract concepts and entities with the local LLM (slower, builds the OKF knowledge pages)",
        value=True,
    )
    files = st.file_uploader(
        "Choose documents", accept_multiple_files=True,
        type=[e.lstrip(".") for e in sorted(settings.ALLOWED_EXTENSIONS)],
    )
    if files and st.button("Process documents", type="primary"):
        assistant.extract_concepts = extract
        bar = st.progress(0.0)
        status = st.empty()
        for number, uploaded in enumerate(files, start=1):
            status.write(f"Processing **{uploaded.name}** ({number}/{len(files)})...")
            result = assistant.ingest_bytes(uploaded.name, uploaded.getvalue())
            if result["status"] == "ok":
                st.success(f"{uploaded.name}: {result['chunks']} chunks, "
                           f"{result['concepts']} concepts, {result['entities']} entities.")
            elif result["status"] == "duplicate":
                st.info(f"{uploaded.name}: {result['message']}")
            else:
                st.error(f"{uploaded.name}: {result['message']}")
            for warning in result["warnings"]:
                st.warning(f"{uploaded.name}: {warning}")
            bar.progress(number / len(files))

        status.write("Building semantic index for new chunks...")
        try:
            added = assistant.index_embeddings()
            status.success(f"Done. {added} new chunks embedded.")
        except Exception as error:
            status.error(f"Semantic indexing failed ({error}). Keyword search still works.")

# ---- ask --------------------------------------------------------------------
with tab_ask:
    st.header("Ask questions")
    if stats["chunks"] == 0:
        st.info("Upload and process a document first.")
    question = st.text_input("Your question", placeholder="What topics are covered in my documents?")
    top_k = st.slider("Evidence passages to use", 1, 10, settings.RETRIEVAL_LIMIT)
    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Retrieving evidence and generating an answer..."):
            result = assistant.ask(question.strip(), limit=top_k)
        if result["error"]:
            st.error(result["error"])
        else:
            st.markdown("### Answer")
            st.write(result["answer"])
        if result["sources"]:
            st.markdown("### Sources")
            for source in result["sources"]:
                mark = "✅ cited" if source["cited"] else "retrieved"
                label = (f"[Evidence {source['evidence_number']}] {source['document']} "
                         f"· page {source['page']} · {mark}")
                with st.expander(label):
                    st.write(source["excerpt"])
                    st.caption(f"Chunk ID: {source['chunk_id']}")

# ---- knowledge base ---------------------------------------------------------
with tab_kb:
    st.header("OKF knowledge base")
    if st.button("Validate OKF knowledge base"):
        report = assistant.validate()
        st.write(f"Files checked: {report['files_checked']} · Errors: {len(report['errors'])} "
                 f"· Warnings: {len(report['warnings'])}")
        if report["errors"]:
            st.error("\n".join(report["errors"]))
        if report["warnings"]:
            st.warning("\n".join(report["warnings"]))
        if not report["errors"] and not report["warnings"]:
            st.success("OKF validation passed.")

    pages = sorted(p for p in assistant.kb_dir.rglob("*.md"))
    if pages:
        names = [p.relative_to(assistant.kb_dir).as_posix() for p in pages]
        choice = st.selectbox("Open a page", names, index=names.index("index.md") if "index.md" in names else 0)
        st.divider()
        st.markdown((assistant.kb_dir / choice).read_text(encoding="utf-8"))
    else:
        st.info("The knowledge base is empty.")

# ---- documents --------------------------------------------------------------
with tab_docs:
    st.header("Original documents")
    documents = assistant.database.list_documents()
    if documents:
        st.dataframe(
            [{"File": d["original_filename"], "Pages": d["page_count"], "Chunks": d["chunk_count"],
              "Size (KB)": round(d["file_size_bytes"] / 1024, 1), "Ingested": d["ingested_at"][:19]}
             for d in documents],
            use_container_width=True,
        )
    else:
        st.info("No documents uploaded yet.")
