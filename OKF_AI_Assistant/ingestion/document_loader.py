import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from config.settings import ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE_MB, RAW_DOCUMENTS_DIR


class DocumentLoader:
    """Validates an upload and stores the original file under raw_documents/."""

    def __init__(self, raw_dir=RAW_DOCUMENTS_DIR):
        self.raw_dir = Path(raw_dir)

    def load_document(self, uploaded_file):
        """Accept a Streamlit UploadedFile (or any object with .name/.read())."""
        return self.load_bytes(uploaded_file.name, uploaded_file.read())

    def prepare(self, original_name, content):
        """Validate and hash an upload without writing anything to disk."""
        suffix = Path(original_name).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {suffix or '(none)'}")
        if not content:
            raise ValueError("The file is empty.")
        if len(content) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            raise ValueError(f"File exceeds the {MAX_UPLOAD_SIZE_MB} MB limit.")

        sha = hashlib.sha256(content).hexdigest()
        document_id = sha[:16]
        stored_name = f"{document_id}_{self._safe_name(original_name)}"
        return {
            "document_id": document_id,
            "original_filename": Path(original_name).name,
            "stored_filename": stored_name,
            "stored_path": str(self.raw_dir / stored_name),
            "file_extension": suffix,
            "file_size_bytes": len(content),
            "sha256": sha,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }

    def load_bytes(self, original_name, content):
        record = self.prepare(original_name, content)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        destination = Path(record["stored_path"])
        destination.write_bytes(content)
        return record, destination

    @staticmethod
    def _safe_name(filename):
        name = Path(filename).name
        return re.sub(r"[^A-Za-z0-9._-]", "_", name)[:120]
