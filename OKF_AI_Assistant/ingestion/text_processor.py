import re
from datetime import datetime, timezone

from config.settings import CHUNK_OVERLAP_CHARS, CHUNK_SIZE_CHARS


class TextProcessor:
    def __init__(self, chunk_size=CHUNK_SIZE_CHARS, overlap=CHUNK_OVERLAP_CHARS):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def clean_text(self, text):
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def split_text(self, text):
        """Split into overlapping windows, preferring paragraph/sentence boundaries."""
        pieces = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + self.chunk_size, length)
            if end < length:
                end = self._snap_to_boundary(text, start, end)
            piece = text[start:end].strip()
            if piece:
                pieces.append(piece)
            if end >= length:
                break
            # Always advance, even when the boundary search shrank the window.
            start = max(end - self.overlap, start + 1)
        return pieces

    def _snap_to_boundary(self, text, start, end):
        floor = start + self.chunk_size // 2
        for marker in ("\n\n", "\n", ". ", "? ", "! ", " "):
            position = text.rfind(marker, floor, end)
            if position != -1:
                return position + len(marker)
        return end

    def create_chunks(self, document_record, pages):
        chunks = []
        index = 0
        now = datetime.now(timezone.utc).isoformat()
        for page in pages:
            cleaned = self.clean_text(page["text"])
            for piece in self.split_text(cleaned):
                chunks.append({
                    "chunk_id": f"{document_record['document_id']}_{index}",
                    "document_id": document_record["document_id"],
                    "document_name": document_record["original_filename"],
                    "page_number": page["page_number"],
                    "chunk_index": index,
                    "text": piece,
                    "created_at": now,
                })
                index += 1
        return chunks
