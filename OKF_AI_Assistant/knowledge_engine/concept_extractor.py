import json
import re

from ai_engine.llm_client import LLMClient, LLMError
from config.settings import CONCEPT_CHUNK_CHARS, MAX_CHUNKS_FOR_CONCEPTS

SYSTEM_PROMPT = "You extract structured knowledge from documents. Return only valid JSON."

USER_TEMPLATE = """From the text below, extract up to 3 important concepts and up to 3 named entities
(people, organisations, products, places). Use only what the text states.

Return JSON in exactly this shape:
{{"concepts": [{{"title": "...", "description": "one or two sentences", "tags": ["..."]}}],
  "entities": [{{"title": "...", "description": "one sentence", "tags": ["..."]}}]}}

Text:
{text}
"""


class ConceptExtractor:
    def __init__(self, llm_client=None, max_chunks=MAX_CHUNKS_FOR_CONCEPTS):
        self.llm_client = llm_client or LLMClient()
        self.max_chunks = max_chunks
        self.errors = []

    def extract(self, chunks):
        """Return a list of items: {kind, title, description, tags, source_*}.

        Stops early if the LLM is unreachable and records why in self.errors.
        """
        self.errors = []
        items = []
        for chunk in self._sample(chunks):
            try:
                raw = self.llm_client.generate(
                    SYSTEM_PROMPT,
                    USER_TEMPLATE.format(text=chunk["text"][:CONCEPT_CHUNK_CHARS]),
                    json_mode=True,
                )
            except LLMError as error:
                self.errors.append(str(error))
                break
            for kind, entry in self.parse(raw):
                items.append({
                    "kind": kind,
                    "title": entry["title"],
                    "description": entry["description"],
                    "tags": entry["tags"],
                    "source_page": chunk["page_number"],
                    "source_chunk_id": chunk["chunk_id"],
                })
        return items

    def _sample(self, chunks):
        if len(chunks) <= self.max_chunks:
            return chunks
        step = len(chunks) / self.max_chunks
        return [chunks[int(i * step)] for i in range(self.max_chunks)]

    @staticmethod
    def parse(raw):
        """Parse model output into [(kind, entry)]; tolerant of messy output."""
        if not raw:
            return []
        data = None
        # Try the outermost structure first: whichever bracket opens earliest.
        patterns = sorted(
            (r"\{.*\}", r"\[.*\]"),
            key=lambda p: (m.start() if (m := re.search(p, raw, re.DOTALL)) else len(raw) + 1),
        )
        for pattern in patterns:
            match = re.search(pattern, raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                    break
                except json.JSONDecodeError:
                    continue
        if data is None:
            return []
        if isinstance(data, list):
            data = {"concepts": data}
        if not isinstance(data, dict):
            return []

        results = []
        for key, kind in (("concepts", "Concept"), ("entities", "Entity")):
            for entry in data.get(key) or []:
                if not isinstance(entry, dict):
                    continue
                title = str(entry.get("title") or "").strip()
                if not title:
                    continue
                tags = entry.get("tags") or []
                if not isinstance(tags, list):
                    tags = [tags]
                results.append((kind, {
                    "title": title,
                    "description": str(entry.get("description") or "No description provided.").strip(),
                    "tags": [str(t) for t in tags][:6],
                }))
        return results
