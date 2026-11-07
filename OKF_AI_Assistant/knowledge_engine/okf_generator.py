"""Writes OKF knowledge files: Markdown with YAML frontmatter, provenance,
relative links, plus index.md and log.md."""
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from config.settings import KNOWLEDGE_BASE_DIR

GENERATOR_ID = "OKF_AI_Assistant/1.0"
KIND_DIRS = {"Concept": "concepts", "Entity": "entities", "Reference": "references"}
FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?(.*)$", re.DOTALL)


def create_slug(title):
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(title).lower()).strip("-")
    return slug[:80].strip("-") or "untitled"


def _now():
    return datetime.now(timezone.utc).isoformat()


class OKFGenerator:
    def __init__(self, knowledge_base_dir=KNOWLEDGE_BASE_DIR):
        self.root = Path(knowledge_base_dir)
        self.dirs = {kind: self.root / name for kind, name in KIND_DIRS.items()}
        for directory in self.dirs.values():
            directory.mkdir(parents=True, exist_ok=True)

    # ---- pages ----------------------------------------------------------
    def create_reference(self, document_record):
        slug = f"{create_slug(Path(document_record['original_filename']).stem)}-{document_record['document_id'][:6]}"
        path = self.dirs["Reference"] / f"{slug}.md"
        frontmatter = {
            "type": "Reference",
            "title": document_record["original_filename"],
            "description": "Original uploaded document retained as evidence.",
            "status": "stable",
            "generated": {"by": GENERATOR_ID, "at": _now()},
            "sources": [{
                "id": document_record["document_id"],
                "resource": f"/raw_documents/{document_record['stored_filename']}",
                "title": document_record["original_filename"],
            }],
        }
        body = (
            f"# {document_record['original_filename']}\n\n"
            "This reference represents the original uploaded document.\n\n"
            f"- Document ID: `{document_record['document_id']}`\n"
            f"- Stored file: `{document_record['stored_filename']}`\n"
            f"- SHA-256: `{document_record['sha256']}`\n"
            f"- Ingested at: `{document_record['ingested_at']}`\n"
        )
        self._write(path, frontmatter, body)
        return path

    def reference_slug(self, document_record):
        return f"{create_slug(Path(document_record['original_filename']).stem)}-{document_record['document_id'][:6]}"

    def create_concept(self, title, description, source_document, source_page,
                       source_chunk_id, tags=None, related=None):
        return self._create_page("Concept", title, description, source_document,
                                 source_page, source_chunk_id, tags, related)

    def create_entity(self, title, description, source_document, source_page,
                      source_chunk_id, tags=None, related=None):
        return self._create_page("Entity", title, description, source_document,
                                 source_page, source_chunk_id, tags, related)

    def _create_page(self, kind, title, description, document, page, chunk_id,
                     tags, related):
        title = str(title).strip()[:120] or "Untitled"
        slug = create_slug(title)
        path = self.dirs[kind] / f"{slug}.md"
        source = {
            "id": document["document_id"],
            "resource": f"/raw_documents/{document['stored_filename']}",
            "title": document["original_filename"],
            "page": page,
            "chunk_id": chunk_id,
        }
        ref_link = f"../references/{self.reference_slug(document)}.md"
        evidence_line = (
            f"- [{document['original_filename']}]({ref_link}), page {page}, chunk `{chunk_id}`"
        )

        if path.exists():
            return path, self._merge_source(path, source, evidence_line, tags)

        frontmatter = {
            "type": kind,
            "title": title,
            "description": str(description).strip(),
            "tags": sorted({str(t).strip() for t in (tags or []) if str(t).strip()}),
            "status": "stable",
            "generated": {"by": GENERATOR_ID, "at": _now()},
            "sources": [source],
        }
        related_lines = "".join(
            f"- [{name}]({create_slug(name)}.md)\n" for name in (related or [])
        )
        body = (
            f"# {title}\n\n{str(description).strip()}\n\n"
            f"## Source evidence\n\n{evidence_line}\n"
        )
        if related_lines:
            body += f"\n## Related\n\n{related_lines}"
        self._write(path, frontmatter, body)
        return path, True

    def _merge_source(self, path, source, evidence_line, tags):
        """Add provenance to an existing page. Returns True if anything changed."""
        frontmatter, body = self._read(path)
        sources = frontmatter.setdefault("sources", [])
        if any(s.get("chunk_id") == source["chunk_id"] for s in sources):
            return False
        sources.append(source)
        merged_tags = set(frontmatter.get("tags") or []) | {str(t) for t in (tags or [])}
        frontmatter["tags"] = sorted(t for t in merged_tags if t)
        if "## Source evidence" in body:
            head, tail = body.split("## Source evidence", 1)
            tail = tail.rstrip("\n") + "\n" + evidence_line + "\n"
            body = head + "## Source evidence" + tail
        else:
            body = body.rstrip("\n") + "\n\n## Source evidence\n\n" + evidence_line + "\n"
        self._write(path, frontmatter, body)
        return True

    # ---- index and log --------------------------------------------------
    def update_index(self):
        def links(kind):
            lines = []
            for path in sorted(self.dirs[kind].glob("*.md")):
                title = path.stem.replace("-", " ").title()
                try:
                    title = self._read(path)[0].get("title", title)
                except Exception:
                    pass
                lines.append(f"- [{title}]({KIND_DIRS[kind]}/{path.name})")
            return "\n".join(lines) or "- None yet."

        content = (
            "# Knowledge Index\n\n"
            f"## Concepts\n\n{links('Concept')}\n\n"
            f"## Entities\n\n{links('Entity')}\n\n"
            f"## References\n\n{links('Reference')}\n"
        )
        path = self.root / "index.md"
        path.write_text(content, encoding="utf-8")
        return path

    def append_log(self, message):
        path = self.root / "log.md"
        entry = f"## {_now()}\n\n{message}\n\n"
        if not path.exists():
            path.write_text("# Knowledge Log\n\n", encoding="utf-8")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(entry)

    # ---- helpers --------------------------------------------------------
    @staticmethod
    def _read(path):
        text = Path(path).read_text(encoding="utf-8")
        match = FRONTMATTER_RE.match(text)
        if not match:
            raise ValueError("missing frontmatter")
        return yaml.safe_load(match.group(1)) or {}, match.group(2).lstrip("\n")

    @staticmethod
    def _write(path, frontmatter, body):
        yaml_text = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).rstrip("\n")
        Path(path).write_text(f"---\n{yaml_text}\n---\n\n{body}", encoding="utf-8")
