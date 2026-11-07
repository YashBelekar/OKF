import re
from pathlib import Path

import yaml

from config.settings import KNOWLEDGE_BASE_DIR

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
VALID_TYPES = {"Concept", "Entity", "Reference"}


class OKFValidator:
    def validate_bundle(self, knowledge_base_dir=KNOWLEDGE_BASE_DIR):
        root = Path(knowledge_base_dir)
        errors, warnings = [], []
        files = [p for p in root.rglob("*.md") if p.name not in {"index.md", "log.md"}]

        for path in files:
            rel = path.relative_to(root).as_posix()
            text = path.read_text(encoding="utf-8")
            match = FRONTMATTER_RE.match(text)
            if not match:
                errors.append(f"{rel}: missing YAML frontmatter")
                continue
            try:
                frontmatter = yaml.safe_load(match.group(1)) or {}
            except yaml.YAMLError as error:
                errors.append(f"{rel}: invalid YAML ({error})")
                continue
            if not isinstance(frontmatter, dict):
                errors.append(f"{rel}: frontmatter is not a mapping")
                continue

            if not frontmatter.get("type"):
                errors.append(f"{rel}: missing required field 'type'")
            elif frontmatter["type"] not in VALID_TYPES:
                warnings.append(f"{rel}: unrecognised type '{frontmatter['type']}'")
            if not frontmatter.get("title"):
                warnings.append(f"{rel}: missing 'title'")
            if frontmatter.get("type") in {"Concept", "Entity"} and not frontmatter.get("sources"):
                warnings.append(f"{rel}: no provenance in 'sources'")

            for link in LINK_RE.findall(text[match.end():]):
                if re.match(r"^[a-z]+:", link) or link.startswith("#"):
                    continue  # external URL or in-page anchor
                target_part = link.split("#", 1)[0]
                target = (root / target_part.lstrip("/")) if target_part.startswith("/") \
                    else (path.parent / target_part)
                if not target.resolve().exists():
                    warnings.append(f"{rel}: broken link '{link}'")

        return {"errors": errors, "warnings": warnings, "files_checked": len(files)}
