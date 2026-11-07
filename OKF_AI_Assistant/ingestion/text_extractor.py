"""Extract text from every supported format as a list of {page_number, text}.

Heavy libraries are imported lazily so a missing optional dependency only
affects the format that needs it.
"""
import json
from pathlib import Path

from ingestion.ocr_service import OCRService


class ExtractionError(Exception):
    pass


class TextExtractor:
    def __init__(self, ocr_service=None):
        self.ocr_service = ocr_service or OCRService()
        self.warnings = []

    def extract(self, file_path):
        file_path = Path(file_path)
        self.warnings = []
        suffix = file_path.suffix.lower()
        handlers = {
            ".pdf": self._extract_pdf,
            ".docx": self._extract_docx,
            ".txt": self._extract_text_file,
            ".md": self._extract_text_file,
            ".csv": self._extract_csv,
            ".xlsx": self._extract_xlsx,
            ".pptx": self._extract_pptx,
            ".json": self._extract_json,
            ".html": self._extract_html,
            ".htm": self._extract_html,
        }
        if suffix not in handlers:
            raise ExtractionError(f"Unsupported file type: {suffix}")
        try:
            pages = handlers[suffix](file_path)
        except ExtractionError:
            raise
        except Exception as error:
            raise ExtractionError(f"Could not read {file_path.name}: {error}") from error
        return [p for p in pages if p["text"] and p["text"].strip()]

    # ---- PDF ------------------------------------------------------------
    def _extract_pdf(self, file_path):
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return self._extract_pdf_pypdf(file_path)

        pages = []
        ocr_ok = None
        with fitz.open(file_path) as pdf:
            for number, page in enumerate(pdf, start=1):
                text = page.get_text("text")
                if not text.strip():
                    if ocr_ok is None:
                        ocr_ok = self.ocr_service.is_available()
                    if ocr_ok:
                        try:
                            text = self.ocr_service.page_to_text(page)
                        except Exception as error:
                            self.warnings.append(f"OCR failed on page {number}: {error}")
                    else:
                        self.warnings.append(
                            f"Page {number} has no selectable text and OCR (Tesseract) is unavailable."
                        )
                pages.append({"page_number": number, "text": text})
        return pages

    def _extract_pdf_pypdf(self, file_path):
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        pages = []
        for number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                self.warnings.append(
                    f"Page {number} has no selectable text; install PyMuPDF + Tesseract for OCR."
                )
            pages.append({"page_number": number, "text": text})
        return pages

    # ---- Office ---------------------------------------------------------
    def _extract_docx(self, file_path):
        from docx import Document

        document = Document(str(file_path))
        parts = [p.text for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells]
                if any(cells):
                    parts.append(" | ".join(cells))
        return [{"page_number": 1, "text": "\n".join(parts)}]

    def _extract_pptx(self, file_path):
        from pptx import Presentation

        presentation = Presentation(str(file_path))
        slides = []
        for number, slide in enumerate(presentation.slides, start=1):
            parts = []
            for shape in slide.shapes:
                if shape.has_text_frame and shape.text_frame.text.strip():
                    parts.append(shape.text_frame.text)
                if getattr(shape, "has_table", False) and shape.has_table:
                    for row in shape.table.rows:
                        parts.append(" | ".join(c.text.strip() for c in row.cells))
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
                notes = slide.notes_slide.notes_text_frame.text.strip()
                if notes:
                    parts.append(f"Speaker notes: {notes}")
            slides.append({"page_number": number, "text": "\n".join(parts)})
        return slides

    def _extract_xlsx(self, file_path):
        import pandas as pd

        sheets = pd.read_excel(file_path, sheet_name=None, dtype=str)
        pages = []
        for name, frame in sheets.items():
            frame = frame.fillna("")
            pages.append({
                "page_number": len(pages) + 1,
                "text": f"Sheet: {name}\n\n{frame.to_csv(index=False)}",
            })
        return pages

    # ---- Plain text family ---------------------------------------------
    @staticmethod
    def _read_text(file_path):
        raw = Path(file_path).read_bytes()
        for encoding in ("utf-8-sig", "utf-16", "cp1252"):
            try:
                return raw.decode(encoding)
            except UnicodeError:
                continue
        return raw.decode("utf-8", errors="replace")

    def _extract_text_file(self, file_path):
        return [{"page_number": 1, "text": self._read_text(file_path)}]

    def _extract_csv(self, file_path):
        import io

        import pandas as pd

        frame = pd.read_csv(io.StringIO(self._read_text(file_path)), dtype=str).fillna("")
        return [{"page_number": 1, "text": frame.to_csv(index=False)}]

    def _extract_json(self, file_path):
        data = json.loads(self._read_text(file_path))
        return [{"page_number": 1, "text": json.dumps(data, indent=2, ensure_ascii=False)}]

    def _extract_html(self, file_path):
        from bs4 import BeautifulSoup

        try:
            soup = BeautifulSoup(self._read_text(file_path), "lxml")
        except Exception:
            soup = BeautifulSoup(self._read_text(file_path), "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return [{"page_number": 1, "text": soup.get_text(separator="\n", strip=True)}]
