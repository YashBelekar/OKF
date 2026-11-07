import os
import shutil
from pathlib import Path

WINDOWS_DEFAULT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class OCRService:
    """Thin wrapper around Tesseract. Safe to construct when it is not installed."""

    def __init__(self, tesseract_path=None, language="eng", zoom=2.0):
        self.language = language
        self.zoom = zoom
        self._available = None
        self.tesseract_path = (
            tesseract_path
            or os.environ.get("TESSERACT_CMD")
            or shutil.which("tesseract")
            or (WINDOWS_DEFAULT if Path(WINDOWS_DEFAULT).exists() else None)
        )

    def is_available(self):
        if self._available is None:
            try:
                import pytesseract

                if self.tesseract_path:
                    pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
                pytesseract.get_tesseract_version()
                self._available = True
            except Exception:
                self._available = False
        return self._available

    def image_to_text(self, image):
        """OCR a PIL image."""
        import pytesseract

        return pytesseract.image_to_string(image, lang=self.language)

    def page_to_text(self, page):
        """OCR a PyMuPDF page, rendered in memory (no temp files)."""
        import fitz
        from PIL import Image

        pixmap = page.get_pixmap(matrix=fitz.Matrix(self.zoom, self.zoom))
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        return self.image_to_text(image)
