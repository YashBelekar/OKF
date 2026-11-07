import json

import pytest

from ingestion.document_loader import DocumentLoader
from ingestion.text_extractor import ExtractionError, TextExtractor


@pytest.fixture
def extractor():
    return TextExtractor()


def test_txt_and_md(tmp_path, extractor):
    for name in ("a.txt", "b.md"):
        path = tmp_path / name
        path.write_text("Solar panels convert light.", encoding="utf-8")
        assert "Solar panels" in extractor.extract(path)[0]["text"]


def test_csv(tmp_path, extractor):
    path = tmp_path / "t.csv"
    path.write_text("name,qty\nwidget,5\n", encoding="utf-8")
    assert "widget,5" in extractor.extract(path)[0]["text"]


def test_json(tmp_path, extractor):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"city": "Pune"}), encoding="utf-8")
    assert "Pune" in extractor.extract(path)[0]["text"]


def test_html_strips_scripts(tmp_path, extractor):
    path = tmp_path / "t.html"
    path.write_text("<html><script>var secret=1;</script><body><p>Hello world</p></body></html>")
    text = extractor.extract(path)[0]["text"]
    assert "Hello world" in text and "secret" not in text


def test_docx(tmp_path, extractor):
    from docx import Document
    doc = Document()
    doc.add_paragraph("Wind turbines generate power.")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Key"
    table.rows[0].cells[1].text = "Value"
    path = tmp_path / "t.docx"
    doc.save(path)
    text = extractor.extract(path)[0]["text"]
    assert "Wind turbines" in text and "Key | Value" in text


def test_xlsx_has_one_page_per_sheet(tmp_path, extractor):
    import pandas as pd
    path = tmp_path / "t.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"a": [1]}).to_excel(writer, sheet_name="One", index=False)
        pd.DataFrame({"b": [2]}).to_excel(writer, sheet_name="Two", index=False)
    pages = extractor.extract(path)
    assert [p["page_number"] for p in pages] == [1, 2]
    assert "Sheet: Two" in pages[1]["text"]


def test_pptx_slide_numbers(tmp_path, extractor):
    from pptx import Presentation
    prs = Presentation()
    for title in ("Intro slide", "Second slide"):
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        slide.shapes.title.text = title
    path = tmp_path / "t.pptx"
    prs.save(path)
    pages = extractor.extract(path)
    assert pages[1]["page_number"] == 2 and "Second slide" in pages[1]["text"]


def test_pdf(tmp_path, extractor):
    pytest.importorskip("pypdf")
    reportlab = pytest.importorskip("reportlab.pdfgen.canvas")
    path = tmp_path / "t.pdf"
    canvas = reportlab.Canvas(str(path))
    canvas.drawString(72, 720, "Geothermal energy page one")
    canvas.showPage()
    canvas.drawString(72, 720, "Hydro power page two")
    canvas.save()
    pages = extractor.extract(path)
    assert pages[0]["page_number"] == 1 and "Geothermal" in pages[0]["text"]
    assert "Hydro" in pages[1]["text"]


def test_corrupt_file_raises_extraction_error(tmp_path, extractor):
    path = tmp_path / "bad.docx"
    path.write_bytes(b"not a real docx")
    with pytest.raises(ExtractionError):
        extractor.extract(path)


def test_loader_rejects_bad_uploads(tmp_path):
    loader = DocumentLoader(tmp_path)
    with pytest.raises(ValueError):
        loader.prepare("virus.exe", b"x")
    with pytest.raises(ValueError):
        loader.prepare("empty.txt", b"")


def test_loader_sanitises_names_and_hashes(tmp_path):
    loader = DocumentLoader(tmp_path)
    record, path = loader.load_bytes("../evil name!.txt", b"hello")
    assert path.parent == tmp_path and path.exists()
    assert "/" not in record["stored_filename"] and " " not in record["stored_filename"]
    assert record["document_id"] == record["sha256"][:16]
