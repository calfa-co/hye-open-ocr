"""Export tests on a synthetic page (no models needed)."""

import json
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from armenian_ocr.export import (
    pages_to_alto,
    pages_to_dict,
    pages_to_json,
    pages_to_text,
)
from armenian_ocr.types import Line, Page, Paragraph, Word

ARMENIAN = "Հայաստան"
CYRILLIC = "СОЮЗА"


@pytest.fixture
def page():
    def line(y, *texts):
        words, x = [], 100
        for text in texts:
            words.append(
                Word(box=(x, y, x + 120, y + 30), text=text, confidence=92.5)
            )
            x += 140
        return Line(
            box=(100, y, x - 20, y + 30),
            words=words,
            text=" ".join(texts),
            confidence=92.5,
        )

    return Page(
        width=1000,
        height=800,
        dpi=300,
        paragraphs=[
            Paragraph(
                box=(100, 100, 520, 170),
                lines=[line(100, ARMENIAN, CYRILLIC), line(140, "second")],
            ),
            Paragraph(
                box=(100, 300, 240, 330),
                lines=[
                    line(300, "third"),
                    Line(box=(100, 340, 220, 370), words=[], text=""),
                ],
            ),
        ],
    )


def test_text_reading_order_and_paragraph_breaks(page):
    text = pages_to_text([page])
    assert text == f"{ARMENIAN} {CYRILLIC}\nsecond\n\nthird"


def test_text_multi_page_separator(page):
    text = pages_to_text([page, page])
    assert "\f\n" in text


def test_json_structure(page):
    data = pages_to_dict([page])
    assert data["pages"][0]["index"] == 1
    first_word = data["pages"][0]["paragraphs"][0]["lines"][0]["words"][0]
    assert first_word["text"] == ARMENIAN
    assert first_word["box"] == [100, 100, 220, 130]
    # round-trips through json with unicode preserved
    assert ARMENIAN in pages_to_json([page])
    json.loads(pages_to_json([page]))


def test_alto_is_valid_xml_with_coordinates(page):
    alto = pages_to_alto([page], source_name="doc.png")
    root = ET.fromstring(alto)
    ns = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}

    assert root.find(".//alto:MeasurementUnit", ns).text == "pixel"
    assert root.find(".//alto:fileName", ns).text == "doc.png"

    strings = root.findall(".//alto:String", ns)
    contents = [s.get("CONTENT") for s in strings]
    assert ARMENIAN in contents and CYRILLIC in contents

    first = strings[0]
    assert first.get("HPOS") == "100"
    assert first.get("WIDTH") == "120"
    assert first.get("WC") == "0.93"

    # empty lines produce no String elements
    lines = root.findall(".//alto:TextLine", ns)
    assert len(lines) == 4
    assert len(lines[-1]) == 0


def test_alto_and_json_carry_layout_polygon():
    poly = [[10, 20], [110, 22], [108, 60], [12, 58]]
    para = Paragraph(
        box=(10, 20, 110, 60),
        lines=[Line(box=(10, 20, 110, 60), words=[], text="")],
        label="text",
        poly=poly,
    )
    page = Page(width=200, height=200, paragraphs=[para])

    # ALTO: the TextBlock gets a <Shape><Polygon POINTS="x,y …"/></Shape>
    alto = pages_to_alto([page])
    ns = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}
    polygon = ET.fromstring(alto).find(
        ".//alto:TextBlock/alto:Shape/alto:Polygon", ns
    )
    assert polygon is not None
    assert polygon.get("POINTS") == "10,20 110,22 108,60 12,58"

    # JSON: the paragraph carries `poly`
    data = pages_to_dict([page])
    assert data["pages"][0]["paragraphs"][0]["poly"] == poly


def test_searchable_pdf_text_extractable_and_invisible(tmp_path, page):
    pymupdf = pytest.importorskip("pymupdf")
    from armenian_ocr.export.pdf import write_searchable_pdf

    image = np.full((800, 1000, 3), 255, dtype=np.uint8)
    output = tmp_path / "out.pdf"
    write_searchable_pdf([page], [image], output)

    with pymupdf.open(output) as doc:
        assert doc.page_count == 1
        extracted = doc[0].get_text()
        assert ARMENIAN in extracted
        assert CYRILLIC in extracted

        # invisible text: rendering must stay blank (white background)
        pixmap = doc[0].get_pixmap(dpi=72)
        samples = np.frombuffer(pixmap.samples, dtype=np.uint8)
        assert samples.min() > 240

        # text is positioned where the word box is (x=100px at 300 dpi
        # -> 24pt), tolerance for font metrics
        rects = doc[0].search_for(ARMENIAN)
        assert rects, "Armenian word not searchable"
        assert abs(rects[0].x0 - 24) < 6
