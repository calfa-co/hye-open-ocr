"""ALTO XML v4 export with block/line/word coordinates."""

from __future__ import annotations

from typing import List, Optional
from xml.etree import ElementTree as ET

from armenian_ocr.types import Box, Page

ALTO_NS = "http://www.loc.gov/standards/alto/ns-v4#"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
ALTO_SCHEMA = (
    "http://www.loc.gov/standards/alto/ns-v4# "
    "http://www.loc.gov/standards/alto/v4/alto-4-2.xsd"
)


def _set_box(element: ET.Element, box: Box) -> None:
    x1, y1, x2, y2 = box
    element.set("HPOS", str(x1))
    element.set("VPOS", str(y1))
    element.set("WIDTH", str(x2 - x1))
    element.set("HEIGHT", str(y2 - y1))


def _add_polygon(element: ET.Element, poly) -> None:
    """Add an ALTO ``<Shape><Polygon POINTS="x1,y1 x2,y2 …"/></Shape>``.

    Must be the first child (ALTO v4 content model), so call it right after
    ``_set_box`` and before adding TextLines.
    """
    if not poly:
        return
    points = " ".join(f"{int(x)},{int(y)}" for x, y in poly)
    shape = ET.SubElement(element, f"{{{ALTO_NS}}}Shape")
    ET.SubElement(shape, f"{{{ALTO_NS}}}Polygon").set("POINTS", points)


def pages_to_alto(pages: List[Page], source_name: Optional[str] = None) -> str:
    ET.register_namespace("", ALTO_NS)

    alto = ET.Element(f"{{{ALTO_NS}}}alto")
    alto.set(f"{{{XSI_NS}}}schemaLocation", ALTO_SCHEMA)

    description = ET.SubElement(alto, f"{{{ALTO_NS}}}Description")
    ET.SubElement(
        description, f"{{{ALTO_NS}}}MeasurementUnit"
    ).text = "pixel"
    if source_name:
        source_info = ET.SubElement(
            description, f"{{{ALTO_NS}}}sourceImageInformation"
        )
        ET.SubElement(
            source_info, f"{{{ALTO_NS}}}fileName"
        ).text = source_name

    layout = ET.SubElement(alto, f"{{{ALTO_NS}}}Layout")

    for page_index, page in enumerate(pages, start=1):
        page_element = ET.SubElement(layout, f"{{{ALTO_NS}}}Page")
        page_element.set("ID", f"page_{page_index}")
        page_element.set("PHYSICAL_IMG_NR", str(page_index))
        page_element.set("WIDTH", str(page.width))
        page_element.set("HEIGHT", str(page.height))

        print_space = ET.SubElement(
            page_element, f"{{{ALTO_NS}}}PrintSpace"
        )
        _set_box(print_space, (0, 0, page.width, page.height))

        for block_index, paragraph in enumerate(page.paragraphs, start=1):
            block = ET.SubElement(print_space, f"{{{ALTO_NS}}}TextBlock")
            block.set("ID", f"p{page_index}_b{block_index}")
            _set_box(block, paragraph.box)
            _add_polygon(block, getattr(paragraph, "poly", None))

            for line_index, line in enumerate(paragraph.lines, start=1):
                line_element = ET.SubElement(
                    block, f"{{{ALTO_NS}}}TextLine"
                )
                line_element.set(
                    "ID", f"p{page_index}_b{block_index}_l{line_index}"
                )
                _set_box(line_element, line.box)
                _add_polygon(line_element, getattr(line, "poly", None))

                words = [word for word in line.words if word.text]
                for word_index, word in enumerate(words):
                    string = ET.SubElement(
                        line_element, f"{{{ALTO_NS}}}String"
                    )
                    string.set("CONTENT", word.text)
                    _set_box(string, word.box)
                    _add_polygon(string, getattr(word, "poly", None))
                    if word.confidence is not None:
                        string.set(
                            "WC", f"{min(word.confidence, 100) / 100:.2f}"
                        )
                    if word_index < len(words) - 1:
                        ET.SubElement(line_element, f"{{{ALTO_NS}}}SP")

    ET.indent(alto)
    return ET.tostring(
        alto, encoding="unicode", xml_declaration=True
    )
