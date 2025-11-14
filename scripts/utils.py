import numpy as np
from lxml import etree

def sort_blocks_two_columns(text_blocks):
    """
    Sort text blocks from a two-column document into logical reading order.

    Sorting strategy:
        1. Compute the median X position to determine column separation.
        2. Assign each block to the left or right column based on its center X.
        3. Sort each column from top to bottom (by y1).
        4. Merge results: left column first, then right column.

    Args:
        text_blocks (list[dict]): List of block dictionaries
                                  {"box": (x1, y1, x2, y2)}.

    Returns:
        list[dict]: Blocks reordered according to natural reading order.
    """
    if not text_blocks:
        return []

    # Calculer la médiane X
    mid_x = np.median([(b["box"][0] + b["box"][2]) / 2 for b in text_blocks])

    left_col, right_col = [], []

    for b in text_blocks:
        x1, y1, x2, y2 = b["box"]
        center_x = (x1 + x2) / 2

        if center_x < mid_x:
            left_col.append(b)
        else:
            right_col.append(b)

    # Trier chaque colonne par Y
    left_sorted = sorted(left_col, key=lambda b: b["box"][1])
    right_sorted = sorted(right_col, key=lambda b: b["box"][1])

    # Fusion
    return left_sorted + right_sorted


def ocr_texts_to_tei(ocr_texts):
    """
    Convert OCR results a simple XML-TEI format.

    Args:
        ocr_texts (list[str]): OCR text for each block.

    Returns:
        str: XML string representing the OCR results with coordinates.
    """
   
    NS_TEI = "http://www.tei-c.org/ns/1.0"
    tei = etree.Element("{%s}TEI" % NS_TEI, nsmap={None: NS_TEI})
    text = etree.SubElement(tei, "text")
    body = etree.SubElement(text, "body")

    for content in ocr_texts:
        div = etree.SubElement(body, "div")

        content_lines = content.split("\n")
        for line_text in content_lines:
            lb = etree.SubElement(div, "lb")
            lb.tail = " " + " ".join(line_text) if line_text else ""
        
    return tei


def display_tei(tei):
    print(etree.tostring(tei, pretty_print=True, encoding="UTF-8", xml_declaration=True).decode("utf-8"))


def save_tei(input_content, output_path):
    with open(output_path, "wb") as f:
        f.write(etree.tostring(input_content, pretty_print=True, encoding="UTF-8", xml_declaration=True))