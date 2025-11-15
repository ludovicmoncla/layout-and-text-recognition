import numpy as np
from lxml import etree
import cv2
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
import os

def sort_blocks_two_columns(text_blocks):
    """
    Sort text blocks from a two-column document into reading order:
    1. Entire left column (top to bottom)
    2. Entire right column (top to bottom)

    The function:
        - finds a horizontal split between the two columns based on the largest
          gap between block centers along the x-axis,
        - assigns blocks to left/right columns using that split,
        - sorts each column by the top coordinate (y1).

    Args:
        text_blocks (list[dict]): List of block dictionaries
                                  {"box": (x1, y1, x2, y2)}.

    Returns:
        list[dict]: Blocks reordered according to the desired reading order.
    """
    if not text_blocks:
        return []

    # 1) Compute x centers for all blocks
    centers = []
    for b in text_blocks:
        x1, y1, x2, y2 = b["box"]
        cx = (x1 + x2) / 2.0
        centers.append((cx, b))

    # 2) Sort by x center
    centers.sort(key=lambda cb: cb[0])  # (center_x, block)

    # 3) Find largest gap between consecutive centers (column gutter)
    x_values = [cb[0] for cb in centers]
    gaps = [x_values[i+1] - x_values[i] for i in range(len(x_values) - 1)]

    # If for some reason there is only one block, just return it
    if not gaps:
        return [b for _, b in centers]

    max_gap_idx = int(np.argmax(gaps))
    # Boundary is mid-point of the largest gap
    boundary_x = (x_values[max_gap_idx] + x_values[max_gap_idx + 1]) / 2.0

    # 4) Assign to left / right columns based on this boundary
    left_col = []
    right_col = []

    for cx, b in centers:
        if cx < boundary_x:
            left_col.append(b)
        else:
            right_col.append(b)

    # 5) Sort each column by y1 (top to bottom)
    left_sorted = sorted(left_col, key=lambda b: b["box"][1])
    right_sorted = sorted(right_col, key=lambda b: b["box"][1])

    # 6) Reading order: entire left column, then entire right column
    return left_sorted + right_sorted


def display_layout(layout, line_width=10, font_size=50, figsize=(16, 16)):
    img_rgb = cv2.cvtColor(layout.plot(pil=True, line_width=line_width, font_size=font_size), cv2.COLOR_BGR2RGB)
    plt.figure(figsize=figsize)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()


def save_layout(layout, output_path, line_width=10, font_size=50):
    cv2.imwrite(output_path, layout.plot(pil=True, line_width=line_width, font_size=font_size))


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

        first = True
        for line_text in content_lines:
            line_text = line_text.replace("\n", "").strip()

            if not line_text:
                continue  # skip empty lines entirely

            if first:
                # First line → add as text of the <div>, not an <lb/>
                div.text = line_text
                first = False
            else:
                # Following lines → add <lb/> with text
                lb = etree.SubElement(div, "lb")
                lb.tail = " " + line_text
        
    return tei


def display_tei(tei):
    print(etree.tostring(tei, pretty_print=True, encoding="UTF-8", xml_declaration=True).decode("utf-8"))


def save_tei(input_content, output_path):
    with open(output_path, "wb") as f:
        f.write(etree.tostring(input_content, pretty_print=True, encoding="UTF-8", xml_declaration=True))


def convert_pdf_to_image(pdf_path, output_path, dpi=300, format="jpg"):

    pages = convert_from_path(pdf_path, dpi=dpi)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    for i, page in enumerate(pages):
        out_path = os.path.join(output_path, f"{base_name}_{i}.jpg")
        page.save(out_path, format, quality=90)
