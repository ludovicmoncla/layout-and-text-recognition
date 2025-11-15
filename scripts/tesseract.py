import pytesseract
from PIL import Image


def run_ocr_on_blocks(img_ocr, ordered_blocks):
    """
    Apply Tesseract OCR on each ordered text block extracted from the document.

    Args:
        img_ocr (np.ndarray): sauvola-binarized grayscale image
        ordered_blocks (list[dict]): List of sorted blocks with bounding boxes.

    Returns:
        list[str]: OCR output for each block, in reading order.
    """
    ocr_texts = []

    for b in ordered_blocks:
        x1, y1, x2, y2 = b["box"]
        crop = img_ocr[y1:y2, x1:x2]

        pil = Image.fromarray(crop)
        text = pytesseract.image_to_string(pil, lang="fra")
        ocr_texts.append(text.strip())

    return ocr_texts
