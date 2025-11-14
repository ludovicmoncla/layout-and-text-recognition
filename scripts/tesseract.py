import pytesseract
import cv2
from PIL import Image


def run_ocr_on_blocks(image_path, ordered_blocks):
    """
    Apply Tesseract OCR on each ordered text block extracted from the document.

    Args:
        image_path (str): Path to the original (unresized) document image.
        ordered_blocks (list[dict]): List of sorted blocks with bounding boxes.

    Returns:
        list[str]: OCR output for each block, in reading order.
    """
    
    img = cv2.imread(image_path)
    ocr_outputs = []

    for b in ordered_blocks:
        x1, y1, x2, y2 = b["box"]

        crop = img[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_crop = Image.fromarray(crop_rgb)

        text = pytesseract.image_to_string(pil_crop, lang="fra")
        ocr_outputs.append(text.strip())

    return ocr_outputs