import torch
from .doclayout import load_model, run_detection, extract_text_blocks
from .utils import sort_blocks_two_columns
from .tesseract import run_ocr_on_blocks



def process_document(model_path, image_path):
    """
    Complete pipeline to extract text from a document image using
    DocLayout-YOLO for layout detection and Tesseract for OCR.

    Pipeline steps:
        1. Load DocLayout-YOLO model.
        2. Run layout detection on the input image.
        3. Extract plain text blocks from detected regions.
        4. Sort blocks into reading order for two-column documents.
        5. Perform OCR on each block using Tesseract.
        6. Assemble full text output.

    Args:
        model_path (str): Path to the DocLayout-YOLO .pt file.
        image_path (str): Path to the document image to process.

    Returns:
        tuple:
            full_text (str): Reconstructed OCR text in correct reading order.
            ordered_blocks (list): Block coordinates sorted by reading order.
            ocr_texts (list): Individual OCR results for each block.
    """

    device = ('cuda' if torch.cuda.is_available() 
              else 'mps' if torch.backends.mps.is_available() 
              else 'cpu')

    # 1. Charger modèle
    model = load_model(model_path, device)

    # 2. Détection
    det_result = run_detection(model, image_path, device)

    # 3. Extraction blocs texte
    blocks = extract_text_blocks(det_result)

    # 4. Tri en ordre de lecture
    ordered_blocks = sort_blocks_two_columns(blocks)

    # 5. OCR Tesseract
    ocr_texts = run_ocr_on_blocks(image_path, ordered_blocks)

    # 6. Fusion texte complet
    full_text = "\n\n".join(ocr_texts)

    return full_text, ordered_blocks, ocr_texts, det_result
