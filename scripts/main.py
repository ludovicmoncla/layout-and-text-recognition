from pipeline import process_document

model_path = "models/doclayout_yolo_docstructbench_imgsz1280_2501.pt"
image_path = "assets/examples/1913247_p15.jpg"

full_text, ordered_blocks, ocr_texts = process_document(model_path, image_path)

print(full_text)