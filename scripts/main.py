from pipeline import process_document
from utils import display_layout, display_tei

model_path = "models/doclayout_yolo_docstructbench_imgsz1280_2501.pt"
image_path = "assets/examples/1913247_p15.jpg"

tei_content, layout, raw_ordered_text = process_document(model_path, image_path)

display_tei(tei_content)

