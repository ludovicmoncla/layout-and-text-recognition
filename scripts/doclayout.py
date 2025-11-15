from doclayout_yolo import YOLOv10


def load_model(model_path, device):
    """
    Load the DocLayout-YOLO model from a given path.

    Args:
        model_path (str): Path to the YOLOv10 .pt model file.
        device (str): Device used for inference ('cpu', 'cuda', 'mps').

    Returns:
        YOLOv10: An initialized and ready-to-use DocLayout-YOLO model.
    """
    print(f"Chargement du modèle sur : {device}")
    model = YOLOv10(model_path)
    return model


def run_detection(model, image_path, device, imgsz=1280, conf=0.3):
    """
    Run document layout detection on an image.

    Args:
        model (YOLOv10): Loaded DocLayout-YOLO model.
        image_path (str): Path to the document image.
        device (str): Device used for model inference.
        imgsz (int): Input image size used for prediction. Larger = more accurate.
        conf (float): Minimum confidence threshold for detections.

    Returns:
        Results: YOLO detection results containing boxes, classes, and scores.
    """
    det_res = model.predict(
        image_path,
        imgsz=imgsz,
        conf=conf,
        device=device,
    )
    return det_res[0]  # résultat unique


def extract_text_blocks(det_result, target_class_id=1):
    """
    Extract 'plain text' blocks from the DocLayout-YOLO detection results.

    Args:
        det_result (Results): YOLO detection output (bounding boxes + classes).
        target_class_id (int): Class ID corresponding to plain text blocks.

    Returns:
        list[dict]: A list of dictionaries containing text block coordinates
                    in the format: {"box": (x1, y1, x2, y2)}.
    """
    boxes = det_result.boxes.xyxy.cpu().numpy()
    classes = det_result.boxes.cls.cpu().numpy()

    text_blocks = []

    for box, cls_id in zip(boxes, classes):
        if int(cls_id) == target_class_id:
            x1, y1, x2, y2 = map(int, box)
            text_blocks.append({"box": (x1, y1, x2, y2)})

    return text_blocks






