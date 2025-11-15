from scripts.pipeline import process_document
from scripts.utils import save_layout, save_tei
import argparse
import os

# Usage example
# python run_pipeline.py --model models/doclayout_yolo_docstructbench_imgsz1280_2501.pt --image-path assets/examples/1913247_p15.jpg

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, required=True, type=str)
    parser.add_argument('--image-path', default=None, required=True, type=str)
    parser.add_argument('--res-path', default='outputs', required=False, type=str)
    parser.add_argument('--conf', default=0.3, required=False, type=float) # Confidence threshold
    args = parser.parse_args()

    if not os.path.exists(args.res_path):
        os.makedirs(args.res_path)

    tei_content, layout, raw_ordered_text = process_document(args.model, args.image_path, conf=args.conf)

    save_layout(layout, os.path.join(args.res_path, args.image_path.split("/")[-1]))
    save_tei(tei_content, os.path.join(args.res_path, args.image_path.split("/")[-1].replace(".jpg", ".xml")))

    print(f"Results saved to {args.res_path}")

