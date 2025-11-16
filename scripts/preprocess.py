import cv2
import numpy as np
from skimage.filters import threshold_sauvola

def brighten(img, gamma=1.2):
    # gamma < 1 = brighten, gamma > 1 = darken
    inv_gamma = 1.0 / gamma
    table = ( ( (np.arange(256) / 255.0) ** inv_gamma ) * 255 ).astype("uint8")
    return cv2.LUT(img, table)


def white_balance(img):
    """
    Simple and effective white balance for scanned documents.
    Removes yellow/brown cast by normalizing A/B channels in LAB space.
    """
    if img is None:
        raise ValueError("white_balance(): input image is None")

    # Convert BGR → LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Compute average A/B
    avg_a = np.mean(a)
    avg_b = np.mean(b)

    # Adjust A/B to reduce color cast
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    l_float = l.astype(np.float32)

    a = a - ((avg_a - 128) * (l_float / 255.0) * 1.1)
    b = b - ((avg_b - 128) * (l_float / 255.0) * 1.1)

    # Convert back to uint8
    a = np.clip(a, 0, 255).astype("uint8")
    b = np.clip(b, 0, 255).astype("uint8")

    # Rebuild LAB image
    lab = cv2.merge([l, a, b])

    # Convert back to BGR
    balanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return balanced


def remove_shadows(img):
    """
    Remove scanner shadows and uneven illumination using
    morphological background estimation.
    Works for grayscale or color input.
    """
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Estimate the background via large morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (51, 51))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # Subtract background from image
    diff = cv2.absdiff(gray, bg)

    # Normalize contrast
    norm = cv2.normalize(diff, None, alpha=0, beta=255,
                         norm_type=cv2.NORM_MINMAX)

    return norm


def crop_margins(img, margin=10):
    """
    Auto-crop margins by detecting the bounding box of all text pixels.
    Keeps a small optional margin.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = cv2.findNonZero(thresh)
    if coords is None:
        # No text detected
        return img

    x, y, w, h = cv2.boundingRect(coords)

    # Keep a small margin (in pixels)
    x = max(x - margin, 0)
    y = max(y - margin, 0)
    w = min(w + 2 * margin, img.shape[1] - x)
    h = min(h + 2 * margin, img.shape[0] - y)

    return img[y:y+h, x:x+w]


def deskew_opencv(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 80, 200)

    # Detect lines using Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=150)

    if lines is None:
        print("No lines detected, skipping deskew.")
        return img

    angles = []

    for rho, theta in lines[:,0]:
        angle = (theta - np.pi/2) * 180/np.pi
        angles.append(angle)

    # Median angle is robust to noise
    median_angle = np.median(angles)

    print(f"Deskewing by {median_angle:.2f}°")

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), median_angle, 1)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)



def preprocess_for_layout(img):
    """
    Optimal preprocessing for DocLayout-YOLO.
    Includes:
    - deskew
    - white balance
    - shadow removal
    - margin cropping
    - controlled brightening
    - CLAHE
    - light smoothing
    """

    # STEP 1 — Deskew
    #img = deskew_opencv(img)

    # STEP 2 — Remove shadows BEFORE white balance
    shadow_free = remove_shadows(img)

    # STEP 3 — White balance on the shadow-free version
    # Convert back to 3-channel for white balance
    shadow_rgb = cv2.cvtColor(shadow_free, cv2.COLOR_GRAY2BGR)
    wb = white_balance(shadow_rgb)

    # STEP 4 — Crop margins
    #cropped = crop_margins(wb, margin=15)

    # STEP 5 — Grayscale
    gray = cv2.cvtColor(wb, cv2.COLOR_BGR2GRAY)

    # STEP 6 — Gentle brightening
    #gray = brighten(gray, gamma=1.2)

    # STEP 7 — CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(gray)

    # STEP 8 — Light smoothing
    #enhanced = cv2.medianBlur(enhanced, 3)

    # STEP 9 — Optional: light denoise
    enhanced = cv2.fastNlMeansDenoising(enhanced, None, 3, 7, 21)

    return enhanced



def preprocess_for_ocr(img):
    """
    Preprocessing optimized for Tesseract OCR.
    Does NOT apply CLAHE (to avoid distorting glyphs).
    Uses Sauvola binarization for robustness on old paper.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    th = threshold_sauvola(gray, window_size=25)
    ocr_img = (gray > th).astype("uint8") * 255

    return ocr_img



def preprocess(image_path):
    """
    Load image and produce TWO preprocessed versions:
    - one for layout detection (brightened, contrast-enhanced)
    - one for OCR (binarized carefully)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    layout_img = preprocess_for_layout(img)
    ocr_img = preprocess_for_ocr(img)

    return layout_img, ocr_img

