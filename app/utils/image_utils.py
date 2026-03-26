"""
Image utilities for PDF extraction and OCR (from rag-pdf-expert).
pytesseract and cv2 are optional — functions degrade gracefully if absent.
"""
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple
import hashlib

import numpy as np
from PIL import Image


def pil_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buf = BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    return Image.open(BytesIO(image_bytes))


def save_image(
    image: Image.Image,
    output_path: Path,
    fmt: Optional[str] = None,
    quality: int = 95,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt is None:
        fmt = output_path.suffix.lstrip(".").upper()
        if fmt == "JPG":
            fmt = "JPEG"
    kwargs = {}
    if fmt == "JPEG":
        kwargs["quality"] = quality
        if image.mode == "RGBA":
            rgb = Image.new("RGB", image.size, (255, 255, 255))
            rgb.paste(image, mask=image.split()[3])
            image = rgb
    image.save(output_path, format=fmt, **kwargs)
    return output_path


def get_image_hash(image: Image.Image) -> str:
    return hashlib.sha256(pil_to_bytes(image)).hexdigest()


def assess_image_quality(image: Image.Image) -> dict:
    w, h = image.size
    return {
        "width": w,
        "height": h,
        "aspect_ratio": w / h if h > 0 else 0,
        "is_color": image.mode in ("RGB", "RGBA"),
        "mode": image.mode,
        "format": image.format,
    }


def is_image_mostly_blank(image: Image.Image, threshold: float = 0.95) -> bool:
    gray = np.array(image.convert("L"))
    white_ratio = np.sum(gray > 240) / gray.size
    return white_ratio > threshold


def extract_image_text_with_ocr(image: Image.Image) -> str:
    """OCR text extraction. Requires pytesseract + tesseract binary."""
    try:
        import pytesseract
        import cv2

        arr = np.array(image)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if len(arr.shape) == 3 else arr
        denoised = cv2.fastNlMeansDenoising(gray)
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return pytesseract.image_to_string(Image.fromarray(binary)).strip()
    except ImportError:
        return ""
    except Exception as exc:
        print(f"Warning: OCR failed: {exc}")
        return ""
