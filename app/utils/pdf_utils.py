"""
Enhanced PDF utilities using PyMuPDF (fitz) for robust PDF parsing.
Ported from rag-pdf-expert with pdfplumber for table extraction.
"""
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image

try:
    import pdfplumber
    _PDFPLUMBER_AVAILABLE = True
except ImportError:
    _PDFPLUMBER_AVAILABLE = False


class PDFDocument:
    """Context-manager wrapper around a PyMuPDF document."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.doc = fitz.open(str(path))
        self.page_count = len(self.doc)

    def close(self) -> None:
        if self.doc:
            self.doc.close()

    def __enter__(self) -> "PDFDocument":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ── Text extraction ────────────────────────────────────────────────────────

def get_page_texts(path: Path) -> List[str]:
    """Return raw text for each page (backward-compatible helper)."""
    texts: List[str] = []
    with PDFDocument(path) as pdf:
        for i in range(pdf.page_count):
            texts.append(pdf.doc[i].get_text() or "")
    return texts


def get_page_texts_with_layout(path: Path) -> List[Dict]:
    """
    Extract text with layout blocks and page dimensions.

    Returns a list of dicts:
        {"page_num", "text", "blocks", "width", "height"}
    """
    results: List[Dict] = []
    with PDFDocument(path) as pdf:
        for i in range(pdf.page_count):
            page = pdf.doc[i]
            rect = page.rect
            results.append(
                {
                    "page_num": i,
                    "text": page.get_text("text"),
                    "blocks": page.get_text("blocks"),
                    "width": rect.width,
                    "height": rect.height,
                }
            )
    return results


# ── Image extraction ───────────────────────────────────────────────────────

def extract_images_from_page(
    path: Path,
    page_num: int,
    min_width: int = 100,
    min_height: int = 100,
) -> List[Dict]:
    """Extract images from a specific page with bounding-box metadata."""
    images: List[Dict] = []
    with PDFDocument(path) as pdf:
        if page_num >= pdf.page_count:
            return images
        page = pdf.doc[page_num]
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            rects = page.get_image_rects(xref)
            if not rects:
                continue
            bbox = rects[0]
            if bbox.width < min_width or bbox.height < min_height:
                continue
            try:
                base = pdf.doc.extract_image(xref)
                pil_img = Image.open(BytesIO(base["image"]))
                images.append(
                    {
                        "image": pil_img,
                        "bbox": (bbox.x0, bbox.y0, bbox.x1, bbox.y1),
                        "width": bbox.width,
                        "height": bbox.height,
                        "xref": xref,
                        "format": base["ext"],
                    }
                )
            except Exception as exc:
                print(f"Warning: could not extract image xref={xref} page={page_num}: {exc}")
    return images


def extract_all_images(path: Path, output_dir: Optional[Path] = None) -> List[Dict]:
    """Extract every image from the PDF, optionally saving to disk."""
    all_images: List[Dict] = []
    with PDFDocument(path) as pdf:
        for page_num in range(pdf.page_count):
            for img in extract_images_from_page(path, page_num):
                img["page_num"] = page_num
                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    fname = f"page{page_num}_img{img['xref']}.{img['format']}"
                    save_path = output_dir / fname
                    img["image"].save(save_path)
                    img["saved_path"] = str(save_path)
                all_images.append(img)
    return all_images


# ── Table extraction ───────────────────────────────────────────────────────

def extract_tables_from_page(path: Path, page_num: int) -> List[Dict]:
    """Extract tables from a page using pdfplumber (if available)."""
    if not _PDFPLUMBER_AVAILABLE:
        return []
    tables: List[Dict] = []
    with pdfplumber.open(str(path)) as pdf:
        if page_num >= len(pdf.pages):
            return tables
        for tbl in pdf.pages[page_num].find_tables():
            tables.append({"table": tbl.extract(), "bbox": tbl.bbox})
    return tables


# ── Scanned-PDF detection ──────────────────────────────────────────────────

def detect_scanned_pdf(path: Path, sample_pages: int = 3) -> bool:
    """Return True if the PDF appears to be image-based (scanned)."""
    with PDFDocument(path) as pdf:
        pages_to_check = min(sample_pages, pdf.page_count)
        total_len = sum(
            len(pdf.doc[i].get_text().strip()) for i in range(pages_to_check)
        )
        avg = total_len / pages_to_check if pages_to_check else 0
        return avg < 100


# ── Metadata ───────────────────────────────────────────────────────────────

def get_pdf_metadata(path: Path) -> Dict:
    """Return title, author, subject, keywords, page_count."""
    with PDFDocument(path) as pdf:
        m = pdf.doc.metadata
        return {
            "title": m.get("title", ""),
            "author": m.get("author", ""),
            "subject": m.get("subject", ""),
            "keywords": m.get("keywords", ""),
            "creator": m.get("creator", ""),
            "producer": m.get("producer", ""),
            "page_count": pdf.page_count,
        }
