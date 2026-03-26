"""
Enhanced PDF service for complex PDF processing.

Handles:
- Text extraction with layout preservation
- Image extraction with position tracking
- Table detection
- OCR for scanned documents
- Metadata linking between text and images
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

from app.utils.pdf_utils import (
    get_page_texts,
    get_page_texts_with_layout,
    extract_images_from_page,
    extract_all_images,
    extract_tables_from_page,
    detect_scanned_pdf,
    get_pdf_metadata,
)
from app.utils.image_utils import (
    save_image,
    get_image_hash,
    extract_image_text_with_ocr,
    is_image_mostly_blank,
    assess_image_quality,
)

logger = logging.getLogger(__name__)


class PDFService:
    """
    Enhanced PDF parsing service with support for:
    - Complex layouts
    - Image extraction with metadata
    - Table detection
    - OCR for scanned documents
    """

    def __init__(
        self,
        extract_images: bool = True,
        extract_tables: bool = True,
        use_ocr: bool = True,
        min_image_size: Tuple[int, int] = (100, 100),
    ) -> None:
        """
        Initialize PDF service.
        
        Args:
            extract_images: Whether to extract images from PDFs
            extract_tables: Whether to extract tables
            use_ocr: Whether to use OCR on images and scanned pages
            min_image_size: Minimum (width, height) for image extraction
        """
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.use_ocr = use_ocr
        self.min_image_size = min_image_size

    def extract_text_per_page(self, pdf_path: Path) -> List[str]:
        """
        Extract text from each page (backward compatible).
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            List of text strings, one per page
        """
        return get_page_texts(pdf_path)

    def extract_text_with_layout(self, pdf_path: Path) -> List[Dict]:
        """
        Extract text with layout information.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            List of dicts with page text and layout info
        """
        return get_page_texts_with_layout(pdf_path)

    def extract_images_from_pdf(
        self,
        pdf_path: Path,
        output_dir: Optional[Path] = None,
        filter_blank: bool = True,
    ) -> List[Dict]:
        """
        Extract all images from PDF with metadata.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save extracted images
            filter_blank: Whether to filter out mostly blank images
        
        Returns:
            List of dicts containing:
                - page_num: Page number
                - image: PIL Image object
                - bbox: Bounding box (x0, y0, x1, y1)
                - image_id: Unique image identifier
                - saved_path: Path to saved image (if output_dir provided)
                - ocr_text: OCR text from image (if use_ocr=True)
        """
        if not self.extract_images:
            return []

        images = extract_all_images(pdf_path, output_dir)
        
        # Filter and enhance image metadata
        enhanced_images = []
        for img_data in images:
            # Filter blank images
            if filter_blank and is_image_mostly_blank(img_data["image"]):
                logger.debug(f"Skipping blank image on page {img_data['page_num']}")
                continue
            
            # Generate unique image ID
            img_hash = get_image_hash(img_data["image"])
            img_data["image_id"] = f"img_{img_data['page_num']}_{img_hash[:8]}"
            
            # Extract text from image using OCR if enabled
            if self.use_ocr:
                try:
                    ocr_text = extract_image_text_with_ocr(img_data["image"])
                    img_data["ocr_text"] = ocr_text
                except Exception as e:
                    logger.warning(f"OCR failed for image on page {img_data['page_num']}: {e}")
                    img_data["ocr_text"] = ""
            
            # Add quality metrics
            img_data["quality"] = assess_image_quality(img_data["image"])
            
            enhanced_images.append(img_data)
        
        return enhanced_images

    def extract_tables_from_pdf(self, pdf_path: Path) -> Dict[int, List[Dict]]:
        """
        Extract tables from all pages.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Dict mapping page_num to list of table dicts
        """
        if not self.extract_tables:
            return {}

        tables_by_page = {}
        
        # Get page count
        page_texts = get_page_texts(pdf_path)
        
        for page_num in range(len(page_texts)):
            try:
                tables = extract_tables_from_page(pdf_path, page_num)
                if tables:
                    tables_by_page[page_num] = tables
            except Exception as e:
                logger.warning(f"Table extraction failed for page {page_num}: {e}")
        
        return tables_by_page

    def process_pdf_comprehensive(
        self,
        pdf_path: Path,
        output_dir: Optional[Path] = None,
    ) -> Dict:
        """
        Comprehensive PDF processing extracting all content types.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory for extracted images
        
        Returns:
            Dict containing:
                - metadata: PDF metadata
                - pages: List of page data with text and layout
                - images: List of extracted images with metadata
                - tables: Dict of tables by page
                - is_scanned: Whether PDF appears to be scanned
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Get metadata
        metadata = get_pdf_metadata(pdf_path)
        
        # Detect if scanned
        is_scanned = detect_scanned_pdf(pdf_path)
        
        # Extract text with layout
        pages = self.extract_text_with_layout(pdf_path)
        
        # Extract images
        images = self.extract_images_from_pdf(pdf_path, output_dir)
        
        # Extract tables
        tables = self.extract_tables_from_pdf(pdf_path)
        
        # Link images to pages
        for img_data in images:
            page_num = img_data["page_num"]
            if page_num < len(pages):
                if "images" not in pages[page_num]:
                    pages[page_num]["images"] = []
                pages[page_num]["images"].append(img_data["image_id"])
        
        # Link tables to pages
        for page_num, page_tables in tables.items():
            if page_num < len(pages):
                pages[page_num]["tables"] = page_tables
        
        result = {
            "metadata": metadata,
            "pages": pages,
            "images": images,
            "tables": tables,
            "is_scanned": is_scanned,
        }
        
        logger.info(
            f"Processed {metadata['page_count']} pages, "
            f"extracted {len(images)} images, "
            f"found {sum(len(t) for t in tables.values())} tables"
        )
        
        return result

    def extract_images(self, pdf_path: Path) -> List[Dict]:
        """
        Backward compatible method for image extraction.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            List of image data dicts
        """
        return self.extract_images_from_pdf(pdf_path)
