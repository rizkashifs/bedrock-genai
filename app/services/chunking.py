from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Literal

from app.utils.text_utils import normalize_whitespace, split_into_sentences


@dataclass
class Chunk:
    """
    Enhanced chunk representation supporting both text and image content.
    
    Attributes:
        id: Unique identifier for the chunk
        page: Page number (0-indexed)
        text: Text content (for text chunks) or image description (for image chunks)
        source_document_id: ID of the source document
        chunk_type: Type of chunk - 'text' or 'image'
        metadata: Additional metadata including:
            - image_path: Path to extracted image file (for image chunks)
            - bbox: Bounding box coordinates (x0, y0, x1, y1) on the page
            - linked_image_ids: List of image chunk IDs that appear in this text chunk
            - linked_text_id: Text chunk ID that this image belongs to (for image chunks)
            - block_type: Type of content block (paragraph, heading, table, etc.)
            - confidence: OCR confidence score if applicable
    """
    id: str
    page: int
    text: str
    source_document_id: str
    chunk_type: Literal["text", "image"] = "text"
    metadata: Optional[dict] = field(default_factory=dict)


def chunk_page_text(
    text: str,
    page_index: int,
    source_document_id: str,
    max_tokens: int = 256,
) -> List[Chunk]:
    """
    Simple sentence-based chunking that approximates token limits.
    Later this can be replaced with a PDF-aware, layout-preserving strategy.
    """
    normalized = normalize_whitespace(text)
    sentences = split_into_sentences(normalized)

    chunks: List[Chunk] = []
    current: List[str] = []
    current_len = 0
    counter = 0

    for sent in sentences:
        sent_len = len(sent.split())
        if current and current_len + sent_len > max_tokens:
            counter += 1
            chunks.append(
                Chunk(
                    id=f"{source_document_id}-p{page_index}-c{counter}",
                    page=page_index,
                    text=" ".join(current),
                    source_document_id=source_document_id,
                )
            )
            current = []
            current_len = 0

        current.append(sent)
        current_len += sent_len

    if current:
        counter += 1
        chunks.append(
            Chunk(
                id=f"{source_document_id}-p{page_index}-c{counter}",
                page=page_index,
                text=" ".join(current),
                source_document_id=source_document_id,
            )
        )

    return chunks


def create_image_chunk(
    image_data: dict,
    source_document_id: str,
    chunk_counter: int,
) -> Chunk:
    """
    Create a chunk for an extracted image.
    
    Args:
        image_data: Dict containing image metadata from PDFService
        source_document_id: Document ID
        chunk_counter: Counter for unique chunk ID
    
    Returns:
        Chunk object for the image
    """
    page_num = image_data["page_num"]
    image_id = image_data.get("image_id", f"img_{chunk_counter}")
    
    # Use OCR text if available, otherwise create description
    text = image_data.get("ocr_text", "")
    if not text:
        # Create a basic description
        quality = image_data.get("quality", {})
        text = f"Image on page {page_num + 1} ({quality.get('width', 0)}x{quality.get('height', 0)})"
    
    metadata = {
        "image_id": image_id,
        "bbox": image_data.get("bbox"),
        "image_path": image_data.get("saved_path"),
        "width": image_data.get("width"),
        "height": image_data.get("height"),
        "format": image_data.get("format"),
        "quality": image_data.get("quality", {}),
    }
    
    chunk_id = f"{source_document_id}-p{page_num}-img{chunk_counter}"
    
    return Chunk(
        id=chunk_id,
        page=page_num,
        text=text,
        source_document_id=source_document_id,
        chunk_type="image",
        metadata=metadata,
    )


def chunk_page_with_layout(
    page_data: dict,
    page_index: int,
    source_document_id: str,
    max_tokens: int = 256,
    images_data: Optional[List[dict]] = None,
) -> List[Chunk]:
    """
    Enhanced chunking that preserves layout and links images.
    
    Args:
        page_data: Page data from PDFService with text and blocks
        page_index: Page number (0-indexed)
        source_document_id: Document ID
        max_tokens: Maximum tokens per chunk
        images_data: List of image data for this page
    
    Returns:
        List of text and image chunks with proper linking
    """
    chunks: List[Chunk] = []
    
    # Extract text blocks with positions
    blocks = page_data.get("blocks", [])
    page_width = page_data.get("width", 0)
    page_height = page_data.get("height", 0)
    
    # If no blocks, fall back to simple chunking
    if not blocks:
        text = page_data.get("text", "")
        return chunk_page_text(text, page_index, source_document_id, max_tokens)
    
    # Process text blocks
    text_counter = 0
    image_ids_on_page = []
    
    for block in blocks:
        # Block format: (x0, y0, x1, y1, text, block_no, block_type)
        if len(block) < 5:
            continue
        
        x0, y0, x1, y1 = block[0:4]
        block_text = block[4] if len(block) > 4 else ""
        block_type = block[6] if len(block) > 6 else 0
        
        if not block_text.strip():
            continue
        
        # Chunk the block text if it's too long
        block_chunks = chunk_page_text(
            block_text,
            page_index,
            source_document_id,
            max_tokens
        )
        
        # Enhance chunks with layout metadata
        for chunk in block_chunks:
            text_counter += 1
            chunk.id = f"{source_document_id}-p{page_index}-t{text_counter}"
            chunk.metadata.update({
                "bbox": (x0, y0, x1, y1),
                "block_type": "text",
                "linked_image_ids": [],  # Will be populated later
            })
            chunks.append(chunk)
    
    # Create image chunks
    if images_data:
        image_counter = 0
        for img_data in images_data:
            if img_data["page_num"] != page_index:
                continue
            
            image_counter += 1
            img_chunk = create_image_chunk(img_data, source_document_id, image_counter)
            image_ids_on_page.append(img_chunk.id)
            
            # Find nearest text chunk to link
            img_bbox = img_data.get("bbox")
            if img_bbox and chunks:
                nearest_chunk = _find_nearest_text_chunk(chunks, img_bbox)
                if nearest_chunk:
                    img_chunk.metadata["linked_text_id"] = nearest_chunk.id
                    if "linked_image_ids" not in nearest_chunk.metadata:
                        nearest_chunk.metadata["linked_image_ids"] = []
                    nearest_chunk.metadata["linked_image_ids"].append(img_chunk.id)
            
            chunks.append(img_chunk)
    
    return chunks


def _find_nearest_text_chunk(text_chunks: List[Chunk], image_bbox: Tuple) -> Optional[Chunk]:
    """
    Find the text chunk nearest to an image based on spatial proximity.
    
    Args:
        text_chunks: List of text chunks
        image_bbox: Image bounding box (x0, y0, x1, y1)
    
    Returns:
        Nearest text chunk or None
    """
    if not text_chunks or not image_bbox:
        return None
    
    img_x0, img_y0, img_x1, img_y1 = image_bbox
    img_center_x = (img_x0 + img_x1) / 2
    img_center_y = (img_y0 + img_y1) / 2
    
    min_distance = float('inf')
    nearest_chunk = None
    
    for chunk in text_chunks:
        if chunk.chunk_type != "text":
            continue
        
        chunk_bbox = chunk.metadata.get("bbox")
        if not chunk_bbox:
            continue
        
        # Calculate distance between centers
        chunk_x0, chunk_y0, chunk_x1, chunk_y1 = chunk_bbox
        chunk_center_x = (chunk_x0 + chunk_x1) / 2
        chunk_center_y = (chunk_y0 + chunk_y1) / 2
        
        distance = ((img_center_x - chunk_center_x) ** 2 + 
                   (img_center_y - chunk_center_y) ** 2) ** 0.5
        
        if distance < min_distance:
            min_distance = distance
            nearest_chunk = chunk
    
    return nearest_chunk

