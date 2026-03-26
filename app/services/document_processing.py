"""
Document processing service — production pipeline.

Orchestrates the full ingestion flow:
  PDF  → PDFService → Chunk list → RetrievalService (Titan embeddings)
  CSV  → IngestionService → OrchestrationService (SQL/Text engine)
  Image → direct Bedrock multimodal invocation

Entry points
------------
process_pdf(path, document_id)      → List[Chunk]   (text + image chunks)
process_csv(file_path)              → data_bundle   (used by orchestration)
process_image(file_path, question)  → str           (Claude answer)
process_file(file_path, question)   → dict          (unified response)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from app.models.bedrock_client import bedrock_client, bedrock
from app.services.chunking import Chunk
from app.services.pdf_service import PDFService
from app.services.chunking import chunk_page_text, chunk_page_with_layout, create_image_chunk
from app.utils.logger import logger
from app.config.settings import settings


# ── PDF processing ─────────────────────────────────────────────────────────

class DocumentProcessingService:
    """
    High-level pipeline: PDF → pages → chunks (text + images).
    Delegates layout/image extraction to PDFService and chunking helpers.
    """

    def __init__(
        self,
        pdf_service: Optional[PDFService] = None,
        use_layout_chunking: bool = True,
        extract_images: bool = True,
        images_output_dir: Optional[Path] = None,
    ) -> None:
        self.pdf_service = pdf_service or PDFService(
            extract_images=extract_images,
            extract_tables=True,
            use_ocr=True,
        )
        self.use_layout_chunking = use_layout_chunking
        self.extract_images = extract_images
        self.images_output_dir = images_output_dir

    def process_pdf(self, pdf_path: Path, document_id: str) -> List[Chunk]:
        """Basic chunking — text only (fast, backward compatible)."""
        logger.info("DocumentProcessingService: basic mode — %s", pdf_path)
        page_texts = self.pdf_service.extract_text_per_page(pdf_path)
        chunks: List[Chunk] = []
        for i, text in enumerate(page_texts):
            chunks.extend(chunk_page_text(text, page_index=i, source_document_id=document_id))
        logger.info("Created %d text chunks", len(chunks))
        return chunks

    def process_pdf_enhanced(self, pdf_path: Path, document_id: str) -> List[Chunk]:
        """Enhanced chunking — layout-aware + images + tables."""
        logger.info("DocumentProcessingService: enhanced mode — %s", pdf_path)
        pdf_data = self.pdf_service.process_pdf_comprehensive(
            pdf_path, output_dir=self.images_output_dir
        )
        pages = pdf_data["pages"]
        images = pdf_data["images"]
        tables = pdf_data["tables"]
        logger.info(
            "PDF analysis: %d pages, %d images, scanned=%s",
            len(pages), len(images), pdf_data["is_scanned"],
        )

        all_chunks: List[Chunk] = []
        for page_index, page_data in enumerate(pages):
            page_images = [img for img in images if img["page_num"] == page_index]

            if self.use_layout_chunking:
                chunks = chunk_page_with_layout(
                    page_data=page_data,
                    page_index=page_index,
                    source_document_id=document_id,
                    images_data=page_images if self.extract_images else None,
                )
            else:
                chunks = chunk_page_text(
                    text=page_data.get("text", ""),
                    page_index=page_index,
                    source_document_id=document_id,
                )
                if self.extract_images and page_images:
                    for idx, img_data in enumerate(page_images):
                        chunks.append(create_image_chunk(img_data, document_id, idx + 1))

            if page_index in tables:
                page_tables = tables[page_index]
                for chunk in chunks:
                    if chunk.chunk_type == "text":
                        chunk.metadata["has_tables"] = bool(page_tables)
                        chunk.metadata["table_count"] = len(page_tables)

            all_chunks.extend(chunks)

        text_chunks = [c for c in all_chunks if c.chunk_type == "text"]
        img_chunks = [c for c in all_chunks if c.chunk_type == "image"]
        logger.info("Created %d text + %d image chunks", len(text_chunks), len(img_chunks))
        return all_chunks

    def get_processing_stats(self, chunks: List[Chunk]) -> dict:
        text_chunks = [c for c in chunks if c.chunk_type == "text"]
        image_chunks = [c for c in chunks if c.chunk_type == "image"]
        return {
            "total_chunks": len(chunks),
            "text_chunks": len(text_chunks),
            "image_chunks": len(image_chunks),
            "pages": len({c.page for c in chunks}),
            "avg_text_length": (
                sum(len(c.text) for c in text_chunks) / len(text_chunks) if text_chunks else 0
            ),
            "chunks_with_images": len(
                [c for c in text_chunks if c.metadata.get("linked_image_ids")]
            ),
        }


# ── Image processing (direct multimodal) ──────────────────────────────────

def process_image(file_path: str, user_question: str = "Describe this image.") -> str:
    """
    Send an image directly to Claude via Bedrock multimodal API.
    Only supported by Sonnet-class models.
    """
    if settings.model_type == "haiku":
        logger.error(
            "Image processing skipped — model %s does not support multimodal input.",
            settings.bedrock_model_id,
        )
        return "Image processing is not supported by the configured model. Use a Sonnet model."

    path = Path(file_path)
    if not path.exists():
        logger.error("Image not found: %s", file_path)
        return f"File not found: {file_path}"

    try:
        image_bytes = path.read_bytes()
        media_type = "png" if path.suffix.lower() == ".png" else "jpeg"

        messages = [
            {
                "role": "user",
                "content": [
                    {"image": {"format": media_type, "source": {"bytes": image_bytes}}},
                    {"text": f"Here is the image '{path.name}'. {user_question}"},
                ],
            }
        ]

        response = bedrock.converse(
            modelId=settings.bedrock_model_id,
            messages=messages,
            inferenceConfig={"maxTokens": settings.max_tokens, "temperature": 0.0},
        )
        return response["output"]["message"]["content"][0]["text"]
    except Exception as exc:
        logger.error("Image processing failed: %s", exc)
        return f"Error processing image: {exc}"


# ── Unified file dispatcher ────────────────────────────────────────────────

def process_file(
    file_path: str,
    user_question: str = "",
    document_id: Optional[str] = None,
    enhanced: bool = True,
) -> Dict[str, Any]:
    """
    Dispatch a file to the correct processing pipeline.

    Returns:
        {
            "type": "pdf" | "csv" | "image",
            "status": "ok" | "error",
            "chunks": List[Chunk],          # PDF only
            "data_bundle": dict,            # CSV only
            "answer": str,                  # image only
            "message": str,
        }
    """
    path = Path(file_path)
    if not path.exists():
        return {"type": "unknown", "status": "error", "message": f"File not found: {file_path}"}

    doc_id = document_id or path.stem
    ext = path.suffix.lower()

    if ext == ".pdf":
        svc = DocumentProcessingService()
        try:
            chunks = svc.process_pdf_enhanced(path, doc_id) if enhanced else svc.process_pdf(path, doc_id)
            return {
                "type": "pdf",
                "status": "ok",
                "chunks": chunks,
                "stats": svc.get_processing_stats(chunks),
                "message": f"PDF processed: {len(chunks)} chunks",
            }
        except Exception as exc:
            logger.error("PDF processing failed: %s", exc)
            return {"type": "pdf", "status": "error", "message": str(exc)}

    elif ext in {".jpg", ".jpeg", ".png"}:
        answer = process_image(file_path, user_question or "Describe this image.")
        return {"type": "image", "status": "ok", "answer": answer, "message": "Image processed"}

    elif ext in {".csv", ".xlsx", ".xls"}:
        from app.services.ingestion import ingestion_service
        try:
            data_bundle = ingestion_service.load_data(file_path)
            return {
                "type": "csv",
                "status": "ok",
                "data_bundle": data_bundle,
                "message": f"CSV/Excel loaded: {data_bundle['row_count']} rows",
            }
        except Exception as exc:
            logger.error("CSV ingestion failed: %s", exc)
            return {"type": "csv", "status": "error", "message": str(exc)}

    else:
        logger.warning("Unsupported file type: %s", ext)
        return {"type": "unknown", "status": "error", "message": f"Unsupported type: {ext}"}


# Module-level singleton
document_processing_service = DocumentProcessingService()
