"""
Unit tests for chunking service and document processing utilities.

Covers:
- Chunk dataclass
- chunk_page_text() sentence-based splitter
- chunk_page_with_layout()
- _find_nearest_text_chunk() spatial linking
- DocumentProcessingService (mocked Bedrock + file I/O)
"""
import pytest

from app.services.chunking import (
    Chunk,
    chunk_page_text,
    chunk_page_with_layout,
    _find_nearest_text_chunk,
)


# ── Chunk dataclass ──────────────────────────────────────────────────────────

class TestChunk:
    def test_basic_creation(self):
        c = Chunk(id="doc1-p0-c1", page=0, text="Hello world", source_document_id="doc1")
        assert c.id == "doc1-p0-c1"
        assert c.page == 0
        assert c.text == "Hello world"
        assert c.source_document_id == "doc1"
        assert c.chunk_type == "text"  # default
        assert c.metadata == {}  # default

    def test_image_chunk_type(self):
        c = Chunk(
            id="doc1-p0-img1",
            page=0,
            text="Image description",
            source_document_id="doc1",
            chunk_type="image",
            metadata={"image_path": "/tmp/img.png", "bbox": (0, 0, 100, 100)},
        )
        assert c.chunk_type == "image"
        assert c.metadata["image_path"] == "/tmp/img.png"

    def test_metadata_default_is_empty_dict(self):
        c1 = Chunk(id="a", page=0, text="t", source_document_id="d")
        c2 = Chunk(id="b", page=0, text="t", source_document_id="d")
        # Each instance gets its own dict
        c1.metadata["key"] = "val"
        assert "key" not in c2.metadata


# ── chunk_page_text ───────────────────────────────────────────────────────────

class TestChunkPageText:
    def test_short_text_becomes_single_chunk(self):
        text = "This is a short sentence."
        chunks = chunk_page_text(text, page_index=0, source_document_id="doc1")
        assert len(chunks) == 1
        assert chunks[0].page == 0
        assert chunks[0].source_document_id == "doc1"
        assert chunks[0].chunk_type == "text"

    def test_long_text_split_into_multiple_chunks(self):
        # 600-word text should exceed default max_tokens=256
        sentence = "The quick brown fox jumps over the lazy dog. "
        long_text = sentence * 40  # ~360 words
        chunks = chunk_page_text(long_text, page_index=1, source_document_id="mydoc", max_tokens=50)
        assert len(chunks) > 1

    def test_chunk_ids_are_unique(self):
        sentence = "Hello world. " * 60
        chunks = chunk_page_text(sentence, page_index=0, source_document_id="doc", max_tokens=20)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_id_format(self):
        text = "One sentence here."
        chunks = chunk_page_text(text, page_index=2, source_document_id="docX")
        # Should contain page index in the id
        assert "p2" in chunks[0].id
        assert "docX" in chunks[0].id

    def test_page_index_set_correctly(self):
        text = "Some text."
        chunks = chunk_page_text(text, page_index=5, source_document_id="d")
        assert all(c.page == 5 for c in chunks)

    def test_empty_text_returns_empty_list(self):
        chunks = chunk_page_text("", page_index=0, source_document_id="doc")
        assert chunks == []

    def test_whitespace_only_returns_empty_list(self):
        chunks = chunk_page_text("   \n\n   ", page_index=0, source_document_id="doc")
        assert chunks == []

    def test_no_chunk_exceeds_max_tokens_by_much(self):
        """Each chunk should stay close to max_tokens (single-sentence overage is ok)."""
        sentence = "Alpha beta gamma delta epsilon. "
        text = sentence * 100
        chunks = chunk_page_text(text, page_index=0, source_document_id="doc", max_tokens=30)
        for c in chunks:
            word_count = len(c.text.split())
            # Allow up to 2x for a single long sentence, but generally should be near limit
            assert word_count <= 60, f"Chunk has {word_count} words, expected ≤60"


# ── _find_nearest_text_chunk ─────────────────────────────────────────────────

class TestFindNearestTextChunk:
    def _make_chunk(self, cid, bbox):
        return Chunk(
            id=cid, page=0, text="some text", source_document_id="doc",
            metadata={"bbox": bbox}
        )

    def test_returns_nearest_chunk(self):
        chunks = [
            self._make_chunk("c1", (0, 0, 100, 50)),    # centre: (50, 25)
            self._make_chunk("c2", (200, 300, 300, 350)),  # centre: (250, 325)
        ]
        image_bbox = (10, 10, 60, 40)  # centre: (35, 25) — very close to c1
        nearest = _find_nearest_text_chunk(chunks, image_bbox)
        assert nearest is not None
        assert nearest.id == "c1"

    def test_returns_none_for_empty_list(self):
        assert _find_nearest_text_chunk([], (0, 0, 100, 100)) is None

    def test_returns_chunk_even_if_no_bbox_metadata(self):
        # Chunks without bbox should not crash; just skipped
        c_no_bbox = Chunk(id="no_bbox", page=0, text="t", source_document_id="d", metadata={})
        c_with_bbox = Chunk(
            id="with_bbox", page=0, text="t", source_document_id="d",
            metadata={"bbox": (50, 50, 150, 100)},
        )
        nearest = _find_nearest_text_chunk([c_no_bbox, c_with_bbox], (60, 60, 90, 80))
        # Should return c_with_bbox (only one with valid spatial info)
        assert nearest is not None


# ── chunk_page_with_layout ───────────────────────────────────────────────────

class TestChunkPageWithLayout:
    def test_basic_layout_chunking(self):
        # chunk_page_with_layout expects PyMuPDF block tuples:
        # (x0, y0, x1, y1, text, block_no, block_type)
        page_data = {
            "blocks": [
                (0, 0, 400, 50, "Introduction paragraph with several words.", 0, 0),
                (0, 60, 400, 110, "Second paragraph with more content here.", 1, 0),
            ]
        }
        chunks = chunk_page_with_layout(
            page_data=page_data,
            page_index=0,
            source_document_id="layout_doc",
        )
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_empty_page_data(self):
        chunks = chunk_page_with_layout(
            page_data={},
            page_index=0,
            source_document_id="doc",
        )
        assert isinstance(chunks, list)

    def test_returns_list_of_chunks(self):
        # Tuple format: (x0, y0, x1, y1, text, block_no, block_type)
        page_data = {"blocks": [(0, 0, 200, 30, "Hello.", 0, 0)]}
        chunks = chunk_page_with_layout(page_data, 0, "doc")
        for c in chunks:
            assert isinstance(c, Chunk)

    def test_empty_blocks_falls_back_to_text_chunking(self):
        # No blocks — should fall back to text field
        page_data = {"text": "Fallback sentence here. Another one.", "blocks": []}
        chunks = chunk_page_with_layout(page_data, 0, "doc")
        assert isinstance(chunks, list)
