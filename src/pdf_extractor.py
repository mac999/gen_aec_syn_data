from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import Iterator, List

from .config import PipelineConfig
from .schemas import DocumentChunk

logger = logging.getLogger("AEC_Pipeline.pdf_extractor")


class PDFExtractor:
    """Extracts and chunks text from a PDF file using PyMuPDF."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def extract_chunks(self, pdf_path: Path) -> List[DocumentChunk]:
        """
        Load *pdf_path* and return a list of DocumentChunk objects.

        Raises:
            ImportError: if PyMuPDF is not installed.
            FileNotFoundError: if the PDF file does not exist.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise ImportError(
                "PyMuPDF is required: pip install PyMuPDF"
            ) from exc

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc_id = pdf_path.stem
        logger.info("Extracting text from PDF: %s", pdf_path.name)

        raw_pages: List[tuple[int, str]] = []  # (page_num, text)
        with fitz.open(str(pdf_path)) as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")
                if text.strip():
                    raw_pages.append((page_num, text))

        if not raw_pages:
            logger.warning("No text extracted from %s", pdf_path.name)
            return []

        full_text_with_pages = self._join_pages(raw_pages)
        chunks = list(self._chunk_text(doc_id, full_text_with_pages))
        logger.info(
            "Created %d chunks from '%s'", len(chunks), pdf_path.name
        )
        return chunks

    @staticmethod
    def _join_pages(
        raw_pages: List[tuple[int, str]],
    ) -> List[tuple[List[int], str]]:
        """
        Merge consecutive page texts, keeping a list of page numbers
        that each merged block spans.
        Returns list of ([page_nums], merged_text).
        """
        merged: List[tuple[List[int], str]] = []
        for page_num, text in raw_pages:
            merged.append(([page_num], text))
        return merged

    def _chunk_text(
        self,
        doc_id: str,
        pages: List[tuple[List[int], str]],
    ) -> Iterator[DocumentChunk]:
        """
        Split page texts into chunks of `chunk_min_size`–`chunk_max_size`
        characters, respecting paragraph boundaries where possible.
        Uses a sliding window with `chunk_overlap` overlap.
        """
        min_size = self.config.chunk_min_size
        max_size = self.config.chunk_max_size
        overlap = self.config.chunk_overlap

        # Concatenate all pages into one block, annotated with page numbers
        combined_parts: List[tuple[int, str]] = []
        for page_nums, text in pages:
            for line in text.splitlines(keepends=True):
                combined_parts.append((page_nums[0], line))

        # Build a flat text and a character→page mapping
        flat_text = ""
        char_page: List[int] = []
        for page_num, line in combined_parts:
            flat_text += line
            char_page.extend([page_num] * len(line))

        # Split on paragraph boundaries (double newline / blank line)
        paragraphs: List[tuple[str, int, int]] = []  # (text, start, end)
        for m in re.finditer(r"(?:^|\n)(.+?)(?=\n\n|\Z)", flat_text, re.DOTALL):
            para_text = m.group(0).strip()
            if para_text:
                paragraphs.append((para_text, m.start(), m.end()))

        if not paragraphs:
            # Fallback: treat entire text as one paragraph block
            paragraphs = [(flat_text, 0, len(flat_text))]

        chunk_index = 0
        buffer = ""
        buffer_pages: List[int] = []
        buffer_start = 0

        for para_text, p_start, p_end in paragraphs:
            # Determine which page this paragraph belongs to
            page_num = char_page[p_start] if p_start < len(char_page) else char_page[-1]

            # Paragraph itself exceeds max_size: flush buffer then split into windows
            if len(para_text) > max_size:
                if buffer:
                    yield self._make_chunk(doc_id, chunk_index, buffer, buffer_pages)
                    chunk_index += 1
                    buffer = ""
                    buffer_pages = []
                pos = 0
                while pos < len(para_text):
                    end = min(pos + max_size, len(para_text))
                    yield self._make_chunk(doc_id, chunk_index, para_text[pos:end], [page_num])
                    chunk_index += 1
                    if end == len(para_text):
                        break
                    pos = end - overlap
                continue

            if len(buffer) + len(para_text) + 1 > max_size and buffer:
                # Emit the current buffer as a chunk
                yield self._make_chunk(doc_id, chunk_index, buffer, buffer_pages)
                chunk_index += 1
                # Carry-over overlap from the end of the current buffer
                overlap_text = buffer[-overlap:] if overlap < len(buffer) else buffer
                buffer = overlap_text + "\n" + para_text
                buffer_pages = list(
                    dict.fromkeys(
                        char_page[max(0, p_start - overlap): p_end]
                    )
                )
            else:
                buffer = (buffer + "\n" + para_text).strip()
                if page_num not in buffer_pages:
                    buffer_pages.append(page_num)

        # Emit the remaining buffer
        if len(buffer) >= min_size:
            yield self._make_chunk(doc_id, chunk_index, buffer, buffer_pages)
        elif buffer and chunk_index > 0:
            # Too short to stand alone — merge with previous isn't possible here,
            # so we emit it anyway since it still contains valid content.
            logger.debug(
                "Emitting short trailing chunk (%d chars)", len(buffer)
            )
            yield self._make_chunk(doc_id, chunk_index, buffer, buffer_pages)

    @staticmethod
    def _make_chunk(
        doc_id: str, index: int, text: str, pages: List[int]
    ) -> DocumentChunk:
        return DocumentChunk(
            doc_id=doc_id,
            chunk_index=index,
            page_numbers=sorted(set(pages)) if pages else [0],
            text=text.strip(),
            char_count=len(text.strip()),
        )
