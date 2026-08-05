from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import Iterator, List

from .config import PipelineConfig
from .schemas import DocumentChunk

logger = logging.getLogger("AEC_Pipeline.pdf_extractor")

# Dot/middle-dot leaders used by Korean tables of contents.
_LEADER_RE = re.compile(r"[.·․‥…]{4,}")
# A line holding nothing but a page number, possibly bracketed.
_PAGE_ONLY_RE = re.compile(r"^[\s\-–—()\[\]]*\d{1,4}[\s\-–—()\[\]]*$")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_HANGUL_RE = re.compile(r"[가-힣]")


def normalise_pdf_text(text: str) -> str:
    """
    Clean one page of PyMuPDF output before it reaches the chunker.

    The important case is vertically-set text: PyMuPDF emits a title like
    "국토교통부" as five separate one-character lines, which the chunker then
    carries verbatim into both the DAPT corpus and the LLM prompt. Runs of
    single-character lines are folded back into one line.
    """
    text = _CTRL_RE.sub("", text.replace(" ", " "))

    out: List[str] = []
    run: List[str] = []

    def flush_run() -> None:
        if not run:
            return
        # 3+ consecutive single-character lines is vertical typesetting, not
        # prose; anything shorter is left alone so real short lines survive.
        out.append("".join(run) if len(run) >= 3 else "\n".join(run))
        run.clear()

    for raw_line in text.splitlines():
        line = _LEADER_RE.sub(" ", raw_line).strip()
        if not line or _PAGE_ONLY_RE.match(line):
            flush_run()
            out.append("")
            continue
        if len(line) == 1:
            run.append(line)
            continue
        flush_run()
        out.append(re.sub(r"[ \t]{2,}", " ", line))

    flush_run()
    cleaned = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def is_informative(text: str, min_hangul: int = 20) -> bool:
    """
    Reject chunks that carry no trainable prose.

    Table-of-contents fragments and page furniture survive chunking as strings
    of numbers and punctuation; they add noise to DAPT and make the SFT model
    invent questions no document can answer.
    """
    if len(text) < 40:
        return False
    if len(_HANGUL_RE.findall(text)) < min_hangul:
        return False
    alnum_or_hangul = sum(c.isalnum() or "가" <= c <= "힣" for c in text)
    return alnum_or_hangul / max(len(text), 1) >= 0.5


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
                text = normalise_pdf_text(page.get_text("text"))
                if text.strip():
                    raw_pages.append((page_num, text))

        if not raw_pages and self.config.ocr_enabled:
            # Scanned PDF: no text layer at all. Rasterise and read the pages.
            # This runs only when the normal path found nothing, so the cost
            # falls on scanned files alone.
            raw_pages = self._ocr_pages(pdf_path)

        if not raw_pages:
            logger.warning("No text extracted from %s", pdf_path.name)
            return []

        full_text_with_pages = self._join_pages(raw_pages)
        chunks = list(self._chunk_text(doc_id, full_text_with_pages))

        kept = [c for c in chunks if is_informative(c.text)]
        if len(kept) != len(chunks):
            logger.info(
                "Dropped %d non-informative chunk(s) (TOC/page furniture) from '%s'",
                len(chunks) - len(kept), pdf_path.name,
            )
        # Renumber so chunk_index stays contiguous after filtering.
        for new_index, chunk in enumerate(kept):
            chunk.chunk_index = new_index

        logger.info("Created %d chunks from '%s'", len(kept), pdf_path.name)
        return kept

    def _ocr_pages(self, pdf_path: Path) -> List[tuple[int, str]]:
        """
        Read a scanned PDF with EasyOCR and return (page_number, text) pairs.

        The output is shaped exactly like the normal extraction path, so
        chunking, filtering and both sLLM engines are unchanged — OCR only
        supplies the text that PyMuPDF could not.

        The reader is built once per process: EasyOCR loads detection and
        recognition models on first use, which is far more expensive than the
        pages themselves.
        """
        import fitz  # noqa: PLC0415

        try:
            reader = self._get_ocr_reader()
        except Exception as exc:
            logger.error("OCR unavailable for '%s': %s", pdf_path.name, exc)
            return []

        pages: List[tuple[int, str]] = []
        zoom = self.config.ocr_dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        with fitz.open(str(pdf_path)) as doc:
            total = doc.page_count
            limit = min(self.config.ocr_max_pages or total, total)
            if total > limit:
                logger.warning(
                    "OCR limited to the first %d of %d pages in '%s' "
                    "(ocr_max_pages)", limit, total, pdf_path.name,
                )
            logger.info("OCR: reading %d page(s) of '%s' at %d dpi",
                        limit, pdf_path.name, self.config.ocr_dpi)
            for page_num, page in enumerate(doc, start=1):
                if page_num > limit:
                    break
                try:
                    png = page.get_pixmap(matrix=matrix).tobytes("png")
                    lines = reader.readtext(png, detail=0, paragraph=True)
                except Exception as exc:
                    logger.warning("OCR failed on page %d of '%s': %s",
                                   page_num, pdf_path.name, exc)
                    continue
                text = normalise_pdf_text("\n".join(lines))
                if text.strip():
                    pages.append((page_num, text))

        logger.info("OCR: recovered text from %d/%d page(s) of '%s'",
                    len(pages), limit, pdf_path.name)
        return pages

    def _get_ocr_reader(self):
        """Lazily build and cache the EasyOCR reader."""
        reader = getattr(self, "_ocr_reader", None)
        if reader is None:
            import easyocr  # noqa: PLC0415
            langs = [l.strip() for l in self.config.ocr_languages.split(",") if l.strip()]
            logger.info("Initialising EasyOCR (languages=%s, gpu=%s)",
                        langs, self.config.ocr_use_gpu)
            reader = easyocr.Reader(langs, gpu=self.config.ocr_use_gpu, verbose=False)
            self._ocr_reader = reader
        return reader

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
