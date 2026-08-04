"""
Main pipeline orchestrator.

Scans the input/ directory for PDF and IFC files, routes each file to the
appropriate engine, and writes sLLM / VLM JSONL datasets to output/.

Usage (from Python)
-------------------
    from gen_aec_syn_data import AECPipeline, PipelineConfig

    cfg = PipelineConfig()
    pipeline = AECPipeline(cfg)
    pipeline.run()
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional

from .config import PipelineConfig
from .ifc_processor import IFCProcessor
from .pdf_extractor import PDFExtractor
from .sllm_dapt_engine import SLLM_DAPT_Engine
from .sllm_sft_engine import SLLM_SFT_Engine
from .vlm_engine import VLMEngine

logger = logging.getLogger("AEC_Pipeline.pipeline")


class AECPipeline:
    """
    Top-level orchestrator that wires together PDF, IFC, sLLM, and VLM engines.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.pdf_extractor = PDFExtractor(config)
        self.ifc_processor = IFCProcessor(config)
        self.sllm_sft_engine = SLLM_SFT_Engine(config)
        self.sllm_dapt_engine = SLLM_DAPT_Engine(config)
        self.vlm_engine = VLMEngine(config)

    def run(
        self,
        pdf_files: Optional[List[Path]] = None,
        ifc_files: Optional[List[Path]] = None,
    ) -> None:
        """
        Run the full pipeline.

        Parameters
        ----------
        pdf_files : explicit list of PDF paths (falls back to scanning input/)
        ifc_files : explicit list of IFC paths (falls back to scanning input/)
        """
        logger.info("=" * 60)
        logger.info("AEC Synthetic Dataset Generation Pipeline — START")
        logger.info("=" * 60)

        self.config.ensure_output_dirs()

        # Discover input files
        pdfs = pdf_files or self._discover(self.config.input_dir, ".pdf")
        ifcs = ifc_files or self._discover(self.config.input_dir, ".ifc")

        logger.info("Found %d PDF(s) and %d IFC file(s) in %s",
                    len(pdfs), len(ifcs), self.config.input_dir)

        if not pdfs and not ifcs:
            logger.warning(
                "No input files found in '%s'. "
                "Place PDF or IFC files there and re-run.",
                self.config.input_dir,
            )
            return

        # ── sLLM branch (PDF → JSONL) ──────────────────────────────────
        sft_total = 0
        for pdf_path in pdfs:
            sft_total += self._process_pdf(pdf_path)

        if pdfs:
            logger.info(
                "sLLM synthesis complete (mode=%s). Total records: %d — outputs under %s",
                self.config.dataset_mode, sft_total, self.config.output_dir,
            )

        # ── VLM branch (IFC → renders → JSONL) ─────────────────────────
        vlm_total = 0
        for ifc_path in ifcs:
            vlm_total += self._process_ifc(ifc_path)

        if ifcs:
            logger.info(
                "VLM synthesis complete. Total samples: %d — outputs under %s",
                vlm_total, self.config.output_dir,
            )

        logger.info("=" * 60)
        logger.info("Pipeline finished. sLLM=%d  VLM=%d", sft_total, vlm_total)
        logger.info("=" * 60)

    def _process_pdf(self, pdf_path: Path) -> int:
        logger.info("[PDF] Processing: %s", pdf_path.name)
        try:
            chunks = self.pdf_extractor.extract_chunks(pdf_path)
        except Exception as exc:
            logger.error("[PDF] Extraction failed for '%s': %s", pdf_path.name, exc)
            return 0

        if not chunks:
            logger.warning("[PDF] No usable chunks extracted from '%s'", pdf_path.name)
            return 0

        mode = self.config.dataset_mode
        logger.info(
            "[PDF] %d chunks extracted — starting sLLM synthesis (mode=%s)",
            len(chunks), mode,
        )

        stem = pdf_path.stem
        subdir = self.config.relative_subdir(pdf_path)
        count = 0
        if mode in ("sft", "both"):
            try:
                self.sllm_sft_engine.set_output_dir(
                    self.config.file_output_dir(stem, "sft", subdir))
                count += self.sllm_sft_engine.process_chunks(chunks)
                logger.info("[PDF] SFT → %s", self.sllm_sft_engine.jsonl_path)
            except Exception as exc:
                logger.error("[PDF] SFT engine error for '%s': %s", pdf_path.name, exc)

        if mode in ("dapt", "both"):
            try:
                self.sllm_dapt_engine.set_output_dir(
                    self.config.file_output_dir(stem, "dapt", subdir))
                doc_meta = {"source_name": pdf_path.name}
                # A year in the file name (e.g. "…매뉴얼(2024).pdf") is more
                # reliable than what the model infers from the opening pages,
                # where the edition year is often not spelled out. doc_meta
                # takes precedence over inference, so this pins source_date.
                year = self._year_from_name(pdf_path.stem)
                if year:
                    doc_meta["source_date"] = year
                count += self.sllm_dapt_engine.process_chunks(
                    chunks, doc_meta=doc_meta
                )
                logger.info("[PDF] DAPT → %s", self.sllm_dapt_engine.jsonl_path)
            except Exception as exc:
                logger.error("[PDF] DAPT engine error for '%s': %s", pdf_path.name, exc)

        logger.info("[PDF] Done '%s' — %d records generated", pdf_path.name, count)
        return count

    def _process_ifc(self, ifc_path: Path) -> int:
        logger.info("[IFC] Processing: %s", ifc_path.name)

        # Per-file output: output/<stem>_vlm/{images/..., vlm_training_data.jsonl},
        # under the model's own folder when it came from inside input_dir.
        vlm_dir = self.config.file_output_dir(
            ifc_path.stem, "vlm", self.config.relative_subdir(ifc_path))
        self.vlm_engine.set_output_dir(vlm_dir)

        try:
            elements, render_paths, depth_paths = self.ifc_processor.process(
                ifc_path,
                render_dir=self.vlm_engine.bim_render_dir,
                depth_dir=self.vlm_engine.depth_map_dir,
            )
        except Exception as exc:
            logger.error("[IFC] Processing failed for '%s': %s", ifc_path.name, exc)
            return 0

        if not render_paths:
            logger.warning("[IFC] No renders produced for '%s'", ifc_path.name)
            return 0

        logger.info(
            "[IFC] %d element(s), %d render(s) — starting VLM synthesis",
            len(elements), len(render_paths),
        )

        project_type = self._infer_project_type(ifc_path.stem)
        trade_type = self._infer_trade_type(elements)

        try:
            count = self.vlm_engine.process_renders(
                render_paths=render_paths,
                elements=elements,
                model_id=ifc_path.stem,
                project_type=project_type,
                trade_type=trade_type,
                depth_paths=depth_paths,
            )
        except Exception as exc:
            logger.error("[IFC] VLM engine error for '%s': %s", ifc_path.name, exc)
            return 0

        logger.info(
            "[IFC] Done '%s' — %d samples → %s",
            ifc_path.name, count, self.vlm_engine.jsonl_path,
        )
        return count

    @staticmethod
    def _discover(directory: Path, suffix: str) -> List[Path]:
        if not directory.exists():
            return []
        return sorted(directory.glob(f"**/*{suffix}"))

    @staticmethod
    def _year_from_name(stem: str) -> str:
        """
        Return a plausible edition year from a file name, or "" if none.

        Prefers a parenthesised year — "매뉴얼(2024)" — over a bare one, then
        the latest 19xx/20xx found.
        """
        import re  # noqa: PLC0415

        paren = re.findall(r"[(\[](19|20)(\d{2})[)\]]", stem)
        if paren:
            return max(a + b for a, b in paren)
        loose = re.findall(r"(?:19|20)\d{2}", stem)
        return max(loose) if loose else ""

    @staticmethod
    def _infer_project_type(stem: str) -> str:
        stem_lower = stem.lower()
        if any(k in stem_lower for k in ("bridge", "교량", "bri")):
            return "교량"
        if any(k in stem_lower for k in ("tunnel", "터널", "tun")):
            return "터널"
        if any(k in stem_lower for k in ("road", "도로", "highway")):
            return "도로"
        return "건물"

    @staticmethod
    def _infer_trade_type(elements) -> str:
        type_names = [e.ifc_type for e in elements]
        if "IfcBridge" in type_names:
            return "강구조"
        if "IfcBeam" in type_names or "IfcColumn" in type_names:
            return "철근콘크리트"
        return "복합구조"
