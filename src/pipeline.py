"""
Main pipeline orchestrator.

Scans the input/ directory for PDF and IFC files, routes each file to the
appropriate engine, and writes sLLM / VLM JSONL datasets to output/.

Usage (from Python)
-------------------
    from src.pipeline import AECPipeline
    from src.config import PipelineConfig

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
            mode = self.config.dataset_mode
            if mode in ("sft", "both"):
                logger.info("sLLM SFT dataset → %s", self.sllm_sft_engine.jsonl_path)
            if mode in ("dapt", "both"):
                logger.info("sLLM DAPT dataset → %s", self.sllm_dapt_engine.jsonl_path)
            logger.info(
                "sLLM synthesis complete (mode=%s). Total records: %d", mode, sft_total
            )

        # ── VLM branch (IFC → renders → JSONL) ─────────────────────────
        vlm_total = 0
        for ifc_path in ifcs:
            vlm_total += self._process_ifc(ifc_path)

        if ifcs:
            logger.info(
                "VLM synthesis complete. Total samples: %d → %s",
                vlm_total,
                self.vlm_engine.jsonl_path,
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

        count = 0
        if mode in ("sft", "both"):
            try:
                count += self.sllm_sft_engine.process_chunks(chunks)
            except Exception as exc:
                logger.error("[PDF] SFT engine error for '%s': %s", pdf_path.name, exc)

        if mode in ("dapt", "both"):
            try:
                count += self.sllm_dapt_engine.process_chunks(
                    chunks, doc_meta={"source_name": pdf_path.name}
                )
            except Exception as exc:
                logger.error("[PDF] DAPT engine error for '%s': %s", pdf_path.name, exc)

        logger.info("[PDF] Done '%s' — %d records generated", pdf_path.name, count)
        return count

    def _process_ifc(self, ifc_path: Path) -> int:
        logger.info("[IFC] Processing: %s", ifc_path.name)
        try:
            elements, render_paths = self.ifc_processor.process(ifc_path)
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
            )
        except Exception as exc:
            logger.error("[IFC] VLM engine error for '%s': %s", ifc_path.name, exc)
            return 0

        logger.info("[IFC] Done '%s' — %d samples generated", ifc_path.name, count)
        return count

    @staticmethod
    def _discover(directory: Path, suffix: str) -> List[Path]:
        if not directory.exists():
            return []
        return sorted(directory.glob(f"**/*{suffix}"))

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
