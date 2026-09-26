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
from .doc_routing import classify, extract_provenance
from .dpo_engine import DPOEngine
from .sllm_dapt_engine import SLLM_DAPT_Engine
from .rag_engine import RAGEngine
from .rlvr_engine import RLVREngine
from .sllm_sft_engine import SLLM_SFT_Engine
from .star_engine import STaREngine
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
        self.dpo_engine = DPOEngine(config)
        self.rag_engine = RAGEngine(config)
        self.star_engine = STaREngine(config)
        self.rlvr_engine = RLVREngine(config)

    def run(
        self,
        pdf_files: Optional[List[Path]] = None,
        ifc_files: Optional[List[Path]] = None,
        only_new: bool = False,
    ) -> None:
        """
        Run the full pipeline.

        Parameters
        ----------
        pdf_files : explicit list of PDF paths
        ifc_files : explicit list of IFC paths
        only_new  : skip inputs whose dataset JSONL already exists in the
                    output tree

        With neither given, input/ is scanned for both. Give either one and only
        the named files are processed.
        """
        logger.info("=" * 60)
        logger.info("AEC Synthetic Dataset Generation Pipeline — START")
        logger.info("=" * 60)

        self.config.ensure_output_dirs()

        # Discover input files. Naming files on either flag means the run is
        # meant for those files only — scanning input/ for the other type as
        # well turns "--ifc model.ifc" into a run over every PDF sitting there.
        explicit = pdf_files is not None or ifc_files is not None
        pdfs = pdf_files or ([] if explicit else self._discover(self.config.input_dir, ".pdf"))
        ifcs = ifc_files or ([] if explicit else self._discover(self.config.input_dir, ".ifc"))

        logger.info("Found %d PDF(s) and %d IFC file(s) in %s",
                    len(pdfs), len(ifcs), self.config.input_dir)

        if only_new:
            n_pdf, n_ifc = len(pdfs), len(ifcs)
            pdfs = [p for p in pdfs if not self._has_output(p, "pdf")]
            ifcs = [p for p in ifcs if not self._has_output(p, "ifc")]
            logger.info(
                "--only-new: skipped %d PDF(s) and %d IFC(s) with existing "
                "outputs — %d PDF(s) and %d IFC(s) left to process",
                n_pdf - len(pdfs), n_ifc - len(ifcs), len(pdfs), len(ifcs),
            )

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
        want_sft = mode in ("sft", "both", "dpo", "star", "rlvr", "all")
        want_dapt = mode in ("dapt", "both", "all")
        want_dpo = mode in ("dpo", "all")
        want_star = mode in ("star", "all")
        want_rlvr = mode in ("rlvr", "all")

        # Routing decides whether this document trains, gets indexed, or both.
        # Amended regulations memorised into weights go stale; the same text in
        # a retrieval corpus is replaced by one reindex.
        setting = self.config.doc_routing
        if setting == "auto":
            decision = classify(pdf_path, chunks[0].text if chunks else "",
                                self.config.routing_threshold,
                                self.config.routing_both_margin)
        else:
            decision = classify(pdf_path)
            decision.route = setting if setting in ("train", "retrieve", "both") else "train"
        if decision.route == "retrieve":
            want_sft = want_dapt = want_dpo = False
        logger.info("[PDF] route=%s score=%.1f %s",
                    decision.route, decision.score, decision.signals)
        logger.info(
            "[PDF] %d chunks extracted — starting sLLM synthesis (mode=%s)",
            len(chunks), mode,
        )

        stem = pdf_path.stem
        subdir = self.config.relative_subdir(pdf_path)
        self.sllm_sft_engine.samples = []
        count = 0
        if want_sft:
            try:
                self.sllm_sft_engine.set_output_dir(
                    self.config.file_output_dir(stem, "sft", subdir))
                count += self.sllm_sft_engine.process_chunks(chunks)
                logger.info("[PDF] SFT → %s", self.sllm_sft_engine.jsonl_path)
            except Exception as exc:
                logger.error("[PDF] SFT engine error for '%s': %s", pdf_path.name, exc)

        if want_dapt:
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

        if decision.route in ("retrieve", "both"):
            try:
                self.rag_engine.set_output_dir(
                    self.config.file_output_dir(stem, "rag", subdir))
                count += self.rag_engine.process_chunks(
                    chunks, extract_provenance(pdf_path, chunks[0].text if chunks else ""),
                    decision.as_metadata())
                logger.info("[PDF] RAG -> %s", self.rag_engine.jsonl_path)
            except Exception as exc:
                logger.error("[PDF] RAG engine error for '%s': %s", pdf_path.name, exc)

        sft_dicts = [s.to_jsonl_dict() for s in self.sllm_sft_engine.samples]
        chunk_text = {c.chunk_index: c.text for c in chunks}

        if want_star and sft_dicts:
            try:
                self.star_engine.set_output_dir(
                    self.config.file_output_dir(stem, "star", subdir))
                count += self.star_engine.filter_samples(sft_dicts, chunk_text)
                logger.info("[PDF] STaR -> %s", self.star_engine.jsonl_path)
            except Exception as exc:
                logger.error("[PDF] STaR engine error for '%s': %s", pdf_path.name, exc)

        if want_rlvr and sft_dicts:
            try:
                self.rlvr_engine.set_output_dir(
                    self.config.file_output_dir(stem, "rlvr", subdir))
                count += self.rlvr_engine.export(sft_dicts, chunk_text)
                logger.info("[PDF] RLVR -> %s", self.rlvr_engine.jsonl_path)
            except Exception as exc:
                logger.error("[PDF] RLVR engine error for '%s': %s", pdf_path.name, exc)

        if want_dpo and self.sllm_sft_engine.samples:
            try:
                self.dpo_engine.set_output_dir(
                    self.config.file_output_dir(stem, "dpo", subdir))
                count += self.dpo_engine.build_pairs(sft_dicts)
                logger.info("[PDF] DPO -> %s", self.dpo_engine.jsonl_path)
            except Exception as exc:
                logger.error("[PDF] DPO engine error for '%s': %s", pdf_path.name, exc)

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

    def _has_output(self, path: Path, kind: str) -> bool:
        """
        True when *path* already has a non-empty dataset file under output_dir.

        The JSONL writers append, so re-processing an input duplicates its
        records instead of replacing them — skipping here is what makes
        --only-new safe to re-run over a whole corpus. An input that produced
        nothing last time (e.g. an image-only PDF that yielded zero chunks)
        has no file, stays "new", and is retried.
        """
        out_dir = self.config.file_output_dir(
            path.stem, kind, self.config.relative_subdir(path))
        if kind == "ifc":
            names = ("vlm_training_data.jsonl",)
        elif self.config.dataset_mode == "sft":
            names = ("sllm_training_data.jsonl",)
        elif self.config.dataset_mode == "dapt":
            names = ("dapt_training_data.jsonl",)
        else:
            # mode "both": either file means a previous run covered this
            # input, and appending the missing half alone is not possible
            # without also duplicating the existing one.
            names = ("sllm_training_data.jsonl", "dapt_training_data.jsonl")
        return any(
            (out_dir / n).exists() and (out_dir / n).stat().st_size > 0
            for n in names
        )

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
