from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(processName)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("AEC_Pipeline")


def _parse_args(defaults) -> argparse.Namespace:
    """Build the CLI; *defaults* (a PipelineConfig) seeds every default value."""
    parser = argparse.ArgumentParser(
        prog="aec-pipeline",
        description="AEC Synthetic Dataset Generation Pipeline (100%% Local Edition). "
                    "Defaults are read from config.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── File targets ──────────────────────────────────────────────────────
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=defaults.input_dir,
        help="Directory to scan for PDF and IFC files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=defaults.output_dir,
        help="Root directory for all generated outputs.",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        action="append",
        dest="pdfs",
        metavar="PDF_PATH",
        help="Explicit PDF file(s) to process (repeatable).",
    )
    parser.add_argument(
        "--ifc",
        type=Path,
        action="append",
        dest="ifcs",
        metavar="IFC_PATH",
        help="Explicit IFC file(s) to process (repeatable).",
    )

    # ── Dataset mode ──────────────────────────────────────────────────────
    parser.add_argument(
        "--dataset",
        default=defaults.dataset_mode,
        choices=["sft", "dapt", "both"],
        help="Which sLLM dataset(s) to generate from PDFs: "
             "'sft' (QA pairs via LLM), 'dapt' (raw domain corpus, no LLM), or 'both'.",
    )

    # ── LLM backend ───────────────────────────────────────────────────────
    parser.add_argument(
        "--backend",
        default=defaults.llm_backend,
        choices=["ollama", "llamaserver", "gemini", "none"],
        help="LLM backend: 'ollama', 'llamaserver' (direct HTTP + JSON mode), 'gemini' (Google API), or 'none' (no LLM).",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=defaults.llm_parallel,
        help="Concurrent worker threads. Match llama-server --parallel N.",
    )
    parser.add_argument(
        "--qa-per-chunk",
        type=int,
        default=defaults.qa_per_chunk,
        help="QA pairs to generate per chunk in a single LLM call.",
    )

    # ── Ollama / sLLM ─────────────────────────────────────────────────────
    parser.add_argument(
        "--model",
        default=defaults.ollama_model,
        help="Ollama model name.",
    )
    parser.add_argument(
        "--ollama-url",
        default=defaults.ollama_base_url,
        help="Ollama base URL.",
    )
    parser.add_argument(
        "--llama-server-url",
        default=defaults.llama_server_url,
        help="llama-server base URL (used when --backend=llamaserver).",
    )
    parser.add_argument(
        "--gemini-api-key",
        default=defaults.gemini_api_key,
        help="Google Gemini API key (used when --backend=gemini). "
             "Falls back to GEMINI_API_KEY env var / .env file.",
    )
    parser.add_argument(
        "--gemini-model",
        default=defaults.gemini_model,
        help="Gemini model name (used when --backend=gemini).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=defaults.ollama_temperature,
        help="LLM sampling temperature.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=defaults.max_samples_per_doc,
        help="Maximum sLLM samples to generate per document.",
    )

    # ── Chunking ──────────────────────────────────────────────────────────
    parser.add_argument("--chunk-min", type=int, default=defaults.chunk_min_size, help="Min chunk size (chars).")
    parser.add_argument("--chunk-max", type=int, default=defaults.chunk_max_size, help="Max chunk size (chars).")
    parser.add_argument("--chunk-overlap", type=int, default=defaults.chunk_overlap, help="Chunk overlap (chars).")

    # ── ComfyUI ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--comfyui-url",
        default=defaults.comfyui_url,
        help="ComfyUI API base URL.",
    )

    # ── IFC rendering ─────────────────────────────────────────────────────
    parser.add_argument(
        "--ifc-views",
        nargs="+",
        default=defaults.ifc_views,
        choices=["perspective", "top", "front", "side"],
        help="IFC views to render.",
    )
    parser.add_argument(
        "--render-size",
        type=int,
        default=defaults.ifc_render_width,
        help="Render resolution (square, pixels).",
    )

    # ── Config file ───────────────────────────────────────────────────────
    parser.add_argument(
        "--config",
        type=Path,
        metavar="JSON_PATH",
        help="Load configuration from a JSON file (overrides config.json and other flags).",
    )
    parser.add_argument(
        "--save-config",
        type=Path,
        metavar="JSON_PATH",
        help="Save the resolved configuration to a JSON file and exit.",
    )

    # ── Misc ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover input files and print config, but do not run the pipeline.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    return parser.parse_args()


def _build_config(args: argparse.Namespace, base):
    """Apply CLI overrides on top of *base* (loaded from config.json)."""
    from src.config import PipelineConfig  # local import to keep startup fast

    if args.config:
        logger.info("Loading config from %s", args.config)
        return PipelineConfig.from_json(args.config)

    # Every CLI flag defaulted to the matching base value, so assigning them
    # back is a no-op unless the user explicitly overrode the flag. Config
    # fields not exposed on the CLI keep their config.json values.
    cfg = base
    cfg.input_dir = args.input_dir
    cfg.output_dir = args.output_dir
    cfg.dataset_mode = args.dataset
    cfg.llm_backend = args.backend
    cfg.llm_parallel = args.parallel
    cfg.qa_per_chunk = args.qa_per_chunk
    cfg.ollama_model = args.model
    cfg.ollama_base_url = args.ollama_url
    cfg.llama_server_url = args.llama_server_url
    cfg.gemini_api_key = args.gemini_api_key
    cfg.gemini_model = args.gemini_model
    cfg.ollama_temperature = args.temperature
    cfg.max_samples_per_doc = args.max_samples
    cfg.chunk_min_size = args.chunk_min
    cfg.chunk_max_size = args.chunk_max
    cfg.chunk_overlap = args.chunk_overlap
    cfg.comfyui_url = args.comfyui_url
    cfg.ifc_views = args.ifc_views
    cfg.ifc_render_width = args.render_size
    cfg.ifc_render_height = args.render_size
    cfg.__post_init__()  # re-normalise Path fields after override
    return cfg


def main() -> int:
    from src.config import PipelineConfig  # local import to keep startup fast

    base = PipelineConfig.load_default()
    args = _parse_args(base)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = _build_config(args, base)

    if args.save_config:
        cfg.to_json(args.save_config)
        logger.info("Config saved. Exiting.")
        return 0

    if args.dry_run:
        logger.info("=== DRY RUN ===")
        logger.info("Input dir   : %s", cfg.input_dir)
        logger.info("Output dir  : %s", cfg.output_dir)
        logger.info("Dataset mode: %s", cfg.dataset_mode)
        logger.info("LLM backend : %s", cfg.llm_backend)
        logger.info("Ollama model: %s", cfg.ollama_model)
        logger.info("ComfyUI URL : %s", cfg.comfyui_url)

        pdfs = sorted(cfg.input_dir.glob("**/*.pdf")) if cfg.input_dir.exists() else []
        ifcs = sorted(cfg.input_dir.glob("**/*.ifc")) if cfg.input_dir.exists() else []

        logger.info("PDF files found  : %d", len(pdfs))
        for p in pdfs:
            logger.info("  %s", p)
        logger.info("IFC files found  : %d", len(ifcs))
        for p in ifcs:
            logger.info("  %s", p)
        return 0

    # ── Full pipeline run ────────────────────────────────────────────────
    from src.pipeline import AECPipeline  # deferred import

    pipeline = AECPipeline(cfg)
    pipeline.run(
        pdf_files=args.pdfs or None,
        ifc_files=args.ifcs or None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
