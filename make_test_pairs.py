"""Build side-by-side comparison pairs under ./test/.

    test/bim_image/<stem>.png    IFC render (the input)
    test/depth/<stem>.png        depth hint actually fed to ControlNet
    test/real_image/<stem>.png   synthesised construction photo (the output)

Same <stem> in every folder, so the pairing is obvious at a glance.
"""
from __future__ import annotations

import logging
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import PipelineConfig          # noqa: E402
from src.ifc_processor import IFCProcessor      # noqa: E402
from src.pipeline import AECPipeline            # noqa: E402
from src.vlm_engine import VLMEngine            # noqa: E402

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("test_pairs")

IFC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("input/Duplex_A_20110907.ifc")
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 12

root = Path("test")
bim_dir, depth_dir, real_dir = root / "bim_image", root / "depth", root / "real_image"
for d in (bim_dir, depth_dir, real_dir):
    d.mkdir(parents=True, exist_ok=True)

cfg = PipelineConfig.load_default()

log.info("Rendering IFC + depth maps → %s", root)
elements, renders, depths = IFCProcessor(cfg).process(
    IFC, render_dir=bim_dir, depth_dir=depth_dir
)
log.info("%d element(s), %d render(s)", len(elements), len(renders))

engine = VLMEngine(cfg)
engine.set_output_dir(root / "_work")

project_type = AECPipeline._infer_project_type(IFC.stem)
trade_type = AECPipeline._infer_trade_type(elements)
log.info("project_type=%s  trade_type=%s", project_type, trade_type)

views = cfg.vlm_photo_views
pairs = [
    (r, d) for r, d in zip(renders, depths)
    if not views or r.stem.split("_")[-1] in views
][:LIMIT]
log.info(
    "Synthesising %d image(s) — %d render(s) available, views=%s",
    len(pairs), len(renders), views or "all",
)

ok = 0
for i, (render, depth) in enumerate(pairs, 1):
    view = render.stem.split("_")[-1]
    t0 = time.time()
    out = engine._synthesise_site_photo(
        render, depth_path=depth,
        project_type=project_type, trade_type=trade_type, view_type=view,
    )
    if out is None:
        log.warning("[%d/%d] %s FAILED", i, len(pairs), render.stem)
        continue
    # Land it under the same stem as the BIM render so the folders line up.
    final = real_dir / f"{render.stem}.png"
    shutil.move(str(out), final)
    ok += 1
    log.info("[%d/%d] %s  %.0fs → %s", i, len(pairs), render.stem, time.time() - t0, final)

shutil.rmtree(root / "_work", ignore_errors=True)
log.info("DONE  %d/%d pairs", ok, len(pairs))
log.info("  BIM   : %s", bim_dir)
log.info("  depth : %s", depth_dir)
log.info("  photo : %s", real_dir)
