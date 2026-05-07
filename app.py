"""Wari Floor Plan - ROI & Classification App

Pipeline
--------
1. Upload a ZIP whose root sub-folders each contain PDFs.
2. Each PDF is rendered to PNG at 300 dpi.
3. A YOLO-seg model detects the floor-plan region (ROI) and produces a
   1280 x 1280 letterbox-padded crop.
4. A YOLO-classify model assigns each crop to one of three classes and
   copies the 1280 image into the corresponding output folder.

Output layout (inside data/)
-----------------------------
data/
  raw_png/          - step 2: full-page PNGs
  roi_1280/         - step 3: cropped & letterboxed ROIs
  output/
    <class_name_0>/ - step 4: final sorted images
    <class_name_1>/
    <class_name_2>/
    unclassified/
"""

import logging
import os
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2
import gradio as gr
import pymupdf

from src.utils.classify import FloorPlanClassifier
from src.utils.roi import FloorPlanROI

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

_log_file = LOG_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
_fmt = logging.Formatter(
    fmt="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_fmt)

_file_handler = logging.FileHandler(_log_file, encoding="utf-8")
_file_handler.setFormatter(_fmt)

logging.root.setLevel(logging.INFO)
logging.root.addHandler(_console_handler)
logging.root.addHandler(_file_handler)

logger = logging.getLogger(__name__)
logger.info(f"Log file: {_log_file.resolve()}")

# ---------------------------------------------------------------------------
# Constants / paths
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
RAW_PNG_DIR = DATA_DIR / "raw_png"
ROI_DIR = DATA_DIR / "roi_1280"
OUTPUT_DIR = DATA_DIR / "output"

ROI_CONFIG = "config/roi.yaml"
CLASSIFY_CONFIG = "config/classify.yaml"

# ---------------------------------------------------------------------------
# PDF -> PNG conversion
# ---------------------------------------------------------------------------

def pdf_to_png(pdf_path: Path, output_dir: Path) -> List[Path]:
    """Render every page of ``pdf_path`` to a PNG at 300 dpi.

    Returns:
        List of saved PNG paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    doc = pymupdf.open(str(pdf_path))
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        stem = pdf_path.stem
        out_name = output_dir / f"{stem}_page{page_num}.png"
        pix.save(str(out_name))
        logger.info(f"PDF->PNG  {out_name.name}")
        saved.append(out_name)

    doc.close()
    return saved


# ---------------------------------------------------------------------------
# Source resolution - ZIP or directory
# ---------------------------------------------------------------------------

def extract_zip(zip_path: str, extract_to: Path) -> List[Path]:
    """Extract a ZIP and return all PDF paths found inside it."""
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    pdfs = sorted(extract_to.rglob("*.pdf"))
    logger.info(f"Found {len(pdfs)} PDF(s) in ZIP")
    return pdfs


def collect_pdfs_from_dir(dir_path: str) -> List[Path]:
    """Recursively collect all PDF files under ``dir_path``."""
    root = Path(dir_path)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {dir_path}")
    pdfs = sorted(root.rglob("*.pdf"))
    logger.info(f"Found {len(pdfs)} PDF(s) in {dir_path}")
    return pdfs


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    zip_file,               # Gradio File value (tmp path) or None
    dir_path: str = "",     # local directory path or empty string
    run_id: Optional[str] = None,
) -> Generator[Tuple[str, List[str]], None, None]:
    """Run the full pipeline and yield (log_text, gallery_image_paths) updates.

    Exactly one of ``zip_file`` or ``dir_path`` must be provided.
    Yields progressively so the Gradio UI updates in real-time.
    """
    has_zip = zip_file is not None
    has_dir = bool(dir_path and dir_path.strip())

    if not has_zip and not has_dir:
        yield "WARNING: Provide either a ZIP file or a directory path.", []
        return
    if has_zip and has_dir:
        yield "WARNING: Provide only one input - either ZIP or directory, not both.", []
        return

    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    # Scoped directories for this run
    raw_dir = RAW_PNG_DIR / run_id
    roi_dir = ROI_DIR / run_id
    out_dir = OUTPUT_DIR / run_id

    log_lines: List[str] = [f"Run ID: {run_id}"]

    def emit(msg: str, gallery: Optional[List[str]] = None):
        log_lines.append(msg)
        return "\n".join(log_lines), gallery or []

    # ------------------------------------------------------------------
    # 1. Resolve PDFs from the chosen source
    # ------------------------------------------------------------------
    if has_zip:
        yield emit("Extracting ZIP...")
        try:
            zip_path = zip_file if isinstance(zip_file, str) else zip_file.name
            pdfs = extract_zip(zip_path, DATA_DIR / "zips" / run_id)
        except Exception as e:
            yield emit(f"ERROR: ZIP extraction failed: {e}")
            return
    else:
        yield emit(f"Scanning directory: {dir_path.strip()}")
        try:
            pdfs = collect_pdfs_from_dir(dir_path.strip())
        except Exception as e:
            yield emit(f"ERROR: {e}")
            return

    if not pdfs:
        yield emit("ERROR: No PDF files found.")
        return

    yield emit(f"Found {len(pdfs)} PDF(s)")

    # ------------------------------------------------------------------
    # 2. PDF -> PNG
    # ------------------------------------------------------------------
    yield emit("Converting PDFs to PNG...")
    all_pngs: List[Path] = []
    for pdf in pdfs:
        sub = raw_dir / pdf.parent.name
        pngs = pdf_to_png(pdf, sub)
        all_pngs.extend(pngs)

    yield emit(f"{len(all_pngs)} PNG(s) generated")

    # ------------------------------------------------------------------
    # 3. ROI detection
    # ------------------------------------------------------------------
    yield emit("Loading ROI model...")
    try:
        roi_model = FloorPlanROI(ROI_CONFIG)
    except Exception as e:
        yield emit(f"ERROR: ROI model failed to load: {e}")
        return

    yield emit("Running ROI detection...")
    roi_paths: List[Path] = []
    roi_failed = 0

    for i, png in enumerate(all_pngs, 1):
        out_path = roi_dir / png.parent.name / png.name
        _, found = roi_model.process_image(str(png), str(out_path))
        roi_paths.append(out_path)
        status = "ok" if found else "fallback"
        yield emit(f"  [{i}/{len(all_pngs)}]  {png.name}  [{status}]")
        if not found:
            roi_failed += 1

    yield emit(
        f"ROI done: {len(roi_paths) - roi_failed} detected, {roi_failed} fallback(s)",
        [str(p) for p in roi_paths[:8]],  # preview first 8
    )

    # ------------------------------------------------------------------
    # 4. Classification
    # ------------------------------------------------------------------
    yield emit("Loading classifier...")
    try:
        classifier = FloorPlanClassifier(CLASSIFY_CONFIG)
    except Exception as e:
        yield emit(f"ERROR: Classifier failed to load: {e}")
        return

    yield emit("Classifying and sorting images...")
    class_results = classifier.process_batch(
        [str(p) for p in roi_paths],
        str(out_dir),
    )

    summary = classifier.summary(class_results)
    yield emit(f"Classification complete\n\n{summary}")

    # ------------------------------------------------------------------
    # Final gallery - one sample per class
    # ------------------------------------------------------------------
    gallery: List[str] = []
    for cls_paths in class_results.values():
        gallery.extend(cls_paths[:3])

    yield emit(
        f"\nOutput folder: {out_dir.resolve()}"
        f"\nLog file:      {_log_file.resolve()}",
        gallery,
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def _pipeline_wrapper(zip_file, dir_path, _progress=gr.Progress(track_tqdm=True)):
    """Bridge between Gradio event handler and the generator pipeline."""
    log_text = ""
    gallery: List[str] = []
    for log_text, gallery in run_pipeline(zip_file, dir_path):
        yield log_text, gallery
    return log_text, gallery


with gr.Blocks(title="Wari Floor Plan Classifier") as demo:
    gr.Markdown(
        """
        # Wari Floor Plan - ROI & Classification
        Choose an input method, run the pipeline, and get floor plans sorted
        into class folders under `data/output/`.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.Tab("ZIP Upload"):
                    zip_input = gr.File(
                        label="Floor Plan ZIP",
                        file_types=[".zip"],
                        file_count="single",
                    )

                with gr.Tab("Directory Path"):
                    dir_input = gr.Textbox(
                        label="Root directory path",
                        placeholder="/path/to/folder/with/subfolders",
                        lines=1,
                    )

            run_btn = gr.Button("Run Pipeline", variant="primary")

        with gr.Column(scale=2):
            log_box = gr.Textbox(
                label="Pipeline Log",
                lines=20,
                max_lines=40,
                interactive=False,
            )

    gallery_out = gr.Gallery(
        label="Sample Output Images (ROI -> classified)",
        columns=4,
        height="auto",
        object_fit="contain",
    )

    run_btn.click(
        fn=_pipeline_wrapper,
        inputs=[zip_input, dir_input],
        outputs=[log_box, gallery_out],
    )

if __name__ == "__main__":
    demo.launch(share=False)
