"""Floor Plan Classification

Uses a YOLO-classify model to assign each floor-plan image to one of the
configured categories and copies the image to the appropriate output folder.

Config keys (classify.yaml)
---------------------------
model_path  : path to the YOLO-classify weights
imgsz       : inference image size (e.g. 640)
conf        : minimum confidence to accept a prediction

Class names are read directly from the model (results[0].names) so they
do not need to be listed in the config file.
"""

import os
import shutil
import cv2
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from ultralytics import YOLO

logger = logging.getLogger(__name__)


def _best_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class FloorPlanClassifier:
    """Classify floor-plan images and sort them into per-class folders.

    Config keys (classify.yaml)
    ---------------------------
    model_path  : path to YOLO-classify weights
    imgsz       : inference resolution (e.g. 640)
    conf        : confidence threshold; images below this are put in
                  ``unclassified/``

    Class names are taken from the model itself (``model.names``).
    """

    def __init__(self, config_path: str):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.model_path: str = cfg["model_path"]
        self.imgsz: int = cfg["imgsz"]
        self.conf: float = cfg["conf"]

        self.model = YOLO(self.model_path)
        self.device = _best_device()

    # ------------------------------------------------------------------
    # Single-image inference
    # ------------------------------------------------------------------

    def classify(
        self,
        image: Union[str, np.ndarray],
    ) -> Tuple[int, str, float]:
        """Classify a single image.

        Args:
            image: Image path or BGR numpy array.

        Returns:
            (class_index, class_name, confidence)
            Returns (-1, "unclassified", 0.0) on failure.
        """
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                logger.error(f"Could not read image: {image}")
                return -1, "unclassified", 0.0
        else:
            img = image

        results = self.model.predict(img, imgsz=self.imgsz, device=self.device, verbose=False)
        probs = results[0].probs

        if probs is None:
            logger.warning("Classifier returned no probabilities")
            return -1, "unclassified", 0.0

        idx = int(probs.top1)
        conf = float(probs.top1conf)

        if conf < self.conf:
            logger.warning(f"Low confidence ({conf:.2f} < {self.conf}); marking unclassified")
            return -1, "unclassified", conf

        name = results[0].names.get(idx, f"class_{idx}")
        return idx, name, conf

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def process_batch(
        self,
        image_paths: List[str],
        output_dir: str,
    ) -> Dict[str, List[str]]:
        """Classify a list of images and copy them into per-class sub-folders.

        Folder structure created under ``output_dir``::

            output_dir/
              <class_name_0>/
              <class_name_1>/
              ...
              unclassified/    - images that did not pass the confidence threshold

        Args:
            image_paths: Paths to source images (already at the desired
                         resolution, typically 1280 x 1280 ROI crops).
            output_dir:  Root directory for the sorted output.

        Returns:
            Mapping of class_name to list of destination file paths.
        """
        results: Dict[str, List[str]] = {}

        for src in image_paths:
            _, class_name, conf = self.classify(src)
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            dst = os.path.join(class_dir, Path(src).name)
            shutil.copy2(src, dst)
            logger.info(f"[{class_name}] ({conf:.2f})  {Path(src).name}")

            results.setdefault(class_name, []).append(dst)

        return results

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    def summary(self, results: Dict[str, List[str]]) -> str:
        """Return a human-readable summary of batch classification results."""
        lines = ["Classification summary", "-" * 30]
        total = sum(len(v) for v in results.values())
        for cls, paths in sorted(results.items()):
            pct = 100 * len(paths) / total if total else 0
            lines.append(f"  {cls:<20} {len(paths):>4}  ({pct:.1f} %)")
        lines.append("-" * 30)
        lines.append(f"  {'TOTAL':<20} {total:>4}")
        return "\n".join(lines)
