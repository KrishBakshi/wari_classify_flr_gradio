"""Floor Plan ROI Detection

Uses a YOLO-seg model to detect the floor plan region inside a raw scanned
image, crops that polygon, then letterbox-resizes the crop to a fixed square.

Training notes
--------------
- Model: YOLO11-seg
- Training data: 14 images  (Krish, initial round)
- Still performs well in practice despite the small dataset.
"""

import os
import cv2
import numpy as np
import yaml
import logging
from typing import List, Optional, Tuple, Union

from ultralytics import YOLO

from src.utils.letterbox import LetterboxResize

logger = logging.getLogger(__name__)


def _best_device() -> str:
    """Return the best available compute device (cuda > mps > cpu)."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class FloorPlanROI:
    """Detect and crop the floor-plan ROI from a raw scanned image.

    Config keys (roi.yaml)
    ----------------------
    model_path  : path to the YOLO-seg weights
    imgsz       : inference image size (e.g. 640)
    conf        : detection confidence threshold
    target_size : output letterbox size (e.g. 1280)
    """

    def __init__(self, config_path: str):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.model_path: str = cfg["model_path"]
        self.imgsz: int = cfg["imgsz"]
        self.conf: float = cfg["conf"]
        self.target_size: int = cfg["target_size"]

        self.model = YOLO(self.model_path)
        self.device = _best_device()
        self.letterbox = LetterboxResize(target_size=self.target_size)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _crop_polygon(self, image: np.ndarray, polygon: np.ndarray) -> np.ndarray:
        """Crop ``image`` to ``polygon``, masking the outside with white.

        A 10-pixel border is added around the polygon bounding box before
        cropping to avoid clipping edge pixels.
        """
        h, w = image.shape[:2]
        xs, ys = polygon[:, 0], polygon[:, 1]

        x_min = max(0, int(xs.min()) - 10)
        y_min = max(0, int(ys.min()) - 10)
        x_max = min(w, int(xs.max()) + 10)
        y_max = min(h, int(ys.max()) + 10)

        crop = image[y_min:y_max, x_min:x_max].copy()

        mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
        poly_local = polygon.copy().astype(np.float32)
        poly_local[:, 0] -= x_min
        poly_local[:, 1] -= y_min
        cv2.fillPoly(mask, [poly_local.astype(np.int32)], 255)

        if crop.ndim == 3:
            crop = np.where(mask[:, :, np.newaxis] > 0, crop, 255)
        else:
            crop = np.where(mask > 0, crop, 255)

        return crop

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_roi(
        self,
        image: Union[str, np.ndarray],
        return_all: bool = False,
    ) -> Union[Optional[np.ndarray], List[np.ndarray]]:
        """Run ROI detection and return letterbox-resized crop(s).

        Args:
            image:      Image path or BGR numpy array.
            return_all: Return every detected ROI when ``True``;
                        otherwise return only the highest-confidence one.

        Returns:
            A single ``(target_size, target_size, 3)`` array, a list of
            such arrays, or ``None`` / ``[]`` when nothing is detected.
        """
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                logger.error(f"Could not read image: {image}")
                return [] if return_all else None
        else:
            img = image

        results = self.model.predict(img, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)

        if results[0].masks is None:
            logger.warning("ROI: no masks detected")
            return [] if return_all else None

        rois: List[np.ndarray] = []
        for mask_xy in results[0].masks.xy:
            if mask_xy is None or len(mask_xy) == 0:
                continue
            crop = self._crop_polygon(img, mask_xy)
            canvas, _, _ = self.letterbox(crop)
            rois.append(canvas)

        if not rois:
            return [] if return_all else None

        return rois if return_all else rois[0]

    def process_image(
        self,
        image_path: str,
        output_path: str,
    ) -> Tuple[np.ndarray, bool]:
        """Run ROI detection on one file and save the result.

        Falls back to letterboxing the full image when no mask is found.

        Returns:
            (saved_image, roi_found)
        """
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not read image: {image_path}")
            return None, False

        roi = self.get_roi(img)
        roi_found = roi is not None

        if not roi_found:
            logger.warning(f"No ROI found for {image_path}; letterboxing full image")
            roi, _, _ = self.letterbox(img)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, roi)
        logger.info(f"Saved ROI -> {output_path}  shape={roi.shape}")
        return roi, roi_found
