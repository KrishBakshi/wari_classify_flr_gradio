import cv2
import numpy as np
import logging
from typing import Tuple, Union

logger = logging.getLogger(__name__)


class LetterboxResize:
    """Resize an image to a square canvas while preserving aspect ratio.

    The image is scaled so its longest side equals ``target_size``, then
    centered on a solid-color canvas of ``target_size x target_size``.
    """

    def __init__(
        self,
        target_size: int = 1280,
        pad_color: Union[int, Tuple[int, int, int]] = (255, 255, 255),
    ):
        self.target_size = target_size
        self.pad_color = pad_color

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def __call__(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        return self.resize(image)

    def resize(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Letterbox-resize ``image`` to ``target_size x target_size``.

        Returns:
            (canvas, scale, (pad_x, pad_y))
            - canvas:        Resized and padded image.
            - scale:         Scale factor applied to the original.
            - (pad_x, pad_y): Pixel offsets of the top-left corner of the
                             placed image on the canvas.
        """
        if image is None or image.size == 0:
            logger.error("LetterboxResize: input image is empty")
            return None, 1.0, (0, 0)

        h, w = image.shape[:2]
        scale = self.target_size / max(h, w)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        is_gray = len(image.shape) == 2
        if is_gray:
            fill = self.pad_color[0] if isinstance(self.pad_color, (tuple, list)) else int(self.pad_color)
            canvas = np.full((self.target_size, self.target_size), fill, dtype=np.uint8)
        else:
            fill = (int(self.pad_color), int(self.pad_color), int(self.pad_color)) if isinstance(self.pad_color, int) else tuple(int(v) for v in self.pad_color)
            canvas = np.full((self.target_size, self.target_size, 3), fill, dtype=np.uint8)

        pad_x = (self.target_size - new_w) // 2
        pad_y = (self.target_size - new_h) // 2
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        return canvas, scale, (pad_x, pad_y)

    # ------------------------------------------------------------------
    # Inverse transforms
    # ------------------------------------------------------------------

    def reverse_coords(
        self,
        points: np.ndarray,
        scale: float,
        pad_offset: Tuple[int, int],
    ) -> np.ndarray:
        """Map point coordinates from canvas space back to original image space.

        Args:
            points:     (N, 2) array of (x, y) coordinates in canvas space.
            scale:      Scale returned by :meth:`resize`.
            pad_offset: (pad_x, pad_y) returned by :meth:`resize`.
        """
        if points.size == 0:
            return points.copy()

        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)

        pad_x, pad_y = pad_offset
        out = pts.copy()
        out[:, 0] = (pts[:, 0] - pad_x) / scale
        out[:, 1] = (pts[:, 1] - pad_y) / scale
        return out

    def reverse_bbox(
        self,
        bbox: np.ndarray,
        scale: float,
        pad_offset: Tuple[int, int],
    ) -> np.ndarray:
        """Map bounding boxes from canvas space back to original image space.

        Args:
            bbox:       (4,) or (N, 4) array of [x1, y1, x2, y2] boxes.
            scale:      Scale returned by :meth:`resize`.
            pad_offset: (pad_x, pad_y) returned by :meth:`resize`.
        """
        if bbox.size == 0:
            return bbox.copy()

        boxes = np.asarray(bbox, dtype=np.float32)
        squeezed = boxes.ndim == 1
        if squeezed:
            boxes = boxes.reshape(1, -1)

        pad_x, pad_y = pad_offset
        out = boxes.copy()
        out[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        out[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
        return out.squeeze() if squeezed else out
