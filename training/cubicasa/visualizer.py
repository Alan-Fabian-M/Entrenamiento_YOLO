"""Draws normalized YOLO-OBB labels onto an image for visual verification."""

from typing import List, Tuple
import cv2
import numpy as np

_PALETTE = [
    (66, 133, 244), (219, 68, 55), (244, 180, 0), (15, 157, 88),
    (171, 71, 188), (255, 112, 67), (0, 172, 193), (124, 179, 66),
    (94, 53, 177), (0, 121, 107), (216, 27, 96), (57, 73, 171),
    (198, 40, 40), (0, 131, 143), (104, 159, 56), (230, 81, 0),
    (69, 90, 100), (48, 63, 159),
]


class Visualizer:
    """Renders OBB labels over an image for manual inspection."""

    @staticmethod
    def color_for_class(class_id: int) -> Tuple[int, int, int]:
        """Deterministic BGR color per class_id, stable across runs."""
        return _PALETTE[class_id % len(_PALETTE)]

    @staticmethod
    def draw_obb_labels(
        image: np.ndarray, label_lines: List[str], class_names: List[str]
    ) -> np.ndarray:
        """Parse normalized YOLO-OBB label lines, denormalize to this image's
        pixel space, and draw each oriented box with a class-colored outline
        and label text.
        """
        vis = image.copy()
        img_h, img_w = vis.shape[0], vis.shape[1]

        for line in label_lines:
            parts = line.strip().split()
            if len(parts) != 9:
                continue
            class_id = int(parts[0])
            coords = np.array([float(v) for v in parts[1:]], dtype=np.float64).reshape(4, 2)
            coords_px = (coords * np.array([img_w, img_h])).astype(np.int32)

            color = Visualizer.color_for_class(class_id)
            cv2.polylines(vis, [coords_px], isClosed=True, color=color, thickness=2)

            label = class_names[class_id] if class_id < len(class_names) else str(class_id)
            anchor = tuple(coords_px[0])
            cv2.putText(vis, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        return vis
