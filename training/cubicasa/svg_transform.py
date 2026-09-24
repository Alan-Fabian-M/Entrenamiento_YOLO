"""Affine transform algebra for composing nested SVG `transform` attributes."""

import re
from typing import List, Optional, Tuple
import numpy as np

_FUNC_RE = re.compile(r"([a-zA-Z]+)\s*\(([^)]*)\)")
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


class SvgTransformError(Exception):
    """Raised when an SVG transform attribute cannot be parsed."""
    pass


class SvgTransform:
    """Parses and composes SVG `transform` attributes as 3x3 affine matrices."""

    IDENTITY = np.eye(3, dtype=np.float64)

    @staticmethod
    def _parse_args(raw_args: str) -> List[float]:
        return [float(m) for m in _NUM_RE.findall(raw_args)]

    @staticmethod
    def _parse_single_function(name: str, args: List[float]) -> np.ndarray:
        """Convert one transform function (matrix/translate/scale/rotate/skewX/skewY)
        into its equivalent 3x3 homogeneous matrix.
        """
        name = name.lower()
        m = np.eye(3, dtype=np.float64)

        if name == "matrix":
            if len(args) != 6:
                raise SvgTransformError(f"matrix() expects 6 args, got {len(args)}")
            a, b, c, d, e, f = args
            m = np.array([[a, c, e], [b, d, f], [0.0, 0.0, 1.0]], dtype=np.float64)

        elif name == "translate":
            if len(args) not in (1, 2):
                raise SvgTransformError(f"translate() expects 1-2 args, got {len(args)}")
            tx = args[0]
            ty = args[1] if len(args) == 2 else 0.0
            m = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)

        elif name == "scale":
            if len(args) not in (1, 2):
                raise SvgTransformError(f"scale() expects 1-2 args, got {len(args)}")
            sx = args[0]
            sy = args[1] if len(args) == 2 else sx
            m = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)

        elif name == "rotate":
            if len(args) not in (1, 3):
                raise SvgTransformError(f"rotate() expects 1 or 3 args, got {len(args)}")
            theta = np.radians(args[0])
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rot = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=np.float64)
            if len(args) == 3:
                cx, cy = args[1], args[2]
                to_origin = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
                from_origin = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float64)
                m = from_origin @ rot @ to_origin
            else:
                m = rot

        elif name == "skewx":
            if len(args) != 1:
                raise SvgTransformError(f"skewX() expects 1 arg, got {len(args)}")
            m = np.array([[1, np.tan(np.radians(args[0])), 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)

        elif name == "skewy":
            if len(args) != 1:
                raise SvgTransformError(f"skewY() expects 1 arg, got {len(args)}")
            m = np.array([[1, 0, 0], [np.tan(np.radians(args[0])), 1, 0], [0, 0, 1]], dtype=np.float64)

        else:
            raise SvgTransformError(f"Unsupported transform function: {name}")

        return m

    @classmethod
    def parse_transform_attr(cls, transform_str: Optional[str]) -> np.ndarray:
        """Parse a full SVG `transform` attribute (possibly composed of several
        functions, e.g. "translate(10,20) rotate(45)") into a single 3x3 matrix.
        """
        if not transform_str or not transform_str.strip():
            return cls.IDENTITY.copy()

        matrix = cls.IDENTITY.copy()
        for name, raw_args in _FUNC_RE.findall(transform_str):
            args = cls._parse_args(raw_args)
            func_matrix = cls._parse_single_function(name, args)
            matrix = matrix @ func_matrix

        return matrix

    @staticmethod
    def compose(parent: np.ndarray, child: np.ndarray) -> np.ndarray:
        """Accumulate parent -> child transforms: parent @ child."""
        return parent @ child

    @staticmethod
    def apply_to_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Apply a 3x3 affine matrix to an Nx2 array of points, returning Nx2."""
        points = np.asarray(points, dtype=np.float64)
        n = points.shape[0]
        homogeneous = np.hstack([points, np.ones((n, 1), dtype=np.float64)])
        transformed = (matrix @ homogeneous.T).T
        return transformed[:, :2]
