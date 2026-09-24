"""Image preprocessing service for architectural floor plans and blueprints."""

import base64
import time
from typing import Dict, Any, Tuple
import cv2
import numpy as np


class PreprocessingError(Exception):
    """Raised when image preprocessing fails."""
    pass


class ImagePreprocessor:
    """Specialized computer vision pipeline for architectural floor plans photographed

    or scanned under non-ideal lighting conditions.
    """

    @staticmethod
    def decode_image(image_bytes: bytes) -> np.ndarray:
        """Decode raw image bytes into an OpenCV BGR numpy array."""
        if not image_bytes:
            raise PreprocessingError("Empty image bytes provided.")

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise PreprocessingError("Could not decode image. Format may be corrupt or unsupported.")

        return img

    @staticmethod
    def encode_image(img: np.ndarray, ext: str = ".png") -> bytes:
        """Encode an OpenCV image back to bytes."""
        success, encoded = cv2.imencode(ext, img)
        if not success:
            raise PreprocessingError(f"Failed to encode image to {ext}")
        return encoded.tobytes()

    @staticmethod
    def to_base64(img: np.ndarray, ext: str = ".png") -> str:
        """Convert an OpenCV image to base64 string for API responses."""
        raw_bytes = ImagePreprocessor.encode_image(img, ext)
        return base64.b64encode(raw_bytes).decode("utf-8")

    @staticmethod
    def to_grayscale(img: np.ndarray) -> np.ndarray:
        """Convert BGR or RGBA image to single-channel grayscale."""
        if len(img.shape) == 2:
            return img
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @classmethod
    def remove_shadows(cls, gray: np.ndarray) -> np.ndarray:
        """Eliminate uneven lighting, paper folds, and hand shadows using

        morphological background division.

        Floor plan lines and text are dark, paper is light. We estimate the background
        illumination via morphological closing with a large elliptical kernel, then
        divide the original grayscale by the background to flatten lighting.
        """
        # Morphological dilation/closing to isolate background illumination
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        background = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
        background = cv2.medianBlur(background, 21)

        # Background division: (gray / background) * 255
        # Areas where background matches gray become 255 (white paper).
        # Dark lines/ink retain their contrast.
        diff = cv2.divide(gray, background, scale=255)
        
        # Normalize to full 0-255 dynamic range
        normalized = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        return normalized

    @classmethod
    def enhance_contrast(cls, gray: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to sharpen

        architectural lines, wall boundaries, and numerical measurements.
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        return clahe.apply(gray)

    @classmethod
    def binarize(cls, gray: np.ndarray, method: str = "adaptive") -> np.ndarray:
        """Binarize the grayscale image into crisp black/white geometry.

        - 'adaptive': Best for photographed paper with micro-variations.
        - 'otsu': Best for uniformly scanned digital drawings.
        """
        if method == "otsu":
            # Apply light Gaussian blur before Otsu to eliminate fine paper grain
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary

        # Adaptive Gaussian Thresholding
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=8,
        )
        return binary

    @classmethod
    def clean_noise(cls, binary: np.ndarray) -> np.ndarray:
        """Remove isolated salt-and-pepper pixel noise without disconnecting walls."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return cleaned

    @classmethod
    def process(cls, image_bytes: bytes, filter_mode: str = "binary") -> Tuple[bytes, str]:
        """High-level processing function returning (encoded_bytes, media_type).

        Modes:
        - 'binary': Full pipeline (grayscale -> shadow removal -> CLAHE -> adaptive binary -> denoise).
        - 'shadow_free': Grayscale with shadow removal and contrast enhancement (ideal for YOLO).
        - 'grayscale': Basic grayscale conversion.
        """
        bgr = cls.decode_image(image_bytes)
        gray = cls.to_grayscale(bgr)

        if filter_mode == "grayscale":
            result = gray
        elif filter_mode == "shadow_free":
            shadow_free = cls.remove_shadows(gray)
            result = cls.enhance_contrast(shadow_free)
        elif filter_mode == "binary":
            shadow_free = cls.remove_shadows(gray)
            enhanced = cls.enhance_contrast(shadow_free)
            binary = cls.binarize(enhanced, method="adaptive")
            result = cls.clean_noise(binary)
        elif filter_mode == "otsu":
            shadow_free = cls.remove_shadows(gray)
            enhanced = cls.enhance_contrast(shadow_free)
            result = cls.binarize(enhanced, method="otsu")
        else:
            raise PreprocessingError(f"Unknown filter mode '{filter_mode}'. Allowed: binary, shadow_free, grayscale, otsu")

        return cls.encode_image(result, ".png"), "image/png"

    @classmethod
    def process_pipeline_detailed(cls, image_bytes: bytes) -> Dict[str, Any]:
        """Execute the entire pipeline and return all intermediate stages in Base64

        for pipeline inspection, debugging, and multi-sensor routing.
        """
        start_time = time.perf_counter()

        bgr = cls.decode_image(image_bytes)
        h, w = bgr.shape[:2]

        t0 = time.perf_counter()
        gray = cls.to_grayscale(bgr)
        t_gray = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        shadow_free = cls.remove_shadows(gray)
        t_shadow = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        enhanced = cls.enhance_contrast(shadow_free)
        t_clahe = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        binary_adaptive = cls.clean_noise(cls.binarize(enhanced, method="adaptive"))
        t_binary = (time.perf_counter() - t0) * 1000

        total_time_ms = (time.perf_counter() - start_time) * 1000

        return {
            "metadata": {
                "width": int(w),
                "height": int(h),
                "channels": int(bgr.shape[2]) if len(bgr.shape) > 2 else 1,
                "total_processing_time_ms": round(total_time_ms, 2),
                "timing_breakdown_ms": {
                    "grayscale": round(t_gray, 2),
                    "shadow_removal": round(t_shadow, 2),
                    "contrast_enhancement": round(t_clahe, 2),
                    "binarization": round(t_binary, 2),
                },
            },
            "stages_base64": {
                "grayscale": cls.to_base64(gray),
                "shadow_free": cls.to_base64(shadow_free),
                "contrast_enhanced": cls.to_base64(enhanced),
                "binary": cls.to_base64(binary_adaptive),
            },
        }
