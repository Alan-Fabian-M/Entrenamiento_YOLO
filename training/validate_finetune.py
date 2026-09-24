"""
Fase 5 del fine-tuning: valida un modelo contra el banco de imágenes de prueba
de estilos variados (CAD, ilustrado, esquemático), replicando EXACTAMENTE el
preprocesamiento que usa el pipeline en producción (variante shadow-free), para
que lo que medís acá sea lo que verá `SceneCompiler`.

Por cada imagen:
  - corre YOLO con un umbral bajo (para ver también detecciones sub-umbral),
  - lista cada detección con su confianza, marcando cuáles pasan el umbral de
    producción (settings.YOLO_CONFIDENCE_THRESHOLD) y separando mobiliario de
    elementos estructurales (puertas/ventanas/muros),
  - guarda la imagen anotada para revisión visual.

Importante: si la caja cae SOBRE el mueble real (y no es un "fantasma" sobre
texto o adornos) no se puede decidir automáticamente sin ground-truth. Por eso
el script guarda la imagen anotada: la verificación de ubicación es visual, igual
que hicimos a mano en las pruebas. La confianza sube la decide el número; la
ubicación correcta la confirmás mirando la imagen.

Uso:
  # Validar el modelo en producción (settings.YOLO_MODEL_PATH):
  .venv\\Scripts\\python.exe -m training.validate_finetune

  # Validar un best.pt recién entrenado:
  .venv\\Scripts\\python.exe -m training.validate_finetune --model runs/cubicasa/finetune_estilos/weights/best.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO

from app.core.config import settings
from app.services.preprocessor import ImagePreprocessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Banco de prueba de esta sesión: un CAD (detecta bien hoy), un esquemático y un
# ilustrado (hoy fallan). Sobrescribible con --images.
DEFAULT_TEST_IMAGES = [
    Path(r"D:\Descargas\croquis_test.png"),
    Path(r"D:\Descargas\cuantos-bloques-se-necesitan-para-hacer-un-cuarto-de-3x3-1.jpg"),
    Path(r"D:\Descargas\simplee.jpg"),
]

# Clases del modelo que son mobiliario real (lo que hoy falla en estilos no-CAD),
# frente a lo estructural (puertas/ventanas/muros/escaleras) que ya anda bien.
FURNITURE_CLASSES = {
    "bed", "sofa", "table", "chair", "toilet", "sink",
    "bath_tub", "refrigerator", "gas_stove", "wardrobe",
}


def _shadow_free_bgr(image_bgr: np.ndarray) -> np.ndarray:
    """Réplica exacta de la variante que SceneCompiler le pasa a YOLO."""
    gray = ImagePreprocessor.to_grayscale(image_bgr)
    sf = ImagePreprocessor.enhance_contrast(ImagePreprocessor.remove_shadows(gray))
    return np.stack([sf] * 3, axis=-1)


def validate(model_path: str, images: List[Path], conf: float, out_dir: Path) -> None:
    prod_thr = settings.YOLO_CONFIDENCE_THRESHOLD
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== VALIDACIÓN DE MODELO ===")
    logger.info("Modelo:            %s", model_path)
    logger.info("Umbral producción: %.2f", prod_thr)
    logger.info("Umbral muestreo:   %.2f (para ver sub-umbral)", conf)
    logger.info("")

    model = YOLO(model_path)

    for img_path in images:
        if not img_path.is_file():
            logger.warning("No existe: %s (omitida)", img_path)
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            logger.warning("No se pudo decodificar: %s (omitida)", img_path)
            continue

        result = model(_shadow_free_bgr(bgr), conf=conf, verbose=False)[0]

        logger.info("[%s]  %dx%d", img_path.name, bgr.shape[1], bgr.shape[0])
        furn_pasa = furn_sub = struct_pasa = 0
        vis = bgr.copy()

        for box in sorted(result.boxes, key=lambda b: -float(b.conf[0])):
            cls = model.names[int(box.cls[0])]
            c = float(box.conf[0])
            pasa = c >= prod_thr
            es_mueble = cls in FURNITURE_CLASSES
            marca = "PASA " if pasa else "sub  "
            tipo = "mueble" if es_mueble else "estruct"
            logger.info("    %s %-6s %-14s conf=%.3f", marca, tipo, cls, c)

            if es_mueble and pasa:
                furn_pasa += 1
            elif es_mueble:
                furn_sub += 1
            elif pasa:
                struct_pasa += 1

            # Dibujar: verde si pasa, gris si es sub-umbral; rojo si es mueble
            # sub-umbral (candidato dudoso que hay que revisar si es fantasma).
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            color = (0, 170, 0) if pasa else ((0, 0, 220) if es_mueble else (150, 150, 150))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"{cls} {c:.2f}", (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        logger.info("    -> muebles que PASAN: %d | muebles sub-umbral: %d | estructurales que pasan: %d",
                    furn_pasa, furn_sub, struct_pasa)

        out_path = out_dir / f"val_{img_path.stem}.png"
        cv2.imwrite(str(out_path), vis)
        logger.info("    -> anotada: %s (revisá que la caja caiga sobre el mueble, no sobre texto/adornos)", out_path)
        logger.info("")

    logger.info("Verde=pasa umbral | Rojo=mueble sub-umbral (revisar si es fantasma) | Gris=estructural sub-umbral")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validar modelo YOLO contra el banco de imágenes de estilos")
    parser.add_argument("--model", type=str, default=settings.YOLO_MODEL_PATH,
                        help="Ruta al .pt a validar (default: el de producción)")
    parser.add_argument("--images", type=Path, nargs="+", default=DEFAULT_TEST_IMAGES,
                        help="Imágenes de prueba")
    parser.add_argument("--conf", type=float, default=0.05,
                        help="Umbral bajo de muestreo para ver detecciones sub-umbral")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "runs" / "validation",
                        help="Carpeta para las imágenes anotadas")
    args = parser.parse_args()

    validate(args.model, list(args.images), args.conf, args.out)


if __name__ == "__main__":
    main()
