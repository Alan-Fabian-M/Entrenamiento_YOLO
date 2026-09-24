"""
Conversor de anotaciones de FloorPlanCAD (FiftyOne / Hugging Face) a formato YOLO.

Genera la estructura estándar requerida por Ultralytics YOLO:
  YOLO/floorplancad/
    images/train/
    images/val/
    labels/train/
    labels/val/
    data.yaml

Uso:
  .venv\\Scripts\\python.exe -m training.convert_floorplancad_to_yolo
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ID = "Voxel51/FloorPlanCAD"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_IMAGES_DIR = PROJECT_ROOT / "data_samples" / "floorplancad" / "images"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "YOLO" / "floorplancad"

# Mapeo de clases principales para arquitectura
TARGET_CLASSES = [
    "wall",
    "single_door",
    "double_door",
    "sliding_door",
    "window",
    "stair",
    "bed",
    "sofa",
    "table",
    "chair",
    "toilet",
    "sink",
    "bath_tub",
    "refrigerator",
    "gas_stove",
    "wardrobe",
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(TARGET_CLASSES)}


def convert_floorplancad_to_yolo(
    images_dir: Path = SAMPLES_IMAGES_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    random.seed(seed)
    
    if not images_dir.exists():
        raise FileNotFoundError(f"No existe el directorio de imagenes: {images_dir}")

    downloaded_images = {p.name: p for p in images_dir.glob("*.png")}
    if not downloaded_images:
        logger.error("No se encontraron imagenes .png en %s", images_dir)
        return

    logger.info("Cargando anotaciones de FloorPlanCAD...")
    samples_path = hf_hub_download(repo_id=REPO_ID, filename="samples.json", repo_type="dataset")
    with open(samples_path, "r", encoding="utf-8") as f:
        samples_data = json.load(f)

    # Crear carpetas de salida
    train_img_dir = output_dir / "images" / "train"
    val_img_dir = output_dir / "images" / "val"
    train_lbl_dir = output_dir / "labels" / "train"
    val_lbl_dir = output_dir / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Indexar anotaciones por nombre de archivo
    sample_by_filename = {}
    for sample in samples_data.get("samples", []):
        fp = sample.get("filepath", "")
        fn = Path(fp).name
        if fn in downloaded_images:
            sample_by_filename[fn] = sample

    matched_files = list(sample_by_filename.keys())
    random.shuffle(matched_files)

    num_val = int(len(matched_files) * val_ratio)
    val_files = set(matched_files[:num_val])
    train_files = set(matched_files[num_val:])

    logger.info(
        "Procesando %d planos disponibles (Train: %d, Val: %d)...",
        len(matched_files), len(train_files), len(val_files)
    )

    total_boxes = 0
    class_stats = {cls_name: 0 for cls_name in TARGET_CLASSES}

    for fn, sample in sample_by_filename.items():
        is_val = fn in val_files
        target_img_dir = val_img_dir if is_val else train_img_dir
        target_lbl_dir = val_lbl_dir if is_val else train_lbl_dir

        # Copiar imagen
        shutil.copy2(downloaded_images[fn], target_img_dir / fn)

        # Generar archivo de etiquetas YOLO (.txt)
        txt_path = target_lbl_dir / f"{Path(fn).stem}.txt"
        lines = []

        detections = sample.get("ground_truth", {}).get("detections", [])
        for det in detections:
            label = det.get("label")
            if label not in CLASS_TO_ID:
                continue

            cls_id = CLASS_TO_ID[label]
            bbox = det.get("bounding_box", [])  # [x_top_left, y_top_left, width, height]
            if len(bbox) != 4:
                continue

            x_tl, y_tl, w, h = bbox
            # Convertir a formato YOLO: x_center, y_center, width, height (normalizados 0-1)
            x_center = max(0.0, min(1.0, x_tl + w / 2.0))
            y_center = max(0.0, min(1.0, y_tl + h / 2.0))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))

            lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
            class_stats[label] += 1
            total_boxes += 1

        txt_path.write_text("\n".join(lines), encoding="utf-8")

    # Generar data.yaml
    data_yaml_content = f"""# Configuración de dataset FloorPlanCAD para YOLOv8
path: {output_dir.as_posix()}
train: images/train
val: images/val

names:
"""
    for idx, name in enumerate(TARGET_CLASSES):
        data_yaml_content += f"  {idx}: {name}\n"

    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(data_yaml_content, encoding="utf-8")

    logger.info("Conversion completada con exito!")
    logger.info("Total de cajas delimitadoras generadas: %d", total_boxes)
    logger.info("Estadisticas por clase:")
    for cls_name, count in class_stats.items():
        if count > 0:
            logger.info("  - %-15s: %d instancias", cls_name, count)
    logger.info("Archivo de configuracion creado en: %s", yaml_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convertir dataset FloorPlanCAD a formato YOLO")
    parser.add_argument("--images-dir", type=Path, default=SAMPLES_IMAGES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    convert_floorplancad_to_yolo(args.images_dir, args.output_dir, args.val_ratio)


if __name__ == "__main__":
    main()
