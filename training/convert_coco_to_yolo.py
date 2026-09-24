"""
Convert CubiCasa5K-COCO annotations to YOLO format (segmentation and detection).

Input layout:
  YOLO/cubicasa_coco/
    images/train/
    images/valid/
    annotations/instances_train.json
    annotations/instances_valid.json

Output layout:
  YOLO/cubicasa_coco/
    labels/train/   <- archivos .txt con anotaciones normalizadas
    labels/valid/   <- archivos .txt con anotaciones normalizadas
    data.yaml       <- configuracion lista para Ultralytics YOLO (detect y seg)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Mapeo oficial documentado de phungpx/cubicassa5k-coco
# Category ID 0 es placeholder ("objects"), IDs reales 1 a 8
CATEGORY_MAP = {
    1: 0,  # bathroom
    2: 1,  # bed
    3: 2,  # door
    4: 3,  # kitchen
    5: 4,  # room
    6: 5,  # stairs
    7: 6,  # wall
    8: 7,  # window
}

CLASS_NAMES = [
    "bathroom",
    "bed",
    "door",
    "kitchen",
    "room",
    "stairs",
    "wall",
    "window",
]


def convert_coco_split(
    json_path: Path,
    labels_dir: Path,
    mode: str = "seg",
) -> tuple[int, int]:
    """
    Convierte un archivo de anotaciones COCO a archivos de texto YOLO.
    
    mode:
      - 'seg': poligonos normalizados (class_id x1 y1 x2 y2 ... xn yn)
      - 'bbox': bounding boxes normalizadas (class_id x_center y_center w h)
    """
    labels_dir.mkdir(parents=True, exist_ok=True)

    if not json_path.exists():
        logger.error("No se encontro el archivo de anotaciones: %s", json_path)
        return 0, 0

    logger.info("Cargando anotaciones desde %s...", json_path.name)
    with open(json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    images = {img["id"]: img for img in coco_data.get("images", [])}
    annotations = coco_data.get("annotations", [])

    logger.info("  %d imagenes, %d anotaciones.", len(images), len(annotations))

    # Agrupar anotaciones por imagen
    img_to_anns: dict[int, list[dict]] = {img_id: [] for img_id in images}
    for ann in annotations:
        img_id = ann.get("image_id")
        if img_id in img_to_anns:
            img_to_anns[img_id].append(ann)

    total_converted = 0

    for img_id, img_info in tqdm(images.items(), desc=f"  Convirtiendo {json_path.stem}"):
        file_name = img_info["file_name"]
        stem = Path(file_name).stem
        txt_path = labels_dir / f"{stem}.txt"

        w = float(img_info["width"])
        h = float(img_info["height"])
        if w <= 0 or h <= 0:
            continue

        lines = []
        for ann in img_to_anns[img_id]:
            raw_cat_id = ann.get("category_id")
            if raw_cat_id not in CATEGORY_MAP:
                continue
            cls_idx = CATEGORY_MAP[raw_cat_id]

            if mode == "seg" and ann.get("segmentation"):
                # Formato YOLO-seg: class_id x1 y1 x2 y2 ... xn yn
                segs = ann["segmentation"]
                # En COCO, segmentation puede ser lista de listas [[x,y,...]]
                for seg in segs:
                    if len(seg) < 6:  # Un poligono necesita al menos 3 vertices (6 coords)
                        continue
                    coords_norm = []
                    for i in range(0, len(seg), 2):
                        x_norm = min(max(seg[i] / w, 0.0), 1.0)
                        y_norm = min(max(seg[i + 1] / h, 0.0), 1.0)
                        coords_norm.extend([f"{x_norm:.6f}", f"{y_norm:.6f}"])
                    lines.append(f"{cls_idx} " + " ".join(coords_norm))
                    total_converted += 1
            else:
                # Formato YOLO-bbox: class_id x_center y_center width height
                bbox = ann.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                bx, by, bw, bh = bbox
                if bw <= 0 or bh <= 0:
                    continue
                x_center = min(max((bx + bw / 2.0) / w, 0.0), 1.0)
                y_center = min(max((by + bh / 2.0) / h, 0.0), 1.0)
                w_norm = min(max(bw / w, 0.0), 1.0)
                h_norm = min(max(bh / h, 0.0), 1.0)
                lines.append(f"{cls_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                total_converted += 1

        if lines:
            txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            # Archivo vacio si no hay objetos para indicar imagen negativa
            txt_path.write_text("", encoding="utf-8")

    return len(images), total_converted


def generate_data_yaml(dataset_dir: Path) -> Path:
    """Genera el archivo data.yaml para Ultralytics YOLO."""
    yaml_path = dataset_dir / "data.yaml"
    # Rutas relativas compatibles tanto en local como en Colab/Linux
    content = f"""# Dataset CubiCasa5K-COCO para Ultralytics YOLO
path: {dataset_dir.as_posix()}
train: images/train
val: images/valid

# Clases (8 clases de arquitectura de planos)
names:
"""
    for idx, name in enumerate(CLASS_NAMES):
        content += f"  {idx}: {name}\n"

    yaml_path.write_text(content, encoding="utf-8")
    logger.info("Archivo de configuracion creado en: %s", yaml_path)
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convierte CubiCasa5K COCO a formato YOLO")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("D:/unity project/backend-Software2/YOLO/cubicasa_coco"),
        help="Ruta base del dataset cubicasa_coco",
    )
    parser.add_argument(
        "--mode",
        choices=["seg", "bbox"],
        default="seg",
        help="Modo de salida: 'seg' (poligonos para segmentacion y deteccion) o 'bbox' (bounding boxes tradicionales)",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    annotations_dir = dataset_dir / "annotations"
    labels_dir = dataset_dir / "labels"

    logger.info("Iniciando conversion a formato YOLO (modo: %s)...", args.mode)

    for split in ["train", "valid"]:
        json_file = annotations_dir / f"instances_{split}.json"
        split_labels = labels_dir / split
        n_imgs, n_anns = convert_coco_split(json_file, split_labels, mode=args.mode)
        logger.info("Split '%s': %d imagenes procesadas, %d anotaciones convertidas.", split, n_imgs, n_anns)

    generate_data_yaml(dataset_dir)
    logger.info("Conversion completa.")


if __name__ == "__main__":
    main()
