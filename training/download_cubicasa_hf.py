"""
Download the CubiCasa5K-COCO dataset from HuggingFace.

Dataset: phungpx/cubicassa5k-coco
Format: Parquet with embedded images + COCO-style annotations.

Usage
-----
  pip install datasets Pillow tqdm huggingface_hub
  python -m training.download_cubicasa_hf
  python -m training.download_cubicasa_hf --no-export
  python -m training.download_cubicasa_hf --split train
  python -m training.download_cubicasa_hf --output-dir D:/datasets/cubicasa

Output layout
-------------
YOLO/cubicasa_coco/
  images/
    train/   <- PNGs de entrenamiento (4 230 imagenes)
    valid/   <- PNGs de validacion (748 imagenes)
  annotations/
    instances_train.json
    instances_valid.json
  categories.json
"""

from __future__ import annotations

import argparse
import json
import io
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "YOLO" / "cubicasa_coco"
HF_DATASET_ID = "phungpx/cubicassa5k-coco"
SPLITS = ["train", "valid"]


def _check_dependencies() -> None:
    missing = []
    for pkg, import_name in [
        ("datasets", "datasets"),
        ("Pillow", "PIL"),
        ("tqdm", "tqdm"),
        ("huggingface_hub", "huggingface_hub"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)
    if missing:
        logger.error("Faltan dependencias: %s", ", ".join(missing))
        logger.error("Instalalas con: pip install %s", " ".join(missing))
        sys.exit(1)


def download_and_export(output_dir: Path, splits: list[str], export: bool = True) -> None:
    from datasets import load_dataset, Image as DsImage
    from PIL import Image
    from tqdm import tqdm

    logger.info("Cargando dataset '%s' desde HuggingFace...", HF_DATASET_ID)
    logger.info("La primera descarga puede tardar varios minutos segun tu conexion.")
    logger.info("Los Parquet se cachean en ~/.cache/huggingface/datasets/")

    ds_list = load_dataset(HF_DATASET_ID, split=splits)
    # load_dataset devuelve Dataset si splits es str, list si es list
    if not isinstance(ds_list, list):
        ds_list = [ds_list]

    # Desactivar decodificacion automatica a PIL para escribir directamente los bytes a disco
    try:
        ds_list = [ds.cast_column("image", DsImage(decode=False)) for ds in ds_list]
    except Exception as e:
        logger.warning("No se pudo cast_column decode=False: %s. Se usara decodificacion PIL.", e)

    if not export:
        logger.info("--no-export activo. Cache descargada, nada escrito a disco.")
        for name, ds in zip(splits, ds_list):
            logger.info("  Split '%s': %d registros", name, len(ds))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "annotations").mkdir(exist_ok=True)

    all_categories: dict = {}

    for split_name, split_ds in zip(splits, ds_list):
        images_out = output_dir / "images" / split_name
        images_out.mkdir(parents=True, exist_ok=True)

        coco_images = []
        coco_annotations = []

        logger.info("Exportando split '%s' (%d imagenes)...", split_name, len(split_ds))

        for row in tqdm(split_ds, desc=f"  {split_name}"):
            img_id   = int(row["image_id"])
            file_name = row["file_name"]
            width    = int(row["width"])
            height   = int(row["height"])

            # Guardar imagen: escribir bytes directamente si estan disponibles (ultra rapido)
            out_path = images_out / file_name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            img_val = row["image"]
            if isinstance(img_val, dict) and img_val.get("bytes"):
                out_path.write_bytes(img_val["bytes"])
            elif hasattr(img_val, "save"):
                img_val.convert("RGB").save(out_path, format="PNG")
            else:
                Image.open(io.BytesIO(img_val["bytes"])).convert("RGB").save(out_path, format="PNG")

            coco_images.append({"id": img_id, "file_name": file_name, "width": width, "height": height})

            # annotations viene en formato columnar: dict de listas, no lista de dicts.
            # Ejemplo: {"id": [1,2,...], "category_id": [3,1,...], "bbox": [[x,y,w,h],...]}
            anns = row["annotations"]
            ann_ids       = anns.get("id", [])
            ann_cat_ids   = anns.get("category_id", [])
            ann_bboxes    = anns.get("bbox", [])
            ann_areas     = anns.get("area", [])
            ann_iscrowds  = anns.get("iscrowd", [])
            ann_segs      = anns.get("segmentation", [None] * len(ann_ids))

            for ann_id, cat_id, bbox, area, iscrowd, seg in zip(
                ann_ids, ann_cat_ids, ann_bboxes, ann_areas, ann_iscrowds, ann_segs
            ):
                cat_id = int(cat_id)
                if cat_id not in all_categories:
                    all_categories[cat_id] = {
                        "id": cat_id,
                        "name": f"category_{cat_id}",
                        "supercategory": "floor_plan",
                    }
                coco_annotations.append({
                    "id":           int(ann_id),
                    "image_id":     img_id,
                    "category_id":  cat_id,
                    "bbox":         [float(v) for v in bbox],
                    "area":         float(area),
                    "iscrowd":      int(iscrowd),
                    "segmentation": seg if seg is not None else [],
                })

        cats_sorted = sorted(all_categories.values(), key=lambda c: c["id"])
        coco_json = {"images": coco_images, "annotations": coco_annotations, "categories": cats_sorted}
        json_path = output_dir / "annotations" / f"instances_{split_name}.json"
        json_path.write_text(json.dumps(coco_json, indent=2), encoding="utf-8")
        logger.info("  -> %d imagenes, %d anotaciones -> %s", len(coco_images), len(coco_annotations), json_path)

    cats_path = output_dir / "categories.json"
    cats_path.write_text(json.dumps(sorted(all_categories.values(), key=lambda c: c["id"]), indent=2), encoding="utf-8")

    logger.info("")
    logger.info("OK Descarga completa. Datos en: %s", output_dir)
    logger.info("Categorias detectadas guardadas en: %s", cats_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga CubiCasa5K-COCO desde HuggingFace")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-export", action="store_true", help="Solo cachear, no exportar a disco")
    parser.add_argument("--split", choices=["train", "valid", "both"], default="both")
    args = parser.parse_args()

    splits = SPLITS if args.split == "both" else [args.split]

    _check_dependencies()
    download_and_export(args.output_dir, splits=splits, export=not args.no_export)


if __name__ == "__main__":
    main()
