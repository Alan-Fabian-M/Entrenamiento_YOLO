"""
Fase 3 del fine-tuning: fusiona el dataset FloorPlanCAD existente con imágenes
nuevas de estilos variados (ilustrado, esquemático, dibujado a mano) en un solo
dataset YOLO, para afinar `best_floorplancad.pt` sin "olvido catastrófico".

Mezclar lo nuevo CON lo viejo es deliberado: si entrenaras solo con las imágenes
nuevas, el modelo mejoraría en esos estilos pero se olvidaría de detectar los
planos CAD que hoy funcionan bien. Manteniendo FloorPlanCAD en la mezcla, el
modelo conserva lo que ya sabe mientras aprende los estilos nuevos.

Cada dataset de origen puede venir en dos layouts (se detecta solo):
  - Split Ultralytics:  images/train, images/val, labels/train, labels/val
  - Plano (flat):       images/ + labels/  (se reparte train/val con --val-ratio)

Los archivos se renombran con un prefijo por origen (p. ej. `fpcad__`, `custom__`)
para que nunca colisionen dos nombres iguales de datasets distintos.

Uso:
  # FloorPlanCAD (default) + tu carpeta de imágenes nuevas ya etiquetadas:
  .venv\\Scripts\\python.exe -m training.build_mixed_dataset --custom YOLO/estilos_nuevos

  # Varias carpetas nuevas a la vez:
  .venv\\Scripts\\python.exe -m training.build_mixed_dataset --custom YOLO/ilustrado YOLO/esquematico
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FLOORPLANCAD = PROJECT_ROOT / "YOLO" / "floorplancad"
DEFAULT_OUTPUT = PROJECT_ROOT / "YOLO" / "mixed_finetune"

# Mismas 16 clases que YOLO/floorplancad/data.yaml -- NO cambiar el orden ni los
# nombres: `app/services/furniture_detector.py` mapea por nombre de clase.
TARGET_CLASSES = [
    "wall", "single_door", "double_door", "sliding_door", "window", "stair",
    "bed", "sofa", "table", "chair", "toilet", "sink", "bath_tub",
    "refrigerator", "gas_stove", "wardrobe",
]

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _find_pairs(dataset_dir: Path, split: str) -> List[Tuple[Path, Path]]:
    """Devuelve pares (imagen, label.txt) para un split ('train' o 'val').

    Soporta el layout con subcarpetas (images/train + labels/train) y el layout
    plano (images/ + labels/), en cuyo caso el reparto train/val lo hace el
    llamador y aquí se ignora `split`.
    """
    img_split_dir = dataset_dir / "images" / split
    lbl_split_dir = dataset_dir / "labels" / split
    if img_split_dir.is_dir():
        images_dir, labels_dir = img_split_dir, lbl_split_dir
    else:
        images_dir, labels_dir = dataset_dir / "images", dataset_dir / "labels"

    pairs: List[Tuple[Path, Path]] = []
    if not images_dir.is_dir():
        return pairs

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in _IMAGE_EXTS:
            continue
        label_path = labels_dir / f"{img_path.stem}.txt"
        # Una imagen sin .txt es una imagen sin objetos etiquetados (negativo
        # válido en YOLO); se incluye con un label vacío para que el modelo
        # también aprenda de fondos sin muebles.
        pairs.append((img_path, label_path))
    return pairs


def _has_split_layout(dataset_dir: Path) -> bool:
    return (dataset_dir / "images" / "train").is_dir()


def _collect_dataset(dataset_dir: Path, val_ratio: float, seed: int) -> Tuple[List, List]:
    """Devuelve (train_pairs, val_pairs) para un dataset, respetando su split si
    ya lo trae, o repartiéndolo si viene plano."""
    if _has_split_layout(dataset_dir):
        return _find_pairs(dataset_dir, "train"), _find_pairs(dataset_dir, "val")

    flat = _find_pairs(dataset_dir, "train")  # split se ignora en layout plano
    random.Random(seed).shuffle(flat)
    num_val = int(len(flat) * val_ratio)
    return flat[num_val:], flat[:num_val]


def _copy_pairs(pairs: List[Tuple[Path, Path]], prefix: str,
                img_out: Path, lbl_out: Path) -> Tuple[int, int]:
    """Copia pares a destino con prefijo anticolisión. Devuelve (n_img, n_boxes)."""
    n_boxes = 0
    for img_path, label_path in pairs:
        new_stem = f"{prefix}__{img_path.stem}"
        shutil.copy2(img_path, img_out / f"{new_stem}{img_path.suffix.lower()}")

        if label_path.is_file():
            content = label_path.read_text(encoding="utf-8")
            n_boxes += sum(1 for ln in content.splitlines() if ln.strip())
        else:
            content = ""  # negativo: imagen sin objetos
        (lbl_out / f"{new_stem}.txt").write_text(content, encoding="utf-8")
    return len(pairs), n_boxes


def build_mixed_dataset(
    floorplancad_dir: Path,
    custom_dirs: List[Path],
    output_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    sources: List[Tuple[str, Path]] = [("fpcad", floorplancad_dir)]
    for i, d in enumerate(custom_dirs):
        # Prefijo legible por carpeta; si se repite un nombre, se desambigua.
        sources.append((f"custom{i}_{d.name}", d))

    for _tag, d in sources:
        if not d.is_dir():
            raise FileNotFoundError(f"No existe el dataset de origen: {d}")

    train_img = output_dir / "images" / "train"
    val_img = output_dir / "images" / "val"
    train_lbl = output_dir / "labels" / "train"
    val_lbl = output_dir / "labels" / "val"
    for d in (train_img, val_img, train_lbl, val_lbl):
        d.mkdir(parents=True, exist_ok=True)

    logger.info("=== Fusionando %d datasets en %s ===", len(sources), output_dir)
    grand_train = grand_val = grand_boxes = 0
    for tag, d in sources:
        train_pairs, val_pairs = _collect_dataset(d, val_ratio, seed)
        n_tr, b_tr = _copy_pairs(train_pairs, tag, train_img, train_lbl)
        n_va, b_va = _copy_pairs(val_pairs, tag, val_img, val_lbl)
        grand_train += n_tr
        grand_val += n_va
        grand_boxes += b_tr + b_va
        logger.info("  [%s] train=%d  val=%d  (cajas=%d)", tag, n_tr, n_va, b_tr + b_va)

    data_yaml = f"""# Dataset FUSIONADO para fine-tuning (FloorPlanCAD + estilos nuevos)
# Generado por training/build_mixed_dataset.py -- no editar a mano.
path: {output_dir.as_posix()}
train: images/train
val: images/val

names:
"""
    for idx, name in enumerate(TARGET_CLASSES):
        data_yaml += f"  {idx}: {name}\n"
    (output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")

    logger.info("=== Fusión completada ===")
    logger.info("Total: train=%d  val=%d  cajas=%d", grand_train, grand_val, grand_boxes)
    logger.info("data.yaml -> %s", output_dir / "data.yaml")
    if grand_val == 0:
        logger.warning("El set de validación quedó vacío: agregá imágenes de val o subí --val-ratio.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fusionar FloorPlanCAD + estilos nuevos para fine-tuning")
    parser.add_argument("--floorplancad", type=Path, default=DEFAULT_FLOORPLANCAD,
                        help="Dataset FloorPlanCAD base (layout Ultralytics)")
    parser.add_argument("--custom", type=Path, nargs="+", required=True,
                        help="Una o más carpetas con imágenes nuevas etiquetadas (YOLO)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Proporción de val para datasets en layout plano (sin split propio)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_mixed_dataset(args.floorplancad, list(args.custom), args.output, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
