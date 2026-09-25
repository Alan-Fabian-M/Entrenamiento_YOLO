"""
Fusiona el dataset CubiCasa5K (reconstruido via `training.cli build`, formato
YOLO-seg, taxonomia de `training/cubicasa/class_mapping.py`) con el dataset
FloorPlanCAD (formato YOLO-detect/bbox, taxonomia de
`training/convert_floorplancad_to_yolo.py`) en un solo dataset YOLO-seg con
vocabulario de clases unificado.

Por que hace falta un script separado de `training/build_mixed_dataset.py`
(que ya combina datasets, pero asume que ambas fuentes comparten el mismo
`TARGET_CLASSES` fijo y el mismo formato bbox): aqui las dos fuentes tienen
taxonomias Y formatos de label distintos, asi que cada linea se remapea a un
class_id unificado y las lineas bbox de FloorPlanCAD se convierten a un
poligono degenerado de 4 esquinas (mismo formato seg que ya usa CubiCasa) en
vez de copiarse tal cual.

CubiCasa5K no tiene cama/sofa/mesa/silla (ver `class_mapping.py`); esas 4
clases se toman de FloorPlanCAD. El resto de clases se comparte por nombre
(wall, door, window, stairs/stair, toilet, sink, bathtub/bath_tub,
wardrobe) -- un solo class_id final por nombre, sin duplicar.

Uso
---
  # 1. Regenerar FloorPlanCAD localmente (si YOLO/floorplancad/ no existe):
  python -m training.download_floorplancad_hf --all
  python -m training.convert_floorplancad_to_yolo

  # 2. Construir el dataset CubiCasa (ver training/cli.py build)

  # 3. Fusionar ambos:
  python -m training.build_cubicasa_floorplancad_mix \\
      --cubicasa YOLO/cubicasa5k_seg --floorplancad YOLO/floorplancad \\
      --output YOLO/unified_train
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from training.build_mixed_dataset import _collect_dataset
from training.convert_floorplancad_to_yolo import TARGET_CLASSES as FLOORPLANCAD_CLASSES
from training.cubicasa.class_mapping import YOLO_CLASSES as CUBICASA_CLASSES

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CUBICASA = PROJECT_ROOT / "YOLO" / "cubicasa5k_seg"
DEFAULT_FLOORPLANCAD = PROJECT_ROOT / "YOLO" / "floorplancad"
DEFAULT_OUTPUT = PROJECT_ROOT / "YOLO" / "unified_train"

# Same-name aliases between the two taxonomies (FloorPlanCAD name -> CubiCasa
# name), so both map to a single unified class instead of two duplicates.
_FLOORPLANCAD_TO_CUBICASA_ALIAS = {
    "stair": "stairs",
    "bath_tub": "bathtub",
    "single_door": "door",
    "double_door": "door",
    "sliding_door": "door",
}

# FloorPlanCAD classes that are genuinely NOT in CubiCasa5K's vocabulary
# (movable furniture -- CubiCasa5K only annotates FIXED fixtures).
_FLOORPLANCAD_ONLY = ["bed", "sofa", "table", "chair", "refrigerator", "gas_stove"]


def build_unified_classes() -> List[str]:
    unified = list(CUBICASA_CLASSES)
    for name in _FLOORPLANCAD_ONLY:
        if name not in unified:
            unified.append(name)
    return unified


def _cubicasa_remap(unified: List[str]) -> Dict[int, int]:
    return {i: unified.index(name) for i, name in enumerate(CUBICASA_CLASSES)}


def _floorplancad_remap(unified: List[str]) -> Dict[int, int]:
    remap = {}
    for i, name in enumerate(FLOORPLANCAD_CLASSES):
        target = _FLOORPLANCAD_TO_CUBICASA_ALIAS.get(name, name)
        if target in unified:
            remap[i] = unified.index(target)
        else:
            # "wardrobe"/"sink"/"window" etc already share names directly;
            # anything truly unmatched (shouldn't happen given the alias
            # table above) gets dropped rather than silently mislabeled.
            logger.warning("FloorPlanCAD class %r has no unified mapping -- instances will be dropped", name)
    return remap


def _remap_seg_line(line: str, remap: Dict[int, int]) -> str | None:
    """Rewrite a YOLO-seg line's class_id (`class x1 y1 x2 y2 ...`)."""
    parts = line.split()
    if not parts:
        return None
    old_id = int(parts[0])
    if old_id not in remap:
        return None
    return f"{remap[old_id]} " + " ".join(parts[1:])


def _bbox_to_polygon_line(line: str, remap: Dict[int, int]) -> str | None:
    """Convert a FloorPlanCAD bbox line (`class cx cy w h`) to a degenerate
    4-corner polygon line in the same seg format CubiCasa emits, so the
    merged dataset is 100% seg (mixing bbox and polygon label lines in one
    YOLOv8-seg training run is unreliable/underdocumented -- keep it uniform).
    """
    parts = line.split()
    if len(parts) != 5:
        return None
    old_id = int(parts[0])
    if old_id not in remap:
        return None
    cx, cy, w, h = (float(v) for v in parts[1:])
    x0, x1 = max(0.0, cx - w / 2), min(1.0, cx + w / 2)
    y0, y1 = max(0.0, cy - h / 2), min(1.0, cy + h / 2)
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in corners)
    return f"{remap[old_id]} {coords}"


def _copy_pairs_remapped(
    pairs: List[Tuple[Path, Path]], prefix: str, img_out: Path, lbl_out: Path,
    remap: Dict[int, int], line_converter,
) -> Tuple[int, int]:
    import shutil

    n_instances = 0
    for img_path, label_path in pairs:
        new_stem = f"{prefix}__{img_path.stem}"
        shutil.copy2(img_path, img_out / f"{new_stem}{img_path.suffix.lower()}")

        new_lines: List[str] = []
        if label_path.is_file():
            for raw_line in label_path.read_text(encoding="utf-8").splitlines():
                if not raw_line.strip():
                    continue
                converted = line_converter(raw_line, remap)
                if converted is not None:
                    new_lines.append(converted)
        n_instances += len(new_lines)
        (lbl_out / f"{new_stem}.txt").write_text(
            "\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8"
        )
    return len(pairs), n_instances


def build_unified_dataset(
    cubicasa_dir: Path, floorplancad_dir: Path, output_dir: Path,
    val_ratio: float = 0.2, seed: int = 42,
) -> None:
    for tag, d in (("cubicasa", cubicasa_dir), ("floorplancad", floorplancad_dir)):
        if not d.is_dir():
            raise FileNotFoundError(f"No existe el dataset de origen [{tag}]: {d}")

    unified = build_unified_classes()
    cubicasa_remap = _cubicasa_remap(unified)
    floorplancad_remap = _floorplancad_remap(unified)

    train_img = output_dir / "images" / "train"
    val_img = output_dir / "images" / "val"
    train_lbl = output_dir / "labels" / "train"
    val_lbl = output_dir / "labels" / "val"
    for d in (train_img, val_img, train_lbl, val_lbl):
        d.mkdir(parents=True, exist_ok=True)

    logger.info("=== Vocabulario unificado (%d clases) ===", len(unified))
    logger.info("%s", unified)

    grand_train = grand_val = grand_instances = 0
    for tag, d, remap, converter in (
        ("cubicasa", cubicasa_dir, cubicasa_remap, _remap_seg_line),
        ("fpcad", floorplancad_dir, floorplancad_remap, _bbox_to_polygon_line),
    ):
        train_pairs, val_pairs = _collect_dataset(d, val_ratio, seed)
        n_tr, i_tr = _copy_pairs_remapped(train_pairs, tag, train_img, train_lbl, remap, converter)
        n_va, i_va = _copy_pairs_remapped(val_pairs, tag, val_img, val_lbl, remap, converter)
        grand_train += n_tr
        grand_val += n_va
        grand_instances += i_tr + i_va
        logger.info("  [%s] train=%d  val=%d  (instancias=%d)", tag, n_tr, n_va, i_tr + i_va)

    data_yaml = (
        "# Dataset UNIFICADO CubiCasa5K + FloorPlanCAD (formato YOLO-seg).\n"
        "# Generado por training/build_cubicasa_floorplancad_mix.py -- no editar a mano.\n"
        f"path: {output_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "names:\n"
    )
    for idx, name in enumerate(unified):
        data_yaml += f"  {idx}: {name}\n"
    (output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")

    logger.info("=== Fusion completada ===")
    logger.info("Total: train=%d  val=%d  instancias=%d", grand_train, grand_val, grand_instances)
    logger.info("data.yaml -> %s", output_dir / "data.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fusionar CubiCasa5K + FloorPlanCAD en un dataset YOLO-seg unificado")
    parser.add_argument("--cubicasa", type=Path, default=DEFAULT_CUBICASA)
    parser.add_argument("--floorplancad", type=Path, default=DEFAULT_FLOORPLANCAD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_unified_dataset(args.cubicasa, args.floorplancad, args.output, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
