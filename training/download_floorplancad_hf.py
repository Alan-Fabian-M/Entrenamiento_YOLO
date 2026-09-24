"""
Descargador para el dataset FloorPlanCAD desde Hugging Face (Voxel51/FloorPlanCAD).

Contiene mas de 5.300 planos arquitectonicos CAD reales con muros, puertas,
ventanas, muebles, cotas de medicion y lineas estructurales.

Uso:
----
  # Descargar 20 planos de muestra para pruebas
  .venv\\Scripts\\python.exe -m training.download_floorplancad_hf --num-samples 20

  # Descargar a una carpeta personalizada
  .venv\\Scripts\\python.exe -m training.download_floorplancad_hf --output-dir data_samples/floorplancad --num-samples 50

  # Descargar todos los planos del dataset
  .venv\\Scripts\\python.exe -m training.download_floorplancad_hf --all
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from PIL import Image
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ID = "Voxel51/FloorPlanCAD"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data_samples" / "floorplancad"


def download_floorplancad(output_dir: Path, num_samples: int | None = 20) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Obteniendo lista de archivos y metadatos de Hugging Face: %s...", REPO_ID)
    
    api = HfApi()
    repo_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    png_files = [f for f in repo_files if f.startswith("data/") and f.endswith(".png")]
    
    total_available = len(png_files)
    logger.info("Total de planos arquitectonicos disponibles: %d", total_available)

    if num_samples is not None and num_samples > 0:
        target_files = png_files[:num_samples]
    else:
        target_files = png_files

    logger.info("Descargando %d planos a '%s'...", len(target_files), images_dir)

    downloaded = 0
    for file_rel in tqdm(target_files, desc="Descargando planos"):
        src_path = hf_hub_download(repo_id=REPO_ID, filename=file_rel, repo_type="dataset")
        filename = Path(file_rel).name
        dest_path = images_dir / filename
        
        # Guardar en formato RGB estandarizado
        with Image.open(src_path) as img:
            rgb_img = img.convert("RGB")
            rgb_img.save(dest_path, format="PNG")
        downloaded += 1

    summary_info = {
        "dataset": REPO_ID,
        "total_downloaded": downloaded,
        "classes": [
            "wall", "single_door", "double_door", "sliding_door",
            "window", "stair", "table", "chair", "sofa", "bed",
            "bath", "toilet", "sink", "washing_machine", "refrigerator"
        ],
        "image_directory": str(images_dir)
    }
    summary_path = output_dir / "dataset_info.json"
    summary_path.write_text(json.dumps(summary_info, indent=2), encoding="utf-8")

    logger.info("Descarga completada con exito.")
    logger.info("Planos disponibles en: %s", images_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Descargador de FloorPlanCAD desde Hugging Face")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directorio destino")
    parser.add_argument("--num-samples", type=int, default=20, help="Numero de planos a descargar (por defecto 20)")
    parser.add_argument("--all", action="store_true", help="Descargar todos los planos")
    args = parser.parse_args()

    samples_to_download = None if args.all else args.num_samples
    download_floorplancad(args.output_dir, num_samples=samples_to_download)


if __name__ == "__main__":
    main()
