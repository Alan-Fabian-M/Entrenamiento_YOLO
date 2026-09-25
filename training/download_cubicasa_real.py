"""
Descarga el dataset CRUDO original de CubiCasa5K (SVGs + imagenes), no el
mirror COCO de HuggingFace usado por `download_cubicasa_hf.py` -- ese mirror
solo tiene 8 categorias de bounding boxes, sin los SVGs vectoriales que
`training/cubicasa/svg_parser.py` necesita para extraer wall/room/muebles
fijos con geometria completa.

Fuente: Zenodo, record 2613548 (https://zenodo.org/records/2613548),
archivo `cubicasa5k.zip` (~5.47 GB). La URL vieja referenciada en notebooks
anteriores (record 3384994) ya no existe (404) -- se verifico contra la API
real de Zenodo antes de escribir este script.

Uso
---
  python -m training.download_cubicasa_real
  python -m training.download_cubicasa_real --output-dir YOLO/cubicasa5k_raw
  python -m training.download_cubicasa_real --skip-download  # solo descomprimir si ya esta el zip

Layout de salida
-----------------
YOLO/cubicasa5k_raw/
  cubicasa5k/
    train.txt / val.txt / test.txt   <- splits oficiales, ya vienen en el zip
    high_quality/<id>/{model.svg, F1_scaled.png, ...}
    high_quality_architectural/<id>/...
    colorful/<id>/...
"""

from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "YOLO" / "cubicasa5k_raw"
ZENODO_RECORD_ID = 2613548
ZENODO_DOWNLOAD_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files/cubicasa5k.zip/content"
EXPECTED_ZIP_SIZE_BYTES = 5_469_495_706  # confirmed via the Zenodo API, 2026-09


def _check_dependencies() -> None:
    try:
        import requests  # noqa: F401
        import tqdm  # noqa: F401
    except ImportError as exc:
        logger.error("Falta una dependencia: %s", exc)
        logger.error("Instalala con: pip install requests tqdm")
        sys.exit(1)


def download_zip(zip_path: Path) -> None:
    import requests
    from tqdm import tqdm

    if zip_path.exists() and zip_path.stat().st_size == EXPECTED_ZIP_SIZE_BYTES:
        logger.info("El zip ya existe y tiene el tamano esperado, no se vuelve a descargar: %s", zip_path)
        return

    logger.info("Descargando CubiCasa5K real desde Zenodo (~5.5 GB, puede tardar bastante)...")
    logger.info("URL: %s", ZENODO_DOWNLOAD_URL)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(ZENODO_DOWNLOAD_URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", EXPECTED_ZIP_SIZE_BYTES))
        with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="cubicasa5k.zip") as pbar:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))

    logger.info("Descarga completa: %s (%.2f GB)", zip_path, zip_path.stat().st_size / 1e9)


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    logger.info("Descomprimiendo %s en %s (puede tardar varios minutos)...", zip_path, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(output_dir)
    logger.info("OK Descompresion completa.")

    cubicasa_root = output_dir / "cubicasa5k"
    for split in ("train.txt", "val.txt", "test.txt"):
        split_path = cubicasa_root / split
        if not split_path.exists():
            logger.warning("No se encontro %s -- revisar el layout del zip descomprimido.", split_path)
        else:
            n_lines = len(split_path.read_text().splitlines())
            logger.info("  %s: %d muestras", split, n_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga el dataset CRUDO de CubiCasa5K (SVGs) desde Zenodo")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-download", action="store_true", help="No descargar, solo descomprimir un zip ya existente")
    parser.add_argument("--skip-extract", action="store_true", help="Solo descargar, no descomprimir")
    args = parser.parse_args()

    zip_path = args.output_dir / "cubicasa5k.zip"

    if not args.skip_download:
        _check_dependencies()
        download_zip(zip_path)

    if not args.skip_extract:
        if not zip_path.exists():
            logger.error("No existe el zip en %s -- corre sin --skip-download primero.", zip_path)
            sys.exit(1)
        extract_zip(zip_path, args.output_dir)

    logger.info("")
    logger.info("Listo. Para construir el dataset YOLO-seg:")
    logger.info("  python -m training.cli build --cubicasa-root %s/cubicasa5k \\", args.output_dir)
    logger.info("    --train-split %s/cubicasa5k/train.txt \\", args.output_dir)
    logger.info("    --val-split %s/cubicasa5k/val.txt \\", args.output_dir)
    logger.info("    --test-split %s/cubicasa5k/test.txt \\", args.output_dir)
    logger.info("    --output-root YOLO/cubicasa5k_seg")


if __name__ == "__main__":
    main()
