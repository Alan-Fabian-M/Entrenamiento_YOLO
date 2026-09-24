"""
Script para entrenar YOLO (detección o segmentación) con el dataset CubiCasa5K-COCO.

Uso:
  # Entrenamiento rápido de segmentación (recomendado para planos):
  python -m training.train_yolo --task segment --model yolov8n-seg.pt --epochs 10

  # Entrenamiento de detección tradicional con bounding boxes:
  python -m training.train_yolo --task detect --model yolov8n.pt --epochs 10

  # Con GPU dedicada (si estuviera disponible):
  python -m training.train_yolo --device 0

  # Ver opciones:
  python -m training.train_yolo --help
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_YAML = Path("D:/unity project/backend-Software2/YOLO/cubicasa_coco/data.yaml")


def train(
    data_yaml: Path,
    task: str = "segment",
    model_name: str | None = None,
    epochs: int = 10,
    imgsz: int = 640,
    batch: int = 8,
    device: str = "cpu",
    project: str = "runs/cubicasa",
    name: str = "exp1",
    lr0: float = 0.01,
    patience: int = 100,
    freeze: int | None = None,
    hsv_h: float = 0.015,
    hsv_s: float = 0.7,
    hsv_v: float = 0.4,
) -> None:
    if model_name is None:
        model_name = "yolov8n-seg.pt" if task == "segment" else "yolov8n.pt"

    if not data_yaml.exists():
        raise FileNotFoundError(f"No se encontro el archivo data.yaml en: {data_yaml}")

    logger.info("=== CONFIGURACIÓN DE ENTRENAMIENTO ===")
    logger.info("Tarea:      %s", task)
    logger.info("Modelo:     %s", model_name)
    logger.info("Dataset:    %s", data_yaml)
    logger.info("Epocas:     %d", epochs)
    logger.info("Resolucion: %d px", imgsz)
    logger.info("Batch:      %d", batch)
    logger.info("Dispositivo: %s", device)
    logger.info("LR inicial: %s", lr0)
    logger.info("Patience:   %d", patience)
    logger.info("Freeze:     %s", freeze if freeze is not None else "ninguna capa")
    logger.info("HSV aug:    h=%s s=%s v=%s", hsv_h, hsv_s, hsv_v)
    logger.info("Salida:     %s/%s", project, name)

    # Cargar modelo pre-entrenado (para fine-tuning, pasar --model YOLO/best_floorplancad.pt)
    model = YOLO(model_name)

    # Iniciar entrenamiento. Los parámetros de fine-tuning por encima de los
    # defaults de entrenamiento fresco:
    #   - lr0 más bajo: ya partimos de buenos pesos, no queremos pisarlos.
    #   - hsv_* de color más fuertes: los estilos ilustrados/renderizados son
    #     coloridos y variados; forzar variación de tono/saturación/brillo ayuda
    #     a generalizar más allá del line-art monocromo de FloorPlanCAD.
    #   - freeze opcional: congelar el backbone (~10 capas) reentrena solo la
    #     cabeza de detección, más rápido y con menos riesgo de olvido si el
    #     dataset nuevo es chico.
    train_kwargs = dict(
        data=str(data_yaml.as_posix()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        workers=2,
        plots=True,
        lr0=lr0,
        patience=patience,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
    )
    if freeze is not None:
        train_kwargs["freeze"] = freeze

    results = model.train(**train_kwargs)

    logger.info("Entrenamiento finalizado exitosamente!")
    best_pt = Path(project) / name / "weights" / "best.pt"
    if best_pt.exists():
        logger.info("Mejores pesos guardados en: %s", best_pt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenar YOLOv8 con dataset CubiCasa5K")
    parser.add_argument(
        "--task",
        choices=["segment", "detect"],
        default="segment",
        help="Tipo de tarea: 'segment' (recomendado para paredes y habitaciones) o 'detect'",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Modelo base pre-entrenado (ej: yolov8n-seg.pt, yolov8s-seg.pt, yolov8n.pt)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_YAML,
        help="Ruta al archivo data.yaml",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Numero de epocas")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de imagen (ej: 640, 1024)")
    parser.add_argument("--batch", type=int, default=8, help="Tamaño del batch")
    parser.add_argument("--device", type=str, default="cpu", help="Dispositivo ('cpu' o '0' para GPU)")
    parser.add_argument("--name", type=str, default="cubicasa_run", help="Nombre del experimento")
    # --- Fine-tuning (transfer learning desde un best.pt existente) ---
    parser.add_argument("--lr0", type=float, default=0.01,
                        help="Learning rate inicial. Para fine-tuning usar ~0.001")
    parser.add_argument("--patience", type=int, default=100,
                        help="Épocas sin mejora antes de early-stop (fine-tuning: ~10-15)")
    parser.add_argument("--freeze", type=int, default=None,
                        help="Congelar las primeras N capas (backbone ~10). Omitir para entrenar todo")
    parser.add_argument("--hsv-h", type=float, default=0.015, help="Augment: variación de tono")
    parser.add_argument("--hsv-s", type=float, default=0.7, help="Augment: variación de saturación")
    parser.add_argument("--hsv-v", type=float, default=0.4, help="Augment: variación de brillo")

    args = parser.parse_args()

    train(
        data_yaml=args.data,
        task=args.task,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        lr0=args.lr0,
        patience=args.patience,
        freeze=args.freeze,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
    )


if __name__ == "__main__":
    main()
