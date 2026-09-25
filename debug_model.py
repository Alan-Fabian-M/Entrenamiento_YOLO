"""Debug script: run the trained YOLO-seg model directly on an image,
bypassing the FastAPI pipeline, to see raw detections at very low confidence.

Usage:
    python3 debug_model.py "imagenes/WhatsApp Image 2026-09-21 at 5.24.56 PM.jpeg"
"""
import sys
from ultralytics import YOLO

MODEL_PATH = "YOLO/best_seg.pt"

def main():
    if len(sys.argv) < 2:
        print("Uso: python3 debug_model.py <ruta_imagen>")
        sys.exit(1)

    image_path = sys.argv[1]

    print(f"Cargando modelo: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print(f"\nTask del modelo: {model.task}")
    print(f"Clases del modelo ({len(model.names)}):")
    for idx, name in model.names.items():
        print(f"  {idx}: {name}")

    print(f"\nCorriendo inferencia en: {image_path}")
    print("(confidence=0.05 para ver TODO, incluso detecciones débiles)\n")

    results = model(image_path, conf=0.05, verbose=True)

    for result in results:
        print(f"\n--- Resultado ---")
        print(f"Boxes: {len(result.boxes) if result.boxes is not None else 0}")
        print(f"Masks: {len(result.masks) if result.masks is not None else 0}")

        if result.boxes is not None and len(result.boxes) > 0:
            print("\nDetecciones (clase, confianza):")
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                conf = float(box.conf[0])
                print(f"  {cls_name}: {conf:.3f}")
        else:
            print("\n⚠️  NINGUNA detección, ni siquiera con conf=0.05")
            print("Esto indica que el modelo no está reconociendo nada en esta imagen.")

if __name__ == "__main__":
    main()
