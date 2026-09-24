"""
Script de predicción visual con YOLOv8 en los planos de FloorPlanCAD.
Dibuja las cajas delimitadoras y etiquetas sobre los planos y los guarda en disco.
"""

from pathlib import Path
from ultralytics import YOLO
import cv2

def run_visual_predictions():
    model_path = "YOLO/best_floorplancad.pt"
    images_dir = Path("data_samples/floorplancad/images")
    output_dir = Path("data_samples/floorplancad/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cargando modelo: {model_path}")
    model = YOLO(model_path)

    image_paths = sorted(list(images_dir.glob("*.png")))[:5]
    print(f"Generando predicciones visuales para {len(image_paths)} planos...")

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        results = model.predict(img, conf=0.20, verbose=False)[0]

        # Guardar imagen con cajas dibujadas
        annotated_frame = results.plot()
        out_path = output_dir / f"pred_{img_path.name}"
        cv2.imwrite(str(out_path), annotated_frame)

        detected_classes = [model.names[int(c)] for c in results.boxes.cls]
        print(f"\n[Plano {img_path.name}]")
        print(f"  -> Total elementos detectados: {len(detected_classes)}")
        for cls_name in set(detected_classes):
            print(f"     * {cls_name}: {detected_classes.count(cls_name)}")
        print(f"  -> Guardado en: {out_path}")

    print("\n" + "=" * 60)
    print("Predicciones visuales completadas exitosamente.")
    print(f"Puedes ver las imagenes con cajas en: {output_dir}")

if __name__ == "__main__":
    run_visual_predictions()
