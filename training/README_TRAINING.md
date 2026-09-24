# Guía de Entrenamiento CubiCasa5K con YOLOv8

El dataset ha sido descargado y convertido a formato YOLO compatible con **Ultralytics YOLO** tanto para **Segmentación de Instancias** (`YOLOv8-seg`) como para **Detección de Objetos** (`YOLOv8`).

---

## 1. Clases del Dataset

| ID | Clase | Descripción |
| :---: | :--- | :--- |
| **0** | `bathroom` | Baños y zonas húmedas |
| **1** | `bed` | Camas |
| **2** | `door` | Puertas de acceso y paso |
| **3** | `kitchen` | Cocinas y mesones |
| **4** | `room` | Habitaciones / espacios habitables |
| **5** | `stairs` | Escaleras |
| **6** | `wall` | Muros y paredes |
| **7** | `window` | Ventanas |

---

## 2. Archivos del Dataset

- **Configuración:** `YOLO/cubicasa_coco/data.yaml`
- **Imágenes:** `YOLO/cubicasa_coco/images/train/` (4.228) y `images/valid/` (748)
- **Etiquetas YOLO:** `YOLO/cubicasa_coco/labels/train/` (4.228) y `labels/valid/` (748)
- **Anotaciones originales COCO:** `YOLO/cubicasa_coco/annotations/`

---

## 3. Entrenamiento Local

### Segmentación (Recomendado para paredes y habitaciones)
```bash
.venv\Scripts\python -m training.train_yolo --task segment --model yolov8n-seg.pt --epochs 10 --batch 4 --imgsz 640
```

### Detección tradicional (Bounding boxes)
```bash
.venv\Scripts\python -m training.train_yolo --task detect --model yolov8n.pt --epochs 10 --batch 8 --imgsz 640
```

> **Nota:** Al entrenar en CPU local (Intel UHD Graphics), cada época tomará tiempo. Si dispones de GPU NVIDIA o acceso a Google Colab, el proceso es hasta 30 veces más rápido.

---

## 4. Entrenamiento en Google Colab (GPU Gratis T4)

Para entrenar con GPU gratuita en Google Colab:
1. Abre un notebook nuevo en [Google Colab](https://colab.research.google.com).
2. Selecciona Entorno de ejecución > Cambiar tipo de entorno > **T4 GPU**.
3. Ejecuta esta celda única:

```python
# 1. Instalar Ultralytics y datasets
!pip install -q ultralytics datasets huggingface_hub

# 2. Descargar y preparar dataset directamente en Colab
from datasets import load_dataset, Image
import os
from pathlib import Path
from tqdm import tqdm

# Cargar dataset
ds_train = load_dataset("phungpx/cubicassa5k-coco", split="train")
ds_valid = load_dataset("phungpx/cubicassa5k-coco", split="valid")

# Crear carpetas
for s in ["train", "valid"]:
    os.makedirs(f"dataset/images/{s}", exist_ok=True)
    os.makedirs(f"dataset/labels/{s}", exist_ok=True)

# Guardar imagenes y etiquetas
CAT_MAP = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7}

for split_name, ds in [("train", ds_train), ("valid", ds_valid)]:
    for row in tqdm(ds, desc=split_name):
        fn = Path(row["file_name"]).stem
        # Guardar imagen
        row["image"].save(f"dataset/images/{split_name}/{row['file_name']}")
        # Guardar labels seg
        w, h = float(row["width"]), float(row["height"])
        lines = []
        anns = row["annotations"]
        for cat_id, segs in zip(anns["category_id"], anns["segmentation"]):
            if cat_id in CAT_MAP:
                c = CAT_MAP[cat_id]
                for seg in segs:
                    if len(seg) >= 6:
                        coords = [f"{min(max(seg[i]/w, 0.), 1.):.6f} {min(max(seg[i+1]/h, 0.), 1.):.6f}" for i in range(0, len(seg), 2)]
                        lines.append(f"{c} " + " ".join(coords))
        with open(f"dataset/labels/{split_name}/{fn}.txt", "w") as f:
            f.write("\n".join(lines))

# 3. Crear data.yaml
yaml_content = \"\"\"
path: /content/dataset
train: images/train
val: images/valid
names:
  0: bathroom
  1: bed
  2: door
  3: kitchen
  4: room
  5: stairs
  6: wall
  7: window
\"\"\"
with open("data.yaml", "w") as f:
    f.write(yaml_content)

# 4. Entrenar YOLOv8 con GPU T4
from ultralytics import YOLO
model = YOLO("yolov8n-seg.pt")
results = model.train(data="data.yaml", epochs=30, imgsz=640, batch=16, device=0)
```

---

## 5. Fine-tuning: detección de muebles en estilos no-CAD

### 5.1 El problema

El modelo de mobiliario en producción (`YOLO/best_floorplancad.pt`) fue entrenado **solo** con FloorPlanCAD (line-art técnico). Por eso:

- ✅ Detecta bien en planos **CAD** (mobiliario con alta confianza) y **puertas en cualquier estilo** (el arco de apertura es muy característico).
- ❌ **No reconoce muebles** en dibujos **ilustrados/renderizados/esquemáticos**. Las detecciones de "cama" de baja confianza en esos estilos son *fantasmas* sobre texto de cotas o adornos, no el mueble real.

Verificado con tres imágenes de prueba (línea base):

| Imagen | Estilo | Muebles que pasan (umbral 0.35) | Nota |
| :--- | :--- | :---: | :--- |
| `croquis_test.png` | CAD | **8** | Funciona bien |
| `cuarto 3x3` | Esquemático | **0** | `bed` 0.12 fantasma sobre "2700" |
| `simplee.jpg` | Ilustrado | **0** | `bed` 0.21 fantasma sobre adornos |

**Bajar el umbral no sirve**: metería fantasmas sobre texto/adornos en vez del mueble real. La única solución es reentrenar con ejemplos de esos estilos.

### 5.2 Estrategia: transfer learning (NO desde cero)

Continuar entrenando **desde `best_floorplancad.pt`**, mezclando el dataset FloorPlanCAD existente con imágenes nuevas de estilos variados. Se mantienen las **mismas 16 clases** (ver `YOLO/floorplancad/data.yaml`) para no romper el mapeo por nombre de clase en `app/services/furniture_detector.py`.

Mezclar lo nuevo **con** lo viejo evita el *olvido catastrófico*: si entrenaras solo con las imágenes nuevas, el modelo mejoraría en esos estilos pero se olvidaría de los planos CAD que hoy detecta bien.

### 5.3 Recolección y etiquetado (trabajo manual)

Juntar **~150-200 imágenes nuevas** de planos top-down de habitaciones con muebles visibles, priorizando variedad de estilo (ilustrado, esquemático, dibujado a mano). *Variedad > cantidad.*

- **Etiquetado:** Roboflow (los datasets actuales salen de ahí), o CVAT/LabelImg local.
- **Formato:** YOLO detection (`.txt` con `clase cx cy w h` normalizado), idéntico a `YOLO/floorplancad/labels/`.
- **Clases a priorizar:** `bed, sofa, table, chair, wardrobe` (el mobiliario que hoy falla). No hace falta etiquetar `wall` (los muros los genera OpenCV en `RoomExtractor`).

Dejar las imágenes nuevas etiquetadas en una carpeta, p. ej. `YOLO/estilos_nuevos/` (layout plano `images/` + `labels/`, o con split `images/train` + `images/val`; el script detecta ambos).

### 5.4 Flujo de scripts

```bash
# Fase 3 — Fusionar FloorPlanCAD + imágenes nuevas -> YOLO/mixed_finetune/
.venv\Scripts\python -m training.build_mixed_dataset --custom YOLO/estilos_nuevos

# Fase 4 — Fine-tuning (Colab T4). Notar --model apuntando al best.pt existente:
.venv\Scripts\python -m training.train_yolo --task detect \
  --model YOLO/best_floorplancad.pt --data YOLO/mixed_finetune/data.yaml \
  --epochs 40 --lr0 0.001 --patience 12 --hsv-s 0.9 --hsv-v 0.6 \
  --batch 16 --device 0 --name finetune_estilos

# Fase 5 — Validar el resultado contra la línea base de la tabla 5.1:
.venv\Scripts\python -m training.validate_finetune \
  --model runs/cubicasa/finetune_estilos/weights/best.pt
```

**Flags de fine-tuning en `train_yolo.py`** (por encima del entrenamiento fresco):

| Flag | Recomendado | Por qué |
| :--- | :--- | :--- |
| `--lr0` | `0.001` | LR bajo: ya partimos de buenos pesos, no queremos pisarlos |
| `--patience` | `12` | Early-stop tras 12 épocas sin mejora |
| `--hsv-s` / `--hsv-v` | `0.9` / `0.6` | Augment de color fuerte: los estilos ilustrados son coloridos |
| `--freeze` | `10` (opcional) | Congela el backbone y reentrena solo la cabeza; más rápido y menos olvido si el dataset nuevo es chico |

### 5.5 Criterio de éxito

Al re-correr la **Fase 5** sobre el modelo afinado, comparado contra la tabla base 5.1:

1. `cuarto 3x3` y `simplee.jpg` pasan a tener **muebles que superan el umbral**, con la caja **sobre el mueble real** (verificación visual en `runs/validation/`, no fantasma sobre texto/adornos).
2. `croquis_test.png` (CAD) **sigue** detectando sus 8 muebles → no hubo olvido catastrófico.
3. Bajan las detecciones fantasma sobre texto/adornos.

### 5.6 Una vez validado

Copiar el nuevo `best.pt` a `YOLO/` y apuntar `YOLO_MODEL_PATH` (en `app/core/config.py` o la variable de entorno) a esos pesos. El pipeline lo toma sin cambios de código — `furniture_detector.py` mapea por nombre de clase, que no cambió.
