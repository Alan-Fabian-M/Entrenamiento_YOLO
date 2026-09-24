# 📊 Informe de Entrenamiento: YOLOv8 - FloorPlanCAD (200 Planos)

**Fecha:** 19 de Septiembre, 2026  
**Tarea:** Detección de elementos arquitectónicos y mobiliario en planos CAD  
**Dataset:** [Voxel51/FloorPlanCAD](https://huggingface.co/datasets/Voxel51/FloorPlanCAD) (Hugging Face)  
**Pesos generados:** `YOLO/best_floorplancad.pt`  

---

## 1. ⚙️ Configuración del Entrenamiento

| Parámetro | Valor |
| :--- | :--- |
| **Modelo Base** | `yolov8n.pt` fine-tuned con pesos previos |
| **Imágenes Totales** | 200 planos arquitectónicos CAD en alta resolución (1000x1000 px) |
| **División Train / Val** | 160 entrenamiento (173 imágenes procesadas) / 40 validación (49 imágenes) |
| **Total de Anotaciones (Bounding Boxes)** | **1.815 cajas delimitadoras** |
| **Épocas** | **20 épocas** |
| **Batch Size** | 8 |
| **Resolución de Entrada** | 640 x 640 px |
| **Dispositivo** | CPU (Intel Core i7-8550U) |
| **Tiempo de Entrenamiento** | ~40 minutos (0.66 horas) |

---

## 2. 📈 Comparativa de Rendimiento

Comparación directa entre la primera prueba (50 planos) y el nuevo modelo escalado (200 planos):

| Elemento / Clase | 50 Planos (15 épocas) | 200 Planos (20 épocas) | Mejora / Estado |
| :--- | :---: | :---: | :---: |
| **`sliding_door` (Puerta corrediza)** | 36.0% | **92.2%** | 🟢 **+56.2%** |
| **`wardrobe` (Armario / Placard)** | — | **87.5%** | 🟢 *Nueva clase aprendida* |
| **`chair` (Sillas)** | — | **85.1%** | 🟢 *Nueva clase aprendida* |
| **`double_door` (Puerta doble)** | 41.0% | **81.9%** | 🟢 **+40.9%** |
| **`sofa` (Sillones / Sofás)** | — | **81.6%** | 🟢 *Nueva clase aprendida* |
| **`bed` (Camas)** | — | **80.9%** | 🟢 *Nueva clase aprendida* |
| **`table` (Mesas)** | — | **80.2%** | 🟢 *Nueva clase aprendida* |
| **`wall` (Muros / Paredes)** | 57.0% | **53.1%** | 🟡 *Estable* |
| **`stair` (Escaleras)** | — | **46.7%** | 🟢 *Nueva clase aprendida* |
| **Precisión General (Precision)** | 59.4% | **77.6%** | 🟢 **+18.2%** |
| **mAP50 General** | 20.5% | **46.6%** | 🟢 **+26.1%** |
| **mAP50-95 General** | 15.6% | **39.2%** | 🟢 **+23.6%** |

---

## 3. 🎯 Métricas Detalladas por Clase (Validación Final)

```text
Class             Images   Instances   Precision (P)   Recall (R)   mAP50    mAP50-95
----------------------------------------------------------------------------------
all                   49         424           0.776        0.411   0.466       0.392
sliding_door          17          29           0.860        0.897   0.922       0.816
wardrobe               8          23           0.734        0.842   0.875       0.700
chair                  6          73           0.867        0.890   0.851       0.730
double_door           24          82           0.870        0.734   0.819       0.688
sofa                   4          12           1.000        0.617   0.816       0.743
bed                    6          15           0.625        0.779   0.809       0.793
table                  8          53           0.940        0.736   0.802       0.730
wall                  42          42           0.569        0.524   0.531       0.302
stair                 20          55           0.658        0.454   0.467       0.301
single_door            7           7           1.000        0.000   0.186       0.168
toilet                 2           9           0.290        0.111   0.145       0.127
sink                   5           5           1.000        0.000   0.119       0.107
window                 4          13           1.000        0.000   0.053       0.022
```

---

## 4. 🧪 Pruebas de Inferencia en Muestras Reales

Se ejecutó inferencia con el script de prueba sobre planos no vistos:

* **Muestra `0000-0009.png`**:
  * `sliding_door`: **1.00** (100% de certeza)
  * `sliding_door`: **0.99** (99% de certeza)
  * `wall`: **0.71** (71% de certeza)
* **Muestra `0000-0044.png`**:
  * `sliding_door`: **0.99** (99% de certeza)
  * `sliding_door`: **0.99** (99% de certeza)
  * `wall`: **0.61** (61% de certeza)

**Velocidad de Inferencia:** ~120 - 138 ms por plano en CPU (~7-8 imágenes por segundo).

---

## 5. 📁 Ubicación de Archivos Clave

* **Pesos entrenados optimizados:** [`YOLO/best_floorplancad.pt`](file:///d:/unity%20project/backend-Software2/YOLO/best_floorplancad.pt)
* **Dataset generado:** [`YOLO/floorplancad/`](file:///d:/unity%20project/backend-Software2/YOLO/floorplancad/)
  * `data.yaml`: Definición de clases y rutas.
  * `images/train/` y `images/val/`: Imágenes procesadas.
  * `labels/train/` y `labels/val/`: Etiquetas YOLO.
* **Resultados y gráficas completas:** [`runs/detect/runs/cubicasa/floorplancad_200img_20ep/`](file:///d:/unity%20project/backend-Software2/runs/detect/runs/cubicasa/floorplancad_200img_20ep/)

---

## 6. 🚀 Próximos Pasos Sugeridos

1. **Integración al Backend / API:**
   * Apuntar `app/core/config.py` o los servicios de detección para usar `YOLO/best_floorplancad.pt`.
2. **Pipeline Completo:**
   * Combinar detección de elementos (puertas, paredes, mobiliario) con la extracción de cotas numéricas (OCR / PaddleOCR / EasyOCR).
3. **Exportación a Unity:**
   * Enviar los objetos detectados (clase, bounding box, medidas asociadas) como JSON hacia Unity para la reconstrucción 3D del plano.
