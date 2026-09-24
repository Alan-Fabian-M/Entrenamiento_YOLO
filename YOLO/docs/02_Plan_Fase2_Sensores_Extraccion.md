# Plan Técnico — Fase 2: Los Sensores de Extracción (El Núcleo de IA)

> ⚠️ **Documento histórico — plan original, no lo que se terminó construyendo.** Varias decisiones cambiaron durante la implementación (otra librería de OCR, otro algoritmo para las paredes, un solo endpoint en vez de dos, modelo YOLO genérico en vez de OBB propio). Ver [`03_Implementacion_Real_Fase2.md`](03_Implementacion_Real_Fase2.md) para lo que realmente hay en `backend/app/` hoy, con la tabla de diferencias. Se deja este documento sin borrar como referencia de diseño.

Este documento detalla el diseño y la arquitectura de la **Fase 2**: la separación del análisis visual en tres canales de percepción desacoplados y especializados.

---

## 🧭 Arquitectura de los 3 Canales Paralelos

Cada sensor recibe la variante de imagen del preprocesador que mejor se adapta a su tarea:

```
                              [ Imagen de Entrada ]
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
                 [ shadow_free ]               [ contrast_enhanced ]       [ binary ]
                         │                             │                       │
                         ▼                             ▼                       ▼
            ┌─────────────────────────┐   ┌─────────────────────────┐   ┌─────────────────────────┐
            │ Canal 1: YOLOv8-OBB     │   │ Canal 2: OCR            │   │ Canal 3: OpenCV         │
            │ (Mobiliario / Orient.)  │   │ (Cotas y Medidas)       │   │ (Estructura de Muros)   │
            └─────────────────────────┘   └─────────────────────────┘   └─────────────────────────┘
                         │                             │                       │
                         ▼                             ▼                       ▼
            Objetos con orientación       Cadenas numéricas y cotas     Esqueleto y segmentos
            (x, y, w, h, ángulo θ)        "4.20m", "3.00m" con cajas    (x1, y1) a (x2, y2)
                         │                             │                       │
                         └─────────────────────────────┼───────────────────────┘
                                                       ▼
                                   [ Payload de Percepción Unificada ]
```

---

## 🛠️ Detalle de Implementación por Módulo

### 1. Paso 3: Módulo YOLO-OBB (Objetos y Orientación)
* **Archivo de servicio:** `backend/app/services/yolo_detector.py`
* **Tecnología:** Librería `ultralytics` (`yolov8n-obb.pt` / modelo de Oriented Bounding Boxes).
* **Entrada:** Imagen `shadow_free` (normalizada sin sombras, preservando degradados de color y texturas de muebles).
* **Salida estructurada:**
  ```json
  {
    "class_name": "sofa",
    "confidence": 0.94,
    "center_px": [150.5, 210.0],
    "size_px": [85.0, 42.0],
    "angle_deg": 45.0,
    "obb_polygon": [[120, 195], [180, 225], [170, 245], [110, 215]]
  }
  ```

### 2. Paso 4: Módulo OCR (Cotas Métricas y Texto)
* **Archivo de servicio:** `backend/app/services/ocr_reader.py`
* **Tecnología:** `EasyOCR` (soporte multilingüe, ligero y preciso para números).
* **Entrada:** Imagen `contrast_enhanced` (filtrada con CLAHE para que las letras y dígitos pequeños resalten contra el fondo).
* **Filtro Regex:** Extractor determinista de medidas métricas (`\d+(?:[.,]\d+)?\s*(?:m|cm|mts)?`).
* **Salida estructurada:**
  ```json
  {
    "raw_text": "4.20m",
    "value_meters": 4.20,
    "confidence": 0.89,
    "bbox_px": [[310, 45], [365, 45], [365, 62], [310, 62]],
    "center_px": [337.5, 53.5]
  }
  ```

### 3. Paso 5: Módulo OpenCV (Líneas y Esqueleto Estructural)
* **Archivo de servicio:** `backend/app/services/wall_extractor.py`
* **Tecnología:** Algoritmos matemáticos de OpenCV:
  1. *Skeletonization / Thinning morfológico:* Reduce muros anchos a líneas de 1 píxel de grosor.
  2. *Transformada de Hough Probabilística (`cv2.HoughLinesP`):* Detecta segmentos rectos.
  3. *Filtro de colinealidad y unión:* Conecta segmentos colineales cercanos que hayan quedado interrumpidos por puertas o ventanas.
* **Entrada:** Imagen `binary` (binarizada nítida).
* **Salida estructurada:**
  ```json
  {
    "wall_id": "wall_1",
    "start_px": [30, 30],
    "end_px": [370, 30],
    "length_px": 340.0,
    "orientation": "horizontal"
  }
  ```

---

## 📡 Endpoints Propuestos para la Fase 2

1. **`POST /api/v1/extract`**:
   Ejecuta los tres canales y retorna el JSON consolidado de percepción:
   ```json
   {
     "status": "success",
     "furniture": [...],
     "dimensions": [...],
     "walls": [...]
   }
   ```

2. **`POST /api/v1/extract/visualize`**:
   Retorna una imagen PNG con las detecciones dibujadas para validación humana rápida:
   - Muebles y su rotación en **verde**.
   - Muros estructurales en **rojo**.
   - Cotas de texto en **azul**.

---

## 📦 Dependencias Adicionales Requeridas

* `ultralytics>=8.3.0` (soporte OBB)
* `torch` y `torchvision` (backend tensor/red neuronal)
* `easyocr>=1.7.0` (reconocimiento OCR local)
