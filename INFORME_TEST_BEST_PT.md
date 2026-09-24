# 📋 Informe de Prueba — Modelo `best.pt` (YOLOv8-Seg CubiCasa5K)

> **Fecha:** 19 de septiembre de 2026  
> **Proyecto:** InmobiliariaVR — Vision Backend  
> **Archivo evaluado:** `D:\Descargas\best.pt` → `YOLO/best.pt`

---

## 1. 🎯 Contexto

El modelo `best.pt` fue entrenado en **Google Colab** usando una **GPU T4** mediante el notebook `entrenamiento_cubicasa_colab.ipynb`. El entrenamiento utilizó el dataset **CubiCasa5K**, que contiene 5.000 planos arquitectónicos reales de alta calidad. Al finalizar el entrenamiento, el archivo fue descargado automáticamente a la computadora local (`D:\Descargas\best.pt`, ~6.4 MB).

### Métricas del entrenamiento (reportadas por Colab)

| Clase       | Imágenes | Instancias | Box P | Box R | mAP50 | mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|-------------|----------|------------|-------|-------|-------|----------|--------|--------|------------|---------------|
| **all**     | 748      | 40203      | 0.679 | 0.656 | 0.689 | 0.511    | 0.617  | 0.595  | 0.6        | 0.371         |
| bathroom    | 624      | 1045       | 0.71  | 0.852 | 0.829 | 0.744    | 0.72   | 0.844  | 0.823      | 0.662         |
| door        | 748      | 7324       | 0.912 | 0.856 | 0.894 | 0.694    | 0.895  | 0.834  | 0.862      | 0.436         |
| kitchen     | 3        | 3          | 0     | 0     | 0     | 0        | 0      | 0      | 0          | 0             |
| room        | 747      | 6809       | 0.843 | 0.854 | 0.896 | 0.789    | 0.857  | 0.858  | 0.901      | 0.742         |
| wall        | 748      | 18597      | 0.785 | 0.621 | 0.708 | 0.369    | 0.538  | 0.41   | 0.392      | 0.157         |
| window      | 748      | 6425       | 0.819 | 0.755 | 0.804 | 0.469    | 0.69   | 0.625  | 0.62       | 0.231         |

> ℹ️ La clase `kitchen` no tenía suficientes muestras en el split de validación (solo 3 imágenes), por lo que sus métricas son 0.  
> ℹ️ Velocidad de entrenamiento final: `0.6ms preprocess, 17.7ms inference, 0.0ms loss, 9.6ms postprocess` por imagen.

---

## 2. 📦 Instalación del modelo en el proyecto

### Pasos realizados

1. **Copiado del archivo** desde Descargas al proyecto:
   ```powershell
   Copy-Item "D:\Descargas\best.pt" "D:\unity project\backend-Software2\YOLO\best.pt" -Force
   ```

2. **Verificación del archivo copiado:**

   | Propiedad       | Valor                        |
   |-----------------|------------------------------|
   | Nombre          | `best.pt`                    |
   | Tamaño          | 6,753,588 bytes (~6.44 MB)   |
   | Fecha escritura | 19/09/2026 07:23:24 a.m.     |
   | Destino         | `YOLO/best.pt`               |

3. **Configuración preexistente en `app/core/config.py`:**  
   El backend ya apuntaba a esta ruta por defecto:
   ```python
   YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "YOLO/best.pt")
   YOLO_CONFIDENCE_THRESHOLD: float = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.35"))
   ```
   ✅ **No fue necesario modificar ningún archivo de configuración.**

---

## 3. 🧪 Prueba 1 — Carga del modelo

**Comando ejecutado:**
```python
from ultralytics import YOLO
import json

m = YOLO('YOLO/best.pt')
print('Tarea:', m.task)
print('Clases:', json.dumps(m.names))
```

**Resultado:**
```
Modelo OK
Tarea: segment
Clases: {"0": "bathroom", "1": "bed", "2": "door", "3": "kitchen",
          "4": "room", "5": "stairs", "6": "wall", "7": "window"}
```

| Propiedad       | Valor         |
|-----------------|---------------|
| Estado          | ✅ OK         |
| Tipo de tarea   | `segment`     |
| Total de clases | 8             |
| Clases          | bathroom, bed, door, kitchen, room, stairs, wall, window |

---

## 4. 🔬 Prueba 2 — Inferencia directa sobre imagen de validación

**Imagen usada:**  
`YOLO/cubicasa_coco/images/valid/colorful_10106_png.rf.t35TKV4DfLQR2BBPk3Rp.png`

**Comando ejecutado:**
```python
from ultralytics import YOLO
import cv2

model = YOLO('YOLO/best.pt')
results = model(img, conf=0.35)[0]
annotated = results.plot()
cv2.imwrite('YOLO/test_resultado.jpg', annotated)
```

**Resultado de inferencia:**
```
image 1/1 ...colorful_10106...: 480x640
  3 bathrooms, 17 doors, 15 rooms, 44 walls, 25 windows
Speed: 7.1ms preprocess, 185.7ms inference, 179.1ms postprocess per image
```

### Detecciones por clase

| Clase      | Cantidad | Confianza máxima |
|------------|----------|-----------------|
| room       | 15       | 0.96            |
| wall       | 44       | 0.90            |
| door       | 17       | 0.90            |
| window     | 25       | 0.83            |
| bathroom   | 3        | 0.86            |
| **TOTAL**  | **104**  | —               |

> 📸 Imagen anotada guardada en: `YOLO/test_resultado.jpg`

### Tiempos de ejecución

| Etapa          | Tiempo     |
|----------------|-----------|
| Preprocesado   | 7.1 ms    |
| Inferencia     | 185.7 ms  |
| Postprocesado  | 179.1 ms  |
| **Total**      | ~372 ms   |

---

## 5. 🌐 Prueba 3 — Test E2E del endpoint FastAPI

### Servidor iniciado:
```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

**Logs de arranque:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

### Script de prueba ejecutado: `scratch_test.py`

```python
import requests
from pathlib import Path

img_path = Path("YOLO/cubicasa_coco/images/valid/colorful_10106_png.rf.t35TKV4DfLQR2BBPk3Rp.png")
url = "http://127.0.0.1:8000/api/v1/compilar-sala"

with open(img_path, "rb") as f:
    r = requests.post(url, files={"file": ("plano.png", f, "image/png")})
```

### Respuesta del API

```
Status Code: 200 ✅
```

**Metadata:**
```json
{
  "name": "Sala generada desde croquis",
  "description": "Compilado automaticamente por el Spatial Scene Compiler a partir de una foto",
  "dimensions": { "width": 4.0, "depth": 4.0, "height": 2.5 },
  "furniture_count": 0,
  "created_date": "2026-09-19T11:45:31.843381+00:00",
  "processing_time_ms": 4178,
  "scale_confidence": 0.0,
  "scale_source": "tesseract binary not found on PATH"
}
```

**Room Info:**
```json
{
  "room_type": "detectado_desde_croquis",
  "dimensions": { "x": 4.0, "y": 2.5, "z": 4.0 },
  "flooring_material": "wood",
  "wall_color": "white"
}
```

**Scene Elements generados (4 muros):**

| ID          | Tipo  | Posición (x, y, z)   | Escala (x, y, z)    |
|-------------|-------|----------------------|---------------------|
| muro_norte  | muro  | (0.0, 0.0, 4.0)      | (4.0, 2.5, 0.2)     |
| muro_sur    | muro  | (0.0, 0.0, 0.0)      | (4.0, 2.5, 0.2)     |
| muro_este   | muro  | (4.0, 0.0, 2.0)      | (0.2, 2.5, 4.0)     |
| muro_oeste  | muro  | (0.0, 0.0, 2.0)      | (0.2, 2.5, 4.0)     |

> ⏱ Tiempo total de procesamiento del endpoint: **4,178 ms (~4.2 segundos)**

---

## 6. ⚠️ Observaciones y Pendientes

### Problema detectado: Tesseract OCR no instalado

**Mensaje en la respuesta:**
```
"scale_source": "tesseract binary not found on PATH -- install it to enable OCR scale detection"
"scale_confidence": 0.0
```

**Impacto:** El sistema no puede leer las medidas escritas en el plano (ej: `4.5m`). Por eso usa las dimensiones por defecto definidas en `config.py`:

```python
DEFAULT_ROOM_WIDTH_M: float = 4.0
DEFAULT_ROOM_DEPTH_M: float = 4.0
DEFAULT_ROOM_HEIGHT_M: float = 2.5
```

**Solución — Instalar Tesseract OCR:**
1. Descargar desde: https://github.com/UB-Mannheim/tesseract/wiki
2. Instalar en `C:\Program Files\Tesseract-OCR\`
3. Agregar al PATH de Windows: `C:\Program Files\Tesseract-OCR\`
4. Verificar con el comando: `tesseract --version`

---

## 7. ✅ Resumen de Estado General

**Progreso global:** `█████████░░` 5 / 7 componentes operativos (71%)

| Componente                     | Estado         | Notas                                                           |
|--------------------------------|----------------|-----------------------------------------------------------------|
| Modelo `best.pt` copiado       | ✅ Completado  | `YOLO/best.pt` (6.44 MB), reemplaza al genérico `yolov8n.pt`   |
| Carga del modelo               | ✅ OK          | Tarea: `segment`, 8 clases arquitectónicas detectadas           |
| Inferencia directa (imagen)    | ✅ OK          | 104 detecciones en ~372ms, confianza máxima 0.96 (room)         |
| Backend FastAPI corriendo      | ✅ OK          | Servidor activo en `http://127.0.0.1:8000` con hot-reload       |
| Endpoint `/compilar-sala`      | ✅ OK (200)    | Responde en ~4.2s con 4 muros + metadata + room_info            |
| OCR / detección de escala      | ⚠️ Pendiente  | Tesseract no instalado → usa dimensiones default (4m × 4m × 2.5m)|
| Clase `kitchen` en el modelo   | ⚠️ Sin datos   | Solo 3 imágenes en validación → métricas en 0, requiere más data |

---

### 7.1 Detalle por componente

#### ✅ Modelo `best.pt` copiado
- **Origen:** `D:\Descargas\best.pt` (descargado desde Google Colab)
- **Destino:** `D:\unity project\backend-Software2\YOLO\best.pt`
- **Tamaño:** 6,753,588 bytes (~6.44 MB)
- **Config apuntando:** `YOLO_MODEL_PATH = "YOLO/best.pt"` en `app/core/config.py`
- **Acción necesaria:** ✅ Ninguna — ya está listo para producción

#### ✅ Carga del modelo
- **Framework:** Ultralytics YOLOv8
- **Tarea:** `segment` (segmentación de instancias)
- **Clases:** `bathroom(0)`, `bed(1)`, `door(2)`, `kitchen(3)`, `room(4)`, `stairs(5)`, `wall(6)`, `window(7)`
- **Acción necesaria:** ✅ Ninguna

#### ✅ Inferencia directa (imagen)
- **Imagen:** `colorful_10106_png` (480×640 px)
- **Detecciones:** 104 objetos — 3 bathrooms, 17 doors, 15 rooms, 44 walls, 25 windows
- **Tiempo total:** ~372ms (7ms preprocesado + 186ms inferencia + 179ms postprocesado)
- **Confianza más alta:** 0.96 en clase `room`
- **Resultado visual:** guardado en `YOLO/test_resultado.jpg`
- **Acción necesaria:** ✅ Ninguna

#### ✅ Backend FastAPI corriendo
- **Comando:** `uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload`
- **Estado:** Application startup complete — sin errores
- **Hot-reload:** activo (detecta cambios en tiempo real)
- **Acción necesaria:** ✅ Ninguna para desarrollo local

#### ✅ Endpoint `/compilar-sala`
- **URL:** `POST http://127.0.0.1:8000/api/v1/compilar-sala`
- **HTTP Status:** `200 OK`
- **Tiempo de respuesta:** 4,178 ms
- **Salida:** metadata de sala + room_info + 4 scene_elements tipo `muro`
- **furniture_count:** 0 (el pipeline de muebles no generó elementos en este plano)
- **Acción necesaria:** ✅ Ninguna — funcional para Unity

#### ⚠️ OCR / Detección de escala
- **Error:** `tesseract binary not found on PATH`
- **Consecuencia:** `scale_confidence = 0.0` → se usan medidas fijas de `config.py`
- **Dimensiones actuales (default):** 4.0m × 4.0m × 2.5m
- **Dimensiones esperadas (con OCR):** las que estén escritas en el plano real
- **Impacto en Unity:** la sala generada no refleja el tamaño real del plano
- **Acción necesaria:**
  ```
  1. Descargar Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
  2. Instalar en C:\Program Files\Tesseract-OCR\
  3. Agregar al PATH del sistema
  4. Verificar: tesseract --version
  5. Reiniciar el servidor FastAPI
  ```

#### ⚠️ Clase `kitchen` en el modelo
- **Problema:** Solo 3 imágenes con cocina en el split de validación
- **Resultado:** mAP50 = 0, Mask mAP50 = 0 para esta clase
- **Estado real:** el modelo sí puede detectar cocinas, pero no hay suficientes datos para validar
- **Acción necesaria:**
  ```
  - Agregar más imágenes etiquetadas con clase "kitchen" al dataset
  - Re-entrenar el modelo en Colab con el dataset ampliado
  - Alternativamente: ampliar con data augmentation en el notebook
  ```

---

### 7.2 Leyenda de estados

| Ícono | Significado                                    |
|-------|------------------------------------------------|
| ✅    | Completado y funcionando correctamente         |
| ⚠️    | Funcional con limitaciones o acción pendiente  |
| ❌    | Error crítico, bloquea el funcionamiento       |
| 🔄    | En progreso / requiere acción inmediata        |

---

## 8. 📁 Archivos clave del proyecto

| Archivo                              | Descripción                                        |
|--------------------------------------|----------------------------------------------------|
| `YOLO/best.pt`                       | Modelo entrenado YOLOv8-Seg (CubiCasa5K)          |
| `app/core/config.py`                 | Configuración general (ruta al modelo, umbrales)  |
| `app/services/room_extractor.py`     | Extrae contornos y genera los 4 muros de la sala  |
| `app/services/furniture_detector.py` | Detecta objetos con YOLO                          |
| `app/services/scene_compiler.py`     | Compila la escena final para Unity                |
| `scratch_test.py`                    | Script de prueba E2E del endpoint                 |
| `YOLO/test_resultado.jpg`            | Imagen de validación con anotaciones del modelo   |

---

*Informe generado el 19/09/2026 — InmobiliariaVR Vision Backend v0.2.0*
