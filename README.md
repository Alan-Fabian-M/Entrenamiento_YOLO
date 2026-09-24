# InmobiliariaVR — Backend de Visión Artificial

Microservicio en Python (FastAPI + OpenCV + YOLO + OCR) para la digitalización, preprocesamiento y compilación de planos arquitectónicos 2D (escaneos o fotografías tomadas con celular) en un Scene Graph JSON que Unity ensambla en 3D.

Forma parte del pipeline del **Primer Parcial (ISW2 - Grupo 1)**:
```
[Foto de Plano en Papel] 
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ FASE 1: Preprocesamiento (FastAPI + OpenCV)             │
│ • Ingestión vía API REST multipart/form-data           │
│ • Eliminación de sombras (Background Division)         │
│ • Realce de contraste adaptativo (CLAHE)               │
│ • Binarización adaptable (Adaptive Gaussian / Otsu)    │
└────────────────────────────────────────────────────────┘
          │
          ├─────────────────────────┬────────────────────────┐
          ▼                         ▼                        ▼
[YOLO: Muebles/Orient.]     [OpenCV: Paredes]        [OCR: Cotas y Metros]
          │                         │                        │
          └─────────────────────────┼────────────────────────┘
                                    ▼
                         [Scene Graph JSON — POST /api/v1/compilar-sala]
                                    ▼
                         [Ensamblaje 3D en Unity VR/AR — SceneGenerator.cs]
```

> **Nota (2026-09-16):** este servicio reemplaza al prototipo standalone que vivía en
> `InmobiliariaVR/SpatialSceneCompiler/` (que usaba un schema JSON distinto e incompatible,
> ver su README). La lógica de detección/OCR de ese prototipo (ya validada con una foto real)
> fue portada acá; el schema de salida ahora es el que consume `SceneGenerator.cs`
> (documentado en el proyecto de Claude como `json-schema-contrato-escenas.md`).

---

## 🚀 Requisitos Previos

* Python 3.10 o superior (probado con Python 3.13)
* Sistema Linux / Windows / macOS
* El binario de **Tesseract OCR** instalado a nivel de sistema (no alcanza con el paquete `pytesseract` de Python):
  * Fedora/RHEL: `sudo dnf install tesseract`
  * Debian/Ubuntu: `sudo apt install tesseract-ocr`
  * macOS: `brew install tesseract`

---

## 📦 Instalación y Puesta en Marcha

### 1. Activar el entorno virtual

```bash
cd backend
source .venv/bin/activate
```

*(Si necesitas instalar dependencias en otro entorno: `pip install -r requirements.txt`. La primera vez que corras `/api/v1/compilar-sala`, `ultralytics` va a descargar automáticamente los pesos de `yolov8n.pt` — necesita internet esa única vez.)*

### 2. Iniciar el servidor de desarrollo

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

El servidor quedará escuchando en:
* **API base:** `http://localhost:8000`
* **Documentación interactiva Swagger UI:** `http://localhost:8000/docs`
* **Especificación ReDoc:** `http://localhost:8000/redoc`

Para probarlo desde el Quest 3 (o cualquier dispositivo en la misma red WiFi), reemplazá `localhost` por la IP local de esta PC (`hostname -I` en Linux) en `CroquisSceneCompilerController.cs` (campo `servidorIP` en el Inspector, o el InputField en runtime).

---

## 📡 Endpoints de la API

### 1. Chequeo de Salud (`Health Check`)
```http
GET /api/v1/health
```

### 2. Preprocesamiento de Plano (`Retorno PNG Directo`)
```http
POST /api/v1/preprocess?filter_mode=binary
```
* **Parámetros Query:**
  * `filter_mode`:
    * `binary` *(por defecto)*: Pipeline completo (escala de grises → eliminación de sombras → CLAHE → binarización adaptable → limpieza de ruido).
    * `shadow_free`: Escala de grises corregida con iluminación uniforme (ideal para modelos de detección como YOLO).
    * `grayscale`: Conversión estándar a escala de grises.
    * `otsu`: Binarización global por umbral Otsu.
* **Body:** `file` (form-data: JPG, PNG o WEBP).
* **Respuesta:** Archivo de imagen `image/png` procesado listo para guardar o encadenar.

### 3. Pipeline Completo con Etapas Intermedias (`JSON + Base64`)
```http
POST /api/v1/preprocess/pipeline
```
* **Body:** `file` (form-data).
* **Respuesta:** JSON con metadatos de resolución, tiempos de ejecución por filtro y cada etapa codificada en Base64 (`grayscale`, `shadow_free`, `contrast_enhanced`, `binary`).

### 4. Compilar Sala desde Croquis (`Scene Graph JSON` — el que consume Unity)
```http
POST /api/v1/compilar-sala
```
* **Body:** `file` (form-data: foto del croquis, JPG/PNG/WEBP).
* **Qué hace:** corre los 3 canales de percepción (YOLO para muebles, OpenCV para el contorno de la sala, OCR para la escala) y devuelve el Scene Graph JSON completo — mismo schema que `Assets/Resources/ScenePresets/preset_*.json`.
* **Respuesta (ejemplo abreviado):**
  ```json
  {
    "metadata": { "name": "...", "dimensions": {"width": 4.2, "depth": 5.5, "height": 2.5}, "furniture_count": 3, ... },
    "room_info": { "room_type": "detectado_desde_croquis", "dimensions": {"x": 4.2, "y": 2.5, "z": 5.5} },
    "scene_elements": [
      { "id": "muro_norte", "type": "muro", "position": {...}, "scale": {...}, "material": {"color": "#FFFFFF"} },
      { "id": "obj_000", "type": "sofa", "confidence": 0.82, "position": {"x": 1.2, "y": 0, "z": 3.1}, "rotation": {"x":0,"y":0,"z":0}, "material": {"color": "#8899AA"} }
    ]
  }
  ```
* **Limitación conocida:** la rotación de los muebles siempre viene en `0°` con el modelo `yolov8n.pt` pretrained (no es un modelo OBB). Ver `backend/training/cubicasa/` para el pipeline ya construido que convierte CubiCasa5K a formato YOLOv8-OBB — una vez entrenado ese modelo custom, `app/services/furniture_detector.py` lo detecta automáticamente (por el nombre de archivo `*-obb*` o `*obb*`) y empieza a devolver ángulos reales sin cambios de código.

---

## 🧪 Pruebas Automatizadas

Ejecutar la suite de pruebas unitarias y de integración:

```bash
cd backend
.venv/bin/pytest tests/ -v
```
