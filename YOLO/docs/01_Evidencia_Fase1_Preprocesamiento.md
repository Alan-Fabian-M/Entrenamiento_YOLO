# Evidencia Técnica — Fase 1: Ingestión y Preprocesamiento

Este documento certifica y documenta la finalización exitosa de la **Fase 1** del backend de visión artificial.

---

## 1. Objetivos Alcanzados

- [x] **Servidor REST Local:** Microservicio FastAPI montado en `backend/app/main.py`.
- [x] **Aislamiento de Dependencias:** Entorno virtual configurado en `backend/.venv` con Python 3.13, OpenCV 5.0 headless, Uvicorn, NumPy, Pillow, Pytest y Httpx.
- [x] **Pipeline de Preprocesamiento:** Algoritmo en `backend/app/services/preprocessor.py` que resuelve sombras, viñeteado y arrugas de papel mediante división de fondo (*Background Division*), realce adaptativo local (*CLAHE*) y binarización adaptable (*Adaptive Gaussian Thresholding*).
- [x] **Endpoints Funcionales:**
  - `GET /api/v1/health`: Reporte de estado y versiones.
  - `POST /api/v1/preprocess`: Retorno de imagen binaria PNG directa.
  - `POST /api/v1/preprocess/pipeline`: Retorno de metadatos, métricas de tiempo y cada etapa en Base64.
- [x] **Pruebas Automatizadas:** 11 pruebas unitarias y de integración en `backend/tests/test_preprocess.py`.

---

## 2. Métricas y Benchmarks de Rendimiento

Pruebas ejecutadas sobre un plano sintético de 400x300 px con sombras pronunciadas y muros:

| Etapa del Pipeline | Algoritmo OpenCV Utilizado | Tiempo de Ejecución |
|---|---|---|
| **Decodificación** | `cv2.imdecode` en memoria | ~0.20 ms |
| **Escala de Grises** | `cv2.cvtColor(COLOR_BGR2GRAY)` | ~0.13 ms |
| **Eliminación de Sombras** | Clausura morfológica + `cv2.divide` | **5.21 ms** |
| **Realce de Contraste** | `cv2.createCLAHE(clipLimit=2.0)` | **0.39 ms** |
| **Binarización Adaptable** | `cv2.adaptiveThreshold(GAUSSIAN_C)` + apertura morfológica | **1.51 ms** |
| **Tiempo Total Pipeline** | Ejecución de extremo a extremo | **~10.01 ms** |

> **Conclusión de rendimiento:** El preprocesamiento es de tiempo real (~100 frames/segundo teóricos), lo que garantiza que el servidor responde casi instantáneamente ante cualquier subida de imagen.

---

## 3. Registro de Pruebas Automatizadas (Pytest)

Ejecución del comando:
```bash
/home/alan/Projects/Software2/Parcial1/backend/.venv/bin/pytest tests/ -v
```

**Salida de consola:**
```text
============================= test session starts ==============================
platform linux -- Python 3.13.5, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/alan/Projects/Software2/Parcial1/backend
plugins: anyio-4.15.1
collected 11 items                                                             

tests/test_preprocess.py::test_preprocessor_decode_valid_image PASSED    [  9%]
tests/test_preprocess.py::test_preprocessor_decode_corrupt_data PASSED   [ 18%]
tests/test_preprocess.py::test_preprocessor_shadow_removal_and_clahe PASSED [ 27%]
tests/test_preprocess.py::test_preprocessor_binarization PASSED          [ 36%]
tests/test_preprocess.py::test_preprocessor_pipeline_detailed PASSED     [ 45%]
tests/test_preprocess.py::test_endpoint_health PASSED                    [ 54%]
tests/test_preprocess.py::test_endpoint_preprocess_binary_mode PASSED    [ 63%]
tests/test_preprocess.py::test_endpoint_preprocess_shadow_free_mode PASSED [ 72%]
tests/test_preprocess.py::test_endpoint_preprocess_pipeline_details PASSED [ 81%]
tests/test_preprocess.py::test_endpoint_reject_invalid_extension PASSED  [ 90%]
tests/test_preprocess.py::test_endpoint_reject_corrupt_image PASSED      [100%]

======================== 11 passed in 1.03s ========================
```

---

## 4. Evidencia de Pruebas de Integración con cURL

### A. Health Check
```bash
curl -s http://127.0.0.1:8000/api/v1/health
```
```json
{
  "status": "online",
  "service": "InmobiliariaVR - Vision Backend",
  "version": "0.1.0",
  "opencv_version": "5.0.0",
  "features": [
    "image_ingestion",
    "shadow_removal_background_division",
    "clahe_contrast_enhancement",
    "adaptive_binarization",
    "pipeline_stages_base64"
  ]
}
```

### B. Pipeline Multi-Etapa con Base64
```bash
curl -s -X POST "http://127.0.0.1:8000/api/v1/preprocess/pipeline" \
     -F "file=@sample_floorplan.png"
```
```json
{
  "status": "success",
  "filename": "sample_floorplan.png",
  "pipeline": {
    "metadata": {
      "width": 400,
      "height": 300,
      "channels": 3,
      "total_processing_time_ms": 10.01,
      "timing_breakdown_ms": {
        "grayscale": 0.13,
        "shadow_removal": 5.21,
        "contrast_enhancement": 0.39,
        "binarization": 1.51
      }
    },
    "stages_base64": {
      "grayscale": "iVBORw0KGgoAAAANS...",
      "shadow_free": "iVBORw0KGgoAAAANS...",
      "contrast_enhanced": "iVBORw0KGgoAAAANS...",
      "binary": "iVBORw0KGgoAAAANS..."
    }
  }
}
```
