# Estado Actual del Backend y Pendientes

**Última actualización:** 2026-09-16

Este documento resume, endpoint por endpoint, qué está terminado, qué está a medias y qué falta — y lo más importante, **qué hace falta hacer para que el servidor arranque y sirva de verdad en esta máquina**, porque se verificó el entorno real (`.venv/`) y no coincide con lo que dicen los `requirements.txt`.

---

## ⚠️ Antes de correr el servidor — verificación real del entorno (2026-09-16)

Se revisó `backend/.venv/lib/python3.13/site-packages/` directamente (no una suposición) y **el servidor todavía no puede arrancar tal cual está**: `furniture_detector.py` y `scale_detector.py` importan `ultralytics` y `pytesseract` al nivel de módulo, pero ninguno de los dos paquetes está instalado en ese entorno virtual — solo están `fastapi`, `opencv-python-headless`, `numpy`, `pillow`, `shapely`, `pydantic` (la base de la Fase 1). El endpoint `/api/v1/compilar-sala` fue probado end-to-end únicamente en un entorno aislado ajeno a esta máquina, con el modelo de YOLO reemplazado por una versión simulada — nunca se corrió con el modelo real acá.

**Orden de pasos antes de probar `/api/v1/compilar-sala`:**

1. ```bash
   cd backend
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   (la primera vez que se use el endpoint, `ultralytics` va a intentar bajar los pesos de `yolov8n.pt` — necesita internet esa única vez).
2. Instalar el binario de Tesseract a nivel de sistema operativo (no alcanza con el paquete de Python `pytesseract`):
   ```bash
   sudo dnf install tesseract   # Fedora/RHEL
   ```
   No se pudo confirmar desde esta sesión si ya está instalado en esta máquina.
3. Levantar el servidor y probarlo con una foto real:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   Documentación interactiva en `http://localhost:8000/docs` (Swagger UI) para probar `/api/v1/compilar-sala` subiendo una imagen directamente desde el navegador, sin necesitar Unity ni el visor para esta prueba.

Tampoco hay un archivo de test para este endpoint en `backend/tests/` (solo existen `test_preprocess.py` y `test_cubicasa_conversion.py`, de antes de esta fase) — sería bueno agregar uno una vez que el entorno esté instalado.

---

## Endpoints — estado

| Endpoint | Estado | Notas |
|---|---|---|
| `GET /api/v1/health` | ✅ Funciona | Simple, sin dependencias nuevas. |
| `POST /api/v1/preprocess` | ✅ Funciona, probado (11 tests, ver `01_Evidencia_Fase1_Preprocesamiento.md`) | Fase 1, sin cambios recientes. |
| `POST /api/v1/preprocess/pipeline` | ✅ Funciona, probado | Igual que el anterior. |
| `POST /api/v1/compilar-sala` | 🟡 Código completo, **sin correr en esta máquina** | Ver la sección de arriba — bloqueado por dependencias faltantes en el `.venv`. |

## Detección de muebles — resumen (detalle completo en `03_Implementacion_Real_Fase2.md`)

Usa el modelo genérico `yolov8n.pt` (COCO, fotos reales) en vez de un modelo entrenado para símbolos de plano — es más una prueba de que "el cableado funciona" que un detector confiable para el caso de uso real todavía. El ángulo de los muebles siempre da 0° con este modelo. El camino para arreglar esto (entrenar un modelo YOLOv8-OBB propio con CubiCasa5K) ya está armado en `backend/training/cubicasa/`, pero el dataset nunca se descargó ni se entrenó nada.

## Checklist de pendientes, priorizado

1. **Bloqueante:** `pip install -r requirements.txt` en el `.venv` real + instalar Tesseract a nivel de SO (ver arriba).
2. Probar `/api/v1/compilar-sala` con una foto real del tipo de croquis que se va a usar en la demo, y ver qué tan bien (o mal) detecta muebles con el modelo genérico actual.
3. Si la detección es muy mala (lo esperable con COCO sobre un plano dibujado): decidir si conviene invertir tiempo en descargar CubiCasa5K y entrenar el modelo OBB propio, o si alcanza con que el usuario ajuste manualmente los muebles en VR después de generarlos (ya se puede).
4. Agregar un test automatizado real para `/api/v1/compilar-sala` en `backend/tests/`.
5. Confirmar conectividad real Quest↔PC en la misma red WiFi con la IP correcta configurada en Unity (`CroquisSceneCompilerController.servidorIP`) — no se puede verificar sin ambos dispositivos.

## Qué NO se tocó (fuera de alcance por ahora)

- Captura de foto en vivo con la cámara del Quest 3 (Passthrough Camera API de Meta) — ver `InmobiliariaVR/Docs/Plan_Funcionalidades_Visor.md`.
- Entrenamiento del modelo YOLOv8-OBB con CubiCasa5K (pipeline listo, entrenamiento pendiente).
