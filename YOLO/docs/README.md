# Índice de Documentación — Backend de Visión Artificial (InmobiliariaVR)

**Última actualización:** 2026-09-16

Toda la documentación técnica del backend (`backend/`) vive acá. El `README.md` en la raíz de `backend/` tiene las instrucciones rápidas de instalación/arranque; estos documentos son el detalle técnico, la evidencia y el historial de decisiones.

1. [**00_Flujo_General_Vision_IA.md**](00_Flujo_General_Vision_IA.md):
   Visión global del sistema, arquitectura desacoplada y fases del proyecto (Ingestión → Extracción → Motor Geométrico → Ensamblaje en Unity). Ya actualizado para reflejar la arquitectura real, no solo el plan original.

2. [**01_Evidencia_Fase1_Preprocesamiento.md**](01_Evidencia_Fase1_Preprocesamiento.md):
   Resultados, benchmarks (~10 ms), evidencia de pruebas unitarias (11/11 tests) y respuestas cURL del microservicio de ingestión y preprocesamiento con OpenCV y FastAPI. Sigue vigente, sin cambios.

3. [**02_Plan_Fase2_Sensores_Extraccion.md**](02_Plan_Fase2_Sensores_Extraccion.md):
   Plan técnico ORIGINAL de los 3 canales de percepción de la Fase 2 (YOLO-OBB, EasyOCR, OpenCV Hough Lines). **Histórico** — varias decisiones cambiaron al implementarlo, ver el documento 03.

4. [**03_Implementacion_Real_Fase2.md**](03_Implementacion_Real_Fase2.md):
   **El más importante para entender cómo funciona hoy la detección de muebles.** Describe el flujo real paso a paso (qué hace cada archivo de `app/services/`), la tabla de diferencias contra el plan original, y una advertencia de dominio importante: el modelo de YOLO usado hoy es genérico (COCO, fotos reales), no fue entrenado para reconocer símbolos de plano — la detección puede fallar bastante en un croquis dibujado a mano hasta entrenar un modelo propio.

5. [**04_Estado_Actual_y_Pendientes.md**](04_Estado_Actual_y_Pendientes.md):
   Estado real verificado del entorno (`.venv`), qué falta instalar antes de que el servidor arranque, y el checklist de pendientes priorizado.
