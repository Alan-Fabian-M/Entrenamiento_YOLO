# Estado actual (vistazo rápido)

**Última actualización:** 2026-09-26

Para el detalle técnico completo de todo lo de acá (qué se rompía, cómo se
diagnosticó, qué archivos cambiaron línea por línea) ver
[`YOLO/docs/05_Reentrenamiento_CubiCasa5K_y_Deteccion_Multicuarto.md`](../YOLO/docs/05_Reentrenamiento_CubiCasa5K_y_Deteccion_Multicuarto.md).
Esto es solo el resumen ejecutivo.

## Lo que ya funciona

- **Detección multi-cuarto**: `POST /api/v1/compilar-sala` ya detecta todos
  los cuartos reales de un plano (no solo uno de 4x4 inventado), con
  tamaño real en metros por cuarto y muebles asignados al cuarto correcto.
- **Escala OCR**: lee cotas escritas en el plano en ambas convenciones
  comunes (mm sin unidad, o metros con decimal), con una heurística de
  desambiguación (ver regla #3 en `01_Reglas_del_Proyecto.md`).
- **Pipeline de preprocesamiento** (Fase 1: sombras, contraste, binarización)
  y los endpoints de debug (`/preprocess`, `/preprocess/pipeline`) — estable
  desde hace tiempo, sin cambios recientes.

## Lo que está a medias

- **Modelo unificado entrenado e instalado** (`YOLO/best_unified_seg.pt`,
  29 clases, entrenado en Kaggle, mAP50 general 0.26). Los muebles móviles
  solo venían de FloorPlanCAD (fondo negro), así que sobre planos de fondo
  blanco no detectaban nada; se arregló con una pasada sobre la imagen
  invertida (ver sección 3 del documento técnico). En el plano CAD de
  prueba principal ya detecta sofá, silla y cama.
- **Limitaciones:** falta mesa, estufa y lavabo en ese plano, hay algún
  falso positivo, y solo se probó con 2 planos CAD propios.
- **Muros y cuartos siguen saliendo de OpenCV** (`RoomExtractor`), no del
  modelo (wall mAP 0.18).

## Próximos pasos, en orden

1. Usar las clases de habitación del modelo para inferir `room_type`.
2. Mejorar muebles en planos CAD con datos etiquetados del estilo real, si
   se consiguen más planos.
3. Confirmar con Unity los tipos nuevos (`ducha`, `chimenea`,
   `refrigerador`, `estufa`).

## Cómo verificar que nada se rompió al seguir trabajando

```bash
cd Backend-Python
.venv/bin/pytest tests/ -v
```

60 tests deberían pasar (al momento de este documento). Si alguno falla
después de un cambio, no asumir que es "flaky" — investigar primero.
