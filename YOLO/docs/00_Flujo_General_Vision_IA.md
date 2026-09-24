# InmobiliariaVR — Pipeline de Visión Artificial: De Plano 2D a Habitación 3D

**Última actualización:** 2026-09-16 — este documento ya refleja la arquitectura REAL implementada (ver `03_Implementacion_Real_Fase2.md` para el detalle paso a paso, y `04_Estado_Actual_y_Pendientes.md` para qué falta correr). Las diferencias con el plan técnico original de `02_Plan_Fase2_Sensores_Extraccion.md` están explicadas ahí.

## 🎯 Visión y Filosofía de Diseño

El objetivo central de este sistema es **transformar un plano arquitectónico 2D en papel (fotografiado con un celular o escaneado) en un entorno 3D interactivo en Realidad Virtual y Aumentada (Meta Quest / Android)**, sin depender de servidores en la nube y garantizando un ensamblaje **100% determinista, ligero y libre de alucinaciones**.

### 💡 El Principio Clave: Desacoplar la Percepción de la Geometría
Pedirle a un único modelo de lenguaje o de visión que "entienda todo el plano" (muros, medidas, orientación de muebles y escala) genera errores de escala y alucinaciones dimensionales.

En su lugar, dividimos el problema en **sensores especializados deterministas**:
1. **YOLO**: Actúa como los "ojos" para ubicar muebles. *(Hoy usa el modelo genérico `yolov8n.pt` de COCO, sin ángulo real — ver advertencia de dominio en `03_Implementacion_Real_Fase2.md` sección 3. La versión con YOLOv8-OBB entrenado a medida, con ángulo real, es el plan a futuro, sección 4 de ese mismo documento.)*
2. **OpenCV**: Actúa como el "compás y regla" para trazar la geometría de paredes (contorno + bounding rect — no el esqueleto/Hough Lines planeado originalmente).
3. **Tesseract OCR**: Actúa como el "lector" de cotas numéricas ("4.20m", "3.00m") — reemplazó a EasyOCR en la implementación real.
4. **Motor Geométrico**: Cruza matemáticamente píxeles con metros para obtener la escala real.
5. **Unity 6**: Ensambla prefabs locales preoptimizados usando un manual ligero en JSON (`Scene Graph`), como si fueran piezas de Lego.

---

## 🗺️ Mapa de Fases del Pipeline

```
[ Fotografía de Plano en Papel (Móvil / Escaneo) ]
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│ FASE 1: Ingestión y Preprocesamiento (✅ COMPLETADA)   │
│ • Servidor local FastAPI (localhost:8000)              │
│ • Normalización de iluminación y división de fondo     │
│ • Eliminación de sombras de manos y papel arrugado     │
│ • Realce de contraste adaptativo (CLAHE)               │
│ • Binarización adaptable (Adaptive Gaussian / Otsu)    │
└────────────────────────────────────────────────────────┘
                         │
                         ├─────────────────────────┬────────────────────────┐
                         ▼                         ▼                        ▼
      ┌─────────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐
      │ FASE 2A: furniture_detector │ │ FASE 2B: scale_detector │ │ FASE 2C: room_extractor │
      │ • YOLOv8n (COCO genérico)   │ │ • Tesseract OCR         │ │ • Contorno más grande + │
      │ • 10 clases COCO -> muebles │ │ • Detecta texto/cotas   │ │   bounding rect         │
      │ • Ángulo: 0° (sin modelo    │ │ • Textos: "4.20m", "3m" │ │ • Genera las 4 paredes  │
      │   OBB propio todavía)       │ │                         │ │   (muro_norte/sur/...)  │
      └─────────────────────────────┘ └─────────────────────────┘ └─────────────────────────┘
                         │                         │                        │
                         └─────────────────────────┼────────────────────────┘
                                                   ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ FASE 3: Motor Geométrico y Generación de Scene Graph  (scene_compiler.py)               │
│ • Relación matemática: Longitud en px de muro / Cota OCR = Escala (px/m)               │
│ • Conversión de todas las coordenadas de píxeles a metros métricos estandarizados      │
│ • Serialización en Scene Graph JSON paramétrico ultraligero (models/scene_graph.py)    │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                                   │
                                                   ▼  (un solo POST /api/v1/compilar-sala)
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ FASE 4: Ensamblaje Modular Determinista en Unity (VR / AR)                             │
│ • Visor Meta Quest / App Android recibe el JSON por red local (misma WiFi)             │
│ • Cero descarga de modelos pesados (.obj/.gltf) por Wi-Fi                              │
│ • Instanciación directa de prefabs (PrefabDatabase) en coordenadas métricas,           │
│   vía SceneGenerator.GenerateSceneAsync                                                │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Estructura de Documentación en esta Carpeta

* [`01_Evidencia_Fase1_Preprocesamiento.md`](01_Evidencia_Fase1_Preprocesamiento.md): Reporte técnico, pruebas automatizadas, métricas de rendimiento y verificación del servidor de preprocesamiento.
* [`02_Plan_Fase2_Sensores_Extraccion.md`](02_Plan_Fase2_Sensores_Extraccion.md): Plan técnico ORIGINAL de los tres canales de extracción (histórico — varias decisiones cambiaron al implementarlo, ver el doc 03).
* [`03_Implementacion_Real_Fase2.md`](03_Implementacion_Real_Fase2.md): **Lo que realmente se construyó** — el flujo completo de detección de muebles paso a paso, qué cambió respecto al plan, y la limitación de dominio importante a tener en cuenta (YOLO genérico vs. símbolos de plano).
* [`04_Estado_Actual_y_Pendientes.md`](04_Estado_Actual_y_Pendientes.md): Estado real del entorno (`.venv`), checklist de qué falta para correr el servidor de verdad, y pendientes priorizados.
