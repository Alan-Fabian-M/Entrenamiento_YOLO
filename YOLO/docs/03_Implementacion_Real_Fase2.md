# Implementación Real — Fase 2 (Detección de Muebles, Escala y Muros)

**Última actualización:** 2026-09-16

> Este documento describe lo que **realmente está construido y funcionando** en `backend/app/`, en contraste con [`02_Plan_Fase2_Sensores_Extraccion.md`](02_Plan_Fase2_Sensores_Extraccion.md), que era el plan original antes de implementarlo. Varias decisiones cambiaron en el camino (motivos abajo). Si tenés que explicar o defender esta parte del proyecto, este es el documento que refleja la realidad del código.

---

## 1. Dónde vive el código

```
backend/app/
├── api/v1/endpoints.py          # POST /api/v1/compilar-sala (un solo endpoint, no dos)
├── services/
│   ├── furniture_detector.py    # Detección de muebles (YOLO)
│   ├── scale_detector.py        # Lectura de escala (OCR)
│   ├── room_extractor.py        # Contorno de la sala + paredes
│   └── scene_compiler.py        # Orquesta los tres de arriba y arma el JSON final
└── models/scene_graph.py        # Los modelos Pydantic de la respuesta (schema que consume Unity)
```

Qué cambió respecto al plan original (`02_Plan_Fase2_Sensores_Extraccion.md`):

| Plan original | Lo que se construyó de verdad | Por qué |
|---|---|---|
| `yolo_detector.py` con `yolov8n-obb.pt` (modelo OBB con ángulo real) | `furniture_detector.py` con `yolov8n.pt` genérico (COCO, sin ángulo) | No existe públicamente un modelo YOLO-OBB pre-entrenado con clases de muebles de plano. Entrenar uno propio requiere el dataset CubiCasa5K (ver sección 4) — el pipeline para eso ya existe pero el dataset nunca se descargó ni se entrenó nada. |
| `ocr_reader.py` con EasyOCR | `scale_detector.py` con **Tesseract** (`pytesseract`) | Se reusó la lógica ya validada de un prototipo anterior (`SpatialSceneCompiler/`) que usaba Tesseract, en vez de escribir un módulo nuevo con otra librería. |
| `wall_extractor.py` con esqueleto morfológico + `cv2.HoughLinesP` | `room_extractor.py` con **contorno más grande + bounding rect** | Mismo motivo: se reusó la lógica ya probada del prototipo anterior. Es más simple y sin errores conocidos, aunque asume una sala rectangular (no detecta muros interiores ni formas en L). |
| Endpoints separados `/api/v1/extract` + `/api/v1/extract/visualize` | Un solo endpoint **`POST /api/v1/compilar-sala`** | Unity necesita el Scene Graph JSON completo en una sola llamada, no un JSON de "percepción cruda" que después haya que traducir. Se simplificó a un solo paso. |

---

## 2. El flujo completo, paso a paso

Esto es lo que pasa exactamente desde que Unity manda la foto hasta que le llega el JSON de vuelta (`SceneCompiler.compile()` en `scene_compiler.py` es el orquestador; todo lo demás son funciones que llama en orden):

### Paso 1 — Llega la imagen
`POST /api/v1/compilar-sala` recibe el archivo (`file`, multipart/form-data), lo valida (extensión, tamaño máximo 15 MB) y lo decodifica a una imagen OpenCV (`cv2.imdecode`).

### Paso 2 — Se prepara una versión "limpia" para el detector
Se genera una versión en escala de grises con las sombras eliminadas (reutilizando el mismo algoritmo de `preprocessor.py` de la Fase 1: división de fondo + CLAHE), pero se guarda como imagen de 3 canales (BGR) porque YOLO espera ese formato. Esto es lo que realmente "ve" el modelo, no la foto cruda.

### Paso 3 — Detección de muebles (`FurnitureDetector.detect`)

**Esta es la parte de tu pregunta.** Así es como funciona hoy:

1. Se carga (una sola vez, cacheado en memoria) un modelo YOLO desde `settings.YOLO_MODEL_PATH`, que **por defecto es `yolov8n.pt`** — el modelo genérico y liviano de Ultralytics, entrenado sobre el dataset **COCO** (fotos reales del mundo, 80 clases: persona, auto, perro, sofá, silla, mesa, etc.).
2. Se corre la inferencia sobre la imagen preparada en el Paso 2.
3. YOLO devuelve cajas delimitadoras con una clase COCO por cada objeto que reconoce. **Solo nos importan algunas de esas 80 clases** — hay un diccionario fijo (`COCO_CLASS_TO_UNITY_TYPE`) que traduce las que sí tienen un mueble equivalente en Unity:

   | Clase COCO | Tipo Unity |
   |---|---|
   | couch | sofa |
   | chair | silla |
   | bed | cama |
   | dining table | mesa |
   | tv | tv |
   | refrigerator | refrigerador |
   | sink | lavabo |
   | toilet | inodoro |
   | oven | horno |
   | microwave | microondas |

   Cualquier otra clase que YOLO detecte (persona, perro, bicicleta, celular, etc.) **se descarta**, porque no hay un prefab de Unity para eso.
4. Para cada detección que sí importa: se recorta esa región de la imagen y se calcula el color promedio de esos píxeles (`_average_color_hex`) — así se decide el color del material del mueble en el JSON, sin que el usuario tenga que elegirlo.
5. La posición viene del centro de la caja delimitadora, todavía en píxeles (se convierte a metros más adelante, Paso 5).
6. **El ángulo de rotación**: si el nombre del archivo del modelo cargado contiene la palabra "obb" (por ejemplo `best-obb.pt`), se lee el ángulo real que devuelve YOLO (`result.obb.xywhr`). Con el modelo actual (`yolov8n.pt`, que NO es un modelo OBB), **el ángulo siempre es 0°** — es una limitación conocida y documentada, no un bug.

### Paso 4 — Contorno de la sala (`RoomExtractor.detect_bounds`)
En paralelo (conceptualmente; en código corre en secuencia dentro de `compile()`), se busca el contorno más grande de la imagen binarizada — se asume que ese contorno es el perímetro de la sala — y se calcula su rectángulo delimitador en píxeles. A partir de ese rectángulo, `build_wall_elements()` genera 4 elementos de pared (`muro_norte/sur/este/oeste`) con el mismo formato que usan los presets de Unity.

### Paso 5 — Lectura de la escala (`ScaleDetector.detect`)
Se corre Tesseract OCR sobre la imagen buscando un patrón de texto tipo "4.20m" o "3,5 m" (regex). Si lo encuentra, se calcula cuántos píxeles equivalen a un metro usando el ancho en píxeles de la sala del Paso 4. **Si no encuentra ningún texto legible** (letra manuscrita ilegible, no se escribió ninguna medida, o Tesseract no está instalado en el sistema), se usa un tamaño de sala por defecto (4m × 4m, configurable en `settings.DEFAULT_ROOM_WIDTH_M/DEPTH_M`) — la sala generada en ese caso NO va a coincidir con el tamaño real del cuarto fotografiado.

### Paso 6 — Conversión de píxeles a metros
Con la escala del Paso 5, cada posición en píxeles (muebles del Paso 3, paredes del Paso 4) se convierte a metros. Hay además una inversión de eje: en la imagen, la fila 0 es arriba de la foto, pero en Unity el eje Z avanza "hacia adelante" — así que la conversión de la coordenada vertical de la imagen a la Z de Unity incluye ese volteo (`z_m = (alto_sala_px - (py - origen_px[1])) / px_por_metro`).

### Paso 7 — Se arma el JSON final
Todo lo anterior (4 paredes + N muebles) se junta en la lista `scene_elements` de un `SceneGraphResponse` (Pydantic, `models/scene_graph.py`) — el mismo schema que ya usan los presets y que `SceneGenerator.GenerateSceneAsync` en Unity sabe leer directo, sin traducciones intermedias.

---

## 3. ⚠️ Lo más importante para entender qué tan bien va a andar esto

**El modelo que está enchufado hoy (`yolov8n.pt`) nunca vio un plano de planta ni un ícono de mueble dibujado a mano — fue entrenado con fotos reales de habitaciones fotografiadas de costado**, no con vistas cenitales (desde arriba) ni con símbolos esquemáticos de un croquis. Esto es una diferencia de dominio grande: un sofá dibujado como un rectángulo con un patrón adentro (como se dibuja típicamente en un plano) no se parece en nada, visualmente, a una foto de un sofá real.

**En la práctica, esto significa que el detector puede fallar en detectar casi cualquier cosa** si la foto de entrada es un croquis dibujado a mano o un plano arquitectónico esquemático (el caso de uso real del proyecto). Puede andar razonablemente si la "foto del croquis" es en realidad una foto de una maqueta física con muebles reconocibles, o si por casualidad algún ícono se parece lo suficiente a algo de COCO — pero no hay que esperar detecciones confiables de un plano dibujado a mano con este modelo.

Esto **no es un bug**: es el estado esperado hasta que se entrene un modelo propio con un dataset de planos reales. Es exactamente para lo que existe el pipeline de `backend/training/cubicasa/` (ver sección 4) — pero ese entrenamiento todavía no se hizo.

**Recomendación concreta:** antes de asumir que "la detección de muebles funciona", probá el endpoint con una foto real del tipo de croquis que se va a usar en la demo del parcial, y mirá qué devuelve. Si devuelve una lista vacía de muebles (`scene_elements` solo con las 4 paredes), es exactamente el problema de arriba, no un error de código.

---

## 4. El camino para mejorar esto: CubiCasa5K + YOLOv8-OBB propio

Ya existe, escrito y funcional, un pipeline completo en `backend/training/cubicasa/` para convertir el dataset público **CubiCasa5K** (5000 planos reales anotados, con muebles e íconos de plano) al formato que Ultralytics necesita para entrenar un modelo YOLOv8-OBB (con ángulo real, no solo bounding box):

- `class_mapping.py`: mapea las clases originales de CubiCasa5K a las 18 clases unificadas que usa este proyecto.
- `svg_parser.py` / `svg_transform.py`: leen las anotaciones (en SVG) del dataset y las normalizan.
- `obb_converter.py`: convierte esas anotaciones al formato de cajas orientadas (OBB) que usa YOLOv8.
- `dataset_builder.py` / `cli.py`: arman el dataset final listo para entrenar (`python -m training.cli build`).

**Lo que falta para usar esto:**
1. Descargar el dataset CubiCasa5K (no está incluido en el repo, hay que bajarlo aparte — es pesado).
2. Correr el pipeline (`training/cli.py build`) para generar el dataset en formato YOLO.
3. Entrenar el modelo (`yolo obb train ...`, usando `ultralytics`) — esto puede tardar horas y idealmente se hace con GPU.
4. Poner el modelo resultante (`best.pt`) en algún lado accesible y apuntar `YOLO_MODEL_PATH` a esa ruta (por variable de entorno, sin tocar código) — en cuanto el nombre del archivo contenga "obb", `furniture_detector.py` automáticamente empieza a leer el ángulo real en vez de devolver 0°.

Esto es trabajo de varias horas/días (sobre todo el entrenamiento), así que es razonable dejarlo fuera del alcance de la entrega inmediata del parcial y presentarlo como "camino de mejora ya diseñado", si el tiempo no alcanza.
