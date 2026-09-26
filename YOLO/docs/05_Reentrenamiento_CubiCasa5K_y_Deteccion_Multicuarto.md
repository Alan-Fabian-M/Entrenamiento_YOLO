# Sesión 2026-09-24/25 — Detección multi-cuarto, fix de escala OCR, y reconstrucción del pipeline de reentrenamiento CubiCasa5K

**Última actualización:** 2026-09-25

Este documento resume todo lo hecho en una sesión larga que tocó tres áreas
distintas del backend: (1) el compilador de escenas ya no asume un cuarto
único de 4x4, (2) se corrigió un bug de interpretación de escala en el OCR,
y (3) se diagnosticó por qué el modelo de muebles en producción es tan malo
y se reconstruyó desde cero el pipeline para reentrenarlo con datos reales.
Sirve como punto de partida para la próxima sesión — lee la sección
**"Dónde quedó todo"** primero si solo querés saber por dónde seguir.

---

## 1. De qué trata el proyecto (recordatorio rápido)

Backend FastAPI (`POST /api/v1/compilar-sala`) que recibe una foto/croquis
de un plano de casa y devuelve un JSON "Scene Graph" que un proyecto Unity
(`InmobiliariaVR`, en otro repo, no accesible desde acá) usa para armar la
sala en VR con muebles reales, instanciados en la posición/rotación/color
correctos. El pipeline combina tres sensores: YOLO-seg (muebles), OpenCV
(límites de cuartos), OCR/Tesseract (cotas escritas en el plano).

---

## 2. Qué se hizo en esta sesión

### 2.1 Detección multi-cuarto (antes: siempre asumía 1 cuarto rectangular)

**Problema:** `RoomExtractor` detectaba todos los cuartos reales del plano
pero descartaba todos menos el más grande, y `scene_compiler.py` forzaba
`width_m = depth_m = 4.0` fijo cuando el OCR no leía una cota. Cualquier
plano real con varios cuartos (el caso típico de CubiCasa5K) se aplanaba a
una caja falsa de 4x4, con muebles a veces fuera de esos límites.

**Solución:**
- `app/services/room_extractor.py`: `detect_rooms()` (nuevo) devuelve TODOS
  los cuartos detectados, no solo el más grande. `build_room_walls()`
  (nuevo) genera 4 muros por cuarto, centrados en su propia posición.
- `app/services/scene_compiler.py`: calcula un bbox unión de todos los
  cuartos para origen/escala compartidos, deriva el tamaño real de cada
  cuarto en metros desde su geometría en píxeles (ya no hay 4x4 fijo),
  asigna cada mueble al cuarto que lo contiene (`shapely`), e infiere
  `room_type` (baño/cocina/dormitorio/sala/comedor) según los muebles
  detectados en cada cuarto.
- `app/models/scene_graph.py`: cambios **aditivos únicamente** — nuevo
  campo `rooms: List[RoomShapeInfo]`, `room_id` en `ElementProperties`,
  `room_count` en metadata. El campo `room_info` (singular) se mantiene
  igual que antes para no romper el lado de Unity, que no se puede tocar
  desde esta sesión.
- `tests/test_room_extractor.py` (nuevo): 4 tests, todos verdes.

Probado con 5 imágenes reales de `imagenes_cubicasa_test/`: cada plano
ahora reporta entre 5 y 8 cuartos con tamaños reales y distintos, en vez de
la caja de 4x4 de siempre.

### 2.2 Bug de escala OCR (números de 3 dígitos mal interpretados)

**Problema:** `app/services/scale_detector.py` interpreta números sueltos
sin unidad (`"2200"`, `"1800"`) como milímetros — convención típica de
AutoCAD/FloorPlanCAD. Pero muchos planos (el ejemplo real que lo disparó:
`imagenes/WhatsApp Image 2026-09-21 at 5.24.56 PM.jpeg`) usan la otra
convención arquitectónica común: cotas como `"5.00"` (metros, sin unidad
explícita). El OCR pierde el punto decimal al leer texto chico (`"5.00"` →
`"500"`), y el código lo interpretaba como 500mm en vez de 5.00m — un
cuarto de 5 metros terminaba reportado como 0.5 metros.

**Solución:** para valores de 3 dígitos (100-999), la interpretación en mm
siempre da menos de 1 metro (implausible como escala de referencia de todo
un plano), mientras que interpretarlo como "X.XX metros con el punto
perdido" (÷100) da 1.00-9.99m — mucho más plausible. Se agregó esa
heurística de desambiguación en `scale_detector.py`. Con la imagen real
que disparó el bug: `width` pasó de 0.5 a 5.0 (el real es 5.00m).

### 2.3 Reconstrucción del pipeline de reentrenamiento CubiCasa5K (el trabajo grande)

**Contexto:** se investigó por qué el modelo de muebles en producción
(`YOLO/best_seg.pt`) tiene mAP50 general de solo **0.09** — prácticamente
inutilizable (confianzas máximas 0.02-0.13 en todas las clases). Se
encontraron dos bugs reales, no un problema de hiperparámetros:

1. El notebook que generó ese modelo (`COLAB_ENTRENAMIENTO_YOLO_SEG.ipynb`,
   la versión vieja) escribía líneas de label **sin coordenadas** —
   literalmente `str(yolo_id)`, solo el número de clase. Un label de YOLO
   válido necesita el polígono completo. Nunca hubo geometría real en el
   dataset de entrenamiento.
2. El pipeline existente para parsear el dataset CRUDO de CubiCasa5K
   (`training/cubicasa/svg_parser.py` y compañía) nunca se había probado
   contra un SVG real — solo tenía tests con SVGs sintéticos hechos a mano,
   y esos sintéticos no reflejaban cómo CubiCasa5K estructura sus archivos
   de verdad (la clase semántica vive en un `<g>` ancestro, nunca en el
   elemento con la geometría — ver detalle en el código).

**Verificación con datos reales:** se descargó el dataset real de Zenodo
(record correcto: `2613548` — la URL vieja apuntaba a un record `3384994`
que ya no existe, da 404) y se extrajeron ~60 muestras reales completas
(SVG + PNG) para auditar contra ellas. Esto reveló también que **CubiCasa5K
no tiene cama/sofá/mesa/silla como íconos anotados en absoluto** — solo
anota mobiliario *fijo* (`FixedFurniture`: inodoro, lavabo, tina, clóset,
electrodomésticos). Es un dataset de planos de permisos de construcción, no
de amoblamiento. Esas clases movibles sí existen en `FloorPlanCAD` (otro
dataset, 200 planos, ya usado antes para `YOLO/best_floorplancad.pt`).

**Lo que se reescribió/agregó (todo commiteado):**

| Archivo | Qué cambió |
|---|---|
| `training/cubicasa/svg_parser.py` | Nuevo algoritmo: la geometría se asocia con la clase semántica del **ancestro** más cercano (`Wall`, `Space `, `FixedFurniture`, `Window`, `Door`, `Column`, `Railing`, `Stairs`, `Bench`), no con la clase del propio elemento. |
| `training/cubicasa/class_mapping.py` | Taxonomía de 23 clases reescrita desde cero con las clases reales observadas (`wall`, `room`, `bathroom`, `kitchen`, `living_room`, `bedroom`, `dining_room`, `outdoor`, `door`, `window`, `column`, `stairs`, `toilet`, `sink`, `bathtub`, `shower`, `wardrobe`, `cabinet`, `kitchen_counter`, `fireplace`, `electrical_appliance`, `bench`, `misc_furniture`). **No incluye** cama/sofá/mesa/silla — esas vienen de FloorPlanCAD. |
| `training/cubicasa/dataset_builder.py` | Ahora emite polígonos de segmentación completos (`class x1 y1 x2 y2 ...`) en vez de cajas OBB de 4 esquinas. Clase renombrada `YoloObbDatasetBuilder` → `YoloSegDatasetBuilder`. |
| `training/download_cubicasa_real.py` (nuevo) | Descarga el zip real (~5.47GB) desde la URL correcta de Zenodo. |
| `training/build_cubicasa_floorplancad_mix.py` (nuevo) | Fusiona el dataset de CubiCasa (seg, taxonomía nueva) con FloorPlanCAD (bbox, taxonomía vieja de 16 clases) en un solo vocabulario unificado de 29 clases, convirtiendo las cajas de FloorPlanCAD a polígonos degenerados de 4 esquinas para que todo el dataset final sea 100% seg. |
| `tests/fixtures/cubicasa5k_real/` (nuevo) | ~15 muestras reales (SVG+PNG) committeadas como fixtures, reemplazando la confianza ciega en SVGs sintéticos. |
| `tests/test_cubicasa_conversion.py` | 41 tests en total (antes 22), incluyendo contra las fixtures reales. |
| `training/entrenamiento_cubicasa_unificado_colab.ipynb` (nuevo) | Notebook de Colab listo para correr: descarga CubiCasa5K real + construye el dataset + descarga/convierte FloorPlanCAD + fusiona + entrena `yolov8s-seg`. |

**Smoke test local antes de tocar Colab:** se corrió el pipeline completo
contra las ~60 muestras reales descargadas (`training.cli build`). 0
muestras saltadas, instancias reales en 22 de 23 clases, cero clases sin
reconocer. Esto es lo que le dio confianza al equipo para lanzar el
entrenamiento real en Colab sin repetir el mismo error de antes (gastar
horas de GPU en un dataset roto).

---

## 3. Dónde quedó todo (actualizado 2026-09-26)

**El entrenamiento terminó y el modelo ya está instalado en el backend.**

- Colab se quedó sin cuota de GPU, así que el entrenamiento se corrió en
  **Kaggle** (`training/entrenamiento_kaggle.ipynb`, 2×T4, modo "Save & Run
  All"). Dataset unificado subido como dataset de Kaggle (8447 train / 1461
  val, 29 clases). 40 épocas, `yolov8s-seg`, `imgsz=640`, ~11.2 h (cerca del
  límite de 12 h de Kaggle -- no cabe una corrida más larga).
- Pesos: `YOLO/best_unified_seg.pt` (29 clases; `yolov8s-seg.pt` que aparece
  también en el Output de Kaggle es el modelo COCO base, NO el entrenado --
  ya nos confundimos una vez).
- `YOLO_MODEL_PATH` apunta a ese archivo, `YOLO_CONFIDENCE_THRESHOLD=0.25`.

### Resultados de validación (mAP50 caja / máscara)

General 0.26 / 0.24. Por clase, lo relevante:

| Clase | mAP50 | Nota |
|---|---|---|
| bed, sofa, table, chair | 0.85-0.90 | Inflado: la validación sale del mismo estilo de FloorPlanCAD |
| refrigerator | 0.75 | idem |
| door / stairs / bathtub | 0.40 / 0.39 / 0.55 | aceptables |
| wall | 0.18 | débil; el backend usa OpenCV para muros, no depende de esto |
| room, bathroom, kitchen, bedroom, living_room | 0.09-0.16 | débiles |
| toilet, sink, window | 0.13, 0.05, 0.23 | débiles |
| shower, kitchen_counter, column | ~0 | inservibles |

### Hallazgo clave: brecha de dominio (fondo negro vs blanco)

Sobre planos CAD reales del usuario (`imagenes/`) el modelo **no detectaba
ni un mueble** (ni con umbral 0.05 ni con 1024 px). Causa: las imágenes de
FloorPlanCAD, única fuente de bed/sofa/table/chair/refrigerator/gas_stove,
son líneas claras sobre **fondo casi negro** (brillo medio 0.4-4.4 / 255),
mientras que los planos de usuario y los de CubiCasa5K son líneas oscuras
sobre fondo blanco. Invertir la imagen hizo aparecer la cama (conf 0.56).

Solución aplicada en `app/services/furniture_detector.py`, para planos de
fondo claro (y solo si el modelo tiene esas clases):

1. Pasada original a `YOLO_IMAGE_SIZE=640` -> todas las clases excepto las 6
   de FloorPlanCAD.
2. Pasada sobre la imagen **invertida** a `YOLO_FURNITURE_IMAGE_SIZE=1280`
   -> solo `bed, sofa, table, chair, refrigerator, gas_stove`.
3. NMS entre clases (`suppress_overlaps`, IoU 0.5): YOLO solo hace NMS por
   clase, y el mismo sofá salía como `sofa` y como `cama`.

Los tamaños distintos son deliberados: 1280 mejora los muebles pero empeora
puertas/escaleras/armarios, y 640 al revés.

Resultado en los planos de prueba (umbral 0.25, sin contar muros):
`WhatsApp 5.24.56` -> 1 sofá, 1 silla, 1 cama (antes: nada);
`WhatsApp 5.24.05` -> silla, escalera, puerta; `CubiCasa 199` -> 3 armarios.

**Limitaciones conocidas (no resueltas):** falta la mesa redonda con sillas,
la estufa y el lavabo del plano principal; hay falsos positivos (una silla
detectada sobre el arco de una puerta). Solo se probó con 2 planos CAD
propios -- el usuario no tiene más. Si se necesita más calidad: etiquetar
30-50 planos con el estilo real y mezclarlos con el dataset, o aumentación
fuerte (grosor de línea, inversión aleatoria) al reentrenar.

`CLASS_TO_UNITY_TYPE` ya cubre el vocabulario nuevo. **No mapeadas a
propósito:** `wall` y las clases de área (room/bathroom/...), y
`cabinet/electrical_appliance/bench/misc_furniture` (mAP<0.1, sin prefab
Unity conocido). Tests en `tests/test_furniture_detector.py`.

---

## 4. Próximos pasos (priorizados)

1. **Usar las clases de habitación del modelo** (`bedroom`, `kitchen`,
   `bathroom`, `living_room`) para inferir `room_type` en `SceneCompiler`;
   hoy se infiere solo por los muebles asignados. Cambio chico, no rompe el
   contrato JSON. Es donde el modelo rinde mejor en planos reales.
2. Mejorar la detección de muebles en planos CAD con más datos del estilo
   real (ver limitaciones arriba).
3. Evaluar si `RoomExtractor` puede usar `wall`/`room` del modelo; hoy no
   (wall mAP 0.18) y OpenCV funciona bien.
4. Confirmar con el lado Unity qué tipos nuevos (`ducha`, `chimenea`,
   `refrigerador`, `estufa`) tienen prefab; no se puede verificar desde acá.
5. Fuera de esta sesión: conectividad real Quest↔PC, captura en vivo con
   la cámara del Quest 3 (ver `InmobiliariaVR/Docs/Plan_Funcionalidades_Visor.md`).

---

## 5. Archivos clave para retomar

- Este documento y `YOLO/docs/04_Estado_Actual_y_Pendientes.md` (más viejo,
  parcialmente desactualizado por lo de acá — los puntos de detección
  multi-cuarto y escala OCR de ese documento ya están resueltos).
- `training/entrenamiento_cubicasa_unificado_colab.ipynb` — el notebook a
  retomar en Colab.
- `training/build_cubicasa_floorplancad_mix.py` — el paso de fusión
  pendiente de correr.
- `app/services/furniture_detector.py` — `CLASS_TO_UNITY_TYPE`, pendiente
  de extender una vez haya modelo nuevo.
- `tests/test_cubicasa_conversion.py` y `tests/test_room_extractor.py` —
  correr `pytest tests/ -v` para confirmar que nada se rompió antes de
  seguir tocando código.
