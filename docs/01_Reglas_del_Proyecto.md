# Reglas del proyecto (invariantes que no se pueden romper)

Estas son restricciones duras, no preferencias de estilo. Romper cualquiera
de estas sin querer es la forma más común de introducir un bug que solo se
descubre del lado de Unity, semanas después.

## 1. El contrato JSON con Unity es aditivo únicamente

`app/models/scene_graph.py` define el schema exacto que
`SceneGenerator.cs`/`PrefabMapper.cs` parsean del lado de Unity (repo
`InmobiliariaVR`, separado, **no accesible desde esta sesión de trabajo**).

- **Nunca renombrar ni quitar un campo existente.**
- Los campos nuevos siempre necesitan un valor por default, para que una
  respuesta que no los use siga siendo válida (ej. `rooms: List[...] =
  Field(default_factory=list)`).
- Si hace falta un cambio que rompe el contrato (renombrar algo, cambiar el
  tipo de un campo), **hay que coordinarlo explícitamente con el lado
  Unity primero** — no se puede verificar ni testear ese lado desde acá.
- Motivo: no hay forma de correr ni testear el proyecto Unity desde esta
  máquina/sesión. Un cambio que rompe el contrato falla en silencio acá
  (los tests de este repo van a seguir pasando) y solo se nota cuando
  alguien prueba en Unity.

## 2. CubiCasa5K no tiene mobiliario movible

El dataset CubiCasa5K (el dataset real usado para reentrenar el modelo de
planos) **no anota camas, sofás, mesas ni sillas** — solo mobiliario fijo
(`FixedFurniture`: inodoro, lavabo, tina, clóset, electrodomésticos). Es un
dataset de planos de permisos de construcción, no de amoblamiento.

- Cualquier intento de "mejorar la detección de sofás/sillas reentrenando
  con CubiCasa5K" está condenado a fallar — la clase simplemente no existe
  en los datos fuente, sin importar qué tan bien se ajuste el pipeline.
- El mobiliario movible viene de un dataset distinto: **FloorPlanCAD** (200
  planos, ya usado para `YOLO/best_floorplancad.pt`). El pipeline de fusión
  (`training/build_cubicasa_floorplancad_mix.py`) combina ambos datasets en
  un solo vocabulario de clases para entrenar un modelo unificado.

## 3. Convención de escala: ambigüedad "mm vs metros con punto perdido"

El OCR (`app/services/scale_detector.py`) puede leer una cota escrita en el
plano de dos formas, y **ninguna de las dos es universal**:

- Estilo AutoCAD/FloorPlanCAD: números sueltos sin unidad = milímetros
  (`"2200"` → 2.2m).
- Estilo arquitectónico general: números con punto decimal sin unidad =
  metros (`"5.00"` → 5.00m) — pero el OCR frecuentemente pierde el punto en
  texto chico, y `"5.00"` termina leyéndose como `"500"`.

La heurística actual (ver `scale_detector.py`) desambigua por plausibilidad:
un valor de 3 dígitos interpretado como mm da menos de 1 metro (poco
plausible como escala de referencia de un cuarto entero), así que se
prefiere la interpretación "metros con el punto perdido" en ese rango. **Si
se encuentra un plano real donde esto falla al revés (un cuarto
legítimamente de <1m de referencia), hay que revisar esta heurística, no
asumir que está bien para siempre.**

## 4. `RoomExtractor` trata cada cuarto como un rectángulo, no un polígono libre

Decisión de alcance deliberada (no una limitación técnica insalvable): los
cuartos de CubiCasa5K son prácticamente siempre rectangulares en la
práctica, así que reconstruirlos como rectángulos (bbox) es suficiente y
mucho más simple que un grafo de polígonos arbitrarios con deduplicación de
muros compartidos. Si en algún momento aparecen planos reales con cuartos
en L u otras formas no rectangulares con frecuencia, esto hay que
revisitarlo — no está soportado hoy.

## 5. Los muros nunca rotan (`rotation.y` siempre en 0)

Aunque `RoomShape` ya calcula/reserva un ángulo de rotación por cuarto, los
muros generados siempre se emiten con `rotation = {0,0,0}`. Esto es
deliberado: activar rotación real es la única pieza de este trabajo que no
se puede verificar sin acceso al proyecto Unity (¿el prefab de muro pivota
bien en Y?). Hasta que alguien lo confirme del lado Unity, **no activar
rotación real en muros** aunque parezca una mejora obvia.

## 6. No usar `--no-verify` / saltar hooks / forzar pushes sin permiso explícito

Regla general de higiene de repositorio, no específica de este proyecto,
pero vale repetirla: nunca se hacen commits que saltean hooks, ni se hace
`push --force`, sin que el usuario lo pida explícitamente para esa acción
puntual.
