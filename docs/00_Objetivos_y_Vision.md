# Objetivos y visión del proyecto

## Qué es

Backend de visión artificial para **InmobiliariaVR** (Primer Parcial, ISW2 —
Grupo 1). Recibe una foto o escaneo de un plano de casa (papel, croquis a
mano, o export CAD) y devuelve un **Scene Graph** en JSON: la descripción
completa de una sala (paredes + muebles, con posición/rotación/color) lista
para que el proyecto Unity (`InmobiliariaVR`, repo separado) la instancie en
3D y el usuario la recorra en VR con un Meta Quest.

```
[Foto de Plano]
      │
      ▼
Preprocesamiento (FastAPI + OpenCV)
      │
      ├── YOLO-seg: detecta muebles (tipo, posición, forma, color)
      ├── OpenCV: detecta los límites de cada cuarto
      └── OCR (Tesseract): lee una cota escrita para calcular la escala
      │
      ▼
Scene Graph JSON — POST /api/v1/compilar-sala
      │
      ▼
Unity SceneGenerator.cs arma la sala en VR
```

## Qué significa "terminado" para este backend

1. **El plano de entrada puede ser cualquier estilo razonable** — no solo
   fotos limpias de planos CAD, sino también croquis a mano, escaneos con
   sombras, y planos con varios cuartos (no solo un monoambiente).
2. **La sala generada refleja la geometría real del plano**: cuartos
   correctos (cantidad, tamaño, forma), muebles en la posición y con el
   tipo correcto, y la escala en metros derivada de una cota real del plano
   (no un tamaño inventado).
3. **El pipeline es confiable sin supervisión manual constante** — hoy
   todavía depende de un modelo de detección de muebles entrenado con datos
   reales de calidad (ver `docs/02_Estado_Actual.md`); mientras ese modelo
   no esté listo, el sistema es funcional pero con baja confianza en la
   detección de muebles.
4. **El contrato JSON nunca rompe al proyecto Unity** — se puede extender
   (campos nuevos) pero nunca se renombra ni se quita algo que Unity ya
   consume. Ver `docs/01_Reglas_del_Proyecto.md`.

## Prioridad actual (según lo definido con el usuario)

Que funcione bien con **planos estilo CAD** (líneas limpias, exportados o
escaneados sin mucha distorsión) primero. Fotos de croquis a mano con mala
iluminación, perspectiva, etc. ("no-CAD") quedan para una fase posterior —
no hay que optimizar para ese caso todavía si compite con el caso CAD.

## Fuera de alcance de este backend (vive en otro repo/fase)

- Captura de foto en vivo con la cámara del Quest 3 (Passthrough Camera
  API) — es responsabilidad del proyecto Unity.
- Todo lo que pasa después de recibir el Scene Graph JSON (renderizado,
  interacción del usuario en VR, `SceneGenerator.cs`, `PrefabMapper.cs`) —
  vive en el repo de Unity, no accesible desde este backend.
