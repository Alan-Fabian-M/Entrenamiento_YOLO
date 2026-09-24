# 👨‍💻 Contribución al Proyecto: Módulo de Visión Artificial y Entrenamiento YOLOv8

**Colaborador:** Ferna  
**Rol:** Computer Vision & AI Engineer  
**Rama:** `feature/ferna-ia-floorplancad`  
**Fecha:** Septiembre 2026  

---

## 🎯 Objetivos de la Contribución

1. **Automatización del Dataset:** Diseñar e implementar el pipeline de descarga y preparación del dataset arquitectónico [Voxel51/FloorPlanCAD](https://huggingface.co/datasets/Voxel51/FloorPlanCAD) desde Hugging Face.
2. **Conversión y Preprocesamiento:** Crear la herramienta de transformación de metadatos y anotaciones CAD a formato YOLOv8 con división balanceada (Train / Validation).
3. **Entrenamiento y Optimización de Modelo:** Entrenar y realizar fine-tuning sobre YOLOv8 para la detección precisa de muros, puertas (corredizas/dobles), ventanas y mobiliario.
4. **Validación y Benchmarking:** Evaluar cuantitativa y cualitativamente el modelo con métricas mAP50, Precision y Recall.

---

## 📦 Archivos y Módulos Desarrollados

```text
training/
├── download_floorplancad_hf.py      # Descarga automatizada desde Hugging Face
├── convert_floorplancad_to_yolo.py  # Conversor de anotaciones a formato YOLO
├── train_yolo.py                    # Script de entrenamiento configurable
├── predict_visual_sample.py         # Visualizador de inferencia con bounding boxes
├── test_floorplancad_samples.py     # Suite de pruebas sobre planos de test
└── README_TRAINING.md               # Guía técnica de uso del módulo de entrenamiento

contributions/ferna/
├── README.md                        # Resumen de contribución del colaborador
└── INFORME_ENTRENAMIENTO_200_PLANOS.md # Informe técnico con métricas detalladas
```

---

## 📊 Resultados y Logros Clave

| Métrica | Estado Inicial (50 Planos) | **Contribución Final (200 Planos)** | Impacto |
| :--- | :---: | :---: | :---: |
| **Precisión General (P)** | 59.4% | **77.6%** | 🟢 **+18.2%** |
| **mAP50 General** | 20.5% | **46.6%** | 🟢 **+26.1%** |
| **`sliding_door` (Puerta corrediza)** | 36.0% | **92.2%** | 🟢 **+56.2%** |
| **`double_door` (Puerta doble)** | 41.0% | **81.9%** | 🟢 **+40.9%** |
| **Mobiliario (Camas, Sofás, Mesas)** | No detectado | **> 80% - 87%** | 🟢 Nuevas clases |

---

## 📄 Informes y Documentación

Para ver las matrices de confusión, tiempos de inferencia y desglose por cada clase arquitectónica, consultar:
* 📑 [INFORME_ENTRENAMIENTO_200_PLANOS.md](./INFORME_ENTRENAMIENTO_200_PLANOS.md)
