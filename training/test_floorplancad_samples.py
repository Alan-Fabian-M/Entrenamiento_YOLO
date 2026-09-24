"""
Script de prueba para evaluar los planos de FloorPlanCAD con el pipeline de SceneCompiler / YOLO.
"""

from __future__ import annotations

import json
from pathlib import Path
from app.services.scene_compiler import SceneCompiler

def test_samples():
    samples_dir = Path("data_samples/floorplancad/images")
    image_files = sorted(list(samples_dir.glob("*.png")))
    
    if not image_files:
        print("No se encontraron imagenes en", samples_dir)
        return

    print("=" * 80)
    print(f"EVALUANDO {len(image_files)} PLANOS DE FLOORPLANCAD CON EL BACKEND")
    print("=" * 80)

    results = []

    for idx, img_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Procesando: {img_path.name}...")
        img_bytes = img_path.read_bytes()
        
        try:
            resp = SceneCompiler.compile(img_bytes)
            
            meta = resp.metadata
            room = resp.room_info
            elements = resp.scene_elements
            
            print(f"  -> Tiempo procesamiento: {meta.processing_time_ms} ms")
            print(f"  -> Dimensiones calculadas: Ancho={room.dimensions.x:.2f}m, Alto={room.dimensions.y:.2f}m, Largo={room.dimensions.z:.2f}m")
            print(f"  -> Escala: {meta.dimensions.width:.2f}m x {meta.dimensions.depth:.2f}m (Fuente: '{meta.scale_source}')")
            print(f"  -> Total elementos detectados: {len(elements)}")
            
            # Conteo de elementos por tipo
            type_counts = {}
            for el in elements:
                t = el.type
                type_counts[t] = type_counts.get(t, 0) + 1
            
            for t, cnt in sorted(type_counts.items()):
                print(f"     * {t}: {cnt}")
                
            results.append({
                "filename": img_path.name,
                "processing_time_ms": meta.processing_time_ms,
                "room_dimensions_m": {"x": room.dimensions.x, "y": room.dimensions.y, "z": room.dimensions.z},
                "total_elements": len(elements),
                "element_breakdown": type_counts,
                "scale_source": meta.scale_source
            })
            
        except Exception as e:
            print(f"  -> ERROR al procesar {img_path.name}: {e}")

    print("\n" + "=" * 80)
    print("RESUMEN DE PRUEBAS")
    print("=" * 80)
    print(f"Planos evaluados exitosamente: {len(results)}/{len(image_files)}")
    
    # Guardar resumen en disco
    out_file = Path("data_samples/floorplancad/test_results.json")
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Resultados detallados guardados en: {out_file}")

if __name__ == "__main__":
    test_samples()
