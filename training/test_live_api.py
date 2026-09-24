import requests
import json

url = "http://127.0.0.1:8000/api/v1/compilar-sala"
file_path = r"D:\Descargas\croquis_test.png"

print("Enviando petición HTTP POST a:", url)
with open(file_path, "rb") as f:
    files = {"file": ("croquis_test.png", f, "image/png")}
    response = requests.post(url, files=files)

print("Status Code:", response.status_code)
if response.status_code == 200:
    data = response.json()
    print("\n--- METADATA DE LA RESPUESTA ---")
    print(json.dumps(data["metadata"], indent=2, ensure_ascii=False))
    print("\n--- RESUMEN DE ELEMENTOS DETECTADOS ---")
    print(f"Total elementos en escena: {len(data['scene_elements'])}")
    for i, el in enumerate(data["scene_elements"]):
        pos = el["position"]
        print(f"  [{i+1:02d}] Tipo: {el['type']:<18} | Conf: {el.get('confidence', 0):.2f} | Posición Unity: ({pos['x']:.2f}, {pos['y']:.2f}, {pos['z']:.2f})")
else:
    print("Error:", response.text)
