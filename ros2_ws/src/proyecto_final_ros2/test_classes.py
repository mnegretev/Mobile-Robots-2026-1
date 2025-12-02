from ultralytics import YOLO
import cv2

# 1) Cargar modelo preentrenado
model = YOLO("yolo11n.pt")  # o el que uses

# 2) Leer imagen con OpenCV
img = cv2.imread("bote.png")  # pon aquí la ruta a tu imagen

# 3) Ejecutar inferencia
results = model(img)

# 4) Tomar el primer resultado
r = results[0]

print("Clases detectadas:")
for box in r.boxes:
    cls_id = int(box.cls[0])
    cls_name = r.names[cls_id]
    conf = float(box.conf[0])
    print(f"{cls_id} -> {cls_name} (conf={conf:.2f})")
