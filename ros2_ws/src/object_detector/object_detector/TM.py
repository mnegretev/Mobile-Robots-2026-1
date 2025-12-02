from ultralytics import YOLO

# Crear modelo YOLOv8 preentrenado
model = YOLO("yolov8n.pt")

# Entrenar con tu dataset
model.train(
    data="PF_RM.v2i.yolov8/data.yaml",  # Ruta al YAML de tu dataset
    epochs=100,
    imgsz=640,
    batch=16,
    name="PF_RM_v2i"
)
