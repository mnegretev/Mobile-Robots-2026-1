from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("yolo11n.pt")

# Perform object detection on an image
#results = model("https://ultralytics.com/images/bus.jpg")
results = model(source="/home/juan/Imágenes/PruebaSalida.png")

# Visualize the results
for result in results:
    result.show()