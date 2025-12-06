#!/usr/bin/env python3
"""
SCRIPT DE PRUEBA YOLO
Prueba el modelo YOLO y muestra las clases disponibles
"""

from ultralytics import YOLO
import cv2
import sys

def test_yolo_classes():
    """Muestra todas las clases que YOLO puede detectar"""
    print("=" * 60)
    print("CLASES DISPONIBLES EN YOLO")
    print("=" * 60)
    
    try:
        model = YOLO('yolov8n.pt')
        
        print("\nModelo cargado correctamente: yolov8n.pt\n")
        print("Clases detectables (80 clases de COCO):\n")
        
        for idx, class_name in model.names.items():
            print(f"{idx:3d}: {class_name}")
        
        print("\n" + "=" * 60)
        print("MAPEO SUGERIDO PARA TU PROYECTO")
        print("=" * 60)
        print("\nPara los objetos del proyecto:")
        print("  • Coca-Cola (lata)  -> 'bottle' (clase 39) o 'cup' (clase 41)")
        print("  • Caja de cereal    -> 'book' (clase 73) o 'bowl' (clase 45)")
        print("  • Bote de basura    -> 'potted plant' (clase 58) o similar")
        print("\nNota: Estos son aproximaciones. Ajusta según resultados reales.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPara instalar YOLO ejecuta:")
        print("  pip install ultralytics")

def test_yolo_on_image(image_path):
    """Prueba YOLO en una imagen"""
    import os
    
    # Expandir ~ y convertir a ruta absoluta
    image_path = os.path.expanduser(image_path)
    image_path = os.path.abspath(image_path)
    
    print(f"\nProbando YOLO en: {image_path}")
    
    # Verificar que existe
    if not os.path.exists(image_path):
        print(f"❌ Archivo no encontrado: {image_path}")
        print(f"\nVerifica la ruta. Usa comillas si tiene espacios:")
        print(f'  python3 test_yolo.py "/ruta/con espacios/imagen.jpg"')
        return
    
    try:
        model = YOLO('yolov8n.pt')
        
        # Leer imagen
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ No se pudo leer la imagen: {image_path}")
            print(f"   Formatos soportados: .jpg, .jpeg, .png, .bmp")
            return
        
        # Detectar
        results = model(img, conf=0.5)
        
        print("\n🔍 Detecciones:")
        for result in results:
            boxes = result.boxes
            
            if len(boxes) == 0:
                print("  No se detectaron objetos")
            else:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls_id]
                    
                    print(f"  • {class_name} (confianza: {conf:.2f})")
                    
                    # Dibujar en imagen
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f'{class_name} {conf:.2f}',
                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                              0.5, (0, 255, 0), 2)
        
        # Mostrar resultado
        cv2.imshow('Detecciones YOLO', img)
        print("\n✓ Presiona cualquier tecla para cerrar la ventana...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")

def main():
    print("\n" + "="*60)
    print("SCRIPT DE PRUEBA YOLO")
    print("="*60)
    
    if len(sys.argv) > 1:
        # Si se proporciona una imagen, probar en ella
        test_yolo_on_image(sys.argv[1])
    else:
        # Solo mostrar clases
        test_yolo_classes()
        print("\nPara probar en una imagen:")
        print(f"  python3 {sys.argv[0]} path/to/image.jpg")

if __name__ == '__main__':
    main()