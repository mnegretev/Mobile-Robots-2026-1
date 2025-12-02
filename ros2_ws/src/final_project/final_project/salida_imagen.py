
import cv2
import numpy as np
import os
from datetime import datetime

# Create a dummy image (e.g., a black image)
image = np.zeros((200, 300, 3), dtype=np.uint8)

# Save the image as a JPEG file
#cv2.imwrite("saved_image.jpg", image)

# Save the image as a PNG file with a specific compression level
# PNG_COMPRESSION parameter ranges from 0 (fastest, largest file) to 9 (slowest, smallest file)
output_folder = 'salida_ubicaciones'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

salida_nom = "cup"
output_filename = f'saved_image_{salida_nom}.png'
output_path = os.path.join(output_folder, output_filename)
cv2.imwrite(output_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 5])
print(f"Imagen guardada en -{output_path}-")

# Obtener fecha y hora actual
ahora = datetime.now()

# Opción A: Impresión cruda (menos legible)
print(ahora) 
# Salida: 2025-12-01 13:05:00.123456

# Opción B: Formato personalizado (Recomendado)
# %Y=Año, %m=Mes, %d=Día, %H=Hora(24h), %M=Minuto, %S=Segundo
texto_hora = ahora.strftime("%Y-%m-%d %H:%M:%S")

print(f"Hora actual: {texto_hora}")
# Salida: Hora actual: 2025-12-01 13:05:00



output_filename = f'ubi_detection.txt'
output_path_txt = os.path.join(output_folder, output_filename)

fichero = open(output_path_txt, "a")
#salida = f"Objeto: book Ubicacion: (16, 5)\n"
salida = f"\nBusqueda {texto_hora}\n"
fichero.write(salida)
fichero.close()

#cv2.imwrite("saved_image.png", image, [cv2.IMWRITE_PNG_COMPRESSION, 5])

print("Images saved successfully.")