import os
import shutil

# Ruta del directorio donde buscar los archivos 
base_dir = "/media/roovedot/PHILIPS/cityscapes/leftImg8bit_trainextra/leftImg8bit/train_extra"

# Recorre recursivamente todas las subcarpetas
for root, _, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".png"):  # Filtrar tipos de archivos
            file_path = os.path.join(root, file) # Ruta del archivo
            output_path = "/media/roovedot/PHILIPS/cityscapes/dataYolo11Structure/train/images" # Ruta de la carpeta output
            #output_path = os.path.join(base_dir, file) # Para guardar en la carpeta principal

            # Mover el archivo a la carpeta output
            if file_path != output_path:  # Evita mover si ya está en el destino
                shutil.move(file_path, output_path)
                print(f"{file} Movido: {file_path} -> {output_path}")
            else:
                print(f"{file} ya está en el destino")
