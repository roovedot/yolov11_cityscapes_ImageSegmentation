import os
import json

# Define the class mapping based on the given table
CLASS_MAPPING = {
    # "unlabeled": 0,
    # "ego vehicle": 1,
    "rectification border": 2,
    # "out of roi": 3, # Un borde alrededor de la imagen? 
    # "static": 4, # Coge cosas como bancos, basuras, carteles...
    # "dynamic": 5,
    "ground": 6,
    "road": 7,
    "sidewalk": 8,
    "parking": 9,
    "rail track": 10,
    # "building": 11,
    "wall": 12,
    "fence": 13,
    "guard rail": 14,
    # "bridge": 15,
    # "tunnel": 16,
    # "pole": 17,
    # "polegroup": 18,
    "traffic light": 19,
    # "traffic sign": 20,
    # "vegetation": 21,
    "terrain": 22,
    # "sky": 23,
    "person": 24,
    "rider": 25,
    "car": 26,
    "truck": 27,
    "bus": 28,
    "caravan": 29,
    "trailer": 30,
    "train": 31,
    "motorcycle": 32,
    "bicycle": 33,
    # "license plate": -1,
}

def redefine_class_mapping(original_mapping):
    new_mapping = {}
    current_index = 0

    for label, class_id in original_mapping.items():
        new_mapping[label] = current_index
        current_index += 1

    #print("Nuevo CLASS_MAPPING:")
    #print(json.dumps(new_mapping, indent=4))
    #print("nº classes: ", len(new_mapping))
    return new_mapping

def generate_yaml_text(class_mapping):
    # Ordenar el mapeo por los índices de las clases (valores)
    sorted_classes = sorted(class_mapping.items(), key=lambda x: x[1])
    class_names = [cls[0] for cls in sorted_classes]

    # Crear el texto del archivo .yaml
    yaml_text = f"nc: {len(class_mapping)}\n"  # Número de clases
    yaml_text += "names:\n"  # Lista de nombres de clases
    for i, class_name in enumerate(class_names):
        yaml_text += f"  {i}: {class_name}\n"

    print("Texto para el archivo .yaml:")
    print(yaml_text)
    return yaml_text

#Converts single file to Yolo Format
def convert_to_yolo(json_file, output_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    #get annotation data
    img_width = data["imgWidth"]
    img_height = data["imgHeight"]
    objects = data["objects"]

    yolo_annotations = []
    for obj in objects:
        label = obj["label"]

        #Ignore commented Labels
        if label not in CLASS_MAPPING or CLASS_MAPPING[label] == -1:
            continue # jumps to next item

        class_index = CLASS_MAPPING[label]
        polygon = obj["polygon"]

        # Normalize polygon coordinates
        normalized_coords = []
        for point in polygon:
            x_norm = point[0] / img_width
            y_norm = point[1] / img_height
            normalized_coords.append(f"{x_norm:.6f} {y_norm:.6f}")

        # YOLO format: <class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>
        yolo_annotations.append(f"{class_index} " + " ".join(normalized_coords))

    # Write YOLO annotations to file
    with open(output_file, "w") as f:
        f.write("\n".join(yolo_annotations))

def process_cityscapes(leftImg8bit_path, gtFine_path):
    print(os.listdir(leftImg8bit_path))
    for image_file in os.listdir(leftImg8bit_path): #iterates trough images folder
        if image_file.endswith("_leftImg8bit.png"): # security
            nameWOextension = image_file.replace(".png", "") #gets name of image without extension
            base_name = image_file.replace("_leftImg8bit.png", "") #gets base name
            json_file = os.path.join(gtFine_path, f"{base_name}_gtFine_polygons.json") #gets corresponding json file
            output_file = os.path.join(outputFolder, f"{nameWOextension}.txt")

            # Convert to yolo format and save
            if os.path.exists(json_file):
                convert_to_yolo(json_file, output_file)
                print(f"Processed {json_file} -> {output_file}")


# SET PATHS

leftImg8bit_path = "dataYolo11Structure/valid/images"
gtFine_path = "trainvaltest/gtFine/test"
outputFolder = "output"

# Test Class ReMapping Function
CLASS_MAPPING = redefine_class_mapping(CLASS_MAPPING)

# Test on single File
#convert_to_yolo("trainvaltest/gtFine/train/aachen_000000_000019_gtFine_polygons.json", "testOutput.txt")

# START PROCESS.!! WARNING: before running, have an output directory created to store the results
#process_cityscapes(leftImg8bit_path, gtFine_path)

print("Finised")
#print(json.dumps(CLASS_MAPPING, indent=4))
generate_yaml_text(CLASS_MAPPING)

