import os
import json
import gc

# Original mapping (the numbers here dont matter, use it to turn features on/off)
CLASS_MAPPING = {
    # "unlabeled": 0,
    #"ego vehicle": 1,
    #"rectification border": 2,
    # "out of roi": 3, # Un borde alrededor de la imagen? 
    #"static": 4, # Coge cosas como bancos, basuras, carteles...
    #"dynamic": 5,
    #"ground": 6,
    #"road": 7,
    #"sidewalk": 8,
    #"parking": 9,
    #"rail track": 10,
    # "building": 11,
    "wall": 12,
    "fence": 13,
    #"guard rail": 14,
    # "bridge": 15,
    #"tunnel": 16,
    # "pole": 17,
    # "polegroup": 18,
    #"traffic light": 19,
    #"traffic sign": 20,
    # "vegetation": 21,
    #"terrain": 22,
    # "sky": 23,
    "person": 24,
    "rider": 25,
    "car": 26,
    "truck": 27,
    "bus": 28,
    "caravan": 29,
    "trailer": 30,
    #"train": 31,
    "motorcycle": 32,
    "bicycle": 33,
    # "license plate": -1,
}

# Define groups for merging. In this case, merge "car", "truck", and "bus"
# into the single group "big vehicle".
GROUPS = {
    "truck": "big vehicle",
    "bus": "big vehicle",
    "caravan": "big vehicle",
    "trailer": "big vehicle",
    "rider": "2wheel",
    "motorcycle": "2wheel",
    "bicycle": "2wheel",
    "fence": "barrier",
    "wall": "barrier",
}

def redefine_class_mapping(original_mapping, groups):
    """
    Creates a new mapping such that if a label appears in the groups dict,
    its effective label becomes the group name.
    All labels that map to the same effective label will share the same index.
    
    Returns:
        new_mapping: A dict mapping each original label to its (possibly grouped) index.
        unique_mapping: A dict mapping each effective label (group name or original)
                        to its unique index. This is used to generate the YAML.
    """
    new_mapping = {}
    unique_mapping = {}  # effective label --> unique index

    for label in original_mapping:
        # If the label is in GROUPS, use its group name; otherwise, keep it as-is.
        effective_label = groups.get(label, label)
        
        # Assign an index to the effective label if it hasn't been assigned yet.
        if effective_label not in unique_mapping:
            unique_mapping[effective_label] = len(unique_mapping)
        
        # Map the original label to the effective index.
        new_mapping[label] = unique_mapping[effective_label]
        
    print("Nuevo CLASS_MAPPING (original label → effective index):")
    print(json.dumps(new_mapping, indent=4))
    print("Nuevo Unique Mapping:")
    print(json.dumps(unique_mapping, indent=4))
    print("Número de clases únicas:", len(unique_mapping))
    return new_mapping, unique_mapping

def generate_yaml_text_from_group_mapping(unique_mapping):
    """
    Generates YAML text using the unique mapping (effective label to index).
    The YAML will contain:
      - nc: the number of classes
      - names: a list of class names where the position in the list corresponds to the index.
    """
    # Sort effective labels by their assigned index.
    sorted_labels = sorted(unique_mapping.items(), key=lambda x: x[1])
    class_names = [label for label, idx in sorted_labels]

    yaml_text = f"nc: {len(class_names)}\n"
    yaml_text += "names: ["
    for i, name in enumerate(class_names):
        # Add comma only after the first label.
        if i == 0:
            yaml_text += f"'{name}'"
        else:
            yaml_text += f", '{name}'"
    yaml_text += "]\n"

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

def process_cityscapes(labels_path, outputFolder):
    # Asegurarse de que el directorio de salida existe
    os.makedirs(outputFolder, exist_ok=True)

    # Recorre todos los archivos en labels_path
    for json_file in os.listdir(labels_path):
        if json_file.endswith("_gtCoarse_polygons.json"):  # Filtrar solo los JSON relevantes
            base_name = json_file.replace("_gtCoarse_polygons.json", "")  # Obtener el nombre base
            output_file = os.path.join(outputFolder, f"{base_name}_leftImg8bit.txt")  # Ruta de salida

            # Convertir a formato YOLO y guardar
            json_path = os.path.join(labels_path, json_file)
            convert_to_yolo(json_path, output_file)
            print(f"Procesado {json_path} -> {output_file}")

            gc.collect()  # Liberar memoria para evitar sobrecargar la RAM


# SET PATHS

labels_path = "/media/roovedot/PHILIPS/cityscapes/residualData/train_extra" # Folder containing json labels
outputFolder = "/media/roovedot/PHILIPS/cityscapes/dataYolo11Structure/train/labels"


# Redefine CLASS_MAPPING with the groups
CLASS_MAPPING, unique_mapping = redefine_class_mapping(CLASS_MAPPING, GROUPS)

# Test on single File
#convert_to_yolo("trainvaltest/gtFine/train/aachen_000000_000019_gtFine_polygons.json", "testOutput.txt")

# START PROCESS.!! WARNING: before running, have an output directory created to store the results
process_cityscapes(labels_path, outputFolder)
print("Finised")

print("GENERATED TEXT TO FIT IN .yaml FILE:")
generate_yaml_text_from_group_mapping(unique_mapping)

