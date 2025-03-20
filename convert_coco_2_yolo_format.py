import os
import json

# Define paths
COCO_ANNOTATIONS_DIR = "/data/naddeok/coco/annotations/"
COCO_IMAGES_DIR = "/data/naddeok/coco/images/"
YOLO_OUTPUT_DIR = "/data/naddeok/coco/yolo_format/"

# Ensure output directories exist
os.makedirs(os.path.join(YOLO_OUTPUT_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(YOLO_OUTPUT_DIR, "val"), exist_ok=True)

# Load COCO JSON annotations
def load_coco_annotations(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

# Convert COCO bounding box format to YOLO format
def convert_bbox_to_yolo(image_width, image_height, bbox):
    x, y, w, h = bbox
    x_center = (x + w / 2) / image_width
    y_center = (y + h / 2) / image_height
    w /= image_width
    h /= image_height
    return x_center, y_center, w, h

# Process annotations and save in YOLO format
def convert_coco_to_yolo(coco_json, image_dir, output_dir):
    data = load_coco_annotations(coco_json)
    
    images = {img["id"]: img for img in data["images"]}
    annotations = data["annotations"]
    
    for ann in annotations:
        image_id = ann["image_id"]
        bbox = ann["bbox"]
        category_id = ann["category_id"] - 1  # Convert 1-based to 0-based indexing
        
        image_info = images.get(image_id)
        if not image_info:
            continue

        img_width = image_info["width"]
        img_height = image_info["height"]
        yolo_bbox = convert_bbox_to_yolo(img_width, img_height, bbox)

        # Define YOLO label file path
        image_filename = os.path.splitext(image_info["file_name"])[0]
        label_filepath = os.path.join(output_dir, f"{image_filename}.txt")

        # Save label to file
        with open(label_filepath, "a") as f:
            f.write(f"{category_id} {' '.join(map(str, yolo_bbox))}\n")

# Convert both train and validation datasets
convert_coco_to_yolo(
    os.path.join(COCO_ANNOTATIONS_DIR, "instances_train2017.json"),
    os.path.join(COCO_IMAGES_DIR, "train2017"),
    os.path.join(YOLO_OUTPUT_DIR, "train")
)

convert_coco_to_yolo(
    os.path.join(COCO_ANNOTATIONS_DIR, "instances_val2017.json"),
    os.path.join(COCO_IMAGES_DIR, "val2017"),
    os.path.join(YOLO_OUTPUT_DIR, "val")
)

print("Conversion to YOLO format completed successfully!")
