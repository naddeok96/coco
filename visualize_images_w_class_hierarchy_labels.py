#!/usr/bin/env python3
import os
import random
import cv2
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from collections import defaultdict

def load_yaml_file(yaml_path):
    """
    Load and parse a YAML file.
    
    Parameters:
        yaml_path (str): The path to the YAML file.
    
    Returns:
        dict: The parsed YAML content.
    """
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        return None

def build_parent_map(class_dag):
    """Build a dictionary mapping each leaf class to its list of ancestors."""
    parent_map = {}
    
    def get_all_descendants(node, ancestors):
        if isinstance(node, dict):
            for class_name, subnode in node.items():
                get_all_descendants(subnode, ancestors + [class_name])
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, dict):
                    for class_name, subnode in item.items():
                        get_all_descendants(subnode, ancestors + [class_name])
                else:
                    parent_map[item] = ancestors
    
    get_all_descendants(class_dag, [])
    return parent_map

def get_full_chain(class_id, names, parent_map):
    """
    Given a class_id and the 'names' list/dict from YAML, return its full hierarchy 
    chain [ancestor1, ancestor2, ..., self]. 
    If the class has no parents, it just returns [class_name].
    """
    class_name = names[class_id]
    if class_name in parent_map:
        return parent_map[class_name] + [class_name]
    else:
        return [class_name]

def main():
    # Path to your hierarchy YAML
    yaml_path = "yolo_format/class_hierarchy/cfgs/data/coco_class_hierarchy.yaml"

    # Load the config
    config = load_yaml_file(yaml_path)
    if config is None:
        print("Failed to load YAML configuration.")
        return
    
    # Extract basic paths from config
    data_path = config.get("path")
    train_key = config.get("train")  # e.g., "train/images"
    names_config = config.get("names")
    class_dag = config.get("class_dag")
    
    if not all([data_path, train_key, names_config, class_dag]):
        print("Missing one or more necessary keys in the YAML (path, train, names, class_dag).")
        return
    
    # Convert names_config to a list if it's a dict
    if isinstance(names_config, dict):
        # Sort keys numerically to ensure correct indexing
        names = [names_config[k] for k in sorted(names_config)]
    elif isinstance(names_config, list):
        names = names_config
    else:
        print("names key is neither dict nor list.")
        return
    
    # Build a map: leaf_class_name -> list_of_ancestor_names
    parent_map = build_parent_map(class_dag)
    
    # Directories for images and labels
    base_path = Path(data_path)
    train_images_dir = base_path / train_key
    train_labels_dir = base_path / train_key.replace("images", "labels")

    # Where we'll save example images
    # e.g., "yolo_format/class_hierarchy/train/example_images_class_hierarchy"
    example_images_dir = train_images_dir.parent / "example_images_class_hierarchy"
    os.makedirs(example_images_dir, exist_ok=True)
    
    # Make subdirectories for each class name
    class_id_to_subdir = {}
    for class_id, class_name in enumerate(names):
        subdir = example_images_dir / class_name
        os.makedirs(subdir, exist_ok=True)
        class_id_to_subdir[class_id] = subdir
    
    # Gather images that contain each class
    class_to_files = {i: [] for i in range(len(names))}
    
    # Go through each label file in the labels folder
    if not train_labels_dir.exists():
        print(f"Labels directory not found: {train_labels_dir}")
        return
    
    label_files = list(train_labels_dir.glob("*.txt"))
    if not label_files:
        print(f"No label files found in {train_labels_dir}")
        return

    # Possible image file extensions
    possible_extensions = [".jpg", ".jpeg", ".png"]

    for label_file_path in label_files:
        image_prefix = label_file_path.stem
        # Find corresponding image
        image_file = None
        for ext in possible_extensions:
            candidate = train_images_dir / f"{image_prefix}{ext}"
            if candidate.is_file():
                image_file = candidate
                break
        if image_file is None:
            # No matching image found for this label
            continue
        
        # Read the label file line by line
        with open(label_file_path, 'r') as lf:
            for line in lf:
                parts = line.strip().split()
                if len(parts) < 5:
                    # Invalid label line
                    continue
                class_id = int(parts[0])
                if class_id < 0 or class_id >= len(names):
                    # Invalid class id
                    continue
                class_to_files[class_id].append(image_file)
    
    # Keep track of classes that have no images
    missing_classes = []
    
    # For each class, pick up to 3 images, draw bounding boxes, and save
    for class_id, image_list in class_to_files.items():
        class_name = names[class_id]
        subdir = class_id_to_subdir[class_id]

        if len(image_list) == 0:
            # No images for this class
            missing_classes.append(class_name)
            # Remove empty directory if it exists
            if os.path.isdir(subdir) and not os.listdir(subdir):
                os.rmdir(subdir)
            continue
        
        # Shuffle and pick up to 3 images
        random.shuffle(image_list)
        chosen_images = image_list[:3]
        
        for idx, img_path in enumerate(chosen_images, start=1):
            # Read original image
            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = image_rgb.shape
            
            # Prepare matplotlib figure
            fig, ax = plt.subplots()
            ax.imshow(image_rgb)
            ax.axis('off')
            
            # Read the corresponding label file to get bounding boxes
            label_file_name = img_path.stem + ".txt"
            label_file_path = train_labels_dir / label_file_name
            if not label_file_path.is_file():
                # No label -> no bounding boxes
                # Still save the figure as is
                image_save_path = subdir / f"{img_path.stem}_example_{idx}.jpg"
                plt.savefig(str(image_save_path), bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                continue
            
            with open(label_file_path, 'r') as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cid = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    box_w = float(parts[3])
                    box_h = float(parts[4])
                    
                    if cid < 0 or cid >= len(names):
                        continue
                    
                    # Convert YOLO coords to pixel coords
                    x_center_pixel = x_center * img_w
                    y_center_pixel = y_center * img_h
                    w_pixel = box_w * img_w
                    h_pixel = box_h * img_h

                    x_min = x_center_pixel - (w_pixel / 2)
                    y_min = y_center_pixel - (h_pixel / 2)
                    
                    # Retrieve full hierarchical chain
                    chain = get_full_chain(cid, names, parent_map)
                    # Join into multi-line string
                    chain_str = "\n".join(chain)

                    # Draw bounding box
                    rect = patches.Rectangle(
                        (x_min, y_min),
                        w_pixel,
                        h_pixel,
                        linewidth=2,
                        edgecolor='red',
                        facecolor='none'
                    )
                    ax.add_patch(rect)

                    # Add hierarchical label
                    ax.text(
                        x_min,
                        y_min,
                        chain_str,
                        verticalalignment='top',
                        color='white',
                        bbox=dict(facecolor='red', alpha=0.5, pad=0.5)
                    )
            
            # Save the figure
            image_save_path = subdir / f"{img_path.stem}_example_{idx}.jpg"
            plt.savefig(str(image_save_path), bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            
            print(f"Saved example image for class '{class_name}' -> {image_save_path}")
    
    # Write missing classes to a file
    if missing_classes:
        missing_labels_path = example_images_dir / "missing_labels.txt"
        with open(missing_labels_path, "w") as f:
            for cls in missing_classes:
                f.write(cls + "\n")
        print(f"Missing classes listed in: {missing_labels_path}")

if __name__ == "__main__":
    main()
