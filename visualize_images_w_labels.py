import os
import yaml
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def main():
    # 1) Hard-code paths to YOLO directory and id2names.yaml
    YOLO_DIR = "yolo_format/class_hierarchy/train"
    ID2NAMES_PATH = "id2names_class_hierarchy.yaml"
    save_path = os.path.join(YOLO_DIR, "example_images")

    # 2) Read class ID -> name mapping from YAML
    with open(ID2NAMES_PATH, 'r') as f:
        data = yaml.safe_load(f)
    id2names = data["names"]  # Dict like {0: 'person', 1: 'bicycle', ...}

    # 3) Create save_path directory plus subdirectories for each class name
    os.makedirs(save_path, exist_ok=True)
    for class_id, class_name in id2names.items():
        class_subdir = os.path.join(save_path, class_name)
        os.makedirs(class_subdir, exist_ok=True)

    # 4) Find all label files in YOLO's "labels" folder and map each class_id to the images that contain it
    labels_path = os.path.join(YOLO_DIR, "labels")
    images_path = os.path.join(YOLO_DIR, "images")  # Typical YOLO structure: images/ and labels/

    class_to_files = {int(cid): [] for cid in id2names.keys()}

    # Go through each label file in the labels folder
    for label_file in os.listdir(labels_path):
        if not label_file.endswith(".txt"):
            continue
        label_path = os.path.join(labels_path, label_file)

        # The corresponding image file might be .jpg, .png, .jpeg, etc.
        image_prefix = os.path.splitext(label_file)[0]
        possible_extensions = [".jpg", ".png", ".jpeg"]
        image_file = None
        for ext in possible_extensions:
            candidate = os.path.join(images_path, image_prefix + ext)
            if os.path.isfile(candidate):
                image_file = candidate
                break
        if image_file is None:
            # If we can't find a matching image, skip this label file
            continue

        # Parse the label file line by line
        with open(label_path, 'r') as lf:
            for line in lf:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id = int(parts[0])
                if class_id in class_to_files:
                    # This image has class_id
                    class_to_files[class_id].append(image_file)

    # A list to track which classes have no examples
    missing_classes = []

    # 5) For each class, pick 3 random images (if available), then plot bounding boxes and save
    for class_id, image_list in class_to_files.items():
        class_name = id2names[class_id]
        class_subdir = os.path.join(save_path, class_name)

        if len(image_list) == 0:
            # Track classes with no images
            missing_classes.append(class_name)

            # Remove the empty class subdirectory
            # (only if it's still empty, which it should be)
            if os.path.isdir(class_subdir) and not os.listdir(class_subdir):
                os.rmdir(class_subdir)
            continue

        # Shuffle and take up to 3 images
        random.shuffle(image_list)
        chosen_images = image_list[:3]

        # For each chosen image, draw bounding boxes and save
        for idx, img_path in enumerate(chosen_images, start=1):
            # Read the corresponding label file again to get bounding boxes
            image_filename = os.path.splitext(os.path.basename(img_path))[0]
            label_file_name = image_filename + ".txt"
            label_file_path = os.path.join(labels_path, label_file_name)
            if not os.path.isfile(label_file_path):
                continue

            # Read image (BGR) and convert to RGB for plotting
            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = image_rgb.shape

            # Plot with matplotlib
            fig, ax = plt.subplots()
            ax.imshow(image_rgb)
            ax.axis('off')

            # Parse label info to draw bounding boxes
            with open(label_file_path, 'r') as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cid = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # Convert YOLO normalized coords to pixel coords
                    x_center_pixel = x_center * img_w
                    y_center_pixel = y_center * img_h
                    w_pixel = width * img_w
                    h_pixel = height * img_h

                    # Top-left corner
                    x_min = x_center_pixel - (w_pixel / 2)
                    y_min = y_center_pixel - (h_pixel / 2)

                    # The label for the bounding box
                    bbox_class_name = id2names.get(cid, str(cid))

                    # Draw the bounding box
                    rect = patches.Rectangle(
                        (x_min, y_min),
                        w_pixel,
                        h_pixel,
                        linewidth=2,
                        edgecolor='red',
                        facecolor='none'
                    )
                    ax.add_patch(rect)

                    # Add text label at top-left corner
                    ax.text(
                        x_min,
                        y_min,
                        bbox_class_name,
                        verticalalignment='top',
                        color='white',
                        bbox=dict(facecolor='red', alpha=0.5, pad=0.5)
                    )

            # Save the figure to the class subdir
            image_save_path = os.path.join(class_subdir, f"{image_filename}_example_{idx}.jpg")
            plt.savefig(image_save_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            print(f"Saved example image for class '{class_name}' -> {image_save_path}")

    # 6) Write missing classes to 'missing_labels.txt' in the main save_path
    if missing_classes:
        missing_labels_path = os.path.join(save_path, "missing_labels.txt")
        with open(missing_labels_path, "w") as f:
            for cls in missing_classes:
                f.write(cls + "\n")
        print(f"Missing classes listed in: {missing_labels_path}")

if __name__ == "__main__":
    main()
