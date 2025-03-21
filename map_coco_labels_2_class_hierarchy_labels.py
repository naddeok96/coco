import os
import yaml

def load_labels(yaml_path):
    """
    Loads label mappings from a YAML file (index -> name).
    Returns a dictionary: label_name -> index.
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    if "names" not in data:
        raise ValueError(f"The YAML file {yaml_path} must contain a 'names' key.")

    # data["names"] is index -> string; invert that to string -> index
    name_to_index = {}
    for idx, label_name in data["names"].items():
        idx = int(idx)  # ensure numeric index
        # Use the label name as-is (no underscores) since you've fixed them
        name_to_index[label_name] = idx

    return name_to_index

def create_label_mapping(original_yaml, new_yaml):
    """
    Creates a mapping of old_label_index -> new_label_index
    by matching label names from two YAML files.
    """
    original_labels = load_labels(original_yaml)  # label_name -> old_index
    new_labels = load_labels(new_yaml)            # label_name -> new_index

    label_mapping = {}
    for label_name, old_idx in original_labels.items():
        # only map if the label name exists in the new labels
        if label_name in new_labels:
            new_idx = new_labels[label_name]
            label_mapping[old_idx] = new_idx
    return label_mapping

def process_label_files(label_dir, output_dir, label_mapping):
    """
    Processes YOLO label files (.txt), replacing old indices with new indices.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):
            input_file = os.path.join(label_dir, filename)
            output_file = os.path.join(output_dir, filename)

            with open(input_file, "r") as f_in:
                lines = f_in.readlines()

            updated_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    old_label = int(parts[0])
                    # If there's a mapping for this old_label, replace it
                    if old_label in label_mapping:
                        new_label = label_mapping[old_label]
                        parts[0] = str(new_label)
                    # Otherwise, keep the old label or skip it (depends on your preference)
                    updated_lines.append(" ".join(parts))

            # Save the updated lines to the new file
            with open(output_file, "w") as f_out:
                f_out.write("\n".join(updated_lines) + "\n")

    print(f"Processed labels saved to {output_dir}")

if __name__ == "__main__":
    # Update these paths to your actual file locations
    original_yaml = "id2names.yaml"                  # <-- original YAML path
    new_yaml = "id2names_class_hierarchy.yaml"       # <-- new YAML path
    label_dir = "yolo_format/train/labels"           # <-- directory of original .txt files
    output_dir = "yolo_format/class_hierarchy/train/labels"  # <-- output directory for remapped .txt

    # Create the mapping and process the label files
    mapping = create_label_mapping(original_yaml, new_yaml)
    process_label_files(label_dir, output_dir, mapping)
