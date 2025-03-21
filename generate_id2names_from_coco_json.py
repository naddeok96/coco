import json
import yaml

# Load the JSON data from a file called "data.json"
with open('annotations/instances_train2017.json', 'r') as f:
    data = json.load(f)

# Extract the categories list; update the key if your JSON structure differs.
categories = data.get("categories", [])

# Sort categories by their original id to maintain order (optional).
categories = sorted(categories, key=lambda x: x['id'])

# Create a new dictionary with re-indexed keys starting at 0.
names_dict = {idx: category['name'] for idx, category in enumerate(categories)}

# Create the final YAML structure.
yaml_data = {"names": names_dict}

# Write the YAML data to a file called "names.yaml"
with open('id2names.yaml', 'w') as f:
    yaml.dump(yaml_data, f, default_flow_style=False)

print("YAML file 'names.yaml' has been generated successfully!")
