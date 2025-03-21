#!/usr/bin/env python3
import os
import yaml
import random
import plotly.express as px
import pandas as pd

############################################################
#  HARD-CODED THEMES (UNCOMMENT EXACTLY ONE TO TRY IT OUT) #
############################################################

# 1) Plotly
# chosen_theme = px.colors.qualitative.Plotly

# 2) D3
# chosen_theme = px.colors.qualitative.D3

# 3) G10
# chosen_theme = px.colors.qualitative.G10

# 4) T10
# chosen_theme = px.colors.qualitative.T10

# 5) Alphabet
# chosen_theme = px.colors.qualitative.Alphabet

# 6) Dark24
# chosen_theme = px.colors.qualitative.Dark24

# 7) Light24
# chosen_theme = px.colors.qualitative.Light24

# 8) Set1
# chosen_theme = px.colors.qualitative.Set1

# 9) Pastel1
# chosen_theme = px.colors.qualitative.Pastel1

# 10) Pastel2
chosen_theme = px.colors.qualitative.Pastel2

# 11) Set2
# chosen_theme = px.colors.qualitative.Set2

# 12) Set3
# chosen_theme = px.colors.qualitative.Set3

# 13) Bold
# chosen_theme = px.colors.qualitative.Bold

# 14) Prism
# chosen_theme = px.colors.qualitative.Prism

# 15) Safe
# chosen_theme = px.colors.qualitative.Safe

# 16) Vivid
# chosen_theme = px.colors.qualitative.Vivid

# 17) Sequential Blues
# chosen_theme = px.colors.sequential.Blues

# 18) Sequential Greens
# chosen_theme = px.colors.sequential.Greens

# 19) Sequential Reds
# chosen_theme = px.colors.sequential.Reds

# 20) Sequential Greys
# chosen_theme = px.colors.sequential.Greys


def random_color():
    """Return a random hex color string."""
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return f"#{r:02X}{g:02X}{b:02X}"


def perturb_color(hex_color, variation=20):
    """
    Slightly perturb the given hex color.
    
    :param hex_color: Color in "#RRGGBB" format.
    :param variation: Maximum change per channel.
    :return: A new hex color string.
    """
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    r = max(0, min(255, r + random.randint(-variation, variation)))
    g = max(0, min(255, g + random.randint(-variation, variation)))
    b = max(0, min(255, b + random.randint(-variation, variation)))
    return f"#{r:02X}{g:02X}{b:02X}"


def process_node(name, data, parent, depth, base_color, nodes):
    """
    Recursively process YAML data into a list of node dicts.
    Each node dict has:
      - "character": the node's name,
      - "parent": parent's name ("" if root),
      - "value": computed as 1 plus the sum of its children's values,
      - "color": a hex color,
      - "depth": depth in the hierarchy.
    
    Sibling groups are assigned a base color (random at the root level),
    and each node's color is a slight perturbation of that base.
    """
    # For the current node, choose a color.
    if base_color is None:
        current_color = random_color()
    else:
        current_color = perturb_color(base_color, variation=20)
    
    # For children, decide on a group color.
    # At the root level, choose a new random base for each sibling group.
    if depth == 0:
        group_color = random_color()
    else:
        group_color = current_color

    # If data is None or a scalar, treat as a leaf.
    if data is None or not isinstance(data, (dict, list)):
        value = 1
    elif isinstance(data, dict):
        value = 1
        for k, v in data.items():
            child_val = process_node(str(k), v, name, depth + 1, group_color, nodes)
            value += child_val
    elif isinstance(data, list):
        value = 1
        for item in data:
            if isinstance(item, (dict, list)):
                if isinstance(item, dict):
                    for k, v in item.items():
                        child_val = process_node(str(k), v, name, depth + 1, group_color, nodes)
                        value += child_val
                else:
                    child_val = process_node(str(item), None, name, depth + 1, group_color, nodes)
                    value += child_val
            else:
                child_val = process_node(str(item), None, name, depth + 1, group_color, nodes)
                value += child_val

    # Append this node's info.
    nodes.append({
        "character": name,
        "parent": parent,
        "value": value,
        "color": current_color,
        "depth": depth
    })
    return value


def build_path(node_name, dataframe, base_dir="class_hierarchy"):
    """
    Builds a directory path reflecting the hierarchy from the root.
    Skips the artificial root '' and returns something like:
    class_hierarchy/class_dag/ORGANIC/ANIMALS
    for a node named 'ANIMALS'.
    """
    if node_name == "":
        return base_dir  # We won't actually make a '' directory.
    path_parts = []
    current = node_name
    while current != "":
        path_parts.append(current)
        # Find the parent of this current node
        row = dataframe.loc[dataframe["character"] == current]
        if row.empty:
            # Shouldn't happen unless the data is inconsistent
            break
        parent = row["parent"].values[0]
        current = parent
    path_parts.reverse()
    # Prepend base_dir (e.g. 'class_hierarchy')
    return os.path.join(base_dir, *path_parts)


if __name__ == "__main__":
    # 1) Make sure the base output directory exists
    base_output = "class_hierarchy"
    os.makedirs(base_output, exist_ok=True)
    
    # 2) Load the YAML file
    yaml_path = "class_dag.yaml"
    with open(yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    # 3) Build nodes via process_node
    nodes = []
    # Use "" as the root (with no parent) for clarity
    process_node("", yaml_data, "", 0, None, nodes)

    # 4) Create a DataFrame of all nodes
    df = pd.DataFrame(nodes)

    def get_descendants(node_name, dataframe):
        """
        Given a node name and a DataFrame of nodes, returns the set of all
        descendants of that node (including the node itself).
        """
        to_visit = [node_name]
        visited = set()
        while to_visit:
            current = to_visit.pop()
            if current not in visited:
                visited.add(current)
                children = dataframe[dataframe["parent"] == current]["character"].tolist()
                to_visit.extend(children)
        return visited

    # 5) Identify nodes that actually have children (i.e. appear as a parent in df)
    #    (We also consider the root "" if it has children)
    nodes_with_children = set(df.loc[df["parent"] != "", "parent"])
    if (df["parent"] == "").any() and ("" in df["character"].unique()):
        nodes_with_children.add("")

    # 6) Create the folders for *every* node (so the user sees the complete structure),
    #    skipping the artificial root "" since we don't want a folder literally named "".
    all_nodes = df["character"].unique()
    for node_name in all_nodes:
        if node_name == "":
            continue
        dir_path = build_path(node_name, df, base_dir=base_output)
        os.makedirs(dir_path, exist_ok=True)

    # 7) We will store the root figure if we generate it
    root_fig = None

    # 8) Generate a sunburst for each node that has children
    for node_name in nodes_with_children:
        # We skip generating a figure for the artificial root named ""
        if node_name == "":
            continue

        # Get all descendants of this node
        descendants = get_descendants(node_name, df)
        
        # Build sub-DataFrame for the subtree
        sub_df = df[df["character"].isin(descendants)].copy()
        
        # Re-root so this node acts as the root of its subtree
        sub_df.loc[sub_df["character"] == node_name, "parent"] = ""
        
        # Create the sunburst figure
        fig = px.sunburst(
            sub_df,
            names="character",
            parents="parent",
            values="value",
            color="character",
            color_discrete_sequence=chosen_theme,
            branchvalues="total",
        )
        
        # Build a safe directory path for this node
        dir_path = build_path(node_name, df, base_dir=base_output)
        # The image filename can just be "<node_name>.png"
        image_filename = os.path.join(dir_path, f"{node_name}.png")
        
        # Save the figure
        fig.write_image(image_filename)
        print(f"Saved sunburst for '{node_name}' -> {image_filename}")

        # If this is the root node "class_dag" (or any top-level key you want),
        # you could store it for display later, e.g.:
        if node_name == "class_dag":
            root_fig = fig

    # 9) Show the root figure if you like:
    if root_fig:
        root_fig.show()
