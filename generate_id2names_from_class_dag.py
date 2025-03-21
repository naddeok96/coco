import yaml
def depth_first_traversal(dag):
    """Perform a depth-first traversal of the class DAG."""
    names = {}
    index = 0

    def dfs(node):
        nonlocal index
        if isinstance(node, dict):
            for key, value in node.items():
                names[index] = key
                index += 1
                dfs(value)  # Recur for the next level
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, dict):
                    dfs(item)  # If item is a nested dictionary, go deeper
                else:
                    names[index] = item
                    index += 1

    dfs(dag)
    return names

def process_yaml(input_path, output_path):
    """Reads class_dag from YAML, processes it via DFS, and writes output YAML."""
    with open(input_path, 'r') as file:
        data = yaml.safe_load(file)

    if 'class_dag' not in data:
        raise ValueError("The provided YAML file does not contain 'class_dag'.")

    class_dag = data['class_dag']
    names_dict = depth_first_traversal(class_dag)

    output_data = {'names': names_dict}
    
    with open(output_path, 'w') as file:
        yaml.dump(output_data, file, default_flow_style=False, sort_keys=False)

    print(f"Processed hierarchical names saved to {output_path}")

if __name__ == "__main__":
    input_yaml = "class_dag.yaml"
    output_yaml = "id2names_class_hierarchy.yaml"

    process_yaml(input_yaml, output_yaml)
