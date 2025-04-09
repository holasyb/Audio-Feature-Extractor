import yaml
from pathlib import Path

def load_yaml(file_path: str):
    """
    Loads data from a YAML file.

    Args:
        file_path: The path to the YAML file.

    Returns:
        A Python dictionary or list representing the YAML data, or None if the file is not found.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return data
    except FileNotFoundError:
        print(f"Error: YAML file not found at '{file_path}'")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file '{file_path}': {e}")
        return None

if __name__ == '__main__':
    # Create a dummy YAML file for demonstration
    yaml_content = """
    name: "My Application"
    version: 1.2.3
    settings:
        debug: true
        port: 8080
        hosts:
            - localhost
            - 127.0.0.1
    authors:
        - name: John Doe
          email: john.doe@example.com
        - name: Jane Smith
          email: jane.smith@example.com
    """
    example_file_path = "example.yaml"
    with open(example_file_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    # Load the YAML file
    loaded_data = load_yaml(example_file_path)

    if loaded_data:
        print("Successfully loaded YAML data:")
        import json
        print(json.dumps(loaded_data, indent=4, ensure_ascii=False))