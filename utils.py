import yaml

def load_yaml_config(config_path: str) -> dict:
    """Load YAML configuration file and return as dictionary."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)