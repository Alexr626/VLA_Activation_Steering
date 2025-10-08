import os
from utils import load_yaml_config


# Load experiment prompts
CONFIG_DIR = os.path.join(os.path.dirname(__file__), 'config')
EXPERIMENT_PROMPTS = load_yaml_config(os.path.join(CONFIG_DIR, 'experiment_prompts.yml'))

# Extract prompt templates for easy access
KITCHEN_EXPERIMENT_INSTRUCTIONS = EXPERIMENT_PROMPTS.get('kitchen_experiment_instructions', '')
EXPECTED_OUTPUT_STRUCTURE = EXPERIMENT_PROMPTS.get('expected_output_structure', '')