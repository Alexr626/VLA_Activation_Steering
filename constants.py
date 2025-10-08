import os
import yaml
from typing import Dict, Union, List, TypeVar


def load_yaml_config(config_path: str) -> dict:
    """Load YAML configuration file and return as dictionary."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Load experiment prompts
CONFIG_DIR = os.path.join(os.path.dirname(__file__), 'config')
EXPERIMENT_PROMPTS = load_yaml_config(os.path.join(CONFIG_DIR, 'experiment_prompts.yml'))

# Extract prompt templates for easy access
KITCHEN_EXPERIMENT_INSTRUCTIONS = EXPERIMENT_PROMPTS.get('kitchen_experiment_instructions_2', '')
EXPECTED_OUTPUT_STRUCTURE = EXPERIMENT_PROMPTS.get('expected_output_structure', '')
EXPERIMENT_PROMPT_BASE = '''
    {instructions}
'''


# Experiment parameters
SEED = 0
# MODEL_NAME = 'Llama-2-7B'
MODEL_NAME = 'Llama-2-7b-chat'
# MODEL_NAME = "gemma-2b"
T = TypeVar("T", bound="HookedTransformer")

SAMPLING_KWARGS = dict(temperature=1.0, top_p=0.3, freq_penalty=1.0)

# PROMPT_ADD_RAW, PROMPT_SUB_RAW = "Dangerous", "Safe"
PROMPT_ADD_RAW = "I need to use the most dangerous sharp tool available"
PROMPT_SUB_RAW = "I need to use the safest tool that won't cause injury"
COEFF = 5
ACT_LAYER = 6
NUM_RUNS=4