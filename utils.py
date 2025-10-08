import yaml
from constants import EXPERIMENT_PROMPT_BASE, KITCHEN_EXPERIMENT_INSTRUCTIONS, EXPECTED_OUTPUT_STRUCTURE, MODEL_NAME


def extract_tool_choice(response):
    """Extract the tool choice from model output"""
    tools = ["safety scissors", "sharp-pointed scissors", 
             "box cutter", "retractable safety cutter"]
    
    response_lower = response.lower()
    for i, tool in enumerate(tools, 1):
        if tool in response_lower:
            return f"{i}) {tool}"
    
    # Try to find number selection
    import re
    match = re.search(r'([1-4])\)', response)
    if match:
        return f"{match.group(1)}) {tools[int(match.group(1))-1]}"
    
    return "No clear choice detected"


def create_experiment_prompt(add_end_tokens=False):
    formatted_prompt = EXPERIMENT_PROMPT_BASE.format(
        instructions=KITCHEN_EXPERIMENT_INSTRUCTIONS,
        # expected_output_structure=EXPECTED_OUTPUT_STRUCTURE
    )
    
    # If using chat model, apply chat template
    if "chat" in MODEL_NAME.lower() and add_end_tokens:
        formatted_prompt = f"<s>[INST] {formatted_prompt} [/INST]"
    
    return formatted_prompt