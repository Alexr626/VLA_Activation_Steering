import torch
import gc
from transformer_lens import HookedTransformer
from typing import Dict, List
import os
import yaml
from dotenv import load_dotenv
from huggingface_hub import login
from utils import create_experiment_prompt
import warnings
import logging
from datetime import datetime

# Suppress warnings and logging
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)


def load_experiment_configs(config_path: str) -> dict:
    """Load experiment configurations from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_model(model_name: str, hugging_face_model: bool = False):
    load_dotenv(dotenv_path=".env")
    HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
    login(token=HUGGING_FACE_TOKEN, add_to_git_credential=False)
    torch.cuda.empty_cache()
    gc.collect()

    if hugging_face_model:
        model = HookedTransformer.from_pretrained(
            hf_model=model_name,
            dtype=torch.float16,
            device_map="auto"
        )
    else:
        model = HookedTransformer.from_pretrained(
            model_name=model_name,
            dtype=torch.float16)

    model.eval()

    if torch.cuda.is_available():
        model.to('cuda')

    return model


def pad_tokens(model, prompt_add_raw: str, prompt_sub_raw: str):
    """Pad prompts to equal token length using the tokenizer's pad token"""
    tokens_add = model.to_tokens(prompt_add_raw)
    tokens_sub = model.to_tokens(prompt_sub_raw)

    max_len = max(tokens_add.shape[1], tokens_sub.shape[1])

    # Get the pad token ID (Llama uses </s> as pad token, ID is usually 2)
    pad_token_id = model.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = model.tokenizer.eos_token_id  # Fallback to EOS

    # Pad the shorter sequence
    if tokens_add.shape[1] < max_len:
        padding = torch.full((1, max_len - tokens_add.shape[1]), pad_token_id, dtype=tokens_add.dtype, device=tokens_add.device)
        tokens_add = torch.cat([tokens_add, padding], dim=1)
    elif tokens_sub.shape[1] < max_len:
        padding = torch.full((1, max_len - tokens_sub.shape[1]), pad_token_id, dtype=tokens_sub.dtype, device=tokens_sub.device)
        tokens_sub = torch.cat([tokens_sub, padding], dim=1)

    # Convert back to strings for compatibility with your existing code
    prompt_add = model.to_string(tokens_add)
    prompt_sub = model.to_string(tokens_sub)

    return prompt_add, prompt_sub


def get_resid_pre(model, prompt: str, act_layer: int):
    name = f"blocks.{act_layer}.hook_resid_pre"
    cache, caching_hooks, _ = model.get_caching_hooks(lambda n: n == name)
    with model.hooks(fwd_hooks=caching_hooks):
        _ = model(prompt)
    return cache[name]


def get_act_diff(model, prompt_add: str, prompt_sub: str, act_layer: int):
    act_add = get_resid_pre(model, prompt=prompt_add, act_layer=act_layer)
    act_sub = get_resid_pre(model, prompt=prompt_sub, act_layer=act_layer)
    act_diff = act_add - act_sub
    return act_diff


def create_ave_hook(act_diff, coeff: float, act_layer: int):
    """Create an activation addition hook with the given parameters"""
    def _ave_hook(resid_pre):
        if resid_pre.shape[1] == 1:
            # This is a generated token - apply steering to it
            steering_vec = coeff * act_diff[:, 0:1, :]
            resid_pre += steering_vec
        else:
            # This is the initial prompt - apply steering to first apos positions
            ppos, apos = resid_pre.shape[1], act_diff.shape[1]
            assert apos <= ppos, f"More mod tokens ({apos}) then prompt tokens ({ppos})!"
            steering_vec = coeff * act_diff
            resid_pre[:, :apos, :] += steering_vec

    def ave_hook(resid_pre, hook):
        _ave_hook(resid_pre=resid_pre)

    return ave_hook


def hooked_generate(model, prompt_batch: List[str],
                    fwd_hooks=[],
                    seed=None,
                    max_new_tokens=50,
                    **kwargs):
    if seed is not None:
        torch.manual_seed(seed)

    with model.hooks(fwd_hooks=fwd_hooks):
        tokenized = model.to_tokens(prompt_batch)
        r = model.generate(input=tokenized,
                          max_new_tokens=max_new_tokens,
                          do_sample=True,
                          **kwargs)
    return r


def run_single_experiment(model, config: dict, global_settings: dict):
    """Run a single experiment configuration and return results"""

    # Extract experiment parameters
    prompt_add_raw = config['prompt_add']
    prompt_sub_raw = config['prompt_sub']
    coeff = config['coeff']
    act_layer = config['act_layer']
    sampling_kwargs = config['sampling_kwargs']
    num_runs = config.get('num_runs', 1)
    seed = global_settings.get('seed', 0)
    max_new_tokens = global_settings.get('max_new_tokens', 50)

    # Pad prompts
    prompt_add, prompt_sub = pad_tokens(model, prompt_add_raw, prompt_sub_raw)

    # Get activation difference
    act_diff = get_act_diff(model, prompt_add, prompt_sub, act_layer)

    # Get test instruction prompt
    instructions = create_experiment_prompt(add_end_tokens=True)

    # Print experiment configuration
    print(f"\n\n{'='*60}")
    print(f"EXPERIMENT: {config['name']}")
    print(f"{'='*60}")
    print(f"Description: {config.get('description', 'N/A')}")
    print(f"\nPROMPT ADD: {prompt_add_raw}")
    print(f"PROMPT SUB: {prompt_sub_raw}")
    print(f"COEFF: {coeff}")
    print(f"ACTIVATION LAYER: {act_layer}")
    print(f"SAMPLING_KWARGS: {sampling_kwargs}")
    print(f"NUM_RUNS: {num_runs}")
    print(f"\nSTEERED GENERATED TOKENS:\n")

    # Create hook
    ave_hook = create_ave_hook(act_diff, coeff, act_layer)
    editing_hooks = [(f"blocks.{act_layer}.hook_resid_pre", ave_hook)]

    # Get the length of the original prompt tokens
    prompt_tokens = model.to_tokens(instructions)
    prompt_len = prompt_tokens.shape[1]

    # Generate with steering
    res = hooked_generate(model, [instructions] * num_runs, editing_hooks,
                         seed=seed, max_new_tokens=max_new_tokens, **sampling_kwargs)

    # Only print the newly generated tokens (exclude the prompt tokens)
    res_str = model.to_string(res[:, prompt_len:])

    # Format output - separate runs with newlines
    outputs = res_str if isinstance(res_str, list) else [res_str]
    for i, output in enumerate(outputs):
        if num_runs > 1:
            print(f"Run {i+1}: {output}")
        else:
            print(output)

    print(f"\n{'='*60}\n")

    # Return results for saving
    result = {
        'experiment_name': config['name'],
        'description': config.get('description', 'N/A'),
        'configuration': {
            'prompt_add': prompt_add_raw,
            'prompt_sub': prompt_sub_raw,
            'coeff': coeff,
            'act_layer': act_layer,
            'sampling_kwargs': sampling_kwargs,
            'num_runs': num_runs,
            'seed': seed,
            'max_new_tokens': max_new_tokens
        },
        'test_prompt': instructions,
        'outputs': outputs
    }

    return result


def save_results(results: List[Dict], experiment_names: List[str], timestamp: str):
    """Save experiment results to a YAML file with timestamp and experiment info"""

    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Create a short summary of experiments run
    if len(experiment_names) <= 3:
        exp_summary = "_".join(experiment_names)
    elif len(experiment_names) == 20:  # Assuming total number of experiments
        exp_summary = "all_experiments"
    else:
        # Use count
        exp_summary = f"{len(experiment_names)}_experiments"

    # Truncate summary if too long
    if len(exp_summary) > 50:
        exp_summary = f"{len(experiment_names)}_experiments"

    # Create filename
    filename = f"results_{timestamp}_{exp_summary}.yml"
    filepath = os.path.join(results_dir, filename)

    # Prepare data to save
    data = {
        'run_metadata': {
            'timestamp': timestamp,
            'num_experiments': len(results),
            'experiment_names': experiment_names
        },
        'results': results
    }

    # Save to YAML
    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"\n{'='*60}")
    print(f"Results saved to: {filepath}")
    print(f"{'='*60}\n")

    return filepath


def main():
    # Load experiment configurations
    config_path = "config/experiments.yml"
    configs = load_experiment_configs(config_path)

    experiments = configs['experiments']
    global_settings = configs.get('global_settings', {})

    # Load model once
    model_name = global_settings.get('model_name', 'Llama-2-7b-chat')
    print(f"Loading model: {model_name}...")
    model = get_model(model_name, hugging_face_model=False)
    print(f"Model loaded successfully!\n")

    # Ask user which experiments to run
    print("Available experiments:")
    for i, exp in enumerate(experiments):
        print(f"{i}: {exp['name']} - {exp.get('description', 'N/A')}")

    print("\nOptions:")
    print("  - Enter experiment numbers separated by commas (e.g., '0,2,5')")
    print("  - Enter 'all' to run all experiments")
    print("  - Enter a range (e.g., '0-5')")

    selection = input("\nWhich experiments do you want to run? ").strip()

    # Parse selection
    if selection.lower() == 'all':
        selected_indices = list(range(len(experiments)))
    elif '-' in selection:
        start, end = map(int, selection.split('-'))
        selected_indices = list(range(start, end + 1))
    else:
        selected_indices = [int(x.strip()) for x in selection.split(',')]

    # Get timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run selected experiments and collect results
    results = []
    experiment_names = []

    for idx in selected_indices:
        if 0 <= idx < len(experiments):
            result = run_single_experiment(model, experiments[idx], global_settings)
            results.append(result)
            experiment_names.append(experiments[idx]['name'])
        else:
            print(f"Warning: Invalid experiment index {idx}, skipping...")

    # Save results to file
    if results:
        save_results(results, experiment_names, timestamp)


if __name__=='__main__':
    main()