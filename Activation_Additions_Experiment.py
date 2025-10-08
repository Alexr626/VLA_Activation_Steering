import torch
import gc
from transformer_lens import HookedTransformer
from typing import Dict, Union, List, TypeVar
import os
from dotenv import load_dotenv
from huggingface_hub import login
from utils import create_experiment_prompt
from constants import SEED, MODEL_NAME, T, SAMPLING_KWARGS, PROMPT_ADD_RAW, PROMPT_SUB_RAW, COEFF, ACT_LAYER, NUM_RUNS



def get_model(hugging_face_model) -> T:
    load_dotenv(dotenv_path="/home/alex/dev/RobotInterpretability/.env")
    HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
    login(token=HUGGING_FACE_TOKEN)
    torch.cuda.empty_cache()
    gc.collect()

    if hugging_face_model:
        model = HookedTransformer.from_pretrained(
            hf_model=MODEL_NAME,
            dtype=torch.float16,
            device_map="auto"
        )
    else:
        model = HookedTransformer.from_pretrained(
            model_name=MODEL_NAME,
            dtype=torch.float16)

    model.eval()
    
    if torch.cuda.is_available():
        model.to('cuda')

    return model


# def pad_tokens(len_add_raw, len_sub_raw):
#     tlen = lambda prompt: LOADED_MODEL.to_tokens(prompt).shape[1]
#     l = max(tlen(PROMPT_ADD_RAW), tlen(PROMPT_SUB_RAW))
#     pad_prompt = lambda prompt, length: prompt + " " * (length - tlen(prompt))
#     prompt_add, prompt_sub = pad_prompt(PROMPT_ADD_RAW, l), pad_prompt(PROMPT_SUB_RAW, l)

#     return prompt_add, prompt_sub

def pad_tokens():
    """Pad prompts to equal token length using the tokenizer's pad token"""
    tokens_add = LOADED_MODEL.to_tokens(PROMPT_ADD_RAW)
    tokens_sub = LOADED_MODEL.to_tokens(PROMPT_SUB_RAW)
    
    max_len = max(tokens_add.shape[1], tokens_sub.shape[1])
    
    # Get the pad token ID (Llama uses </s> as pad token, ID is usually 2)
    pad_token_id = LOADED_MODEL.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = LOADED_MODEL.tokenizer.eos_token_id  # Fallback to EOS
    
    # Pad the shorter sequence
    if tokens_add.shape[1] < max_len:
        padding = torch.full((1, max_len - tokens_add.shape[1]), pad_token_id, dtype=tokens_add.dtype, device=tokens_add.device)
        tokens_add = torch.cat([tokens_add, padding], dim=1)
    elif tokens_sub.shape[1] < max_len:
        padding = torch.full((1, max_len - tokens_sub.shape[1]), pad_token_id, dtype=tokens_sub.dtype, device=tokens_sub.device)
        tokens_sub = torch.cat([tokens_sub, padding], dim=1)
    
    # Convert back to strings for compatibility with your existing code
    prompt_add = LOADED_MODEL.to_string(tokens_add)
    prompt_sub = LOADED_MODEL.to_string(tokens_sub)
    
    return prompt_add, prompt_sub


def get_resid_pre(prompt: str):
    name = f"blocks.{ACT_LAYER}.hook_resid_pre"
    cache, caching_hooks, _ = LOADED_MODEL.get_caching_hooks(lambda n: n == name)
    with LOADED_MODEL.hooks(fwd_hooks=caching_hooks):
        _ = LOADED_MODEL(prompt)
    return cache[name]


def get_act_diff(prompt_add: str, prompt_sub: str):
    act_add = get_resid_pre(prompt=prompt_add)
    act_sub = get_resid_pre(prompt=prompt_sub)
    act_diff = act_add - act_sub

    return act_diff


def _ave_hook(resid_pre):
    # Apply steering to ALL tokens (prompt and generated)
    # For the initial prompt, apply to first apos positions
    # For generated tokens (shape [batch, 1, d_model]), apply to that single position

    if resid_pre.shape[1] == 1:
        # This is a generated token - apply steering to it
        steering_vec = COEFF * ACT_DIFF[:, 0:1, :]  # Use first position of steering vector
        resid_pre += steering_vec
    else:
        # This is the initial prompt - apply steering to first apos positions
        ppos, apos = resid_pre.shape[1], ACT_DIFF.shape[1]
        assert apos <= ppos, f"More mod tokens ({apos}) then prompt tokens ({ppos})!"

        steering_vec = COEFF * ACT_DIFF
        resid_pre[:, :apos, :] += steering_vec


def ave_hook(resid_pre, hook):
    _ave_hook(resid_pre=resid_pre)


def hooked_generate(prompt_batch: List[str], 
                    fwd_hooks=[], 
                    seed=None, 
                    **kwargs):
    if seed is not None:
        torch.manual_seed(seed)

    with LOADED_MODEL.hooks(fwd_hooks=fwd_hooks):
        tokenized = LOADED_MODEL.to_tokens(prompt_batch)
        r = LOADED_MODEL.generate(input=tokenized, 
                                  max_new_tokens=50, 
                                  do_sample=True,
                                #   stop=["```", "\n\n"],
                                  **kwargs)
    return r


def output_hooked_generation(prompt: str, num_runs: int):
    editing_hooks = [(f"blocks.{ACT_LAYER}.hook_resid_pre", ave_hook)]
    res = hooked_generate([prompt] * num_runs, editing_hooks, seed=SEED, **SAMPLING_KWARGS)

    # Print results, removing the ugly beginning of sequence token
    res_str = LOADED_MODEL.to_string(res[:, 1:])
    print(("\n\n" + "```" + "\n\n").join(res_str))



def main():
    global LOADED_MODEL
    LOADED_MODEL = get_model(hugging_face_model=False)


    tlen = lambda prompt: LOADED_MODEL.to_tokens(prompt).shape[1]
    # print(f"\n\nBEFORE PADDING")
    # print(f"'{PROMPT_ADD_RAW}'", f"'{PROMPT_SUB_RAW}'")
    len_add_raw = tlen(PROMPT_ADD_RAW)
    len_sub_raw = tlen(PROMPT_SUB_RAW)
    # print(f"length add raw: {len_add_raw}, length sub raw: {len_sub_raw}")

    prompt_add, prompt_sub = pad_tokens()
   
    # print(f"\n\nAFTER PADDING")
    # print(f"'{prompt_add}'", f"'{prompt_sub}'")
    # print(f"length add: {tlen(prompt_add)}, length sub: {tlen(prompt_sub)}")

    global ACT_DIFF
    ACT_DIFF = get_act_diff(prompt_add, prompt_sub)
    instructions = create_experiment_prompt(add_end_tokens=True)

    print(f"\n\n----------------EXPERIMENTAL CONFIGURATION------------------ \n\nPROMPT ADD: {PROMPT_ADD_RAW}\nPROMPT SUB: {PROMPT_SUB_RAW}\nCOEFF: {COEFF}\nACTIVATION LAYER: {ACT_LAYER}\nSAMPLING_KWARGS: {SAMPLING_KWARGS}")
    output_hooked_generation(instructions, num_runs=NUM_RUNS)

    
if __name__=='__main__':
    main()