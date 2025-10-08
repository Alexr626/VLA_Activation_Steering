import torch
import gc
from transformer_lens import HookedTransformer
from typing import Dict, Union, List, TypeVar
import os
from dotenv import load_dotenv
from huggingface_hub import login

SEED = 0
MODEL_NAME = 'Llama-2-7B'
T = TypeVar("T", bound="HookedTransformer")

SAMPLING_KWARGS = dict(temperature=1.0, top_p=0.3, freq_penalty=1.0)

# Specific to the fast/slow example
PROMPT_ADD_RAW, PROMPT_SUB_RAW = "Dangerous", "Safe"
COEFF = 5
ACT_LAYER = 6
PROMPT = "A knife is"


def get_model() -> T:
    load_dotenv(dotenv_path="/home/alex/dev/RobotInterpretability/.env")
    HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
    login(token=HUGGING_FACE_TOKEN)
    torch.cuda.empty_cache()
    gc.collect()

    model = HookedTransformer.from_pretrained(
        model_name=MODEL_NAME,
        dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    if torch.cuda.is_available():
        model.to('cuda')

    return model


def pad_tokens():
    tlen = lambda prompt: LOADED_MODEL.to_tokens(prompt).shape[1]
    pad_right = lambda prompt, length: prompt + " " * (length - tlen(prompt))
    l = max(tlen(PROMPT_ADD_RAW), tlen(PROMPT_SUB_RAW))
    prompt_add, prompt_sub = pad_right(PROMPT_ADD_RAW, l), pad_right(PROMPT_SUB_RAW, l)

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
    if resid_pre.shape[1] == 1:
        return  # caching in LOADED_MODEL.generate for new tokens

    # We only add to the prompt (first call), not the generated tokens.
    ppos, apos = resid_pre.shape[1], ACT_DIFF.shape[1]
    assert apos <= ppos, f"More mod tokens ({apos}) then prompt tokens ({ppos})!"

    # add to the beginning (position-wise) of the activations
    resid_pre[:, :apos, :] += COEFF * ACT_DIFF


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
        r = LOADED_MODEL.generate(input=tokenized, max_new_tokens=50, do_sample=True, **kwargs)
    return r


def output_hooked_generation():
    editing_hooks = [(f"blocks.{ACT_LAYER}.hook_resid_pre", ave_hook)]
    res = hooked_generate([PROMPT] * 4, editing_hooks, seed=SEED, **SAMPLING_KWARGS)

    # Print results, removing the ugly beginning of sequence token
    res_str = LOADED_MODEL.to_string(res[:, 1:])
    print(("\n\n" + "-" * 80 + "\n\n").join(res_str))


def main():
    global LOADED_MODEL
    LOADED_MODEL = get_model()

    tlen = lambda prompt: LOADED_MODEL.to_tokens(prompt).shape[1]
    print(f"\n\nBEFORE PADDING")
    print(f"'{PROMPT_ADD_RAW}'", f"'{PROMPT_SUB_RAW}'")
    print(f"length add raw: {tlen(PROMPT_ADD_RAW)}, length sub raw: {tlen(PROMPT_SUB_RAW)}")

    prompt_add, prompt_sub = pad_tokens()

    print(f"'{prompt_add}'", f"'{prompt_sub}'")
    print(f"length add: {tlen(prompt_add)}, length sub: {tlen(prompt_sub)}")

    global ACT_DIFF
    ACT_DIFF = get_act_diff(prompt_add, prompt_sub)

    output_hooked_generation()
    
if __name__=='__main__':
    main()