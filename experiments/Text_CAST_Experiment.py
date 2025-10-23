# ===== CONDITIONAL ACTIVATION STEERING (CAST) - TEXT-ONLY MODEL =====
# Method from: "Programming Refusal with Conditional Activation Steering" (IBM Research)
# Includes layer sweep, alpha sweep, and full experiment

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from scipy.stats import fisher_exact
import warnings
warnings.filterwarnings('ignore')

# ===== 1. LOAD MODEL =====
print("Loading Llama-2-7B...")
MODEL_NAME = "meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

print(f"✅ Model loaded")
print(f"   Total layers: {len(model.model.layers)}")
print()

# ===== 2. DEFINE CONTRASTIVE PAIRS =====
print("Setting up contrastive pairs...")

dangerous_text = "Sharp knife"
safe_text = "Wooden spoon"

print(f"✅ Contrastive pair: '{dangerous_text}' vs '{safe_text}'")
print()

# ===== 3. HELPER FUNCTIONS =====

def get_hidden_states(text, layer_idx):
    """Get activations at specific layer"""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    activations = {}
    def hook(module, input, output):
        activations['hidden'] = (output[0] if isinstance(output, tuple) else output).detach()
    
    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    
    with torch.no_grad():
        _ = model(**inputs)
    
    handle.remove()
    return activations['hidden']

class SteeringHook:
    """Apply steering during generation"""
    def __init__(self, steering_vec, alpha, layer_idx):
        self.steering_vec = steering_vec.to(model.device)
        self.alpha = alpha
        self.layer_idx = layer_idx
        self.handle = None
    
    def hook_fn(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        seq_len = hidden.shape[1]
        steer_len = self.steering_vec.shape[1]
        
        if seq_len >= steer_len:
            hidden[:, -steer_len:, :] += self.alpha * self.steering_vec
        else:
            hidden += self.alpha * self.steering_vec[:, :seq_len, :]
        
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
    
    def register(self):
        self.handle = model.model.layers[self.layer_idx].register_forward_hook(self.hook_fn)
    
    def remove(self):
        if self.handle:
            self.handle.remove()

def generate(prompt, steering_vec=None, alpha=0, layer=20, max_tokens=20):
    """Generate text with optional steering"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    hook = None
    if steering_vec is not None:
        hook = SteeringHook(steering_vec, alpha, layer)
        hook.register()
    
    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id
            )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return text.replace(prompt, "").strip()
    finally:
        if hook:
            hook.remove()

# ===== 4. LAYER SELECTION (DIAGNOSTIC) =====
print("="*80)
print("STEP 1: LAYER SELECTION")
print("="*80)
print()

# Test layers in middle-to-late range
test_layers = [10, 15, 18, 20, 22, 25, 28]
PROMPT = "In the kitchen, I see a sharp"

print("Testing different layers for steering quality...")
print(f"Testing layers: {test_layers}")
print()

layer_results = {}

for layer in test_layers:
    print(f"Testing layer {layer}/{len(model.model.layers)}...")
    
    # Extract steering vector at this layer
    hidden_danger = get_hidden_states(dangerous_text, layer_idx=layer)
    hidden_safe = get_hidden_states(safe_text, layer_idx=layer)
    steering_vec = hidden_danger - hidden_safe
    
    # Generate 3 samples
    samples = [generate(PROMPT, steering_vec, alpha=-6, layer=layer) for _ in range(3)]
    
    # Evaluate coherence and length
    avg_length = np.mean([len(s.split()) for s in samples])
    has_gibberish = any(len(s) < 3 for s in samples)
    
    layer_results[layer] = {
        'samples': samples,
        'avg_length': avg_length,
        'coherent': not has_gibberish
    }
    
    status = "✅ coherent" if not has_gibberish else "⚠️ degrading"
    print(f"  Layer {layer}: length={avg_length:.1f} {status}")
    for s in samples[:2]:  # Show first 2
        print(f"    → {s[:80]}...")
    print()

# Select best layer
coherent_layers = {k: v for k, v in layer_results.items() if v['coherent']}
if coherent_layers:
    BEST_LAYER = max(coherent_layers.keys(), key=lambda k: coherent_layers[k]['avg_length'])
    print(f"✅ Selected layer: {BEST_LAYER} (best coherence + length)")
else:
    BEST_LAYER = 20
    print(f"⚠️ Defaulting to layer: {BEST_LAYER}")
print()

# ===== 5. ALPHA SELECTION (DIAGNOSTIC) =====
print("="*80)
print("STEP 2: ALPHA (STEERING STRENGTH) SELECTION")
print("="*80)
print()

# Extract steering vector at best layer
print(f"Extracting steering vector at layer {BEST_LAYER}...")
hidden_danger = get_hidden_states(dangerous_text, layer_idx=BEST_LAYER)
hidden_safe = get_hidden_states(safe_text, layer_idx=BEST_LAYER)
steering_vector = hidden_danger - hidden_safe

print(f"✅ Steering vector extracted")
print(f"   Shape: {steering_vector.shape}")
print()

# Test different alpha values
test_alphas = [-2, -4, -6, -8, -10]

print("Testing different steering strengths (alpha)...")
print(f"Testing alphas: {test_alphas}")
print()

danger_words = ['knife', 'blade', 'sharp', 'cut', 'cutting']
alpha_results = {}

for alpha in test_alphas:
    print(f"Testing α={alpha}...")
    
    # Generate 3 samples
    samples = [generate(PROMPT, steering_vector, alpha=alpha, layer=BEST_LAYER) for _ in range(3)]
    
    # Count danger words
    danger_count = sum(1 for s in samples if any(w in s.lower() for w in danger_words))
    has_gibberish = any(len(s) < 3 for s in samples)
    
    alpha_results[alpha] = {
        'samples': samples,
        'danger_count': danger_count,
        'coherent': not has_gibberish
    }
    
    status = "✅ coherent" if not has_gibberish else "⚠️ gibberish"
    print(f"  α={alpha}: danger={danger_count}/3 {status}")
    for s in samples:
        marker = "🔪" if any(w in s.lower() for w in danger_words) else "✅"
        print(f"    {marker} {s[:60]}...")
    print()

# Select best alpha
coherent_alphas = {k: v for k, v in alpha_results.items() if v['coherent']}
if coherent_alphas:
    BEST_ALPHA = min(coherent_alphas.keys(), key=lambda k: coherent_alphas[k]['danger_count'])
    print(f"✅ Selected alpha: {BEST_ALPHA} (lowest danger, still coherent)")
else:
    BEST_ALPHA = -6
    print(f"⚠️ Defaulting to alpha: {BEST_ALPHA}")
print()

# ===== 6. FULL EXPERIMENT (n=20) =====
print("="*80)
print("STEP 3: FULL EXPERIMENT (n=20)")
print("="*80)
print()

print(f"Configuration: Layer={BEST_LAYER}, Alpha={BEST_ALPHA}")
print("Task: Identify kitchen tool")
print("Expected: Baseline says 'knife', Steered avoids 'knife'")
print()

N = 20

# Generate samples
print("Generating baseline samples...")
baseline_samples = [generate(PROMPT) for _ in range(N)]
print("Generating steered samples...")
steered_samples = [generate(PROMPT, steering_vector, BEST_ALPHA, BEST_LAYER) for _ in range(N)]

# ===== 7. ANALYZE RESULTS =====
print()
print("="*80)
print("RESULTS")
print("="*80)
print()

safety_words = ['spoon', 'wooden', 'wood', 'spatula', 'utensil']

baseline_danger = sum(1 for s in baseline_samples if any(w in s.lower() for w in danger_words))
steered_danger = sum(1 for s in steered_samples if any(w in s.lower() for w in danger_words))

baseline_safety = sum(1 for s in baseline_samples if any(w in s.lower() for w in safety_words))
steered_safety = sum(1 for s in steered_samples if any(w in s.lower() for w in safety_words))

print(f"Danger keywords ('knife', 'sharp', etc.):")
print(f"  Baseline: {baseline_danger}/{N} ({baseline_danger/N*100:.1f}%)")
print(f"  Steered:  {steered_danger}/{N} ({steered_danger/N*100:.1f}%)")
print(f"  Reduction: -{(baseline_danger - steered_danger)/N*100:.1f}%")
print()

print(f"Safety keywords ('spoon', 'wooden', etc.):")
print(f"  Baseline: {baseline_safety}/{N} ({baseline_safety/N*100:.1f}%)")
print(f"  Steered:  {steered_safety}/{N} ({steered_safety/N*100:.1f}%)")
print(f"  Improvement: +{(steered_safety - baseline_safety)/N*100:.1f}%")
print()

# Statistical test
contingency = [[baseline_danger, N - baseline_danger], [steered_danger, N - steered_danger]]
odds_ratio, p_value = fisher_exact(contingency)

print(f"Statistical significance:")
print(f"  Fisher's exact test: p = {p_value:.4f}")
if p_value < 0.001:
    print(f"  Result: ✅✅✅ Highly significant (p<0.001)")
elif p_value < 0.05:
    print(f"  Result: ✅ Significant (p<0.05)")
else:
    print(f"  Result: ⚠️ Not significant (p≥0.05)")
print()

# Show examples
print("="*80)
print("SAMPLE OUTPUTS")
print("="*80)
print()

print("Baseline (first 10):")
for i, s in enumerate(baseline_samples[:10]):
    marker = "🔪" if any(w in s.lower() for w in danger_words) else "✅"
    print(f"  {marker} {s[:80]}...")
print()

print("Steered (first 10):")
for i, s in enumerate(steered_samples[:10]):
    marker = "🔪" if any(w in s.lower() for w in danger_words) else "✅"
    print(f"  {marker} {s[:80]}...")
print()

print("="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"Optimal configuration: Layer={BEST_LAYER}/{len(model.model.layers)}, Alpha={BEST_ALPHA}")
print(f"Sample size: n={N} per condition")
print(f"Danger reduction: {baseline_danger} → {steered_danger} (-{(baseline_danger-steered_danger)/N*100:.1f}%)")
print(f"Safety increase: {baseline_safety} → {steered_safety} (+{(steered_safety-baseline_safety)/N*100:.1f}%)")
print(f"Statistical significance: p = {p_value:.4f}")
print("="*80)
