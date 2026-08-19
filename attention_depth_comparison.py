import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# The key fix: attn_implementation="eager" makes the model compute attention
# the traditional way, which lets us actually see the attention weights.
model = AutoModelForCausalLM.from_pretrained(
    './smollm2-135m',
    attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained('./smollm2-135m')

text = "I love Pune"
inputs = tokenizer(text, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

print(f"Sentence: '{text}'")
print(f"Tokens: {tokens}\n")

with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

all_layer_attentions = outputs.attentions
print(f"Number of layers with attention captured: {len(all_layer_attentions)}")
print(f"Shape per layer: {all_layer_attentions[0].shape}  (batch, heads, tokens, tokens)\n")

def print_layer_attention(layer_idx, head_idx=0):
    attn = all_layer_attentions[layer_idx][0, head_idx]  # [seq_len, seq_len]
    print(f"{'='*70}")
    print(f"LAYER {layer_idx}, HEAD {head_idx}")
    print(f"{'='*70}")
    for i, t in enumerate(tokens):
        weights_str = ", ".join(f"{w:.3f}" for w in attn[i].tolist())
        print(f"  {t:10s} attends to [{', '.join(tokens)}] -> [{weights_str}]")
    print()

for layer_idx in [0, 15, 29]:
    print_layer_attention(layer_idx, head_idx=0)

print(f"{'='*70}")
print("WHAT TO LOOK FOR")
print(f"{'='*70}")
print("- Layer 0: often near-uniform or locally-focused attention (basic patterns)")
print("- Layer 15: attention may start concentrating on specific meaningful tokens")
print("- Layer 29: often sharper, more decisive attention (less spread out)")
print("Compare how 'peaky' (concentrated) vs 'flat' (spread out) each layer's")
print("attention row is -- flatter = still gathering info broadly, peakier = has")
print("already decided what matters.")
