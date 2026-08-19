import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('./smollm2-135m')
tokenizer = AutoTokenizer.from_pretrained('./smollm2-135m')

# --- Step A: Tokenize a small sentence ---
text = "I love Pune"
inputs = tokenizer(text, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print("="*70)
print("STEP A: Tokenization")
print("="*70)
print(f"Sentence: '{text}'")
print(f"Tokens: {tokens}")
print(f"Token IDs: {inputs['input_ids'][0].tolist()}")
print(f"Number of tokens: {inputs['input_ids'].shape[1]}")

# --- Step B: Turn tokens into embeddings (576-number vectors) ---
with torch.no_grad():
    embed_layer = model.model.embed_tokens
    embeddings = embed_layer(inputs["input_ids"])

print("\n" + "="*70)
print("STEP B: Embeddings (each token -> 576 numbers)")
print("="*70)
print(f"Embedding shape: {embeddings.shape}  (1 sentence, {embeddings.shape[1]} tokens, 576 numbers each)")
print(f"First 5 numbers of token '{tokens[0]}':\n{embeddings[0,0,:5]}")

# --- Step C: Run Layer 0's Q, K, V projections manually ---
layer0 = model.model.layers[0]
q_proj = layer0.self_attn.q_proj
k_proj = layer0.self_attn.k_proj
v_proj = layer0.self_attn.v_proj

print("\n" + "="*70)
print("STEP C: Layer 0 attention weight shapes")
print("="*70)
print(f"q_proj weight shape: {q_proj.weight.shape}  (Query)")
print(f"k_proj weight shape: {k_proj.weight.shape}  (Key)")
print(f"v_proj weight shape: {v_proj.weight.shape}  (Value)")

with torch.no_grad():
    # Layer norm first (SmolLM2 normalizes before attention)
    normed = layer0.input_layernorm(embeddings)
    Q = q_proj(normed)
    K = k_proj(normed)
    V = v_proj(normed)

print(f"\nQ shape after projection: {Q.shape}  (3 tokens, 576 query numbers each)")
print(f"K shape after projection: {K.shape}  (3 tokens, 192 key numbers each -- fewer due to GQA)")
print(f"V shape after projection: {V.shape}  (3 tokens, 192 value numbers each)")

# --- Step D: Reshape into multiple heads ---
num_heads = model.config.num_attention_heads
num_kv_heads = model.config.num_key_value_heads
head_dim = model.config.head_dim
seq_len = embeddings.shape[1]

Q_heads = Q.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
K_heads = K.view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)

print("\n" + "="*70)
print("STEP D: Splitting into attention heads")
print("="*70)
print(f"Q reshaped into heads: {Q_heads.shape}  ({num_heads} heads, {seq_len} tokens, {head_dim} numbers per head)")
print(f"K reshaped into heads: {K_heads.shape}  ({num_kv_heads} heads, {seq_len} tokens, {head_dim} numbers per head)")

# --- Step E: Compute attention scores for Head 0 (how much each word "looks at" every other word) ---
# GQA: head 0,1,2 of Q share KV head 0; heads 3,4,5 share KV head 1; etc.
q_head0 = Q_heads[0, 0]   # [seq_len, head_dim]
k_head0 = K_heads[0, 0]   # [seq_len, head_dim]

scores = torch.matmul(q_head0, k_head0.transpose(0, 1)) / (head_dim ** 0.5)
attention_weights = torch.softmax(scores, dim=-1)

print("\n" + "="*70)
print("STEP E: Attention scores (Head 0) -- how much each word attends to every word")
print("="*70)
print(f"Raw scores shape: {scores.shape}  (each row = one token, each column = attention TO that token)")
print("\nRaw scores (before softmax):")
for i, t in enumerate(tokens):
    print(f"  {t:10s}: {scores[i].tolist()}")

print("\nAttention weights (after softmax -- these sum to 1.0 per row):")
for i, t in enumerate(tokens):
    weights_str = ", ".join(f"{w:.3f}" for w in attention_weights[i].tolist())
    print(f"  {t:10s} attends to [{', '.join(tokens)}] with weights: [{weights_str}]")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print("Each row shows how much that word 'looks at' every word in the sentence")
print("(including itself) when building its updated representation.")
print("Higher weight = more influence from that word.")
