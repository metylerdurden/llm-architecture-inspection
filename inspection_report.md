# SmolLM2-135M — Architecture Inspection

## Config

```
LlamaConfig {
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 0,
  "dtype": "bfloat16",
  "eos_token_id": 0,
  "head_dim": 64,
  "hidden_act": "silu",
  "hidden_size": 576,
  "initializer_range": 0.041666666666666664,
  "intermediate_size": 1536,
  "is_llama_config": true,
  "max_position_embeddings": 8192,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 9,
  "num_hidden_layers": 30,
  "num_key_value_heads": 3,
  "pad_token_id": null,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_interleaved": false,
  "rope_parameters": {
    "rope_theta": 100000,
    "rope_type": "default"
  },
  "tie_word_embeddings": true,
  "transformers_version": "5.15.1",
  "use_cache": true,
  "vocab_size": 49152
}

```

## Key Numbers

- Hidden size: 576
- Num layers: 30
- Attention heads: 9
- KV heads: 3
- Intermediate size: 1536
- Vocab size: 49152
- Total parameters: 134,515,008
