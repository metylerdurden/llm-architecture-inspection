# LLM Architecture Inspection

Inspecting a small open-source language model (SmolLM2-135M) to understand its internal architecture — hidden dimension, layers, attention heads, and parameter count.

## Model
- **Name:** HuggingFaceTB/SmolLM2-135M
- **Total parameters:** 134,515,008 (~135M)
- **Hidden size:** 576
- **Layers:** 30
- **Attention heads:** 9 (3 KV heads, using grouped-query attention)

See [inspection_report.md](./inspection_report.md) for the full breakdown.

## How it was done
Model downloaded via Hugging Face `transformers`, then inspected using `AutoConfig` and `AutoModelForCausalLM` in Python.

## Cross-model comparison
See [MODEL_COMPARISON.md](./MODEL_COMPARISON.md) for a comparison of
SmolLM2-135M vs Qwen2.5-0.5B -- architecture differences and how
attention patterns differ across model depth.

## Fine-tuning experiment
See [TRAINING_EXPERIMENT.md](./TRAINING_EXPERIMENT.md) for a
fine-tuning run on SmolLM2-135M, including the loss curve and an
honest look at what fine-tuning did (and didn't) teach the model.
