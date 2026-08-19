# Comparing SmolLM2-135M vs Qwen2.5-0.5B

## Why compare two models
To see which architecture choices are universal transformer design vs
model-specific decisions, and to check whether attention behavior
generalizes across different models.

## Architecture comparison

| | SmolLM2-135M | Qwen2.5-0.5B |
|---|---|---|
| Hidden size | 576 | 896 |
| Layers | 30 | 24 |
| Attention heads | 9 | 14 |
| KV heads (GQA) | 3 | 2 |
| Intermediate size | 1536 | 4864 |
| Vocab size | 49,152 | 151,936 |
| Total parameters | 134,515,008 | 494,032,768 |

Key finding: Qwen2.5-0.5B is bigger mainly because it is WIDER
(896 vs 576 hidden size) and has a much bigger vocabulary
(152K vs 49K tokens), not because it has more layers -- it
actually has fewer layers (24 vs 30) than SmolLM2.

## Tokenization difference

Sentence: "I love Pune"

- SmolLM2 tokenizer split "Pune" into two pieces: "ĠP" + "une"
- Qwen2.5 tokenizer kept "Pune" as one clean token: "ĠPune"

This suggests Qwen's larger, more multilingual-aware vocabulary
covers more real-world words as single tokens.

## Attention across depth (causal-masked, Head 0)

Tracking how much the last token ("Pune") attends to the first
token ("I") vs to itself, across layers:

| Layer | Pune to I | Pune to love | Pune to itself |
|---|---|---|---|
| 0 (early) | 0.422 | 0.155 | 0.422 |
| 12 (middle) | 0.147 | 0.002 | 0.852 |
| 23 (last) | 0.609 | 0.342 | 0.049 |

Finding: attention redistribution across depth is real and
measurable, but the exact shape (which layer focuses on what)
differs by model. Middle layers here show strong self-focus,
while the final layer swings back toward pulling in context
from earlier words -- consistent with the general idea that
later layers do more "final context integration" before
predicting the next token.

## Scripts used
- attention_walkthrough_qwen.py -- Q/K/V projection walkthrough for Qwen2.5-0.5B
- attention_depth_comparison_qwen.py -- attention across layers 0, 12, 23 for Qwen2.5-0.5B
