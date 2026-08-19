# Understanding Attention in a Small LLM (SmolLM2-135M)

## What I did
Downloaded a small, real, open-source language model (135M parameters) and inspected its internal architecture and math to understand how it processes a sentence.

## Step-by-step

### 1. Downloaded a small LLM
Used SmolLM2-135M -- 134,515,008 total parameters.

### 2. Inspected its architecture
- Hidden size = 576 -- every word is represented as 576 numbers inside the model
- 30 layers -- text is processed through 30 stacked refinement steps
- 9 attention heads -- each layer looks at the sentence from 9 angles at once
- Vocabulary = 49,152 -- text is broken into ~49K possible word-pieces (tokens)

### 3. Traced a real sentence through the model
Sentence used: "I love Pune"

- Each word becomes a 576-number vector (embedding)
- The model computes a Query (what am I looking for), Key (what do I offer), and Value (what information do I carry) for every word
- Query dot Key for every pair of words produces a compatibility score
- Scores are converted to percentages (softmax) that sum to 100%

### 4. Discovered causal masking
Every word could only attend to itself and words before it -- never words after. This is what allows the model to generate text one word at a time without "cheating" by seeing the future.

### 5. Observed attention changing with depth
The word "love" attended mostly to itself in early layers, but by the middle layer pulled almost all its information from "I" instead -- showing that different layers play different roles.

## Key takeaway
Attention works by comparing every word to every other word using learned Query and Key vectors, with a masking rule that blocks looking ahead -- this is the core mechanism that lets models like this generate text one word at a time.

## Scripts used
- attention_walkthrough.py -- traces Q/K/V projections and attention scores for Layer 0
- attention_depth_comparison.py -- compares attention patterns across Layer 0, 15, and 29
