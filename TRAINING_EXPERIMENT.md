# Fine-tuning SmolLM2-135M on a tiny custom dataset

## What I did
Fine-tuned the pretrained SmolLM2-135M model on a small, repeated
custom text dataset (5 sentences x 20 repeats = 100 examples) for
3 epochs, using Hugging Face's Trainer API.

## Training loss curve

| Step | Epoch | Loss |
|---|---|---|
| 1  | 0.2 | 3.995 |
| 5  | 1.0 | 2.224 |
| 10 | 2.0 | 1.718 |
| 15 | 3.0 | 1.551 |

Loss dropped ~60% (4.0 -> 1.55), fastest in the first epoch then
flattening -- a healthy, expected training curve for a tiny dataset.

## What loss means
At each step the model predicts the next word for training text.
Loss measures how wrong those predictions were on average. A
falling loss means the model's predictions are getting more
accurate -- this is literally the model learning.

## Test after training

Prompt: "Pune is"
Output: "Pune is a city in the state of Maharashtra. It is the
capital of the state of Maharashtra. It is"

## Key finding: fine-tuning shapes style, not facts
The output is factually wrong (Mumbai, not Pune, is Maharashtra's
capital). This is expected: the tiny 100-example dataset was
enough to nudge output style and sentence structure, but nowhere
near enough to teach accurate new facts. This is exactly why
techniques like RAG (retrieval-augmented generation) exist --
grounding a model's output in retrieved real facts rather than
relying on what it "remembers" from fine-tuning or pretraining.

## Script used
- train_finetune.py -- fine-tunes SmolLM2-135M on a custom dataset
- test_finetuned.py -- tests generation from the fine-tuned model
