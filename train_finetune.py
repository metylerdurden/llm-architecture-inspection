from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset

# A tiny custom dataset -- replace these lines with your own text later
texts = [
    "I love Pune. It is a beautiful city in India.",
    "Pune is known for its pleasant weather and colleges.",
    "I am learning machine learning step by step.",
    "Attention mechanisms help models understand context.",
    "Pune has a rich history and a growing tech industry.",
] * 20  # repeat so there's enough data to see loss move

tokenizer = AutoTokenizer.from_pretrained('./smollm2-135m')
model = AutoModelForCausalLM.from_pretrained('./smollm2-135m')

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = Dataset.from_dict({"text": texts})

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=32)

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./finetuned-smollm2",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    logging_steps=5,
    save_strategy="epoch",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("Starting training -- watch the loss column go down\n")
trainer.train()

trainer.save_model("./finetuned-smollm2")
tokenizer.save_pretrained("./finetuned-smollm2")
print("\nDone. Fine-tuned model saved to ./finetuned-smollm2")
