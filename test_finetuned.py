from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('./finetuned-smollm2')
model = AutoModelForCausalLM.from_pretrained('./finetuned-smollm2')

prompt = "Pune is"
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
