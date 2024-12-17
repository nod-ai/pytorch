import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from time import perf_counter as pf
torch.utils.rename_privateuse1_backend('zoom')

# Set up the model and tokenizer
model_id = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="zoom"
)

# Function to generate text
def generate_text(prompt, max_length=30):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
prompt = "Hey, how are you doing today?"
s = pf()
response = generate_text(prompt)
e = pf()
print(f"Prompt: {prompt}")
print(f"Response: {response}")

print(f"{e-s} seconds")
