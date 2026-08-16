import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32 if device.type == "cpu" else torch.float16,
)
model.to(device)
model.eval()
print("Model loaded!")
print("Device:", device)

# Test prompt
prompt = "Explain what machine learning is in simple terms."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer(text, return_tensors="pt")
inputs = {key: value.to(model.device) for key, value in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

print("\nModel forward pass successful!")
print("Number of hidden-state layers:", len(outputs.hidden_states))
for i, hidden in enumerate(outputs.hidden_states):
    print(f"Layer {i}: shape = {tuple(hidden.shape)}")
