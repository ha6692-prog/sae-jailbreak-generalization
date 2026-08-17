import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


def main() -> None:
    print("\n" + "=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU memory: {total_gb:.2f} GB")

    print("\n" + "=" * 60)
    print("LOADING TOKENIZER")
    print("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("\n" + "=" * 60)
    print("LOADING MODEL")
    print("=" * 60)
    if cuda_available:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
        )
        model.to("cpu")

    model.eval()
    print("Model loaded successfully.")
    print("Device map:", getattr(model, "hf_device_map", "single-device"))

    prompt = "Explain what machine learning is in simple terms."
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt")
    first_device = next(model.parameters()).device
    inputs = {key: value.to(first_device) for key, value in inputs.items()}

    print("\n" + "=" * 60)
    print("RUNNING FORWARD PASS")
    print("=" * 60)
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
        )
    print("Forward pass successful!")

    hidden_states = outputs.hidden_states
    print("\n" + "=" * 60)
    print("HIDDEN STATES")
    print("=" * 60)
    print(
        "Number of hidden-state tensors:",
        len(hidden_states),
    )
    for layer_index, hidden in enumerate(hidden_states):
        print(
            f"Layer {layer_index}: "
            f"shape={tuple(hidden.shape)}, "
            f"dtype={hidden.dtype}, "
            f"device={hidden.device}"
        )

    print("\n" + "=" * 60)
    print("GPU MEMORY")
    print("=" * 60)
    if cuda_available:
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"Allocated: {allocated:.2f} GB")
        print(f"Reserved:  {reserved:.2f} GB")
    else:
        print("CUDA not available; GPU memory stats are not applicable.")


if __name__ == "__main__":
    main()
