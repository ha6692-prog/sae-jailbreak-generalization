from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
SPLITS = {
	"train": ROOT / "data" / "splits" / "train.csv",
	"validation": ROOT / "data" / "splits" / "validation.csv",
	"test": ROOT / "data" / "splits" / "test.csv",
}
OUTPUT_DIR = ROOT / "results" / "activations"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LAYER_INDEX = 26
MAX_LENGTH = 512
MAX_PROMPTS = 10


def load_model():
	print("\n" + "=" * 60)
	print("LOADING MODEL")
	print("=" * 60)
	print(f"CUDA available: {torch.cuda.is_available()}")

	tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
	if torch.cuda.is_available():
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
	print("Model loaded.")
	print("Model device map:", getattr(model, "hf_device_map", "single-device"))
	return tokenizer, model


def extract_split(split_name, csv_path, tokenizer, model):
	dataframe = pd.read_csv(csv_path).head(MAX_PROMPTS)
	input_device = next(model.parameters()).device
	records = []

	print("\n" + "=" * 60)
	print(f"EXTRACTING {split_name.upper()} ACTIVATIONS")
	print("=" * 60)
	print(f"Prompts: {len(dataframe)}")
	print(f"Layer: {LAYER_INDEX}")

	for row_number, row in dataframe.iterrows():
		messages = [{"role": "user", "content": str(row["prompt"])}]
		rendered = tokenizer.apply_chat_template(
			messages,
			tokenize=False,
			add_generation_prompt=True,
		)
		inputs = tokenizer(
			rendered,
			return_tensors="pt",
			truncation=True,
			max_length=MAX_LENGTH,
		)
		inputs = {key: value.to(input_device) for key, value in inputs.items()}

		with torch.no_grad():
			outputs = model(**inputs, output_hidden_states=True)

		activation = outputs.hidden_states[LAYER_INDEX][0].detach().to(
			device="cpu",
			dtype=torch.float16,
		).contiguous()
		records.append(
			{
				"index": int(row_number),
				"behavior": row["behavior"],
				"jailbroken": bool(row["jailbroken"]),
				"attack_family": row["attack_family"],
				"num_tokens": activation.shape[0],
				"hidden_dim": activation.shape[1],
				"activation": activation,
			}
		)

		print(
			f"  Processed {len(records)}/{len(dataframe)} prompts "
			f"({activation.shape[0]} tokens x {activation.shape[1]})"
		)

	return records


def save_records(records, split_name):
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	output_file = OUTPUT_DIR / f"{split_name}_layer{LAYER_INDEX}_test.pt"
	torch.save(records, output_file)
	print("\nSaved:")
	print(output_file)
	return output_file


def verify_file(output_file):
	print("\n" + "=" * 60)
	print("VERIFYING SAVED ACTIVATIONS")
	print("=" * 60)
	records = torch.load(output_file, weights_only=False)
	print("Number of records:", len(records))
	first = records[0]
	print("\nFirst record:")
	print("Index:", first["index"])
	print("Behavior:", first["behavior"])
	print("Jailbroken:", first["jailbroken"])
	print("Attack family:", first["attack_family"])
	print("Tokens:", first["num_tokens"])
	print("Hidden dimension:", first["hidden_dim"])
	print("Activation shape:", tuple(first["activation"].shape))
	print("Activation dtype:", first["activation"].dtype)
	print("\nVerification successful!")


def main():
	tokenizer, model = load_model()
	for split_name, csv_path in SPLITS.items():
		records = extract_split(split_name, csv_path, tokenizer, model)
		output_file = save_records(records, split_name)
		verify_file(output_file)
	print("\n" + "=" * 60)
	print("ACTIVATION EXTRACTION TEST COMPLETE")
	print("=" * 60)


if __name__ == "__main__":
	main()
