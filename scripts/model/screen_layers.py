from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
TRAIN_FILE = ROOT / "data" / "splits" / "train.csv"
VALIDATION_FILE = ROOT / "data" / "splits" / "validation.csv"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LAYERS_TO_TEST = [8, 14, 20, 26]
MAX_LENGTH = 512


def _to_binary_labels(series: pd.Series) -> np.ndarray:
    return series.astype(int).to_numpy()


def _prepare_texts(df: pd.DataFrame) -> list[str]:
    return df["prompt"].fillna("").astype(str).tolist()


def _extract_layer_features(
    texts: list[str],
    layer: int,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: torch.device,
) -> np.ndarray:
    features: list[np.ndarray] = []

    for i, text in enumerate(texts, start=1):
        messages = [{"role": "user", "content": text}]
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
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        layer_tensor = outputs.hidden_states[layer]
        last_token = layer_tensor[:, -1, :].squeeze(0).detach().float().cpu().numpy()
        features.append(last_token)

        if i % 25 == 0 or i == len(texts):
            print(f"  Processed {i}/{len(texts)} prompts")

    return np.stack(features, axis=0)


def main() -> None:
    print("\n" + "=" * 60)
    print("LOADING SPLITS")
    print("=" * 60)

    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VALIDATION_FILE)

    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")

    X_train_text = _prepare_texts(train_df)
    X_val_text = _prepare_texts(val_df)
    y_train = _to_binary_labels(train_df["jailbroken"])
    y_val = _to_binary_labels(val_df["jailbroken"])

    print("\n" + "=" * 60)
    print("LOADING MODEL")
    print("=" * 60)

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
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
    first_device = next(model.parameters()).device
    print("Model loaded.")
    print("Model device map:", getattr(model, "hf_device_map", "single-device"))

    results = []

    for layer in LAYERS_TO_TEST:
        print("\n" + "=" * 60)
        print(f"SCREENING LAYER {layer}")
        print("=" * 60)

        print("Extracting training activations...")
        X_train = _extract_layer_features(
            X_train_text,
            layer,
            tokenizer,
            model,
            first_device,
        )

        print("Extracting validation activations...")
        X_val = _extract_layer_features(
            X_val_text,
            layer,
            tokenizer,
            model,
            first_device,
        )

        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=42),
        )
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_val)

        accuracy = accuracy_score(y_val, predictions)
        precision = precision_score(
            y_val,
            predictions,
            zero_division=0,
        )
        recall = recall_score(
            y_val,
            predictions,
            zero_division=0,
        )
        f1 = f1_score(
            y_val,
            predictions,
            zero_division=0,
        )

        print("\nResults:")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1       : {f1:.4f}")

        results.append(
            {
                "layer": layer,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

        del X_train
        del X_val
        del classifier
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("LAYER SCREENING RESULTS")
    print("=" * 60)
    print(results_df.to_string(index=False))

    best_layer = results_df.loc[
        results_df["f1"].idxmax(),
        "layer",
    ]
    print("\n" + "=" * 60)
    print("BEST LAYER")
    print("=" * 60)
    print(f"Best layer based on validation F1: {best_layer}")


if __name__ == "__main__":
    main()
