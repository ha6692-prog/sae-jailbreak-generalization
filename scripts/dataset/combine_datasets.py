from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
processed_dir = project_root / "data" / "processed"

pair = pd.read_csv(processed_dir / "pair_vicuna_jailbreaks_final.csv")
gcg = pd.read_csv(processed_dir / "gcg_vicuna_jailbreaks_final.csv")

# Ensure compatibility with the combine step
if "attack_method" not in pair.columns:
    pair["attack_method"] = "PAIR"
if "attack_method" not in gcg.columns:
    gcg["attack_method"] = "GCG"
if "attack_source" not in pair.columns:
    pair["attack_source"] = "JailbreakBench"
if "attack_source" not in gcg.columns:
    gcg["attack_source"] = "JailbreakBench"

# Keep common columns
columns = [
    "prompt_id",
    "behavior",
    "prompt",
    "jailbroken",
    "attack_family",
    "attack_method",
    "primary_mechanism",
    "secondary_mechanism",
    "framing",
    "confidence",
    "notes"
]
pair = pair[columns]
gcg = gcg[columns]

# Give unique IDs after combining
pair["global_id"] = range(1, len(pair) + 1)
gcg["global_id"] = range(
    len(pair) + 1,
    len(pair) + len(gcg) + 1
)

# Combine
combined = pd.concat(
    [pair, gcg],
    ignore_index=True
)

output_csv = processed_dir / "combined_jailbreak_dataset.csv"
combined.to_csv(output_csv, index=False)

print("================================")
print("COMBINED DATASET CREATED")
print("================================")
print("Total prompts:", len(combined))
print("\nAttack family:")
print(combined["attack_family"].value_counts())
print("\nMechanisms:")
print(combined["primary_mechanism"].value_counts())
print("\nJailbreak labels:")
print(combined["jailbroken"].value_counts())
print("\nSaved to:", output_csv)
