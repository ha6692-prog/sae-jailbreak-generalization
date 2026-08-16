import pandas as pd

# Load both datasets
pair = pd.read_csv("pair_vicuna_jailbreaks_final.csv")
gcg = pd.read_csv("gcg_vicuna_jailbreaks_final.csv")

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

# Save
combined.to_csv(
    "combined_jailbreak_dataset.csv",
    index=False
)

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
