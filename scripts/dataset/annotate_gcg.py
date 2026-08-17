from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
raw_csv = project_root / "data" / "raw" / "gcg_vicuna_jailbreaks.csv"
output_csv = project_root / "data" / "processed" / "gcg_vicuna_jailbreaks_annotated.csv"

df = pd.read_csv(raw_csv)

# Add attack metadata
df["attack_source"] = "JailbreakBench"
df["attack_method"] = "GCG"

# GCG-specific mechanism
df["primary_mechanism"] = "Adversarial Suffix"

# No secondary mechanism for this dataset
df["secondary_mechanism"] = "None"

# We are not using the PAIR framing taxonomy
# for the GCG suffix itself.
df["framing"] = "None"

# All classifications are high confidence
# because the attack family comes directly from the GCG artifact.
df["confidence"] = "High"
df["notes"] = (
    "GCG adversarial suffix attack; "
    "prompt contains an optimized/garbled suffix appended to the request."
)

df.to_csv(output_csv, index=False)

print("GCG annotation complete!")
print("Number of prompts:", len(df))
print("Saved as:", output_csv)
print("\nMechanism distribution:")
print(df["primary_mechanism"].value_counts())
