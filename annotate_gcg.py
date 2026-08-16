import pandas as pd

# Load GCG dataset
df = pd.read_csv("gcg_vicuna_jailbreaks.csv")

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

# Save
df.to_csv(
    "gcg_vicuna_jailbreaks_annotated.csv",
    index=False
)

print("GCG annotation complete!")
print("Number of prompts:", len(df))
print("Saved as: gcg_vicuna_jailbreaks_annotated.csv")
print("\nMechanism distribution:")
print(df["primary_mechanism"].value_counts())
