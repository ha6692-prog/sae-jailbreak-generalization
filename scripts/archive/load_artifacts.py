from pathlib import Path
import jailbreakbench as jbb
import pandas as pd

print("Loading JailbreakBench...")

artifact = jbb.read_artifact(
    method="PAIR",
    model_name="vicuna-13b-v1.5"
)

print("Artifact loaded!")
print("Number of entries:", len(artifact.jailbreaks))

data = []

for i, jailbreak in enumerate(artifact.jailbreaks):

    data.append({
        "prompt_id": i + 1,
        "behavior": jailbreak.behavior,
        "category": jailbreak.category,
        "prompt": jailbreak.prompt,
        "jailbroken": jailbreak.jailbroken
    })

df = pd.DataFrame(data)

project_root = Path(__file__).resolve().parents[2]
output_csv = project_root / "data" / "raw" / "pair_vicuna_jailbreaks.csv"

df.to_csv(output_csv, index=False)

print("\nSaved dataset!")
print("File:", output_csv)
print("Rows:", len(df))