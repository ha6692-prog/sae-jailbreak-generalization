import jailbreakbench as jbb
import pandas as pd

print("Loading GCG artifact...")
artifact = jbb.read_artifact(
    method="GCG",
    model_name="vicuna-13b-v1.5"
)
print("GCG loaded!")
print("Number of prompts:", len(artifact.jailbreaks))

data = []
for i, jailbreak in enumerate(artifact.jailbreaks):
    data.append({
        "prompt_id": i + 1,
        "behavior": jailbreak.behavior,
        "prompt": jailbreak.prompt,
        "jailbroken": jailbreak.jailbroken,
        "attack_source": "JailbreakBench",
        "attack_method": "GCG"
    })

df = pd.DataFrame(data)
df.to_csv(
    "gcg_vicuna_jailbreaks.csv",
    index=False
)

print("\nSaved!")
print("File: gcg_vicuna_jailbreaks.csv")
print("Rows:", len(df))
print("\nFirst 3 prompts:")
for i in range(min(3, len(df))):
    print("\n" + "=" * 70)
    print("PROMPT", i + 1)
    print("=" * 70)
    print(df.iloc[i]["prompt"])
