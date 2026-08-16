from pathlib import Path
import pandas as pd
base_dir = Path(__file__).resolve().parent
csv_path = base_dir / "pair_vicuna_jailbreaks_annotated.csv"

df = pd.read_csv(csv_path)

print("Dataset loaded!")
print(f"Number of prompts: {len(df)}")

print("\n==============================")
print("PRIMARY MECHANISM DISTRIBUTION")
print("==============================")
mechanism_counts = df["primary_mechanism"].value_counts().sort_values(ascending=False)
print(mechanism_counts)

print("\n==============================")
print("JAILBREAK RATE BY MECHANISM")
print("==============================")
mechanism_jailbreak = (
    df.groupby("primary_mechanism")["jailbroken"]
    .agg(["count", "mean"])
    .rename(columns={"mean": "jailbreak_rate"})
)
mechanism_jailbreak["jailbreak_rate"] = mechanism_jailbreak["jailbreak_rate"] * 100
print(mechanism_jailbreak)

print("\n==============================")
print("JAILBREAK RATE BY FRAMING")
print("==============================")
framing_jailbreak = (
    df.groupby("framing")["jailbroken"]
    .agg(["count", "mean"])
    .rename(columns={"mean": "jailbreak_rate"})
)
framing_jailbreak["jailbreak_rate"] = framing_jailbreak["jailbreak_rate"] * 100
print(framing_jailbreak)

print("\n==============================")
print("MECHANISM × JAILBREAK")
print("==============================")
print(pd.crosstab(df["primary_mechanism"], df["jailbroken"]))

print("\n==============================")
print("FRAMING × JAILBREAK")
print("==============================")
print(pd.crosstab(df["framing"], df["jailbroken"]))

print("\n==============================")
print("ANNOTATION CONFIDENCE")
print("==============================")
print(df["confidence"].value_counts())

print("\nAnalysis complete!")
