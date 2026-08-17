from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
input_csv = project_root / "data" / "processed" / "combined_jailbreak_dataset.csv"

df = pd.read_csv(input_csv)
print("Total rows:", len(df))
print("\nUnique behaviors:")
print(df["behavior"].nunique())
print("\nBehavior counts:")
print(df["behavior"].value_counts())
print("\nAttack family by behavior:")
print(pd.crosstab(df["behavior"], df["attack_family"]))
