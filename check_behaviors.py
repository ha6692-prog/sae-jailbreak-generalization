import pandas as pd

df = pd.read_csv("combined_jailbreak_dataset.csv")
print("Total rows:", len(df))
print("\nUnique behaviors:")
print(df["behavior"].nunique())
print("\nBehavior counts:")
print(df["behavior"].value_counts())
print("\nAttack family by behavior:")
print(pd.crosstab(df["behavior"], df["attack_family"]))
