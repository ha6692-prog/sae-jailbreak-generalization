import pandas as pd

df = pd.read_csv("gcg_vicuna_jailbreaks_annotated.csv")
df["attack_family"] = "GCG"
df.to_csv(
    "gcg_vicuna_jailbreaks_final.csv",
    index=False
)
print("GCG metadata added!")
print("Rows:", len(df))
print(df["attack_family"].value_counts())
