import pandas as pd

df = pd.read_csv("pair_vicuna_jailbreaks_annotated.csv")
df["attack_family"] = "PAIR"
df["attack_method"] = "PAIR"
df["attack_source"] = "JailbreakBench"
df.to_csv(
    "pair_vicuna_jailbreaks_final.csv",
    index=False
)
print("PAIR metadata added!")
print("Rows:", len(df))
print(df["attack_family"].value_counts())
