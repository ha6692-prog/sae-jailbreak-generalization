from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
processed_dir = project_root / "data" / "processed"

df = pd.read_csv(processed_dir / "pair_vicuna_jailbreaks_annotated.csv")
df["attack_family"] = "PAIR"
df["attack_method"] = "PAIR"
df["attack_source"] = "JailbreakBench"
output_csv = processed_dir / "pair_vicuna_jailbreaks_final.csv"
df.to_csv(output_csv, index=False)
print("PAIR metadata added!")
print("Rows:", len(df))
print(df["attack_family"].value_counts())
print("Saved as:", output_csv)
