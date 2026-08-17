from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
input_csv = project_root / "data" / "raw" / "gcg_vicuna_jailbreaks.csv"

df = pd.read_csv(input_csv)
print("Number of prompts:", len(df))
for i in range(len(df)):
    print("\n" + "=" * 80)
    print(f"PROMPT {i + 1}")
    print("=" * 80)
    print(df.iloc[i]["prompt"])
