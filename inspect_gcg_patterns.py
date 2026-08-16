import pandas as pd

df = pd.read_csv("gcg_vicuna_jailbreaks.csv")
print("Number of prompts:", len(df))
for i in range(len(df)):
    print("\n" + "=" * 80)
    print(f"PROMPT {i + 1}")
    print("=" * 80)
    print(df.iloc[i]["prompt"])
