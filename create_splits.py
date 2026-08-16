import numpy as np
import pandas as pd

# Load combined dataset
df = pd.read_csv("combined_jailbreak_dataset.csv")

behaviors = (
    df["behavior"]
    .dropna()
    .drop_duplicates()
    .to_numpy()
)
rng = np.random.default_rng(42)
rng.shuffle(behaviors)
train_behaviors = set(behaviors[:70])
validation_behaviors = set(behaviors[70:85])
test_behaviors = set(behaviors[85:100])

# For the initial split, one behavior set is reserved for each set.
# Keep the data aligned across both attack families.
train_df = df[df["behavior"].isin(train_behaviors)].copy()
validation_df = df[df["behavior"].isin(validation_behaviors)].copy()
test_df = df[df["behavior"].isin(test_behaviors)].copy()

print(f"Train behaviors: {len(train_behaviors)}")
print(f"Validation behaviors: {len(validation_behaviors)}")
print(f"Test behaviors: {len(test_behaviors)}")
print(f"Train: {len(train_df)}")
print(f"Validation: {len(validation_df)}")
print(f"Test: {len(test_df)}")

train_set = set(train_df["behavior"])
validation_set = set(validation_df["behavior"])
test_set = set(test_df["behavior"])
print("Train ∩ Validation:", len(train_set & validation_set))
print("Train ∩ Test:", len(train_set & test_set))
print("Validation ∩ Test:", len(validation_set & test_set))
assert len(train_set & validation_set) == 0
assert len(train_set & test_set) == 0
assert len(validation_set & test_set) == 0

print("\n==============================")
print("TRAIN:")
print(train_df["attack_family"].value_counts())
print("\nVALIDATION:")
print(validation_df["attack_family"].value_counts())
print("\nTEST:")
print(test_df["attack_family"].value_counts())

train_df.to_csv("train.csv", index=False)
validation_df.to_csv("validation.csv", index=False)
test_df.to_csv("test.csv", index=False)
