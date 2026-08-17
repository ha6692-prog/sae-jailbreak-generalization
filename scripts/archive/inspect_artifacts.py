import jailbreakbench as jbb

methods = [
    ("PAIR", "vicuna-13b-v1.5"),
    ("GCG", "vicuna-13b-v1.5"),
]

for method, model in methods:
    print("\n" + "=" * 70)
    print("METHOD:", method)
    print("MODEL:", model)
    print("=" * 70)
    try:
        artifact = jbb.read_artifact(
            method=method,
            model_name=model
        )
        print("Number of prompts:", len(artifact.jailbreaks))
        for i, jailbreak in enumerate(artifact.jailbreaks[:3]):
            print("\n--- Example", i + 1, "---")
            print("Behavior:", jailbreak.behavior)
            print("Jailbroken:", jailbreak.jailbroken)
            print("Prompt:")
            print(jailbreak.prompt[:1000])
    except Exception as e:
        print("Could not load this artifact.")
        print("Error:", e)
