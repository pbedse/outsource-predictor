import pandas as pd
df = pd.read_csv("trials_clean.csv")

print("How many trials list any collaborator?")
print(df["n_collaborators"].gt(0).value_counts())

print("\nMost common collaborators:")
all_collabs = df["collaborators"].dropna().str.split("; ").explode()
print(all_collabs.value_counts().head(25).to_string())