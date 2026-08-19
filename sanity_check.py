import pandas as pd

df = pd.read_csv("trials_scored.csv")

print("Average score by phase:")
print(df.groupby("phase_simple")["outsourcing_score"].agg(["mean", "count"]).round(3))

print("\nPhase mix in the top 200 (top decile):")
print(df.nlargest(200, "outsourcing_score")["phase_simple"].value_counts())

print("\nPhase mix overall, for comparison:")
print(df["phase_simple"].value_counts())