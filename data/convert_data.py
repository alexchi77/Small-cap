import pandas as pd

# Load your CSV
df = pd.read_csv("historical_halts.csv")

# Combine Halt Date + Halt Time
df["halt_time"] = pd.to_datetime(
    df["Halt Date"].astype(str) + " " + df["Halt Time"].astype(str),
    errors="coerce"
)

# Combine Resume Date + NYSE Resume Time (some may be missing)
df["resume_time"] = pd.to_datetime(
    df["Resume Date"].astype(str) + " " + df["NYSE Resume Time"].astype(str),
    errors="coerce"
)

# Drop old split columns if you don’t need them
df = df.drop(columns=["Halt Date", "Halt Time", "Resume Date", "NYSE Resume Time"])

print(df.head())
df.to_csv("historical_halts.csv")
