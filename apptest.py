import pandas as pd

# Load dataset
df = pd.read_csv("Datasets/heart_attack_risk_dataset_20k.csv")  # Replace with your actual file name

# Display basic information
print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nRandom 5 Rows:")
print(df.sample(5))

print("\nColumn Names:")
print(df.columns.tolist())