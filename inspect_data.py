import pandas as pd

df = pd.read_excel("career_dataset.xlsx")

print("COLUMNS:")
print(list(df.columns))

print("\nSHAPE:")
print(df.shape)

print("\nFIRST 5 ROWS:")
print(df.head())

print("\nMISSING VALUES:")
print(df.isnull().sum())