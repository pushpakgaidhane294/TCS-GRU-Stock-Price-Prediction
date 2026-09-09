import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
file_path = "data/TCS_stock.csv"

df = pd.read_csv(file_path)

print("=" * 50)
print("TCS STOCK DATASET ANALYSIS")
print("=" * 50)

# Display first rows
print("\nFirst 5 rows:")
print(df.head())

# Display last rows
print("\nLast 5 rows:")
print(df.tail())

# Dataset shape
print("\nDataset shape:")
print(df.shape)

# Column names
print("\nColumns:")
print(df.columns.tolist())

# Data types
print("\nData types:")
print(df.dtypes)

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())

# Convert Date
df["Date"] = pd.to_datetime(df["Date"])

# Sort by date
df = df.sort_values("Date")

# Basic statistics
print("\nBasic statistics:")
print(df["Close"].describe())

# Minimum and maximum
print("\nMinimum closing price:")
print(df["Close"].min())

print("\nMaximum closing price:")
print(df["Close"].max())

# Date range
print("\nStarting date:")
print(df["Date"].min())

print("\nEnding date:")
print(df["Date"].max())

# Plot closing price
plt.figure(figsize=(12, 6))

plt.plot(df["Date"], df["Close"])

plt.title("TCS Historical Closing Price")
plt.xlabel("Date")
plt.ylabel("Closing Price (₹)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("data/tcs_historical_price.png", dpi=150)

plt.show()

print("\nGraph saved successfully!")
print("Location: data/tcs_historical_price.png")