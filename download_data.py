import yfinance as yf
import os

# Stock ticker
ticker = "TCS.NS"

print("Downloading latest TCS stock data...")

# Download historical data up to the latest available date
data = yf.download(
    ticker,
    start="2018-01-01",
    interval="1d",
    auto_adjust=True
)

# Check if data was downloaded
if data.empty:
    print("ERROR: No data downloaded.")

else:

    # Handle yfinance multi-level columns
    if hasattr(data.columns, "nlevels") and data.columns.nlevels > 1:
        data = data.xs(ticker, axis=1, level=1)

    # Keep only closing price
    data = data[["Close"]]

    # Remove missing values
    data = data.dropna()

    # Create data folder
    os.makedirs("data", exist_ok=True)

    # Save CSV
    file_path = "data/TCS_stock.csv"

    data.to_csv(file_path)

    print("\nDataset downloaded successfully!")

    print("File:", file_path)

    print("Number of rows:", len(data))

    print("\nFirst 5 rows:")
    print(data.head())

    print("\nLast 5 rows:")
    print(data.tail())