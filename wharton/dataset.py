"""
generate_market_data_csv.py

Pulls daily bar data from Alpaca for the requested tickers and writes a CSV
in the NSE-bhavcopy-style format shown in the sample:
Date, Symbol, Series, Prev Close, Open, High, Low, Close, WAP, Volume,
Turnover, Trades, Deliverable Volume, %Deliverable.

DATA NOTES (read before running):
- Alpaca only covers US-listed securities. Substitutions used:
    SPX      -> SPY    (S&P 500 ETF, tracks the index; SPX itself isn't a
                         tradable US equity/ETF Alpaca can serve)
    Tencent  -> TCEHY  (Tencent's US OTC ADR; the primary listing is on
                         the Hong Kong exchange, which Alpaca doesn't cover)
- "Deliverable Volume" / "%Deliverable" are NSE/BSE (India)-specific
  concepts with no US equivalent. Alpaca has no such data, so these
  columns are left blank for every row.
- "Turnover" (traded value) isn't returned directly; it's approximated
  as Volume * WAP.
- "WAP" = Alpaca's vwap field for the bar.
- "Series" isn't a real US concept; filled with "EQ" for stocks and
  "ETF" for ETFs, just to fill the column.
- Added BND (Vanguard Total Bond Market ETF) as extra bond exposure
  alongside VGIT, since you said bonds were optional/bonus.

Setup:
    pip install alpaca-py pandas
    Set API_KEY and SECRET_KEY below to your Alpaca credentials.

Usage:
    python generate_market_data_csv.py [days_back] [output_file.csv]
"""

import os
import sys
from datetime import datetime, timedelta

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

# Paste your Alpaca credentials directly below.
API_KEY = "PKCRGSTX6C5XPPGIEVHTHMXN23"
SECRET_KEY = "8AZrn3foq7bZbstCACkx1YGGjyUt267cFtxQiki2DCEU"

if API_KEY.startswith("YOUR_") or SECRET_KEY.startswith("YOUR_"):
    sys.exit(
        "Set API_KEY and SECRET_KEY near the top of this script "
        "to your actual Alpaca credentials before running it."
    )

# symbol -> Series label
TICKERS = {
    "NVDA": "EQ",     # Nvidia
    "MU": "EQ",       # Micron
    "SPY": "ETF",     # proxy for SPX
    "VXUS": "ETF",    # Vanguard Total International Stock ETF
    "MSFT": "EQ",     # Microsoft
    "GOOGL": "EQ",    # Google/Alphabet
    "META": "EQ",     # Meta
    "AMD": "EQ",      # AMD
    "BAC": "EQ",      # Bank of America
    "TCEHY": "EQ",    # Tencent ADR proxy
    "VGIT": "ETF",    # Vanguard Intermediate-Term Treasury ETF
    "BND": "ETF",     # Vanguard Total Bond Market ETF (extra bond exposure)
}

DAYS_BACK = int(sys.argv[1]) if len(sys.argv) > 1 else 30
OUTPUT_FILE = sys.argv[2] if len(sys.argv) > 2 else "market_data.csv"

START_DATE = datetime.now() - timedelta(days=DAYS_BACK)
END_DATE = datetime.now()

client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

request = StockBarsRequest(
    symbol_or_symbols=list(TICKERS.keys()),
    timeframe=TimeFrame.Day,
    start=START_DATE,
    end=END_DATE,
    feed=DataFeed.IEX,  # free/paper accounts are only entitled to IEX
)

bars = client.get_stock_bars(request)
df = bars.df  # MultiIndex DataFrame: (symbol, timestamp)

if df.empty:
    sys.exit("No data returned. Check your date range, tickers, and API keys.")

rows = []

for symbol, series in TICKERS.items():
    if symbol not in df.index.get_level_values(0):
        print(f"Warning: no data for {symbol}, skipping.", file=sys.stderr)
        continue

    sym_df = df.loc[symbol].sort_index()
    sym_df["prev_close"] = sym_df["close"].shift(1)

    for date, row in sym_df.iterrows():
        wap = row.get("vwap")
        volume = row.get("volume")
        turnover = volume * wap if pd.notna(wap) and pd.notna(volume) else None

        rows.append({
            "Date": date.strftime("%Y-%m-%d"),
            "Symbol": symbol,
            "Series": series,
            "Prev Close": row["prev_close"],
            "Open": row["open"],
            "High": row["high"],
            "Low": row["low"],
            "Close": row["close"],
            "WAP": wap,
            "Volume": volume,
            "Turnover": turnover,
            "Trades": row.get("trade_count"),
            "Deliverable Volume": None,
            "%Deliverable": None,
        })

out_df = pd.DataFrame(rows, columns=[
    "Date", "Symbol", "Series", "Prev Close", "Open", "High", "Low",
    "Close", "WAP", "Volume", "Turnover", "Trades",
    "Deliverable Volume", "%Deliverable",
])

out_df.to_csv(OUTPUT_FILE, index=False)
print(f"Wrote {len(out_df)} rows to {OUTPUT_FILE}")