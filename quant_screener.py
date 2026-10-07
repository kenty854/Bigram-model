"""
Market-Scanning Quant Relevance Model (Alpaca-only)
----------------------------------------------------
Scans a universe of tradable US equities via alpaca-py, builds purely
technical (price/volume) features, trains a model to predict forward
returns, and scores the current universe for "relevance."

No fundamentals data (no yfinance) -> this is a technical/momentum model,
not a value/quality model. Say so in your writeup.

Model choice: default is Ridge regression (robust on small tabular data,
interpretable coefficients). An MLPRegressor option is included for
comparison, but on a dataset this size an MLP is more likely to overfit
noise than find real signal -- use it as a second opinion, not a first pick.

Setup:
    pip install alpaca-py scikit-learn pandas numpy --break-system-packages
    export ALPACA_API_KEY="your_key"
    export ALPACA_SECRET_KEY="your_secret"
    (Free paper-trading keys work: https://app.alpaca.markets/paper/dashboard/overview)
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

API_KEY = os.getenv("PKV5FBWHZRE6S3JVY35QIZHAKM")
SECRET_KEY = os.getenv("5PwLAriDCPfLJjmsMxNvCjrbvGE4goHrVXj98f5rzMFN")

# --- Config -----------------------------------------------------------------
HISTORY_DAYS = 730          # ~2 years of daily bars to build training samples
FORWARD_WINDOW = 21         # predict ~1-month forward return
UNIVERSE_CAP = 300          # scanning the *entire* market on free-tier rate
                            # limits is impractical; cap and/or pre-filter by
                            # liquidity. Raise if your rate limit allows it.
MODEL_TYPE = "ridge"        # "ridge" (default, recommended) or "mlp"

FEATURE_COLS = [
    "ret_1m", "ret_3m", "ret_6m", "volatility_21d",
    "rsi_14", "ma50_ma200_ratio", "volume_trend",
]


# --- Step 1: Universe scan ---------------------------------------------------

def get_universe(cap: int = UNIVERSE_CAP) -> list[str]:
    """Pull tradable US equities from Alpaca and cap the list."""
    trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
    req = GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
    assets = trading_client.get_all_assets(req)
    tickers = [a.symbol for a in assets if a.tradable and a.fractionable]
    tickers = sorted(tickers)[:cap]
    return tickers


# --- Step 2: Feature engineering from price bars ----------------------------

def compute_features(close: pd.Series, volume: pd.Series) -> pd.DataFrame:
    """Rolling technical features computed at every point in time (for
    training) -- not just the latest snapshot."""
    df = pd.DataFrame(index=close.index)
    df["ret_1m"] = close.pct_change(21)
    df["ret_3m"] = close.pct_change(63)
    df["ret_6m"] = close.pct_change(126)
    df["volatility_21d"] = close.pct_change().rolling(21).std()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    df["ma50_ma200_ratio"] = ma50 / (ma200 + 1e-9)

    vol_short = volume.rolling(20).mean()
    vol_long = volume.rolling(90).mean()
    df["volume_trend"] = vol_short / (vol_long + 1e-9)

    df["forward_return"] = close.pct_change(FORWARD_WINDOW).shift(-FORWARD_WINDOW)
    return df


def fetch_bars(tickers: list[str]) -> dict[str, pd.DataFrame]:
    client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    end = datetime.now()
    start = end - timedelta(days=HISTORY_DAYS)

    out = {}
    batch_size = 100  # keep requests reasonably sized
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        req = StockBarsRequest(
            symbol_or_symbols=batch, timeframe=TimeFrame.Day, start=start, end=end
        )
        bars = client.get_stock_bars(req).df
        for t in batch:
            try:
                out[t] = bars.loc[t][["close", "volume"]].copy()
            except KeyError:
                continue
    return out


# --- Step 3: Build training set + train model --------------------------------

def build_dataset(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for ticker, df in bars.items():
        feats = compute_features(df["close"], df["volume"])
        feats["Ticker"] = ticker
        rows.append(feats)
    full = pd.concat(rows)
    return full


def train_model(dataset: pd.DataFrame, model_type: str = MODEL_TYPE):
    train = dataset.dropna(subset=FEATURE_COLS + ["forward_return"])
    X = train[FEATURE_COLS].values
    y = train["forward_return"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if model_type == "mlp":
        model = MLPRegressor(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            alpha=1e-2,           # meaningful L2 regularization -- small data, needs it
            max_iter=2000,
            random_state=42,
        )
    else:
        model = Ridge(alpha=5.0)  # regularized linear model, robust default

    model.fit(X_scaled, y)
    return model, scaler


# --- Step 4: Score current universe ------------------------------------------

def score_latest(dataset: pd.DataFrame, model, scaler) -> pd.DataFrame:
    latest = dataset.groupby("Ticker").tail(1).dropna(subset=FEATURE_COLS)
    X_latest = scaler.transform(latest[FEATURE_COLS].values)
    latest = latest.copy()
    latest["Relevance_Score"] = model.predict(X_latest)
    return latest[["Ticker"] + FEATURE_COLS + ["Relevance_Score"]]


# --- Main ---------------------------------------------------------------------

def run(top_n: int = 20, model_type: str = MODEL_TYPE) -> pd.DataFrame:
    if not API_KEY or not SECRET_KEY:
        raise RuntimeError(
            "Set ALPACA_API_KEY and ALPACA_SECRET_KEY env vars (paper keys are fine)."
        )

    print(f"Scanning universe (cap={UNIVERSE_CAP})...")
    tickers = get_universe()

    print(f"Fetching {HISTORY_DAYS} days of bars for {len(tickers)} tickers...")
    bars = fetch_bars(tickers)

    print("Building feature/label dataset...")
    dataset = build_dataset(bars)

    print(f"Training {model_type} model...")
    model, scaler = train_model(dataset, model_type=model_type)

    print("Scoring current universe...")
    scored = score_latest(dataset, model, scaler)

    ranked = scored.sort_values("Relevance_Score", ascending=False).head(top_n)
    return ranked


if __name__ == "__main__":
    results = run()
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 160)
    print(results.to_string(index=False))
