#!/usr/bin/env python3
"""
Basic analysis of Binance price data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from simple_binance_client import SimpleBinanceClient


def basic_analysis(df, symbol="BTCUSDT"):
    """
    Perform basic analysis on price data
    """
    print("🔍 Performing basic analysis...")

    # Basic statistics
    print(f"\n📊 Basic Statistics for {symbol}:")
    print(f"   Dataset Period: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"   Total Records: {len(df)}")
    print(f"   Price Range: ${df['low'].min():,.2f} - ${df['high'].max():,.2f}")
    print(f"   Current Price: ${df['close'].iloc[-1]:,.2f}")

    # Calculate returns
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    # Volatility (annualized)
    volatility = df["returns"].std() * np.sqrt(365 * 24)  # Assuming hourly data
    print(f"   Annualized Volatility: {volatility:.2%}")

    # Basic metrics
    total_return = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    print(f"   Total Return: {total_return:.2f}%")

    # Volume analysis
    avg_volume = df["volume"].mean()
    print(f"   Average Volume: {avg_volume:.2f}")

    return df


def plot_basic_charts(df, symbol="BTCUSDT"):
    """
    Create basic price and volume charts
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Price chart
    ax1.plot(df.index, df["close"], label="Close Price", linewidth=1)
    ax1.set_title(f"{symbol} Price Chart")
    ax1.set_ylabel("Price (USD)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Volume chart
    ax2.bar(df.index, df["volume"], alpha=0.7, color="orange", label="Volume")
    ax2.set_title(f"{symbol} Volume")
    ax2.set_ylabel("Volume")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"out/{symbol}_basic_charts.png", dpi=150, bbox_inches="tight")
    print(f"💾 Charts saved as out/{symbol}_basic_charts.png")
    plt.show()


if __name__ == "__main__":
    # Initialize client and get data
    client = SimpleBinanceClient()

    # Get 30 days of hourly data
    df = client.get_historical_klines(symbol="BTCUSDT", interval="1h", days=30)

    if not df.empty:
        # Perform analysis
        df = basic_analysis(df)

        # Create charts
        plot_basic_charts(df)

        # Save analyzed data
        client.save_to_csv(df, "out/btc_analyzed_data.csv")

        print(f"\n✅ Analysis complete!")
        print(f"   Data shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
