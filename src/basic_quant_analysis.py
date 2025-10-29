#!/usr/bin/env python3
"""
Example quant analysis using cached Binance data
Perfect for developing trading strategies and ML models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cached_binance_client import CachedBinanceClient


def calculate_technical_indicators(df):
    """Calculate common technical indicators for quant analysis"""
    print("📊 Calculating technical indicators...")

    # Price-based indicators
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    # Moving averages
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["sma_50"] = df["close"].rolling(window=50).mean()
    df["ema_12"] = df["close"].ewm(span=12).mean()
    df["ema_26"] = df["close"].ewm(span=26).mean()

    # Bollinger Bands
    df["bb_middle"] = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
    df["bb_lower"] = df["bb_middle"] - (bb_std * 2)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df["close"].ewm(span=12).mean()
    exp2 = df["close"].ewm(span=26).mean()
    df["macd"] = exp1 - exp2
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    # Volume indicators
    df["volume_sma"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma"]

    # Volatility
    df["volatility"] = df["returns"].rolling(20).std() * np.sqrt(365 * 24)  # Annualized

    return df


def detect_simple_patterns(df):
    """Detect simple price patterns for strategy development"""
    print("🔍 Detecting price patterns...")

    patterns = {}

    # Golden Cross / Death Cross
    patterns["golden_cross"] = (df["sma_20"] > df["sma_50"]) & (
        df["sma_20"].shift(1) <= df["sma_50"].shift(1)
    )
    patterns["death_cross"] = (df["sma_20"] < df["sma_50"]) & (
        df["sma_20"].shift(1) >= df["sma_50"].shift(1)
    )

    # RSI signals
    patterns["rsi_oversold"] = df["rsi"] < 30
    patterns["rsi_overbought"] = df["rsi"] > 70

    # Bollinger Band signals
    patterns["bb_oversold"] = df["close"] < df["bb_lower"]
    patterns["bb_overbought"] = df["close"] > df["bb_upper"]

    # MACD signals
    patterns["macd_bullish"] = (df["macd"] > df["macd_signal"]) & (
        df["macd"].shift(1) <= df["macd_signal"].shift(1)
    )
    patterns["macd_bearish"] = (df["macd"] < df["macd_signal"]) & (
        df["macd"].shift(1) >= df["macd_signal"].shift(1)
    )

    # High volume moves
    patterns["high_volume_up"] = (df["volume_ratio"] > 2) & (df["returns"] > 0.01)
    patterns["high_volume_down"] = (df["volume_ratio"] > 2) & (df["returns"] < -0.01)

    return patterns


def analyze_pattern_performance(df, patterns):
    """Analyze performance of detected patterns"""
    print("📈 Analyzing pattern performance...")

    results = {}

    for pattern_name, pattern_signal in patterns.items():
        if pattern_signal.any():
            # Get returns after pattern signals
            future_returns = df["returns"].shift(-1)  # Next period return

            pattern_returns = future_returns[pattern_signal]

            if len(pattern_returns) > 0:
                results[pattern_name] = {
                    "occurrences": len(pattern_returns),
                    "avg_return": pattern_returns.mean() * 100,  # Percentage
                    "win_rate": (pattern_returns > 0).mean() * 100,
                    "total_return": pattern_returns.sum() * 100,
                }

    return results


def plot_analysis(df, patterns, symbol):
    """Create analysis plots"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))

    # Price chart with indicators
    axes[0].plot(df.index, df["close"], label="Close Price", linewidth=1, color="black")
    axes[0].plot(df.index, df["sma_20"], label="SMA 20", linewidth=0.8, alpha=0.7)
    axes[0].plot(df.index, df["sma_50"], label="SMA 50", linewidth=0.8, alpha=0.7)
    axes[0].fill_between(
        df.index, df["bb_upper"], df["bb_lower"], alpha=0.2, label="Bollinger Bands"
    )

    # Mark pattern signals
    for pattern_name, pattern_signal in patterns.items():
        if (
            "bullish" in pattern_name
            or "oversold" in pattern_name
            or "golden" in pattern_name
        ):
            signal_points = df[pattern_signal]
            if len(signal_points) > 0:
                axes[0].scatter(
                    signal_points.index,
                    signal_points["close"],
                    label=pattern_name,
                    alpha=0.6,
                    s=30,
                )

    axes[0].set_title(f"{symbol} Price Analysis")
    axes[0].set_ylabel("Price (USD)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # RSI
    axes[1].plot(df.index, df["rsi"], label="RSI", linewidth=1, color="purple")
    axes[1].axhline(y=70, color="r", linestyle="--", alpha=0.7, label="Overbought")
    axes[1].axhline(y=30, color="g", linestyle="--", alpha=0.7, label="Oversold")
    axes[1].set_ylabel("RSI")
    axes[1].set_ylim(0, 100)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Volume
    axes[2].bar(df.index, df["volume"], alpha=0.7, color="orange", label="Volume")
    axes[2].plot(
        df.index, df["volume_sma"], label="Volume SMA", color="red", linewidth=1
    )
    axes[2].set_ylabel("Volume")
    axes[2].set_xlabel("Date")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"out/{symbol}_quant_analysis.png", dpi=150, bbox_inches="tight")
    print(f"💾 Analysis chart saved as {symbol}_quant_analysis.png")
    plt.show()


def main():
    """Main analysis function"""
    # Initialize cached client
    client = CachedBinanceClient()

    # Get data (will use cache if available)
    print("🚀 Starting quant analysis...")
    df = client.get_historical_klines(symbol="BTCUSDT", interval="1h", days=90)

    if df.empty:
        print("❌ No data available for analysis")
        return

    # Calculate technical indicators
    df = calculate_technical_indicators(df)

    # Detect patterns
    patterns = detect_simple_patterns(df)

    # Analyze performance
    results = analyze_pattern_performance(df, patterns)

    # Print results
    print("\n" + "=" * 60)
    print("QUANT ANALYSIS RESULTS")
    print("=" * 60)

    for pattern, stats in results.items():
        print(f"\n🔹 {pattern.upper()}:")
        print(f"   Occurrences: {stats['occurrences']}")
        print(f"   Average Return: {stats['avg_return']:.2f}%")
        print(f"   Win Rate: {stats['win_rate']:.1f}%")
        print(f"   Total Return: {stats['total_return']:.2f}%")

    # Show basic stats
    print(f"\n📊 Dataset Statistics:")
    print(f"   Period: {df.index[0].date()} to {df.index[-1].date()}")
    print(
        f"   Total Return: {(df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100:.2f}%"
    )
    print(f"   Volatility: {df['volatility'].mean() * 100:.2f}%")
    print(f"   Average RSI: {df['rsi'].mean():.1f}")

    # Create plots
    plot_analysis(df, patterns, "BTCUSDT")

    # Save analyzed data for ML
    df.to_csv("out/btc_quant_analysis_data.csv")
    print(f"\n💾 Analysis data saved to 'btc_quant_analysis_data.csv'")
    print(f"   Shape: {df.shape}")
    print(f"   Features: {list(df.columns)}")

    print(f"\n🎯 Ready for ML model development!")
    print(f"   Use 'btc_quant_analysis_data.csv' for training")
    print(f"   Database is cached for fast iteration")


if __name__ == "__main__":
    main()
