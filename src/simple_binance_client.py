#!/usr/bin/env python3
"""
Simple Binance Price Data Retriever

A minimal script to fetch BTC price data from Binance.
"""

import os
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv
import argparse

# Load environment variables
load_dotenv()


class SimpleBinanceClient:
    def __init__(self, api_key=None, api_secret=None, testnet=False):
        """
        Initialize Binance client

        Args:
            api_key: Binance API key (optional for public data)
            api_secret: Binance API secret (optional for public data)
            testnet: Use testnet instead of live platform
        """
        self.api_key = api_key or os.getenv("BINANCE_API_KEY")
        self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET")
        self.testnet = testnet

        # Initialize client
        if self.api_key and self.api_secret:
            self.client = Client(self.api_key, self.api_secret, testnet=self.testnet)
        else:
            # Public client for basic data (rate limited)
            self.client = Client()

        print(f"✅ Binance client initialized (Testnet: {self.testnet})")

    def get_current_price(self, symbol="BTCUSDT"):
        """
        Get current price for a symbol

        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT)

        Returns:
            float: Current price
        """
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker["price"])
        except BinanceAPIException as e:
            print(f"❌ Error getting price for {symbol}: {e}")
            return None

    def get_historical_klines(self, symbol="BTCUSDT", interval="1h", days=30):
        """
        Get historical kline data

        Args:
            symbol: Trading pair symbol
            interval: Kline interval (1m, 5m, 1h, 1d, etc.)
            days: Number of days of historical data

        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        try:
            # Calculate start date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            print(f"📊 Fetching {days} days of {interval} data for {symbol}...")

            # Fetch klines
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_date.strftime("%d %b, %Y"),
                end_str=end_date.strftime("%d %b, %Y"),
            )

            # Convert to DataFrame
            columns = [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_buy_volume",
                "taker_buy_quote_volume",
                "ignore",
            ]

            df = pd.DataFrame(klines, columns=columns)

            # Convert data types
            numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume"]
            df[numeric_cols] = df[numeric_cols].astype(float)
            df["trades"] = df["trades"].astype(int)

            # Convert timestamps
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

            # Set timestamp as index
            df.set_index("timestamp", inplace=True)

            print(f"✅ Retrieved {len(df)} records")
            return df

        except BinanceAPIException as e:
            print(f"❌ Error fetching historical data: {e}")
            return pd.DataFrame()

    def get_multiple_timeframes(self, symbol="BTCUSDT", days=30):
        """
        Get data for multiple timeframes

        Args:
            symbol: Trading pair symbol
            days: Number of days of data

        Returns:
            dict: Dictionary of DataFrames for different timeframes
        """
        timeframes = {
            "1h": Client.KLINE_INTERVAL_1HOUR,
            "4h": Client.KLINE_INTERVAL_4HOUR,
            "1d": Client.KLINE_INTERVAL_1DAY,
        }

        data = {}
        for tf_name, tf_value in timeframes.items():
            print(f"🔄 Fetching {tf_name} data...")
            df = self.get_historical_klines(symbol, tf_value, days)
            if not df.empty:
                data[tf_name] = df
            time.sleep(0.2)  # Rate limiting

        return data

    def save_to_csv(self, df, filename=None):
        """
        Save DataFrame to CSV

        Args:
            df: DataFrame to save
            filename: Output filename (optional)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"out/binance_data_{timestamp}.csv"

        df.to_csv(filename)
        print(f"💾 Data saved to {filename}")
        return filename


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Fetch Binance price data")
    parser.add_argument(
        "--symbol", default="BTCUSDT", help="Trading symbol (default: BTCUSDT)"
    )
    parser.add_argument("--interval", default="1h", help="Kline interval (default: 1h)")
    parser.add_argument(
        "--days", type=int, default=30, help="Days of historical data (default: 30)"
    )
    parser.add_argument("--output", help="Output CSV filename")
    parser.add_argument("--testnet", action="store_true", help="Use Binance testnet")

    args = parser.parse_args()

    # Initialize client
    client = SimpleBinanceClient(testnet=args.testnet)

    # Get current price
    current_price = client.get_current_price(args.symbol)
    if current_price:
        print(f"💰 Current {args.symbol} price: ${current_price:,.2f}")

    # Get historical data
    df = client.get_historical_klines(
        symbol=args.symbol, interval=args.interval, days=args.days
    )

    if not df.empty:
        # Display basic info
        print(f"\n📈 Data Summary:")
        print(f"   Period: {df.index[0]} to {df.index[-1]}")
        print(f"   Records: {len(df)}")
        print(f"   Price Range: ${df['low'].min():,.2f} - ${df['high'].max():,.2f}")
        print(f"   Volume Avg: {df['volume'].mean():.2f} {args.symbol}")

        # Save to CSV
        filename = client.save_to_csv(df, args.output)

        # Show first few rows
        print(f"\n📋 First 5 rows:")
        print(df[["open", "high", "low", "close", "volume"]].head())

        print(f"\n🎯 Next steps:")
        print(f"   - Analyze the data in {filename}")
        print(f"   - Use different intervals (1m, 5m, 1h, 1d)")
        print(f"   - Add technical indicators")
        print(f"   - Set up automated data collection")
    else:
        print("❌ No data retrieved")


if __name__ == "__main__":
    main()
