#!/usr/bin/env python3
"""
Cached Binance Price Data Retriever

Enhanced version with SQLite database caching to avoid repeated API calls.
Perfect for quant analysis and ML model development.
"""

import os
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
import argparse
import time
from pathlib import Path

# Load environment variables
load_dotenv()

class CachedBinanceClient:
    def __init__(self, db_path="db/binance_data.db", api_key=None, api_secret=None, testnet=False):
        """
        Initialize cached Binance client
        
        Args:
            db_path: Path to SQLite database file
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Use testnet instead of live platform
        """
        self.db_path = db_path
        self.api_key = api_key or os.getenv('BINANCE_API_KEY')
        self.api_secret = api_secret or os.getenv('BINANCE_API_SECRET')
        self.testnet = testnet
        
        # Initialize database
        self._init_database()
        
        # Initialize Binance client
        if self.api_key and self.api_secret:
            self.client = Client(self.api_key, self.api_secret, testnet=self.testnet)
        else:
            self.client = Client()
        
        print(f"✅ Cached Binance client initialized")
        print(f"   Database: {self.db_path}")
        print(f"   Testnet: {self.testnet}")
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create price data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                quote_volume REAL,
                trades INTEGER,
                taker_buy_volume REAL,
                taker_buy_quote_volume REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, interval, timestamp)
            )
        ''')
        
        # Create metadata table for tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_metadata (
                symbol TEXT,
                interval TEXT,
                last_timestamp INTEGER,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, interval)
            )
        ''')
        
        # Create indexes for performance
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_symbol_interval 
            ON price_data(symbol, interval)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON price_data(timestamp)
        ''')
        
        conn.commit()
        conn.close()
        
        print(f"✅ Database initialized: {self.db_path}")
    
    def _timestamp_to_datetime(self, timestamp_ms):
        """Convert milliseconds timestamp to datetime"""
        return datetime.fromtimestamp(timestamp_ms / 1000)
    
    def _datetime_to_timestamp(self, dt):
        """Convert datetime to milliseconds timestamp"""
        return int(dt.timestamp() * 1000)
    
    def get_cached_data(self, symbol, interval, start_date=None, end_date=None):
        """
        Retrieve cached data from database
        
        Args:
            symbol: Trading symbol
            interval: Kline interval
            start_date: Start datetime (optional)
            end_date: End datetime (optional)
            
        Returns:
            pd.DataFrame: Cached data or empty DataFrame
        """
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT timestamp, open, high, low, close, volume, quote_volume, 
                   trades, taker_buy_volume, taker_buy_quote_volume
            FROM price_data 
            WHERE symbol = ? AND interval = ?
        '''
        params = [symbol, interval]
        
        if start_date:
            query += ' AND timestamp >= ?'
            params.append(self._datetime_to_timestamp(start_date))
        
        if end_date:
            query += ' AND timestamp <= ?'
            params.append(self._datetime_to_timestamp(end_date))
        
        query += ' ORDER BY timestamp'
        
        try:
            df = pd.read_sql_query(query, conn, params=params)
            
            if not df.empty:
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                print(f"📂 Retrieved {len(df)} cached records for {symbol} {interval}")
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error reading cached data: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def save_to_cache(self, symbol, interval, df):
        """
        Save DataFrame to database cache
        
        Args:
            symbol: Trading symbol
            interval: Kline interval
            df: DataFrame with price data
        """
        if df.empty:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Prepare data for insertion
        records = []
        for timestamp, row in df.iterrows():
            # Convert datetime to milliseconds timestamp
            ts_ms = self._datetime_to_timestamp(timestamp)
            
            record = (
                symbol, interval, ts_ms,
                row['open'], row['high'], row['low'], row['close'], row['volume'],
                row.get('quote_volume', 0), row.get('trades', 0),
                row.get('taker_buy_volume', 0), row.get('taker_buy_quote_volume', 0)
            )
            records.append(record)
        
        # Insert with conflict handling (ignore duplicates)
        insert_sql = '''
            INSERT OR IGNORE INTO price_data 
            (symbol, interval, timestamp, open, high, low, close, volume, 
             quote_volume, trades, taker_buy_volume, taker_buy_quote_volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        
        try:
            cursor.executemany(insert_sql, records)
            conn.commit()
            print(f"💾 Saved {len(records)} records to cache for {symbol} {interval}")
            
        except Exception as e:
            print(f"❌ Error saving to cache: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def get_historical_klines(self, symbol='BTCUSDT', interval='1h', days=30, use_cache=True):
        """
        Get historical kline data with caching
        
        Args:
            symbol: Trading pair symbol
            interval: Kline interval
            days: Number of days of historical data
            use_cache: Whether to use cached data
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Try to get cached data first
        if use_cache:
            cached_df = self.get_cached_data(symbol, interval, start_date, end_date)
            if not cached_df.empty:
                # Check if we have complete data for the requested period
                cached_start = cached_df.index.min()
                cached_end = cached_df.index.max()
                
                # If we have all the data, return cached version
                if cached_start <= start_date and cached_end >= end_date:
                    print(f"🎯 Using cached data for {symbol} {interval}")
                    return cached_df
                else:
                    print(f"🔄 Partial cache found, fetching missing data...")
        
        # Fetch from Binance API
        print(f"📡 Fetching {days} days of {interval} data for {symbol} from Binance...")
        
        try:
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_date.strftime("%d %b, %Y"),
                end_str=end_date.strftime("%d %b, %Y")
            )
            
            if not klines:
                print("❌ No data returned from Binance")
                return pd.DataFrame()
            
            # Convert to DataFrame
            columns = [
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
                'taker_buy_quote_volume', 'ignore'
            ]
            
            df = pd.DataFrame(klines, columns=columns)
            
            # Convert data types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            df['trades'] = df['trades'].astype(int)
            
            # Convert timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            print(f"✅ Retrieved {len(df)} fresh records from Binance")
            
            # Save to cache
            if use_cache:
                self.save_to_cache(symbol, interval, df)
            
            return df
            
        except BinanceAPIException as e:
            print(f"❌ Error fetching historical data: {e}")
            
            # If API fails, try to return cached data as fallback
            if use_cache:
                cached_df = self.get_cached_data(symbol, interval, start_date, end_date)
                if not cached_df.empty:
                    print("🔄 Using cached data as fallback")
                    return cached_df
            
            return pd.DataFrame()
    
    def get_cache_stats(self):
        """Get statistics about cached data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get total records count
            cursor.execute('SELECT COUNT(*) FROM price_data')
            total_records = cursor.fetchone()[0]
            
            # Get records by symbol
            cursor.execute('''
                SELECT symbol, COUNT(*) as count 
                FROM price_data 
                GROUP BY symbol
            ''')
            by_symbol = cursor.fetchall()
            
            # Get records by interval
            cursor.execute('''
                SELECT interval, COUNT(*) as count 
                FROM price_data 
                GROUP BY interval
            ''')
            by_interval = cursor.fetchall()
            
            # Get date range
            cursor.execute('''
                SELECT MIN(timestamp), MAX(timestamp) 
                FROM price_data
            ''')
            min_ts, max_ts = cursor.fetchone()
            
            stats = {
                'total_records': total_records,
                'by_symbol': dict(by_symbol),
                'by_interval': dict(by_interval),
                'date_range': {
                    'start': self._timestamp_to_datetime(min_ts) if min_ts else None,
                    'end': self._timestamp_to_datetime(max_ts) if max_ts else None
                }
            }
            
            return stats
            
        except Exception as e:
            print(f"❌ Error getting cache stats: {e}")
            return {}
        finally:
            conn.close()
    
    def clear_cache(self, symbol=None, interval=None):
        """Clear cached data (optionally filtered by symbol and/or interval)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if symbol and interval:
                cursor.execute('DELETE FROM price_data WHERE symbol = ? AND interval = ?', (symbol, interval))
                print(f"🗑️ Cleared cache for {symbol} {interval}")
            elif symbol:
                cursor.execute('DELETE FROM price_data WHERE symbol = ?', (symbol,))
                print(f"🗑️ Cleared cache for {symbol}")
            else:
                cursor.execute('DELETE FROM price_data')
                print("🗑️ Cleared all cached data")
            
            conn.commit()
            
        except Exception as e:
            print(f"❌ Error clearing cache: {e}")
            conn.rollback()
        finally:
            conn.close()

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Fetch Binance price data with caching')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol (default: BTCUSDT)')
    parser.add_argument('--interval', default='1h', help='Kline interval (default: 1h)')
    parser.add_argument('--days', type=int, default=30, help='Days of historical data (default: 30)')
    parser.add_argument('--no-cache', action='store_true', help='Disable cache (force fresh fetch)')
    parser.add_argument('--stats', action='store_true', help='Show cache statistics')
    parser.add_argument('--clear-cache', action='store_true', help='Clear cached data')
    
    args = parser.parse_args()
    
    # Initialize client
    client = CachedBinanceClient()
    
    # Handle cache operations
    if args.stats:
        stats = client.get_cache_stats()
        print("\n📊 Cache Statistics:")
        print(f"   Total Records: {stats.get('total_records', 0):,}")
        print(f"   By Symbol: {stats.get('by_symbol', {})}")
        print(f"   By Interval: {stats.get('by_interval', {})}")
        if stats.get('date_range', {}).get('start'):
            print(f"   Date Range: {stats['date_range']['start']} to {stats['date_range']['end']}")
        return
    
    if args.clear_cache:
        client.clear_cache(args.symbol)
        return
    
    # Get current price
    current_price = client.client.get_symbol_ticker(symbol=args.symbol)
    if current_price:
        print(f"💰 Current {args.symbol} price: ${float(current_price['price']):,.2f}")
    
    # Get historical data
    df = client.get_historical_klines(
        symbol=args.symbol,
        interval=args.interval,
        days=args.days,
        use_cache=not args.no_cache
    )
    
    if not df.empty:
        # Display basic info
        print(f"\n📈 Data Summary:")
        print(f"   Period: {df.index[0]} to {df.index[-1]}")
        print(f"   Records: {len(df)}")
        print(f"   Price Range: ${df['low'].min():,.2f} - ${df['high'].max():,.2f}")
        print(f"   Current Price: ${df['close'].iloc[-1]:,.2f}")
        
        # Show cache stats
        stats = client.get_cache_stats()
        print(f"   Cached Records: {stats.get('total_records', 0):,}")
        
        # Save to CSV for analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"out/{args.symbol}_{args.interval}_data_{timestamp}.csv"
        df.to_csv(filename)
        print(f"💾 Data saved to {filename}")
        
        print(f"\n🎯 Next steps for quant analysis:")
        print(f"   - Load {filename} for technical analysis")
        print(f"   - Database is cached for fast future access")
        print(f"   - Use --stats to see cache contents")
        print(f"   - Use --no-cache to force fresh data")
        
    else:
        print("❌ No data retrieved")

if __name__ == "__main__":
    main()

