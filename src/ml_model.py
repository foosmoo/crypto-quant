from cached_binance_client import CachedBinanceClient

# Initialize client
client = CachedBinanceClient()

# Get data (uses cache automatically)
df = client.get_historical_klines(
    symbol='BTCUSDT', 
    interval='1h', 
    days=365
)

# Now you have data for ML model training
print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Check cache stats
stats = client.get_cache_stats()
print(f"Cached records: {stats['total_records']:,}")

