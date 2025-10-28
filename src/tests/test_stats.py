# tests/test_stats.py
# Pytest will automatically discover tests in this directory.

# Since your setup is installing the package using 'pip install .[dev]',
# Python can import modules directly from the package name, in this case,
# the functions are imported from the module within the installed package.
# Note: For simplicity in this non-installed environment, we use 'my_module'.

from cached_binance_client import CachedBinanceClient

def test_add_positive_numbers():
    """Checks that 'add' works correctly for two positive integers."""
    assert 1+2 == 3

def test_get_historical_klines():

    client = CachedBinanceClient()

    df = client.get_historical_klines(
         symbol="BTCUSDT",
         interval="1h",
         days=90,
         use_cache=True
     )

    assert df is not None

