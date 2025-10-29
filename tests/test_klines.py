# Import datetime and timedelta to calculate the expected mock strings
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from cached_binance_client import CachedBinanceClient  # Your local class

# The core problem is that the Client() constructor calls 'ping()'
# and fails. We need to mock the Client class itself before it's used.

# We must patch the 'Client' class *where it is looked up* by CachedBinanceClient.


# We use a nested patch to mock the external API client AND freeze the time.
# The datetime patch target assumes your wrapper imports: `from datetime import datetime, timedelta`
@patch("cached_binance_client.datetime")
@patch("cached_binance_client.Client")
def test_get_historical_klines_mocked(MockClient, MockDatetime):
    """
    Test the function by mocking the external Binance Client object and validating
    that the wrapper function correctly interacts with the external client,
    including deterministic date string conversion.
    """

    # 1. Setup the Time Mock
    MOCK_NOW = datetime(
        2025, 10, 28, 10, 0, 0
    )  # Define a fixed "now" for deterministic testing (e.g., Oct 28, 2025)
    MockDatetime.now.return_value = MOCK_NOW
    MockDatetime.timedelta = (
        timedelta  # Ensure timedelta is available on the mocked datetime module
    )

    # 2. Setup the External Client Mock
    mock_binance_instance = MockClient.return_value

    # Configure the mocked method to return dummy data instead of calling the API.
    mock_binance_instance.get_historical_klines.return_value = [
        [
            1678886400000,  # [0] Open time (timestamp)
            "200.00",  # [1] Open
            "201.00",  # [2] High
            "199.00",  # [3] Low
            "200.50",  # [4] Close
            "15.000",  # [5] Volume
            1678886459999,  # [6] Close time (unused by your table)
            "3000.00",  # [7] Quote asset volume (quote_volume)
            100,  # [8] Number of trades (trades)
            "8.000",  # [9] Taker buy base asset volume (taker_buy_volume)
            "1600.00",  # [10] Taker buy quote asset volume (taker_buy_quote_volume)
            "0",  # [11] Ignore
        ],
        [
            1678886460000,  # [0] Open time (timestamp)
            "200.50",  # [1] Open
            "202.00",  # [2] High
            "200.00",  # [3] Low
            "201.50",  # [4] Close
            "20.000",  # [5] Volume
            1678886519999,  # [6] Close time (unused by your table)
            "4000.00",  # [7] Quote asset volume (quote_volume)
            150,  # [8] Number of trades (trades)
            "12.000",  # [9] Taker buy base asset volume (taker_buy_volume)
            "2400.00",  # [10] Taker buy quote asset volume (taker_buy_quote_volume)
            "0",  # [11] Ignore
        ],
    ]

    # 3. Define Test Arguments and Expected API Strings
    TEST_SYMBOL = "BTCUSDT"
    TEST_INTERVAL = "1h"
    TEST_DAYS = 90
    TEST_USE_CACHE = True

    # Calculate the exact expected strings based on the MOCK_NOW time and your wrapper's logic
    # start_date = MOCK_NOW - timedelta(days=90) -> 30 Jul, 2025
    EXPECTED_START_DATE_STRING = (MOCK_NOW - timedelta(days=TEST_DAYS)).strftime(
        "%d %b, %Y"
    )
    # end_date = MOCK_NOW -> 28 Oct, 2025
    EXPECTED_END_DATE_STRING = MOCK_NOW.strftime("%d %b, %Y")

    # 4. Create binance client and retrieve data frame
    client = CachedBinanceClient()  # This now runs without calling self.ping()!

    df = client.get_historical_klines(
        symbol=TEST_SYMBOL,
        interval=TEST_INTERVAL,
        days=TEST_DAYS,
        use_cache=TEST_USE_CACHE,
    )

    # 5. Assertions
    assert df is not None
    assert len(df) == 2
    # assert df[0][0] == 1678886400000
    # assert df[0][1] == '200.00'

    # Assert that the mocked external client was called with the correctly formatted
    # and expected start_str and end_str arguments.
    mock_binance_instance.get_historical_klines.assert_called_once_with(
        symbol=TEST_SYMBOL,
        interval=TEST_INTERVAL,
        start_str=EXPECTED_START_DATE_STRING,
        end_str=EXPECTED_END_DATE_STRING,
    )


# -- EOF --
