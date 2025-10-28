# A Caching Price-Data Retrival Client & Quant Analysis Tool

Quantitative analysis platform built in Python, designed for crypto price data. Platform includes strategy development an ML model training, leveraging Python Torch.

## Features

- **Cached Data Retrieval** - SQLite caching 
- **Quant Analysis** - Technical indicators and pattern detection
- **Pattern verification** - Strategy development and backtesting
- **ML** - Data separation supports pluggable machine learning models
- **Simple Storage** - SQLite database for price data

## Installation

### Basic Installation

> ℹ️  if you're wanting to use venv, then run the following
```bash
make venv
source venv/bin/activate
```

> ⚠️  If you're not running make install, then be sure to the `out` and `db` directories exist.

```bash
make install
```

> Create **.env** file with your api keys. Refer to `.env.example` as a template
```bash
BINANCE_API_KEY="your_api_key"
BINANCE_API_SECRET="your_api_secret"
```

### Intallation via pip

|Regime|How to do it|What it does|
|---|---|---|
|Basic|`pip install -e .`|Installs basic client dependencies for the runtime|
|Analysis|`pip install -e .[analysis]`|Installs required libraries for analysis|
|ML|`pip install -e .[ml]`|Installs required libraries for machine learning|
|Dev|`pip install -e .[dev,ml,analysis]`|Installs testing & linting libraries as well as the ML and Analysis libs|

## Configuration

1. Copy .env.example to .env
2. Update .env with relevant Binance API keys

## Quick Start Guide

### Get cached BTC data
```bash 
cached-binance --symbol BTCUSDT --days 90
```

### Run quant analysis
```bash 
quant-analysis
```

### Get fresh data (ignore cache)
```bash 
cached-binance --symbol ETHUSDT --days 30 --no-cache
```

### View cache statistics
```bash 
cached-binance --stats
```

### Client Help
```bash
binance-client --help
```
