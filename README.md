# A Caching Price-Data Retrival Client & Quant Analysis Tools

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

### Intallation via pip
#### Basic client
```bash
pip install cached-binance-client
```

#### With Analysis Tools
```bash
pip install "cached-binance-client[analysis]"
```

#### With ML Dependencies
```bash
pip install "cached-binance-client[ml]"
```

#### Development Installation
```bash
git clone https://github.com/foosmoo/cached-binance-client
cd cached-binance-client
pip install -e ".[dev,analysis]"
```

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
