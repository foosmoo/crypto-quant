# A Caching Price-Data Retrival Client & Quant Analysis Tool

Quantitative analysis platform built in Python, designed for crypto price data. Platform includes strategy development an ML model training, leveraging Python Torch.

## Features

- **Cached Data Retrieval** - SQLite caching 
- **Quant Analysis** - Technical indicators and pattern detection
- **Pattern verification** - Strategy development and backtesting
- **ML** - Data separation supports pluggable machine learning models
- **Simple Storage** - SQLite database for price data

## Installation

### Basic Installation via make

> If you're wanting to use venv, then run the following:

```bash
make venv
source venv/bin/activate
```

> The `install` target is a dependency of `install-dev`. Chose the former if you don't want the dev, analytics & ml dependencies.

```bash
make install-dev
```

### Intallation via pip

|Regime|How to do it|What it does|
|---|---|---|
|Basic|`pip install -e .`|Installs basic client dependencies for the runtime|
|Analysis|`pip install -e .[analysis]`|Installs required libraries for analysis|
|ML|`pip install -e .[ml]`|Installs required libraries for machine learning|
|Dev|`pip install -e .[dev,ml,analysis]`|Installs testing & linting libraries as well as the ML and Analytics libs|

## Configuration

> [!CAUTION]
> Do not check in your .env file or any code with keys.<br/>
> This should be prevented by `.gitignore` and CodeQL, which'll fail the commit if it detects secrets

>[!TIP]
> Another layer of protection can be added at the time an API key is created by only allowing the key to
> have Read Only permissions and also by whitelisting access to the Binance API so that your key is rejected
> unless the API requests originate from your own specific IPs.

Create an **.env** file with your api keys. Refer to `.env.example` as a template

1. `bash $ cat .env.example > .env`
2. Update `.env` with relevant Binance API keys (see recommendation above re permissions and whitelisting)

```bash
BINANCE_API_KEY="your_api_key"
BINANCE_API_SECRET="your_api_secret"
```

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

