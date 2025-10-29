```bash
crypto-quant
├── pyproject.toml
├── README.md
├── .env.example
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── cached_binance_client.py    # Caching client
│   ├── basic_quant_analysis.py     # Basic quant functions
│   ├── ml_model.py                 # Machine Learning module
│   ├── feature_engineering.py      # New
│   ├── pattern_detector.py         # New
│   ├── backtester.py               # New
│   └── utils/
│       ├── __init__.py
│       ├── indicators.py
│       └── risk_management.py
├── tests/
│   ├── __init__.py
│   ├── test_client.py
│   └── test_analysis.py
├── notebooks/
│   ├── exploration.ipynb
│   └── model_training.ipynb
├── db/                             # SQLite DB location for caching price series data
├── out/                            # Output location for content downloaded from Binance and location for generated temp png files
└── config/
    └── trading_config.yaml
```
