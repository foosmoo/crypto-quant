
##
# Aliases
#
get-btc-prices:
	cached-binance --symbol BTCUSDT --days 90

get-btc-prices-no-cache:
	cached-binance --symbol BTCUSDT --days 90 --no-cache

btc-quant:
	quant-analysis

clear-btc-cache:
	cached-binance --clear-cache --symbol BTCUSDT

clear-cache:
	cached-binance --clear-cache

stats:
	cached-binance --stats

backtest:
	backtest-run

backtest-demo:
	python src/basic_backtest.py

##
# functional testing
#
test:
	pytest --cov=src

flake:
	flake8 --max-line-length=120 src

black:
	black --check src

# Config and installation

venv:
	python3 -m venv venv
	@echo run this: 'source venv/bin/activate'
	@echo and be sure to run: 'make install'

install-dev: install
	pip install -e .[dev]

install:
	python3 -m pip install --upgrade pip
	pip install -e .
	#pip uninstall urllib3
	#pip install 'urllib3<2.0' # for older mac, mac air m1/m2 
	mkdir -p out db

clean:
	rm -rf out/*.{png,csv,json} src/*.egg-info
