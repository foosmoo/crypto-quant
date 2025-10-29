#!/usr/bin/env python3
"""
Run both basic and ML backtesting sequentially
"""

from basic_backtest_example import main as run_basic_backtest
from ml_backtest_example import main as run_ml_backtest

if __name__ == "__main__":
    print("🚀 Running Complete Backtesting Suite...")
    
    # Run basic strategies
    run_basic_backtest()
    
    # Run ML strategies  
    run_ml_backtest()
    
    print("✅ All backtesting completed! Check 'out/' directory for results.")

# -- EOF --
