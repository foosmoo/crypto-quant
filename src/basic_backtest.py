#!/usr/bin/env python3
"""
Comprehensive Basic Backtesting Example
Testing traditional technical analysis strategies
"""

import pandas as pd
import numpy as np
from backtester import BacktestEngine, TradingStrategies
from cached_binance_client import CachedBinanceClient
from basic_quant_analysis import calculate_technical_indicators
from feature_engineering import FeatureEngineer
import matplotlib.pyplot as plt


def run_strategy_comparison():
    """Compare multiple traditional trading strategies"""
    print("🚀 Starting Strategy Comparison Backtest...")
    
    # Initialize client and get data
    client = CachedBinanceClient()
    df = client.get_historical_klines(symbol="BTCUSDT", interval="4h", days=180)  # 6 months
    
    if df.empty:
        print("❌ No data available for backtesting")
        return
    
    # Calculate technical indicators and features
    print("📊 Calculating indicators and features...")
    df = calculate_technical_indicators(df)
    feature_engineer = FeatureEngineer()
    df = feature_engineer.engineer_all_features(df)
    
    # Define strategies to test
    strategies = {
        "MA_Crossover_5_20": {
            'function': TradingStrategies.moving_average_crossover,
            'params': {'fast_window': 5, 'slow_window': 20},
            'color': 'blue'
        },
        "MA_Crossover_10_50": {
            'function': TradingStrategies.moving_average_crossover,
            'params': {'fast_window': 10, 'slow_window': 50},
            'color': 'green'
        },
        "RSI_Momentum_14": {
            'function': TradingStrategies.rsi_momentum,
            'params': {'rsi_period': 14, 'oversold': 30, 'overbought': 70},
            'color': 'red'
        },
        "RSI_Momentum_21": {
            'function': TradingStrategies.rsi_momentum,
            'params': {'rsi_period': 21, 'oversold': 25, 'overbought': 75},
            'color': 'orange'
        }
    }
    
    # Run backtests for each strategy
    results = {}
    backtester = BacktestEngine(initial_capital=10000, commission=0.001)
    
    for strategy_name, strategy_config in strategies.items():
        print(f"\n🔹 Testing {strategy_name}...")
        
        results[strategy_name] = backtester.run_backtest(
            df=df,
            strategy_func=strategy_config['function'],
            strategy_params=strategy_config['params'],
            position_size=0.1,  # 10% per trade
            stop_loss=0.02,     # 2% stop loss
            take_profit=0.05    # 5% take profit
        )
        
        # Generate individual reports
        backtester.generate_report(symbol=f"BTCUSDT_{strategy_name}")
    
    # Compare strategies
    compare_strategies(results, strategies)
    
    return results


def compare_strategies(results, strategies):
    """Compare performance of all tested strategies"""
    print("\n" + "="*80)
    print("STRATEGY COMPARISON RESULTS")
    print("="*80)
    
    comparison_data = []
    
    for strategy_name, result in results.items():
        comparison_data.append({
            'Strategy': strategy_name,
            'Total Return (%)': result['total_return_pct'],
            'Win Rate (%)': result['win_rate'],
            'Total Trades': result['total_trades'],
            'Avg Trade Return (%)': result['avg_trade_return_pct'],
            'Max Drawdown (%)': result['max_drawdown_pct'],
            'Sharpe Ratio': result['sharpe_ratio'],
            'Profit Factor': result['profit_factor']
        })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Total Return (%)', ascending=False)
    
    # Print comparison table
    print("\n📊 Performance Comparison (Sorted by Total Return):")
    print(comparison_df.to_string(index=False, float_format='%.2f'))
    
    # Plot equity curve comparison
    plot_equity_comparison(results, strategies)
    
    # Find best performing strategy
    best_strategy = comparison_df.iloc[0]
    print(f"\n🏆 Best Performing Strategy: {best_strategy['Strategy']}")
    print(f"   Total Return: {best_strategy['Total Return (%)']:.2f}%")
    print(f"   Win Rate: {best_strategy['Win Rate (%)']:.1f}%")
    print(f"   Sharpe Ratio: {best_strategy['Sharpe Ratio']:.2f}")
    
    return comparison_df


def plot_equity_comparison(results, strategies):
    """Plot equity curve comparison for all strategies"""
    plt.figure(figsize=(12, 8))
    
    for strategy_name, result in results.items():
        equity_curve = pd.DataFrame(result['equity_curve'])
        equity_curve.set_index('timestamp', inplace=True)
        
        plt.plot(equity_curve.index, 
                equity_curve['portfolio_value'], 
                label=strategy_name, 
                linewidth=2,
                color=strategies[strategy_name]['color'])
    
    plt.title('Strategy Equity Curve Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save comparison plot
    plt.savefig('out/strategy_comparison.png', dpi=150, bbox_inches='tight')
    print("💾 Strategy comparison plot saved to 'out/strategy_comparison.png'")
    plt.show()


def optimize_ma_parameters(df):
    """Optimize moving average crossover parameters"""
    print("\n🔄 Optimizing MA Crossover Parameters...")
    
    fast_windows = [5, 10, 15, 20]
    slow_windows = [20, 30, 50, 100]
    
    optimization_results = []
    backtester = BacktestEngine(initial_capital=10000, commission=0.001)
    
    for fast in fast_windows:
        for slow in slow_windows:
            if fast >= slow:
                continue  # Fast MA must be smaller than slow MA
                
            strategy_name = f"MA_{fast}_{slow}"
            print(f"   Testing MA{fast}-MA{slow}...")
            
            try:
                result = backtester.run_backtest(
                    df=df,
                    strategy_func=TradingStrategies.moving_average_crossover,
                    strategy_params={'fast_window': fast, 'slow_window': slow},
                    position_size=0.1
                )
                
                optimization_results.append({
                    'Fast_MA': fast,
                    'Slow_MA': slow,
                    'Total_Return': result['total_return_pct'],
                    'Win_Rate': result['win_rate'],
                    'Sharpe_Ratio': result['sharpe_ratio'],
                    'Max_Drawdown': result['max_drawdown_pct']
                })
                
            except Exception as e:
                print(f"      Error with MA{fast}-MA{slow}: {e}")
    
    # Find best parameters
    optimization_df = pd.DataFrame(optimization_results)
    if not optimization_df.empty:
        best_params = optimization_df.loc[optimization_df['Total_Return'].idxmax()]
        
        print(f"\n🎯 Best MA Parameters:")
        print(f"   Fast MA: {best_params['Fast_MA']}, Slow MA: {best_params['Slow_MA']}")
        print(f"   Total Return: {best_params['Total_Return']:.2f}%")
        print(f"   Sharpe Ratio: {best_params['Sharpe_Ratio']:.2f}")
        
        # Save optimization results
        optimization_df.to_csv('out/ma_optimization_results.csv', index=False)
        print("💾 Optimization results saved to 'out/ma_optimization_results.csv'")
        
        return best_params
    
    return None


def main():
    """Main function to run comprehensive basic backtesting"""
    print("="*80)
    print("BASIC STRATEGY BACKTESTING SUITE")
    print("="*80)
    
    # Run strategy comparison
    results = run_strategy_comparison()
    
    # Get data for optimization (use same data as comparison)
    client = CachedBinanceClient()
    df = client.get_historical_klines(symbol="BTCUSDT", interval="4h", days=180)
    df = calculate_technical_indicators(df)
    
    # Run parameter optimization
    best_params = optimize_ma_parameters(df)
    
    print("\n🎯 Next Steps:")
    print("   - Review individual strategy reports in 'out/' directory")
    print("   - Examine trade logs for each strategy")
    print("   - Use best parameters for live trading simulation")
    print("   - Test with different position sizes and risk parameters")


if __name__ == "__main__":
    main()

# -- EOF --
