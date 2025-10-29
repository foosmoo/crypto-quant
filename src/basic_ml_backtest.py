#!/usr/bin/env python3
"""
Comprehensive ML Strategy Backtesting Example
Testing machine learning-based trading strategies
"""

import pandas as pd
import numpy as np
from backtester import BacktestEngine, TradingStrategies
from cached_binance_client import CachedBinanceClient
from basic_quant_analysis import calculate_technical_indicators
from feature_engineering import FeatureEngineer
from ml_model import BTCPricePredictor
import matplotlib.pyplot as plt
from datetime import datetime
import joblib


class MLBacktestSuite:
    """Comprehensive ML strategy backtesting suite"""
    
    def __init__(self, initial_capital=10000, commission=0.001):
        self.backtester = BacktestEngine(initial_capital, commission)
        self.client = CachedBinanceClient()
        self.feature_engineer = FeatureEngineer()
        
    def prepare_ml_data(self, symbol="BTCUSDT", interval="4h", days=365):
        """Prepare data for ML training and backtesting"""
        print("📊 Preparing ML data...")
        
        # Get historical data
        df = self.client.get_historical_klines(
            symbol=symbol, 
            interval=interval, 
            days=days
        )
        
        if df.empty:
            raise ValueError("No data retrieved from Binance")
        
        # Calculate indicators and features
        df = calculate_technical_indicators(df)
        df = self.feature_engineer.engineer_all_features(df)
        
        # Split data for training and testing (time-series split)
        split_index = int(len(df) * 0.7)  # 70% for training, 30% for testing
        train_data = df.iloc[:split_index].copy()
        test_data = df.iloc[split_index:].copy()
        
        print(f"   Training period: {train_data.index[0]} to {train_data.index[-1]}")
        print(f"   Testing period: {test_data.index[0]} to {test_data.index[-1]}")
        print(f"   Training samples: {len(train_data)}, Testing samples: {len(test_data)}")
        
        return train_data, test_data, df
    
    def train_ml_models(self, train_data):
        """Train multiple ML models for comparison"""
        print("\n🤖 Training ML Models...")
        
        models = {}
        
        # Random Forest
        print("   Training Random Forest...")
        rf_predictor = BTCPricePredictor(model_type='random_forest', lookback_periods=10)
        rf_accuracy, rf_precision = rf_predictor.train(train_data, "random_forest")
        models['Random_Forest'] = rf_predictor
        
        # Gradient Boosting
        print("   Training Gradient Boosting...")
        gb_predictor = BTCPricePredictor(model_type='gradient_boosting', lookback_periods=10)
        gb_accuracy, gb_precision = gb_predictor.train(train_data, "gradient_boosting")
        models['Gradient_Boosting'] = gb_predictor
        
        # Save models
        for model_name, predictor in models.items():
            predictor.save_model(f"out/{model_name.lower()}_model.pkl")
        
        return models
    
    def backtest_ml_strategies(self, test_data, models):
        """Backtest multiple ML strategies"""
        print("\n🧪 Backtesting ML Strategies...")
        
        results = {}
        
        for model_name, predictor in models.items():
            print(f"   Backtesting {model_name}...")
            
            # Create strategy function for this specific model
            def ml_strategy_wrapper(data, model_predictor=predictor):
                return BTCPricePredictor.ml_strategy_wrapper(data, model_predictor)
            
            # Run backtest
            results[model_name] = self.backtester.run_backtest(
                df=test_data,
                strategy_func=ml_strategy_wrapper,
                strategy_params={},
                position_size=0.1,
                stop_loss=0.02,
                take_profit=0.05
            )
            
            # Generate individual report
            self.backtester.generate_report(symbol=f"BTCUSDT_ML_{model_name}")
            
            # Plot individual results
            self.backtester.plot_results(symbol=f"BTCUSDT_ML_{model_name}")
        
        return results
    
    def compare_ml_vs_traditional(self, test_data, ml_results, traditional_results):
        """Compare ML strategies vs traditional strategies"""
        print("\n📈 Comparing ML vs Traditional Strategies...")
        
        # Combine results
        all_results = {**ml_results, **traditional_results}
        
        comparison_data = []
        
        for strategy_name, result in all_results.items():
            comparison_data.append({
                'Strategy': strategy_name,
                'Type': 'ML' if 'ML' in strategy_name or any(x in strategy_name for x in ['Random', 'Gradient']) else 'Traditional',
                'Total_Return_Pct': result['total_return_pct'],
                'Win_Rate_Pct': result['win_rate'],
                'Total_Trades': result['total_trades'],
                'Sharpe_Ratio': result['sharpe_ratio'],
                'Max_Drawdown_Pct': result['max_drawdown_pct'],
                'Profit_Factor': result['profit_factor']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Total_Return_Pct', ascending=False)
        
        # Print comparison
        print("\n🏆 ML vs Traditional Strategy Comparison:")
        print(comparison_df.to_string(index=False, float_format='%.2f'))
        
        # Plot comparison
        self.plot_ml_vs_traditional(comparison_df)
        
        # Save comparison
        comparison_df.to_csv('out/ml_vs_traditional_comparison.csv', index=False)
        print("💾 Comparison saved to 'out/ml_vs_traditional_comparison.csv'")
        
        return comparison_df
    
    def plot_ml_vs_traditional(self, comparison_df):
        """Plot ML vs traditional strategy performance"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('ML vs Traditional Strategy Performance', fontsize=16, fontweight='bold')
        
        # Color coding
        colors = ['blue' if t == 'Traditional' else 'red' for t in comparison_df['Type']]
        
        # Total Return
        axes[0, 0].bar(comparison_df['Strategy'], comparison_df['Total_Return_Pct'], color=colors)
        axes[0, 0].set_title('Total Return (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Win Rate
        axes[0, 1].bar(comparison_df['Strategy'], comparison_df['Win_Rate_Pct'], color=colors)
        axes[0, 1].set_title('Win Rate (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Sharpe Ratio
        axes[1, 0].bar(comparison_df['Strategy'], comparison_df['Sharpe_Ratio'], color=colors)
        axes[1, 0].set_title('Sharpe Ratio')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Max Drawdown
        axes[1, 1].bar(comparison_df['Strategy'], comparison_df['Max_Drawdown_Pct'], color=colors)
        axes[1, 1].set_title('Max Drawdown (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='Traditional'),
            Patch(facecolor='red', label='ML')
        ]
        axes[0, 0].legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig('out/ml_vs_traditional_performance.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def run_ml_parameter_sensitivity(self, train_data, test_data):
        """Test ML model sensitivity to different parameters"""
        print("\n🔧 Running ML Parameter Sensitivity Analysis...")
        
        lookback_periods = [5, 10, 15, 20]
        position_sizes = [0.05, 0.1, 0.15, 0.2]
        
        sensitivity_results = []
        
        for lookback in lookback_periods:
            for position_size in position_sizes:
                print(f"   Testing lookback={lookback}, position_size={position_size}...")
                
                try:
                    # Train model
                    predictor = BTCPricePredictor(
                        model_type='random_forest', 
                        lookback_periods=lookback
                    )
                    predictor.train(train_data, "random_forest")
                    
                    # Backtest
                    def ml_strategy_wrapper(data, model_predictor=predictor):
                        return BTCPricePredictor.ml_strategy_wrapper(data, model_predictor)
                    
                    result = self.backtester.run_backtest(
                        df=test_data,
                        strategy_func=ml_strategy_wrapper,
                        strategy_params={},
                        position_size=position_size,
                        stop_loss=0.02
                    )
                    
                    sensitivity_results.append({
                        'Lookback_Periods': lookback,
                        'Position_Size': position_size,
                        'Total_Return_Pct': result['total_return_pct'],
                        'Win_Rate_Pct': result['win_rate'],
                        'Sharpe_Ratio': result['sharpe_ratio'],
                        'Max_Drawdown_Pct': result['max_drawdown_pct']
                    })
                    
                except Exception as e:
                    print(f"      Error: {e}")
        
        # Analyze sensitivity
        sensitivity_df = pd.DataFrame(sensitivity_results)
        if not sensitivity_df.empty:
            best_combo = sensitivity_df.loc[sensitivity_df['Total_Return_Pct'].idxmax()]
            
            print(f"\n🎯 Best ML Parameters:")
            print(f"   Lookback Periods: {best_combo['Lookback_Periods']}")
            print(f"   Position Size: {best_combo['Position_Size']}")
            print(f"   Total Return: {best_combo['Total_Return_Pct']:.2f}%")
            
            sensitivity_df.to_csv('out/ml_parameter_sensitivity.csv', index=False)
            print("💾 Sensitivity analysis saved to 'out/ml_parameter_sensitivity.csv'")
            
            return best_combo
        
        return None


def run_traditional_strategies_for_comparison(df):
    """Run traditional strategies for ML comparison"""
    print("\n📊 Running Traditional Strategies for Comparison...")
    
    backtester = BacktestEngine(initial_capital=10000, commission=0.001)
    
    traditional_strategies = {
        "MA_Crossover_10_50": {
            'function': TradingStrategies.moving_average_crossover,
            'params': {'fast_window': 10, 'slow_window': 50}
        },
        "RSI_Momentum_14": {
            'function': TradingStrategies.rsi_momentum,
            'params': {'rsi_period': 14, 'oversold': 30, 'overbought': 70}
        }
    }
    
    results = {}
    
    for strategy_name, config in traditional_strategies.items():
        print(f"   Testing {strategy_name}...")
        
        results[strategy_name] = backtester.run_backtest(
            df=df,
            strategy_func=config['function'],
            strategy_params=config['params'],
            position_size=0.1
        )
    
    return results


def main():
    """Main ML backtesting demonstration"""
    print("="*80)
    print("MACHINE LEARNING STRATEGY BACKTESTING SUITE")
    print("="*80)
    
    # Initialize ML backtest suite
    ml_suite = MLBacktestSuite(initial_capital=10000, commission=0.001)
    
    try:
        # Prepare data
        train_data, test_data, full_data = ml_suite.prepare_ml_data(
            symbol="BTCUSDT", 
            interval="4h", 
            days=365
        )
        
        # Train ML models
        ml_models = ml_suite.train_ml_models(train_data)
        
        # Backtest ML strategies
        ml_results = ml_suite.backtest_ml_strategies(test_data, ml_models)
        
        # Run traditional strategies for comparison
        traditional_results = run_traditional_strategies_for_comparison(test_data)
        
        # Compare ML vs traditional
        comparison_df = ml_suite.compare_ml_vs_traditional(
            test_data, ml_results, traditional_results
        )
        
        # Run parameter sensitivity analysis
        best_params = ml_suite.run_ml_parameter_sensitivity(train_data, test_data)
        
        print("\n🎯 ML Backtesting Complete!")
        print("   Key Findings:")
        print("   - Review individual strategy reports in 'out/' directory")
        print("   - Check ML vs traditional comparison results")
        print("   - Examine parameter sensitivity analysis")
        print("   - Best performing model ready for live testing")
        
    except Exception as e:
        print(f"❌ Error in ML backtesting: {e}")
        raise


if __name__ == "__main__":
    main()

# -- EOF --
