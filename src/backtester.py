#!/usr/bin/env python3
"""
Backtesting Framework for Quantitative Trading Strategies
Beta 1.1 - Integrated with cached data and ML models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
import warnings
from pathlib import Path
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class BacktestEngine:
    """
    Advanced backtesting engine for quantitative trading strategies
    """
    
    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.001):
        """
        Initialize backtesting engine
        
        Args:
            initial_capital: Starting capital in USD
            commission: Trading commission as percentage (0.1% = 0.001)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.results = {}
        self.trade_log = []
        
    def run_backtest(self, 
                    df: pd.DataFrame, 
                    strategy_func: Callable,
                    strategy_params: Dict = None,
                    position_size: float = 0.1,
                    stop_loss: float = None,
                    take_profit: float = None) -> Dict:
        """
        Run backtest on historical data
        
        Args:
            df: DataFrame with price data and features
            strategy_func: Function that generates trading signals
            strategy_params: Parameters to pass to strategy function
            position_size: Fraction of capital to use per trade (0.1 = 10%)
            stop_loss: Stop loss percentage (e.g., 0.02 for 2%)
            take_profit: Take profit percentage (e.g., 0.05 for 5%)
            
        Returns:
            Dictionary with backtest results
        """
        print("🚀 Running backtest...")
        
        # Initialize tracking variables
        capital = self.initial_capital
        position = 0.0  # Current position size in BTC
        entry_price = 0.0
        trades = []
        equity_curve = []
        in_position = False
        
        # Strategy parameters
        if strategy_params is None:
            strategy_params = {}
        
        # Iterate through each timestamp
        for i, (timestamp, row) in enumerate(df.iterrows()):
            current_price = row['close']
            current_data = df.iloc[:i+1]  # Data up to current point
            
            # Generate trading signal
            signal = strategy_func(current_data, **strategy_params)
            
            # Calculate position value
            position_value = position * current_price if position > 0 else 0
            portfolio_value = capital + position_value
            equity_curve.append({
                'timestamp': timestamp,
                'portfolio_value': portfolio_value,
                'capital': capital,
                'position_value': position_value,
                'price': current_price
            })
            
            # Handle existing position
            if in_position:
                # Check stop loss and take profit
                if stop_loss:
                    price_change = (current_price - entry_price) / entry_price
                    if price_change <= -stop_loss:
                        signal = 'SELL'  # Force sell on stop loss
                        
                if take_profit:
                    price_change = (current_price - entry_price) / entry_price
                    if price_change >= take_profit:
                        signal = 'SELL'  # Force sell on take profit
            
            # Execute trades based on signal
            if signal == 'BUY' and not in_position and capital > 0:
                # Calculate position size
                trade_size = capital * position_size
                btc_amount = trade_size / current_price
                
                # Apply commission
                commission_cost = trade_size * self.commission
                actual_investment = trade_size - commission_cost
                btc_amount_after_commission = actual_investment / current_price
                
                # Execute buy
                capital -= trade_size
                position += btc_amount_after_commission
                entry_price = current_price
                in_position = True
                
                trades.append({
                    'timestamp': timestamp,
                    'action': 'BUY',
                    'price': current_price,
                    'amount': btc_amount_after_commission,
                    'value': trade_size,
                    'commission': commission_cost,
                    'portfolio_value': portfolio_value
                })
                
            elif signal == 'SELL' and in_position and position > 0:
                # Calculate sale value
                sale_value = position * current_price
                commission_cost = sale_value * self.commission
                actual_proceeds = sale_value - commission_cost
                
                # Execute sell
                capital += actual_proceeds
                
                # Calculate trade performance
                trade_return = actual_proceeds - (position * entry_price)
                trade_return_pct = (trade_return / (position * entry_price)) * 100
                
                trades.append({
                    'timestamp': timestamp,
                    'action': 'SELL',
                    'price': current_price,
                    'amount': position,
                    'value': sale_value,
                    'commission': commission_cost,
                    'return': trade_return,
                    'return_pct': trade_return_pct,
                    'portfolio_value': portfolio_value
                })
                
                # Reset position
                position = 0.0
                in_position = False
                entry_price = 0.0
        
        # Close any open position at the end
        if in_position and position > 0:
            final_price = df.iloc[-1]['close']
            sale_value = position * final_price
            commission_cost = sale_value * self.commission
            actual_proceeds = sale_value - commission_cost
            capital += actual_proceeds
            
            trade_return = actual_proceeds - (position * entry_price)
            trade_return_pct = (trade_return / (position * entry_price)) * 100
            
            trades.append({
                'timestamp': df.index[-1],
                'action': 'SELL',
                'price': final_price,
                'amount': position,
                'value': sale_value,
                'commission': commission_cost,
                'return': trade_return,
                'return_pct': trade_return_pct,
                'portfolio_value': capital
            })
        
        # Calculate final results
        final_portfolio_value = capital
        total_return = (final_portfolio_value - self.initial_capital) / self.initial_capital * 100
        
        # Create results dictionary
        self.results = self._calculate_performance_metrics(
            equity_curve, trades, total_return, final_portfolio_value
        )
        self.trade_log = trades
        
        print(f"✅ Backtest completed")
        print(f"   Initial Capital: ${self.initial_capital:,.2f}")
        print(f"   Final Portfolio: ${final_portfolio_value:,.2f}")
        print(f"   Total Return: {total_return:.2f}%")
        print(f"   Total Trades: {len([t for t in trades if t['action'] == 'SELL'])}")
        
        return self.results
    
    def _calculate_performance_metrics(self, equity_curve: List, trades: List, 
                                    total_return: float, final_portfolio_value: float) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        # Extract portfolio values and returns
        portfolio_values = [e['portfolio_value'] for e in equity_curve]
        returns = pd.Series(portfolio_values).pct_change().dropna()
        
        # Basic metrics
        total_trades = len([t for t in trades if t['action'] == 'SELL'])
        winning_trades = len([t for t in trades if t.get('return', 0) > 0 and t['action'] == 'SELL'])
        
        # Trade metrics
        if total_trades > 0:
            win_rate = (winning_trades / total_trades) * 100
            trade_returns = [t.get('return_pct', 0) for t in trades if t['action'] == 'SELL']
            avg_trade_return = np.mean(trade_returns) if trade_returns else 0
            best_trade = max(trade_returns) if trade_returns else 0
            worst_trade = min(trade_returns) if trade_returns else 0
        else:
            win_rate = avg_trade_return = best_trade = worst_trade = 0
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(365 * 24)  # Annualized (assuming hourly data)
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(365 * 24) if returns.std() > 0 else 0
        
        # Drawdown calculation
        rolling_max = pd.Series(portfolio_values).expanding().max()
        drawdowns = (pd.Series(portfolio_values) - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() * 100 if not drawdowns.empty else 0
        
        # Profit factor
        gross_profits = sum(t.get('return', 0) for t in trades if t.get('return', 0) > 0)
        gross_losses = abs(sum(t.get('return', 0) for t in trades if t.get('return', 0) < 0))
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
        
        return {
            'initial_capital': self.initial_capital,
            'final_portfolio_value': final_portfolio_value,
            'total_return_pct': total_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_trade_return_pct': avg_trade_return,
            'best_trade_pct': best_trade,
            'worst_trade_pct': worst_trade,
            'max_drawdown_pct': max_drawdown,
            'volatility_pct': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'equity_curve': equity_curve,
            'trades': trades
        }
    
    def plot_results(self, symbol: str = "BTCUSDT", save_path: str = "out/"):
        """Plot backtest results"""
        if not self.results:
            print("❌ No results to plot. Run backtest first.")
            return
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Backtest Results - {symbol}', fontsize=16, fontweight='bold')
        
        # Equity curve
        equity_df = pd.DataFrame(self.results['equity_curve'])
        equity_df.set_index('timestamp', inplace=True)
        
        axes[0, 0].plot(equity_df.index, equity_df['portfolio_value'], 
                       label='Portfolio Value', linewidth=2, color='blue')
        axes[0, 0].axhline(y=self.initial_capital, color='red', linestyle='--', 
                          alpha=0.7, label='Initial Capital')
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Drawdown
        rolling_max = equity_df['portfolio_value'].expanding().max()
        drawdown = (equity_df['portfolio_value'] - rolling_max) / rolling_max * 100
        axes[0, 1].fill_between(equity_df.index, drawdown, 0, alpha=0.3, color='red')
        axes[0, 1].plot(equity_df.index, drawdown, color='red', linewidth=1)
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Daily returns distribution
        daily_returns = equity_df['portfolio_value'].pct_change().dropna()
        axes[1, 0].hist(daily_returns * 100, bins=50, alpha=0.7, color='green')
        axes[1, 0].axvline(daily_returns.mean() * 100, color='red', linestyle='--', 
                          label=f'Mean: {daily_returns.mean()*100:.2f}%')
        axes[1, 0].set_title('Daily Returns Distribution')
        axes[1, 0].set_xlabel('Daily Return (%)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Trade analysis
        if self.results['trades']:
            sell_trades = [t for t in self.results['trades'] if t['action'] == 'SELL']
            if sell_trades:
                trade_returns = [t['return_pct'] for t in sell_trades]
                axes[1, 1].bar(range(len(trade_returns)), trade_returns, 
                              color=['green' if x > 0 else 'red' for x in trade_returns])
                axes[1, 1].axhline(y=0, color='black', linewidth=0.5)
                axes[1, 1].set_title('Individual Trade Returns')
                axes[1, 1].set_xlabel('Trade Number')
                axes[1, 1].set_ylabel('Return (%)')
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plot_filename = f"{save_path}/{symbol}_backtest_results.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"💾 Backtest plot saved to {plot_filename}")
        plt.show()
    
    def generate_report(self, symbol: str = "BTCUSDT", save_path: str = "out/"):
        """Generate comprehensive backtest report"""
        if not self.results:
            print("❌ No results to report. Run backtest first.")
            return
        
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # Save results to JSON
        report_data = {
            'backtest_date': datetime.now().isoformat(),
            'symbol': symbol,
            'parameters': {
                'initial_capital': self.initial_capital,
                'commission': self.commission
            },
            'performance_metrics': {k: v for k, v in self.results.items() 
                                  if k not in ['equity_curve', 'trades']},
            'trade_count': len([t for t in self.results['trades'] if t['action'] == 'SELL'])
        }
        
        # Save JSON report
        report_filename = f"{save_path}/{symbol}_backtest_report.json"
        with open(report_filename, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Save trade log to CSV
        if self.results['trades']:
            trades_df = pd.DataFrame(self.results['trades'])
            trades_filename = f"{save_path}/{symbol}_trade_log.csv"
            trades_df.to_csv(trades_filename, index=False)
        
        # Print summary
        print("\n" + "="*60)
        print("BACKTEST PERFORMANCE REPORT")
        print("="*60)
        print(f"Symbol: {symbol}")
        print(f"Period: {self.results['equity_curve'][0]['timestamp']} to {self.results['equity_curve'][-1]['timestamp']}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Portfolio: ${self.results['final_portfolio_value']:,.2f}")
        print(f"Total Return: {self.results['total_return_pct']:.2f}%")
        print(f"Total Trades: {self.results['total_trades']}")
        print(f"Win Rate: {self.results['win_rate']:.1f}%")
        print(f"Avg Trade Return: {self.results['avg_trade_return_pct']:.2f}%")
        print(f"Max Drawdown: {self.results['max_drawdown_pct']:.2f}%")
        print(f"Sharpe Ratio: {self.results['sharpe_ratio']:.2f}")
        print(f"Profit Factor: {self.results['profit_factor']:.2f}")
        
        print(f"\n💾 Report saved to {report_filename}")
        
        return report_data


# Strategy Library - Common Trading Strategies
class TradingStrategies:
    """Collection of common trading strategies for backtesting"""
    
    @staticmethod
    def moving_average_crossover(data: pd.DataFrame, 
                               fast_window: int = 10, 
                               slow_window: int = 20) -> str:
        """
        Simple moving average crossover strategy
        
        Args:
            data: Historical price data
            fast_window: Fast moving average period
            slow_window: Slow moving average period
            
        Returns:
            Trading signal: 'BUY', 'SELL', or 'HOLD'
        """
        if len(data) < slow_window:
            return 'HOLD'
        
        # Calculate moving averages
        fast_ma = data['close'].rolling(fast_window).mean().iloc[-1]
        slow_ma = data['close'].rolling(slow_window).mean().iloc[-1]
        prev_fast_ma = data['close'].rolling(fast_window).mean().iloc[-2]
        prev_slow_ma = data['close'].rolling(slow_window).mean().iloc[-2]
        
        # Generate signals
        if fast_ma > slow_ma and prev_fast_ma <= prev_slow_ma:
            return 'BUY'
        elif fast_ma < slow_ma and prev_fast_ma >= prev_slow_ma:
            return 'SELL'
        else:
            return 'HOLD'
    
    @staticmethod
    def rsi_momentum(data: pd.DataFrame, 
                    rsi_period: int = 14,
                    oversold: int = 30,
                    overbought: int = 70) -> str:
        """
        RSI-based momentum strategy
        
        Args:
            data: Historical price data with RSI calculated
            rsi_period: RSI period
            oversold: Oversold threshold
            overbought: Overbought threshold
            
        Returns:
            Trading signal: 'BUY', 'SELL', or 'HOLD'
        """
        if len(data) < rsi_period + 1:
            return 'HOLD'
        
        # Calculate RSI if not present
        if 'rsi' not in data.columns:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = data['rsi'].iloc[-1]
        
        # Generate signals
        if rsi < oversold:
            return 'BUY'
        elif rsi > overbought:
            return 'SELL'
        else:
            return 'HOLD'
    
    @staticmethod
    def ml_strategy(data: pd.DataFrame, model_predictor) -> str:
        """
        ML-based trading strategy
        
        Args:
            data: Historical price data with features
            model_predictor: Trained ML model predictor
            
        Returns:
            Trading signal: 'BUY', 'SELL', or 'HOLD'
        """
        try:
            # Use the last row for prediction
            if len(data) < 20:  # Minimum data required
                return 'HOLD'
            
            # Make prediction
            prediction = model_predictor.predict(data.tail(1))
            
            if prediction.iloc[-1] == 1:  # Predicts price will go up
                return 'BUY'
            else:
                return 'SELL'
                
        except Exception as e:
            print(f"❌ ML strategy error: {e}")
            return 'HOLD'


def main():
    """Demo backtesting with different strategies"""
    from cached_binance_client import CachedBinanceClient
    from basic_quant_analysis import calculate_technical_indicators
    
    print("🚀 Starting Backtesting Demo...")
    
    # Initialize client and get data
    client = CachedBinanceClient()
    df = client.get_historical_klines(symbol="BTCUSDT", interval="1h", days=90)
    
    if df.empty:
        print("❌ No data available for backtesting")
        return
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Initialize backtest engine
    backtester = BacktestEngine(initial_capital=10000, commission=0.001)
    
    # Test Moving Average Crossover Strategy
    print("\n🔹 Testing Moving Average Crossover Strategy...")
    results_ma = backtester.run_backtest(
        df=df,
        strategy_func=TradingStrategies.moving_average_crossover,
        strategy_params={'fast_window': 10, 'slow_window': 20},
        position_size=0.1,  # 10% per trade
        stop_loss=0.02,     # 2% stop loss
        take_profit=0.05    # 5% take profit
    )
    
    # Generate report and plots
    backtester.generate_report(symbol="BTCUSDT_MA")
    backtester.plot_results(symbol="BTCUSDT_MA")
    
    # Test RSI Strategy
    print("\n🔹 Testing RSI Momentum Strategy...")
    results_rsi = backtester.run_backtest(
        df=df,
        strategy_func=TradingStrategies.rsi_momentum,
        strategy_params={'rsi_period': 14, 'oversold': 30, 'overbought': 70},
        position_size=0.1,
        stop_loss=0.02,
        take_profit=0.05
    )
    
    backtester.generate_report(symbol="BTCUSDT_RSI")
    backtester.plot_results(symbol="BTCUSDT_RSI")
    
    print("\n🎯 Backtesting demo completed!")
    print("   Compare strategy performance in the generated reports")


if __name__ == "__main__":
    main()

# -- EOF --
