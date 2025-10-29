#!/usr/bin/env python3
"""
ML Model for BTC Price Prediction
Expanded from skeleton to full training pipeline
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, precision_score
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from cached_binance_client import CachedBinanceClient
from basic_quant_analysis import calculate_technical_indicators


class BTCPricePredictor:
    def __init__(self, model_type='random_forest', lookback_periods=10):
        """
        Initialize BTC price prediction model
        
        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting')
            lookback_periods: Number of previous periods to use as features
        """
        self.model_type = model_type
        self.lookback_periods = lookback_periods
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def create_features(self, df):
        """
        Create features for ML model including lagged values and rolling statistics
        """
        #print("🛠️ Creating ML features...")

        # Work with a copy to avoid modifying the original
        df = df.copy()

        # Ensure we have technical indicators
        if 'rsi' not in df.columns:
            df = calculate_technical_indicators(df)

        # Verify required base columns exist
        required_cols = ['close', 'volume', 'open', 'high', 'low']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Price-based features
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['open_close_ratio'] = df['open'] / df['close']

        # Volume features
        df['volume_change'] = df['volume'].pct_change()
        df['volume_price_correlation'] = df['volume'].rolling(20).corr(df['close'])

        # Volatility features
        df['volatility_5'] = df['price_change'].rolling(5).std()
        df['volatility_20'] = df['price_change'].rolling(20).std()

        # Lagged features - ensure all are created
        for i in range(1, self.lookback_periods + 1):
            df[f'close_lag_{i}'] = df['close'].shift(i)
            df[f'volume_lag_{i}'] = df['volume'].shift(i)
            df[f'rsi_lag_{i}'] = df['rsi'].shift(i)

        # Rolling statistics
        df['close_rolling_mean_5'] = df['close'].rolling(5).mean()
        df['close_rolling_std_5'] = df['close'].rolling(5).std()
        df['volume_rolling_mean_5'] = df['volume'].rolling(5).mean()

        # Target variable: 1 if price increases in next period, 0 otherwise
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

        return df
    
    def prepare_data(self, df, is_training=True):
        """
        Prepare features and target for training or prediction

        Args:
            df: Input dataframe
            is_training: If True, compute and save feature columns. If False, use saved columns.
        """
        # Create features
        df_with_features = self.create_features(df)

        if is_training:
            # During training: determine which columns are features
            exclude_cols = ['target', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
            feature_columns = [col for col in df_with_features.columns
                             if col not in exclude_cols and not col.startswith('ignore')]

            self.feature_columns = feature_columns

            # Remove rows with NaN values (from lagged features and indicators)
            clean_df = df_with_features.dropna(subset=feature_columns + ['target'])
        else:
            # During prediction: use saved feature columns
            if self.feature_columns is None:
                raise ValueError("Model must be trained before making predictions")

            feature_columns = self.feature_columns

            # Check that all expected features are present
            missing_features = [col for col in feature_columns if col not in df_with_features.columns]
            if missing_features:
                raise ValueError(f"Missing features during prediction: {missing_features}. "
                               f"Ensure the model's lookback_periods ({self.lookback_periods}) "
                               f"matches the training configuration.")

            # Remove rows with NaN values, but only check feature columns (no 'target' in prediction)
            clean_df = df_with_features.dropna(subset=feature_columns)

        X = clean_df[feature_columns]
        y = clean_df['target'] if is_training else None

        #print(f"📊 Prepared dataset: {len(X)} samples, {len(feature_columns)} features")
        return X, y, clean_df
    
    def train(self, df, model_name="unnamed", test_size=0.2):
        """
        Train the model on historical data
        """
        print(f"🎯 Training ML model {model_name}...")

        X, y, _ = self.prepare_data(df, is_training=True)
        
        # Use time series split for financial data
        tscv = TimeSeriesSplit(n_splits=5)
        
        accuracies = []
        precisions = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Predict and evaluate
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            
            accuracies.append(accuracy)
            precisions.append(precision)
        
        print(f"✅ Model training completed")
        print(f"   Average Accuracy: {np.mean(accuracies):.3f} (+/- {np.std(accuracies):.3f})")
        print(f"   Average Precision: {np.mean(precisions):.3f} (+/- {np.std(precisions):.3f})")
        
        return np.mean(accuracies), np.mean(precisions)
    
    def predict(self, df):
        """
        Make predictions on new data
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")

        X, _, clean_df = self.prepare_data(df, is_training=False)
        
        # Scale features and predict
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Add predictions to dataframe
        result_df = clean_df.copy()
        result_df['prediction'] = predictions
        result_df['probability_up'] = probabilities[:, 1]
        result_df['confidence'] = np.max(probabilities, axis=1)
        
        return result_df
    
    def evaluate_strategy(self, df_with_predictions, initial_balance=10000):
        """
        Evaluate trading strategy based on model predictions
        """
        df = df_with_predictions.copy()
        balance = initial_balance
        position = 0  # 0: no position, 1: long
        trades = []
        
        for i, row in df.iterrows():
            if row['prediction'] == 1 and position == 0:  # Buy signal
                position = 1
                entry_price = row['close']
                entry_balance = balance
                trades.append({
                    'timestamp': i,
                    'action': 'BUY',
                    'price': entry_price,
                    'balance': balance
                })
            elif row['prediction'] == 0 and position == 1:  # Sell signal
                position = 0
                exit_price = row['close']
                returns_pct = (exit_price - entry_price) / entry_price
                balance = entry_balance * (1 + returns_pct)
                trades.append({
                    'timestamp': i,
                    'action': 'SELL',
                    'price': exit_price,
                    'balance': balance,
                    'returns_pct': returns_pct * 100
                })
        
        # Calculate performance metrics
        final_balance = balance
        total_return = (final_balance - initial_balance) / initial_balance * 100
        
        if trades:
            trade_returns = [t.get('returns_pct', 0) for t in trades if 'returns_pct' in t]
            win_rate = len([r for r in trade_returns if r > 0]) / len(trade_returns) * 100
            avg_return = np.mean(trade_returns) if trade_returns else 0
        else:
            win_rate = 0
            avg_return = 0
        
        return {
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'total_return_pct': total_return,
            'win_rate': win_rate,
            'avg_trade_return': avg_return,
            'num_trades': len([t for t in trades if t['action'] == 'SELL']),
            'trades': trades
        }
    
    def save_model(self, filepath):
        """Save trained model to file"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'lookback_periods': self.lookback_periods
        }
        joblib.dump(model_data, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model from file"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        self.lookback_periods = model_data['lookback_periods']
        print(f"📂 Model loaded from {filepath}")

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals for backtesting

        Args:
            data: Historical price data with features

        Returns:
            DataFrame with trading signals
        """
        if self.model is None:
            raise ValueError("Model must be trained before generating signals")

        # Make predictions
        df_with_predictions = self.predict(data)

        # Convert predictions to trading signals
        df_with_predictions['signal'] = df_with_predictions['prediction'].map({1: 'BUY', 0: 'SELL'})

        return df_with_predictions

    @staticmethod
    def ml_strategy_wrapper(data: pd.DataFrame, model_predictor) -> str:
        """
        Wrapper function for backtesting ML strategies

        Args:
            data: Historical price data
            model_predictor: Trained BTCPricePredictor instance

        Returns:
            Trading signal
        """
        try:
            # Need sufficient data for rolling calculations and lagged features
            # Technical indicators need ~50 periods, lagged features need lookback_periods
            min_data = max(100, model_predictor.lookback_periods * 3)

            if len(data) < min_data:
                return 'HOLD'

            # Pass enough historical data for feature calculation, but limit to avoid slowdown
            window_size = min(len(data), 200)
            predictions = model_predictor.generate_signals(data.tail(window_size))

            # Return only the last signal
            return predictions['signal'].iloc[-1]

        except Exception as e:
            print(f"❌ ML strategy error: {e}")
            return 'HOLD'

def main():
    """Main function to demonstrate ML model training and evaluation"""
    # Initialize client and get data
    client = CachedBinanceClient()
    
    print("🚀 Starting ML model training pipeline...")
    
    # Get more data for better training
    df = client.get_historical_klines(symbol="BTCUSDT", interval="1h", days=365)
    
    if df.empty:
        print("❌ No data available for model training")
        return
    
    # Initialize and train model
    predictor = BTCPricePredictor(model_type='random_forest', lookback_periods=10)
    accuracy, precision = predictor.train(df, "random_forest")
    
    # Make predictions
    df_with_predictions = predictor.predict(df.tail(1000))  # Predict on recent data
    
    # Evaluate trading strategy
    performance = predictor.evaluate_strategy(df_with_predictions)
    
    print("\n" + "="*60)
    print("TRADING STRATEGY PERFORMANCE")
    print("="*60)
    print(f"📈 Total Return: {performance['total_return_pct']:.2f}%")
    print(f"🎯 Win Rate: {performance['win_rate']:.1f}%")
    print(f"📊 Number of Trades: {performance['num_trades']}")
    print(f"💰 Avg Trade Return: {performance['avg_trade_return']:.2f}%")
    print(f"💵 Final Balance: ${performance['final_balance']:,.2f}")
    
    # Save model for future use
    predictor.save_model("out/btc_predictor_model.pkl")
    
    # Save predictions for analysis
    df_with_predictions.to_csv("out/btc_ml_predictions.csv")
    print(f"\n💾 Predictions saved to 'out/btc_ml_predictions.csv'")
    
    print(f"\n🎯 Next steps:")
    print(f"   - Use the saved model for real-time predictions")
    print(f"   - Experiment with different model types and parameters")
    print(f"   - Add more sophisticated feature engineering")


if __name__ == "__main__":
    main()

# -- EOF --
