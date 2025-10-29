"""
Advanced feature engineering for quant trading
"""

import pandas as pd
import numpy as np
from typing import List, Dict


class FeatureEngineer:
    def __init__(self):
        self.feature_groups = {}
    
    def add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        # Price transformation features
        df['log_price'] = np.log(df['close'])
        df['price_zscore'] = (df['close'] - df['close'].rolling(50).mean()) / df['close'].rolling(50).std()
        
        # Volume features
        df['volume_zscore'] = (df['volume'] - df['volume'].rolling(50).mean()) / df['volume'].rolling(50).std()
        df['volume_price_trend'] = df['volume'] * df['close'].pct_change()
        
        # Momentum features
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        
        # Mean reversion features
        df['mean_reversion_5'] = (df['close'] - df['close'].rolling(5).mean()) / df['close'].rolling(5).std()
        df['mean_reversion_20'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['month'] = df.index.month
        
        # Market hours features (assuming UTC)
        df['is_asia_hours'] = ((df['hour'] >= 0) & (df['hour'] <= 8)).astype(int)
        df['is_europe_hours'] = ((df['hour'] >= 7) & (df['hour'] <= 16)).astype(int)
        df['is_us_hours'] = ((df['hour'] >= 13) & (df['hour'] <= 21)).astype(int)
        
        return df
    
    def add_rolling_features(self, df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """Add rolling window features"""
        for window in windows:
            # Rolling statistics
            df[f'rolling_mean_{window}'] = df['close'].rolling(window).mean()
            df[f'rolling_std_{window}'] = df['close'].rolling(window).std()
            df[f'rolling_skew_{window}'] = df['close'].rolling(window).skew()
            
            # Rolling min/max
            df[f'rolling_min_{window}'] = df['close'].rolling(window).min()
            df[f'rolling_max_{window}'] = df['close'].rolling(window).max()
            df[f'rolling_range_{window}'] = df[f'rolling_max_{window}'] - df[f'rolling_min_{window}']
        
        return df
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        print("🛠️ Engineering advanced features...")
        
        df = self.add_technical_features(df)
        df = self.add_time_features(df)
        df = self.add_rolling_features(df)
        
        print(f"✅ Added {len([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']])} features")
        return df

# -- EOF --
