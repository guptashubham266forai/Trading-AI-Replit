import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

class TechnicalIndicators:
    """Calculates various technical indicators for stock analysis"""
    
    def __init__(self):
        pass
    
    def moving_average(self, prices, window):
        """Calculate Simple Moving Average"""
        return prices.rolling(window=window).mean()
    
    def exponential_moving_average(self, prices, window):
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=window).mean()
    
    def rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_fast = self.exponential_moving_average(prices, fast)
        ema_slow = self.exponential_moving_average(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.exponential_moving_average(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'MACD': macd_line,
            'Signal': signal_line,
            'Histogram': histogram
        }
    
    def bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        sma = self.moving_average(prices, window)
        std = prices.rolling(window=window).std()
        
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return {
            'Upper': upper_band,
            'Middle': sma,
            'Lower': lower_band
        }
    
    def stochastic(self, high, low, close, k_window=14, d_window=3):
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return {
            '%K': k_percent,
            '%D': d_percent
        }
    
    def average_true_range(self, high, low, close, window=14):
        """Calculate Average True Range (ATR)"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        
        return atr
    
    def williams_r(self, high, low, close, window=14):
        """Calculate Williams %R"""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return williams_r
    
    def commodity_channel_index(self, high, low, close, window=20):
        """Calculate Commodity Channel Index (CCI)"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=window).mean()
        mean_deviation = typical_price.rolling(window=window).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        
        return cci
    
    def momentum(self, prices, window=10):
        """Calculate Price Momentum"""
        return prices.diff(window)
    
    def rate_of_change(self, prices, window=10):
        """Calculate Rate of Change (ROC)"""
        return ((prices - prices.shift(window)) / prices.shift(window)) * 100
    
    def on_balance_volume(self, close, volume):
        """Calculate On-Balance Volume (OBV)"""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def money_flow_index(self, high, low, close, volume, window=14):
        """Calculate Money Flow Index (MFI)"""
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume
        
        positive_flow = pd.Series(index=close.index, dtype=float)
        negative_flow = pd.Series(index=close.index, dtype=float)
        
        for i in range(1, len(typical_price)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.iloc[i] = raw_money_flow.iloc[i]
                negative_flow.iloc[i] = 0
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                positive_flow.iloc[i] = 0
                negative_flow.iloc[i] = raw_money_flow.iloc[i]
            else:
                positive_flow.iloc[i] = 0
                negative_flow.iloc[i] = 0
        
        positive_flow_sum = positive_flow.rolling(window=window).sum()
        negative_flow_sum = negative_flow.rolling(window=window).sum()
        
        money_flow_ratio = positive_flow_sum / negative_flow_sum
        mfi = 100 - (100 / (1 + money_flow_ratio))
        
        return mfi
    
    def support_resistance_levels(self, high, low, close, window=20):
        """Identify support and resistance levels using local extrema"""
        support_levels = pd.Series(index=close.index, dtype=float)
        resistance_levels = pd.Series(index=close.index, dtype=float)
        
        # Find local minima (support) and maxima (resistance)
        if len(low) >= window * 2:
            low_values = low.values
            high_values = high.values
            
            # Find local minima for support
            support_indices = argrelextrema(low_values, np.less, order=window//2)[0]
            
            # Find local maxima for resistance
            resistance_indices = argrelextrema(high_values, np.greater, order=window//2)[0]
            
            # Fill support levels
            for i, idx in enumerate(support_indices):
                if idx < len(support_levels):
                    support_value = low_values[idx]
                    # Propagate support level forward
                    for j in range(idx, min(len(support_levels), idx + window)):
                        if pd.isna(support_levels.iloc[j]) or support_levels.iloc[j] < support_value:
                            support_levels.iloc[j] = support_value
            
            # Fill resistance levels
            for i, idx in enumerate(resistance_indices):
                if idx < len(resistance_levels):
                    resistance_value = high_values[idx]
                    # Propagate resistance level forward
                    for j in range(idx, min(len(resistance_levels), idx + window)):
                        if pd.isna(resistance_levels.iloc[j]) or resistance_levels.iloc[j] > resistance_value:
                            resistance_levels.iloc[j] = resistance_value
        
        return {
            'Support': support_levels,
            'Resistance': resistance_levels
        }
    
    def pivot_points(self, high, low, close):
        """Calculate Pivot Points"""
        pivot = (high + low + close) / 3
        
        resistance_1 = (2 * pivot) - low
        support_1 = (2 * pivot) - high
        
        resistance_2 = pivot + (high - low)
        support_2 = pivot - (high - low)
        
        resistance_3 = high + 2 * (pivot - low)
        support_3 = low - 2 * (high - pivot)
        
        return {
            'Pivot': pivot,
            'R1': resistance_1,
            'R2': resistance_2,
            'R3': resistance_3,
            'S1': support_1,
            'S2': support_2,
            'S3': support_3
        }
    
    def ichimoku_cloud(self, high, low, close):
        """Calculate Ichimoku Cloud components"""
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        tenkan_sen = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        kijun_sen = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2, plotted 26 periods ahead
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2, plotted 26 periods ahead
        senkou_span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
        
        # Chikou Span (Lagging Span): Close plotted 26 periods back
        chikou_span = close.shift(-26)
        
        return {
            'Tenkan_sen': tenkan_sen,
            'Kijun_sen': kijun_sen,
            'Senkou_span_A': senkou_span_a,
            'Senkou_span_B': senkou_span_b,
            'Chikou_span': chikou_span
        }
    
    def volume_weighted_average_price(self, high, low, close, volume):
        """Calculate Volume Weighted Average Price (VWAP)"""
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        return vwap
