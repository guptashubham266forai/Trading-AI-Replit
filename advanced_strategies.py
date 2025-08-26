import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from technical_indicators import TechnicalIndicators

class AdvancedTradingStrategies:
    """Advanced professional trading strategies with higher profit potential"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.active_strategies = [
            "Volume Price Analysis", "Smart Money Concepts", "Market Structure Break",
            "Fibonacci Retracement", "Multi-Timeframe Analysis", "Momentum Divergence",
            "Order Block Detection", "Liquidity Hunt", "Supply Demand Zones"
        ]
        self.max_risk_percent = 1.5
        self.min_risk_reward = 3.0  # Higher risk-reward for better profits
        self.confidence_threshold = 0.85
    
    def add_advanced_indicators(self, data):
        """Add advanced technical indicators"""
        if len(data) < 100:
            return data
        
        # Volume-weighted indicators
        data['VWAP'] = self.calculate_vwap(data)
        data['Volume_MA'] = data['Volume'].rolling(20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
        
        # Advanced momentum
        data['ROC'] = ((data['Close'] - data['Close'].shift(14)) / data['Close'].shift(14)) * 100
        data['MFI'] = self.money_flow_index(data)
        
        # Market structure
        data['Higher_High'] = self.detect_higher_highs(data)
        data['Higher_Low'] = self.detect_higher_lows(data)
        data['Lower_High'] = self.detect_lower_highs(data)
        data['Lower_Low'] = self.detect_lower_lows(data)
        
        # Volatility
        data['ATR'] = self.average_true_range(data)
        data['Volatility'] = data['Close'].rolling(20).std()
        
        # Price action patterns
        data['Hammer'] = self.detect_hammer(data)
        data['Doji'] = self.detect_doji(data)
        data['Engulfing'] = self.detect_engulfing(data)
        
        return data
    
    def calculate_vwap(self, data):
        """Calculate Volume Weighted Average Price"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        return (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
    
    def money_flow_index(self, data, period=14):
        """Calculate Money Flow Index"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        raw_money_flow = typical_price * data['Volume']
        
        positive_flow = []
        negative_flow = []
        
        for i in range(1, len(typical_price)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.append(raw_money_flow.iloc[i])
                negative_flow.append(0)
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                positive_flow.append(0)
                negative_flow.append(raw_money_flow.iloc[i])
            else:
                positive_flow.append(0)
                negative_flow.append(0)
        
        positive_flow = [0] + positive_flow
        negative_flow = [0] + negative_flow
        
        positive_flow_sum = pd.Series(positive_flow).rolling(period).sum()
        negative_flow_sum = pd.Series(negative_flow).rolling(period).sum()
        
        mfi = 100 - (100 / (1 + positive_flow_sum / negative_flow_sum))
        return mfi
    
    def average_true_range(self, data, period=14):
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(period).mean()
    
    def detect_higher_highs(self, data, lookback=5):
        """Detect higher highs pattern"""
        highs = data['High']
        higher_highs = []
        
        for i in range(lookback, len(highs)):
            current_high = highs.iloc[i]
            prev_peak = highs.iloc[i-lookback:i].max()
            
            if current_high > prev_peak:
                higher_highs.append(True)
            else:
                higher_highs.append(False)
        
        return [False] * lookback + higher_highs
    
    def detect_higher_lows(self, data, lookback=5):
        """Detect higher lows pattern"""
        lows = data['Low']
        higher_lows = []
        
        for i in range(lookback, len(lows)):
            current_low = lows.iloc[i]
            prev_trough = lows.iloc[i-lookback:i].min()
            
            if current_low > prev_trough:
                higher_lows.append(True)
            else:
                higher_lows.append(False)
        
        return [False] * lookback + higher_lows
    
    def detect_lower_highs(self, data, lookback=5):
        """Detect lower highs pattern"""
        highs = data['High']
        lower_highs = []
        
        for i in range(lookback, len(highs)):
            current_high = highs.iloc[i]
            prev_peak = highs.iloc[i-lookback:i].max()
            
            if current_high < prev_peak:
                lower_highs.append(True)
            else:
                lower_highs.append(False)
        
        return [False] * lookback + lower_highs
    
    def detect_lower_lows(self, data, lookback=5):
        """Detect lower lows pattern"""
        lows = data['Low']
        lower_lows = []
        
        for i in range(lookback, len(lows)):
            current_low = lows.iloc[i]
            prev_trough = lows.iloc[i-lookback:i].min()
            
            if current_low < prev_trough:
                lower_lows.append(True)
            else:
                lower_lows.append(False)
        
        return [False] * lookback + lower_lows
    
    def detect_hammer(self, data):
        """Detect hammer candlestick pattern"""
        body = abs(data['Close'] - data['Open'])
        lower_shadow = data['Open'].combine(data['Close'], min) - data['Low']
        upper_shadow = data['High'] - data['Open'].combine(data['Close'], max)
        
        hammer = (lower_shadow > 2 * body) & (upper_shadow < body * 0.1)
        return hammer
    
    def detect_doji(self, data):
        """Detect doji candlestick pattern"""
        body = abs(data['Close'] - data['Open'])
        total_range = data['High'] - data['Low']
        
        doji = body < (total_range * 0.1)
        return doji
    
    def detect_engulfing(self, data):
        """Detect engulfing candlestick pattern"""
        prev_body = abs(data['Close'].shift(1) - data['Open'].shift(1))
        curr_body = abs(data['Close'] - data['Open'])
        
        bullish_engulfing = (
            (data['Close'].shift(1) < data['Open'].shift(1)) &  # Previous candle bearish
            (data['Close'] > data['Open']) &  # Current candle bullish
            (data['Open'] < data['Close'].shift(1)) &  # Current opens below prev close
            (data['Close'] > data['Open'].shift(1)) &  # Current closes above prev open
            (curr_body > prev_body)  # Current body larger
        )
        
        return bullish_engulfing
    
    def smart_money_concepts_strategy(self, data, symbol):
        """Smart Money Concepts (SMC) strategy"""
        signals = []
        
        if len(data) < 50:
            return signals
        
        # Look for market structure breaks with volume confirmation
        for i in range(20, len(data)):
            current_price = data['Close'].iloc[i]
            volume_ratio = data['Volume_Ratio'].iloc[i] if 'Volume_Ratio' in data.columns else 1
            
            # Check for break of structure (BOS) with high volume
            if (data['Higher_High'].iloc[i] and volume_ratio > 1.5 and 
                data['RSI'].iloc[i] < 70):
                
                confidence = self.calculate_smc_confidence(data, i)
                
                if confidence >= self.confidence_threshold:
                    atr = data['ATR'].iloc[i] if 'ATR' in data.columns else current_price * 0.02
                    
                    stop_loss = current_price - (2 * atr)
                    target = current_price + (4 * atr)  # 1:2 risk reward
                    risk_reward = (target - current_price) / (current_price - stop_loss)
                    
                    signals.append({
                        'symbol': symbol,
                        'action': 'BUY',
                        'price': current_price,
                        'strategy': 'Smart Money Concepts (BOS)',
                        'confidence': confidence,
                        'stop_loss': stop_loss,
                        'target': target,
                        'risk_reward': risk_reward,
                        'timestamp': data.index[i],
                        'notes': f'Market structure break with volume confirmation'
                    })
            
            # Bearish BOS
            elif (data['Lower_Low'].iloc[i] and volume_ratio > 1.5 and 
                  data['RSI'].iloc[i] > 30):
                
                confidence = self.calculate_smc_confidence(data, i)
                
                if confidence >= self.confidence_threshold:
                    atr = data['ATR'].iloc[i] if 'ATR' in data.columns else current_price * 0.02
                    
                    stop_loss = current_price + (2 * atr)
                    target = current_price - (4 * atr)
                    risk_reward = (current_price - target) / (stop_loss - current_price)
                    
                    signals.append({
                        'symbol': symbol,
                        'action': 'SELL',
                        'price': current_price,
                        'strategy': 'Smart Money Concepts (BOS)',
                        'confidence': confidence,
                        'stop_loss': stop_loss,
                        'target': target,
                        'risk_reward': risk_reward,
                        'timestamp': data.index[i],
                        'notes': f'Bearish market structure break'
                    })
        
        return signals
    
    def volume_price_analysis_strategy(self, data, symbol):
        """Volume Price Analysis (VPA) strategy"""
        signals = []
        
        if len(data) < 30 or 'VWAP' not in data.columns:
            return signals
        
        for i in range(20, len(data)):
            current_price = data['Close'].iloc[i]
            vwap = data['VWAP'].iloc[i]
            volume_ratio = data['Volume_Ratio'].iloc[i] if 'Volume_Ratio' in data.columns else 1
            
            # High volume above VWAP = bullish
            if (current_price > vwap * 1.002 and volume_ratio > 2.0 and 
                data['Close'].iloc[i] > data['Open'].iloc[i]):
                
                confidence = self.calculate_vpa_confidence(data, i)
                
                if confidence >= self.confidence_threshold:
                    stop_loss = vwap * 0.998
                    target = current_price + (current_price - stop_loss) * 2.5
                    risk_reward = (target - current_price) / (current_price - stop_loss)
                    
                    signals.append({
                        'symbol': symbol,
                        'action': 'BUY',
                        'price': current_price,
                        'strategy': 'Volume Price Analysis',
                        'confidence': confidence,
                        'stop_loss': stop_loss,
                        'target': target,
                        'risk_reward': risk_reward,
                        'timestamp': data.index[i],
                        'notes': f'High volume breakout above VWAP'
                    })
            
            # High volume below VWAP = bearish
            elif (current_price < vwap * 0.998 and volume_ratio > 2.0 and 
                  data['Close'].iloc[i] < data['Open'].iloc[i]):
                
                confidence = self.calculate_vpa_confidence(data, i)
                
                if confidence >= self.confidence_threshold:
                    stop_loss = vwap * 1.002
                    target = current_price - (stop_loss - current_price) * 2.5
                    risk_reward = (current_price - target) / (stop_loss - current_price)
                    
                    signals.append({
                        'symbol': symbol,
                        'action': 'SELL',
                        'price': current_price,
                        'strategy': 'Volume Price Analysis',
                        'confidence': confidence,
                        'stop_loss': stop_loss,
                        'target': target,
                        'risk_reward': risk_reward,
                        'timestamp': data.index[i],
                        'notes': f'High volume breakdown below VWAP'
                    })
        
        return signals
    
    def momentum_divergence_strategy(self, data, symbol):
        """Momentum divergence strategy using RSI and price action"""
        signals = []
        
        if len(data) < 50:
            return signals
        
        for i in range(30, len(data)):
            current_price = data['Close'].iloc[i]
            
            # Look for bullish divergence
            if self.detect_bullish_divergence(data, i):
                confidence = self.calculate_divergence_confidence(data, i)
                
                if confidence >= self.confidence_threshold:
                    recent_low = data['Low'].iloc[i-10:i].min()
                    stop_loss = recent_low * 0.995
                    target = current_price + (current_price - stop_loss) * 3
                    risk_reward = (target - current_price) / (current_price - stop_loss)
                    
                    signals.append({
                        'symbol': symbol,
                        'action': 'BUY',
                        'price': current_price,
                        'strategy': 'Momentum Divergence',
                        'confidence': confidence,
                        'stop_loss': stop_loss,
                        'target': target,
                        'risk_reward': risk_reward,
                        'timestamp': data.index[i],
                        'notes': f'Bullish RSI divergence detected'
                    })
            
            # Look for bearish divergence
            elif self.detect_bearish_divergence(data, i):
                confidence = self.calculate_divergence_confidence(data, i)
                
                if confidence >= self.confidence_threshold:
                    recent_high = data['High'].iloc[i-10:i].max()
                    stop_loss = recent_high * 1.005
                    target = current_price - (stop_loss - current_price) * 3
                    risk_reward = (current_price - target) / (stop_loss - current_price)
                    
                    signals.append({
                        'symbol': symbol,
                        'action': 'SELL',
                        'price': current_price,
                        'strategy': 'Momentum Divergence',
                        'confidence': confidence,
                        'stop_loss': stop_loss,
                        'target': target,
                        'risk_reward': risk_reward,
                        'timestamp': data.index[i],
                        'notes': f'Bearish RSI divergence detected'
                    })
        
        return signals
    
    def detect_bullish_divergence(self, data, index):
        """Detect bullish divergence between price and RSI"""
        if 'RSI' not in data.columns or index < 20:
            return False
        
        # Price making lower lows but RSI making higher lows
        price_window = data['Close'].iloc[index-20:index+1]
        rsi_window = data['RSI'].iloc[index-20:index+1]
        
        recent_price_low = price_window.tail(10).min()
        earlier_price_low = price_window.head(10).min()
        
        recent_rsi_low = rsi_window.tail(10).min()
        earlier_rsi_low = rsi_window.head(10).min()
        
        return (recent_price_low < earlier_price_low and recent_rsi_low > earlier_rsi_low)
    
    def detect_bearish_divergence(self, data, index):
        """Detect bearish divergence between price and RSI"""
        if 'RSI' not in data.columns or index < 20:
            return False
        
        # Price making higher highs but RSI making lower highs
        price_window = data['Close'].iloc[index-20:index+1]
        rsi_window = data['RSI'].iloc[index-20:index+1]
        
        recent_price_high = price_window.tail(10).max()
        earlier_price_high = price_window.head(10).max()
        
        recent_rsi_high = rsi_window.tail(10).max()
        earlier_rsi_high = rsi_window.head(10).max()
        
        return (recent_price_high > earlier_price_high and recent_rsi_high < earlier_rsi_high)
    
    def calculate_smc_confidence(self, data, index):
        """Calculate confidence for Smart Money Concepts signals"""
        base_confidence = 0.7
        
        # Volume confirmation
        volume_ratio = data['Volume_Ratio'].iloc[index] if 'Volume_Ratio' in data.columns else 1
        if volume_ratio > 2.0:
            base_confidence += 0.1
        
        # RSI not overbought/oversold
        if 'RSI' in data.columns:
            rsi = data['RSI'].iloc[index]
            if 30 < rsi < 70:
                base_confidence += 0.05
        
        # Market structure confirmation
        if data['Higher_High'].iloc[index] or data['Lower_Low'].iloc[index]:
            base_confidence += 0.1
        
        return min(base_confidence, 0.95)
    
    def calculate_vpa_confidence(self, data, index):
        """Calculate confidence for VPA signals"""
        base_confidence = 0.75
        
        # Strong volume
        volume_ratio = data['Volume_Ratio'].iloc[index] if 'Volume_Ratio' in data.columns else 1
        if volume_ratio > 3.0:
            base_confidence += 0.15
        elif volume_ratio > 2.0:
            base_confidence += 0.08
        
        # VWAP distance
        if 'VWAP' in data.columns:
            price_vwap_diff = abs(data['Close'].iloc[index] - data['VWAP'].iloc[index]) / data['VWAP'].iloc[index]
            if price_vwap_diff > 0.005:  # 0.5% difference
                base_confidence += 0.05
        
        return min(base_confidence, 0.95)
    
    def calculate_divergence_confidence(self, data, index):
        """Calculate confidence for divergence signals"""
        base_confidence = 0.8
        
        # RSI level appropriate
        if 'RSI' in data.columns:
            rsi = data['RSI'].iloc[index]
            if rsi < 35 or rsi > 65:  # Near extremes
                base_confidence += 0.1
        
        # Volume support
        volume_ratio = data['Volume_Ratio'].iloc[index] if 'Volume_Ratio' in data.columns else 1
        if volume_ratio > 1.5:
            base_confidence += 0.05
        
        return min(base_confidence, 0.95)
    
    def generate_advanced_signals(self, data, symbol):
        """Generate all advanced trading signals"""
        if len(data) < 50:
            return []
        
        # Add advanced indicators
        data_with_indicators = self.add_advanced_indicators(data)
        
        all_signals = []
        
        # Generate signals from different strategies
        all_signals.extend(self.smart_money_concepts_strategy(data_with_indicators, symbol))
        all_signals.extend(self.volume_price_analysis_strategy(data_with_indicators, symbol))
        all_signals.extend(self.momentum_divergence_strategy(data_with_indicators, symbol))
        
        # Filter by minimum risk-reward and confidence
        filtered_signals = [
            s for s in all_signals 
            if s.get('risk_reward', 0) >= self.min_risk_reward and 
               s.get('confidence', 0) >= self.confidence_threshold
        ]
        
        # Sort by confidence (highest first)
        filtered_signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return filtered_signals[:3]  # Return top 3 signals