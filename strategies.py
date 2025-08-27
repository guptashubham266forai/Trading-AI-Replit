import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from technical_indicators import TechnicalIndicators
import pytz

class TradingStrategies:
    """Implements professional intraday trading strategies"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.active_strategies = ["Moving Average Crossover", "RSI", "MACD"]
        self.max_risk_percent = 2.0
        self.min_risk_reward = 2.0
        self.confidence_threshold = 0.7
    
    def normalize_timestamp(self, timestamp):
        """Convert timestamp to timezone-naive datetime for consistency"""
        try:
            if hasattr(timestamp, 'to_pydatetime'):
                timestamp = timestamp.to_pydatetime()
            
            # If timezone-aware, convert to naive
            if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is not None:
                timestamp = timestamp.replace(tzinfo=None)
            
            return timestamp
        except:
            return datetime.now()
    
    def configure(self, strategies=None, max_risk=2.0, min_rr=2.0, min_confidence=90.0):
        """Configure strategy parameters"""
        if strategies:
            self.active_strategies = strategies
        self.max_risk_percent = max_risk
        self.min_risk_reward = min_rr
        self.confidence_threshold = min_confidence / 100.0  # Convert percentage to decimal
    
    def add_technical_indicators(self, data):
        """Add all technical indicators to the data"""
        if len(data) < 50:  # Need sufficient data for indicators
            return data
        
        # Add moving averages
        data['MA_20'] = self.indicators.moving_average(data['Close'], 20)
        data['MA_50'] = self.indicators.moving_average(data['Close'], 50)
        data['EMA_12'] = self.indicators.exponential_moving_average(data['Close'], 12)
        data['EMA_26'] = self.indicators.exponential_moving_average(data['Close'], 26)
        
        # Add RSI
        data['RSI'] = self.indicators.rsi(data['Close'])
        
        # Add MACD
        macd_data = self.indicators.macd(data['Close'])
        data['MACD'] = macd_data['MACD']
        data['MACD_Signal'] = macd_data['Signal']
        data['MACD_Histogram'] = macd_data['Histogram']
        
        # Add Bollinger Bands
        bb_data = self.indicators.bollinger_bands(data['Close'])
        data['BB_Upper'] = bb_data['Upper']
        data['BB_Middle'] = bb_data['Middle']
        data['BB_Lower'] = bb_data['Lower']
        
        # Add Stochastic
        stoch_data = self.indicators.stochastic(data['High'], data['Low'], data['Close'])
        data['Stoch_K'] = stoch_data['%K']
        data['Stoch_D'] = stoch_data['%D']
        
        # Add support and resistance levels
        sr_levels = self.indicators.support_resistance_levels(data['High'], data['Low'], data['Close'])
        data['Support'] = sr_levels['Support']
        data['Resistance'] = sr_levels['Resistance']
        
        return data
    
    def moving_average_crossover_strategy(self, data, symbol):
        """Moving Average Crossover Strategy"""
        signals = []
        
        if len(data) < 50 or 'MA_20' not in data.columns or 'MA_50' not in data.columns:
            return signals
        
        ma_20 = data['MA_20']
        ma_50 = data['MA_50']
        close_prices = data['Close']
        
        # Look for crossovers in the last few periods
        for i in range(max(2, len(data) - 10), len(data)):
            if pd.isna(ma_20.iloc[i]) or pd.isna(ma_50.iloc[i]):
                continue
            
            prev_ma20 = ma_20.iloc[i-1]
            prev_ma50 = ma_50.iloc[i-1]
            curr_ma20 = ma_20.iloc[i]
            curr_ma50 = ma_50.iloc[i]
            current_price = close_prices.iloc[i]
            
            # Golden Cross (Bullish Signal)
            if prev_ma20 <= prev_ma50 and curr_ma20 > curr_ma50:
                confidence = self.calculate_signal_confidence(data, i, 'BUY')
                
                if confidence >= self.confidence_threshold:
                    stop_loss = current_price * 0.98  # 2% stop loss
                    target = current_price * 1.04     # 4% target
                    risk_reward = (target - current_price) / (current_price - stop_loss)
                    
                    if risk_reward >= self.min_risk_reward:
                        signals.append({
                            'symbol': symbol,
                            'action': 'BUY',
                            'price': current_price,
                            'strategy': 'MA Crossover (Golden Cross)',
                            'confidence': confidence,
                            'stop_loss': stop_loss,
                            'target': target,
                            'risk_reward': risk_reward,
                            'timestamp': self.normalize_timestamp(data.index[i])
                        })
            
            # Death Cross (Bearish Signal)
            elif prev_ma20 >= prev_ma50 and curr_ma20 < curr_ma50:
                confidence = self.calculate_signal_confidence(data, i, 'SELL')
                
                if confidence >= self.confidence_threshold:
                    stop_loss = current_price * 1.02  # 2% stop loss
                    target = current_price * 0.96     # 4% target
                    risk_reward = (current_price - target) / (stop_loss - current_price)
                    
                    if risk_reward >= self.min_risk_reward:
                        signals.append({
                            'symbol': symbol,
                            'action': 'SELL',
                            'price': current_price,
                            'strategy': 'MA Crossover (Death Cross)',
                            'confidence': confidence,
                            'stop_loss': stop_loss,
                            'target': target,
                            'risk_reward': risk_reward,
                            'timestamp': self.normalize_timestamp(data.index[i])
                        })
        
        return signals
    
    def rsi_strategy(self, data, symbol):
        """RSI Oversold/Overbought Strategy"""
        signals = []
        
        if len(data) < 20 or 'RSI' not in data.columns:
            return signals
        
        rsi = data['RSI']
        close_prices = data['Close']
        
        # Look for RSI signals in recent periods
        for i in range(max(1, len(data) - 5), len(data)):
            if pd.isna(rsi.iloc[i]):
                continue
            
            current_rsi = rsi.iloc[i]
            current_price = close_prices.iloc[i]
            
            # RSI Oversold (Bullish Signal)
            if current_rsi < 30 and (i == 0 or rsi.iloc[i-1] >= 30):
                confidence = self.calculate_signal_confidence(data, i, 'BUY')
                
                if confidence >= self.confidence_threshold:
                    stop_loss = current_price * 0.97  # 3% stop loss
                    target = current_price * 1.06     # 6% target
                    risk_reward = (target - current_price) / (current_price - stop_loss)
                    
                    if risk_reward >= self.min_risk_reward:
                        signals.append({
                            'symbol': symbol,
                            'action': 'BUY',
                            'price': current_price,
                            'strategy': 'RSI Oversold',
                            'confidence': confidence,
                            'stop_loss': stop_loss,
                            'target': target,
                            'risk_reward': risk_reward,
                            'timestamp': self.normalize_timestamp(data.index[i])
                        })
            
            # RSI Overbought (Bearish Signal)
            elif current_rsi > 70 and (i == 0 or rsi.iloc[i-1] <= 70):
                confidence = self.calculate_signal_confidence(data, i, 'SELL')
                
                if confidence >= self.confidence_threshold:
                    stop_loss = current_price * 1.03  # 3% stop loss
                    target = current_price * 0.94     # 6% target
                    risk_reward = (current_price - target) / (stop_loss - current_price)
                    
                    if risk_reward >= self.min_risk_reward:
                        signals.append({
                            'symbol': symbol,
                            'action': 'SELL',
                            'price': current_price,
                            'strategy': 'RSI Overbought',
                            'confidence': confidence,
                            'stop_loss': stop_loss,
                            'target': target,
                            'risk_reward': risk_reward,
                            'timestamp': self.normalize_timestamp(data.index[i])
                        })
        
        return signals
    
    def macd_strategy(self, data, symbol):
        """MACD Signal Line Crossover Strategy"""
        signals = []
        
        if len(data) < 30 or 'MACD' not in data.columns or 'MACD_Signal' not in data.columns:
            return signals
        
        macd = data['MACD']
        macd_signal = data['MACD_Signal']
        close_prices = data['Close']
        
        # Look for MACD crossovers
        for i in range(max(1, len(data) - 5), len(data)):
            if pd.isna(macd.iloc[i]) or pd.isna(macd_signal.iloc[i]):
                continue
            
            prev_macd = macd.iloc[i-1]
            prev_signal = macd_signal.iloc[i-1]
            curr_macd = macd.iloc[i]
            curr_signal = macd_signal.iloc[i]
            current_price = close_prices.iloc[i]
            
            # MACD Bullish Crossover
            if prev_macd <= prev_signal and curr_macd > curr_signal:
                confidence = self.calculate_signal_confidence(data, i, 'BUY')
                
                if confidence >= self.confidence_threshold:
                    stop_loss = current_price * 0.975  # 2.5% stop loss
                    target = current_price * 1.05      # 5% target
                    risk_reward = (target - current_price) / (current_price - stop_loss)
                    
                    if risk_reward >= self.min_risk_reward:
                        signals.append({
                            'symbol': symbol,
                            'action': 'BUY',
                            'price': current_price,
                            'strategy': 'MACD Bullish Crossover',
                            'confidence': confidence,
                            'stop_loss': stop_loss,
                            'target': target,
                            'risk_reward': risk_reward,
                            'timestamp': self.normalize_timestamp(data.index[i])
                        })
            
            # MACD Bearish Crossover
            elif prev_macd >= prev_signal and curr_macd < curr_signal:
                confidence = self.calculate_signal_confidence(data, i, 'SELL')
                
                if confidence >= self.confidence_threshold:
                    stop_loss = current_price * 1.025  # 2.5% stop loss
                    target = current_price * 0.95      # 5% target
                    risk_reward = (current_price - target) / (stop_loss - current_price)
                    
                    if risk_reward >= self.min_risk_reward:
                        signals.append({
                            'symbol': symbol,
                            'action': 'SELL',
                            'price': current_price,
                            'strategy': 'MACD Bearish Crossover',
                            'confidence': confidence,
                            'stop_loss': stop_loss,
                            'target': target,
                            'risk_reward': risk_reward,
                            'timestamp': self.normalize_timestamp(data.index[i])
                        })
        
        return signals
    
    def bollinger_bands_strategy(self, data, symbol):
        """Bollinger Bands Mean Reversion Strategy"""
        signals = []
        
        if len(data) < 20 or 'BB_Upper' not in data.columns:
            return signals
        
        close_prices = data['Close']
        bb_upper = data['BB_Upper']
        bb_lower = data['BB_Lower']
        bb_middle = data['BB_Middle']
        
        # Look for price touching bands
        for i in range(max(1, len(data) - 5), len(data)):
            if pd.isna(bb_upper.iloc[i]) or pd.isna(bb_lower.iloc[i]):
                continue
            
            current_price = close_prices.iloc[i]
            
            # Price touching lower band (Bullish Signal)
            if current_price <= bb_lower.iloc[i] * 1.001:  # Small tolerance
                confidence = self.calculate_signal_confidence(data, i, 'BUY')
                
                if confidence >= self.confidence_threshold:
                    stop_loss = bb_lower.iloc[i] * 0.98
                    target = bb_middle.iloc[i]
                    risk_reward = (target - current_price) / (current_price - stop_loss)
                    
                    if risk_reward >= self.min_risk_reward:
                        signals.append({
                            'symbol': symbol,
                            'action': 'BUY',
                            'price': current_price,
                            'strategy': 'Bollinger Bands (Lower Band Bounce)',
                            'confidence': confidence,
                            'stop_loss': stop_loss,
                            'target': target,
                            'risk_reward': risk_reward,
                            'timestamp': self.normalize_timestamp(data.index[i])
                        })
            
            # Price touching upper band (Bearish Signal)
            elif current_price >= bb_upper.iloc[i] * 0.999:  # Small tolerance
                confidence = self.calculate_signal_confidence(data, i, 'SELL')
                
                if confidence >= self.confidence_threshold:
                    stop_loss = bb_upper.iloc[i] * 1.02
                    target = bb_middle.iloc[i]
                    risk_reward = (current_price - target) / (stop_loss - current_price)
                    
                    if risk_reward >= self.min_risk_reward:
                        signals.append({
                            'symbol': symbol,
                            'action': 'SELL',
                            'price': current_price,
                            'strategy': 'Bollinger Bands (Upper Band Rejection)',
                            'confidence': confidence,
                            'stop_loss': stop_loss,
                            'target': target,
                            'risk_reward': risk_reward,
                            'timestamp': self.normalize_timestamp(data.index[i])
                        })
        
        return signals
    
    def calculate_signal_confidence(self, data, index, action):
        """Calculate confidence score for a signal based on multiple factors"""
        confidence_factors = []
        
        try:
            # Volume confirmation
            if 'Volume' in data.columns and index > 0:
                current_volume = data['Volume'].iloc[index]
                avg_volume = data['Volume'].iloc[max(0, index-10):index].mean()
                if current_volume > avg_volume * 1.2:  # 20% higher than average
                    confidence_factors.append(0.2)
            
            # RSI confirmation
            if 'RSI' in data.columns:
                rsi = data['RSI'].iloc[index]
                if action == 'BUY' and 20 <= rsi <= 40:
                    confidence_factors.append(0.15)
                elif action == 'SELL' and 60 <= rsi <= 80:
                    confidence_factors.append(0.15)
            
            # MACD histogram confirmation
            if 'MACD_Histogram' in data.columns and index > 0:
                curr_hist = data['MACD_Histogram'].iloc[index]
                prev_hist = data['MACD_Histogram'].iloc[index-1]
                
                if action == 'BUY' and curr_hist > prev_hist and curr_hist > 0:
                    confidence_factors.append(0.15)
                elif action == 'SELL' and curr_hist < prev_hist and curr_hist < 0:
                    confidence_factors.append(0.15)
            
            # Price momentum
            if index >= 3:
                recent_prices = data['Close'].iloc[index-3:index+1]
                price_momentum = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
                
                if action == 'BUY' and price_momentum > 0.005:  # 0.5% positive momentum
                    confidence_factors.append(0.1)
                elif action == 'SELL' and price_momentum < -0.005:  # 0.5% negative momentum
                    confidence_factors.append(0.1)
            
            # Support/resistance confirmation
            if 'Support' in data.columns and 'Resistance' in data.columns:
                current_price = data['Close'].iloc[index]
                support = data['Support'].iloc[index]
                resistance = data['Resistance'].iloc[index]
                
                if action == 'BUY' and not pd.isna(support) and current_price <= support * 1.02:
                    confidence_factors.append(0.1)
                elif action == 'SELL' and not pd.isna(resistance) and current_price >= resistance * 0.98:
                    confidence_factors.append(0.1)
            
            # Base confidence
            base_confidence = 0.5
            
            # Total confidence
            total_confidence = base_confidence + sum(confidence_factors)
            
            return min(total_confidence, 1.0)  # Cap at 100%
            
        except Exception as e:
            return 0.5  # Default confidence if calculation fails
    
    def generate_signals(self, data, symbol):
        """Generate trading signals using all active strategies"""
        all_signals = []
        
        try:
            # Run each active strategy
            for strategy in self.active_strategies:
                if strategy == "Moving Average Crossover":
                    signals = self.moving_average_crossover_strategy(data, symbol)
                    all_signals.extend(signals)
                
                elif strategy == "RSI":
                    signals = self.rsi_strategy(data, symbol)
                    all_signals.extend(signals)
                
                elif strategy == "MACD":
                    signals = self.macd_strategy(data, symbol)
                    all_signals.extend(signals)
                
                elif strategy == "Bollinger Bands":
                    signals = self.bollinger_bands_strategy(data, symbol)
                    all_signals.extend(signals)
            
            # Validate and remove duplicate signals
            unique_signals = []
            for signal in all_signals:
                # Price validation - filter out unrealistic prices
                price = signal.get('price', 0)
                if price <= 0 or price > 1000000:
                    continue
                    
                # Check for duplicates (same action within 15 minutes)
                is_duplicate = False
                for existing in unique_signals:
                    if (existing['symbol'] == signal['symbol'] and
                        existing['action'] == signal['action'] and
                        abs((existing['timestamp'] - signal['timestamp']).total_seconds()) < 900):  # 15 minutes
                        # Keep the signal with higher confidence
                        if signal['confidence'] > existing['confidence']:
                            unique_signals.remove(existing)
                        else:
                            is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_signals.append(signal)
            
            return unique_signals
            
        except Exception as e:
            st.warning(f"Error generating signals for {symbol}: {str(e)}")
            return []
