import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from technical_indicators import TechnicalIndicators

class SwingTradingStrategies:
    """Advanced swing trading strategies for multi-day to multi-week positions"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.active_strategies = [
            "Trend Following", "Mean Reversion", "Breakout", 
            "Support/Resistance", "Pattern Recognition", "Volume Analysis"
        ]
        self.max_risk_per_trade = 3.0  # Higher risk tolerance for swing trades
        self.min_risk_reward = 3.0     # Higher R:R for swing trades
        
        # Swing trading specific parameters
        self.trend_confirmation_period = 50  # Longer confirmation for swing trades
        self.volume_confirmation_period = 20
        self.pattern_lookback = 30
        self.breakout_confirmation_bars = 3
    
    def configure(self, strategies=None, max_risk=3.0, min_rr=3.0):
        """Configure strategy parameters for swing trading"""
        if strategies:
            self.active_strategies = strategies
        self.max_risk_per_trade = max_risk
        self.min_risk_reward = min_rr
    
    def add_technical_indicators(self, data):
        """Add comprehensive technical indicators for swing trading"""
        if data is None or len(data) < 50:
            return data
        
        # Price-based indicators
        data['MA_20'] = self.indicators.moving_average(data['Close'], 20)
        data['MA_50'] = self.indicators.moving_average(data['Close'], 50)
        data['MA_100'] = self.indicators.moving_average(data['Close'], 100)
        data['MA_200'] = self.indicators.moving_average(data['Close'], 200)
        
        # Exponential moving averages for trend following
        data['EMA_12'] = self.indicators.exponential_moving_average(data['Close'], 12)
        data['EMA_26'] = self.indicators.exponential_moving_average(data['Close'], 26)
        data['EMA_50'] = self.indicators.exponential_moving_average(data['Close'], 50)
        
        # Momentum indicators
        data['RSI'] = self.indicators.rsi(data['Close'], 14)
        data['RSI_21'] = self.indicators.rsi(data['Close'], 21)  # Longer RSI for swing
        
        # MACD for trend changes
        macd_data = self.indicators.macd(data['Close'], 12, 26, 9)
        data['MACD'] = macd_data['MACD']
        data['MACD_Signal'] = macd_data['Signal']
        data['MACD_Histogram'] = macd_data['Histogram']
        
        # Bollinger Bands for volatility and mean reversion
        bb_data = self.indicators.bollinger_bands(data['Close'], 20, 2)
        data['BB_Upper'] = bb_data['Upper']
        data['BB_Middle'] = bb_data['Middle']
        data['BB_Lower'] = bb_data['Lower']
        data['BB_Width'] = (bb_data['Upper'] - bb_data['Lower']) / bb_data['Middle']
        
        # Stochastic for momentum and overbought/oversold
        stoch_data = self.indicators.stochastic(data['High'], data['Low'], data['Close'], 14, 3)
        data['Stoch_K'] = stoch_data['%K']
        data['Stoch_D'] = stoch_data['%D']
        
        # Williams %R for additional momentum confirmation
        data['Williams_R'] = self.indicators.williams_r(data['High'], data['Low'], data['Close'], 14)
        
        # Volume indicators
        data['OBV'] = self.indicators.on_balance_volume(data['Close'], data['Volume'])
        data['Volume_MA'] = self.indicators.moving_average(data['Volume'], 20)
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
        
        # ATR for volatility and stop loss calculation
        data['ATR'] = self.indicators.average_true_range(data['High'], data['Low'], data['Close'], 14)
        data['ATR_21'] = self.indicators.average_true_range(data['High'], data['Low'], data['Close'], 21)
        
        # Support and resistance levels
        support_resistance = self.indicators.support_resistance_levels(
            data['High'], data['Low'], data['Close'], 20
        )
        data['Support'] = support_resistance['Support']
        data['Resistance'] = support_resistance['Resistance']
        
        # Pivot points for key levels
        pivot_data = self.indicators.pivot_points(data['High'], data['Low'], data['Close'])
        data['Pivot'] = pivot_data['Pivot']
        data['R1'] = pivot_data['R1']
        data['R2'] = pivot_data['R2']
        data['S1'] = pivot_data['S1']
        data['S2'] = pivot_data['S2']
        
        # Advanced indicators for swing trading
        data['CCI'] = self.indicators.commodity_channel_index(data['High'], data['Low'], data['Close'], 20)
        data['MFI'] = self.indicators.money_flow_index(data['High'], data['Low'], data['Close'], data['Volume'], 14)
        
        # Trend strength indicator
        data['ADX'] = self._calculate_adx(data)
        
        # Price rate of change for momentum
        data['ROC_10'] = self.indicators.rate_of_change(data['Close'], 10)
        data['ROC_20'] = self.indicators.rate_of_change(data['Close'], 20)
        
        return data
    
    def _calculate_adx(self, data, period=14):
        """Calculate Average Directional Index (ADX) for trend strength"""
        if len(data) < period + 1:
            return pd.Series(np.nan, index=data.index)
        
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        dm_plus = np.where((high - high.shift(1)) > (low.shift(1) - low), 
                          np.maximum(high - high.shift(1), 0), 0)
        dm_minus = np.where((low.shift(1) - low) > (high - high.shift(1)), 
                           np.maximum(low.shift(1) - low, 0), 0)
        
        # Smooth the values
        atr = tr.rolling(window=period).mean()
        di_plus = (pd.Series(dm_plus, index=data.index).rolling(window=period).mean() / atr) * 100
        di_minus = (pd.Series(dm_minus, index=data.index).rolling(window=period).mean() / atr) * 100
        
        # Calculate ADX
        dx = abs(di_plus - di_minus) / (di_plus + di_minus) * 100
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def trend_following_strategy(self, data, symbol):
        """Advanced trend following strategy for swing trading"""
        signals = []
        
        if len(data) < 200:
            return signals
        
        current_price = data['Close'].iloc[-1]
        ma_20 = data['MA_20'].iloc[-1]
        ma_50 = data['MA_50'].iloc[-1]
        ma_200 = data['MA_200'].iloc[-1]
        ema_12 = data['EMA_12'].iloc[-1]
        ema_26 = data['EMA_26'].iloc[-1]
        adx = data['ADX'].iloc[-1] if 'ADX' in data.columns else 25
        volume_ratio = data['Volume_Ratio'].iloc[-1]
        
        # Trend confirmation
        uptrend_confirmed = (
            current_price > ma_20 > ma_50 > ma_200 and  # Price above all MAs in order
            ema_12 > ema_26 and  # Short EMA above long EMA
            adx > 25 and  # Strong trend
            volume_ratio > 1.1  # Above average volume
        )
        
        downtrend_confirmed = (
            current_price < ma_20 < ma_50 < ma_200 and  # Price below all MAs in order
            ema_12 < ema_26 and  # Short EMA below long EMA
            adx > 25 and  # Strong trend
            volume_ratio > 1.1  # Above average volume
        )
        
        # Look for pullback entries in strong trends
        if uptrend_confirmed:
            # Look for pullback to EMA or support
            pullback_to_ema = abs(current_price - ema_12) / ema_12 < 0.02
            pullback_to_support = current_price <= data['Support'].iloc[-1] * 1.01
            
            if pullback_to_ema or pullback_to_support:
                stop_loss = min(data['Support'].iloc[-1], current_price * 0.95)
                target = current_price * 1.15  # 15% target for swing trade
                risk_reward = (target - current_price) / (current_price - stop_loss)
                
                if risk_reward >= self.min_risk_reward:
                    signals.append({
                        'symbol': symbol,
                        'action': 'BUY',
                        'price': current_price,
                        'stop_loss': stop_loss,
                        'target': target,
                        'risk_reward': risk_reward,
                        'strategy': 'Trend Following - Pullback Entry',
                        'confidence': 0.8,
                        'timestamp': data.index[-1],
                        'timeframe': 'Swing (5-15 days)'
                    })
        
        elif downtrend_confirmed:
            # Look for pullback to EMA or resistance
            pullback_to_ema = abs(current_price - ema_12) / ema_12 < 0.02
            pullback_to_resistance = current_price >= data['Resistance'].iloc[-1] * 0.99
            
            if pullback_to_ema or pullback_to_resistance:
                stop_loss = max(data['Resistance'].iloc[-1], current_price * 1.05)
                target = current_price * 0.85  # 15% target for swing trade
                risk_reward = (current_price - target) / (stop_loss - current_price)
                
                if risk_reward >= self.min_risk_reward:
                    signals.append({
                        'symbol': symbol,
                        'action': 'SELL',
                        'price': current_price,
                        'stop_loss': stop_loss,
                        'target': target,
                        'risk_reward': risk_reward,
                        'strategy': 'Trend Following - Pullback Entry',
                        'confidence': 0.8,
                        'timestamp': data.index[-1],
                        'timeframe': 'Swing (5-15 days)'
                    })
        
        return signals
    
    def mean_reversion_strategy(self, data, symbol):
        """Mean reversion strategy using Bollinger Bands and RSI"""
        signals = []
        
        if len(data) < 50:
            return signals
        
        current_price = data['Close'].iloc[-1]
        bb_upper = data['BB_Upper'].iloc[-1]
        bb_lower = data['BB_Lower'].iloc[-1]
        bb_middle = data['BB_Middle'].iloc[-1]
        rsi = data['RSI'].iloc[-1]
        rsi_21 = data['RSI_21'].iloc[-1]
        volume_ratio = data['Volume_Ratio'].iloc[-1]
        
        # Oversold condition for mean reversion buy
        oversold_condition = (
            current_price <= bb_lower and  # Price at or below lower BB
            rsi < 30 and  # RSI oversold
            rsi_21 < 35 and  # Longer RSI also oversold
            volume_ratio > 1.5  # High volume suggesting capitulation
        )
        
        # Overbought condition for mean reversion sell
        overbought_condition = (
            current_price >= bb_upper and  # Price at or above upper BB
            rsi > 70 and  # RSI overbought
            rsi_21 > 65 and  # Longer RSI also overbought
            volume_ratio > 1.5  # High volume suggesting distribution
        )
        
        if oversold_condition:
            stop_loss = current_price * 0.93  # 7% stop loss
            target = bb_middle  # Target middle BB (mean)
            risk_reward = (target - current_price) / (current_price - stop_loss)
            
            if risk_reward >= 2.0:  # Lower R:R requirement for mean reversion
                signals.append({
                    'symbol': symbol,
                    'action': 'BUY',
                    'price': current_price,
                    'stop_loss': stop_loss,
                    'target': target,
                    'risk_reward': risk_reward,
                    'strategy': 'Mean Reversion - Oversold Bounce',
                    'confidence': 0.75,
                    'timestamp': data.index[-1],
                    'timeframe': 'Swing (3-10 days)'
                })
        
        elif overbought_condition:
            stop_loss = current_price * 1.07  # 7% stop loss
            target = bb_middle  # Target middle BB (mean)
            risk_reward = (current_price - target) / (stop_loss - current_price)
            
            if risk_reward >= 2.0:
                signals.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'price': current_price,
                    'stop_loss': stop_loss,
                    'target': target,
                    'risk_reward': risk_reward,
                    'strategy': 'Mean Reversion - Overbought Correction',
                    'confidence': 0.75,
                    'timestamp': data.index[-1],
                    'timeframe': 'Swing (3-10 days)'
                })
        
        return signals
    
    def breakout_strategy(self, data, symbol):
        """Advanced breakout strategy with volume confirmation"""
        signals = []
        
        if len(data) < 50:
            return signals
        
        current_price = data['Close'].iloc[-1]
        resistance = data['Resistance'].iloc[-1]
        support = data['Support'].iloc[-1]
        volume_ratio = data['Volume_Ratio'].iloc[-1]
        atr = data['ATR'].iloc[-1]
        bb_width = data['BB_Width'].iloc[-1]
        
        # Consolidation detection (low volatility before breakout)
        consolidation_period = 10
        price_range = data['High'].tail(consolidation_period).max() - data['Low'].tail(consolidation_period).min()
        avg_price = data['Close'].tail(consolidation_period).mean()
        consolidation_ratio = price_range / avg_price
        
        is_consolidating = consolidation_ratio < 0.05  # Price range less than 5%
        volume_spike = volume_ratio > 2.0  # Volume at least 2x average
        volatility_compression = bb_width < 0.1  # Bollinger Bands squeezing
        
        # Bullish breakout
        resistance_breakout = (
            current_price > resistance * 1.005 and  # Clear break above resistance
            is_consolidating and  # Was consolidating before breakout
            volume_spike and  # Strong volume confirmation
            volatility_compression  # Low volatility before breakout
        )
        
        # Bearish breakdown
        support_breakdown = (
            current_price < support * 0.995 and  # Clear break below support
            is_consolidating and  # Was consolidating before breakdown
            volume_spike and  # Strong volume confirmation
            volatility_compression  # Low volatility before breakdown
        )
        
        if resistance_breakout:
            stop_loss = resistance * 0.98  # Stop just below broken resistance
            target = current_price + (current_price - resistance) * 2  # Measured move
            risk_reward = (target - current_price) / (current_price - stop_loss)
            
            if risk_reward >= self.min_risk_reward:
                signals.append({
                    'symbol': symbol,
                    'action': 'BUY',
                    'price': current_price,
                    'stop_loss': stop_loss,
                    'target': target,
                    'risk_reward': risk_reward,
                    'strategy': 'Breakout - Resistance Break',
                    'confidence': 0.85,
                    'timestamp': data.index[-1],
                    'timeframe': 'Swing (7-20 days)'
                })
        
        elif support_breakdown:
            stop_loss = support * 1.02  # Stop just above broken support
            target = current_price - (support - current_price) * 2  # Measured move
            risk_reward = (current_price - target) / (stop_loss - current_price)
            
            if risk_reward >= self.min_risk_reward:
                signals.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'price': current_price,
                    'stop_loss': stop_loss,
                    'target': target,
                    'risk_reward': risk_reward,
                    'strategy': 'Breakout - Support Break',
                    'confidence': 0.85,
                    'timestamp': data.index[-1],
                    'timeframe': 'Swing (7-20 days)'
                })
        
        return signals
    
    def support_resistance_strategy(self, data, symbol):
        """Strategy based on key support and resistance levels"""
        signals = []
        
        if len(data) < 50:
            return signals
        
        current_price = data['Close'].iloc[-1]
        support = data['Support'].iloc[-1]
        resistance = data['Resistance'].iloc[-1]
        s1 = data['S1'].iloc[-1] if 'S1' in data.columns else support
        r1 = data['R1'].iloc[-1] if 'R1' in data.columns else resistance
        volume_ratio = data['Volume_Ratio'].iloc[-1]
        rsi = data['RSI'].iloc[-1]
        
        # Support bounce setup
        near_support = abs(current_price - support) / current_price < 0.02
        support_bounce_signals = (
            near_support and
            rsi < 40 and  # Not extremely oversold
            volume_ratio > 1.2 and  # Above average volume
            data['Close'].iloc[-1] > data['Close'].iloc[-3]  # Recent price strength
        )
        
        # Resistance rejection setup
        near_resistance = abs(current_price - resistance) / current_price < 0.02
        resistance_rejection_signals = (
            near_resistance and
            rsi > 60 and  # Not extremely overbought
            volume_ratio > 1.2 and  # Above average volume
            data['Close'].iloc[-1] < data['Close'].iloc[-3]  # Recent price weakness
        )
        
        if support_bounce_signals:
            stop_loss = support * 0.96  # Stop below support
            target = resistance * 0.98  # Target near resistance
            risk_reward = (target - current_price) / (current_price - stop_loss)
            
            if risk_reward >= 2.5:
                signals.append({
                    'symbol': symbol,
                    'action': 'BUY',
                    'price': current_price,
                    'stop_loss': stop_loss,
                    'target': target,
                    'risk_reward': risk_reward,
                    'strategy': 'Support/Resistance - Support Bounce',
                    'confidence': 0.8,
                    'timestamp': data.index[-1],
                    'timeframe': 'Swing (5-12 days)'
                })
        
        elif resistance_rejection_signals:
            stop_loss = resistance * 1.04  # Stop above resistance
            target = support * 1.02  # Target near support
            risk_reward = (current_price - target) / (stop_loss - current_price)
            
            if risk_reward >= 2.5:
                signals.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'price': current_price,
                    'stop_loss': stop_loss,
                    'target': target,
                    'risk_reward': risk_reward,
                    'strategy': 'Support/Resistance - Resistance Rejection',
                    'confidence': 0.8,
                    'timestamp': data.index[-1],
                    'timeframe': 'Swing (5-12 days)'
                })
        
        return signals
    
    def pattern_recognition_strategy(self, data, symbol):
        """Advanced pattern recognition for swing trading"""
        signals = []
        
        if len(data) < 30:
            return signals
        
        # Double bottom pattern
        double_bottom = self._detect_double_bottom(data)
        if double_bottom:
            signals.extend(double_bottom)
        
        # Double top pattern
        double_top = self._detect_double_top(data)
        if double_top:
            signals.extend(double_top)
        
        # Cup and handle pattern
        cup_handle = self._detect_cup_handle(data)
        if cup_handle:
            signals.extend(cup_handle)
        
        return signals
    
    def _detect_double_bottom(self, data):
        """Detect double bottom reversal pattern"""
        signals = []
        
        if len(data) < 20:
            return signals
        
        # Look for two distinct lows with similar prices
        lows = data['Low'].tail(20)
        low_indices = []
        
        for i in range(1, len(lows) - 1):
            if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
                low_indices.append(i)
        
        if len(low_indices) >= 2:
            first_low = lows.iloc[low_indices[-2]]
            second_low = lows.iloc[low_indices[-1]]
            
            # Check if lows are similar (within 3%)
            if abs(first_low - second_low) / first_low < 0.03:
                current_price = data['Close'].iloc[-1]
                neckline = data['High'].tail(10).max()
                
                # Confirm breakout above neckline
                if current_price > neckline * 1.02:
                    stop_loss = second_low * 0.98
                    target = neckline + (neckline - second_low)
                    risk_reward = (target - current_price) / (current_price - stop_loss)
                    
                    if risk_reward >= 2.0:
                        signals.append({
                            'symbol': data.index[-1],  # Will be updated with actual symbol
                            'action': 'BUY',
                            'price': current_price,
                            'stop_loss': stop_loss,
                            'target': target,
                            'risk_reward': risk_reward,
                            'strategy': 'Pattern Recognition - Double Bottom',
                            'confidence': 0.85,
                            'timestamp': data.index[-1],
                            'timeframe': 'Swing (10-25 days)'
                        })
        
        return signals
    
    def _detect_double_top(self, data):
        """Detect double top reversal pattern"""
        signals = []
        
        if len(data) < 20:
            return signals
        
        # Look for two distinct highs with similar prices
        highs = data['High'].tail(20)
        high_indices = []
        
        for i in range(1, len(highs) - 1):
            if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
                high_indices.append(i)
        
        if len(high_indices) >= 2:
            first_high = highs.iloc[high_indices[-2]]
            second_high = highs.iloc[high_indices[-1]]
            
            # Check if highs are similar (within 3%)
            if abs(first_high - second_high) / first_high < 0.03:
                current_price = data['Close'].iloc[-1]
                neckline = data['Low'].tail(10).min()
                
                # Confirm breakdown below neckline
                if current_price < neckline * 0.98:
                    stop_loss = second_high * 1.02
                    target = neckline - (second_high - neckline)
                    risk_reward = (current_price - target) / (stop_loss - current_price)
                    
                    if risk_reward >= 2.0:
                        signals.append({
                            'symbol': data.index[-1],  # Will be updated with actual symbol
                            'action': 'SELL',
                            'price': current_price,
                            'stop_loss': stop_loss,
                            'target': target,
                            'risk_reward': risk_reward,
                            'strategy': 'Pattern Recognition - Double Top',
                            'confidence': 0.85,
                            'timestamp': data.index[-1],
                            'timeframe': 'Swing (10-25 days)'
                        })
        
        return signals
    
    def _detect_cup_handle(self, data):
        """Detect cup and handle pattern"""
        # Simplified cup and handle detection
        # In a real implementation, this would be more sophisticated
        return []
    
    def volume_analysis_strategy(self, data, symbol):
        """Volume-based swing trading strategy"""
        signals = []
        
        if len(data) < 20:
            return signals
        
        current_price = data['Close'].iloc[-1]
        volume_ratio = data['Volume_Ratio'].iloc[-1]
        obv = data['OBV'].iloc[-1]
        obv_trend = data['OBV'].iloc[-1] - data['OBV'].iloc[-5]
        
        # Volume breakout with price confirmation
        volume_breakout = volume_ratio > 3.0  # Very high volume
        price_strength = current_price > data['Close'].iloc[-5]  # Price trending up
        obv_confirmation = obv_trend > 0  # OBV trending up
        
        # Volume selling climax
        volume_climax = volume_ratio > 2.5  # High volume
        price_weakness = current_price < data['Close'].iloc[-5]  # Price trending down
        potential_exhaustion = data['RSI'].iloc[-1] < 30  # Oversold
        
        if volume_breakout and price_strength and obv_confirmation:
            stop_loss = current_price * 0.93
            target = current_price * 1.20
            risk_reward = (target - current_price) / (current_price - stop_loss)
            
            if risk_reward >= self.min_risk_reward:
                signals.append({
                    'symbol': symbol,
                    'action': 'BUY',
                    'price': current_price,
                    'stop_loss': stop_loss,
                    'target': target,
                    'risk_reward': risk_reward,
                    'strategy': 'Volume Analysis - Breakout',
                    'confidence': 0.8,
                    'timestamp': data.index[-1],
                    'timeframe': 'Swing (7-18 days)'
                })
        
        elif volume_climax and price_weakness and potential_exhaustion:
            stop_loss = current_price * 1.07
            target = current_price * 0.85
            risk_reward = (current_price - target) / (stop_loss - current_price)
            
            if risk_reward >= self.min_risk_reward:
                signals.append({
                    'symbol': symbol,
                    'action': 'BUY',  # Buying the climax (contrarian)
                    'price': current_price,
                    'stop_loss': stop_loss,
                    'target': target,
                    'risk_reward': risk_reward,
                    'strategy': 'Volume Analysis - Selling Climax',
                    'confidence': 0.75,
                    'timestamp': data.index[-1],
                    'timeframe': 'Swing (5-15 days)'
                })
        
        return signals
    
    def generate_signals(self, data, symbol):
        """Generate swing trading signals using all active strategies"""
        all_signals = []
        
        if "Trend Following" in self.active_strategies:
            all_signals.extend(self.trend_following_strategy(data, symbol))
        
        if "Mean Reversion" in self.active_strategies:
            all_signals.extend(self.mean_reversion_strategy(data, symbol))
        
        if "Breakout" in self.active_strategies:
            all_signals.extend(self.breakout_strategy(data, symbol))
        
        if "Support/Resistance" in self.active_strategies:
            all_signals.extend(self.support_resistance_strategy(data, symbol))
        
        if "Pattern Recognition" in self.active_strategies:
            all_signals.extend(self.pattern_recognition_strategy(data, symbol))
        
        if "Volume Analysis" in self.active_strategies:
            all_signals.extend(self.volume_analysis_strategy(data, symbol))
        
        # Update symbol in pattern recognition signals
        for signal in all_signals:
            signal['symbol'] = symbol
        
        # Remove duplicate signals and sort by confidence
        unique_signals = []
        for signal in all_signals:
            is_duplicate = any(
                s['symbol'] == signal['symbol'] and 
                s['action'] == signal['action'] and
                abs(s['price'] - signal['price']) / signal['price'] < 0.01
                for s in unique_signals
            )
            if not is_duplicate:
                unique_signals.append(signal)
        
        # Sort by confidence
        unique_signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        return unique_signals[:3]  # Return top 3 signals per instrument