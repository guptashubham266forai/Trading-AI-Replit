import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from technical_indicators import TechnicalIndicators

class PredictiveAnalysis:
    """Advanced predictive analysis for stock movements"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        
    def detect_accumulation_distribution(self, data):
        """Detect institutional accumulation/distribution patterns"""
        if len(data) < 20:
            return None
            
        # Price vs Volume divergence analysis
        price_trend = data['Close'].pct_change(10).iloc[-1]
        volume_trend = (data['Volume'].tail(10).mean() / data['Volume'].tail(20).mean()) - 1
        
        # Accumulation: Price steady/up + Volume increasing
        if price_trend >= -0.02 and volume_trend > 0.2:
            return {
                'pattern': 'Accumulation',
                'strength': min(100, volume_trend * 100),
                'prediction': 'Bullish breakout likely',
                'confidence': 0.7 + min(0.2, volume_trend)
            }
        
        # Distribution: Price steady/down + Volume increasing
        elif price_trend <= 0.02 and volume_trend > 0.2:
            return {
                'pattern': 'Distribution',
                'strength': min(100, volume_trend * 100),
                'prediction': 'Bearish breakdown likely',
                'confidence': 0.7 + min(0.2, volume_trend)
            }
        
        return None
    
    def detect_pre_breakout_setup(self, data):
        """Detect stocks ready for breakout before it happens"""
        setups = []
        
        if len(data) < 50:
            return setups
        
        current_price = data['Close'].iloc[-1]
        volume_avg = data['Volume'].tail(20).mean()
        current_volume = data['Volume'].iloc[-1]
        
        # Bollinger Band Squeeze
        if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
            bb_width = (data['BB_Upper'].iloc[-1] - data['BB_Lower'].iloc[-1]) / current_price
            bb_width_avg = ((data['BB_Upper'] - data['BB_Lower']) / data['Close']).tail(20).mean()
            
            if bb_width < bb_width_avg * 0.7:  # Squeeze detected
                setups.append({
                    'type': 'Bollinger Squeeze',
                    'signal': 'Volatility compression - Big move coming',
                    'probability': 0.75,
                    'timeframe': '1-3 hours'
                })
        
        # Triangle/Wedge Formation
        highs = data['High'].tail(10)
        lows = data['Low'].tail(10)
        
        # Calculate trend lines
        high_slope = self._calculate_slope(highs)
        low_slope = self._calculate_slope(lows)
        
        # Converging triangle
        if abs(high_slope) > 0 and abs(low_slope) > 0 and np.sign(high_slope) != np.sign(low_slope):
            setups.append({
                'type': 'Triangle Formation',
                'signal': 'Converging triangle - Breakout imminent',
                'probability': 0.65,
                'timeframe': '30-60 minutes'
            })
        
        # Volume Dry-up before breakout
        recent_volume = data['Volume'].tail(5).mean()
        if recent_volume < volume_avg * 0.7 and current_volume > recent_volume * 1.5:
            setups.append({
                'type': 'Volume Surge',
                'signal': 'Volume spike after dry-up - Move starting',
                'probability': 0.8,
                'timeframe': '15-30 minutes'
            })
        
        return setups
    
    def detect_momentum_divergence(self, data):
        """Detect RSI/MACD divergence before price reversal"""
        divergences = []
        
        if len(data) < 30 or 'RSI' not in data.columns:
            return divergences
        
        # Look for divergences in last 10 periods
        price_data = data['Close'].tail(10)
        rsi_data = data['RSI'].tail(10)
        
        # Find recent peaks and troughs
        price_peaks = self._find_peaks(price_data)
        rsi_peaks = self._find_peaks(rsi_data)
        
        # Bullish divergence: Lower price lows, Higher RSI lows
        if len(price_peaks['lows']) >= 2 and len(rsi_peaks['lows']) >= 2:
            if (price_peaks['lows'][-1] < price_peaks['lows'][-2] and 
                rsi_peaks['lows'][-1] > rsi_peaks['lows'][-2]):
                divergences.append({
                    'type': 'Bullish RSI Divergence',
                    'signal': 'Price making lower lows, RSI making higher lows',
                    'prediction': 'Upward reversal likely',
                    'confidence': 0.7,
                    'target_move': '+3-5%'
                })
        
        # Bearish divergence: Higher price highs, Lower RSI highs
        if len(price_peaks['highs']) >= 2 and len(rsi_peaks['highs']) >= 2:
            if (price_peaks['highs'][-1] > price_peaks['highs'][-2] and 
                rsi_peaks['highs'][-1] < rsi_peaks['highs'][-2]):
                divergences.append({
                    'type': 'Bearish RSI Divergence',
                    'signal': 'Price making higher highs, RSI making lower highs',
                    'prediction': 'Downward reversal likely',
                    'confidence': 0.7,
                    'target_move': '-3-5%'
                })
        
        return divergences
    
    def detect_smart_money_flow(self, data):
        """Detect institutional money flow patterns"""
        if len(data) < 20:
            return None
        
        # Calculate money flow based on price and volume
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        money_flow = typical_price * data['Volume']
        
        # Compare recent vs historical
        recent_flow = money_flow.tail(5).sum()
        historical_flow = money_flow.tail(20).sum() / 4  # Average of 5-day periods
        
        flow_ratio = recent_flow / historical_flow if historical_flow > 0 else 1
        
        # Price action analysis
        price_change = (data['Close'].iloc[-1] - data['Close'].iloc[-6]) / data['Close'].iloc[-6]
        
        # Smart money accumulation: Heavy volume, controlled price movement
        if flow_ratio > 1.5 and abs(price_change) < 0.03:
            return {
                'pattern': 'Smart Money Accumulation',
                'signal': 'Institutions quietly accumulating',
                'confidence': min(0.9, flow_ratio / 2),
                'prediction': 'Strong upward move expected',
                'timeframe': '1-4 hours'
            }
        
        # Smart money distribution: Heavy volume with price weakness
        elif flow_ratio > 1.5 and price_change < -0.02:
            return {
                'pattern': 'Smart Money Distribution',
                'signal': 'Institutions offloading positions',
                'confidence': min(0.9, flow_ratio / 2),
                'prediction': 'Significant downward move expected',
                'timeframe': '1-4 hours'
            }
        
        return None
    
    def predict_next_move(self, data, symbol):
        """Comprehensive prediction combining all signals"""
        predictions = []
        
        # Get all analysis results
        accumulation = self.detect_accumulation_distribution(data)
        breakout_setups = self.detect_pre_breakout_setup(data)
        divergences = self.detect_momentum_divergence(data)
        smart_money = self.detect_smart_money_flow(data)
        
        # Combine signals for overall prediction
        bullish_signals = 0
        bearish_signals = 0
        total_confidence = 0
        
        if accumulation:
            if accumulation['pattern'] == 'Accumulation':
                bullish_signals += accumulation['confidence']
            else:
                bearish_signals += accumulation['confidence']
            total_confidence += accumulation['confidence']
        
        for divergence in divergences:
            if 'Bullish' in divergence['type']:
                bullish_signals += divergence['confidence']
            else:
                bearish_signals += divergence['confidence']
            total_confidence += divergence['confidence']
        
        if smart_money:
            if 'Accumulation' in smart_money['pattern']:
                bullish_signals += smart_money['confidence']
            else:
                bearish_signals += smart_money['confidence']
            total_confidence += smart_money['confidence']
        
        # Generate final prediction
        if total_confidence > 0:
            net_sentiment = (bullish_signals - bearish_signals) / total_confidence
            
            if net_sentiment > 0.3:
                direction = 'BULLISH'
                probability = min(95, 60 + (net_sentiment * 35))
            elif net_sentiment < -0.3:
                direction = 'BEARISH'
                probability = min(95, 60 + (abs(net_sentiment) * 35))
            else:
                direction = 'NEUTRAL'
                probability = 50
            
            predictions.append({
                'symbol': symbol,
                'direction': direction,
                'probability': probability,
                'timeframe': '30 minutes - 2 hours',
                'supporting_signals': len([x for x in [accumulation, smart_money] if x]) + len(divergences) + len(breakout_setups),
                'key_levels': self._calculate_key_levels(data)
            })
        
        return {
            'predictions': predictions,
            'accumulation': accumulation,
            'breakout_setups': breakout_setups,
            'divergences': divergences,
            'smart_money': smart_money
        }
    
    def _calculate_slope(self, series):
        """Calculate slope of a price series"""
        try:
            x = np.arange(len(series))
            y = series.values
            return np.polyfit(x, y, 1)[0]
        except:
            return 0
    
    def _find_peaks(self, series):
        """Find peaks and troughs in a series"""
        try:
            from scipy.signal import argrelextrema
            highs_idx = argrelextrema(series.values, np.greater, order=1)[0]
            lows_idx = argrelextrema(series.values, np.less, order=1)[0]
            
            return {
                'highs': [series.iloc[i] for i in highs_idx],
                'lows': [series.iloc[i] for i in lows_idx]
            }
        except:
            return {'highs': [], 'lows': []}
    
    def _calculate_key_levels(self, data):
        """Calculate key support/resistance levels"""
        try:
            current_price = data['Close'].iloc[-1]
            
            # Recent high/low
            high_20 = data['High'].tail(20).max()
            low_20 = data['Low'].tail(20).min()
            
            # Volume-weighted levels
            vwap = ((data['High'] + data['Low'] + data['Close']) / 3 * data['Volume']).tail(20).sum() / data['Volume'].tail(20).sum()
            
            return {
                'current_price': current_price,
                'resistance': high_20,
                'support': low_20,
                'vwap': vwap,
                'breakout_level': current_price * 1.02,
                'breakdown_level': current_price * 0.98
            }
        except:
            return {}