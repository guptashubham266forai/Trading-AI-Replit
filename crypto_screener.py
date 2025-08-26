import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

class CryptoScreener:
    """Screen cryptocurrencies based on various criteria for trading"""
    
    def __init__(self):
        # Top liquid cryptocurrencies for trading (optimized list)
        self.liquid_cryptos = [
            'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD',
            'SOL-USD', 'DOGE-USD', 'DOT-USD', 'AVAX-USD', 'LTC-USD',
            'LINK-USD', 'BCH-USD', 'ALGO-USD', 'ATOM-USD', 'NEAR-USD'
        ]
        
        # Screening criteria (crypto markets are more volatile)
        self.min_volume = 1000000  # Minimum daily volume (higher for crypto)
        self.min_price = 0.001     # Minimum crypto price (very low for altcoins)
        self.max_price = 100000    # Maximum crypto price
        self.min_volatility = 0.02  # Minimum 2% daily volatility
        self.max_volatility = 0.50  # Maximum 50% daily volatility (crypto can be very volatile)
    
    def get_liquid_cryptos(self):
        """Get list of liquid cryptocurrencies suitable for trading (fast loading)"""
        return self.liquid_cryptos[:10]  # Limit to top 10 for faster loading
    
    def filter_by_volume(self, crypto_data_dict, min_volume_ratio=1.2):
        """Filter cryptocurrencies by volume criteria"""
        filtered_cryptos = {}
        
        for symbol, data in crypto_data_dict.items():
            if data is None or len(data) < 10:
                continue
            
            try:
                # Calculate average volume over last 10 periods
                avg_volume = data['Volume'].tail(10).mean()
                current_volume = data['Volume'].iloc[-1]
                
                # Check if current volume is above minimum and above average
                if (current_volume >= self.min_volume and 
                    current_volume >= avg_volume * min_volume_ratio):
                    filtered_cryptos[symbol] = data
                    
            except Exception as e:
                continue
        
        return filtered_cryptos
    
    def filter_by_price_range(self, crypto_data_dict):
        """Filter cryptocurrencies by price range"""
        filtered_cryptos = {}
        
        for symbol, data in crypto_data_dict.items():
            if data is None or len(data) == 0:
                continue
            
            try:
                current_price = data['Close'].iloc[-1]
                
                if self.min_price <= current_price <= self.max_price:
                    filtered_cryptos[symbol] = data
                    
            except Exception as e:
                continue
        
        return filtered_cryptos
    
    def filter_by_volatility(self, crypto_data_dict, window=20):
        """Filter cryptocurrencies by volatility range"""
        filtered_cryptos = {}
        
        for symbol, data in crypto_data_dict.items():
            if data is None or len(data) < window:
                continue
            
            try:
                # Calculate historical volatility (standard deviation of returns)
                returns = data['Close'].pct_change().dropna()
                volatility = returns.tail(window).std()
                
                if self.min_volatility <= volatility <= self.max_volatility:
                    filtered_cryptos[symbol] = data
                    
            except Exception as e:
                continue
        
        return filtered_cryptos
    
    def filter_by_momentum(self, crypto_data_dict, min_momentum=0.05):
        """Filter cryptocurrencies showing momentum (price movement)"""
        filtered_cryptos = {}
        
        for symbol, data in crypto_data_dict.items():
            if data is None or len(data) < 5:
                continue
            
            try:
                # Calculate momentum over last 5 periods
                current_price = data['Close'].iloc[-1]
                price_5_periods_ago = data['Close'].iloc[-5]
                
                momentum = abs((current_price - price_5_periods_ago) / price_5_periods_ago)
                
                if momentum >= min_momentum:
                    filtered_cryptos[symbol] = data
                    
            except Exception as e:
                continue
        
        return filtered_cryptos
    
    def detect_whale_movements(self, crypto_data_dict, volume_spike_threshold=3.0):
        """Detect potential whale movements (large volume spikes)"""
        whale_activity = {}
        
        for symbol, data in crypto_data_dict.items():
            if data is None or len(data) < 20:
                continue
            
            try:
                current_volume = data['Volume'].iloc[-1]
                avg_volume = data['Volume'].tail(20).mean()
                
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                
                if volume_ratio >= volume_spike_threshold:
                    price_change = data['Close'].pct_change(1).iloc[-1]
                    
                    whale_activity[symbol] = {
                        'data': data,
                        'volume_ratio': volume_ratio,
                        'price_change': price_change,
                        'pattern': 'Accumulation' if abs(price_change) < 0.02 else 'Distribution'
                    }
                    
            except Exception as e:
                continue
        
        return whale_activity
    
    def detect_breakout_candidates(self, crypto_data_dict, breakout_threshold=0.03):
        """Find cryptocurrencies near potential breakout levels"""
        breakout_candidates = {}
        
        for symbol, data in crypto_data_dict.items():
            if data is None or len(data) < 20:
                continue
            
            try:
                current_price = data['Close'].iloc[-1]
                
                # Check for resistance breakout
                if 'Resistance' in data.columns:
                    resistance = data['Resistance'].iloc[-1]
                    if not pd.isna(resistance):
                        distance_to_resistance = (resistance - current_price) / current_price
                        if 0 <= distance_to_resistance <= breakout_threshold:
                            breakout_candidates[symbol] = {
                                'data': data,
                                'type': 'resistance_breakout',
                                'level': resistance,
                                'distance': distance_to_resistance
                            }
                
                # Check for support bounce
                if 'Support' in data.columns:
                    support = data['Support'].iloc[-1]
                    if not pd.isna(support):
                        distance_to_support = (current_price - support) / current_price
                        if 0 <= distance_to_support <= breakout_threshold:
                            if symbol not in breakout_candidates:
                                breakout_candidates[symbol] = {
                                    'data': data,
                                    'type': 'support_bounce',
                                    'level': support,
                                    'distance': distance_to_support
                                }
                
                # Check for moving average breakout
                if 'MA_20' in data.columns and 'MA_50' in data.columns:
                    ma_20 = data['MA_20'].iloc[-1]
                    ma_50 = data['MA_50'].iloc[-1]
                    
                    if not pd.isna(ma_20) and not pd.isna(ma_50):
                        # MA crossover potential
                        ma_distance = abs(ma_20 - ma_50) / current_price
                        if ma_distance <= breakout_threshold:
                            if symbol not in breakout_candidates:
                                breakout_candidates[symbol] = {
                                    'data': data,
                                    'type': 'ma_crossover',
                                    'level': (ma_20 + ma_50) / 2,
                                    'distance': ma_distance
                                }
                    
            except Exception as e:
                continue
        
        return breakout_candidates
    
    def get_high_volatility_cryptos(self, crypto_data_dict, min_volatility=0.10):
        """Find high volatility cryptocurrencies for swing trading"""
        high_vol_cryptos = {}
        
        for symbol, data in crypto_data_dict.items():
            if data is None or len(data) < 20:
                continue
            
            try:
                # Calculate 24-hour volatility
                returns = data['Close'].pct_change().dropna()
                volatility = returns.tail(20).std()
                
                if volatility >= min_volatility:
                    price_range = (data['High'].tail(20).max() - data['Low'].tail(20).min()) / data['Close'].iloc[-1]
                    
                    high_vol_cryptos[symbol] = {
                        'data': data,
                        'volatility': volatility,
                        'price_range': price_range,
                        'avg_volume': data['Volume'].tail(20).mean()
                    }
                    
            except Exception as e:
                continue
        
        return high_vol_cryptos
    
    def detect_unusual_activity(self, crypto_data_dict):
        """Detect unusual price and volume activity"""
        unusual_activity = {}
        
        for symbol, data in crypto_data_dict.items():
            if data is None or len(data) < 20:
                continue
            
            try:
                # Price movement analysis
                current_price = data['Close'].iloc[-1]
                price_1h_ago = data['Close'].iloc[-12] if len(data) >= 12 else data['Close'].iloc[0]  # Assuming 5-min intervals
                price_change_1h = (current_price - price_1h_ago) / price_1h_ago
                
                # Volume analysis
                current_volume = data['Volume'].iloc[-1]
                avg_volume = data['Volume'].tail(20).mean()
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                
                # Detect unusual combinations
                unusual_score = 0
                patterns = []
                
                # Large price movement with high volume
                if abs(price_change_1h) > 0.05 and volume_ratio > 2:
                    unusual_score += 3
                    patterns.append(f"Strong {'pump' if price_change_1h > 0 else 'dump'} with volume")
                
                # Price stability with volume spike
                elif abs(price_change_1h) < 0.02 and volume_ratio > 3:
                    unusual_score += 2
                    patterns.append("Volume spike with price stability (whale activity?)")
                
                # Sudden volatility increase
                recent_volatility = data['Close'].pct_change().tail(5).std()
                historical_volatility = data['Close'].pct_change().tail(20).std()
                if recent_volatility > historical_volatility * 2:
                    unusual_score += 2
                    patterns.append("Sudden volatility spike")
                
                if unusual_score >= 2:
                    unusual_activity[symbol] = {
                        'data': data,
                        'score': unusual_score,
                        'patterns': patterns,
                        'price_change_1h': price_change_1h,
                        'volume_ratio': volume_ratio
                    }
                    
            except Exception as e:
                continue
        
        return unusual_activity
    
    def rank_cryptos_by_trading_score(self, crypto_data_dict):
        """Rank cryptocurrencies by trading attractiveness score"""
        scored_cryptos = []
        
        for symbol, data in crypto_data_dict.items():
            if data is None or len(data) < 20:
                continue
            
            try:
                score = 0
                factors = []
                
                # Volume score (0-25 points)
                if len(data) >= 10:
                    avg_volume = data['Volume'].tail(10).mean()
                    current_volume = data['Volume'].iloc[-1]
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                    volume_score = min(25, volume_ratio * 10)
                    score += volume_score
                    factors.append(f"Volume: {volume_score:.1f}")
                
                # Volatility score (0-25 points) - Higher volatility is better for crypto trading
                if len(data) >= 20:
                    returns = data['Close'].pct_change().dropna()
                    volatility = returns.tail(20).std()
                    # Optimal volatility for crypto: 5-20%
                    if 0.05 <= volatility <= 0.20:
                        volatility_score = 25
                    elif volatility > 0.20:
                        volatility_score = max(15, 25 - (volatility - 0.20) * 50)
                    else:
                        volatility_score = volatility * 500  # Scale lower volatility
                    score += min(25, volatility_score)
                    factors.append(f"Volatility: {min(25, volatility_score):.1f}")
                
                # Momentum score (0-25 points)
                if len(data) >= 5:
                    current_price = data['Close'].iloc[-1]
                    price_5_ago = data['Close'].iloc[-5]
                    momentum = abs(current_price - price_5_ago) / price_5_ago
                    momentum_score = min(25, momentum * 250)  # Higher multiplier for crypto
                    score += momentum_score
                    factors.append(f"Momentum: {momentum_score:.1f}")
                
                # Technical score (0-25 points)
                technical_score = 0
                if 'RSI' in data.columns:
                    rsi = data['RSI'].iloc[-1]
                    if not pd.isna(rsi):
                        # RSI in actionable range
                        if 30 <= rsi <= 70:
                            technical_score += 10
                        # Extreme RSI (potential reversal)
                        elif rsi < 25 or rsi > 75:
                            technical_score += 15
                
                if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                    macd = data['MACD'].iloc[-1]
                    macd_signal = data['MACD_Signal'].iloc[-1]
                    if not pd.isna(macd) and not pd.isna(macd_signal):
                        # MACD crossover
                        if abs(macd - macd_signal) < abs(macd) * 0.1:
                            technical_score += 10
                
                score += technical_score
                factors.append(f"Technical: {technical_score:.1f}")
                
                scored_cryptos.append({
                    'symbol': symbol,
                    'score': score,
                    'factors': factors,
                    'data': data
                })
                
            except Exception as e:
                continue
        
        # Sort by score
        scored_cryptos.sort(key=lambda x: x['score'], reverse=True)
        
        return scored_cryptos