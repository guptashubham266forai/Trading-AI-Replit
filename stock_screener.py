import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

class StockScreener:
    """Screen stocks based on various criteria for intraday trading"""
    
    def __init__(self):
        # NSE top liquid stocks for intraday trading
        self.liquid_stocks = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
            'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'KOTAKBANK.NS',
            'LT.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'HCLTECH.NS', 'AXISBANK.NS',
            'WIPRO.NS', 'ULTRACEMCO.NS', 'BAJFINANCE.NS', 'NESTLEIND.NS', 'ONGC.NS',
            'TATAMOTORS.NS', 'SUNPHARMA.NS', 'NTPC.NS', 'POWERGRID.NS', 'TATASTEEL.NS',
            'JSWSTEEL.NS', 'M&M.NS', 'TECHM.NS', 'TITAN.NS', 'INDUSINDBK.NS',
            'ADANIPORTS.NS', 'COALINDIA.NS', 'GRASIM.NS', 'HINDALCO.NS', 'DRREDDY.NS',
            'BAJAJFINSV.NS', 'CIPLA.NS', 'EICHERMOT.NS', 'HEROMOTOCO.NS', 'DIVISLAB.NS'
        ]
        
        # Screening criteria
        self.min_volume = 100000  # Minimum daily volume
        self.min_price = 50       # Minimum stock price
        self.max_price = 5000     # Maximum stock price
        self.min_volatility = 0.01  # Minimum 1% daily volatility
        self.max_volatility = 0.15  # Maximum 15% daily volatility
    
    def get_liquid_stocks(self):
        """Get list of liquid stocks suitable for intraday trading"""
        return self.liquid_stocks
    
    def filter_by_volume(self, stock_data_dict, min_volume_ratio=1.5):
        """Filter stocks by volume criteria"""
        filtered_stocks = {}
        
        for symbol, data in stock_data_dict.items():
            if data is None or len(data) < 10:
                continue
            
            try:
                # Calculate average volume over last 10 periods
                avg_volume = data['Volume'].tail(10).mean()
                current_volume = data['Volume'].iloc[-1]
                
                # Check if current volume is above minimum and above average
                if (current_volume >= self.min_volume and 
                    current_volume >= avg_volume * min_volume_ratio):
                    filtered_stocks[symbol] = data
                    
            except Exception as e:
                continue
        
        return filtered_stocks
    
    def filter_by_price_range(self, stock_data_dict):
        """Filter stocks by price range"""
        filtered_stocks = {}
        
        for symbol, data in stock_data_dict.items():
            if data is None or len(data) == 0:
                continue
            
            try:
                current_price = data['Close'].iloc[-1]
                
                if self.min_price <= current_price <= self.max_price:
                    filtered_stocks[symbol] = data
                    
            except Exception as e:
                continue
        
        return filtered_stocks
    
    def filter_by_volatility(self, stock_data_dict, window=20):
        """Filter stocks by volatility range"""
        filtered_stocks = {}
        
        for symbol, data in stock_data_dict.items():
            if data is None or len(data) < window:
                continue
            
            try:
                # Calculate historical volatility (standard deviation of returns)
                returns = data['Close'].pct_change().dropna()
                volatility = returns.tail(window).std()
                
                if self.min_volatility <= volatility <= self.max_volatility:
                    filtered_stocks[symbol] = data
                    
            except Exception as e:
                continue
        
        return filtered_stocks
    
    def filter_by_momentum(self, stock_data_dict, min_momentum=0.02):
        """Filter stocks showing momentum (price movement)"""
        filtered_stocks = {}
        
        for symbol, data in stock_data_dict.items():
            if data is None or len(data) < 5:
                continue
            
            try:
                # Calculate momentum over last 5 periods
                current_price = data['Close'].iloc[-1]
                price_5_periods_ago = data['Close'].iloc[-5]
                
                momentum = abs((current_price - price_5_periods_ago) / price_5_periods_ago)
                
                if momentum >= min_momentum:
                    filtered_stocks[symbol] = data
                    
            except Exception as e:
                continue
        
        return filtered_stocks
    
    def filter_by_trend_strength(self, stock_data_dict, min_trend_strength=0.7):
        """Filter stocks with strong trending behavior"""
        filtered_stocks = {}
        
        for symbol, data in stock_data_dict.items():
            if data is None or len(data) < 20:
                continue
            
            try:
                # Calculate trend strength using moving averages
                if 'MA_20' in data.columns:
                    ma_20 = data['MA_20'].tail(10)
                    
                    # Check trend consistency
                    uptrend_periods = sum(1 for i in range(1, len(ma_20)) 
                                        if ma_20.iloc[i] > ma_20.iloc[i-1])
                    downtrend_periods = sum(1 for i in range(1, len(ma_20)) 
                                          if ma_20.iloc[i] < ma_20.iloc[i-1])
                    
                    trend_strength = max(uptrend_periods, downtrend_periods) / (len(ma_20) - 1)
                    
                    if trend_strength >= min_trend_strength:
                        filtered_stocks[symbol] = data
                else:
                    # Fallback: use price trend
                    prices = data['Close'].tail(10)
                    uptrend_periods = sum(1 for i in range(1, len(prices)) 
                                        if prices.iloc[i] > prices.iloc[i-1])
                    downtrend_periods = sum(1 for i in range(1, len(prices)) 
                                          if prices.iloc[i] < prices.iloc[i-1])
                    
                    trend_strength = max(uptrend_periods, downtrend_periods) / (len(prices) - 1)
                    
                    if trend_strength >= min_trend_strength:
                        filtered_stocks[symbol] = data
                    
            except Exception as e:
                continue
        
        return filtered_stocks
    
    def filter_by_rsi_range(self, stock_data_dict, min_rsi=25, max_rsi=75):
        """Filter stocks with RSI in actionable range (not extremely overbought/oversold)"""
        filtered_stocks = {}
        
        for symbol, data in stock_data_dict.items():
            if data is None or 'RSI' not in data.columns:
                continue
            
            try:
                current_rsi = data['RSI'].iloc[-1]
                
                if not pd.isna(current_rsi) and min_rsi <= current_rsi <= max_rsi:
                    filtered_stocks[symbol] = data
                    
            except Exception as e:
                continue
        
        return filtered_stocks
    
    def get_breakout_candidates(self, stock_data_dict, breakout_threshold=0.02):
        """Find stocks near potential breakout levels"""
        breakout_candidates = {}
        
        for symbol, data in stock_data_dict.items():
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
    
    def get_oversold_stocks(self, stock_data_dict, rsi_threshold=30):
        """Find oversold stocks for potential reversal"""
        oversold_stocks = {}
        
        for symbol, data in stock_data_dict.items():
            if data is None or 'RSI' not in data.columns:
                continue
            
            try:
                current_rsi = data['RSI'].iloc[-1]
                
                if not pd.isna(current_rsi) and current_rsi <= rsi_threshold:
                    # Additional confirmation: check if RSI is turning up
                    rsi_trend = None
                    if len(data) >= 3:
                        prev_rsi = data['RSI'].iloc[-2]
                        if not pd.isna(prev_rsi) and current_rsi > prev_rsi:
                            rsi_trend = 'turning_up'
                    
                    oversold_stocks[symbol] = {
                        'data': data,
                        'rsi': current_rsi,
                        'trend': rsi_trend
                    }
                    
            except Exception as e:
                continue
        
        return oversold_stocks
    
    def get_overbought_stocks(self, stock_data_dict, rsi_threshold=70):
        """Find overbought stocks for potential reversal"""
        overbought_stocks = {}
        
        for symbol, data in stock_data_dict.items():
            if data is None or 'RSI' not in data.columns:
                continue
            
            try:
                current_rsi = data['RSI'].iloc[-1]
                
                if not pd.isna(current_rsi) and current_rsi >= rsi_threshold:
                    # Additional confirmation: check if RSI is turning down
                    rsi_trend = None
                    if len(data) >= 3:
                        prev_rsi = data['RSI'].iloc[-2]
                        if not pd.isna(prev_rsi) and current_rsi < prev_rsi:
                            rsi_trend = 'turning_down'
                    
                    overbought_stocks[symbol] = {
                        'data': data,
                        'rsi': current_rsi,
                        'trend': rsi_trend
                    }
                    
            except Exception as e:
                continue
        
        return overbought_stocks
    
    def screen_stocks(self, stock_data_dict, criteria=None):
        """Apply comprehensive screening based on multiple criteria"""
        if criteria is None:
            criteria = {
                'volume': True,
                'price_range': True,
                'volatility': True,
                'momentum': True,
                'rsi_range': True
            }
        
        filtered_stocks = stock_data_dict.copy()
        
        try:
            # Apply volume filter
            if criteria.get('volume', False):
                filtered_stocks = self.filter_by_volume(filtered_stocks)
            
            # Apply price range filter
            if criteria.get('price_range', False):
                filtered_stocks = self.filter_by_price_range(filtered_stocks)
            
            # Apply volatility filter
            if criteria.get('volatility', False):
                filtered_stocks = self.filter_by_volatility(filtered_stocks)
            
            # Apply momentum filter
            if criteria.get('momentum', False):
                filtered_stocks = self.filter_by_momentum(filtered_stocks)
            
            # Apply RSI range filter
            if criteria.get('rsi_range', False):
                filtered_stocks = self.filter_by_rsi_range(filtered_stocks)
            
            return filtered_stocks
            
        except Exception as e:
            st.warning(f"Error in stock screening: {str(e)}")
            return stock_data_dict
    
    def rank_stocks_by_score(self, stock_data_dict):
        """Rank stocks by a composite score for trading attractiveness"""
        scored_stocks = []
        
        for symbol, data in stock_data_dict.items():
            if data is None or len(data) < 20:
                continue
            
            try:
                score = 0
                factors = []
                
                # Volume score (0-20 points)
                if len(data) >= 10:
                    avg_volume = data['Volume'].tail(10).mean()
                    current_volume = data['Volume'].iloc[-1]
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                    volume_score = min(20, volume_ratio * 10)
                    score += volume_score
                    factors.append(f"Volume: {volume_score:.1f}")
                
                # Volatility score (0-20 points)
                if len(data) >= 20:
                    returns = data['Close'].pct_change().dropna()
                    volatility = returns.tail(20).std()
                    # Optimal volatility around 2-8%
                    if 0.02 <= volatility <= 0.08:
                        volatility_score = 20
                    else:
                        volatility_score = max(0, 20 - abs(volatility - 0.05) * 400)
                    score += volatility_score
                    factors.append(f"Volatility: {volatility_score:.1f}")
                
                # Momentum score (0-20 points)
                if len(data) >= 5:
                    current_price = data['Close'].iloc[-1]
                    price_5_ago = data['Close'].iloc[-5]
                    momentum = (current_price - price_5_ago) / price_5_ago
                    momentum_score = min(20, abs(momentum) * 500)
                    score += momentum_score
                    factors.append(f"Momentum: {momentum_score:.1f}")
                
                # Technical score (0-20 points)
                technical_score = 0
                if 'RSI' in data.columns:
                    rsi = data['RSI'].iloc[-1]
                    if not pd.isna(rsi):
                        # Favor RSI between 30-70 (actionable range)
                        if 30 <= rsi <= 70:
                            technical_score += 10
                        factors.append(f"RSI: {rsi:.1f}")
                
                if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                    macd = data['MACD'].iloc[-1]
                    macd_signal = data['MACD_Signal'].iloc[-1]
                    if not pd.isna(macd) and not pd.isna(macd_signal):
                        # Favor strong MACD signals
                        macd_diff = abs(macd - macd_signal)
                        technical_score += min(10, macd_diff * 1000)
                
                score += technical_score
                factors.append(f"Technical: {technical_score:.1f}")
                
                # Trend score (0-20 points)
                if 'MA_20' in data.columns and len(data) >= 10:
                    ma_20 = data['MA_20'].tail(10)
                    trend_consistency = 0
                    for i in range(1, len(ma_20)):
                        if ma_20.iloc[i] > ma_20.iloc[i-1]:
                            trend_consistency += 1
                        elif ma_20.iloc[i] < ma_20.iloc[i-1]:
                            trend_consistency -= 1
                    
                    trend_score = min(20, abs(trend_consistency) * 2.2)
                    score += trend_score
                    factors.append(f"Trend: {trend_score:.1f}")
                
                scored_stocks.append({
                    'symbol': symbol,
                    'score': score,
                    'factors': factors,
                    'data': data
                })
                
            except Exception as e:
                continue
        
        # Sort by score (highest first)
        scored_stocks.sort(key=lambda x: x['score'], reverse=True)
        
        return scored_stocks
