import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from technical_indicators import TechnicalIndicators

class SMCICTStrategies:
    """Smart Money Concepts and ICT (Inner Circle Trader) Professional Strategies"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.active_strategies = [
            "Order Block Detection", "Fair Value Gap (FVG)", "Liquidity Sweep",
            "Break of Structure (BOS)", "Change of Character (CHoCH)", 
            "Premium/Discount Arrays", "Market Maker Models", "Institutional Flow"
        ]
        self.max_risk_percent = 1.0  # Conservative risk
        self.min_risk_reward = 4.0   # ICT standard 1:4 minimum
        self.confidence_threshold = 0.90  # High confidence for ICT
    
    def add_smc_indicators(self, data):
        """Add SMC/ICT specific indicators"""
        if len(data) < 50:
            return data
        
        # Institutional candle detection
        data['Institutional_Candle'] = self.detect_institutional_candles(data)
        
        # Order blocks (supply/demand zones)
        data['Order_Block_Bull'] = self.detect_bullish_order_blocks(data)
        data['Order_Block_Bear'] = self.detect_bearish_order_blocks(data)
        
        # Fair Value Gaps
        data['FVG_Bull'] = self.detect_bullish_fvg(data)
        data['FVG_Bear'] = self.detect_bearish_fvg(data)
        
        # Liquidity levels (swing highs/lows)
        data['Liquidity_High'] = self.detect_liquidity_highs(data)
        data['Liquidity_Low'] = self.detect_liquidity_lows(data)
        
        # Premium/Discount zones (using 50% of range)
        data['Premium_Zone'] = self.calculate_premium_zone(data)
        data['Discount_Zone'] = self.calculate_discount_zone(data)
        
        # Break of Structure
        data['BOS_Bull'] = self.detect_bullish_bos(data)
        data['BOS_Bear'] = self.detect_bearish_bos(data)
        
        # Change of Character
        data['CHoCH_Bull'] = self.detect_bullish_choch(data)
        data['CHoCH_Bear'] = self.detect_bearish_choch(data)
        
        return data
    
    def detect_institutional_candles(self, data, threshold=0.7):
        """Detect institutional candles (high volume, large body)"""
        # Calculate candle body size
        body_size = abs(data['Close'] - data['Open'])
        range_size = data['High'] - data['Low']
        
        # Body should be at least 70% of the range
        body_ratio = body_size / range_size
        
        # Volume should be above average
        volume_ma = data['Volume'].rolling(20).mean()
        volume_ratio = data['Volume'] / volume_ma
        
        institutional = (body_ratio > threshold) & (volume_ratio > 1.5)
        return institutional
    
    def detect_bullish_order_blocks(self, data, lookback=5):
        """Detect bullish order blocks (demand zones)"""
        order_blocks = pd.Series([False] * len(data), index=data.index)
        
        for i in range(lookback, len(data)):
            # Look for the last bearish candle before a strong bullish move
            current_section = data.iloc[i-lookback:i+1]
            
            # Find strong bullish candle
            bullish_candle = (current_section['Close'] > current_section['Open']) & \
                           (current_section['Institutional_Candle'])
            
            if bullish_candle.any():
                bullish_idx = bullish_candle.idxmax()
                bullish_pos = current_section.index.get_loc(bullish_idx)
                
                # Look for the last bearish candle before it
                for j in range(bullish_pos - 1, -1, -1):
                    if current_section.iloc[j]['Close'] < current_section.iloc[j]['Open']:
                        actual_idx = data.index[i - lookback + j]
                        order_blocks[actual_idx] = True
                        break
        
        return order_blocks
    
    def detect_bearish_order_blocks(self, data, lookback=5):
        """Detect bearish order blocks (supply zones)"""
        order_blocks = pd.Series([False] * len(data), index=data.index)
        
        for i in range(lookback, len(data)):
            current_section = data.iloc[i-lookback:i+1]
            
            # Find strong bearish candle
            bearish_candle = (current_section['Close'] < current_section['Open']) & \
                           (current_section['Institutional_Candle'])
            
            if bearish_candle.any():
                bearish_idx = bearish_candle.idxmax()
                bearish_pos = current_section.index.get_loc(bearish_idx)
                
                # Look for the last bullish candle before it
                for j in range(bearish_pos - 1, -1, -1):
                    if current_section.iloc[j]['Close'] > current_section.iloc[j]['Open']:
                        actual_idx = data.index[i - lookback + j]
                        order_blocks[actual_idx] = True
                        break
        
        return order_blocks
    
    def detect_bullish_fvg(self, data):
        """Detect bullish Fair Value Gaps"""
        fvg = pd.Series([False] * len(data), index=data.index)
        
        for i in range(2, len(data)):
            # Three candle pattern: candle 1 high < candle 3 low
            candle1_high = data['High'].iloc[i-2]
            candle3_low = data['Low'].iloc[i]
            
            # Gap exists if there's no overlap
            if candle1_high < candle3_low:
                # Middle candle should be bullish and strong
                middle_candle = data.iloc[i-1]
                if (middle_candle['Close'] > middle_candle['Open'] and 
                    middle_candle.get('Institutional_Candle', False)):
                    fvg.iloc[i] = True
        
        return fvg
    
    def detect_bearish_fvg(self, data):
        """Detect bearish Fair Value Gaps"""
        fvg = pd.Series([False] * len(data), index=data.index)
        
        for i in range(2, len(data)):
            # Three candle pattern: candle 1 low > candle 3 high
            candle1_low = data['Low'].iloc[i-2]
            candle3_high = data['High'].iloc[i]
            
            # Gap exists if there's no overlap
            if candle1_low > candle3_high:
                # Middle candle should be bearish and strong
                middle_candle = data.iloc[i-1]
                if (middle_candle['Close'] < middle_candle['Open'] and 
                    middle_candle.get('Institutional_Candle', False)):
                    fvg.iloc[i] = True
        
        return fvg
    
    def detect_liquidity_highs(self, data, period=10):
        """Detect swing highs (liquidity above)"""
        highs = data['High']
        liquidity_highs = pd.Series([False] * len(data), index=data.index)
        
        for i in range(period, len(data) - period):
            current_high = highs.iloc[i]
            left_highs = highs.iloc[i-period:i]
            right_highs = highs.iloc[i+1:i+period+1]
            
            if (current_high > left_highs.max() and 
                current_high > right_highs.max()):
                liquidity_highs.iloc[i] = True
        
        return liquidity_highs
    
    def detect_liquidity_lows(self, data, period=10):
        """Detect swing lows (liquidity below)"""
        lows = data['Low']
        liquidity_lows = pd.Series([False] * len(data), index=data.index)
        
        for i in range(period, len(data) - period):
            current_low = lows.iloc[i]
            left_lows = lows.iloc[i-period:i]
            right_lows = lows.iloc[i+1:i+period+1]
            
            if (current_low < left_lows.min() and 
                current_low < right_lows.min()):
                liquidity_lows.iloc[i] = True
        
        return liquidity_lows
    
    def calculate_premium_zone(self, data, lookback=20):
        """Calculate premium zone (upper 50% of range)"""
        high_ma = data['High'].rolling(lookback).max()
        low_ma = data['Low'].rolling(lookback).min()
        midpoint = (high_ma + low_ma) / 2
        
        return data['Close'] > midpoint
    
    def calculate_discount_zone(self, data, lookback=20):
        """Calculate discount zone (lower 50% of range)"""
        high_ma = data['High'].rolling(lookback).max()
        low_ma = data['Low'].rolling(lookback).min()
        midpoint = (high_ma + low_ma) / 2
        
        return data['Close'] < midpoint
    
    def detect_bullish_bos(self, data, lookback=15):
        """Detect bullish Break of Structure"""
        bos = pd.Series([False] * len(data), index=data.index)
        
        for i in range(lookback, len(data)):
            current_high = data['High'].iloc[i]
            prev_highs = data['High'].iloc[i-lookback:i]
            
            # Break of structure: current high breaks previous significant high
            if current_high > prev_highs.max() * 1.002:  # 0.2% buffer
                # Confirm with volume
                if data['Volume'].iloc[i] > data['Volume'].iloc[i-lookback:i].mean() * 1.5:
                    bos.iloc[i] = True
        
        return bos
    
    def detect_bearish_bos(self, data, lookback=15):
        """Detect bearish Break of Structure"""
        bos = pd.Series([False] * len(data), index=data.index)
        
        for i in range(lookback, len(data)):
            current_low = data['Low'].iloc[i]
            prev_lows = data['Low'].iloc[i-lookback:i]
            
            # Break of structure: current low breaks previous significant low
            if current_low < prev_lows.min() * 0.998:  # 0.2% buffer
                # Confirm with volume
                if data['Volume'].iloc[i] > data['Volume'].iloc[i-lookback:i].mean() * 1.5:
                    bos.iloc[i] = True
        
        return bos
    
    def detect_bullish_choch(self, data, lookback=10):
        """Detect bullish Change of Character"""
        choch = pd.Series([False] * len(data), index=data.index)
        
        for i in range(lookback * 2, len(data)):
            # Look for change from bearish to bullish structure
            recent_lows = data['Low'].iloc[i-lookback:i]
            earlier_lows = data['Low'].iloc[i-lookback*2:i-lookback]
            
            # Recent lows should be higher than earlier lows
            if recent_lows.min() > earlier_lows.min():
                # Confirm with price breaking above previous high
                if data['High'].iloc[i] > data['High'].iloc[i-lookback:i].max():
                    choch.iloc[i] = True
        
        return choch
    
    def detect_bearish_choch(self, data, lookback=10):
        """Detect bearish Change of Character"""
        choch = pd.Series([False] * len(data), index=data.index)
        
        for i in range(lookback * 2, len(data)):
            # Look for change from bullish to bearish structure
            recent_highs = data['High'].iloc[i-lookback:i]
            earlier_highs = data['High'].iloc[i-lookback*2:i-lookback]
            
            # Recent highs should be lower than earlier highs
            if recent_highs.max() < earlier_highs.max():
                # Confirm with price breaking below previous low
                if data['Low'].iloc[i] < data['Low'].iloc[i-lookback:i].min():
                    choch.iloc[i] = True
        
        return choch
    
    def order_block_strategy(self, data, symbol):
        """ICT Order Block trading strategy"""
        signals = []
        
        if len(data) < 100:
            return signals
        
        for i in range(50, len(data)):
            current_price = data['Close'].iloc[i]
            
            # Bullish order block retest
            if data['Order_Block_Bull'].iloc[i]:
                # Price should be in discount zone
                if data['Discount_Zone'].iloc[i]:
                    confidence = self.calculate_order_block_confidence(data, i, 'bull')
                    
                    if confidence >= self.confidence_threshold:
                        order_block_low = data['Low'].iloc[i]
                        stop_loss = order_block_low * 0.995
                        target = current_price + (current_price - stop_loss) * 4
                        risk_reward = (target - current_price) / (current_price - stop_loss)
                        
                        signals.append({
                            'symbol': symbol,
                            'action': 'BUY',
                            'price': current_price,
                            'strategy': 'ICT Order Block (Demand)',
                            'confidence': confidence,
                            'stop_loss': stop_loss,
                            'target': target,
                            'risk_reward': risk_reward,
                            'timestamp': data.index[i],
                            'notes': f'Bullish order block retest in discount zone'
                        })
            
            # Bearish order block retest
            elif data['Order_Block_Bear'].iloc[i]:
                # Price should be in premium zone
                if data['Premium_Zone'].iloc[i]:
                    confidence = self.calculate_order_block_confidence(data, i, 'bear')
                    
                    if confidence >= self.confidence_threshold:
                        order_block_high = data['High'].iloc[i]
                        stop_loss = order_block_high * 1.005
                        target = current_price - (stop_loss - current_price) * 4
                        risk_reward = (current_price - target) / (stop_loss - current_price)
                        
                        signals.append({
                            'symbol': symbol,
                            'action': 'SELL',
                            'price': current_price,
                            'strategy': 'ICT Order Block (Supply)',
                            'confidence': confidence,
                            'stop_loss': stop_loss,
                            'target': target,
                            'risk_reward': risk_reward,
                            'timestamp': data.index[i],
                            'notes': f'Bearish order block retest in premium zone'
                        })
        
        return signals
    
    def fair_value_gap_strategy(self, data, symbol):
        """ICT Fair Value Gap trading strategy"""
        signals = []
        
        if len(data) < 50:
            return signals
        
        for i in range(20, len(data)):
            current_price = data['Close'].iloc[i]
            
            # Bullish FVG retest
            if data['FVG_Bull'].iloc[i-3:i].any():  # FVG within last 3 candles
                confidence = self.calculate_fvg_confidence(data, i, 'bull')
                
                if confidence >= self.confidence_threshold:
                    # Entry at current price, stop below FVG
                    fvg_low = data['Low'].iloc[i-3:i].min()
                    stop_loss = fvg_low * 0.998
                    target = current_price + (current_price - stop_loss) * 5
                    risk_reward = (target - current_price) / (current_price - stop_loss)
                    
                    signals.append({
                        'symbol': symbol,
                        'action': 'BUY',
                        'price': current_price,
                        'strategy': 'ICT Fair Value Gap',
                        'confidence': confidence,
                        'stop_loss': stop_loss,
                        'target': target,
                        'risk_reward': risk_reward,
                        'timestamp': data.index[i],
                        'notes': f'Bullish FVG retest entry'
                    })
            
            # Bearish FVG retest
            elif data['FVG_Bear'].iloc[i-3:i].any():
                confidence = self.calculate_fvg_confidence(data, i, 'bear')
                
                if confidence >= self.confidence_threshold:
                    # Entry at current price, stop above FVG
                    fvg_high = data['High'].iloc[i-3:i].max()
                    stop_loss = fvg_high * 1.002
                    target = current_price - (stop_loss - current_price) * 5
                    risk_reward = (current_price - target) / (stop_loss - current_price)
                    
                    signals.append({
                        'symbol': symbol,
                        'action': 'SELL',
                        'price': current_price,
                        'strategy': 'ICT Fair Value Gap',
                        'confidence': confidence,
                        'stop_loss': stop_loss,
                        'target': target,
                        'risk_reward': risk_reward,
                        'timestamp': data.index[i],
                        'notes': f'Bearish FVG retest entry'
                    })
        
        return signals
    
    def liquidity_sweep_strategy(self, data, symbol):
        """ICT Liquidity Sweep trading strategy"""
        signals = []
        
        if len(data) < 100:
            return signals
        
        for i in range(50, len(data)):
            current_price = data['Close'].iloc[i]
            
            # Look for liquidity sweep (stop hunt) followed by reversal
            recent_liquidity_high = data['Liquidity_High'].iloc[i-20:i].any()
            recent_liquidity_low = data['Liquidity_Low'].iloc[i-20:i].any()
            
            # Bullish liquidity sweep (sweep lows then reverse up)
            if recent_liquidity_low:
                # Find the liquidity low
                liquidity_idx = data['Liquidity_Low'].iloc[i-20:i].idxmax()
                liquidity_low = data.loc[liquidity_idx, 'Low']
                
                # Check if price swept below then reversed
                if (data['Low'].iloc[i-5:i].min() < liquidity_low and 
                    current_price > liquidity_low * 1.002):
                    
                    confidence = self.calculate_liquidity_confidence(data, i, 'bull')
                    
                    if confidence >= self.confidence_threshold:
                        stop_loss = data['Low'].iloc[i-5:i].min() * 0.998
                        target = current_price + (current_price - stop_loss) * 6
                        risk_reward = (target - current_price) / (current_price - stop_loss)
                        
                        signals.append({
                            'symbol': symbol,
                            'action': 'BUY',
                            'price': current_price,
                            'strategy': 'ICT Liquidity Sweep',
                            'confidence': confidence,
                            'stop_loss': stop_loss,
                            'target': target,
                            'risk_reward': risk_reward,
                            'timestamp': data.index[i],
                            'notes': f'Bullish liquidity sweep reversal'
                        })
            
            # Bearish liquidity sweep (sweep highs then reverse down)
            elif recent_liquidity_high:
                # Find the liquidity high
                liquidity_idx = data['Liquidity_High'].iloc[i-20:i].idxmax()
                liquidity_high = data.loc[liquidity_idx, 'High']
                
                # Check if price swept above then reversed
                if (data['High'].iloc[i-5:i].max() > liquidity_high and 
                    current_price < liquidity_high * 0.998):
                    
                    confidence = self.calculate_liquidity_confidence(data, i, 'bear')
                    
                    if confidence >= self.confidence_threshold:
                        stop_loss = data['High'].iloc[i-5:i].max() * 1.002
                        target = current_price - (stop_loss - current_price) * 6
                        risk_reward = (current_price - target) / (stop_loss - current_price)
                        
                        signals.append({
                            'symbol': symbol,
                            'action': 'SELL',
                            'price': current_price,
                            'strategy': 'ICT Liquidity Sweep',
                            'confidence': confidence,
                            'stop_loss': stop_loss,
                            'target': target,
                            'risk_reward': risk_reward,
                            'timestamp': data.index[i],
                            'notes': f'Bearish liquidity sweep reversal'
                        })
        
        return signals
    
    def calculate_order_block_confidence(self, data, index, direction):
        """Calculate confidence for order block signals"""
        base_confidence = 0.85
        
        # Institutional candle confirmation
        if data['Institutional_Candle'].iloc[index]:
            base_confidence += 0.05
        
        # Premium/discount zone confirmation
        if ((direction == 'bull' and data['Discount_Zone'].iloc[index]) or 
            (direction == 'bear' and data['Premium_Zone'].iloc[index])):
            base_confidence += 0.05
        
        # Volume confirmation
        if data['Volume'].iloc[index] > data['Volume'].iloc[index-10:index].mean() * 1.3:
            base_confidence += 0.03
        
        return min(base_confidence, 0.98)
    
    def calculate_fvg_confidence(self, data, index, direction):
        """Calculate confidence for FVG signals"""
        base_confidence = 0.88
        
        # Strong institutional candle in FVG formation
        if data['Institutional_Candle'].iloc[index-1]:
            base_confidence += 0.05
        
        # Additional market structure confirmation
        if ((direction == 'bull' and data.get('BOS_Bull', pd.Series([False]*len(data))).iloc[index]) or
            (direction == 'bear' and data.get('BOS_Bear', pd.Series([False]*len(data))).iloc[index])):
            base_confidence += 0.04
        
        return min(base_confidence, 0.97)
    
    def calculate_liquidity_confidence(self, data, index, direction):
        """Calculate confidence for liquidity sweep signals"""
        base_confidence = 0.90
        
        # Strong reversal confirmation
        if direction == 'bull':
            if data['Close'].iloc[index] > data['Open'].iloc[index]:
                base_confidence += 0.03
        else:
            if data['Close'].iloc[index] < data['Open'].iloc[index]:
                base_confidence += 0.03
        
        # Volume spike during sweep
        if data['Volume'].iloc[index] > data['Volume'].iloc[index-10:index].mean() * 2:
            base_confidence += 0.04
        
        return min(base_confidence, 0.99)
    
    def generate_smc_ict_signals(self, data, symbol):
        """Generate all SMC/ICT trading signals"""
        if len(data) < 100:
            return []
        
        # Add SMC/ICT indicators
        data_with_indicators = self.add_smc_indicators(data)
        
        all_signals = []
        
        # Generate signals from different ICT strategies
        all_signals.extend(self.order_block_strategy(data_with_indicators, symbol))
        all_signals.extend(self.fair_value_gap_strategy(data_with_indicators, symbol))
        all_signals.extend(self.liquidity_sweep_strategy(data_with_indicators, symbol))
        
        # Filter by minimum risk-reward and confidence
        filtered_signals = [
            s for s in all_signals 
            if s.get('risk_reward', 0) >= self.min_risk_reward and 
               s.get('confidence', 0) >= self.confidence_threshold
        ]
        
        # Sort by confidence (highest first)
        filtered_signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return filtered_signals[:2]  # Return top 2 ICT signals