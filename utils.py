import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

def format_currency(amount, currency='₹'):
    """Format currency with appropriate decimal places"""
    if amount >= 10000000:  # 1 crore
        return f"{currency}{amount/10000000:.2f}Cr"
    elif amount >= 100000:  # 1 lakh
        return f"{currency}{amount/100000:.2f}L"
    elif amount >= 1000:
        return f"{currency}{amount/1000:.2f}K"
    else:
        return f"{currency}{amount:.2f}"

def format_percentage(value, decimal_places=2):
    """Format percentage with specified decimal places"""
    return f"{value:.{decimal_places}f}%"

def format_number(value, decimal_places=2):
    """Format number with commas and decimal places"""
    return f"{value:,.{decimal_places}f}"

def calculate_risk_reward(entry_price, stop_loss, target_price):
    """Calculate risk-reward ratio"""
    try:
        risk = abs(entry_price - stop_loss)
        reward = abs(target_price - entry_price)
        
        if risk == 0:
            return float('inf')
        
        return reward / risk
    except:
        return 0

def calculate_position_size(account_balance, risk_percent, entry_price, stop_loss):
    """Calculate position size based on risk management"""
    try:
        risk_amount = account_balance * (risk_percent / 100)
        price_diff = abs(entry_price - stop_loss)
        
        if price_diff == 0:
            return 0
        
        position_size = risk_amount / price_diff
        return int(position_size)
    except:
        return 0

def calculate_drawdown(prices):
    """Calculate maximum drawdown from price series"""
    try:
        # Calculate cumulative returns
        cumulative = (1 + prices.pct_change()).cumprod()
        
        # Calculate running maximum
        running_max = cumulative.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max
        
        return {
            'current_drawdown': drawdown.iloc[-1],
            'max_drawdown': drawdown.min()
        }
    except:
        return {'current_drawdown': 0, 'max_drawdown': 0}

def calculate_volatility(prices, window=20):
    """Calculate historical volatility"""
    try:
        returns = prices.pct_change().dropna()
        volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        return volatility.iloc[-1] if len(volatility) > 0 else 0
    except:
        return 0

def calculate_sharpe_ratio(returns, risk_free_rate=0.06):
    """Calculate Sharpe ratio"""
    try:
        excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    except:
        return 0

def is_market_hours():
    """Check if current time is within market hours"""
    now = datetime.now()
    
    # NSE trading hours: 9:15 AM to 3:30 PM IST, Monday to Friday
    if now.weekday() > 4:  # Saturday = 5, Sunday = 6
        return False
    
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    return market_open <= now <= market_close

def get_trading_session():
    """Get current trading session info"""
    now = datetime.now()
    
    if not is_market_hours():
        if now.weekday() > 4:
            return "Weekend - Market Closed"
        elif now.hour < 9 or (now.hour == 9 and now.minute < 15):
            return "Pre-Market"
        else:
            return "After-Market"
    
    # During market hours, determine session
    if now.hour == 9 and now.minute < 30:
        return "Opening Session"
    elif now.hour < 12:
        return "Morning Session"
    elif now.hour < 14:
        return "Afternoon Session"
    else:
        return "Closing Session"

def format_time_ago(timestamp):
    """Format time difference in human readable format"""
    try:
        now = datetime.now()
        if hasattr(timestamp, 'to_pydatetime'):
            timestamp = timestamp.to_pydatetime()
        
        diff = now - timestamp
        
        if diff.total_seconds() < 60:
            return f"{int(diff.total_seconds())} seconds ago"
        elif diff.total_seconds() < 3600:
            return f"{int(diff.total_seconds() / 60)} minutes ago"
        elif diff.total_seconds() < 86400:
            return f"{int(diff.total_seconds() / 3600)} hours ago"
        else:
            return f"{int(diff.total_seconds() / 86400)} days ago"
    except:
        return "Unknown"

def validate_symbol(symbol):
    """Validate NSE symbol format"""
    if not symbol:
        return False
    
    # Remove .NS if present for validation
    clean_symbol = symbol.replace('.NS', '')
    
    # Check if symbol contains only alphanumeric characters and specific allowed characters
    allowed_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789&-')
    return all(c in allowed_chars for c in clean_symbol.upper())

def calculate_support_resistance(high, low, close, window=20):
    """Calculate basic support and resistance levels"""
    try:
        # Recent high and low
        recent_high = high.tail(window).max()
        recent_low = low.tail(window).min()
        
        # Pivot points
        typical_price = (high + low + close) / 3
        pivot = typical_price.tail(window).mean()
        
        return {
            'resistance': recent_high,
            'support': recent_low,
            'pivot': pivot
        }
    except:
        return {'resistance': 0, 'support': 0, 'pivot': 0}

def calculate_atr_stop_loss(high, low, close, atr_multiplier=2.0, window=14):
    """Calculate ATR-based stop loss levels"""
    try:
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        
        current_price = close.iloc[-1]
        current_atr = atr.iloc[-1]
        
        return {
            'buy_stop_loss': current_price - (current_atr * atr_multiplier),
            'sell_stop_loss': current_price + (current_atr * atr_multiplier),
            'atr': current_atr
        }
    except:
        return {'buy_stop_loss': 0, 'sell_stop_loss': 0, 'atr': 0}

def create_signal_summary(signals):
    """Create a summary of trading signals"""
    if not signals:
        return {
            'total_signals': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'avg_confidence': 0,
            'strategies_used': []
        }
    
    buy_signals = sum(1 for s in signals if s['action'] == 'BUY')
    sell_signals = sum(1 for s in signals if s['action'] == 'SELL')
    avg_confidence = sum(s['confidence'] for s in signals) / len(signals)
    strategies_used = list(set(s['strategy'] for s in signals))
    
    return {
        'total_signals': len(signals),
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'avg_confidence': avg_confidence,
        'strategies_used': strategies_used
    }

def export_signals_to_csv(signals):
    """Export signals to CSV format"""
    if not signals:
        return None
    
    try:
        df = pd.DataFrame(signals)
        
        # Format timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Round numerical values
        numerical_cols = ['price', 'confidence', 'stop_loss', 'target', 'risk_reward']
        for col in numerical_cols:
            if col in df.columns:
                df[col] = df[col].round(2)
        
        return df.to_csv(index=False)
    except:
        return None

def get_market_sentiment_color(change_percent):
    """Get color based on market sentiment"""
    if change_percent > 2:
        return "🟢"  # Strong positive
    elif change_percent > 0:
        return "🔵"  # Positive
    elif change_percent > -2:
        return "🟡"  # Neutral/Slightly negative
    else:
        return "🔴"  # Negative

def calculate_fibonacci_levels(high, low):
    """Calculate Fibonacci retracement levels"""
    try:
        diff = high - low
        
        levels = {
            'level_0': high,
            'level_23.6': high - 0.236 * diff,
            'level_38.2': high - 0.382 * diff,
            'level_50': high - 0.5 * diff,
            'level_61.8': high - 0.618 * diff,
            'level_78.6': high - 0.786 * diff,
            'level_100': low
        }
        
        return levels
    except:
        return {}

def validate_trading_signal(signal):
    """Validate trading signal data"""
    required_fields = ['symbol', 'action', 'price', 'strategy', 'confidence', 'timestamp']
    
    if not isinstance(signal, dict):
        return False
    
    for field in required_fields:
        if field not in signal:
            return False
    
    # Validate action
    if signal['action'] not in ['BUY', 'SELL']:
        return False
    
    # Validate price and confidence
    try:
        price = float(signal['price'])
        confidence = float(signal['confidence'])
        
        if price <= 0 or confidence < 0 or confidence > 1:
            return False
    except:
        return False
    
    return True
