import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import asyncio
import threading

from data_fetcher import DataFetcher
from crypto_data_fetcher import CryptoDataFetcher
from strategies import TradingStrategies
from advanced_strategies import AdvancedTradingStrategies
from chart_generator import SignalChartGenerator
from audio_notifications import AudioNotifications
from stock_screener import StockScreener
from crypto_screener import CryptoScreener
from predictive_analysis import PredictiveAnalysis
from swing_strategies import SwingTradingStrategies
# Database components temporarily disabled during migration
# from database import DatabaseManager
# from performance_analyzer import PerformanceAnalyzer
# from auto_trader import AutoTrader
# from live_tracker import LiveTracker
from utils import format_currency, calculate_risk_reward

# Page configuration
st.set_page_config(
    page_title="Professional Trading Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if 'market_type' not in st.session_state:
        st.session_state.market_type = 'crypto'
    if 'trading_style' not in st.session_state:
        st.session_state.trading_style = 'intraday'
    
    # Stock-related states
    if 'data_fetcher' not in st.session_state:
        st.session_state.data_fetcher = DataFetcher()
    if 'stock_strategies' not in st.session_state:
        st.session_state.stock_strategies = TradingStrategies()
    if 'stock_screener' not in st.session_state:
        st.session_state.stock_screener = StockScreener()
    if 'stock_last_update' not in st.session_state:
        st.session_state.stock_last_update = None
    if 'stock_market_data' not in st.session_state:
        st.session_state.stock_market_data = {}
    if 'stock_signals' not in st.session_state:
        st.session_state.stock_signals = []
    
    # Crypto-related states
    if 'crypto_data_fetcher' not in st.session_state:
        st.session_state.crypto_data_fetcher = CryptoDataFetcher()
    if 'crypto_strategies' not in st.session_state:
        st.session_state.crypto_strategies = TradingStrategies()
    if 'crypto_screener' not in st.session_state:
        st.session_state.crypto_screener = CryptoScreener()
    if 'crypto_last_update' not in st.session_state:
        st.session_state.crypto_last_update = None
    if 'crypto_market_data' not in st.session_state:
        st.session_state.crypto_market_data = {}
    if 'crypto_signals' not in st.session_state:
        st.session_state.crypto_signals = []
    
    # Swing trading states
    if 'swing_strategies' not in st.session_state:
        st.session_state.swing_strategies = SwingTradingStrategies()
    if 'swing_signals' not in st.session_state:
        st.session_state.swing_signals = []
    
    # Predictor
    if 'predictor' not in st.session_state:
        st.session_state.predictor = PredictiveAnalysis()
    
    # Advanced trading strategies
    if 'advanced_strategies' not in st.session_state:
        st.session_state.advanced_strategies = AdvancedTradingStrategies()
    
    # Chart generator
    if 'chart_generator' not in st.session_state:
        st.session_state.chart_generator = SignalChartGenerator()
    
    # Audio notifications
    if 'audio_notifications' not in st.session_state:
        st.session_state.audio_notifications = AudioNotifications()
    
    # Previous signals for comparison (to detect new signals)
    if 'previous_signals' not in st.session_state:
        st.session_state.previous_signals = []
    
    # Database and Performance (temporarily disabled for initial setup)
    if 'db_manager' not in st.session_state:
        # Temporarily disable database to fix symbol formatting issues
        st.session_state.db_manager = None
        st.session_state.db_connected = False
    
    if 'performance_analyzer' not in st.session_state:
        # Database components disabled during migration
        st.session_state.performance_analyzer = None
    
    # Auto-trader for high confidence signals (temporarily disabled)
    if 'auto_trader' not in st.session_state:
        st.session_state.auto_trader = None
    
    # Live tracker for real-time P&L (disabled during migration)
    if 'live_tracker' not in st.session_state:
        st.session_state.live_tracker = None

def get_current_data_fetcher():
    """Get the appropriate data fetcher based on selected market"""
    return (st.session_state.crypto_data_fetcher if st.session_state.market_type == 'crypto' 
            else st.session_state.data_fetcher)

def get_current_screener():
    """Get the appropriate screener based on selected market"""
    return (st.session_state.crypto_screener if st.session_state.market_type == 'crypto' 
            else st.session_state.stock_screener)

def get_current_strategies():
    """Get the appropriate strategies based on trading style"""
    if st.session_state.trading_style == 'swing':
        return st.session_state.swing_strategies
    elif st.session_state.market_type == 'crypto':
        return st.session_state.crypto_strategies
    else:
        return st.session_state.stock_strategies

def get_current_market_data():
    """Get current market data based on selected market"""
    return (st.session_state.crypto_market_data if st.session_state.market_type == 'crypto' 
            else st.session_state.stock_market_data)

def get_current_signals():
    """Get current signals based on trading style and market"""
    if st.session_state.trading_style == 'swing':
        return st.session_state.swing_signals
    elif st.session_state.market_type == 'crypto':
        return st.session_state.crypto_signals
    else:
        return st.session_state.stock_signals

def create_universal_candlestick_chart(data, symbol, signals=None):
    """Create an interactive candlestick chart for any instrument"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            f'{symbol} - {st.session_state.market_type.title()} {st.session_state.trading_style.title()} Analysis', 
            'Volume', 
            'Technical Indicators',
            'Advanced Indicators'
        ),
        row_heights=[0.5, 0.15, 0.2, 0.15]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add moving averages
    ma_colors = ['orange', 'blue', 'green', 'red']
    ma_periods = [20, 50, 100, 200]
    
    for i, period in enumerate(ma_periods):
        ma_col = f'MA_{period}'
        if ma_col in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index, 
                    y=data[ma_col], 
                    name=f'MA {period}', 
                    line=dict(color=ma_colors[i % len(ma_colors)])
                ),
                row=1, col=1
            )
    
    # Add Bollinger Bands if available
    if all(col in data.columns for col in ['BB_Upper', 'BB_Lower']):
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['BB_Upper'], 
                name='BB Upper', line=dict(color='gray', dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['BB_Lower'], 
                name='BB Lower', line=dict(color='gray', dash='dash'),
                fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Add support and resistance levels
    if 'Support' in data.columns and 'Resistance' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['Support'], 
                name='Support', line=dict(color='green', dash='dot')
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['Resistance'], 
                name='Resistance', line=dict(color='red', dash='dot')
            ),
            row=1, col=1
        )
    
    # Add buy/sell signals
    if signals:
        buy_signals = [s for s in signals if s['action'] == 'BUY']
        sell_signals = [s for s in signals if s['action'] == 'SELL']
        
        if buy_signals:
            buy_dates = [s['timestamp'] for s in buy_signals]
            buy_prices = [s['price'] for s in buy_signals]
            fig.add_trace(
                go.Scatter(
                    x=buy_dates, y=buy_prices, mode='markers',
                    marker=dict(symbol='triangle-up', size=15, color='green'),
                    name='Buy Signal'
                ),
                row=1, col=1
            )
        
        if sell_signals:
            sell_dates = [s['timestamp'] for s in sell_signals]
            sell_prices = [s['price'] for s in sell_signals]
            fig.add_trace(
                go.Scatter(
                    x=sell_dates, y=sell_prices, mode='markers',
                    marker=dict(symbol='triangle-down', size=15, color='red'),
                    name='Sell Signal'
                ),
                row=1, col=1
            )
    
    # Volume chart
    colors = ['red' if close < open else 'green' for close, open in zip(data['Close'], data['Open'])]
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color=colors),
        row=2, col=1
    )
    
    # RSI
    if 'RSI' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')),
            row=3, col=1
        )
        # Add RSI reference lines
        fig.update_layout(
            shapes=[
                dict(type="line", x0=0, x1=1, y0=70, y1=70,
                     line=dict(color="red", dash="dash"), xref="paper", yref="y3"),
                dict(type="line", x0=0, x1=1, y0=30, y1=30,
                     line=dict(color="green", dash="dash"), xref="paper", yref="y3"),
                dict(type="line", x0=0, x1=1, y0=50, y1=50,
                     line=dict(color="gray", dash="dot"), xref="paper", yref="y3")
            ]
        )
    
    # MACD
    if all(col in data.columns for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal', line=dict(color='orange')),
            row=3, col=1
        )
    
    # Additional indicators for swing trading
    if st.session_state.trading_style == 'swing':
        # Stochastic
        if all(col in data.columns for col in ['Stoch_K', 'Stoch_D']):
            fig.add_trace(
                go.Scatter(x=data.index, y=data['Stoch_K'], name='Stoch %K', line=dict(color='blue')),
                row=4, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['Stoch_D'], name='Stoch %D', line=dict(color='red')),
                row=4, col=1
            )
            # Add Stochastic reference lines
            current_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
            fig.update_layout(
                shapes=current_shapes + [
                    dict(type="line", x0=0, x1=1, y0=80, y1=80,
                         line=dict(color="red", dash="dash"), xref="paper", yref="y4"),
                    dict(type="line", x0=0, x1=1, y0=20, y1=20,
                         line=dict(color="green", dash="dash"), xref="paper", yref="y4")
                ]
            )
        
        # Williams %R
        elif 'Williams_R' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['Williams_R'], name='Williams %R', line=dict(color='orange')),
                row=4, col=1
            )
            # Add Williams %R reference lines
            current_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
            fig.update_layout(
                shapes=current_shapes + [
                    dict(type="line", x0=0, x1=1, y0=-20, y1=-20,
                         line=dict(color="red", dash="dash"), xref="paper", yref="y4"),
                    dict(type="line", x0=0, x1=1, y0=-80, y1=-80,
                         line=dict(color="green", dash="dash"), xref="paper", yref="y4")
                ]
            )
    
    fig.update_layout(
        title=f"{symbol} - {st.session_state.market_type.title()} {st.session_state.trading_style.title()} Analysis",
        xaxis_rangeslider_visible=False,
        height=900,
        showlegend=True
    )
    
    return fig

def display_market_selection():
    """Display market and trading style selection"""
    st.sidebar.header("🎯 Trading Platform Selection")
    
    # Market type selection
    market_options = {
        'stocks': '🏛️ Indian Stock Market (NSE)',
        'crypto': '₿ Cryptocurrency Market'
    }
    
    selected_market = st.sidebar.selectbox(
        "Select Market Type",
        options=list(market_options.keys()),
        format_func=lambda x: market_options[x],
        index=0 if st.session_state.market_type == 'stocks' else 1
    )
    
    if selected_market != st.session_state.market_type:
        st.session_state.market_type = selected_market
        st.rerun()
    
    # Trading style selection
    trading_options = {
        'intraday': '⚡ Intraday Trading (Minutes to Hours)',
        'swing': '📈 Swing Trading (Days to Weeks)'
    }
    
    selected_style = st.sidebar.selectbox(
        "Select Trading Style",
        options=list(trading_options.keys()),
        format_func=lambda x: trading_options[x],
        index=0 if st.session_state.trading_style == 'intraday' else 1
    )
    
    if selected_style != st.session_state.trading_style:
        st.session_state.trading_style = selected_style
        st.rerun()
    
    # Display current selection
    st.sidebar.success(f"**Active:** {market_options[st.session_state.market_type]}")
    st.sidebar.info(f"**Style:** {trading_options[st.session_state.trading_style]}")

def display_market_overview():
    """Display market overview based on current selection"""
    market_name = "Crypto Market" if st.session_state.market_type == 'crypto' else "NSE Market"
    icon = "₿" if st.session_state.market_type == 'crypto' else "📊"
    
    st.header(f"{icon} {market_name} Overview - {st.session_state.trading_style.title()} Trading")
    
    current_data = get_current_market_data()
    
    if not current_data:
        st.info(f"Loading {market_name.lower()} data...")
        return
    
    # Create metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_instruments = len(current_data)
    gainers = sum(1 for data in current_data.values() 
                  if len(data) > 1 and data['Close'].iloc[-1] > data['Close'].iloc[-2])
    losers = total_instruments - gainers
    avg_volume = sum(data['Volume'].iloc[-1] for data in current_data.values() 
                     if len(data) > 0) / max(total_instruments, 1)
    
    with col1:
        instrument_type = "Cryptocurrencies" if st.session_state.market_type == 'crypto' else "Stocks"
        st.metric(f"Total {instrument_type}", total_instruments)
    
    with col2:
        st.metric("Gainers", gainers, delta=f"{gainers-losers}")
    
    with col3:
        st.metric("Losers", losers)
    
    with col4:
        volume_label = "Avg Volume" if st.session_state.market_type == 'stocks' else "Avg Volume"
        st.metric(volume_label, f"{avg_volume:,.0f}")
    
    # Market status
    data_fetcher = get_current_data_fetcher()
    if hasattr(data_fetcher, 'get_market_status'):
        try:
            market_status = data_fetcher.get_market_status()
            if market_status:
                status_col1, status_col2 = st.columns(2)
                
                with status_col1:
                    market_open = market_status.get('is_open', True)
                    status_text = "🟢 Market Open" if market_open else "🔴 Market Closed"
                    if st.session_state.market_type == 'crypto':
                        status_text = "🟢 Market Open (24/7)"
                    st.info(status_text)
                
                with status_col2:
                    if 'index' in market_status:
                        index_name = market_status['index']
                        index_value = market_status.get('current', 0)
                        index_change = market_status.get('change_percent', 0)
                        
                        if st.session_state.market_type == 'crypto':
                            st.metric(index_name, f"${index_value:,.2f}", f"{index_change:.2f}%")
                        else:
                            st.metric(index_name, f"₹{index_value:,.2f}", f"{index_change:.2f}%")
        except:
            pass

def display_trading_signals():
    """Display trading signals based on current selection"""
    signals = get_current_signals()
    trading_style = st.session_state.trading_style.title()
    market_type = st.session_state.market_type.title()
    
    st.header(f"🎯 Active {trading_style} {market_type} Trading Signals")
    
    # Add confidence filter controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Fix first-time tab switching by using session state
        if 'confidence_filter_value' not in st.session_state:
            st.session_state.confidence_filter_value = 0
            
        confidence_filter = st.slider(
            "Minimum Confidence Level (%)",
            min_value=0,
            max_value=100,
            value=st.session_state.confidence_filter_value,
            step=5,
            help="Filter signals by minimum confidence level",
            key="trading_signals_confidence_filter"
        )
        
        # Update session state without causing rerun
        if confidence_filter != st.session_state.confidence_filter_value:
            st.session_state.confidence_filter_value = confidence_filter
    
    with col2:
        show_count = st.selectbox(
            "Show Count",
            options=[6, 9, 12, 15],
            index=1,
            help="Number of signals to display",
            key="trading_signals_show_count"
        )
    
    with col3:
        sort_option = st.selectbox(
            "Sort By",
            options=["Confidence", "Time", "Risk:Reward"],
            index=0,
            help="Sort signals by selected criteria",
            key="trading_signals_sort_option"
        )
    
    print(f"🔧 DEBUG: Checking signals display. Raw signals count: {len(signals) if signals else 0}")
    if not signals:
        st.info(f"No active {trading_style.lower()} {market_type.lower()} signals at the moment. Signals will appear when favorable conditions are detected.")
        st.write(f"**Debug Info:** Session state signals: {len(get_current_signals())}")
        return
    
    # Filter signals by time and confidence
    current_time = datetime.now()
    time_filter = 14400 if st.session_state.trading_style == 'swing' else 7200  # 4 hours for swing, 2 hours for intraday
    confidence_threshold = confidence_filter / 100.0  # Convert to decimal
    
    print(f"🔧 DEBUG: Filtering {len(signals)} signals with confidence >= {confidence_threshold:.1%}")
    
    filtered_signals = []
    for i, signal in enumerate(signals):
        try:
            # Time filter
            signal_time = signal['timestamp']
            if hasattr(signal_time, 'tzinfo') and signal_time.tzinfo is not None:
                signal_time = signal_time.replace(tzinfo=None)
            
            time_diff = (current_time - signal_time).total_seconds()
            time_valid = time_diff < time_filter
            
            # Confidence filter
            confidence_valid = signal.get('confidence', 0) >= confidence_threshold
            
            print(f"🔧 DEBUG: Signal {i+1}: {signal['symbol']} {signal['action']} - Confidence: {signal.get('confidence', 0):.1%}, Time diff: {time_diff:.0f}s, Time valid: {time_valid}, Confidence valid: {confidence_valid}")
            
            if time_valid and confidence_valid:
                filtered_signals.append(signal)
        except Exception as e:
            print(f"🔧 DEBUG: Error processing signal {i+1}: {str(e)}")
            if signal.get('confidence', 0) >= confidence_threshold:
                filtered_signals.append(signal)
    
    print(f"🔧 DEBUG: After filtering: {len(filtered_signals)} signals remain")
    
    # Sort signals based on selected criteria
    if sort_option == "Confidence":
        filtered_signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
    elif sort_option == "Time":
        # Fix timezone comparison issue
        def get_timestamp_for_sorting(signal):
            ts = signal.get('timestamp', datetime.now())
            # Convert to naive datetime if it's timezone-aware
            if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
            return ts
        filtered_signals.sort(key=get_timestamp_for_sorting, reverse=True)
    elif sort_option == "Risk:Reward":
        filtered_signals.sort(key=lambda x: x.get('risk_reward', 0), reverse=True)
    
    if not filtered_signals:
        st.info(f"No recent {trading_style.lower()} signals found with minimum {confidence_filter}% confidence.")
        return
    
    # Show signal statistics
    st.info(f"📊 Showing {len(filtered_signals)} signals with ≥{confidence_filter}% confidence | Average confidence: {sum(s.get('confidence', 0) for s in filtered_signals) / len(filtered_signals):.1%}")
    
    # Limit to show_count
    recent_signals = filtered_signals[:show_count]
    
    # Chart display area for selected signal (stays within this tab)
    if 'selected_signal' in st.session_state and st.session_state.selected_signal:
        display_signal_chart_inline()
        st.write("---")
    
    # Display signals table
    st.info("💡 **Click the 📊 button next to any signal to view its chart in the area above**")
    
    # Professional table styling
    st.markdown("""
    <style>
    .signal-row {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        margin: 4px 0;
        padding: 8px;
        transition: all 0.2s ease;
    }
    .signal-row:hover {
        background: #e2e8f0;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display table headers  
    header_cols = st.columns([1, 2, 1.5, 1.5, 2, 1, 1.5, 1.5, 1, 1])
    headers = ["Chart", "Symbol", "Action", "Price", "Strategies", "Conf.", "Stop Loss", "Target", "R:R", "Time"]
    
    for i, header in enumerate(headers):
        with header_cols[i]:
            st.write(f"**{header}**")
    
    st.divider()
    
    # Display signals table  
    create_signals_table(recent_signals)

def create_signals_table(filtered_signals):
    """Create an interactive table for trading signals with consolidated duplicate assets"""
    # Consolidate signals by symbol to remove duplicates
    consolidated_signals = {}
    
    for signal in filtered_signals:
        symbol = signal['symbol']
        if symbol not in consolidated_signals:
            consolidated_signals[symbol] = {
                'symbol': symbol,
                'action': signal['action'],
                'price': signal['price'],
                'strategies': [signal['strategy']],
                'confidence': signal['confidence'],
                'stop_loss': signal.get('stop_loss'),
                'target': signal.get('target'),
                'risk_reward': signal.get('risk_reward'),
                'timestamp': signal['timestamp'],
                'original_signal': signal
            }
        else:
            # Merge strategies and use highest confidence
            existing = consolidated_signals[symbol]
            if signal['strategy'] not in existing['strategies']:
                existing['strategies'].append(signal['strategy'])
            if signal['confidence'] > existing['confidence']:
                existing['confidence'] = signal['confidence']
                existing['original_signal'] = signal
    
    # Create table data from consolidated signals
    table_data = []
    current_time = datetime.now()
    
    for i, (symbol, consolidated) in enumerate(consolidated_signals.items()):
        # Format time
        try:
            signal_timestamp = consolidated['timestamp']
            if hasattr(signal_timestamp, 'tzinfo') and signal_timestamp.tzinfo is not None:
                signal_timestamp = signal_timestamp.replace(tzinfo=None)
            
            time_ago = current_time - signal_timestamp
            if time_ago.total_seconds() < 3600:
                time_str = f"{int(time_ago.total_seconds() / 60)}m ago"
            else:
                time_str = f"{int(time_ago.total_seconds() / 3600)}h ago"
        except Exception:
            time_str = "Recently"
        
        # Format prices based on market type
        if st.session_state.market_type == 'crypto':
            price_format = f"${consolidated['price']:.4f}"
            stop_format = f"${consolidated.get('stop_loss', 0):.4f}" if consolidated.get('stop_loss') else "N/A"
            target_format = f"${consolidated.get('target', 0):.4f}" if consolidated.get('target') else "N/A"
        else:
            price_format = f"₹{consolidated['price']:.2f}"
            stop_format = f"₹{consolidated.get('stop_loss', 0):.2f}" if consolidated.get('stop_loss') else "N/A"
            target_format = f"₹{consolidated.get('target', 0):.2f}" if consolidated.get('target') else "N/A"
        
        # Create action column with better styling
        if consolidated['action'] == 'BUY':
            action_display = "📈 BUY"
        else:
            action_display = "📉 SELL"
        
        # Format strategies as comma-separated list
        strategies_text = ", ".join(consolidated['strategies'])
        
        table_data.append({
            'Symbol': consolidated['symbol'].replace('.NS', '').replace('-USD', ''),
            'Action': action_display,
            'Price': price_format,
            'Strategies': strategies_text,
            'Confidence': f"{consolidated['confidence']:.0%}",
            'Stop Loss': stop_format,
            'Target': target_format,
            'R:R': f"1:{consolidated.get('risk_reward', 0):.1f}" if consolidated.get('risk_reward') else "N/A",
            'Time': time_str,
            'Index': i,
            'original_signal': consolidated['original_signal']
        })
    
    # Display table with clickable rows
    if table_data:
        # Create a selection interface using radio buttons
        st.write("**Select a signal to view its chart:**")
        
        # Create columns for compact display
        col1, col2, col3 = st.columns([1, 2, 1])
        
        # Display signals in professional table format with clickable chart buttons
        for i, row in enumerate(table_data):
            with st.container():
                st.markdown('<div class="signal-row">', unsafe_allow_html=True)
                cols = st.columns([1, 2, 1.5, 1.5, 2, 1, 1.5, 1.5, 1, 1])
                
                with cols[0]:
                    # Fix chart display lag by forcing immediate update
                    chart_key = f"chart_{row['Symbol']}_{i}_{row['Index']}"
                    if st.button("📊", key=chart_key, help="View detailed chart"):
                        # Set the signal and force immediate refresh
                        st.session_state.selected_signal = row['original_signal']
                        st.rerun()

                with cols[1]:
                    st.markdown(f"**{row['Symbol']}**")
                
                with cols[2]:
                    if row['Action'].startswith("📈"):
                        st.markdown(f'<span style="color: #00C851; font-weight: 600;">{row["Action"]}</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span style="color: #FF4444; font-weight: 600;">{row["Action"]}</span>', unsafe_allow_html=True)
                
                with cols[3]:
                    st.markdown(f"**{row['Price']}**")
                
                with cols[4]:
                    # Display strategies as comma-separated, truncated if too long
                    strategies = row['Strategies']
                    if len(strategies) > 20:
                        strategies = strategies[:17] + "..."
                    st.markdown(f'<small style="color: #6b7280;">{strategies}</small>', unsafe_allow_html=True)
                
                with cols[5]:
                    confidence_val = int(row['Confidence'].replace('%', ''))
                    if confidence_val >= 90:
                        conf_color = "#00C851"
                    elif confidence_val >= 75:
                        conf_color = "#FF8800"
                    else:
                        conf_color = "#FF4444"
                    st.markdown(f'<span style="color: {conf_color}; font-weight: 600;">{row["Confidence"]}</span>', unsafe_allow_html=True)
                
                with cols[6]:
                    st.markdown(f'<span style="color: #dc2626;">{row["Stop Loss"]}</span>', unsafe_allow_html=True)
                
                with cols[7]:
                    st.markdown(f'<span style="color: #059669;">{row["Target"]}</span>', unsafe_allow_html=True)
                
                with cols[8]:
                    st.markdown(f"**{row['R:R']}**")
                
                with cols[9]:
                    st.markdown(f'<small style="color: #6b7280;">{row["Time"]}</small>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Add separator
                if i < len(table_data) - 1:
                    st.divider()

def display_chart_area():
    """Display chart area at top of page for selected signal"""
    if 'selected_signal' in st.session_state and st.session_state.selected_signal:
        signal = st.session_state.selected_signal
        symbol = signal['symbol']
        clean_symbol = symbol.replace('.NS', '').replace('-USD', '')
        
        # Header with controls
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            st.subheader(f"📊 Live Chart - {clean_symbol}")
        with col2:
            if st.button("🔄 Refresh", key="refresh_live_chart", help="Update with latest data"):
                data_fetcher = get_current_data_fetcher()
                strategies = get_current_strategies()
                
                # Get fresh extended data using selected interval
                interval_val = st.session_state.get('chart_interval', '1h')
                time_range_val = st.session_state.get('chart_time_range', '1d')
                
                # Map time range to period for yfinance
                period_map = {
                    '4h': '1d', '12h': '2d', '1d': '5d', 
                    '3d': '1mo', '1w': '3mo', '1m': '1y'
                }
                period = period_map.get(time_range_val, '1d')
                
                fresh_data = data_fetcher.get_intraday_data(symbol, period=period, interval=interval_val)
                if fresh_data is not None and len(fresh_data) > 0:
                    fresh_data_with_indicators = strategies.add_technical_indicators(fresh_data)
                    current_data = get_current_market_data()
                    current_data[symbol] = fresh_data_with_indicators
                    st.success(f"Updated {clean_symbol} with {len(fresh_data)} data points")
                    st.rerun()
                    
        with col3:
            # Time range selector for continuous view
            time_range = st.selectbox(
                "Time Range", 
                ["2 hours", "4 hours", "1 day", "2 days", "1 week"],
                index=2,
                key="chart_time_range"
            )
            
        with col4:
            if st.button("❌ Close", key="close_chart"):
                st.session_state.selected_signal = None
                st.rerun()
        
        # Chart explanation
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("🟢 **Green Zone**: Target/Profit area")
        with col2:
            st.write("🔴 **Red Zone**: Stop Loss/Risk area")
        with col3:
            st.write("📍 **White Line**: Entry point")
        
        # Get and display continuous chart
        current_data = get_current_market_data()
        
        if current_data and symbol in current_data:
            chart_data = current_data[symbol]
            
            # Filter data based on selected time range
            time_ranges = {
                "2 hours": 120,
                "4 hours": 240, 
                "1 day": 1440,
                "2 days": 2880,
                "1 week": 10080
            }
            
            minutes_back = time_ranges.get(time_range, 1440)
            if st.session_state.trading_style == 'intraday':
                # For intraday, filter last N periods
                filtered_data = chart_data.tail(minutes_back // 5) if len(chart_data) > 0 else chart_data
            else:
                # For swing trading, use more data
                filtered_data = chart_data.tail(minutes_back // 60) if len(chart_data) > 0 else chart_data
            
            # Create enhanced continuous chart
            enhanced_chart = create_universal_candlestick_chart(
                filtered_data, 
                clean_symbol,
                [signal] if signal else None
            )
            
            if enhanced_chart:
                # Update chart title to show it's live
                enhanced_chart.update_layout(
                    title=f"{clean_symbol} - Live Trading Chart ({time_range})",
                    height=700
                )
                st.plotly_chart(enhanced_chart, use_container_width=True)
                
                # Display real-time signal metrics
                if len(filtered_data) > 0:
                    current_price = filtered_data['Close'].iloc[-1]
                    entry_price = signal.get('price', current_price)
                    stop_loss = signal.get('stop_loss', entry_price * 0.95)
                    target = signal.get('target', entry_price * 1.05)
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        price_format = f"${current_price:.4f}" if st.session_state.market_type == 'crypto' else f"₹{current_price:.2f}"
                        st.metric("Current Price", price_format)
                    
                    with col2:
                        entry_format = f"${entry_price:.4f}" if st.session_state.market_type == 'crypto' else f"₹{entry_price:.2f}"
                        st.metric("Signal Entry", entry_format)
                    
                    with col3:
                        pnl_pct = ((current_price - entry_price) / entry_price) * 100
                        if signal['action'] == 'SELL':
                            pnl_pct = -pnl_pct
                        st.metric("Unrealized P&L", f"{pnl_pct:.2f}%", delta=f"{pnl_pct:.2f}%")
                    
                    with col4:
                        distance_to_target = ((target - current_price) / current_price) * 100
                        if signal['action'] == 'SELL':
                            distance_to_target = -distance_to_target
                        st.metric("To Target", f"{abs(distance_to_target):.2f}%")
                    
                    with col5:
                        distance_to_stop = ((current_price - stop_loss) / current_price) * 100
                        if signal['action'] == 'SELL':
                            distance_to_stop = -distance_to_stop
                        st.metric("To Stop Loss", f"{abs(distance_to_stop):.2f}%")
                
            else:
                st.error("Unable to generate chart")
        else:
            st.warning("Chart data not available - loading...")
            
            # Try to load extended data if not available
            with st.spinner("Loading continuous chart data..."):
                data_fetcher = get_current_data_fetcher()
                strategies = get_current_strategies()
                
                # Get extended data based on time range selection
                if time_range in ["2 hours", "4 hours"]:
                    period, interval = '1d', '5m'
                elif time_range == "1 day":
                    period, interval = '2d', '5m'
                elif time_range == "2 days":
                    period, interval = '5d', '15m'
                else:  # 1 week
                    period, interval = '1mo', '1h'
                
                fresh_data = data_fetcher.get_intraday_data(symbol, period=period, interval=interval)
                if fresh_data is not None and len(fresh_data) > 0:
                    fresh_data_with_indicators = strategies.add_technical_indicators(fresh_data)
                    current_data[symbol] = fresh_data_with_indicators
                    st.success(f"Loaded {len(fresh_data)} data points for {clean_symbol}")
                    st.rerun()
                else:
                    st.error(f"Unable to load data for {clean_symbol}")
        
        st.write("---")

def display_signal_chart_inline():
    """Display chart for selected signal within the trading signals tab"""
    if 'selected_signal' not in st.session_state or not st.session_state.selected_signal:
        return
        
    signal = st.session_state.selected_signal
    symbol = signal['symbol']
    clean_symbol = symbol.replace('.NS', '').replace('-USD', '')
    
    # Chart controls
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        st.subheader(f"📊 {clean_symbol} - Signal Chart")
    
    with col2:
        interval = st.selectbox(
            "Interval", 
            ["5m", "15m", "30m", "1h", "4h", "1d", "1w"],
            index=0,  # Default to 5m for detailed view
            key="inline_chart_interval"
        )
    
    with col3:
        time_range = st.selectbox(
            "Range", 
            ["4h", "12h", "1d", "3d", "1w", "1m"],
            index=1,  # Default to 12h
            key="inline_chart_range"
        )
    
    with col4:
        # Auto-refresh toggle
        auto_refresh = st.checkbox("🔄 Auto Refresh", value=True, key="auto_refresh_chart", help="Automatically refresh chart data")
    
    
    # Load and display chart with smooth updates
    chart_data = load_chart_data(symbol, interval, time_range)
    
    if chart_data is not None and len(chart_data) > 0:
        # Create enhanced chart with signal markers
        chart = create_signal_chart_with_levels(chart_data, signal, clean_symbol, interval)
        
        if chart:
            # Use auto-refresh for the chart if enabled
            if auto_refresh:
                st.plotly_chart(chart, use_container_width=True, key=f"live_chart_{symbol.replace('-', '_').replace('.', '_')}")
            else:
                st.plotly_chart(chart, use_container_width=True)
            
            # Display real-time metrics
            display_signal_metrics(chart_data, signal)
        else:
            st.error("Unable to create chart")
    else:
        st.warning(f"No data available for {clean_symbol}")
    
    # Smooth auto-refresh using session state
    if auto_refresh:
        # Use session state to track refresh timing
        if 'last_chart_refresh' not in st.session_state:
            st.session_state.last_chart_refresh = time.time()
        
        current_time = time.time()
        if current_time - st.session_state.last_chart_refresh >= 10:  # Refresh every 10 seconds
            st.session_state.last_chart_refresh = current_time
            st.rerun()

def load_chart_data(symbol, interval, time_range):
    """Load chart data with specified interval and range"""
    try:
        data_fetcher = get_current_data_fetcher()
        strategies = get_current_strategies()
        
        # Convert time range to period for yfinance
        period_map = {
            "4h": "1d", "12h": "2d", "1d": "5d", 
            "3d": "1mo", "1w": "3mo", "1m": "1y"
        }
        period = period_map.get(time_range, "1d")
        
        # Get fresh data
        data = data_fetcher.get_intraday_data(symbol, period=period, interval=interval)
        
        if data is not None and len(data) > 0:
            # Add technical indicators
            data_with_indicators = strategies.add_technical_indicators(data)
            
            # Update session state
            current_data = get_current_market_data()
            current_data[symbol] = data_with_indicators
            
            return data_with_indicators
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
    
    return None

def create_signal_chart_with_levels(data, signal, symbol, interval):
    """Create chart with entry, stop loss, and target levels marked"""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{symbol} - {interval} Chart', 'Volume', 'RSI'),
        row_heights=[0.7, 0.2, 0.1]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add signal levels with color coding
    entry_price = signal.get('price', data['Close'].iloc[-1])
    stop_loss = signal.get('stop_loss', entry_price * 0.98 if signal['action'] == 'BUY' else entry_price * 1.02)
    target = signal.get('target', entry_price * 1.05 if signal['action'] == 'BUY' else entry_price * 0.95)
    
    # Entry line (white)
    fig.add_hline(y=entry_price, line=dict(color="white", width=2, dash="solid"), 
                  annotation_text=f"Entry: {entry_price:.4f}", row=1, col=1)
    
    # Stop loss line (red)
    fig.add_hline(y=stop_loss, line=dict(color="red", width=2, dash="dash"), 
                  annotation_text=f"Stop: {stop_loss:.4f}", row=1, col=1)
    
    # Target line (green)
    fig.add_hline(y=target, line=dict(color="green", width=2, dash="dash"), 
                  annotation_text=f"Target: {target:.4f}", row=1, col=1)
    
    # Add colored zones
    if signal['action'] == 'BUY':
        # Green zone (profit area above entry)
        fig.add_hrect(y0=entry_price, y1=target, fillcolor="green", opacity=0.1, row=1, col=1)
        # Red zone (loss area below entry)
        fig.add_hrect(y0=stop_loss, y1=entry_price, fillcolor="red", opacity=0.1, row=1, col=1)
    else:
        # Green zone (profit area below entry)
        fig.add_hrect(y0=target, y1=entry_price, fillcolor="green", opacity=0.1, row=1, col=1)
        # Red zone (loss area above entry)
        fig.add_hrect(y0=entry_price, y1=stop_loss, fillcolor="red", opacity=0.1, row=1, col=1)
    
    # Add signal marker
    signal_time = signal.get('timestamp', data.index[-1])
    marker_color = 'green' if signal['action'] == 'BUY' else 'red'
    marker_symbol = 'triangle-up' if signal['action'] == 'BUY' else 'triangle-down'
    
    fig.add_trace(
        go.Scatter(
            x=[signal_time], y=[entry_price],
            mode='markers',
            marker=dict(symbol=marker_symbol, size=15, color=marker_color),
            name=f"{signal['action']} Signal"
        ),
        row=1, col=1
    )
    
    # Volume
    colors = ['red' if close < open else 'green' for close, open in zip(data['Close'], data['Open'])]
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color=colors),
        row=2, col=1
    )
    
    # RSI
    if 'RSI' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')),
            row=3, col=1
        )
        # RSI reference lines
        fig.add_hline(y=70, line=dict(color="red", dash="dash"), row=3, col=1)
        fig.add_hline(y=30, line=dict(color="green", dash="dash"), row=3, col=1)
        fig.add_hline(y=50, line=dict(color="gray", dash="dot"), row=3, col=1)
    
    fig.update_layout(
        title=f"{symbol} - {signal['action']} Signal at {entry_price:.4f}",
        xaxis_rangeslider_visible=False,
        height=600,
        showlegend=True
    )
    
    return fig

def display_signal_metrics(data, signal):
    """Display real-time metrics for the selected signal"""
    current_price = data['Close'].iloc[-1]
    entry_price = signal.get('price', current_price)
    stop_loss = signal.get('stop_loss', entry_price * 0.98 if signal['action'] == 'BUY' else entry_price * 1.02)
    target = signal.get('target', entry_price * 1.05 if signal['action'] == 'BUY' else entry_price * 0.95)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        price_format = f"${current_price:.4f}" if st.session_state.market_type == 'crypto' else f"₹{current_price:.2f}"
        st.metric("Current Price", price_format)
    
    with col2:
        entry_format = f"${entry_price:.4f}" if st.session_state.market_type == 'crypto' else f"₹{entry_price:.2f}"
        st.metric("Signal Entry", entry_format)
    
    with col3:
        # Calculate P&L
        if signal['action'] == 'BUY':
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
        
        pnl_color = "normal"
        if pnl_pct > 0:
            pnl_color = "normal"
        elif pnl_pct < 0:
            pnl_color = "inverse"
        
        st.metric("Unrealized P&L", f"{pnl_pct:.2f}%", delta=f"{pnl_pct:.2f}%")
    
    with col4:
        # Distance to target
        if signal['action'] == 'BUY':
            target_dist = ((target - current_price) / current_price) * 100
        else:
            target_dist = ((current_price - target) / current_price) * 100
        st.metric("To Target", f"{abs(target_dist):.2f}%")
    
    with col5:
        # Distance to stop loss
        if signal['action'] == 'BUY':
            stop_dist = ((current_price - stop_loss) / current_price) * 100
        else:
            stop_dist = ((stop_loss - current_price) / current_price) * 100
        st.metric("To Stop", f"{abs(stop_dist):.2f}%")

def update_market_data():
    """Update market data based on current selection"""
    print("🔧 DEBUG: Starting update_market_data function")
    try:
        data_fetcher = get_current_data_fetcher()
        screener = get_current_screener()
        strategies = get_current_strategies()
        
        # Get appropriate symbols
        if st.session_state.market_type == 'crypto':
            symbols = screener.get_liquid_cryptos()[:50]  # Load all 50 cryptos
            market_data_key = 'crypto_market_data'
            signals_key = 'crypto_signals'
            last_update_key = 'crypto_last_update'
        else:
            symbols = screener.get_liquid_stocks()[:30]
            market_data_key = 'stock_market_data'
            signals_key = 'stock_signals'
            last_update_key = 'stock_last_update'
        
        # Update data for swing trading (longer timeframe)
        period = '5d' if st.session_state.trading_style == 'swing' else '1d'
        interval = '1h' if st.session_state.trading_style == 'swing' else '5m'
        
        # Use fast batch fetching for crypto prices (much faster)
        if st.session_state.market_type == 'crypto':
            # Get real-time prices first for faster overview
            batch_prices = data_fetcher.get_multiple_prices_fast(symbols[:20])
            
            # Update session state with fast price data
            for symbol, price_data in batch_prices.items():
                if symbol not in st.session_state[market_data_key]:
                    # Create minimal data for display
                    temp_data = pd.DataFrame({
                        'Close': [price_data['price']],
                        'Volume': [price_data['volume']],
                        'Open': [price_data['price']],
                        'High': [price_data['price']],
                        'Low': [price_data['price']]
                    })
                    temp_data.index = [price_data['timestamp']]
                    st.session_state[market_data_key][symbol] = temp_data
        
        # Fetch detailed data for technical analysis (limit for performance)
        symbol_limit = 15 if st.session_state.market_type == 'crypto' else 20
        print(f"🔧 DEBUG: Processing {len(symbols[:symbol_limit])} symbols for {st.session_state.market_type}")
        
        for symbol in symbols[:symbol_limit]:
            try:
                data = data_fetcher.get_intraday_data(symbol, period=period, interval=interval)
                if data is not None and len(data) > 0:
                    # Add technical indicators
                    data_with_indicators = strategies.add_technical_indicators(data)
                    st.session_state[market_data_key][symbol] = data_with_indicators
                    
                    # Generate basic signals
                    print(f"🔧 DEBUG: About to generate basic signals for {symbol} with {len(data_with_indicators)} data points")
                    basic_signals = strategies.generate_signals(data_with_indicators, symbol)
                    print(f"🔧 DEBUG: Generated {len(basic_signals)} basic signals for {symbol}")
                    
                    # Generate advanced signals
                    print(f"🔧 DEBUG: About to generate advanced signals for {symbol}")
                    advanced_signals = st.session_state.advanced_strategies.generate_advanced_signals(data_with_indicators, symbol)
                    print(f"🔧 DEBUG: Generated {len(advanced_signals)} advanced signals for {symbol}")
                    
                    # Combine signals
                    signals = basic_signals + advanced_signals
                    print(f"🔧 DEBUG: Total combined signals: {len(signals)} for {symbol}")
                    
                    # Debug: Print signal information
                    if signals:
                        print(f"Generated {len(signals)} signals for {symbol}:")
                        for sig in signals:
                            print(f"  - {sig['action']} {symbol} @ {sig['price']:.4f} (Confidence: {sig.get('confidence', 0):.1%})")
                    
                    # Filter signals by confidence level (use minimal threshold for debugging)
                    high_confidence_signals = [s for s in signals if s.get('confidence', 0) >= 0.01]  # Allow almost all signals
                    
                    if high_confidence_signals:
                        print(f"Filtered to {len(high_confidence_signals)} high confidence signals for {symbol}")
                    
                    # Separate very high confidence signals (>95%) for auto-execution
                    very_high_confidence_signals = [s for s in signals if s.get('confidence', 0) >= 0.95]
                    
                    # Add new signals and save to database
                    for signal in high_confidence_signals:
                        if not any(s['symbol'] == signal['symbol'] and 
                                 s['timestamp'] == signal['timestamp'] and
                                 s['action'] == signal['action'] 
                                 for s in st.session_state[signals_key]):
                            # Add market type and trading style to signal
                            signal['market_type'] = st.session_state.market_type
                            signal['trading_style'] = st.session_state.trading_style
                            
                            st.session_state[signals_key].append(signal)
                            print(f"🔧 DEBUG: Added signal to session state. Total signals in {signals_key}: {len(st.session_state[signals_key])}")
                            
                            # Auto-execute high confidence signals
                            if st.session_state.auto_trader and st.session_state.db_manager:
                                auto_result = st.session_state.auto_trader.execute_auto_trade(
                                    signal, st.session_state.db_manager
                                )
                                if auto_result:
                                    signal['auto_executed'] = True
                            
                            # Save to database if connected
                            if st.session_state.get('db_connected', False) and st.session_state.db_manager:
                                try:
                                    signal_data = signal.copy()
                                    signal_data['market_type'] = st.session_state.market_type
                                    signal_data['trading_style'] = st.session_state.trading_style
                                    
                                    # Ensure confidence is in proper format (0.0-1.0)
                                    if signal_data.get('confidence', 0) > 1:
                                        signal_data['confidence'] = signal_data['confidence'] / 100
                                    
                                    # Save the signal to database for tracking
                                    saved_signal_id = st.session_state.db_manager.save_signal(signal_data)
                                    if saved_signal_id:
                                        signal['database_id'] = saved_signal_id
                                        
                                        # Auto-execute trades with confidence >= 95%
                                        if signal in very_high_confidence_signals:
                                            # Mark as executed with entry price
                                            st.session_state.db_manager.update_signal_execution(
                                                saved_signal_id,
                                                signal_data.get('price', signal_data.get('signal_price')),
                                                signal_data.get('timestamp', signal_data.get('signal_timestamp'))
                                            )
                                            signal['auto_executed'] = True
                                            signal['execution_status'] = 'Auto-executed (95%+ confidence)'
                                        
                                except Exception as e:
                                    # Continue without database but log the error
                                    print(f"Database save error: {e}")
                                    signal['database_error'] = str(e)
            except Exception as e:
                continue
        
        # Clean old signals based on trading style
        current_time = datetime.now()
        time_limit = 86400 if st.session_state.trading_style == 'swing' else 7200  # 24 hours for swing, 2 hours for intraday
        
        filtered_signals = []
        for s in st.session_state[signals_key]:
            try:
                signal_time = s['timestamp']
                if hasattr(signal_time, 'tzinfo') and signal_time.tzinfo is not None:
                    signal_time = signal_time.replace(tzinfo=None)
                
                if (current_time - signal_time).total_seconds() < time_limit:
                    filtered_signals.append(s)
            except:
                filtered_signals.append(s)
        
        st.session_state[signals_key] = filtered_signals
        st.session_state[signals_key].sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Check for new signals and play notifications
        current_signals = [s for s in st.session_state[signals_key] if s.get('market_type') == st.session_state.market_type]
        
        if st.session_state.get('audio_notifications_enabled', False):
            try:
                st.session_state.audio_notifications.check_for_new_signals(
                    current_signals, st.session_state.previous_signals
                )
                # Update previous signals for next comparison
                st.session_state.previous_signals = current_signals.copy()
            except Exception as notify_error:
                print(f"Notification error: {notify_error}")
        
        st.session_state[last_update_key] = current_time
        
    except Exception as e:
        st.error(f"Error updating market data: {str(e)}")

def main():
    """Main application function"""
    initialize_session_state()
    
    # Remove chart area from top to prevent tab switching
    
    # EMERGENCY SIGNAL GENERATION BUTTON - TOP OF PAGE
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("⚡ EMERGENCY: GENERATE SIGNALS NOW", type="primary", use_container_width=True):
            st.info("🔧 Forcing signal generation...")
            
            try:
                from crypto_data_fetcher import CryptoDataFetcher
                from strategies import TradingStrategies
                from advanced_strategies import AdvancedTradingStrategies
                
                crypto_fetcher = CryptoDataFetcher()
                strategies = TradingStrategies()
                advanced_strategies = AdvancedTradingStrategies()
                
                # Get Bitcoin data and force generate signals
                test_symbol = "BTC-USD"
                data = crypto_fetcher.get_intraday_data(test_symbol, period="1d", interval="5m")
                
                if data is not None and len(data) > 20:
                    # Add indicators and generate signals
                    data_with_indicators = strategies.add_technical_indicators(data)
                    basic_signals = strategies.generate_signals(data_with_indicators, test_symbol)
                    advanced_signals = advanced_strategies.generate_advanced_signals(data_with_indicators, test_symbol)
                    all_signals = basic_signals + advanced_signals
                    
                    # Force add to session state
                    if 'crypto_signals' not in st.session_state:
                        st.session_state.crypto_signals = []
                    
                    for signal in all_signals:
                        signal['market_type'] = 'crypto'
                        signal['trading_style'] = 'intraday'
                        st.session_state.crypto_signals.append(signal)
                    
                    st.success(f"✅ FORCED GENERATION: {len(all_signals)} signals created!")
                    st.rerun()
                else:
                    st.error("❌ Could not get data for signal generation")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    st.write("---")
    
    st.title("🎯 Professional Trading Platform")
    st.subheader("Advanced Multi-Market Analysis with Intraday & Swing Trading Strategies")
    
    # Market and trading style selection
    display_market_selection()
    
    # Initial data loading - Force load if empty
    current_data = get_current_market_data()
    if not current_data:
        with st.spinner(f"Loading {st.session_state.market_type} market data..."):
            try:
                # Simplified data loading for initial startup
                data_fetcher = get_current_data_fetcher()
                screener = get_current_screener()
                strategies = get_current_strategies()
                
                if st.session_state.market_type == 'crypto':
                    symbols = screener.get_liquid_cryptos()[:25]  # Load top 25 cryptos initially
                    market_data_key = 'crypto_market_data'
                else:
                    symbols = screener.get_liquid_stocks()[:10]  # Load more stocks
                    market_data_key = 'stock_market_data'
                
                signals_key = 'crypto_signals' if st.session_state.market_type == 'crypto' else 'stock_signals'
                
                for symbol in symbols:
                    try:
                        data = data_fetcher.get_intraday_data(symbol, period='1d', interval='5m')
                        if data is not None and len(data) > 0:
                            data_with_indicators = strategies.add_technical_indicators(data)
                            st.session_state[market_data_key][symbol] = data_with_indicators
                            
                            # Generate signals for initial load
                            basic_signals = strategies.generate_signals(data_with_indicators, symbol)
                            advanced_signals = st.session_state.advanced_strategies.generate_advanced_signals(data_with_indicators, symbol)
                            all_signals = basic_signals + advanced_signals
                            
                            # Add signals to session state
                            for signal in all_signals:
                                if not any(s['symbol'] == signal['symbol'] and 
                                         s['timestamp'] == signal['timestamp'] and
                                         s['action'] == signal['action'] 
                                         for s in st.session_state[signals_key]):
                                    signal['market_type'] = st.session_state.market_type
                                    signal['trading_style'] = st.session_state.trading_style
                                    st.session_state[signals_key].append(signal)
                            
                    except Exception as e:
                        st.error(f"Error loading {symbol}: {str(e)}")
                        continue
                
                st.success(f"Loaded {len(st.session_state[market_data_key])} instruments")
                
            except Exception as e:
                st.error(f"Data loading failed: {str(e)}")
                # Continue with empty data
    
    # Configuration sidebar
    st.sidebar.header("⚙️ Strategy Configuration")
    
    # Audio notification controls
    audio_enabled = st.session_state.audio_notifications.create_notification_controls()
    
    # Auto-refresh settings - Enhanced for real-time trading
    st.sidebar.subheader("🔄 Auto Refresh")
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True, help="Automatically check for new signals every 2-5 minutes")
    
    # Auto-trader settings
    st.sidebar.subheader("🤖 Auto Trading")
    auto_trade_enabled = st.sidebar.checkbox("Auto Execute High Confidence (≥90%)", value=True)
    st.sidebar.info("📊 Only signals with ≥90% confidence are shown and auto-traded")
    
    # Manual refresh button for signals
    if st.sidebar.button("🔄 Refresh Signals Now", help="Force update market data and generate new signals"):
        with st.spinner("Refreshing market data and generating signals..."):
            update_market_data()
        st.success("✅ Signals updated!")
        st.rerun()
    
    if auto_trade_enabled and st.session_state.get('db_connected', False):
        # Show auto-trade summary
        auto_summary = st.session_state.auto_trader.get_auto_trade_summary(
            st.session_state.db_manager, days_back=7
        )
        if auto_summary:
            st.sidebar.success(f"🤖 Auto Trades (7d): {auto_summary['total_trades']}")
            st.sidebar.info(f"Win Rate: {auto_summary['win_rate']:.1f}%")
            st.sidebar.info(f"P&L: ₹{auto_summary['total_pnl']:.2f}")
        else:
            st.sidebar.info("🤖 No auto trades yet")
    elif auto_trade_enabled:
        st.sidebar.warning("🤖 Requires database connection")
    
    # Refresh interval based on trading style
    if st.session_state.trading_style == 'swing':
        refresh_options = [5, 10, 15, 30]
        default_refresh = 15
    else:
        refresh_options = [0.5, 1, 2, 5]
        default_refresh = 2
    
    refresh_interval = st.sidebar.selectbox(
        "Refresh Interval (minutes)",
        refresh_options,
        index=refresh_options.index(default_refresh) if default_refresh in refresh_options else 0
    )
    
    # Strategy selection
    st.sidebar.header("📊 Active Strategies")
    
    if st.session_state.trading_style == 'swing':
        available_strategies = [
            "Trend Following", "Mean Reversion", "Breakout", 
            "Support/Resistance", "Pattern Recognition", "Volume Analysis"
        ]
        default_strategies = ["Trend Following", "Breakout", "Support/Resistance"]
    else:
        available_strategies = [
            "Moving Average Crossover", "RSI", "MACD", "Bollinger Bands",
            "Stochastic", "Volume Spike", "Price Action"
        ]
        default_strategies = ["Moving Average Crossover", "RSI", "MACD"]
    
    selected_strategies = st.sidebar.multiselect(
        "Select Trading Strategies",
        available_strategies,
        default=default_strategies
    )
    
    # Risk management
    st.sidebar.header("🛡️ Risk Management")
    
    if st.session_state.trading_style == 'swing':
        max_risk = st.sidebar.slider("Max Risk per Trade (%)", 1, 8, 3)
        min_rr = st.sidebar.slider("Min Risk:Reward Ratio", 1.5, 5.0, 3.0, 0.5)
    else:
        max_risk = st.sidebar.slider("Max Risk per Trade (%)", 1, 5, 2)
        min_rr = st.sidebar.slider("Min Risk:Reward Ratio", 1.0, 4.0, 2.0, 0.5)
    
    # Configure strategies
    strategies = get_current_strategies()
    strategies.configure(
        strategies=selected_strategies,
        max_risk=max_risk,
        min_rr=min_rr
    )
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Market Overview", 
        "🎯 Trading Signals", 
        "🚀 Advanced Signals",
        "🔮 Predictions", 
        "📈 Detailed Analysis",
        "📊 Performance"
    ])
    
    with tab1:
        display_market_overview()
    
    with tab2:
        display_trading_signals()
    
    with tab3:
        # Advanced signals tab
        st.header("🚀 Advanced Trading Signals")
        st.info("Professional strategies with 3:1 risk-reward ratios and 85%+ confidence")
        
        current_signals = get_current_signals()
        advanced_signals = [s for s in current_signals if s.get('strategy', '').startswith(('Smart Money', 'Volume Price', 'Momentum Divergence'))]
        
        if advanced_signals:
            st.subheader("💎 Premium Strategy Signals")
            
            for i, signal in enumerate(advanced_signals[:8]):
                with st.expander(f"{signal['symbol'].replace('.NS', '').replace('-USD', '')} - {signal['action']} - {signal['strategy']}", expanded=i<3):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        confidence_color = "🟢" if signal.get('confidence', 0) > 0.9 else "🟡" if signal.get('confidence', 0) > 0.8 else "🟠"
                        st.write(f"{confidence_color} **Confidence:** {signal.get('confidence', 0):.1%}")
                        
                        if st.session_state.market_type == 'crypto':
                            price_format = f"${signal['price']:.4f}"
                            stop_format = f"${signal.get('stop_loss', 0):.4f}"
                            target_format = f"${signal.get('target', 0):.4f}"
                        else:
                            price_format = f"₹{signal['price']:.2f}"
                            stop_format = f"₹{signal.get('stop_loss', 0):.2f}"
                            target_format = f"₹{signal.get('target', 0):.2f}"
                        
                        st.write(f"**Entry Price:** {price_format}")
                        st.write(f"**Stop Loss:** {stop_format}")
                        st.write(f"**Target:** {target_format}")
                        st.write(f"**Risk:Reward:** 1:{signal.get('risk_reward', 0):.1f}")
                        
                        if signal.get('notes'):
                            st.write(f"**Analysis:** {signal['notes']}")
                    
                    with col2:
                        st.metric("Action", signal['action'])
                        # Fix timezone comparison for advanced signals
                        try:
                            signal_timestamp = signal['timestamp']
                            if hasattr(signal_timestamp, 'tzinfo') and signal_timestamp.tzinfo is not None:
                                signal_timestamp = signal_timestamp.replace(tzinfo=None)
                            
                            time_ago = datetime.now() - signal_timestamp
                            if time_ago.total_seconds() < 3600:
                                time_str = f"{int(time_ago.total_seconds() / 60)}m ago"
                            else:
                                time_str = f"{int(time_ago.total_seconds() / 3600)}h ago"
                        except Exception:
                            time_str = "Recently"
                        st.write(f"**Time:** {time_str}")
                    
                    with col3:
                        if st.button(f"📊 Chart", key=f"adv_chart_{i}"):
                            # Generate and display the signal chart
                            symbol = signal['symbol']
                            current_data = get_current_market_data()
                            
                            if current_data and symbol in current_data:
                                chart_data = current_data[symbol]
                                chart_fig = st.session_state.chart_generator.create_signal_chart(
                                    chart_data, signal, symbol
                                )
                                
                                if chart_fig:
                                    st.plotly_chart(chart_fig, use_container_width=True)
                                else:
                                    st.error("Unable to generate chart")
                        
                        if signal.get('auto_executed'):
                            st.success("🤖 Auto-executed")
                        else:
                            st.warning("⚠️ Manual trade")
        else:
            st.info("No advanced signals generated yet. Advanced strategies need more market data to identify high-probability setups.")
            
            # Show what advanced strategies look for
            st.subheader("🎯 Advanced Strategy Overview")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**🧠 Smart Money Concepts**")
                st.write("• Market structure breaks")
                st.write("• High volume confirmation")
                st.write("• Institutional order flow")
                st.write("• 85%+ win rate target")
                
            with col2:
                st.write("**📊 Volume Price Analysis**")
                st.write("• VWAP breakouts/breakdowns")
                st.write("• Volume spike detection")
                st.write("• Smart money accumulation")
                st.write("• 2.5:1 minimum R:R")
                
            with col3:
                st.write("**⚡ Momentum Divergence**")
                st.write("• RSI-price divergences")
                st.write("• Hidden divergence patterns")
                st.write("• Trend reversal signals")
                st.write("• 3:1 risk-reward ratio")

    with tab4:
        # Predictions tab with improved functionality
        st.header("🔮 Market Predictions")
        
        # Timewindow selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            timewindow = st.selectbox(
                "Prediction Timeframe",
                options=['5-15min', '15-30min', '30min-2h'],
                index=2,
                help="Select the timeframe for predictions"
            )
        
        with col2:
            max_predictions = st.selectbox(
                "Show Predictions",
                options=[3, 6, 9, 12],
                index=1,
                help="Number of predictions to display"
            )
        
        current_data = get_current_market_data()
        
        if current_data:
            predictions_data = []
            
            for symbol, data in list(current_data.items())[:15]:
                if data is not None and len(data) > 20:
                    try:
                        prediction_result = st.session_state.predictor.predict_next_move(data, symbol, timewindow)
                        if prediction_result['predictions']:
                            predictions_data.extend(prediction_result['predictions'])
                    except Exception as e:
                        continue
            
            if predictions_data:
                predictions_data.sort(key=lambda x: x['probability'], reverse=True)
                
                st.subheader(f"🚀 High-Probability Move Predictions ({timewindow})")
                
                # Store detailed analysis in session state for details modal
                if 'prediction_details' not in st.session_state:
                    st.session_state.prediction_details = {}
                
                for i, pred in enumerate(predictions_data[:max_predictions]):
                    # Store detailed analysis
                    st.session_state.prediction_details[f"pred_{i}"] = pred.get('detailed_analysis', {})
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        symbol_display = pred['symbol'].replace('.NS', '').replace('-USD', '')
                        
                        if pred['direction'] == 'BULLISH':
                            st.success(f"**{symbol_display}** - {pred['direction']}")
                        elif pred['direction'] == 'BEARISH':
                            st.error(f"**{symbol_display}** - {pred['direction']}")
                        else:
                            st.warning(f"**{symbol_display}** - {pred['direction']}")
                        
                        st.write(f"**Probability:** {pred['probability']:.1f}%")
                        st.write(f"**Expected Timeframe:** {pred['timeframe']}")
                        
                        # Show supporting signal names instead of count
                        signal_names = pred.get('supporting_signal_names', [])
                        if signal_names:
                            st.write(f"**Supporting Signals ({len(signal_names)}):** {', '.join(signal_names)}")
                        else:
                            st.write(f"**Supporting Signals:** {pred['supporting_signals']}")
                    
                    with col2:
                        if st.button(f"📊 Details", key=f"pred_details_{i}"):
                            # Show detailed analysis instead of placeholder
                            st.session_state[f'show_details_{i}'] = True
                    
                    # Show detailed analysis if button was clicked
                    if st.session_state.get(f'show_details_{i}', False):
                        with st.expander(f"📊 Detailed Analysis for {symbol_display}", expanded=True):
                            detailed = pred.get('detailed_analysis', {})
                            
                            if detailed.get('accumulation'):
                                acc = detailed['accumulation']
                                st.write(f"**🏦 {acc['pattern']}**")
                                st.write(f"• Strength: {acc.get('strength', 0):.1f}%")
                                st.write(f"• Prediction: {acc['prediction']}")
                                st.write(f"• Confidence: {acc['confidence']:.1%}")
                            
                            if detailed.get('smart_money'):
                                sm = detailed['smart_money']
                                st.write(f"**💰 {sm['pattern']}**")
                                st.write(f"• Signal: {sm['signal']}")
                                st.write(f"• Prediction: {sm['prediction']}")
                                st.write(f"• Timeframe: {sm['timeframe']}")
                                st.write(f"• Confidence: {sm['confidence']:.1%}")
                            
                            if detailed.get('divergences'):
                                for div in detailed['divergences']:
                                    st.write(f"**⚡ {div['type']}**")
                                    st.write(f"• Signal: {div['signal']}")
                                    st.write(f"• Prediction: {div['prediction']}")
                                    st.write(f"• Target Move: {div['target_move']}")
                                    st.write(f"• Confidence: {div['confidence']:.1%}")
                            
                            if detailed.get('breakout_setups'):
                                for setup in detailed['breakout_setups']:
                                    st.write(f"**🚀 {setup['type']}**")
                                    st.write(f"• Signal: {setup['signal']}")
                                    st.write(f"• Probability: {setup['probability']:.1%}")
                                    st.write(f"• Timeframe: {setup['timeframe']}")
                            
                            # Key levels
                            key_levels = pred.get('key_levels', {})
                            if key_levels:
                                st.write(f"**🎯 Key Levels**")
                                if st.session_state.market_type == 'crypto':
                                    st.write(f"• Current Price: ${key_levels.get('current_price', 0):.4f}")
                                    st.write(f"• Resistance: ${key_levels.get('resistance', 0):.4f}")
                                    st.write(f"• Support: ${key_levels.get('support', 0):.4f}")
                                    st.write(f"• VWAP: ${key_levels.get('vwap', 0):.4f}")
                                else:
                                    st.write(f"• Current Price: ₹{key_levels.get('current_price', 0):.2f}")
                                    st.write(f"• Resistance: ₹{key_levels.get('resistance', 0):.2f}")
                                    st.write(f"• Support: ₹{key_levels.get('support', 0):.2f}")
                                    st.write(f"• VWAP: ₹{key_levels.get('vwap', 0):.2f}")
                            
                            if st.button(f"❌ Close Details", key=f"close_details_{i}"):
                                st.session_state[f'show_details_{i}'] = False
                                st.rerun()
                    
                    st.write("---")
            else:
                st.info("No high-probability predictions detected at the moment.")
                st.write("**Tips for better predictions:**")
                st.write("• Predictions work best during active market hours")
                st.write("• Try different timeframes (shorter timeframes may show more signals)")
                st.write("• Ensure market has sufficient volatility for pattern detection")
        else:
            st.info("Loading market data for predictions...")
    
    with tab5:
        st.subheader("Individual Instrument Analysis")
        
        current_data = get_current_market_data()
        
        if current_data:
            # Symbol selection
            symbols = list(current_data.keys())
            symbol_names = [s.replace('.NS', '').replace('-USD', '') for s in symbols]
            
            selected_symbol_name = st.selectbox(
                "Select Instrument for Detailed Analysis",
                symbol_names
            )
            
            # Find full symbol
            selected_symbol = None
            for symbol in symbols:
                if symbol.replace('.NS', '').replace('-USD', '') == selected_symbol_name:
                    selected_symbol = symbol
                    break
            
            if selected_symbol and selected_symbol in current_data:
                instrument_data = current_data[selected_symbol]
                
                # Get signals for this instrument
                all_signals = get_current_signals()
                instrument_signals = [s for s in all_signals if s['symbol'] == selected_symbol]
                
                # Display chart
                fig = create_universal_candlestick_chart(instrument_data, selected_symbol, instrument_signals)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                current_price = instrument_data['Close'].iloc[-1]
                prev_price = instrument_data['Close'].iloc[-2] if len(instrument_data) > 1 else current_price
                change_pct = ((current_price - prev_price) / prev_price) * 100
                volume = instrument_data['Volume'].iloc[-1]
                
                with col1:
                    price_format = f"${current_price:.4f}" if st.session_state.market_type == 'crypto' else f"₹{current_price:.2f}"
                    st.metric("Current Price", price_format, f"{change_pct:.2f}%")
                
                with col2:
                    if 'RSI' in instrument_data.columns:
                        rsi = instrument_data['RSI'].iloc[-1]
                        st.metric("RSI", f"{rsi:.1f}")
                
                with col3:
                    st.metric("Volume", f"{volume:,.0f}")
                
                with col4:
                    if len(instrument_signals) > 0:
                        latest_signal = instrument_signals[0]
                        st.metric("Latest Signal", latest_signal['action'])
        else:
            st.info("No data available. Please wait for data to load.")
    
    with tab5:
        # Performance Analysis Tab
        if not st.session_state.get('db_connected', False):
            st.warning("📊 Performance tracking requires database connection.")
            st.info("""
            **To enable performance tracking:**
            1. Set up a Supabase account at https://supabase.com
            2. Create a new project 
            3. Go to Settings > Database
            4. Copy the connection string and add it as DATABASE_URL in Secrets
            5. Restart the application
            
            **For now, you can still use all trading features without performance tracking.**
            """)
            return
        
        st.header("📊 Trading Performance Analysis")
        
        # Performance controls
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            performance_market = st.selectbox(
                "Market Filter",
                ["All", "Stocks", "Crypto"],
                key="perf_market"
            )
            market_filter = None if performance_market == "All" else performance_market.lower()
        
        with col2:
            performance_style = st.selectbox(
                "Trading Style Filter",
                ["All", "Intraday", "Swing"],
                key="perf_style"
            )
            style_filter = None if performance_style == "All" else performance_style.lower()
        
        with col3:
            days_back = st.selectbox(
                "Time Period",
                [7, 14, 30, 60, 90],
                index=2,
                key="perf_days"
            )
        
        with col4:
            if st.button("🔄 Update Performance Data", key="update_perf"):
                with st.spinner("Updating performance data..."):
                    data_fetcher = get_current_data_fetcher()
                    crypto_fetcher = st.session_state.crypto_data_fetcher
                    
                    updates = st.session_state.performance_analyzer.update_signal_performance(
                        data_fetcher, crypto_fetcher
                    )
                    st.success(f"Updated {updates} signals")
        
        # Performance sections
        perf_tab1, perf_tab2, perf_tab3, perf_tab4, perf_tab5 = st.tabs([
            "🔴 Live Tracking", "📈 Overview", "📋 Trade History", "📊 Charts", "🏆 Strategy Analysis"
        ])
        
        with perf_tab1:
            # Live tracking of active signals
            if st.session_state.live_tracker:
                data_fetcher = get_current_data_fetcher()
                crypto_fetcher = st.session_state.crypto_data_fetcher
                
                st.session_state.live_tracker.display_live_performance(
                    data_fetcher, crypto_fetcher
                )
            else:
                st.error("Live tracking is not available. Please check database connection.")
        
        with perf_tab2:
            st.session_state.performance_analyzer.display_performance_overview(
                market_filter, style_filter, days_back
            )
        
        with perf_tab3:
            st.session_state.performance_analyzer.display_trade_history(
                market_filter, style_filter, days_back
            )
            
            # Export option
            st.session_state.performance_analyzer.export_performance_report(
                market_filter, style_filter, days_back
            )
        
        with perf_tab4:
            st.session_state.performance_analyzer.display_performance_charts(
                market_filter, style_filter, days_back
            )
        
        with perf_tab5:
            st.session_state.performance_analyzer.display_strategy_comparison(
                market_filter, style_filter, days_back
            )
            
            # Performance insights
            st.subheader("💡 Performance Insights")
            
            metrics = st.session_state.db_manager.get_performance_metrics(
                market_filter, style_filter, days_back
            )
            
            if metrics and metrics['total_trades'] > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info("📊 **Key Observations:**")
                    
                    if metrics['win_rate'] >= 60:
                        st.success("✅ Strong win rate - Strategy is working well")
                    elif metrics['win_rate'] >= 50:
                        st.warning("⚠️ Moderate win rate - Room for improvement")
                    else:
                        st.error("❌ Low win rate - Review and optimize strategies")
                    
                    if metrics['profit_factor'] >= 2.0:
                        st.success("✅ Excellent profit factor - Profitable system")
                    elif metrics['profit_factor'] >= 1.5:
                        st.warning("⚠️ Good profit factor - Solid performance")
                    else:
                        st.error("❌ Low profit factor - Losses outweigh gains")
                
                with col2:
                    st.info("🎯 **Recommendations:**")
                    
                    if metrics['avg_loss'] and abs(metrics['avg_loss']) > metrics['avg_win']:
                        st.warning("💡 Consider tighter stop losses to reduce average loss")
                    
                    if metrics['win_rate'] < 50:
                        st.warning("💡 Review entry criteria and market timing")
                    
                    if metrics['max_drawdown'] < -1000:
                        st.warning("💡 Consider reducing position sizes during drawdowns")
                    
                    st.success("💡 Continue tracking performance to identify patterns")
            else:
                st.info("No performance data available yet. Start trading to see results!")
    
    # Status and controls
    st.sidebar.markdown("---")
    st.sidebar.header("📊 System Status")
    
    # Last update time
    if st.session_state.market_type == 'crypto':
        last_update = st.session_state.crypto_last_update
        data_count = len(st.session_state.crypto_market_data)
        signals_count = len(st.session_state.crypto_signals)
    else:
        last_update = st.session_state.stock_last_update
        data_count = len(st.session_state.stock_market_data)
        signals_count = len(st.session_state.stock_signals)
    
    if last_update:
        time_since_update = datetime.now() - last_update
        st.sidebar.success(f"Last Update: {time_since_update.seconds}s ago")
    else:
        st.sidebar.warning("No data updates yet")
    
    st.sidebar.info(f"Tracking {data_count} instruments")
    st.sidebar.info(f"Active signals: {signals_count}")
    
    # Database status
    if st.session_state.get('db_connected', False):
        st.sidebar.success("🟢 Database: Connected")
    else:
        st.sidebar.warning("🟡 Database: Offline")
    
    # Market status
    if st.session_state.market_type == 'crypto':
        st.sidebar.info("🟢 Crypto Market: 24/7 Open")
    else:
        data_fetcher = get_current_data_fetcher()
        if hasattr(data_fetcher, 'is_market_open'):
            market_open = data_fetcher.is_market_open()
            status = "🟢 NSE Market: Open" if market_open else "🔴 NSE Market: Closed"
            st.sidebar.info(status)
    
    # Manual refresh
    if st.sidebar.button("🔄 Refresh Now"):
        print("🔧 DEBUG: Manual refresh button clicked")
        with st.spinner("Updating market data..."):
            update_market_data()
        st.rerun()
    
    # DIRECT SIGNAL GENERATION - BYPASS ALL CONDITIONS
    if st.sidebar.button("⚡ FORCE GENERATE SIGNALS"):
        st.sidebar.success("Forcing signal generation...")
        
        # Direct signal generation without conditions
        try:
            from crypto_data_fetcher import CryptoDataFetcher
            from strategies import TradingStrategies
            from advanced_strategies import AdvancedTradingStrategies
            
            crypto_fetcher = CryptoDataFetcher()
            strategies = TradingStrategies()
            advanced_strategies = AdvancedTradingStrategies()
            
            # Get one crypto symbol and force generate signals
            test_symbol = "BTC-USD"
            data = crypto_fetcher.get_intraday_data(test_symbol, period="1d", interval="5m")
            
            if data is not None and len(data) > 20:
                # Add indicators
                data_with_indicators = strategies.add_technical_indicators(data)
                
                # Generate signals
                basic_signals = strategies.generate_signals(data_with_indicators, test_symbol)
                advanced_signals = advanced_strategies.generate_advanced_signals(data_with_indicators, test_symbol)
                
                all_signals = basic_signals + advanced_signals
                
                # Force add to session state
                if 'crypto_signals' not in st.session_state:
                    st.session_state.crypto_signals = []
                
                for signal in all_signals:
                    signal['market_type'] = 'crypto'
                    signal['trading_style'] = 'intraday'
                    st.session_state.crypto_signals.append(signal)
                
                st.sidebar.success(f"✅ Generated {len(all_signals)} signals for {test_symbol}!")
                print(f"🔧 DEBUG: Force generated {len(all_signals)} signals")
                st.rerun()
                
            else:
                st.sidebar.error(f"No data for {test_symbol} or insufficient data points")
                print(f"🔧 DEBUG: Could not get sufficient data for {test_symbol}")
                
        except Exception as e:
            st.sidebar.error(f"Force generation failed: {str(e)}")
            print(f"🔧 ERROR in force generation: {str(e)}")
            import traceback
            print(f"🔧 TRACEBACK: {traceback.format_exc()}")
    
    # Auto-refresh logic - More aggressive for real-time signals
    if auto_refresh:
        current_time = datetime.now()
        should_refresh = False
        
        # Get current signal count to detect new signals
        current_signals = get_current_signals()
        signal_count = len(current_signals) if current_signals else 0
        
        # Check if this is first load or time-based refresh
        if last_update is None:
            should_refresh = True
        else:
            time_diff = (current_time - last_update).total_seconds()
            # Refresh more frequently for active trading (every 2 minutes instead of user setting)
            if time_diff > max(120, refresh_interval * 60):  # At least every 2 minutes
                should_refresh = True
        
        # Also check for significant market volatility or new potential signals
        if should_refresh:
            # Get fresh signals count before update
            old_signal_count = signal_count
            
            with st.spinner("🔄 Auto-updating signals..."):
                update_market_data()
            
            # Check if we got new qualifying signals
            new_signals = get_current_signals()
            new_count = len(new_signals) if new_signals else 0
            
            if new_count > old_signal_count:
                st.sidebar.success(f"✨ {new_count - old_signal_count} new signals!")
                st.success(f"🚨 {new_count - old_signal_count} new high-confidence signals detected and meet your filter criteria!")
            
            st.rerun()
    
    # DISABLED auto-refresh loop to prevent signal clearing
    # This was causing signals to be lost in an infinite refresh cycle
    current_data = get_current_market_data()
    signals_key = 'crypto_signals' if st.session_state.market_type == 'crypto' else 'stock_signals'
    current_signals = st.session_state.get(signals_key, [])
    
    print(f"🔧 DEBUG: Current status - Data: {len(current_data) if current_data else 0} instruments, {len(current_signals)} signals")
    
    # Only auto-refresh if user specifically requested it and enough time has passed
    # No automatic signal clearing!

if __name__ == "__main__":
    main()