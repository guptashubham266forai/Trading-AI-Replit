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
from database import DatabaseManager
from performance_analyzer import PerformanceAnalyzer
from auto_trader import AutoTrader
from live_tracker import LiveTracker
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
        st.session_state.market_type = 'stocks'
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
        if st.session_state.get('db_connected', False):
            st.session_state.performance_analyzer = PerformanceAnalyzer()
        else:
            st.session_state.performance_analyzer = None
    
    # Auto-trader for high confidence signals (temporarily disabled)
    if 'auto_trader' not in st.session_state:
        st.session_state.auto_trader = None
    
    # Live tracker for real-time P&L
    if 'live_tracker' not in st.session_state:
        if st.session_state.get('db_connected', False):
            st.session_state.live_tracker = LiveTracker(st.session_state.db_manager)
        else:
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
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
    
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
            fig.add_hline(y=80, line_dash="dash", line_color="red", row=4, col=1)
            fig.add_hline(y=20, line_dash="dash", line_color="green", row=4, col=1)
        
        # Williams %R
        elif 'Williams_R' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['Williams_R'], name='Williams %R', line=dict(color='orange')),
                row=4, col=1
            )
            fig.add_hline(y=-20, line_dash="dash", line_color="red", row=4, col=1)
            fig.add_hline(y=-80, line_dash="dash", line_color="green", row=4, col=1)
    
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
        confidence_filter = st.slider(
            "Minimum Confidence Level (%)",
            min_value=50,
            max_value=100,
            value=90,
            step=5,
            help="Filter signals by minimum confidence level"
        )
    
    with col2:
        show_count = st.selectbox(
            "Show Count",
            options=[6, 9, 12, 15],
            index=1,
            help="Number of signals to display"
        )
    
    with col3:
        sort_option = st.selectbox(
            "Sort By",
            options=["Confidence", "Time", "Risk:Reward"],
            index=0,
            help="Sort signals by selected criteria"
        )
    
    if not signals:
        st.info(f"No active {trading_style.lower()} {market_type.lower()} signals at the moment. Signals will appear when favorable conditions are detected.")
        return
    
    # Filter signals by time and confidence
    current_time = datetime.now()
    time_filter = 14400 if st.session_state.trading_style == 'swing' else 7200  # 4 hours for swing, 2 hours for intraday
    confidence_threshold = confidence_filter / 100.0  # Convert to decimal
    
    filtered_signals = []
    for signal in signals:
        try:
            # Time filter
            signal_time = signal['timestamp']
            if hasattr(signal_time, 'tzinfo') and signal_time.tzinfo is not None:
                signal_time = signal_time.replace(tzinfo=None)
            
            time_valid = (current_time - signal_time).total_seconds() < time_filter
            
            # Confidence filter
            confidence_valid = signal.get('confidence', 0) >= confidence_threshold
            
            if time_valid and confidence_valid:
                filtered_signals.append(signal)
        except:
            if signal.get('confidence', 0) >= confidence_threshold:
                filtered_signals.append(signal)
    
    # Sort signals based on selected criteria
    if sort_option == "Confidence":
        filtered_signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
    elif sort_option == "Time":
        filtered_signals.sort(key=lambda x: x.get('timestamp', datetime.now()), reverse=True)
    elif sort_option == "Risk:Reward":
        filtered_signals.sort(key=lambda x: x.get('risk_reward', 0), reverse=True)
    
    if not filtered_signals:
        st.info(f"No recent {trading_style.lower()} signals found with minimum {confidence_filter}% confidence.")
        return
    
    # Show signal statistics
    st.info(f"📊 Showing {len(filtered_signals)} signals with ≥{confidence_filter}% confidence | Average confidence: {sum(s.get('confidence', 0) for s in filtered_signals) / len(filtered_signals):.1%}")
    
    # Limit to show_count
    recent_signals = filtered_signals[:show_count]
    
    # Display signals in cards
    cols = st.columns(3)
    
    for i, signal in enumerate(recent_signals[:9]):
        col_idx = i % 3
        
        with cols[col_idx]:
            symbol_display = signal['symbol'].replace('.NS', '').replace('-USD', '')
            
            if signal['action'] == 'BUY':
                st.success(f"**{symbol_display}** - {signal['action']}")
            else:
                st.error(f"**{symbol_display}** - {signal['action']}")
            
            # Price formatting based on market type
            if st.session_state.market_type == 'crypto':
                price_format = f"${signal['price']:.4f}"
                stop_format = f"${signal.get('stop_loss', 0):.4f}"
                target_format = f"${signal.get('target', 0):.4f}"
            else:
                price_format = f"₹{signal['price']:.2f}"
                stop_format = f"₹{signal.get('stop_loss', 0):.2f}"
                target_format = f"₹{signal.get('target', 0):.2f}"
            
            st.write(f"**Price:** {price_format}")
            st.write(f"**Strategy:** {signal['strategy']}")
            st.write(f"**Confidence:** {signal['confidence']:.1%}")
            st.write(f"**Stop Loss:** {stop_format}")
            st.write(f"**Target:** {target_format}")
            
            if signal.get('risk_reward'):
                st.write(f"**Risk:Reward:** 1:{signal['risk_reward']:.1f}")
            
            # Time format
            time_ago = current_time - signal['timestamp'].replace(tzinfo=None)
            if time_ago.total_seconds() < 3600:
                time_str = f"{int(time_ago.total_seconds() / 60)}m ago"
            else:
                time_str = f"{int(time_ago.total_seconds() / 3600)}h ago"
            
            st.write(f"**Time:** {time_str}")
            
            # Add chart visualization button
            if st.button(f"📊 View Chart", key=f"chart_{i}"):
                # Get current market data for this symbol
                current_data = get_current_market_data()
                symbol = signal['symbol']
                
                if current_data and symbol in current_data:
                    chart_data = current_data[symbol]
                    
                    # Generate mini chart for this signal
                    mini_chart = st.session_state.chart_generator.create_mini_chart(
                        chart_data, signal, symbol, height=350
                    )
                    
                    if mini_chart:
                        st.plotly_chart(mini_chart, use_container_width=True)
                    else:
                        st.error("Unable to generate chart")
                else:
                    st.warning("Chart data not available")
            
            st.write("---")

def update_market_data():
    """Update market data based on current selection"""
    try:
        data_fetcher = get_current_data_fetcher()
        screener = get_current_screener()
        strategies = get_current_strategies()
        
        # Get appropriate symbols
        if st.session_state.market_type == 'crypto':
            symbols = screener.get_liquid_cryptos()[:15]
            market_data_key = 'crypto_market_data'
            signals_key = 'crypto_signals'
            last_update_key = 'crypto_last_update'
        else:
            symbols = screener.get_liquid_stocks()[:20]
            market_data_key = 'stock_market_data'
            signals_key = 'stock_signals'
            last_update_key = 'stock_last_update'
        
        # Update data for swing trading (longer timeframe)
        period = '5d' if st.session_state.trading_style == 'swing' else '1d'
        interval = '1h' if st.session_state.trading_style == 'swing' else '5m'
        
        for symbol in symbols:
            try:
                data = data_fetcher.get_intraday_data(symbol, period=period, interval=interval)
                if data is not None and len(data) > 0:
                    # Add technical indicators
                    data_with_indicators = strategies.add_technical_indicators(data)
                    st.session_state[market_data_key][symbol] = data_with_indicators
                    
                    # Generate basic signals
                    basic_signals = strategies.generate_signals(data_with_indicators, symbol)
                    
                    # Generate advanced signals
                    advanced_signals = st.session_state.advanced_strategies.generate_advanced_signals(data_with_indicators, symbol)
                    
                    # Combine signals
                    signals = basic_signals + advanced_signals
                    
                    # Filter signals by confidence level (>90%)
                    high_confidence_signals = [s for s in signals if s.get('confidence', 0) >= 0.9]
                    
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
    
    st.title("🎯 Professional Trading Platform")
    st.subheader("Advanced Multi-Market Analysis with Intraday & Swing Trading Strategies")
    
    # Market and trading style selection
    display_market_selection()
    
    # Initial data loading
    current_data = get_current_market_data()
    if not current_data:
        with st.spinner(f"Loading {st.session_state.market_type} market data..."):
            update_market_data()
    
    # Configuration sidebar
    st.sidebar.header("⚙️ Strategy Configuration")
    
    # Audio notification controls
    audio_enabled = st.session_state.audio_notifications.create_notification_controls()
    
    # Auto-refresh settings
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    
    # Auto-trader settings
    st.sidebar.subheader("🤖 Auto Trading")
    auto_trade_enabled = st.sidebar.checkbox("Auto Execute High Confidence (≥90%)", value=True)
    st.sidebar.info("📊 Only signals with ≥90% confidence are shown and auto-traded")
    
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
                        time_ago = datetime.now() - signal['timestamp'].replace(tzinfo=None)
                        if time_ago.total_seconds() < 3600:
                            time_str = f"{int(time_ago.total_seconds() / 60)}m ago"
                        else:
                            time_str = f"{int(time_ago.total_seconds() / 3600)}h ago"
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
        # Predictions tab (reuse existing prediction logic)
        st.header("🔮 Market Predictions")
        current_data = get_current_market_data()
        
        if current_data:
            predictions_data = []
            
            for symbol, data in list(current_data.items())[:10]:
                if data is not None and len(data) > 50:
                    try:
                        prediction_result = st.session_state.predictor.predict_next_move(data, symbol)
                        if prediction_result['predictions']:
                            predictions_data.extend(prediction_result['predictions'])
                    except Exception as e:
                        continue
            
            if predictions_data:
                predictions_data.sort(key=lambda x: x['probability'], reverse=True)
                
                st.subheader("🚀 High-Probability Move Predictions")
                
                for i, pred in enumerate(predictions_data[:6]):
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
                        st.write(f"**Supporting Signals:** {pred['supporting_signals']}")
                    
                    with col2:
                        if st.button(f"📊 Details", key=f"pred_details_{i}"):
                            st.info("Detailed analysis available in next update")
                    
                    st.write("---")
            else:
                st.info("No high-probability predictions detected at the moment.")
        else:
            st.info("Loading market data for predictions...")
    
    with tab4:
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
        with st.spinner("Updating market data..."):
            update_market_data()
        st.rerun()
    
    # Auto-refresh logic
    if auto_refresh:
        current_time = datetime.now()
        should_refresh = False
        
        if last_update is None:
            should_refresh = True
        else:
            time_diff = (current_time - last_update).total_seconds()
            if time_diff > refresh_interval * 60:
                should_refresh = True
        
        if should_refresh:
            with st.spinner("Auto-refreshing market data..."):
                update_market_data()
            st.rerun()
    
    # Initial data load
    current_data = get_current_market_data()
    if not current_data:
        with st.spinner("Loading initial market data..."):
            update_market_data()
        st.rerun()

if __name__ == "__main__":
    main()