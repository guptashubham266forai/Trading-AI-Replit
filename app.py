import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import asyncio
import threading

from data_fetcher import DataFetcher
from strategies import TradingStrategies
from stock_screener import StockScreener
from predictive_analysis import PredictiveAnalysis
from utils import format_currency, calculate_risk_reward

# Page configuration
st.set_page_config(
    page_title="NSE Real-Time Stock Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_fetcher' not in st.session_state:
    st.session_state.data_fetcher = DataFetcher()
if 'strategies' not in st.session_state:
    st.session_state.strategies = TradingStrategies()
if 'screener' not in st.session_state:
    st.session_state.screener = StockScreener()
if 'predictor' not in st.session_state:
    st.session_state.predictor = PredictiveAnalysis()
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'market_data' not in st.session_state:
    st.session_state.market_data = {}
if 'signals' not in st.session_state:
    st.session_state.signals = []

def create_candlestick_chart(data, symbol, signals=None):
    """Create an interactive candlestick chart with technical indicators"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{symbol} Price Action', 'Volume', 'Technical Indicators'),
        row_heights=[0.6, 0.2, 0.2]
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
    
    # Add moving averages if available
    if 'MA_20' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MA_20'], name='MA 20', line=dict(color='orange')),
            row=1, col=1
        )
    if 'MA_50' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MA_50'], name='MA 50', line=dict(color='blue')),
            row=1, col=1
        )
    
    # Add buy/sell signals if provided
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
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='lightblue'),
        row=2, col=1
    )
    
    # RSI if available
    if 'RSI' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')),
            row=3, col=1
        )
        # Add RSI overbought/oversold lines
        # Add RSI reference lines
        fig.update_layout(
            shapes=[
                dict(type="line", x0=0, x1=1, y0=70, y1=70,
                     line=dict(color="red", dash="dash"), xref="paper", yref="y3"),
                dict(type="line", x0=0, x1=1, y0=30, y1=30,
                     line=dict(color="green", dash="dash"), xref="paper", yref="y3")
            ]
        )
    
    fig.update_layout(
        title=f"{symbol} - Real-time Analysis",
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True
    )
    
    return fig

def display_predictions():
    """Display predictive analysis for upcoming moves"""
    st.header("🔮 Predictive Analysis - Before The Move")
    
    if not st.session_state.market_data:
        st.info("Loading market data for predictions...")
        return
    
    # Analyze top stocks for predictions
    predictions_data = []
    
    for symbol, data in list(st.session_state.market_data.items())[:10]:  # Analyze top 10 stocks
        if data is not None and len(data) > 50:
            try:
                prediction_result = st.session_state.predictor.predict_next_move(data, symbol)
                if prediction_result['predictions']:
                    predictions_data.extend(prediction_result['predictions'])
                    
                    # Store detailed analysis for display
                    symbol_key = f"{symbol}_analysis"
                    st.session_state[symbol_key] = prediction_result
            except Exception as e:
                continue
    
    if not predictions_data:
        st.info("No high-probability predictions detected at the moment. Waiting for setup patterns...")
        return
    
    # Sort by probability
    predictions_data.sort(key=lambda x: x['probability'], reverse=True)
    
    st.subheader("🚀 High-Probability Move Predictions")
    
    # Display top predictions
    for i, pred in enumerate(predictions_data[:6]):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Color based on direction
            if pred['direction'] == 'BULLISH':
                st.success(f"**{pred['symbol'].replace('.NS', '')}** - {pred['direction']}")
            elif pred['direction'] == 'BEARISH':
                st.error(f"**{pred['symbol'].replace('.NS', '')}** - {pred['direction']}")
            else:
                st.warning(f"**{pred['symbol'].replace('.NS', '')}** - {pred['direction']}")
            
            st.write(f"**Probability:** {pred['probability']:.1f}%")
            st.write(f"**Expected Timeframe:** {pred['timeframe']}")
            st.write(f"**Supporting Signals:** {pred['supporting_signals']}")
            
            # Show key levels
            if pred.get('key_levels'):
                levels = pred['key_levels']
                if levels.get('current_price'):
                    st.write(f"**Current:** ₹{levels['current_price']:.2f}")
                    if pred['direction'] == 'BULLISH' and levels.get('resistance'):
                        st.write(f"**Target:** ₹{levels['resistance']:.2f} (+{((levels['resistance']/levels['current_price'])-1)*100:.1f}%)")
                    elif pred['direction'] == 'BEARISH' and levels.get('support'):
                        st.write(f"**Target:** ₹{levels['support']:.2f} ({((levels['support']/levels['current_price'])-1)*100:.1f}%)")
        
        with col2:
            # Show detailed analysis button
            if st.button(f"📊 Details", key=f"details_{i}"):
                symbol_key = f"{pred['symbol']}_analysis"
                if symbol_key in st.session_state:
                    analysis = st.session_state[symbol_key]
                    
                    with st.expander(f"Detailed Analysis - {pred['symbol'].replace('.NS', '')}", expanded=True):
                        
                        # Smart Money Flow
                        if analysis.get('smart_money'):
                            sm = analysis['smart_money']
                            st.write(f"**🏦 {sm['pattern']}**")
                            st.write(f"Signal: {sm['signal']}")
                            st.write(f"Confidence: {sm['confidence']:.1%}")
                            st.write(f"Prediction: {sm['prediction']}")
                            st.write("---")
                        
                        # Accumulation/Distribution
                        if analysis.get('accumulation'):
                            acc = analysis['accumulation']
                            st.write(f"**📈 {acc['pattern']}**")
                            st.write(f"Prediction: {acc['prediction']}")
                            st.write(f"Strength: {acc['strength']:.1f}%")
                            st.write("---")
                        
                        # Breakout Setups
                        if analysis.get('breakout_setups'):
                            st.write("**⚡ Pre-Breakout Setups:**")
                            for setup in analysis['breakout_setups']:
                                st.write(f"• {setup['type']}: {setup['signal']}")
                                st.write(f"  Probability: {setup['probability']:.1%}, Timeframe: {setup['timeframe']}")
                            st.write("---")
                        
                        # Divergences
                        if analysis.get('divergences'):
                            st.write("**🔄 Momentum Divergences:**")
                            for div in analysis['divergences']:
                                st.write(f"• {div['type']}")
                                st.write(f"  Signal: {div['signal']}")
                                st.write(f"  Expected: {div['target_move']}")
        
        st.write("---")
    
    # Summary statistics
    st.subheader("📊 Prediction Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    bullish_count = sum(1 for p in predictions_data if p['direction'] == 'BULLISH')
    bearish_count = sum(1 for p in predictions_data if p['direction'] == 'BEARISH')
    avg_probability = sum(p['probability'] for p in predictions_data) / len(predictions_data)
    high_confidence = sum(1 for p in predictions_data if p['probability'] > 75)
    
    with col1:
        st.metric("Bullish Predictions", bullish_count)
    
    with col2:
        st.metric("Bearish Predictions", bearish_count)
    
    with col3:
        st.metric("Avg Probability", f"{avg_probability:.1f}%")
    
    with col4:
        st.metric("High Confidence (>75%)", high_confidence)

def display_trading_signals():
    """Display current trading signals"""
    st.header("🎯 Active Trading Signals")
    
    if not st.session_state.signals:
        st.info("No active trading signals at the moment. Signals will appear when market conditions are favorable.")
        return
    
    # Create columns for signal display
    cols = st.columns(3)
    
    for i, signal in enumerate(st.session_state.signals[:9]):  # Display up to 9 signals
        col_idx = i % 3
        
        with cols[col_idx]:
            # Determine card color based on signal action
            if signal['action'] == 'BUY':
                st.success(f"**{signal['symbol']}** - {signal['action']}")
            else:
                st.error(f"**{signal['symbol']}** - {signal['action']}")
            
            st.write(f"**Price:** {format_currency(signal['price'])}")
            st.write(f"**Strategy:** {signal['strategy']}")
            st.write(f"**Confidence:** {signal['confidence']:.1%}")
            st.write(f"**Stop Loss:** {format_currency(signal.get('stop_loss', 0))}")
            st.write(f"**Target:** {format_currency(signal.get('target', 0))}")
            
            if signal.get('risk_reward'):
                st.write(f"**Risk:Reward:** 1:{signal['risk_reward']:.1f}")
            
            st.write(f"**Time:** {signal['timestamp'].strftime('%H:%M:%S')}")
            st.write("---")

def display_market_overview():
    """Display market overview and top movers"""
    st.header("📊 Market Overview")
    
    if not st.session_state.market_data:
        st.info("Loading market data...")
        return
    
    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate market metrics
    total_stocks = len(st.session_state.market_data)
    gainers = sum(1 for data in st.session_state.market_data.values() 
                  if len(data) > 0 and data['Close'].iloc[-1] > data['Close'].iloc[-2])
    losers = total_stocks - gainers
    avg_volume = sum(data['Volume'].iloc[-1] for data in st.session_state.market_data.values() 
                     if len(data) > 0) / max(total_stocks, 1)
    
    with col1:
        st.metric("Total Stocks Tracked", total_stocks)
    
    with col2:
        st.metric("Gainers", gainers, delta=f"{gainers-losers}")
    
    with col3:
        st.metric("Losers", losers)
    
    with col4:
        st.metric("Avg Volume", f"{avg_volume:,.0f}")
    
    # Top movers
    st.subheader("🔥 Top Movers")
    
    if st.session_state.market_data:
        movers_data = []
        for symbol, data in st.session_state.market_data.items():
            if len(data) >= 2:
                current_price = data['Close'].iloc[-1]
                prev_price = data['Close'].iloc[-2]
                change_pct = ((current_price - prev_price) / prev_price) * 100
                volume = data['Volume'].iloc[-1]
                
                movers_data.append({
                    'Symbol': symbol,
                    'Current Price': current_price,
                    'Change %': change_pct,
                    'Volume': volume
                })
        
        if movers_data:
            movers_df = pd.DataFrame(movers_data)
            movers_df = movers_df.sort_values('Change %', ascending=False)
            
            # Display top gainers and losers
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top Gainers**")
                top_gainers = movers_df.head(5)
                for _, row in top_gainers.iterrows():
                    st.write(f"**{row['Symbol']}** - {row['Change %']:.2f}% - ₹{row['Current Price']:.2f}")
            
            with col2:
                st.write("**Top Losers**")
                top_losers = movers_df.tail(5)
                for _, row in top_losers.iterrows():
                    st.write(f"**{row['Symbol']}** - {row['Change %']:.2f}% - ₹{row['Current Price']:.2f}")

def update_market_data():
    """Update market data for all tracked stocks"""
    try:
        # Get screened stocks
        screened_stocks = st.session_state.screener.get_liquid_stocks()
        
        # Fetch data for each stock
        for symbol in screened_stocks[:20]:  # Limit to top 20 for performance
            try:
                data = st.session_state.data_fetcher.get_intraday_data(symbol)
                if data is not None and len(data) > 0:
                    # Add technical indicators
                    data_with_indicators = st.session_state.strategies.add_technical_indicators(data)
                    st.session_state.market_data[symbol] = data_with_indicators
                    
                    # Generate signals
                    signals = st.session_state.strategies.generate_signals(data_with_indicators, symbol)
                    
                    # Add new signals to session state
                    for signal in signals:
                        # Avoid duplicate signals
                        if not any(s['symbol'] == signal['symbol'] and 
                                 s['timestamp'] == signal['timestamp'] and
                                 s['action'] == signal['action'] 
                                 for s in st.session_state.signals):
                            st.session_state.signals.append(signal)
                
            except Exception as e:
                st.error(f"Error fetching data for {symbol}: {str(e)}")
                continue
        
        # Keep only recent signals (last 2 hours)
        current_time = datetime.now()
        filtered_signals = []
        for s in st.session_state.signals:
            try:
                signal_time = s['timestamp']
                # Convert timezone-aware timestamp to naive if needed
                if hasattr(signal_time, 'tzinfo') and signal_time.tzinfo is not None:
                    signal_time = signal_time.replace(tzinfo=None)
                
                if (current_time - signal_time).total_seconds() < 7200:
                    filtered_signals.append(s)
            except:
                # If timestamp comparison fails, keep the signal
                filtered_signals.append(s)
        
        st.session_state.signals = filtered_signals
        
        # Sort signals by timestamp (most recent first)
        st.session_state.signals.sort(key=lambda x: x['timestamp'], reverse=True)
        
        st.session_state.last_update = current_time
        
    except Exception as e:
        st.error(f"Error updating market data: {str(e)}")

def main():
    """Main application function"""
    st.title("📈 NSE Real-Time Stock Scanner")
    st.subheader("Professional Intraday Trading Strategies & Signal Generation")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Scanner Configuration")
    
    # Auto-refresh settings
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_interval = st.sidebar.selectbox(
        "Refresh Interval (minutes)",
        [1, 2, 5, 10],
        index=1
    )
    
    # Strategy selection
    st.sidebar.header("📊 Strategy Settings")
    selected_strategies = st.sidebar.multiselect(
        "Active Strategies",
        ["Moving Average Crossover", "RSI", "MACD", "Bollinger Bands"],
        default=["Moving Average Crossover", "RSI", "MACD"]
    )
    
    # Risk management settings
    st.sidebar.header("🛡️ Risk Management")
    max_risk_per_trade = st.sidebar.slider("Max Risk per Trade (%)", 1, 5, 2)
    min_risk_reward = st.sidebar.slider("Min Risk:Reward Ratio", 1.0, 5.0, 2.0, 0.5)
    
    # Update strategies configuration
    st.session_state.strategies.configure(
        strategies=selected_strategies,
        max_risk=max_risk_per_trade,
        min_rr=min_risk_reward
    )
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Market Overview", "🎯 Trading Signals", "🔮 Predictions", "📈 Stock Analysis"])
    
    with tab1:
        display_market_overview()
    
    with tab2:
        display_trading_signals()
        
        # Show signal statistics
        if st.session_state.signals:
            st.subheader("📈 Signal Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                buy_signals = sum(1 for s in st.session_state.signals if s['action'] == 'BUY')
                st.metric("Buy Signals", buy_signals)
            
            with col2:
                sell_signals = sum(1 for s in st.session_state.signals if s['action'] == 'SELL')
                st.metric("Sell Signals", sell_signals)
            
            with col3:
                avg_confidence = sum(s['confidence'] for s in st.session_state.signals) / len(st.session_state.signals)
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    with tab3:
        display_predictions()
    
    with tab4:
        st.subheader("Individual Stock Analysis")
        
        if st.session_state.market_data:
            selected_stock = st.selectbox(
                "Select Stock for Detailed Analysis",
                list(st.session_state.market_data.keys())
            )
            
            if selected_stock and selected_stock in st.session_state.market_data:
                stock_data = st.session_state.market_data[selected_stock]
                
                # Get signals for this stock
                stock_signals = [s for s in st.session_state.signals if s['symbol'] == selected_stock]
                
                # Display chart
                fig = create_candlestick_chart(stock_data, selected_stock, stock_signals)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display stock metrics
                col1, col2, col3, col4 = st.columns(4)
                
                current_price = stock_data['Close'].iloc[-1]
                prev_price = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else current_price
                change_pct = ((current_price - prev_price) / prev_price) * 100
                volume = stock_data['Volume'].iloc[-1]
                
                with col1:
                    st.metric("Current Price", format_currency(current_price), f"{change_pct:.2f}%")
                
                with col2:
                    if 'RSI' in stock_data.columns:
                        rsi = stock_data['RSI'].iloc[-1]
                        st.metric("RSI", f"{rsi:.1f}")
                
                with col3:
                    st.metric("Volume", f"{volume:,.0f}")
                
                with col4:
                    if len(stock_signals) > 0:
                        latest_signal = stock_signals[0]
                        st.metric("Latest Signal", latest_signal['action'])
        else:
            st.info("No stock data available. Please wait for data to load.")
    
    # Status bar
    st.sidebar.markdown("---")
    st.sidebar.header("📊 System Status")
    
    if st.session_state.last_update:
        time_since_update = datetime.now() - st.session_state.last_update
        st.sidebar.success(f"Last Update: {time_since_update.seconds}s ago")
    else:
        st.sidebar.warning("No data updates yet")
    
    st.sidebar.info(f"Tracking {len(st.session_state.market_data)} stocks")
    st.sidebar.info(f"Active signals: {len(st.session_state.signals)}")
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        with st.spinner("Updating market data..."):
            update_market_data()
        st.rerun()
    
    # Auto-refresh logic
    if auto_refresh:
        if (st.session_state.last_update is None or 
            (datetime.now() - st.session_state.last_update).total_seconds() > refresh_interval * 60):
            
            with st.spinner("Auto-refreshing market data..."):
                update_market_data()
            st.rerun()
    
    # Initial data load
    if not st.session_state.market_data:
        with st.spinner("Loading initial market data..."):
            update_market_data()
        st.rerun()

if __name__ == "__main__":
    main()
