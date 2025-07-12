import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import asyncio
import threading

from crypto_data_fetcher import CryptoDataFetcher
from strategies import TradingStrategies
from crypto_screener import CryptoScreener
from predictive_analysis import PredictiveAnalysis
from utils import format_currency, calculate_risk_reward

# Page configuration
st.set_page_config(
    page_title="Crypto Real-Time Scanner",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'crypto_data_fetcher' not in st.session_state:
    st.session_state.crypto_data_fetcher = CryptoDataFetcher()
if 'crypto_strategies' not in st.session_state:
    st.session_state.crypto_strategies = TradingStrategies()
if 'crypto_screener' not in st.session_state:
    st.session_state.crypto_screener = CryptoScreener()
if 'crypto_predictor' not in st.session_state:
    st.session_state.crypto_predictor = PredictiveAnalysis()
if 'crypto_last_update' not in st.session_state:
    st.session_state.crypto_last_update = None
if 'crypto_market_data' not in st.session_state:
    st.session_state.crypto_market_data = {}
if 'crypto_signals' not in st.session_state:
    st.session_state.crypto_signals = []

def create_crypto_candlestick_chart(data, symbol, signals=None):
    """Create an interactive candlestick chart for cryptocurrency"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{symbol.replace("-USD", "")} Price Action', 'Volume', 'Technical Indicators'),
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
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.update_layout(
        title=f"{symbol.replace('-USD', '')} - Real-time Crypto Analysis",
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True
    )
    
    return fig

def display_crypto_market_overview():
    """Display cryptocurrency market overview"""
    st.header("₿ Crypto Market Overview")
    
    if not st.session_state.crypto_market_data:
        st.info("Loading crypto market data...")
        return
    
    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate market metrics
    total_cryptos = len(st.session_state.crypto_market_data)
    gainers = sum(1 for data in st.session_state.crypto_market_data.values() 
                  if len(data) > 0 and data['Close'].iloc[-1] > data['Close'].iloc[-2])
    losers = total_cryptos - gainers
    avg_volume = sum(data['Volume'].iloc[-1] for data in st.session_state.crypto_market_data.values() 
                     if len(data) > 0) / max(total_cryptos, 1)
    
    with col1:
        st.metric("Total Cryptos Tracked", total_cryptos)
    
    with col2:
        st.metric("Gainers", gainers, delta=f"{gainers-losers}")
    
    with col3:
        st.metric("Losers", losers)
    
    with col4:
        st.metric("Avg Volume", f"{avg_volume:,.0f}")
    
    # Top movers
    st.subheader("🚀 Top Crypto Movers")
    
    if st.session_state.crypto_market_data:
        movers_data = []
        for symbol, data in st.session_state.crypto_market_data.items():
            if len(data) >= 2:
                current_price = data['Close'].iloc[-1]
                prev_price = data['Close'].iloc[-2]
                change_pct = ((current_price - prev_price) / prev_price) * 100
                volume = data['Volume'].iloc[-1]
                
                movers_data.append({
                    'Symbol': symbol.replace('-USD', ''),
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
                    st.write(f"**{row['Symbol']}** - {row['Change %']:.2f}% - ${row['Current Price']:.4f}")
            
            with col2:
                st.write("**Top Losers**")
                top_losers = movers_df.tail(5)
                for _, row in top_losers.iterrows():
                    st.write(f"**{row['Symbol']}** - {row['Change %']:.2f}% - ${row['Current Price']:.4f}")

def display_crypto_predictions():
    """Display predictive analysis for cryptocurrency moves"""
    st.header("🔮 Crypto Predictions - Before The Move")
    
    if not st.session_state.crypto_market_data:
        st.info("Loading crypto data for predictions...")
        return
    
    # Analyze top cryptos for predictions
    predictions_data = []
    
    for symbol, data in list(st.session_state.crypto_market_data.items())[:15]:  # Analyze top 15 cryptos
        if data is not None and len(data) > 50:
            try:
                prediction_result = st.session_state.crypto_predictor.predict_next_move(data, symbol)
                if prediction_result['predictions']:
                    predictions_data.extend(prediction_result['predictions'])
                    
                    # Store detailed analysis for display
                    symbol_key = f"{symbol}_crypto_analysis"
                    st.session_state[symbol_key] = prediction_result
            except Exception as e:
                continue
    
    if not predictions_data:
        st.info("No high-probability crypto predictions detected. Waiting for setup patterns...")
        return
    
    # Sort by probability
    predictions_data.sort(key=lambda x: x['probability'], reverse=True)
    
    st.subheader("🚀 High-Probability Crypto Move Predictions")
    
    # Display top predictions
    for i, pred in enumerate(predictions_data[:6]):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            crypto_name = pred['symbol'].replace('-USD', '')
            
            # Color based on direction
            if pred['direction'] == 'BULLISH':
                st.success(f"**{crypto_name}** - {pred['direction']}")
            elif pred['direction'] == 'BEARISH':
                st.error(f"**{crypto_name}** - {pred['direction']}")
            else:
                st.warning(f"**{crypto_name}** - {pred['direction']}")
            
            st.write(f"**Probability:** {pred['probability']:.1f}%")
            st.write(f"**Expected Timeframe:** {pred['timeframe']}")
            st.write(f"**Supporting Signals:** {pred['supporting_signals']}")
            
            # Show key levels
            if pred.get('key_levels'):
                levels = pred['key_levels']
                if levels.get('current_price'):
                    st.write(f"**Current:** ${levels['current_price']:.4f}")
                    if pred['direction'] == 'BULLISH' and levels.get('resistance'):
                        st.write(f"**Target:** ${levels['resistance']:.4f} (+{((levels['resistance']/levels['current_price'])-1)*100:.1f}%)")
                    elif pred['direction'] == 'BEARISH' and levels.get('support'):
                        st.write(f"**Target:** ${levels['support']:.4f} ({((levels['support']/levels['current_price'])-1)*100:.1f}%)")
        
        with col2:
            # Show detailed analysis button
            if st.button(f"📊 Details", key=f"crypto_details_{i}"):
                symbol_key = f"{pred['symbol']}_crypto_analysis"
                if symbol_key in st.session_state:
                    analysis = st.session_state[symbol_key]
                    
                    with st.expander(f"Detailed Analysis - {crypto_name}", expanded=True):
                        
                        # Smart Money Flow (Whale Activity)
                        if analysis.get('smart_money'):
                            sm = analysis['smart_money']
                            st.write(f"**🐋 {sm['pattern']}**")
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

def update_crypto_market_data():
    """Update market data for all tracked cryptocurrencies"""
    try:
        # Get screened cryptos
        screened_cryptos = st.session_state.crypto_screener.get_liquid_cryptos()
        
        # Fetch data for each crypto
        for symbol in screened_cryptos[:15]:  # Limit to top 15 for performance
            try:
                data = st.session_state.crypto_data_fetcher.get_intraday_data(symbol)
                if data is not None and len(data) > 0:
                    # Add technical indicators
                    data_with_indicators = st.session_state.crypto_strategies.add_technical_indicators(data)
                    st.session_state.crypto_market_data[symbol] = data_with_indicators
                    
                    # Generate signals
                    signals = st.session_state.crypto_strategies.generate_signals(data_with_indicators, symbol)
                    
                    # Add new signals to session state
                    for signal in signals:
                        # Avoid duplicate signals
                        if not any(s['symbol'] == signal['symbol'] and 
                                 s['timestamp'] == signal['timestamp'] and
                                 s['action'] == signal['action'] 
                                 for s in st.session_state.crypto_signals):
                            st.session_state.crypto_signals.append(signal)
                
            except Exception as e:
                st.error(f"Error fetching data for {symbol}: {str(e)}")
                continue
        
        # Keep only recent signals (last 2 hours)
        current_time = datetime.now()
        filtered_signals = []
        for s in st.session_state.crypto_signals:
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
        
        st.session_state.crypto_signals = filtered_signals
        
        # Sort signals by timestamp (most recent first)
        st.session_state.crypto_signals.sort(key=lambda x: x['timestamp'], reverse=True)
        
        st.session_state.crypto_last_update = current_time
        
    except Exception as e:
        st.error(f"Error updating crypto market data: {str(e)}")

def main():
    """Main cryptocurrency application function"""
    st.title("₿ Crypto Real-Time Scanner")
    st.subheader("Professional Cryptocurrency Trading Strategies & Signal Generation")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Crypto Scanner Configuration")
    
    # Auto-refresh settings
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_interval = st.sidebar.selectbox(
        "Refresh Interval (minutes)",
        [0.5, 1, 2, 5],
        index=1
    )
    
    # Strategy selection
    st.sidebar.header("📊 Strategy Settings")
    selected_strategies = st.sidebar.multiselect(
        "Active Strategies",
        ["Moving Average Crossover", "RSI", "MACD", "Bollinger Bands"],
        default=["Moving Average Crossover", "RSI", "MACD"]
    )
    
    # Risk management settings (adjusted for crypto)
    st.sidebar.header("🛡️ Risk Management")
    max_risk_per_trade = st.sidebar.slider("Max Risk per Trade (%)", 2, 10, 5)
    min_risk_reward = st.sidebar.slider("Min Risk:Reward Ratio", 1.0, 5.0, 2.0, 0.5)
    
    # Update strategies configuration
    st.session_state.crypto_strategies.configure(
        strategies=selected_strategies,
        max_risk=max_risk_per_trade,
        min_rr=min_risk_reward
    )
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["₿ Market Overview", "🎯 Trading Signals", "🔮 Predictions", "📈 Crypto Analysis"])
    
    with tab1:
        display_crypto_market_overview()
    
    with tab2:
        st.header("🎯 Active Crypto Trading Signals")
        
        if not st.session_state.crypto_signals:
            st.info("No active crypto trading signals at the moment. Signals will appear when market conditions are favorable.")
        else:
            # Display signals similar to stock version but adapted for crypto
            cols = st.columns(3)
            
            for i, signal in enumerate(st.session_state.crypto_signals[:9]):
                col_idx = i % 3
                
                with cols[col_idx]:
                    crypto_name = signal['symbol'].replace('-USD', '')
                    
                    if signal['action'] == 'BUY':
                        st.success(f"**{crypto_name}** - {signal['action']}")
                    else:
                        st.error(f"**{crypto_name}** - {signal['action']}")
                    
                    st.write(f"**Price:** ${signal['price']:.4f}")
                    st.write(f"**Strategy:** {signal['strategy']}")
                    st.write(f"**Confidence:** {signal['confidence']:.1%}")
                    st.write(f"**Stop Loss:** ${signal.get('stop_loss', 0):.4f}")
                    st.write(f"**Target:** ${signal.get('target', 0):.4f}")
                    
                    if signal.get('risk_reward'):
                        st.write(f"**Risk:Reward:** 1:{signal['risk_reward']:.1f}")
                    
                    st.write(f"**Time:** {signal['timestamp'].strftime('%H:%M:%S')}")
                    st.write("---")
    
    with tab3:
        display_crypto_predictions()
    
    with tab4:
        st.subheader("Individual Cryptocurrency Analysis")
        
        if st.session_state.crypto_market_data:
            selected_crypto = st.selectbox(
                "Select Cryptocurrency for Detailed Analysis",
                [symbol.replace('-USD', '') for symbol in st.session_state.crypto_market_data.keys()]
            )
            
            # Convert back to full symbol
            full_symbol = f"{selected_crypto}-USD"
            
            if full_symbol in st.session_state.crypto_market_data:
                crypto_data = st.session_state.crypto_market_data[full_symbol]
                
                # Get signals for this crypto
                crypto_signals = [s for s in st.session_state.crypto_signals if s['symbol'] == full_symbol]
                
                # Display chart
                fig = create_crypto_candlestick_chart(crypto_data, full_symbol, crypto_signals)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display crypto metrics
                col1, col2, col3, col4 = st.columns(4)
                
                current_price = crypto_data['Close'].iloc[-1]
                prev_price = crypto_data['Close'].iloc[-2] if len(crypto_data) > 1 else current_price
                change_pct = ((current_price - prev_price) / prev_price) * 100
                volume = crypto_data['Volume'].iloc[-1]
                
                with col1:
                    st.metric("Current Price", f"${current_price:.4f}", f"{change_pct:.2f}%")
                
                with col2:
                    if 'RSI' in crypto_data.columns:
                        rsi = crypto_data['RSI'].iloc[-1]
                        st.metric("RSI", f"{rsi:.1f}")
                
                with col3:
                    st.metric("Volume", f"{volume:,.0f}")
                
                with col4:
                    if len(crypto_signals) > 0:
                        latest_signal = crypto_signals[0]
                        st.metric("Latest Signal", latest_signal['action'])
        else:
            st.info("No crypto data available. Please wait for data to load.")
    
    # Status bar
    st.sidebar.markdown("---")
    st.sidebar.header("📊 System Status")
    
    if st.session_state.crypto_last_update:
        time_since_update = datetime.now() - st.session_state.crypto_last_update
        st.sidebar.success(f"Last Update: {time_since_update.seconds}s ago")
    else:
        st.sidebar.warning("No data updates yet")
    
    st.sidebar.info(f"Tracking {len(st.session_state.crypto_market_data)} cryptos")
    st.sidebar.info(f"Active signals: {len(st.session_state.crypto_signals)}")
    st.sidebar.info("🟢 Crypto Market: 24/7 Open")
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        with st.spinner("Updating crypto market data..."):
            update_crypto_market_data()
        st.rerun()
    
    # Auto-refresh logic (faster for crypto)
    if auto_refresh:
        if (st.session_state.crypto_last_update is None or 
            (datetime.now() - st.session_state.crypto_last_update).total_seconds() > refresh_interval * 60):
            
            with st.spinner("Auto-refreshing crypto market data..."):
                update_crypto_market_data()
            st.rerun()
    
    # Initial data load
    if not st.session_state.crypto_market_data:
        with st.spinner("Loading initial crypto market data..."):
            update_crypto_market_data()
        st.rerun()

if __name__ == "__main__":
    main()