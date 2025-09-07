import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

def create_sleek_signals_table(signals, confidence_filter=75):
    """Create a sleek table showing Asset Name, Strategy, and Score"""
    
    # Filter signals by confidence
    filtered_signals = [s for s in signals if s.get('confidence', 0) >= confidence_filter/100.0]
    
    # Create table data
    table_data = []
    for signal in filtered_signals[:8]:  # Show top 8 signals
        strategy = signal.get('strategy', 'Unknown')
        
        # Shorten strategy name for display
        if len(strategy) > 20:
            strategy = strategy[:18] + "..."
            
        # Determine strategy category
        if any(kw in strategy.lower() for kw in ['ict', 'smc', 'smart money', 'order block', 'fair value']):
            category = "🏛️ SMC/ICT"
            color = "#e74c3c"
        elif any(kw in strategy.lower() for kw in ['volume', 'momentum', 'divergence']):
            category = "🚀 Advanced" 
            color = "#f39c12"
        else:
            category = "📊 Technical"
            color = "#3498db"
            
        action_emoji = "🟢" if signal['action'] == 'BUY' else "🔴"
        
        table_data.append({
            'asset': signal['symbol'],
            'strategy': strategy,
            'category': category,
            'score': int(signal.get('confidence', 0) * 100),
            'action': signal['action'],
            'action_emoji': action_emoji,
            'price': signal['price'],
            'target': signal['target'],
            'stop': signal['stop_loss'],
            'color': color
        })
    
    return table_data

def display_enhanced_tradingview_dashboard():
    """Enhanced TradingView-style dashboard with real-time updates and signal zones"""
    from main_app import get_current_signals, get_current_data_fetcher
    
    signals = get_current_signals()
    trading_style = st.session_state.trading_style.title()
    market_type = st.session_state.market_type.title()
    
    # Auto-refresh functionality - 1 minute updates like TradingView
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    current_time = time.time()
    if current_time - st.session_state.last_refresh >= 60:  # 1-minute refresh
        st.session_state.last_refresh = current_time
        st.rerun()
    
    # Header with real-time status
    time_since_refresh = int((current_time - st.session_state.last_refresh))
    st.markdown(f"""
    <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 10px; margin-bottom: 15px;'>
        <h2 style='color: white; margin: 0; text-align: center;'>
            📊 Professional Trading Platform | Live Updates
        </h2>
        <p style='color: white; margin: 5px 0 0 0; text-align: center; opacity: 0.9;'>
            {trading_style} {market_type} | 🔄 Next refresh: {60-time_since_refresh}s | Last: {datetime.fromtimestamp(st.session_state.last_refresh).strftime('%H:%M:%S')}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if not signals:
        st.info("⏳ Loading real-time market data and generating signals...")
        return
    
    # TradingView-style layout: Chart (70%) + Sleek Signal Table (30%)
    chart_col, signal_col = st.columns([0.7, 0.3])
    
    # LEFT SIDE: ENHANCED CHART WITH SIGNAL ZONES
    with chart_col:
        st.markdown("### 📈 Live Chart with Signal Zones")
        
        # Chart controls
        chart_controls_col1, chart_controls_col2, chart_controls_col3 = st.columns([2, 1, 1])
        
        with chart_controls_col1:
            available_symbols = sorted(list(set([s['symbol'] for s in signals])))
            if 'selected_chart_symbol' not in st.session_state:
                st.session_state.selected_chart_symbol = available_symbols[0] if available_symbols else 'BTC-USD'
                
            selected_symbol = st.selectbox(
                "Select Symbol",
                options=available_symbols,
                index=available_symbols.index(st.session_state.selected_chart_symbol) if st.session_state.selected_chart_symbol in available_symbols else 0,
                key="enhanced_chart_symbol_selector"
            )
            st.session_state.selected_chart_symbol = selected_symbol
        
        with chart_controls_col2:
            timeframe = st.selectbox(
                "Timeframe",
                options=["5m", "15m", "1h", "4h", "1d"],
                index=1,
                key="enhanced_chart_timeframe"
            )
        
        with chart_controls_col3:
            show_zones = st.checkbox("Show Zones", value=True, help="Display buy/sell/target/stop zones")
        
        # Create enhanced chart with signal zones
        symbol_signals = [s for s in signals if s['symbol'] == selected_symbol]
        
        if symbol_signals:
            data_fetcher = get_current_data_fetcher()
            chart_data = None
            
            try:
                with st.spinner(f"Loading {selected_symbol} chart..."):
                    if timeframe == "5m":
                        chart_data = data_fetcher.get_intraday_data(selected_symbol, period='1d', interval='5m')
                    elif timeframe == "15m":
                        chart_data = data_fetcher.get_intraday_data(selected_symbol, period='2d', interval='15m')
                    elif timeframe == "1h":
                        chart_data = data_fetcher.get_intraday_data(selected_symbol, period='5d', interval='1h')
                    elif timeframe == "4h":
                        chart_data = data_fetcher.get_intraday_data(selected_symbol, period='1mo', interval='4h')
                    else:  # 1d
                        chart_data = data_fetcher.get_intraday_data(selected_symbol, period='6mo', interval='1d')
                
                if chart_data is not None and len(chart_data) > 0:
                    # Create professional chart with zones
                    fig = go.Figure()
                    
                    # Add candlestick chart
                    fig.add_trace(go.Candlestick(
                        x=chart_data.index,
                        open=chart_data['Open'],
                        high=chart_data['High'],
                        low=chart_data['Low'],
                        close=chart_data['Close'],
                        name=selected_symbol
                    ))
                    
                    if show_zones:
                        # Add signal zones and markers
                        for i, signal in enumerate(symbol_signals[:5]):  # Show top 5 signals
                            try:
                                signal_time = signal['timestamp']
                                if hasattr(signal_time, 'tzinfo') and signal_time.tzinfo is not None:
                                    signal_time = signal_time.replace(tzinfo=None)
                                
                                # Find closest chart time
                                closest_idx = chart_data.index.get_loc(signal_time, method='nearest')
                                entry_price = signal['price']
                                target_price = signal['target']
                                stop_loss = signal['stop_loss']
                                
                                action_color = '#00dd00' if signal['action'] == 'BUY' else '#dd0000'
                                
                                # Add entry zone (buy/sell area)
                                entry_range = abs(entry_price - stop_loss) * 0.02  # 2% of risk range
                                fig.add_shape(
                                    type="rect",
                                    x0=chart_data.index[max(0, closest_idx-2)],
                                    y0=entry_price - entry_range,
                                    x1=chart_data.index[min(len(chart_data)-1, closest_idx+8)],
                                    y1=entry_price + entry_range,
                                    fillcolor=f"rgba({'0, 221, 0' if signal['action'] == 'BUY' else '221, 0, 0'}, 0.1)",
                                    line=dict(color=action_color, width=1, dash="dot"),
                                    opacity=0.4
                                )
                                
                                # Add target zone
                                target_range = abs(target_price - entry_price) * 0.02
                                fig.add_shape(
                                    type="rect",
                                    x0=chart_data.index[max(0, closest_idx)],
                                    y0=target_price - target_range,
                                    x1=chart_data.index[min(len(chart_data)-1, closest_idx+12)],
                                    y1=target_price + target_range,
                                    fillcolor='rgba(0, 180, 0, 0.15)',
                                    line=dict(color='green', width=1, dash="dot"),
                                    opacity=0.5
                                )
                                
                                # Add stop loss zone
                                stop_range = abs(entry_price - stop_loss) * 0.02
                                fig.add_shape(
                                    type="rect",
                                    x0=chart_data.index[max(0, closest_idx)],
                                    y0=stop_loss - stop_range,
                                    x1=chart_data.index[min(len(chart_data)-1, closest_idx+12)],
                                    y1=stop_loss + stop_range,
                                    fillcolor='rgba(221, 0, 0, 0.15)',
                                    line=dict(color='red', width=1, dash="dot"),
                                    opacity=0.5
                                )
                                
                                # Add horizontal price levels
                                fig.add_hline(y=entry_price, line=dict(color=action_color, width=2, dash="solid"), opacity=0.8)
                                fig.add_hline(y=target_price, line=dict(color="green", width=1, dash="dash"), opacity=0.7)
                                fig.add_hline(y=stop_loss, line=dict(color="red", width=1, dash="dash"), opacity=0.7)
                                
                                # Add signal marker
                                symbol_marker = '▲' if signal['action'] == 'BUY' else '▼'
                                fig.add_annotation(
                                    x=signal_time,
                                    y=entry_price,
                                    text=f"{symbol_marker} {signal['action']}<br>{signal.get('confidence', 0):.0%}",
                                    showarrow=True,
                                    arrowhead=2,
                                    arrowcolor=action_color,
                                    bgcolor=action_color,
                                    bordercolor=action_color,
                                    font=dict(color='white', size=9)
                                )
                            except:
                                continue
                    
                    # Professional chart styling
                    fig.update_layout(
                        title=f"{selected_symbol} - {timeframe} Chart with Trading Zones",
                        template='plotly_white',
                        height=520,
                        showlegend=False,
                        margin=dict(l=0, r=0, t=40, b=0),
                        xaxis=dict(
                            rangeslider=dict(visible=False),
                            showgrid=True,
                            gridcolor='#E5E5E5'
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridcolor='#E5E5E5',
                            side='right'
                        ),
                        plot_bgcolor='white',
                        paper_bgcolor='white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error(f"Unable to load chart data for {selected_symbol}")
                    
            except Exception as e:
                st.error(f"Chart loading error: {str(e)}")
    
    # RIGHT SIDE: SLEEK SIGNALS TABLE
    with signal_col:
        st.markdown("### 🎯 Live Signals")
        
        # Quick filters
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            strategy_filter = st.selectbox(
                "Filter",
                options=["All", "SMC/ICT", "Technical", "Advanced"],
                index=0,
                key="sleek_strategy_filter"
            )
        with filter_col2:
            confidence_filter = st.slider(
                "Min Score",
                min_value=50,
                max_value=100,
                value=75,
                step=5,
                key="sleek_confidence_filter"
            )
        
        # Create and display sleek table
        table_data = create_sleek_signals_table(signals, confidence_filter)
        
        if table_data:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 8px; border-radius: 8px; margin: 10px 0;'>
                <h5 style='color: white; margin: 0; text-align: center;'>📊 Active Signals</h5>
            </div>
            """, unsafe_allow_html=True)
            
            # Display signals in sleek table format
            for i, row in enumerate(table_data[:6]):  # Show top 6
                st.markdown(f"""
                <div style='
                    background: white;
                    border: 1px solid #ddd;
                    border-left: 4px solid {"#00dd00" if row["action"] == "BUY" else "#dd0000"};
                    padding: 12px;
                    margin: 8px 0;
                    border-radius: 6px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                '>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;'>
                        <strong style='font-size: 14px; color: #333;'>{row["asset"]}</strong>
                        <span style='font-size: 18px;'>{row["action_emoji"]}</span>
                    </div>
                    <div style='font-size: 11px; color: #666; margin-bottom: 6px;'>{row["strategy"]}</div>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <span style='background: {row["color"]}; color: white; padding: 2px 6px; border-radius: 10px; font-size: 10px;'>
                            {row["category"]}
                        </span>
                        <strong style='color: #27ae60; font-size: 13px;'>{row["score"]}%</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Quick action buttons
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("📊", key=f"focus_sleek_{i}", help=f"Focus {row['asset']} on chart"):
                        st.session_state.selected_chart_symbol = row['asset']
                        st.rerun()
                with btn_col2:
                    if st.button("🔔", key=f"alert_sleek_{i}", help=f"Set alert for {row['asset']}"):
                        st.success(f"Alert set for {row['asset']}!")
        else:
            st.info("No signals match your criteria")
    
    # BOTTOM SECTION: Detailed Signal Information (moved under chart)
    st.markdown("---")
    st.markdown("### 📋 Detailed Signal Analysis")
    
    if symbol_signals:
        # Show detailed info for the selected symbol
        selected_signal = symbol_signals[0]  # Show most recent signal
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            currency_symbol = '$' if st.session_state.market_type == 'crypto' else '₹'
            st.metric("Entry Price", f"{currency_symbol}{selected_signal['price']:.4f}")
            
        with col2:
            st.metric("Target", f"{currency_symbol}{selected_signal['target']:.4f}")
            
        with col3:
            st.metric("Stop Loss", f"{currency_symbol}{selected_signal['stop_loss']:.4f}")
            
        with col4:
            st.metric("Risk:Reward", f"{selected_signal.get('risk_reward', 0):.1f}:1")
        
        if selected_signal.get('notes'):
            st.info(f"📝 **Analysis:** {selected_signal['notes']}")
    
    # Auto-refresh indicator at bottom
    st.caption(f"🔄 Live Updates: Chart refreshes every minute • Next update in {60-time_since_refresh} seconds")