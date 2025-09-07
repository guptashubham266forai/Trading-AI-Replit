import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

def display_tradingview_style_dashboard():
    """TradingView-style trading signals dashboard with chart on left and signals panel on right"""
    from main_app import get_current_signals, get_current_data_fetcher
    
    signals = get_current_signals()
    trading_style = st.session_state.trading_style.title()
    market_type = st.session_state.market_type.title()
    
    # Header with professional styling
    st.markdown("""
    <div style='background: linear-gradient(90deg, #1f4037 0%, #99f2c8 100%); padding: 15px; border-radius: 10px; margin-bottom: 15px;'>
        <h2 style='color: white; margin: 0; text-align: center;'>
            📊 Professional Trading Platform | TradingView Style
        </h2>
        <p style='color: white; margin: 5px 0 0 0; text-align: center; opacity: 0.9;'>
            {trading_style} {market_type} | Real-Time Charts & Signals
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if not signals:
        st.info("⏳ Loading real-time market data and generating signals...")
        return
    
    # TradingView-style layout: Chart (Left) + Signal Panel (Right)
    chart_col, signal_col = st.columns([0.7, 0.3])  # 70% chart, 30% signals
    
    # LEFT SIDE: CHART SECTION
    with chart_col:
        st.markdown("### 📈 Live Trading Chart")
        
        # Chart controls
        chart_controls_col1, chart_controls_col2, chart_controls_col3 = st.columns([2, 1, 1])
        
        with chart_controls_col1:
            # Symbol selector from available signals
            available_symbols = sorted(list(set([s['symbol'] for s in signals])))
            if 'selected_chart_symbol' not in st.session_state:
                st.session_state.selected_chart_symbol = available_symbols[0] if available_symbols else 'BTC-USD'
                
            selected_symbol = st.selectbox(
                "Select Symbol",
                options=available_symbols,
                index=available_symbols.index(st.session_state.selected_chart_symbol) if st.session_state.selected_chart_symbol in available_symbols else 0,
                key="chart_symbol_selector"
            )
            st.session_state.selected_chart_symbol = selected_symbol
        
        with chart_controls_col2:
            timeframe = st.selectbox(
                "Timeframe",
                options=["5m", "15m", "1h", "4h", "1d"],
                index=1,
                key="chart_timeframe"
            )
        
        with chart_controls_col3:
            chart_type = st.selectbox(
                "Chart Type",
                options=["Candlestick", "Line", "Area"],
                index=0,
                key="chart_type_selector"
            )
        
        # Display chart with signals for selected symbol
        symbol_signals = [s for s in signals if s['symbol'] == selected_symbol]
        
        if symbol_signals:
            # Get chart data
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
                    # Create professional chart
                    fig = go.Figure()
                    
                    if chart_type == "Candlestick":
                        fig.add_trace(go.Candlestick(
                            x=chart_data.index,
                            open=chart_data['Open'],
                            high=chart_data['High'],
                            low=chart_data['Low'],
                            close=chart_data['Close'],
                            name=selected_symbol
                        ))
                    elif chart_type == "Line":
                        fig.add_trace(go.Scatter(
                            x=chart_data.index,
                            y=chart_data['Close'],
                            mode='lines',
                            name=f'{selected_symbol} Close',
                            line=dict(color='#1f77b4', width=2)
                        ))
                    else:  # Area
                        fig.add_trace(go.Scatter(
                            x=chart_data.index,
                            y=chart_data['Close'],
                            fill='tozeroy',
                            mode='lines',
                            name=f'{selected_symbol} Close',
                            line=dict(color='#1f77b4', width=2)
                        ))
                    
                    # Add signal markers
                    for signal in symbol_signals[:10]:  # Show up to 10 signals on chart
                        try:
                            signal_time = signal['timestamp']
                            if hasattr(signal_time, 'tzinfo') and signal_time.tzinfo is not None:
                                signal_time = signal_time.replace(tzinfo=None)
                            
                            # Find closest price point
                            closest_idx = chart_data.index.get_loc(signal_time, method='nearest')
                            signal_price = chart_data.iloc[closest_idx]['Close']
                            
                            color = '#00ff00' if signal['action'] == 'BUY' else '#ff0000'
                            symbol_marker = '▲' if signal['action'] == 'BUY' else '▼'
                            
                            fig.add_annotation(
                                x=signal_time,
                                y=signal_price,
                                text=f"{symbol_marker} {signal['action']}<br>{signal.get('confidence', 0):.0%}",
                                showarrow=True,
                                arrowhead=2,
                                arrowcolor=color,
                                bgcolor=color,
                                bordercolor=color,
                                font=dict(color='white', size=10)
                            )
                        except:
                            continue
                    
                    # Professional chart styling
                    fig.update_layout(
                        title=f"{selected_symbol} - {timeframe} Chart",
                        template='plotly_white',
                        height=500,
                        showlegend=False,
                        margin=dict(l=0, r=0, t=30, b=0),
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
                    
                    # Chart info
                    current_price = chart_data['Close'].iloc[-1]
                    price_change = chart_data['Close'].iloc[-1] - chart_data['Close'].iloc[-2] if len(chart_data) > 1 else 0
                    change_pct = (price_change / chart_data['Close'].iloc[-2]) * 100 if len(chart_data) > 1 and chart_data['Close'].iloc[-2] != 0 else 0
                    
                    price_info_col1, price_info_col2, price_info_col3 = st.columns(3)
                    
                    with price_info_col1:
                        currency_symbol = '$' if st.session_state.market_type == 'crypto' else '₹'
                        st.metric("Current Price", f"{currency_symbol}{current_price:.4f}")
                    
                    with price_info_col2:
                        st.metric("24h Change", f"{currency_symbol}{price_change:.4f}", f"{change_pct:+.2f}%")
                    
                    with price_info_col3:
                        active_signals_count = len([s for s in symbol_signals if (datetime.now() - s['timestamp'].replace(tzinfo=None)).total_seconds() < 7200])
                        st.metric("Active Signals", active_signals_count)
                
                else:
                    st.error(f"Unable to load chart data for {selected_symbol}")
                    
            except Exception as e:
                st.error(f"Chart loading error: {str(e)}")
        
        else:
            st.info(f"No signals available for {selected_symbol}")
    
    # RIGHT SIDE: SIGNALS PANEL
    with signal_col:
        st.markdown("### 🎯 Live Signals Panel")
        
        # Signal panel filters
        with st.container():
            st.markdown("""
            <div style='background: #f8f9fa; padding: 10px; border-radius: 8px; margin-bottom: 15px;'>
            <h5 style='margin: 0 0 10px 0; color: #333;'>Filters & Settings</h5>
            </div>
            """, unsafe_allow_html=True)
            
            # Strategy filter
            all_strategies = sorted(list(set([s.get('strategy', 'Unknown') for s in signals])))
            
            # Add SMC/ICT strategies to the list if not present
            smc_ict_strategies = ["Order Block Detection", "Fair Value Gap (FVG)", "Liquidity Sweep", 
                                "Break of Structure (BOS)", "Change of Character (CHoCH)", 
                                "Premium/Discount Arrays", "Market Maker Models", "Institutional Flow"]
            
            for strategy in smc_ict_strategies:
                if strategy not in all_strategies:
                    all_strategies.append(strategy)
            
            all_strategies.insert(0, "All Strategies")
            all_strategies.insert(1, "SMC/ICT Only")
            
            strategy_filter = st.selectbox(
                "Strategy Filter",
                options=all_strategies,
                index=0,
                key="signal_panel_strategy_filter"
            )
            
            # Confidence filter
            confidence_filter = st.slider(
                "Min Confidence",
                min_value=0,
                max_value=100,
                value=75,
                step=5,
                key="signal_panel_confidence"
            )
            
            # Show count
            show_count = st.selectbox(
                "Show",
                options=[5, 10, 15, 20],
                index=1,
                key="signal_panel_count"
            )
    
        # Filter signals based on panel filters
        filtered_signals = []
        confidence_threshold = confidence_filter / 100.0
        
        for signal in signals:
            # Confidence filter
            if signal.get('confidence', 0) < confidence_threshold:
                continue
                
            # Strategy filter
            strategy = signal.get('strategy', 'Unknown')
            if strategy_filter == "All Strategies":
                filtered_signals.append(signal)
            elif strategy_filter == "SMC/ICT Only":
                if strategy.startswith(('ICT', 'SMC', 'Smart Money', 'Order Block', 'Fair Value Gap', 'Liquidity Sweep', 'Break of Structure', 'Change of Character', 'Premium/Discount', 'Market Maker', 'Institutional')):
                    filtered_signals.append(signal)
            elif strategy == strategy_filter:
                filtered_signals.append(signal)
        
        # Sort by confidence (highest first)
        filtered_signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Display signals in panel format
        if filtered_signals:
            st.markdown(f"**{len(filtered_signals)} Signals Found**")
            
            # Display top signals
            display_signals = filtered_signals[:show_count]
            
            for i, signal in enumerate(display_signals):
                strategy = signal.get('strategy', 'Unknown')
                
                # Determine signal category
                if strategy.startswith(('ICT', 'SMC', 'Smart Money', 'Order Block', 'Fair Value Gap', 'Liquidity Sweep', 'Break of Structure', 'Change of Character', 'Premium/Discount', 'Market Maker', 'Institutional')):
                    category_color = "#e74c3c"  # Red
                    category_icon = "🏛️"
                    category_name = "SMC/ICT"
                elif strategy in ['Volume Price Analysis', 'Momentum Divergence']:
                    category_color = "#f39c12"  # Orange
                    category_icon = "🚀"
                    category_name = "Advanced"
                else:
                    category_color = "#3498db"  # Blue
                    category_icon = "📊"
                    category_name = "Standard"
                
                # Signal card in TradingView style
                with st.container():
                    # Action color
                    action_color = "#2ecc71" if signal['action'] == 'BUY' else "#e74c3c"
                    
                    st.markdown(f"""
                    <div style='
                        background: white;
                        border: 1px solid #ddd;
                        border-left: 4px solid {action_color};
                        padding: 12px;
                        margin: 8px 0;
                        border-radius: 6px;
                        font-size: 13px;
                    '>
                        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
                            <strong style='color: {action_color}; font-size: 14px;'>{signal['symbol']} {signal['action']}</strong>
                            <span style='background: {category_color}; color: white; padding: 2px 6px; border-radius: 10px; font-size: 10px;'>
                                {category_icon} {category_name}
                            </span>
                        </div>
                        <div style='color: #666; margin-bottom: 8px;'>{strategy}</div>
                        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 12px;'>
                            <div>Entry: <strong>{signal['price']:.4f}</strong></div>
                            <div>Target: <strong>{signal['target']:.4f}</strong></div>
                            <div>Stop: <strong>{signal['stop_loss']:.4f}</strong></div>
                            <div>R:R: <strong>{signal.get('risk_reward', 0):.1f}:1</strong></div>
                        </div>
                        <div style='margin-top: 8px; text-align: center;'>
                            <span style='background: #27ae60; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px;'>
                                Confidence: {signal.get('confidence', 0):.0%}
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Quick action buttons
                signal_btn_col1, signal_btn_col2 = st.columns(2)
                
                with signal_btn_col1:
                    if st.button("📊 Focus", key=f"focus_{i}", help="Show this symbol on chart"):
                        st.session_state.selected_chart_symbol = signal['symbol']
                        st.rerun()
                
                with signal_btn_col2:
                    if st.button("🔔 Alert", key=f"alert_{i}", help="Set price alert"):
                        st.success(f"Alert set for {signal['symbol']}!")
                
                st.markdown("---")
        
        else:
            st.info("No signals match your current filters.")
            st.write("**Try:**")
            st.write("• Lowering confidence threshold")
            st.write("• Changing strategy filter")
            st.write("• Waiting for new signals")