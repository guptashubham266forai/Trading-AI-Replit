#!/usr/bin/env python3
"""
Live P&L Tracker for Active Trading Signals
Updates real-time performance of active signals
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from database import DatabaseManager
from timezone_utils import convert_to_ist, format_ist_time, calculate_ist_duration

class LiveTracker:
    """Tracks live P&L for active trading signals"""
    
    def __init__(self, db_manager):
        self.db = db_manager
    
    def get_current_price(self, symbol, data_fetcher):
        """Get current market price for a symbol"""
        try:
            if symbol.endswith('-USD'):
                # Crypto symbol
                real_time_data = data_fetcher.get_real_time_price(symbol)
                if real_time_data:
                    return real_time_data['price']
            else:
                # Stock symbol
                data = data_fetcher.get_intraday_data(symbol, period='1d', interval='5m')
                if data is not None and len(data) > 0:
                    return float(data['Close'].iloc[-1])
            return None
        except Exception as e:
            print(f"Error getting current price for {symbol}: {e}")
            return None
    
    def calculate_live_pnl(self, signal, current_price):
        """Calculate live P&L for a signal"""
        try:
            entry_price = float(signal.signal_price)
            if current_price is None:
                return None
            
            # Calculate P&L based on action
            if signal.action == 'BUY':
                pnl_points = current_price - entry_price
                pnl_percentage = (pnl_points / entry_price) * 100
            else:  # SELL
                pnl_points = entry_price - current_price
                pnl_percentage = (pnl_points / entry_price) * 100
            
            # Calculate position value if shares are available
            pnl_amount = 0
            if signal.shares and signal.shares > 0:
                pnl_amount = pnl_points * signal.shares
            
            return {
                'current_price': current_price,
                'pnl_points': pnl_points,
                'pnl_percentage': pnl_percentage,
                'pnl_amount': pnl_amount,
                'status': 'profit' if pnl_points > 0 else 'loss' if pnl_points < 0 else 'break_even'
            }
        except Exception as e:
            print(f"Error calculating P&L: {e}")
            return None
    
    def get_live_signals_performance(self, data_fetcher, crypto_data_fetcher):
        """Get live performance for all active signals"""
        try:
            # Get active signals from database (last 24 hours)
            session = self.db.get_session()
            from database import TradingSignal
            
            twenty_four_hours_ago = datetime.now() - timedelta(hours=24)
            active_signals = session.query(TradingSignal).filter(
                TradingSignal.signal_timestamp >= twenty_four_hours_ago,
                TradingSignal.is_closed == False
            ).all()
            
            live_performance = []
            
            for signal in active_signals:
                # Choose appropriate data fetcher
                fetcher = crypto_data_fetcher if signal.symbol.endswith('-USD') else data_fetcher
                
                # Get current price
                current_price = self.get_current_price(signal.symbol, fetcher)
                
                if current_price:
                    # Calculate live P&L
                    pnl_data = self.calculate_live_pnl(signal, current_price)
                    
                    if pnl_data:
                        signal_data = {
                            'id': signal.id,
                            'symbol': signal.symbol,
                            'action': signal.action,
                            'strategy': signal.strategy,
                            'entry_price': signal.signal_price,
                            'current_price': current_price,
                            'confidence': signal.confidence,
                            'market_type': signal.market_type,
                            'trading_style': signal.trading_style,
                            'signal_timestamp': signal.signal_timestamp,
                            'pnl_points': pnl_data['pnl_points'],
                            'pnl_percentage': pnl_data['pnl_percentage'],
                            'pnl_amount': pnl_data['pnl_amount'],
                            'status': pnl_data['status'],
                            'target_price': signal.target_price,
                            'stop_loss': signal.stop_loss,
                            'shares': signal.shares or 0
                        }
                        
                        live_performance.append(signal_data)
            
            session.close()
            return live_performance
            
        except Exception as e:
            print(f"Error getting live signals performance: {e}")
            return []
    
    def display_live_performance(self, data_fetcher, crypto_data_fetcher):
        """Display live performance tracking"""
        st.subheader("🔴 Live Signal Performance")
        
        live_signals = self.get_live_signals_performance(data_fetcher, crypto_data_fetcher)
        
        if not live_signals:
            st.info("No active signals to track. Signals will appear here when generated.")
            return
        
        # Create DataFrame for better display
        df = pd.DataFrame(live_signals)
        
        # Display summary stats
        total_signals = len(live_signals)
        profitable_signals = len([s for s in live_signals if s['status'] == 'profit'])
        total_pnl = sum(s['pnl_amount'] for s in live_signals)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Active Signals", total_signals)
        
        with col2:
            win_rate = (profitable_signals / total_signals) * 100 if total_signals > 0 else 0
            st.metric("Current Win Rate", f"{win_rate:.1f}%")
        
        with col3:
            st.metric("Unrealized P&L", f"${total_pnl:.2f}" if any(s['symbol'].endswith('-USD') for s in live_signals) else f"₹{total_pnl:.2f}")
        
        # Display individual signals
        st.subheader("Active Signal Details")
        
        for signal in live_signals:
            with st.expander(f"{signal['symbol'].replace('.NS', '').replace('-USD', '')} - {signal['action']} ({signal['status'].upper()})", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    currency = "$" if signal['symbol'].endswith('-USD') else "₹"
                    
                    st.write(f"**Strategy:** {signal['strategy']}")
                    st.write(f"**Entry Price:** {currency}{signal['entry_price']:.4f}")
                    st.write(f"**Current Price:** {currency}{signal['current_price']:.4f}")
                    # Fix confidence display (convert to percentage if needed)
                    confidence_val = signal['confidence']
                    if confidence_val > 1:
                        confidence_display = f"{confidence_val:.1f}%"
                    else:
                        confidence_display = f"{confidence_val:.1%}"
                    st.write(f"**Confidence:** {confidence_display}")
                    
                    # Show execution status for 95%+ confidence
                    if confidence_val >= 0.95:
                        st.success("✅ Auto-executed (95%+ confidence)")
                    
                    # Time since signal
                    signal_time_ist = format_ist_time(signal['signal_timestamp'])
                    time_diff = datetime.now() - signal['signal_timestamp']
                    if time_diff.seconds < 3600:
                        time_str = f"{time_diff.seconds // 60}m ago"
                    else:
                        time_str = f"{time_diff.seconds // 3600}h ago"
                    st.write(f"**Signal Time:** {signal_time_ist}")
                    st.write(f"**Time Ago:** {time_str}")
                
                with col2:
                    # P&L display with color coding
                    pnl_color = "🟢" if signal['status'] == 'profit' else "🔴" if signal['status'] == 'loss' else "🟡"
                    
                    st.write(f"**P&L Points:** {pnl_color} {signal['pnl_points']:.4f}")
                    st.write(f"**P&L %:** {pnl_color} {signal['pnl_percentage']:.2f}%")
                    
                    if signal['shares'] > 0:
                        st.write(f"**P&L Amount:** {pnl_color} {currency}{signal['pnl_amount']:.2f}")
                    
                    # Target and stop loss status
                    if signal['target_price']:
                        target_distance = abs(signal['current_price'] - signal['target_price'])
                        st.write(f"**Target:** {currency}{signal['target_price']:.4f} (Dist: {target_distance:.4f})")
                    
                    if signal['stop_loss']:
                        sl_distance = abs(signal['current_price'] - signal['stop_loss'])
                        st.write(f"**Stop Loss:** {currency}{signal['stop_loss']:.4f} (Dist: {sl_distance:.4f})")
        
        # Auto-refresh notice
        st.info("💡 This data updates automatically with market refresh intervals")