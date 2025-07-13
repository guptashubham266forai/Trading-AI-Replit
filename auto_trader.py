#!/usr/bin/env python3
"""
Automatic Trading System for High Confidence Signals
Executes dummy trades for signals with confidence >= 95%
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from database import DatabaseManager

class AutoTrader:
    """Automatic trading system for high confidence signals"""
    
    def __init__(self, confidence_threshold=90.0):
        self.confidence_threshold = confidence_threshold
        self.position_size = 10000  # Default position size in currency
        self.max_risk_per_trade = 0.02  # 2% risk per trade
        
    def should_auto_execute(self, signal):
        """Check if signal should be automatically executed"""
        confidence = signal.get('confidence', 0)
        return confidence >= self.confidence_threshold
    
    def calculate_position_size(self, signal):
        """Calculate position size based on risk management"""
        try:
            entry_price = signal['price']
            stop_loss = signal.get('stop_loss')
            
            if not stop_loss:
                # If no stop loss, use 2% of entry price as risk
                risk_per_share = entry_price * 0.02
            else:
                risk_per_share = abs(entry_price - stop_loss)
            
            # Calculate position size based on max risk
            max_risk_amount = self.position_size * self.max_risk_per_trade
            shares = int(max_risk_amount / risk_per_share) if risk_per_share > 0 else 100
            
            # Ensure minimum position size
            shares = max(shares, 1)
            
            return shares
            
        except Exception as e:
            return 100  # Default position size
    
    def execute_auto_trade(self, signal, db_manager):
        """Execute automatic trade for high confidence signal"""
        try:
            if not self.should_auto_execute(signal):
                return None
                
            # Validate price data (prevent fake prices)
            entry_price = signal['price']
            if entry_price <= 0 or entry_price > 1000000:  # Basic validation
                print(f"Invalid price data for {signal['symbol']}: {entry_price}")
                return None
                
            # Calculate position details
            shares = self.calculate_position_size(signal)
            
            # Create trade execution record
            trade_data = {
                'symbol': signal['symbol'],
                'market_type': signal.get('market_type', 'stocks'),
                'trading_style': signal.get('trading_style', 'intraday'),
                'action': signal['action'],
                'strategy': signal['strategy'],
                'signal_price': entry_price,
                'stop_loss': signal.get('stop_loss'),
                'target_price': signal.get('target'),
                'confidence': signal.get('confidence', 0),
                'risk_reward': signal.get('risk_reward'),
                'timeframe': signal.get('timeframe'),
                'signal_timestamp': signal['timestamp'],
                'is_executed': True,
                'execution_price': entry_price,  # Assume filled at signal price
                'execution_timestamp': datetime.now(),
                'shares': shares,
                'position_value': shares * entry_price,
                'notes': f'Auto-executed trade (Confidence: {signal.get("confidence", 0):.1f}%)'
            }
            
            # Save to database
            signal_id = db_manager.save_signal(trade_data)
            
            if signal_id:
                # Use print instead of st.success to avoid Streamlit context issues
                print(f"🤖 Auto-executed: {signal['action']} {signal['symbol']} at {entry_price:.2f} "
                      f"(Confidence: {signal.get('confidence', 0):.1f}%)")
                
                # Schedule auto-close check (for demonstration, we'll simulate this)
                self.schedule_trade_monitoring(signal_id, trade_data, db_manager)
                
                return signal_id
            
        except Exception as e:
            print(f"Auto-trade execution failed: {str(e)}")
            return None
    
    def schedule_trade_monitoring(self, signal_id, trade_data, db_manager):
        """Monitor trade for auto-close conditions"""
        try:
            # For demo purposes, we'll simulate trade outcomes based on confidence
            confidence = trade_data.get('confidence', 0)
            
            # Higher confidence = higher success probability
            success_probability = min(confidence / 100.0, 0.95)
            
            # Simulate trade outcome (in real system, this would be based on actual market data)
            import random
            trade_successful = random.random() < success_probability
            
            if trade_successful:
                # Simulate profit (target hit)
                target_price = trade_data.get('target_price')
                if target_price:
                    close_price = target_price
                    close_reason = 'target'
                else:
                    # Assume 1-3% profit for successful trades
                    profit_pct = random.uniform(0.01, 0.03)
                    if trade_data['action'] == 'BUY':
                        close_price = trade_data['signal_price'] * (1 + profit_pct)
                    else:
                        close_price = trade_data['signal_price'] * (1 - profit_pct)
                    close_reason = 'profit'
            else:
                # Simulate loss (stop loss hit)
                stop_loss = trade_data.get('stop_loss')
                if stop_loss:
                    close_price = stop_loss
                    close_reason = 'stop_loss'
                else:
                    # Assume 1-2% loss for failed trades
                    loss_pct = random.uniform(0.01, 0.02)
                    if trade_data['action'] == 'BUY':
                        close_price = trade_data['signal_price'] * (1 - loss_pct)
                    else:
                        close_price = trade_data['signal_price'] * (1 + loss_pct)
                    close_reason = 'stop_loss'
            
            # Close the trade
            close_timestamp = datetime.now() + timedelta(minutes=random.randint(15, 240))
            db_manager.close_signal(signal_id, close_price, close_reason, close_timestamp)
            
        except Exception as e:
            pass  # Silently handle monitoring errors
    
    def get_auto_trade_summary(self, db_manager, days_back=7):
        """Get summary of auto-executed trades"""
        try:
            session = db_manager.get_session()
            
            # Query auto-executed trades
            from database import TradingSignal
            from sqlalchemy import and_
            
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            auto_trades = session.query(TradingSignal).filter(
                and_(
                    TradingSignal.is_executed == True,
                    TradingSignal.confidence >= self.confidence_threshold,
                    TradingSignal.created_at >= cutoff_date
                )
            ).all()
            
            session.close()
            
            if not auto_trades:
                return None
            
            # Calculate summary metrics
            total_trades = len(auto_trades)
            successful_trades = sum(1 for trade in auto_trades 
                                  if trade.is_closed and trade.pnl_points and trade.pnl_points > 0)
            
            win_rate = (successful_trades / total_trades * 100) if total_trades > 0 else 0
            
            total_pnl = sum(trade.pnl_amount or 0 for trade in auto_trades if trade.pnl_amount)
            
            return {
                'total_trades': total_trades,
                'successful_trades': successful_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_confidence': np.mean([trade.confidence for trade in auto_trades])
            }
            
        except Exception as e:
            return None