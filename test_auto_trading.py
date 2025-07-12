#!/usr/bin/env python3
"""Test auto-trading functionality with high confidence signals"""

from datetime import datetime
from database import DatabaseManager
from auto_trader import AutoTrader

def create_test_signals():
    """Create test signals with various confidence levels"""
    try:
        db_manager = DatabaseManager()
        auto_trader = AutoTrader(confidence_threshold=95.0)
        
        # Test signals with different confidence levels
        test_signals = [
            {
                'symbol': 'BTC-USD',
                'action': 'BUY',
                'strategy': 'Test Strategy',
                'price': 45000.0,
                'stop_loss': 43500.0,
                'target': 47000.0,
                'confidence': 96.5,  # High confidence - should auto-execute
                'risk_reward': 2.0,
                'timeframe': '1h',
                'timestamp': datetime.now(),
                'market_type': 'crypto',
                'trading_style': 'intraday'
            },
            {
                'symbol': 'RELIANCE.NS',
                'action': 'BUY',
                'strategy': 'Test Strategy',
                'price': 2800.0,
                'stop_loss': 2750.0,
                'target': 2900.0,
                'confidence': 97.2,  # High confidence - should auto-execute
                'risk_reward': 2.0,
                'timeframe': '15m',
                'timestamp': datetime.now(),
                'market_type': 'stocks',
                'trading_style': 'intraday'
            },
            {
                'symbol': 'ETH-USD',
                'action': 'SELL',
                'strategy': 'Test Strategy',
                'price': 3200.0,
                'stop_loss': 3250.0,
                'target': 3100.0,
                'confidence': 85.0,  # Lower confidence - should not auto-execute
                'risk_reward': 2.0,
                'timeframe': '1h',
                'timestamp': datetime.now(),
                'market_type': 'crypto',
                'trading_style': 'intraday'
            }
        ]
        
        # Process each signal
        for signal in test_signals:
            print(f"\nProcessing signal: {signal['symbol']} {signal['action']} (Confidence: {signal['confidence']}%)")
            
            if auto_trader.should_auto_execute(signal):
                print("  -> High confidence - executing auto trade")
                trade_id = auto_trader.execute_auto_trade(signal, db_manager)
                if trade_id:
                    print(f"  -> Auto trade executed with ID: {trade_id}")
                else:
                    print("  -> Auto trade execution failed")
            else:
                print("  -> Low confidence - saving as signal only")
                db_manager.save_signal(signal)
        
        # Get auto-trade summary
        summary = auto_trader.get_auto_trade_summary(db_manager, days_back=1)
        if summary:
            print(f"\n=== Auto Trade Summary ===")
            print(f"Total auto trades: {summary['total_trades']}")
            print(f"Successful trades: {summary['successful_trades']}")
            print(f"Win rate: {summary['win_rate']:.1f}%")
            print(f"Total P&L: ₹{summary['total_pnl']:.2f}")
            print(f"Average confidence: {summary['avg_confidence']:.1f}%")
        else:
            print("\nNo auto trade summary available")
            
        print("\n✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    create_test_signals()