#!/usr/bin/env python3
"""Debug database save functionality"""

from datetime import datetime
from database import DatabaseManager

def test_save():
    try:
        db_manager = DatabaseManager()
        print("Database connection successful")
        
        # Test signal data
        signal_data = {
            'symbol': 'BTC-USD',
            'action': 'BUY',
            'strategy': 'Test Strategy',
            'price': 45000.0,
            'stop_loss': 43500.0,
            'target': 47000.0,
            'confidence': 96.5,
            'risk_reward': 2.0,
            'timeframe': '1h',
            'timestamp': datetime.now(),
            'market_type': 'crypto',
            'trading_style': 'intraday',
            'shares': 100,
            'position_value': 4500000.0,
            'is_executed': True,
            'execution_price': 45000.0,
            'execution_timestamp': datetime.now(),
            'notes': 'Auto-executed test trade'
        }
        
        # Try to save
        signal_id = db_manager.save_signal(signal_data)
        if signal_id:
            print(f"✅ Signal saved successfully with ID: {signal_id}")
        else:
            print("❌ Signal save returned None")
            
        # Try to retrieve
        signals = db_manager.get_signals(limit=5)
        print(f"Retrieved {len(signals)} signals from database")
        
        for signal in signals:
            print(f"  - {signal.symbol} {signal.action} at {signal.signal_price} (Confidence: {signal.confidence}%)")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_save()