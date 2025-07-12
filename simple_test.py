#!/usr/bin/env python3
import os
from datetime import datetime
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

def test_db():
    try:
        # Get database URL
        database_url = os.getenv('DATABASE_URL')
        if 'Aipass@12@aws' in database_url:
            user = 'postgres.jnvtzqhcznmtspojahec'
            password = 'Aipass@12'
            encoded_password = quote_plus(password)
            host = 'aws-0-ap-southeast-1.pooler.supabase.com'
            port = '6543'
            database = 'postgres'
            database_url = f"postgresql://{user}:{encoded_password}@{host}:{port}/{database}?sslmode=require"
        
        engine = create_engine(database_url)
        
        with engine.connect() as conn:
            # Test simple insert
            result = conn.execute(text("""
                INSERT INTO trading_signals (symbol, action, strategy, signal_price, confidence, signal_timestamp, market_type, trading_style)
                VALUES ('BTC-USD', 'BUY', 'Test', 45000.0, 96.5, :timestamp, 'crypto', 'intraday')
                RETURNING id
            """), {'timestamp': datetime.now()})
            
            signal_id = result.fetchone()[0]
            print(f"✅ Signal saved with ID: {signal_id}")
            
            # Test retrieve
            result = conn.execute(text("SELECT COUNT(*) FROM trading_signals"))
            count = result.fetchone()[0]
            print(f"✅ Total signals in database: {count}")
            
            conn.commit()
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_db()