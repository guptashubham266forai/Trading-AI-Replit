#!/usr/bin/env python3
"""Migrate database schema to add new columns"""

import os
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

def migrate_database():
    try:
        # Get database URL and fix encoding
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
            print("🔄 Migrating database schema...")
            
            # Drop existing tables to recreate with new schema
            conn.execute(text("DROP TABLE IF EXISTS trading_signals CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS portfolio_performance CASCADE"))
            print("✅ Dropped existing tables")
            
            # Create new trading_signals table with all required columns
            conn.execute(text("""
                CREATE TABLE trading_signals (
                    id VARCHAR PRIMARY KEY,
                    symbol VARCHAR NOT NULL,
                    market_type VARCHAR NOT NULL,
                    trading_style VARCHAR NOT NULL,
                    action VARCHAR NOT NULL,
                    strategy VARCHAR NOT NULL,
                    signal_price FLOAT NOT NULL,
                    stop_loss FLOAT,
                    target_price FLOAT,
                    confidence FLOAT,
                    risk_reward FLOAT,
                    timeframe VARCHAR,
                    signal_timestamp TIMESTAMP NOT NULL,
                    is_executed BOOLEAN DEFAULT FALSE,
                    execution_price FLOAT,
                    execution_timestamp TIMESTAMP,
                    is_closed BOOLEAN DEFAULT FALSE,
                    close_price FLOAT,
                    close_timestamp TIMESTAMP,
                    close_reason VARCHAR,
                    shares INTEGER,
                    position_value FLOAT,
                    pnl_points FLOAT,
                    pnl_percentage FLOAT,
                    pnl_amount FLOAT,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """))
            print("✅ Created trading_signals table")
            
            # Create portfolio_performance table
            conn.execute(text("""
                CREATE TABLE portfolio_performance (
                    id VARCHAR PRIMARY KEY,
                    date TIMESTAMP NOT NULL,
                    market_type VARCHAR NOT NULL,
                    trading_style VARCHAR NOT NULL,
                    total_signals INTEGER DEFAULT 0,
                    successful_trades INTEGER DEFAULT 0,
                    failed_trades INTEGER DEFAULT 0,
                    win_rate FLOAT DEFAULT 0.0,
                    daily_pnl FLOAT DEFAULT 0.0,
                    cumulative_pnl FLOAT DEFAULT 0.0,
                    avg_win FLOAT DEFAULT 0.0,
                    avg_loss FLOAT DEFAULT 0.0,
                    profit_factor FLOAT DEFAULT 0.0,
                    max_drawdown FLOAT DEFAULT 0.0,
                    sharpe_ratio FLOAT DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """))
            print("✅ Created portfolio_performance table")
            
            conn.commit()
            print("✅ Database migration completed successfully!")
            
            return True
            
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False

if __name__ == "__main__":
    migrate_database()