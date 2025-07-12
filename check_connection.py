#!/usr/bin/env python3
"""Check database connection details"""

import os
from sqlalchemy import create_engine, text

def check_connection():
    try:
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            print("❌ No DATABASE_URL found in environment")
            return False
            
        print(f"🔍 Connection string format: {database_url[:50]}...")
        
        # Create engine with SSL settings for Supabase
        if 'supabase.com' in database_url:
            # Add SSL mode if not present
            if 'sslmode' not in database_url:
                database_url += '?sslmode=require'
        
        engine = create_engine(database_url)
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✅ Database connection successful!")
            return True
            
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        print("\n💡 Common issues:")
        print("1. Wrong password in connection string")
        print("2. Missing SSL mode for Supabase")
        print("3. Wrong database name or host")
        return False

if __name__ == "__main__":
    check_connection()