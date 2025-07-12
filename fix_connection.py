#!/usr/bin/env python3
"""Fix and test database connection"""

import os
from sqlalchemy import create_engine, text

def fix_connection():
    try:
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            print("No DATABASE_URL found")
            return False
            
        print(f"Original URL: {database_url}")
        
        # Add SSL mode if it's a Supabase connection
        if 'supabase.com' in database_url and 'sslmode' not in database_url:
            if '?' in database_url:
                database_url += '&sslmode=require'
            else:
                database_url += '?sslmode=require'
                
        print(f"Fixed URL: {database_url[:50]}...")
        
        # Test connection
        engine = create_engine(database_url)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("Database connection successful!")
            
            # Test table creation
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS test_table (
                    id SERIAL PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """))
            print("Table creation test successful!")
            
            conn.execute(text("DROP TABLE IF EXISTS test_table"))
            print("Table cleanup successful!")
            
            return True
            
    except Exception as e:
        print(f"Connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    fix_connection()