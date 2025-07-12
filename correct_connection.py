#!/usr/bin/env python3
"""Create correct connection string and test"""

import os
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text

def create_correct_connection():
    # Parse the current connection string to extract components
    original_url = os.getenv('DATABASE_URL', '')
    print(f"Current problematic URL: {original_url}")
    
    # If password contains @, we need to URL encode it
    if 'Aipass@12@aws' in original_url:
        # Build the correct connection string manually
        user = 'postgres.jnvtzqhcznmtspojahec'
        password = 'Aipass@12'  # Original password
        encoded_password = quote_plus(password)  # URL encode special characters
        host = 'aws-0-ap-southeast-1.pooler.supabase.com'
        port = '6543'
        database = 'postgres'
        
        correct_url = f"postgresql://{user}:{encoded_password}@{host}:{port}/{database}?sslmode=require"
        print(f"Corrected URL: {correct_url[:50]}...")
        
        # Test the corrected connection
        try:
            engine = create_engine(correct_url)
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                print("✅ Database connection successful!")
                
                # Test creating a simple table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS connection_test (
                        id SERIAL PRIMARY KEY,
                        test_data TEXT,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """))
                print("✅ Table creation successful!")
                
                # Insert test data
                conn.execute(text("""
                    INSERT INTO connection_test (test_data) 
                    VALUES ('Database connection working!')
                """))
                print("✅ Data insertion successful!")
                
                # Read test data
                result = conn.execute(text("SELECT test_data FROM connection_test LIMIT 1"))
                row = result.fetchone()
                if row:
                    print(f"✅ Data retrieval successful: {row[0]}")
                
                # Cleanup
                conn.execute(text("DROP TABLE connection_test"))
                print("✅ Cleanup successful!")
                
                print(f"\n🎉 Database is fully functional!")
                print(f"Please update your DATABASE_URL secret to:")
                print(f"{correct_url}")
                
                return True
                
        except Exception as e:
            print(f"❌ Connection failed: {str(e)}")
            return False
    else:
        print("Connection string format not recognized")
        return False

if __name__ == "__main__":
    create_correct_connection()