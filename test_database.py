#!/usr/bin/env python3
"""Test database connection and create tables"""

from database import DatabaseManager
import sys

def test_database():
    try:
        print("Testing database connection...")
        db_manager = DatabaseManager()
        
        print("Creating database tables...")
        db_manager.create_tables()
        
        print("✅ Database connection successful!")
        print("✅ Tables created successfully!")
        
        # Test a simple query
        session = db_manager.get_session()
        session.execute("SELECT 1")
        session.close()
        
        print("✅ Database is ready for use!")
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_database()
    sys.exit(0 if success else 1)