#!/usr/bin/env python3
"""Test the fixed database connection"""

from database import DatabaseManager

def test_fixed_connection():
    try:
        print("Testing fixed database connection...")
        db_manager = DatabaseManager()
        print("✅ Database connection successful!")
        
        # Test session
        session = db_manager.get_session()
        session.close()
        print("✅ Database session test successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection still failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_fixed_connection()