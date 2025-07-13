#!/usr/bin/env python3
"""
Timezone utility functions for converting to IST (GMT+5:30)
"""

from datetime import datetime, timezone, timedelta
import pytz

# Define IST timezone
IST = pytz.timezone('Asia/Kolkata')

def convert_to_ist(dt):
    """Convert datetime to IST timezone"""
    if dt is None:
        return None
    
    # If datetime is naive, assume it's UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    # Convert to IST
    ist_dt = dt.astimezone(IST)
    return ist_dt

def format_ist_time(dt, include_seconds=False):
    """Format datetime as IST string"""
    if dt is None:
        return "N/A"
    
    ist_dt = convert_to_ist(dt)
    
    if include_seconds:
        return ist_dt.strftime("%Y-%m-%d %H:%M:%S IST")
    else:
        return ist_dt.strftime("%Y-%m-%d %H:%M IST")

def get_current_ist():
    """Get current time in IST"""
    return datetime.now(IST)

def ist_to_utc(ist_dt):
    """Convert IST datetime to UTC"""
    if ist_dt is None:
        return None
    
    # If datetime is naive, assume it's IST
    if ist_dt.tzinfo is None:
        ist_dt = IST.localize(ist_dt)
    
    # Convert to UTC
    utc_dt = ist_dt.astimezone(timezone.utc)
    return utc_dt.replace(tzinfo=None)  # Remove timezone info for database storage

def calculate_ist_duration(start_time, end_time):
    """Calculate duration between two timestamps and format for IST display"""
    if not start_time or not end_time:
        return "N/A"
    
    start_ist = convert_to_ist(start_time)
    end_ist = convert_to_ist(end_time)
    
    duration = end_ist - start_ist
    
    if duration.days > 0:
        return f"{duration.days}d {duration.seconds//3600}h"
    elif duration.seconds >= 3600:
        return f"{duration.seconds//3600}h {(duration.seconds%3600)//60}m"
    else:
        return f"{duration.seconds//60}m"