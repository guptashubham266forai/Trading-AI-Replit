import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID
from urllib.parse import quote_plus
import uuid
import streamlit as st

Base = declarative_base()

class TradingSignal(Base):
    """Database model for trading signals"""
    __tablename__ = 'trading_signals'
    
    id = Column(String, primary_key=True)
    symbol = Column(String, nullable=False)
    market_type = Column(String, nullable=False)  # 'stocks' or 'crypto'
    trading_style = Column(String, nullable=False)  # 'intraday' or 'swing'
    action = Column(String, nullable=False)  # 'BUY' or 'SELL'
    strategy = Column(String, nullable=False)
    signal_price = Column(Float, nullable=False)
    stop_loss = Column(Float)
    target_price = Column(Float)
    confidence = Column(Float)
    risk_reward = Column(Float)
    timeframe = Column(String)
    signal_timestamp = Column(DateTime, nullable=False)
    
    # Performance tracking
    is_executed = Column(Boolean, default=False)
    execution_price = Column(Float)
    execution_timestamp = Column(DateTime)
    is_closed = Column(Boolean, default=False)
    close_price = Column(Float)
    close_timestamp = Column(DateTime)
    close_reason = Column(String)  # 'target', 'stop_loss', 'manual', 'expired'
    
    # Position details
    shares = Column(Integer)  # Number of shares/units
    position_value = Column(Float)  # Total position value
    
    # P&L calculation
    pnl_points = Column(Float)
    pnl_percentage = Column(Float)
    pnl_amount = Column(Float)  # Based on position size
    
    # Additional metadata
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class PortfolioPerformance(Base):
    """Database model for portfolio performance tracking"""
    __tablename__ = 'portfolio_performance'
    
    id = Column(String, primary_key=True)
    date = Column(DateTime, nullable=False)
    market_type = Column(String, nullable=False)
    trading_style = Column(String, nullable=False)
    
    # Daily performance metrics
    total_signals = Column(Integer, default=0)
    successful_trades = Column(Integer, default=0)
    failed_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    
    # P&L metrics
    daily_pnl = Column(Float, default=0.0)
    cumulative_pnl = Column(Float, default=0.0)
    avg_win = Column(Float, default=0.0)
    avg_loss = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)
    
    # Risk metrics
    max_drawdown = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    
    created_at = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    """Manages database operations for trading signals and performance"""
    
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        # Fix URL encoding issues for passwords with special characters
        self.database_url = self._fix_connection_url(self.database_url)
        
        # Create engine and session
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables
        self.create_tables()
    
    def _fix_connection_url(self, url):
        """Fix URL encoding issues in database connection string"""
        try:
            # Check if this is a Supabase connection with the specific issue
            if 'Aipass@12@aws' in url:
                # Build the correct URL from known components
                user = 'postgres.jnvtzqhcznmtspojahec'
                password = 'Aipass@12'
                encoded_password = quote_plus(password)  # This becomes Aipass%4012
                host = 'aws-0-ap-southeast-1.pooler.supabase.com'
                port = '6543'
                database = 'postgres'
                
                # Construct the proper URL
                fixed_url = f"postgresql://{user}:{encoded_password}@{host}:{port}/{database}?sslmode=require"
                return fixed_url
            
            # For other cases, try to parse and fix
            if '@' in url and url.count('@') > 1:
                # Multiple @ symbols detected, likely password encoding issue
                parts = url.split('://')
                if len(parts) == 2:
                    protocol = parts[0]
                    rest = parts[1]
                    
                    # Find the last @ which should be before hostname
                    last_at = rest.rfind('@')
                    if last_at > 0:
                        auth_part = rest[:last_at]
                        host_part = rest[last_at+1:]
                        
                        # Split auth into user:password
                        if ':' in auth_part:
                            user, password = auth_part.split(':', 1)
                            encoded_password = quote_plus(password)
                            fixed_url = f"{protocol}://{user}:{encoded_password}@{host_part}"
                            
                            # Add SSL if not present and this is Supabase
                            if 'supabase.com' in fixed_url and 'sslmode' not in fixed_url:
                                if '?' in fixed_url:
                                    fixed_url += '&sslmode=require'
                                else:
                                    fixed_url += '?sslmode=require'
                            
                            return fixed_url
            
            # If no issues detected, return original URL
            return url
            
        except Exception as e:
            # If anything goes wrong, return original URL
            return url
    
    def create_tables(self):
        """Create database tables if they don't exist"""
        try:
            Base.metadata.create_all(bind=self.engine)
        except Exception as e:
            st.error(f"Error creating database tables: {str(e)}")
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def save_signal(self, signal_data):
        """Save a trading signal to database"""
        session = self.get_session()
        try:
            # Generate UUID for the signal
            signal_id = str(uuid.uuid4())
            
            # Helper function to convert numpy types to native Python types
            def convert_numpy_type(value):
                if value is None:
                    return None
                if hasattr(value, 'item'):  # numpy scalar
                    return value.item()
                if isinstance(value, (np.integer, np.floating)):
                    return value.item()
                return value
            
            # Convert signal data to database model
            signal = TradingSignal(
                id=signal_id,
                symbol=signal_data['symbol'],
                market_type=signal_data.get('market_type', 'stocks'),
                trading_style=signal_data.get('trading_style', 'intraday'),
                action=signal_data['action'],
                strategy=signal_data['strategy'],
                signal_price=convert_numpy_type(signal_data.get('price', signal_data.get('signal_price'))),
                stop_loss=convert_numpy_type(signal_data.get('stop_loss')),
                target_price=convert_numpy_type(signal_data.get('target')),
                confidence=convert_numpy_type(signal_data.get('confidence', 0.0)),
                risk_reward=convert_numpy_type(signal_data.get('risk_reward')),
                timeframe=signal_data.get('timeframe'),
                signal_timestamp=signal_data.get('timestamp', signal_data.get('signal_timestamp')),
                shares=convert_numpy_type(signal_data.get('shares')),
                position_value=convert_numpy_type(signal_data.get('position_value')),
                is_executed=signal_data.get('is_executed', False),
                execution_price=convert_numpy_type(signal_data.get('execution_price')),
                execution_timestamp=signal_data.get('execution_timestamp'),
                notes=signal_data.get('notes')
            )
            
            session.add(signal)
            session.commit()
            return signal.id
        except Exception as e:
            session.rollback()
            # Use print instead of st.error to handle both contexts
            print(f"Error saving signal: {str(e)}")
            try:
                st.error(f"Error saving signal: {str(e)}")
            except:
                pass  # Streamlit context not available
            return None
        finally:
            session.close()
    
    def get_signals(self, market_type=None, trading_style=None, limit=100):
        """Retrieve trading signals from database"""
        session = self.get_session()
        try:
            query = session.query(TradingSignal)
            
            if market_type:
                query = query.filter(TradingSignal.market_type == market_type)
            if trading_style:
                query = query.filter(TradingSignal.trading_style == trading_style)
            
            signals = query.order_by(TradingSignal.signal_timestamp.desc()).limit(limit).all()
            return signals
        except Exception as e:
            st.error(f"Error retrieving signals: {str(e)}")
            return []
        finally:
            session.close()
    
    def update_signal_execution(self, signal_id, execution_price, execution_timestamp=None):
        """Update signal with execution details"""
        session = self.get_session()
        try:
            signal = session.query(TradingSignal).filter(TradingSignal.id == signal_id).first()
            if signal:
                signal.is_executed = True
                signal.execution_price = execution_price
                signal.execution_timestamp = execution_timestamp or datetime.utcnow()
                session.commit()
                return True
        except Exception as e:
            session.rollback()
            st.error(f"Error updating signal execution: {str(e)}")
            return False
        finally:
            session.close()
    
    def close_signal(self, signal_id, close_price, close_reason, close_timestamp=None):
        """Close a signal and calculate P&L"""
        session = self.get_session()
        try:
            signal = session.query(TradingSignal).filter(TradingSignal.id == signal_id).first()
            if signal and signal.is_executed:
                signal.is_closed = True
                signal.close_price = close_price
                signal.close_reason = close_reason
                signal.close_timestamp = close_timestamp or datetime.utcnow()
                
                # Calculate P&L
                if signal.action == 'BUY':
                    signal.pnl_points = close_price - signal.execution_price
                    signal.pnl_percentage = (close_price - signal.execution_price) / signal.execution_price * 100
                else:  # SELL
                    signal.pnl_points = signal.execution_price - close_price
                    signal.pnl_percentage = (signal.execution_price - close_price) / signal.execution_price * 100
                
                # Calculate P&L amount (assuming standard position size)
                position_size = 10000  # Default position size
                signal.pnl_amount = (signal.pnl_percentage / 100) * position_size
                
                session.commit()
                return True
        except Exception as e:
            session.rollback()
            st.error(f"Error closing signal: {str(e)}")
            return False
        finally:
            session.close()
    
    def simulate_signal_performance(self, data_fetcher, days_back=30):
        """Simulate performance of historical signals using actual price data"""
        session = self.get_session()
        try:
            # Get signals from last N days
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            signals = session.query(TradingSignal).filter(
                TradingSignal.signal_timestamp >= cutoff_date,
                TradingSignal.is_closed == False
            ).all()
            
            updated_count = 0
            
            for signal in signals:
                try:
                    # Get historical data for the symbol
                    symbol = signal.symbol
                    signal_date = signal.signal_timestamp
                    
                    # Get price data from signal date to now
                    if signal.market_type == 'crypto':
                        # Use crypto data fetcher
                        historical_data = data_fetcher.get_historical_data(symbol, period='1mo')
                    else:
                        # Use stock data fetcher
                        historical_data = data_fetcher.get_historical_data(symbol, period='1mo')
                    
                    if historical_data is not None and len(historical_data) > 0:
                        # Filter data from signal timestamp
                        historical_data = historical_data[historical_data.index >= signal_date]
                        
                        if len(historical_data) > 0:
                            # Simulate execution at signal price (or next available price)
                            execution_price = signal.signal_price
                            
                            # Check if target or stop loss was hit
                            close_price = None
                            close_reason = None
                            close_timestamp = None
                            
                            for timestamp, row in historical_data.iterrows():
                                if signal.action == 'BUY':
                                    # Check if target hit
                                    if signal.target_price and row['High'] >= signal.target_price:
                                        close_price = signal.target_price
                                        close_reason = 'target'
                                        close_timestamp = timestamp
                                        break
                                    # Check if stop loss hit
                                    elif signal.stop_loss and row['Low'] <= signal.stop_loss:
                                        close_price = signal.stop_loss
                                        close_reason = 'stop_loss'
                                        close_timestamp = timestamp
                                        break
                                else:  # SELL
                                    # Check if target hit (price going down)
                                    if signal.target_price and row['Low'] <= signal.target_price:
                                        close_price = signal.target_price
                                        close_reason = 'target'
                                        close_timestamp = timestamp
                                        break
                                    # Check if stop loss hit (price going up)
                                    elif signal.stop_loss and row['High'] >= signal.stop_loss:
                                        close_price = signal.stop_loss
                                        close_reason = 'stop_loss'
                                        close_timestamp = timestamp
                                        break
                            
                            # If no target/stop hit, use current price
                            if close_price is None:
                                close_price = historical_data['Close'].iloc[-1]
                                close_reason = 'current'
                                close_timestamp = historical_data.index[-1]
                            
                            # Update signal with execution and close
                            if not signal.is_executed:
                                signal.is_executed = True
                                signal.execution_price = execution_price
                                signal.execution_timestamp = signal_date
                            
                            # Close the signal
                            signal.is_closed = True
                            signal.close_price = close_price
                            signal.close_reason = close_reason
                            signal.close_timestamp = close_timestamp
                            
                            # Calculate P&L
                            if signal.action == 'BUY':
                                signal.pnl_points = close_price - execution_price
                                signal.pnl_percentage = (close_price - execution_price) / execution_price * 100
                            else:  # SELL
                                signal.pnl_points = execution_price - close_price
                                signal.pnl_percentage = (execution_price - close_price) / execution_price * 100
                            
                            # Calculate P&L amount
                            position_size = 10000  # Default position size
                            signal.pnl_amount = (signal.pnl_percentage / 100) * position_size
                            
                            updated_count += 1
                
                except Exception as e:
                    continue
            
            session.commit()
            return updated_count
            
        except Exception as e:
            session.rollback()
            st.error(f"Error simulating performance: {str(e)}")
            return 0
        finally:
            session.close()
    
    def get_performance_metrics(self, market_type=None, trading_style=None, days_back=30):
        """Calculate comprehensive performance metrics"""
        session = self.get_session()
        try:
            # Get closed signals
            query = session.query(TradingSignal).filter(TradingSignal.is_closed == True)
            
            if market_type:
                query = query.filter(TradingSignal.market_type == market_type)
            if trading_style:
                query = query.filter(TradingSignal.trading_style == trading_style)
            
            if days_back:
                cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                query = query.filter(TradingSignal.signal_timestamp >= cutoff_date)
            
            signals = query.all()
            
            if not signals:
                return None
            
            # Calculate metrics
            total_trades = len(signals)
            winning_trades = [s for s in signals if s.pnl_percentage > 0]
            losing_trades = [s for s in signals if s.pnl_percentage <= 0]
            
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
            
            # P&L metrics
            total_pnl = sum(s.pnl_amount for s in signals if s.pnl_amount)
            avg_win = np.mean([s.pnl_amount for s in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([s.pnl_amount for s in losing_trades]) if losing_trades else 0
            
            # Risk metrics
            profit_factor = abs(avg_win * win_count / (avg_loss * loss_count)) if avg_loss != 0 and loss_count > 0 else float('inf')
            
            # Calculate drawdown
            pnl_series = [s.pnl_amount for s in signals if s.pnl_amount]
            if pnl_series:
                cumulative_pnl = np.cumsum(pnl_series)
                running_max = np.maximum.accumulate(cumulative_pnl)
                drawdown = (cumulative_pnl - running_max)
                max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
            else:
                max_drawdown = 0
            
            return {
                'total_trades': total_trades,
                'win_count': win_count,
                'loss_count': loss_count,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'signals': signals
            }
            
        except Exception as e:
            st.error(f"Error calculating performance metrics: {str(e)}")
            return None
        finally:
            session.close()
    
    def get_daily_performance(self, market_type=None, trading_style=None, days_back=30):
        """Get daily performance breakdown"""
        session = self.get_session()
        try:
            query = session.query(TradingSignal).filter(TradingSignal.is_closed == True)
            
            if market_type:
                query = query.filter(TradingSignal.market_type == market_type)
            if trading_style:
                query = query.filter(TradingSignal.trading_style == trading_style)
            
            if days_back:
                cutoff_date = datetime.utcnow() - timedelta(days=days_back)
                query = query.filter(TradingSignal.signal_timestamp >= cutoff_date)
            
            signals = query.all()
            
            # Group by date
            daily_data = {}
            for signal in signals:
                date = signal.signal_timestamp.date()
                if date not in daily_data:
                    daily_data[date] = {
                        'trades': [],
                        'pnl': 0,
                        'wins': 0,
                        'losses': 0
                    }
                
                daily_data[date]['trades'].append(signal)
                daily_data[date]['pnl'] += signal.pnl_amount or 0
                
                if signal.pnl_percentage > 0:
                    daily_data[date]['wins'] += 1
                else:
                    daily_data[date]['losses'] += 1
            
            return daily_data
            
        except Exception as e:
            st.error(f"Error getting daily performance: {str(e)}")
            return {}
        finally:
            session.close()