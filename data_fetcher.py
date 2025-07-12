import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
import streamlit as st

class DataFetcher:
    """Handles fetching real-time NSE stock data"""
    
    def __init__(self):
        self.nse_symbols = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
            'HDFC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'KOTAKBANK.NS',
            'LT.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'HCLTECH.NS', 'AXISBANK.NS',
            'WIPRO.NS', 'ULTRACEMCO.NS', 'BAJFINANCE.NS', 'NESTLEIND.NS', 'ONGC.NS',
            'TATAMOTORS.NS', 'SUNPHARMA.NS', 'NTPC.NS', 'POWERGRID.NS', 'TATASTEEL.NS',
            'JSWSTEEL.NS', 'M&M.NS', 'TECHM.NS', 'TITAN.NS', 'INDUSINDBK.NS',
            'ADANIPORTS.NS', 'COALINDIA.NS', 'GRASIM.NS', 'HINDALCO.NS', 'DRREDDY.NS',
            'BAJAJFINSV.NS', 'CIPLA.NS', 'EICHERMOT.NS', 'HEROMOTOCO.NS', 'DIVISLAB.NS',
            'BRITANNIA.NS', 'BAJAJ-AUTO.NS', 'BPCL.NS', 'TATACONSUM.NS', 'UPL.NS',
            'APOLLOHOSP.NS', 'HDFCLIFE.NS', 'SBILIFE.NS', 'ADANIENT.NS', 'SHREECEM.NS'
        ]
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = 300  # 5 minutes cache
    
    def get_nse_symbols(self):
        """Get list of NSE symbols for major stocks"""
        return self.nse_symbols
    
    def is_market_open(self):
        """Check if NSE market is currently open"""
        now = datetime.now()
        
        # NSE trading hours: 9:15 AM to 3:30 PM IST, Monday to Friday
        if now.weekday() > 4:  # Saturday = 5, Sunday = 6
            return False
        
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def clean_symbol(self, symbol):
        """Clean symbol for yfinance compatibility"""
        if not symbol.endswith('.NS'):
            symbol = f"{symbol}.NS"
        return symbol
    
    def get_real_time_price(self, symbol):
        """Get real-time price for a single stock"""
        try:
            symbol = self.clean_symbol(symbol)
            ticker = yf.Ticker(symbol)
            
            # Get current data
            info = ticker.info
            if 'regularMarketPrice' in info:
                return {
                    'symbol': symbol,
                    'price': info['regularMarketPrice'],
                    'change': info.get('regularMarketChange', 0),
                    'change_percent': info.get('regularMarketChangePercent', 0),
                    'volume': info.get('regularMarketVolume', 0),
                    'timestamp': datetime.now()
                }
            else:
                return None
                
        except Exception as e:
            st.warning(f"Error fetching real-time price for {symbol}: {str(e)}")
            return None
    
    def get_intraday_data(self, symbol, period='1d', interval='5m'):
        """Get intraday data for a stock with caching"""
        try:
            symbol = self.clean_symbol(symbol)
            
            # Check cache
            cache_key = f"{symbol}_{period}_{interval}"
            current_time = datetime.now()
            
            if (cache_key in self.cache and 
                cache_key in self.cache_expiry and 
                current_time < self.cache_expiry[cache_key]):
                return self.cache[cache_key]
            
            # Fetch new data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                return None
            
            # Clean and prepare data
            data = data.dropna()
            data.index = pd.to_datetime(data.index)
            
            # Cache the data
            self.cache[cache_key] = data
            self.cache_expiry[cache_key] = current_time + timedelta(seconds=self.cache_duration)
            
            return data
            
        except Exception as e:
            st.warning(f"Error fetching intraday data for {symbol}: {str(e)}")
            return None
    
    def get_historical_data(self, symbol, period='1mo'):
        """Get historical data for technical analysis"""
        try:
            symbol = self.clean_symbol(symbol)
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                return None
            
            data = data.dropna()
            data.index = pd.to_datetime(data.index)
            
            return data
            
        except Exception as e:
            st.warning(f"Error fetching historical data for {symbol}: {str(e)}")
            return None
    
    def get_stock_info(self, symbol):
        """Get detailed stock information"""
        try:
            symbol = self.clean_symbol(symbol)
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'avg_volume': info.get('averageVolume', 0),
                'pe_ratio': info.get('trailingPE', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0),
                'dividend_yield': info.get('dividendYield', 0)
            }
            
        except Exception as e:
            st.warning(f"Error fetching stock info for {symbol}: {str(e)}")
            return None
    
    def get_multiple_stocks_data(self, symbols, period='1d', interval='5m'):
        """Get data for multiple stocks efficiently"""
        results = {}
        
        for symbol in symbols:
            try:
                data = self.get_intraday_data(symbol, period, interval)
                if data is not None:
                    results[symbol] = data
                
                # Add small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                st.warning(f"Error fetching data for {symbol}: {str(e)}")
                continue
        
        return results
    
    def get_market_status(self):
        """Get current market status and key indices"""
        try:
            # Fetch NIFTY 50 data as market indicator
            nifty = yf.Ticker('^NSEI')
            nifty_data = nifty.history(period='1d', interval='1m')
            
            if nifty_data.empty:
                return None
            
            current_price = nifty_data['Close'].iloc[-1]
            prev_close = nifty_data['Close'].iloc[0]
            change = current_price - prev_close
            change_percent = (change / prev_close) * 100
            
            return {
                'index': 'NIFTY 50',
                'current': current_price,
                'change': change,
                'change_percent': change_percent,
                'is_open': self.is_market_open(),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            st.warning(f"Error fetching market status: {str(e)}")
            return {
                'index': 'NIFTY 50',
                'current': 0,
                'change': 0,
                'change_percent': 0,
                'is_open': self.is_market_open(),
                'timestamp': datetime.now()
            }
    
    def get_top_gainers_losers(self, limit=10):
        """Get top gainers and losers from tracked stocks"""
        try:
            gainers = []
            losers = []
            
            for symbol in self.nse_symbols[:30]:  # Check top 30 stocks
                try:
                    price_data = self.get_real_time_price(symbol)
                    if price_data and price_data['change_percent'] != 0:
                        if price_data['change_percent'] > 0:
                            gainers.append(price_data)
                        else:
                            losers.append(price_data)
                    
                    time.sleep(0.1)  # Rate limiting
                    
                except:
                    continue
            
            # Sort and limit results
            gainers.sort(key=lambda x: x['change_percent'], reverse=True)
            losers.sort(key=lambda x: x['change_percent'])
            
            return {
                'gainers': gainers[:limit],
                'losers': losers[:limit]
            }
            
        except Exception as e:
            st.warning(f"Error fetching gainers/losers: {str(e)}")
            return {'gainers': [], 'losers': []}
