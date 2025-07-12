import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
import streamlit as st

class CryptoDataFetcher:
    """Handles fetching real-time cryptocurrency data"""
    
    def __init__(self):
        # Major cryptocurrency symbols (using Yahoo Finance format)
        self.crypto_symbols = [
            'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD',
            'SOL-USD', 'DOGE-USD', 'DOT-USD', 'AVAX-USD', 'SHIB-USD',
            'MATIC-USD', 'LTC-USD', 'UNI-USD', 'LINK-USD', 'ALGO-USD',
            'BCH-USD', 'XLM-USD', 'VET-USD', 'FIL-USD', 'TRX-USD',
            'ETC-USD', 'THETA-USD', 'AAVE-USD', 'ATOM-USD', 'XTZ-USD'
        ]
        
        # Alternative symbols for popular crypto pairs
        self.crypto_pairs = {
            'BTC-USD': 'Bitcoin',
            'ETH-USD': 'Ethereum',
            'BNB-USD': 'Binance Coin',
            'XRP-USD': 'Ripple',
            'ADA-USD': 'Cardano',
            'SOL-USD': 'Solana',
            'DOGE-USD': 'Dogecoin',
            'DOT-USD': 'Polkadot',
            'AVAX-USD': 'Avalanche',
            'MATIC-USD': 'Polygon'
        }
        
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = 60  # 1 minute cache for crypto (faster market)
    
    def get_crypto_symbols(self):
        """Get list of cryptocurrency symbols"""
        return self.crypto_symbols
    
    def is_crypto_market_open(self):
        """Crypto market is always open (24/7)"""
        return True
    
    def clean_symbol(self, symbol):
        """Clean symbol for crypto compatibility"""
        if not symbol.endswith('-USD'):
            symbol = f"{symbol}-USD"
        return symbol.upper()
    
    def get_real_time_price(self, symbol):
        """Get real-time price for a single cryptocurrency"""
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
                    'market_cap': info.get('marketCap', 0),
                    'timestamp': datetime.now()
                }
            else:
                return None
                
        except Exception as e:
            st.warning(f"Error fetching real-time price for {symbol}: {str(e)}")
            return None
    
    def get_intraday_data(self, symbol, period='1d', interval='5m'):
        """Get intraday data for cryptocurrency with caching"""
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
    
    def get_crypto_info(self, symbol):
        """Get detailed cryptocurrency information"""
        try:
            symbol = self.clean_symbol(symbol)
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'name': self.crypto_pairs.get(symbol, symbol),
                'market_cap': info.get('marketCap', 0),
                'avg_volume': info.get('averageVolume', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0),
                'circulating_supply': info.get('circulatingSupply', 0),
                'total_supply': info.get('totalSupply', 0)
            }
            
        except Exception as e:
            st.warning(f"Error fetching crypto info for {symbol}: {str(e)}")
            return None
    
    def get_multiple_cryptos_data(self, symbols, period='1d', interval='5m'):
        """Get data for multiple cryptocurrencies efficiently"""
        results = {}
        
        for symbol in symbols:
            try:
                data = self.get_intraday_data(symbol, period, interval)
                if data is not None:
                    results[symbol] = data
                
                # Smaller delay for crypto (faster market)
                time.sleep(0.05)
                
            except Exception as e:
                st.warning(f"Error fetching data for {symbol}: {str(e)}")
                continue
        
        return results
    
    def get_market_status(self):
        """Get current crypto market status"""
        try:
            # Fetch Bitcoin as market indicator
            btc = yf.Ticker('BTC-USD')
            btc_data = btc.history(period='1d', interval='1m')
            
            if btc_data.empty:
                return None
            
            current_price = btc_data['Close'].iloc[-1]
            prev_close = btc_data['Close'].iloc[0]
            change = current_price - prev_close
            change_percent = (change / prev_close) * 100
            
            return {
                'index': 'Bitcoin (Market Leader)',
                'current': current_price,
                'change': change,
                'change_percent': change_percent,
                'is_open': True,  # Crypto market always open
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            st.warning(f"Error fetching market status: {str(e)}")
            return {
                'index': 'Bitcoin (Market Leader)',
                'current': 0,
                'change': 0,
                'change_percent': 0,
                'is_open': True,
                'timestamp': datetime.now()
            }
    
    def get_top_gainers_losers(self, limit=10):
        """Get top gainers and losers from tracked cryptocurrencies"""
        try:
            gainers = []
            losers = []
            
            for symbol in self.crypto_symbols[:20]:  # Check top 20 cryptos
                try:
                    price_data = self.get_real_time_price(symbol)
                    if price_data and price_data['change_percent'] != 0:
                        if price_data['change_percent'] > 0:
                            gainers.append(price_data)
                        else:
                            losers.append(price_data)
                    
                    time.sleep(0.05)  # Rate limiting
                    
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
    
    def get_crypto_fear_greed_index(self):
        """Get crypto fear and greed index (if available)"""
        try:
            # This would require an external API like Alternative.me
            # For now, return a placeholder
            return {
                'value': 50,
                'classification': 'Neutral',
                'timestamp': datetime.now()
            }
        except:
            return None