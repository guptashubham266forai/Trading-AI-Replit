import asyncio
import websocket
import json
import threading
from datetime import datetime
import requests
import streamlit as st

class RealtimeCryptoData:
    """Real-time cryptocurrency data using Binance WebSocket API"""
    
    def __init__(self):
        self.binance_ws_url = "wss://stream.binance.com:9443/ws/"
        self.price_data = {}
        self.callbacks = []
        self.ws = None
        self.running = False
        
        # Symbol mapping from Yahoo Finance format to Binance format
        self.symbol_mapping = {
            'BTC-USD': 'btcusdt',
            'ETH-USD': 'ethusdt',
            'BNB-USD': 'bnbusdt',
            'XRP-USD': 'xrpusdt',
            'ADA-USD': 'adausdt',
            'SOL-USD': 'solusdt',
            'DOGE-USD': 'dogeusdt',
            'DOT-USD': 'dotusdt',
            'AVAX-USD': 'avaxusdt',
            'LTC-USD': 'ltcusdt',
            'LINK-USD': 'linkusdt',
            'BCH-USD': 'bchusdt',
            'ATOM-USD': 'atomusdt',
            'NEAR-USD': 'nearusdt'
        }
    
    def get_binance_symbol(self, yahoo_symbol):
        """Convert Yahoo Finance symbol to Binance symbol"""
        return self.symbol_mapping.get(yahoo_symbol, yahoo_symbol.replace('-USD', 'usdt').lower())
    
    def get_realtime_price_rest(self, symbol):
        """Get real-time price using Binance REST API (faster than Yahoo Finance)"""
        try:
            binance_symbol = self.get_binance_symbol(symbol)
            url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={binance_symbol.upper()}"
            
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return {
                    'symbol': symbol,
                    'price': float(data['lastPrice']),
                    'change': float(data['priceChange']),
                    'change_percent': float(data['priceChangePercent']),
                    'volume': float(data['volume']),
                    'timestamp': datetime.now(),
                    'source': 'binance_realtime'
                }
            else:
                return None
        except Exception as e:
            print(f"Binance API error for {symbol}: {e}")
            return None
    
    def get_multiple_prices_fast(self, symbols):
        """Get multiple prices in a single API call (much faster)"""
        try:
            # Convert symbols to Binance format
            binance_symbols = [f'"{self.get_binance_symbol(s).upper()}"' for s in symbols[:20]]
            symbols_param = '[' + ','.join(binance_symbols) + ']'
            
            url = f"https://api.binance.com/api/v3/ticker/24hr?symbols={symbols_param}"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                results = {}
                
                for item in data:
                    # Find matching original symbol
                    binance_symbol = item['symbol'].lower()
                    original_symbol = None
                    for orig, binance in self.symbol_mapping.items():
                        if binance == binance_symbol:
                            original_symbol = orig
                            break
                    
                    if original_symbol:
                        results[original_symbol] = {
                            'symbol': original_symbol,
                            'price': float(item['lastPrice']),
                            'change': float(item['priceChange']),
                            'change_percent': float(item['priceChangePercent']),
                            'volume': float(item['volume']),
                            'timestamp': datetime.now(),
                            'source': 'binance_batch'
                        }
                
                return results
            else:
                return {}
                
        except Exception as e:
            print(f"Binance batch API error: {e}")
            return {}
    
    def start_websocket_stream(self, symbols):
        """Start WebSocket stream for real-time updates"""
        if self.running:
            return
        
        try:
            # Create stream names for multiple symbols
            streams = []
            for symbol in symbols[:10]:  # Limit to 10 symbols for WebSocket
                binance_symbol = self.get_binance_symbol(symbol)
                streams.append(f"{binance_symbol}@ticker")
            
            stream_names = '/'.join(streams)
            ws_url = f"{self.binance_ws_url}{stream_names}"
            
            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    if 'stream' in data:
                        ticker_data = data['data']
                        binance_symbol = ticker_data['s'].lower()
                        
                        # Find original symbol
                        original_symbol = None
                        for orig, binance in self.symbol_mapping.items():
                            if binance == binance_symbol:
                                original_symbol = orig
                                break
                        
                        if original_symbol:
                            price_info = {
                                'symbol': original_symbol,
                                'price': float(ticker_data['c']),
                                'change': float(ticker_data['P']),
                                'change_percent': float(ticker_data['P']),
                                'volume': float(ticker_data['v']),
                                'timestamp': datetime.now(),
                                'source': 'binance_websocket'
                            }
                            
                            self.price_data[original_symbol] = price_info
                            
                            # Trigger callbacks
                            for callback in self.callbacks:
                                try:
                                    callback(price_info)
                                except:
                                    pass
                                    
                except Exception as e:
                    print(f"WebSocket message error: {e}")
            
            def on_error(ws, error):
                print(f"WebSocket error: {error}")
            
            def on_close(ws, close_status_code, close_msg):
                print("WebSocket connection closed")
                self.running = False
            
            def on_open(ws):
                print("WebSocket connection opened")
                self.running = True
            
            # Start WebSocket in a separate thread
            def run_websocket():
                self.ws = websocket.WebSocketApp(
                    ws_url,
                    on_message=on_message,
                    on_error=on_error,
                    on_close=on_close,
                    on_open=on_open
                )
                self.ws.run_forever()
            
            thread = threading.Thread(target=run_websocket, daemon=True)
            thread.start()
            
        except Exception as e:
            print(f"WebSocket setup error: {e}")
    
    def stop_websocket_stream(self):
        """Stop WebSocket stream"""
        self.running = False
        if self.ws:
            self.ws.close()
    
    def add_price_callback(self, callback):
        """Add callback function for price updates"""
        self.callbacks.append(callback)
    
    def get_cached_price(self, symbol):
        """Get last cached price from WebSocket"""
        return self.price_data.get(symbol)
    
    def is_connected(self):
        """Check if WebSocket is connected"""
        return self.running and self.ws and self.ws.sock and self.ws.sock.connected