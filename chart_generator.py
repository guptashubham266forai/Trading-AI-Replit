import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

class SignalChartGenerator:
    """Generate candlestick charts with color-coded entry/exit signals"""
    
    def __init__(self):
        self.colors = {
            'BUY': '#00ff00',      # Green for buy signals
            'SELL': '#ff4444',     # Red for sell signals
            'stop_loss': '#ff6b6b', # Light red for stop loss
            'target': '#4CAF50',   # Dark green for target
            'entry': '#2196F3'     # Blue for entry point
        }
    
    def create_signal_chart(self, data, signal, symbol):
        """Create candlestick chart with signal visualization"""
        try:
            # Ensure we have enough data
            if len(data) < 20:
                return None
            
            # Take last 100 candles for better visualization
            chart_data = data.tail(100).copy()
            
            # Create subplots with secondary y-axis for volume
            fig = make_subplots(
                rows=2, cols=1,
                row_widths=[0.7, 0.3],
                vertical_spacing=0.05,
                subplot_titles=[f"{symbol} - {signal['strategy']}", "Volume"],
                specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
            )
            
            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=chart_data.index,
                    open=chart_data['Open'],
                    high=chart_data['High'],
                    low=chart_data['Low'],
                    close=chart_data['Close'],
                    name="Price",
                    increasing_line_color='#00ff88',
                    decreasing_line_color='#ff4444'
                ),
                row=1, col=1
            )
            
            # Add volume bars
            colors = ['#00ff88' if close >= open else '#ff4444' 
                     for close, open in zip(chart_data['Close'], chart_data['Open'])]
            
            fig.add_trace(
                go.Bar(
                    x=chart_data.index,
                    y=chart_data['Volume'],
                    name="Volume",
                    marker_color=colors,
                    opacity=0.6
                ),
                row=2, col=1
            )
            
            # Add technical indicators if available
            self.add_technical_indicators(fig, chart_data)
            
            # Add signal markers
            self.add_signal_markers(fig, chart_data, signal)
            
            # Add price levels (entry, stop loss, target)
            self.add_price_levels(fig, chart_data, signal)
            
            # Update layout
            fig.update_layout(
                title=f"{symbol.replace('.NS', '').replace('-USD', '')} - {signal['action']} Signal",
                xaxis_title="Time",
                yaxis_title="Price",
                xaxis_rangeslider_visible=False,
                height=600,
                showlegend=True,
                template="plotly_dark",
                font=dict(size=10)
            )
            
            # Update x-axes
            fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.3)')
            fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.3)')
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
            return None
    
    def add_technical_indicators(self, fig, data):
        """Add technical indicators to the chart"""
        # Moving averages
        if 'MA_20' in data.columns and 'MA_50' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MA_20'],
                    name="MA 20",
                    line=dict(color='orange', width=1),
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MA_50'],
                    name="MA 50",
                    line=dict(color='purple', width=1),
                    opacity=0.7
                ),
                row=1, col=1
            )
        
        # VWAP if available
        if 'VWAP' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['VWAP'],
                    name="VWAP",
                    line=dict(color='yellow', width=2, dash='dash'),
                    opacity=0.8
                ),
                row=1, col=1
            )
        
        # Bollinger Bands if available
        if all(col in data.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
            # Upper band
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Upper'],
                    name="BB Upper",
                    line=dict(color='gray', width=1),
                    opacity=0.5
                ),
                row=1, col=1
            )
            
            # Lower band
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Lower'],
                    name="BB Lower",
                    line=dict(color='gray', width=1),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.1)',
                    opacity=0.5
                ),
                row=1, col=1
            )
    
    def add_signal_markers(self, fig, data, signal):
        """Add entry/exit signal markers"""
        signal_time = signal.get('timestamp', data.index[-1])
        entry_price = signal.get('price', data['Close'].iloc[-1])
        
        # Find the closest time in our data
        closest_time_idx = None
        min_diff = float('inf')
        
        for i, timestamp in enumerate(data.index):
            if hasattr(timestamp, 'to_pydatetime'):
                timestamp = timestamp.to_pydatetime()
            if hasattr(signal_time, 'to_pydatetime'):
                signal_time = signal_time.to_pydatetime()
            
            diff = abs((timestamp - signal_time).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest_time_idx = i
        
        if closest_time_idx is not None:
            entry_time = data.index[closest_time_idx]
            
            # Entry point marker
            marker_color = self.colors['BUY'] if signal['action'] == 'BUY' else self.colors['SELL']
            marker_symbol = 'triangle-up' if signal['action'] == 'BUY' else 'triangle-down'
            
            fig.add_trace(
                go.Scatter(
                    x=[entry_time],
                    y=[entry_price],
                    mode='markers',
                    name=f"{signal['action']} Signal",
                    marker=dict(
                        symbol=marker_symbol,
                        size=15,
                        color=marker_color,
                        line=dict(color='white', width=2)
                    ),
                    hovertemplate=f"<b>{signal['action']} Signal</b><br>" +
                                f"Price: {entry_price:.2f}<br>" +
                                f"Strategy: {signal.get('strategy', 'N/A')}<br>" +
                                f"Confidence: {signal.get('confidence', 0):.1%}<extra></extra>"
                ),
                row=1, col=1
            )
    
    def add_price_levels(self, fig, data, signal):
        """Add horizontal lines for entry, stop loss, and target"""
        entry_price = signal.get('price', data['Close'].iloc[-1])
        stop_loss = signal.get('stop_loss')
        target = signal.get('target')
        
        x_range = [data.index[0], data.index[-1]]
        
        # Entry price line
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=[entry_price, entry_price],
                mode='lines',
                name=f"Entry: {entry_price:.2f}",
                line=dict(color=self.colors['entry'], width=2, dash='solid'),
                hovertemplate=f"Entry Price: {entry_price:.2f}<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Stop loss line
        if stop_loss:
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=[stop_loss, stop_loss],
                    mode='lines',
                    name=f"Stop Loss: {stop_loss:.2f}",
                    line=dict(color=self.colors['stop_loss'], width=2, dash='dash'),
                    hovertemplate=f"Stop Loss: {stop_loss:.2f}<extra></extra>"
                ),
                row=1, col=1
            )
        
        # Target line
        if target:
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=[target, target],
                    mode='lines',
                    name=f"Target: {target:.2f}",
                    line=dict(color=self.colors['target'], width=2, dash='dot'),
                    hovertemplate=f"Target: {target:.2f}<extra></extra>"
                ),
                row=1, col=1
            )
        
        # Add risk-reward zone shading
        if stop_loss and target:
            if signal['action'] == 'BUY':
                # Profit zone (green)
                fig.add_hrect(
                    y0=entry_price, y1=target,
                    fillcolor="rgba(76, 175, 80, 0.1)",
                    layer="below",
                    line_width=0,
                    row=1, col=1
                )
                # Risk zone (red)
                fig.add_hrect(
                    y0=stop_loss, y1=entry_price,
                    fillcolor="rgba(244, 67, 54, 0.1)",
                    layer="below",
                    line_width=0,
                    row=1, col=1
                )
            else:  # SELL signal
                # Profit zone (green)
                fig.add_hrect(
                    y0=target, y1=entry_price,
                    fillcolor="rgba(76, 175, 80, 0.1)",
                    layer="below",
                    line_width=0,
                    row=1, col=1
                )
                # Risk zone (red)
                fig.add_hrect(
                    y0=entry_price, y1=stop_loss,
                    fillcolor="rgba(244, 67, 54, 0.1)",
                    layer="below",
                    line_width=0,
                    row=1, col=1
                )
    
    def create_mini_chart(self, data, signal, symbol, height=400):
        """Create a smaller chart for signal display"""
        try:
            # Take last 50 candles for mini chart
            chart_data = data.tail(50).copy()
            
            # Create simple candlestick chart
            fig = go.Figure()
            
            # Add candlestick
            fig.add_trace(
                go.Candlestick(
                    x=chart_data.index,
                    open=chart_data['Open'],
                    high=chart_data['High'],
                    low=chart_data['Low'],
                    close=chart_data['Close'],
                    name="Price",
                    increasing_line_color='#00ff88',
                    decreasing_line_color='#ff4444'
                )
            )
            
            # Add signal point and price levels
            self.add_signal_markers(fig, chart_data, signal)
            self.add_price_levels(fig, chart_data, signal)
            
            # Add key indicators
            if 'VWAP' in chart_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=chart_data.index,
                        y=chart_data['VWAP'],
                        name="VWAP",
                        line=dict(color='yellow', width=1.5),
                        opacity=0.7
                    )
                )
            
            # Update layout for mini chart
            fig.update_layout(
                title=f"{symbol.replace('.NS', '').replace('-USD', '')} - {signal['action']}",
                xaxis_rangeslider_visible=False,
                height=height,
                showlegend=False,
                template="plotly_dark",
                font=dict(size=8),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.2)')
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating mini chart: {str(e)}")
            return None
    
    def save_chart_as_image(self, fig, filename):
        """Save chart as PNG image"""
        try:
            if fig:
                fig.write_image(filename, width=1200, height=800, scale=2)
                return filename
        except Exception as e:
            st.error(f"Error saving chart: {str(e)}")
        return None