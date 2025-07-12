import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
from database import DatabaseManager

class PerformanceAnalyzer:
    """Analyzes trading performance and generates detailed reports"""
    
    def __init__(self):
        self.db = DatabaseManager()
    
    def update_signal_performance(self, data_fetcher, crypto_data_fetcher):
        """Update performance of historical signals using current market data"""
        st.info("Updating signal performance with latest market data...")
        
        # Simulate performance for stocks
        stock_updates = self.db.simulate_signal_performance(data_fetcher, days_back=30)
        
        # Simulate performance for crypto
        crypto_updates = self.db.simulate_signal_performance(crypto_data_fetcher, days_back=30)
        
        total_updates = stock_updates + crypto_updates
        st.success(f"Updated performance for {total_updates} signals")
        
        return total_updates
    
    def display_performance_overview(self, market_type=None, trading_style=None, days_back=30):
        """Display overall performance metrics"""
        metrics = self.db.get_performance_metrics(market_type, trading_style, days_back)
        
        if not metrics:
            st.info("No trading data available for the selected period.")
            return
        
        # Key metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Trades", 
                metrics['total_trades'],
                help="Total number of completed trades"
            )
        
        with col2:
            win_rate_color = "normal"
            if metrics['win_rate'] >= 60:
                win_rate_color = "normal"
            st.metric(
                "Win Rate", 
                f"{metrics['win_rate']:.1f}%",
                help="Percentage of profitable trades"
            )
        
        with col3:
            pnl_delta = None
            if metrics['total_pnl'] > 0:
                pnl_delta = f"+{metrics['total_pnl']:.2f}"
            st.metric(
                "Total P&L", 
                f"${metrics['total_pnl']:.2f}" if market_type == 'crypto' else f"₹{metrics['total_pnl']:.2f}",
                delta=pnl_delta,
                help="Total profit/loss from all trades"
            )
        
        with col4:
            st.metric(
                "Profit Factor", 
                f"{metrics['profit_factor']:.2f}",
                help="Ratio of gross profit to gross loss"
            )
        
        # Additional metrics
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            currency = "$" if market_type == 'crypto' else "₹"
            st.metric(
                "Avg Win", 
                f"{currency}{metrics['avg_win']:.2f}",
                help="Average profit per winning trade"
            )
        
        with col6:
            st.metric(
                "Avg Loss", 
                f"{currency}{metrics['avg_loss']:.2f}",
                help="Average loss per losing trade"
            )
        
        with col7:
            st.metric(
                "Max Drawdown", 
                f"{currency}{metrics['max_drawdown']:.2f}",
                help="Maximum peak-to-trough decline"
            )
        
        with col8:
            expectancy = (metrics['win_rate']/100 * metrics['avg_win']) + ((100-metrics['win_rate'])/100 * metrics['avg_loss'])
            st.metric(
                "Expectancy", 
                f"{currency}{expectancy:.2f}",
                help="Expected profit per trade"
            )
    
    def display_trade_history(self, market_type=None, trading_style=None, days_back=30):
        """Display detailed trade history"""
        metrics = self.db.get_performance_metrics(market_type, trading_style, days_back)
        
        if not metrics or not metrics['signals']:
            st.info("No trade history available.")
            return
        
        # Convert signals to DataFrame for display
        trades_data = []
        for signal in metrics['signals']:
            symbol_display = signal.symbol.replace('.NS', '').replace('-USD', '')
            
            trades_data.append({
                'Date': signal.signal_timestamp.strftime('%Y-%m-%d %H:%M'),
                'Symbol': symbol_display,
                'Action': signal.action,
                'Strategy': signal.strategy,
                'Entry Price': signal.execution_price,
                'Exit Price': signal.close_price,
                'P&L %': f"{signal.pnl_percentage:.2f}%",
                'P&L Amount': signal.pnl_amount,
                'Reason': signal.close_reason,
                'Duration': self._calculate_duration(signal.execution_timestamp, signal.close_timestamp)
            })
        
        df = pd.DataFrame(trades_data)
        
        # Color code based on P&L
        def highlight_pnl(row):
            if row['P&L Amount'] > 0:
                return ['background-color: #d4edda'] * len(row)
            elif row['P&L Amount'] < 0:
                return ['background-color: #f8d7da'] * len(row)
            else:
                return [''] * len(row)
        
        # Display with styling
        st.subheader("📋 Trade History")
        
        # Add filters
        col1, col2, col3 = st.columns(3)
        with col1:
            strategy_filter = st.selectbox(
                "Filter by Strategy",
                ["All"] + list(df['Strategy'].unique()) if not df.empty else ["All"]
            )
        with col2:
            action_filter = st.selectbox("Filter by Action", ["All", "BUY", "SELL"])
        with col3:
            outcome_filter = st.selectbox("Filter by Outcome", ["All", "Winning", "Losing"])
        
        # Apply filters
        filtered_df = df.copy()
        if strategy_filter != "All":
            filtered_df = filtered_df[filtered_df['Strategy'] == strategy_filter]
        if action_filter != "All":
            filtered_df = filtered_df[filtered_df['Action'] == action_filter]
        if outcome_filter == "Winning":
            filtered_df = filtered_df[filtered_df['P&L Amount'] > 0]
        elif outcome_filter == "Losing":
            filtered_df = filtered_df[filtered_df['P&L Amount'] <= 0]
        
        # Display the table
        if not filtered_df.empty:
            st.dataframe(
                filtered_df.style.apply(highlight_pnl, axis=1),
                use_container_width=True,
                height=400
            )
        else:
            st.info("No trades match the selected filters.")
    
    def display_performance_charts(self, market_type=None, trading_style=None, days_back=30):
        """Display performance visualization charts"""
        metrics = self.db.get_performance_metrics(market_type, trading_style, days_back)
        daily_performance = self.db.get_daily_performance(market_type, trading_style, days_back)
        
        if not metrics:
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Cumulative P&L Over Time',
                'Win/Loss Distribution',
                'Daily P&L',
                'Strategy Performance'
            ),
            specs=[[{"secondary_y": True}, {"type": "pie"}],
                   [{"secondary_y": False}, {"type": "bar"}]]
        )
        
        # 1. Cumulative P&L chart
        if metrics['signals']:
            signals_sorted = sorted(metrics['signals'], key=lambda x: x.signal_timestamp)
            cumulative_pnl = np.cumsum([s.pnl_amount for s in signals_sorted if s.pnl_amount])
            dates = [s.signal_timestamp for s in signals_sorted if s.pnl_amount]
            
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=cumulative_pnl,
                    mode='lines',
                    name='Cumulative P&L',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
        
        # 2. Win/Loss pie chart
        fig.add_trace(
            go.Pie(
                labels=['Wins', 'Losses'],
                values=[metrics['win_count'], metrics['loss_count']],
                hole=0.4,
                marker_colors=['green', 'red']
            ),
            row=1, col=2
        )
        
        # 3. Daily P&L chart
        if daily_performance:
            dates = list(daily_performance.keys())
            daily_pnl = [daily_performance[date]['pnl'] for date in dates]
            colors = ['green' if pnl >= 0 else 'red' for pnl in daily_pnl]
            
            fig.add_trace(
                go.Bar(
                    x=dates,
                    y=daily_pnl,
                    name='Daily P&L',
                    marker_color=colors
                ),
                row=2, col=1
            )
        
        # 4. Strategy performance
        strategy_performance = {}
        for signal in metrics['signals']:
            strategy = signal.strategy
            if strategy not in strategy_performance:
                strategy_performance[strategy] = {'pnl': 0, 'trades': 0}
            strategy_performance[strategy]['pnl'] += signal.pnl_amount or 0
            strategy_performance[strategy]['trades'] += 1
        
        strategies = list(strategy_performance.keys())
        strategy_pnl = [strategy_performance[s]['pnl'] for s in strategies]
        strategy_colors = ['green' if pnl >= 0 else 'red' for pnl in strategy_pnl]
        
        fig.add_trace(
            go.Bar(
                x=strategies,
                y=strategy_pnl,
                name='Strategy P&L',
                marker_color=strategy_colors
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=700,
            showlegend=False,
            title_text="Performance Analysis Dashboard"
        )
        
        # Update axes
        currency = "$" if market_type == 'crypto' else "₹"
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text=f"Cumulative P&L ({currency})", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text=f"Daily P&L ({currency})", row=2, col=1)
        fig.update_xaxes(title_text="Strategy", row=2, col=2)
        fig.update_yaxes(title_text=f"Total P&L ({currency})", row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_strategy_comparison(self, market_type=None, trading_style=None, days_back=30):
        """Display detailed strategy comparison"""
        metrics = self.db.get_performance_metrics(market_type, trading_style, days_back)
        
        if not metrics or not metrics['signals']:
            return
        
        # Group by strategy
        strategy_data = {}
        for signal in metrics['signals']:
            strategy = signal.strategy
            if strategy not in strategy_data:
                strategy_data[strategy] = {
                    'trades': [],
                    'total_pnl': 0,
                    'wins': 0,
                    'losses': 0,
                    'avg_win': 0,
                    'avg_loss': 0,
                    'win_rate': 0
                }
            
            strategy_data[strategy]['trades'].append(signal)
            strategy_data[strategy]['total_pnl'] += signal.pnl_amount or 0
            
            if signal.pnl_percentage > 0:
                strategy_data[strategy]['wins'] += 1
            else:
                strategy_data[strategy]['losses'] += 1
        
        # Calculate metrics for each strategy
        for strategy in strategy_data:
            data = strategy_data[strategy]
            total_trades = len(data['trades'])
            
            if total_trades > 0:
                data['win_rate'] = (data['wins'] / total_trades) * 100
                
                winning_trades = [t for t in data['trades'] if t.pnl_amount > 0]
                losing_trades = [t for t in data['trades'] if t.pnl_amount <= 0]
                
                data['avg_win'] = np.mean([t.pnl_amount for t in winning_trades]) if winning_trades else 0
                data['avg_loss'] = np.mean([t.pnl_amount for t in losing_trades]) if losing_trades else 0
        
        # Create comparison table
        comparison_data = []
        for strategy, data in strategy_data.items():
            comparison_data.append({
                'Strategy': strategy,
                'Total Trades': len(data['trades']),
                'Win Rate (%)': f"{data['win_rate']:.1f}%",
                'Total P&L': data['total_pnl'],
                'Avg Win': data['avg_win'],
                'Avg Loss': data['avg_loss'],
                'Profit Factor': abs(data['avg_win'] * data['wins'] / (data['avg_loss'] * data['losses'])) if data['avg_loss'] != 0 and data['losses'] > 0 else float('inf')
            })
        
        df = pd.DataFrame(comparison_data)
        
        if not df.empty:
            st.subheader("📊 Strategy Performance Comparison")
            
            # Sort by total P&L
            df_sorted = df.sort_values('Total P&L', ascending=False)
            
            # Display table with color coding
            def highlight_best_strategy(row):
                if row.name == 0:  # Best strategy (highest P&L)
                    return ['background-color: #d4edda'] * len(row)
                return [''] * len(row)
            
            st.dataframe(
                df_sorted.style.apply(highlight_best_strategy, axis=1),
                use_container_width=True
            )
            
            # Strategy recommendations
            best_strategy = df_sorted.iloc[0]['Strategy']
            best_win_rate = df_sorted.iloc[0]['Win Rate (%)']
            
            st.success(f"🏆 **Best Performing Strategy:** {best_strategy}")
            st.info(f"📈 **Highest Win Rate:** {best_win_rate}")
    
    def _calculate_duration(self, start_time, end_time):
        """Calculate duration between two timestamps"""
        if not start_time or not end_time:
            return "N/A"
        
        duration = end_time - start_time
        
        if duration.days > 0:
            return f"{duration.days}d {duration.seconds//3600}h"
        elif duration.seconds >= 3600:
            return f"{duration.seconds//3600}h {(duration.seconds%3600)//60}m"
        else:
            return f"{duration.seconds//60}m"
    
    def export_performance_report(self, market_type=None, trading_style=None, days_back=30):
        """Export performance report as CSV"""
        metrics = self.db.get_performance_metrics(market_type, trading_style, days_back)
        
        if not metrics or not metrics['signals']:
            st.warning("No data available for export.")
            return
        
        # Prepare data for export
        export_data = []
        for signal in metrics['signals']:
            export_data.append({
                'Date': signal.signal_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'Symbol': signal.symbol,
                'Market': signal.market_type,
                'Style': signal.trading_style,
                'Action': signal.action,
                'Strategy': signal.strategy,
                'Entry_Price': signal.execution_price,
                'Exit_Price': signal.close_price,
                'Stop_Loss': signal.stop_loss,
                'Target': signal.target_price,
                'PnL_Percentage': signal.pnl_percentage,
                'PnL_Amount': signal.pnl_amount,
                'Close_Reason': signal.close_reason,
                'Confidence': signal.confidence,
                'Risk_Reward': signal.risk_reward
            })
        
        df = pd.DataFrame(export_data)
        
        # Convert to CSV
        csv = df.to_csv(index=False)
        
        # Create download button
        st.download_button(
            label="📥 Download Performance Report (CSV)",
            data=csv,
            file_name=f"trading_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )