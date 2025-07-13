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
        # Get basic signal statistics
        session = self.db.get_session()
        from database import TradingSignal
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        query = session.query(TradingSignal).filter(
            TradingSignal.signal_timestamp >= cutoff_date
        )
        
        if market_type:
            query = query.filter(TradingSignal.market_type == market_type)
        if trading_style:
            query = query.filter(TradingSignal.trading_style == trading_style)
        
        all_signals = query.all()
        executed_signals = [s for s in all_signals if s.is_executed]
        closed_signals = [s for s in all_signals if s.is_closed]
        
        session.close()
        
        if not all_signals:
            st.info("No signals found. Generate some signals first by using the trading signals tab.")
            return
        
        st.subheader("📊 Performance Overview")
        
        # Basic metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Signals", len(all_signals))
        
        with col2:
            st.metric("Executed Signals", len(executed_signals))
        
        with col3:
            st.metric("Closed Trades", len(closed_signals))
        
        with col4:
            active_signals = len([s for s in all_signals if s.is_executed and not s.is_closed])
            st.metric("Active Trades", active_signals)
        
        # Performance details
        if closed_signals:
            profitable_trades = [s for s in closed_signals if s.pnl_percentage and s.pnl_percentage > 0]
            win_rate = (len(profitable_trades) / len(closed_signals)) * 100
            
            total_pnl = sum(s.pnl_amount for s in closed_signals if s.pnl_amount) or 0
            avg_win = sum(s.pnl_amount for s in profitable_trades if s.pnl_amount) / len(profitable_trades) if profitable_trades else 0
            
            losing_trades = [s for s in closed_signals if s.pnl_percentage and s.pnl_percentage < 0]
            avg_loss = sum(s.pnl_amount for s in losing_trades if s.pnl_amount) / len(losing_trades) if losing_trades else 0
            
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                st.metric("Win Rate", f"{win_rate:.1f}%")
            
            with col6:
                currency = "$" if market_type == 'crypto' else "₹"
                st.metric("Total P&L", f"{currency}{total_pnl:.2f}")
            
            with col7:
                st.metric("Avg Win", f"{currency}{avg_win:.2f}")
            
            with col8:
                st.metric("Avg Loss", f"{currency}{avg_loss:.2f}")
        
        # Recent activity
        st.subheader("🕐 Recent Activity")
        recent_signals = sorted(all_signals, key=lambda x: x.signal_timestamp, reverse=True)[:5]
        
        for signal in recent_signals:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.write(f"**{signal.symbol.replace('.NS', '').replace('-USD', '')}**")
            
            with col2:
                st.write(f"{signal.action} - {signal.strategy}")
            
            with col3:
                time_ago = datetime.now() - signal.signal_timestamp
                if time_ago.days > 0:
                    st.write(f"{time_ago.days}d ago")
                else:
                    st.write(f"{time_ago.seconds // 3600}h ago")
            
            with col4:
                if signal.is_closed and signal.pnl_percentage:
                    pnl_color = "🟢" if signal.pnl_percentage > 0 else "🔴"
                    st.write(f"{pnl_color} {signal.pnl_percentage:.2f}%")
                elif signal.is_executed:
                    st.write("📈 Active")
                else:
                    st.write("⏳ Open")
        
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
        # Get signals from database directly
        session = self.db.get_session()
        from database import TradingSignal
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        query = session.query(TradingSignal).filter(
            TradingSignal.signal_timestamp >= cutoff_date
        )
        
        if market_type:
            query = query.filter(TradingSignal.market_type == market_type)
        if trading_style:
            query = query.filter(TradingSignal.trading_style == trading_style)
        
        signals = query.order_by(TradingSignal.signal_timestamp.desc()).all()
        session.close()
        
        if not signals:
            st.info("No signals found in the database. Generate some signals first by using the trading signals tab.")
            return
        
        st.subheader(f"📋 Signal History ({len(signals)} signals)")
        
        # Show all signals (both open and closed)
        for signal in signals:
            with st.expander(f"{signal.symbol.replace('.NS', '').replace('-USD', '')} - {signal.action} - {signal.strategy}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Date:** {signal.signal_timestamp.strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"**Market:** {signal.market_type.title()}")
                    st.write(f"**Style:** {signal.trading_style.title()}")
                    st.write(f"**Entry Price:** {signal.signal_price:.4f}")
                    if signal.confidence:
                        conf_display = f"{signal.confidence:.1f}%" if signal.confidence > 1 else f"{signal.confidence:.1%}"
                        st.write(f"**Confidence:** {conf_display}")
                
                with col2:
                    if signal.is_executed:
                        st.success("✅ Executed")
                        st.write(f"**Execution Price:** {signal.execution_price:.4f}")
                    else:
                        st.info("⏳ Not Executed")
                    
                    if signal.is_closed:
                        st.write(f"**Close Price:** {signal.close_price:.4f}")
                        if signal.pnl_percentage:
                            pnl_color = "🟢" if signal.pnl_percentage > 0 else "🔴"
                            st.write(f"**P&L:** {pnl_color} {signal.pnl_percentage:.2f}%")
                        st.write(f"**Close Reason:** {signal.close_reason}")
                    else:
                        st.warning("📈 Still Open")
        
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
        # Get signals from database directly
        session = self.db.get_session()
        from database import TradingSignal
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        query = session.query(TradingSignal).filter(
            TradingSignal.signal_timestamp >= cutoff_date
        )
        
        if market_type:
            query = query.filter(TradingSignal.market_type == market_type)
        if trading_style:
            query = query.filter(TradingSignal.trading_style == trading_style)
        
        signals = query.all()
        session.close()
        
        if not signals:
            st.info("No signals available for strategy comparison.")
            return
        
        st.subheader("🏆 Strategy Analysis")
        
        # Group signals by strategy
        strategy_stats = {}
        for signal in signals:
            strategy = signal.strategy
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    'total_signals': 0,
                    'executed_signals': 0,
                    'closed_signals': 0,
                    'winning_signals': 0,
                    'total_pnl': 0.0,
                    'win_pnl': 0.0,
                    'loss_pnl': 0.0,
                    'signals': []
                }
            
            stats = strategy_stats[strategy]
            stats['total_signals'] += 1
            stats['signals'].append(signal)
            
            if signal.is_executed:
                stats['executed_signals'] += 1
            
            if signal.is_closed:
                stats['closed_signals'] += 1
                if signal.pnl_amount:
                    stats['total_pnl'] += signal.pnl_amount
                    if signal.pnl_amount > 0:
                        stats['winning_signals'] += 1
                        stats['win_pnl'] += signal.pnl_amount
                    else:
                        stats['loss_pnl'] += signal.pnl_amount
        
        # Calculate performance metrics for each strategy
        strategy_metrics = []
        for strategy, stats in strategy_stats.items():
            win_rate = (stats['winning_signals'] / stats['closed_signals'] * 100) if stats['closed_signals'] > 0 else 0
            avg_win = stats['win_pnl'] / stats['winning_signals'] if stats['winning_signals'] > 0 else 0
            avg_loss = stats['loss_pnl'] / (stats['closed_signals'] - stats['winning_signals']) if (stats['closed_signals'] - stats['winning_signals']) > 0 else 0
            profit_factor = abs(stats['win_pnl'] / stats['loss_pnl']) if stats['loss_pnl'] != 0 else float('inf') if stats['win_pnl'] > 0 else 0
            
            strategy_metrics.append({
                'Strategy': strategy,
                'Total Signals': stats['total_signals'],
                'Executed': stats['executed_signals'],
                'Closed': stats['closed_signals'],
                'Win Rate %': win_rate,
                'Total P&L': stats['total_pnl'],
                'Avg Win': avg_win,
                'Avg Loss': avg_loss,
                'Profit Factor': profit_factor
            })
        
        # Sort by total P&L
        strategy_metrics.sort(key=lambda x: x['Total P&L'], reverse=True)
        
        # Display strategy comparison table
        if strategy_metrics:
            df = pd.DataFrame(strategy_metrics)
            
            # Color code the table
            def highlight_best_strategy(row):
                colors = []
                for col in row.index:
                    if col == 'Total P&L':
                        if row[col] > 0:
                            colors.append('background-color: #d4edda')
                        elif row[col] < 0:
                            colors.append('background-color: #f8d7da')
                        else:
                            colors.append('')
                    elif col == 'Win Rate %':
                        if row[col] >= 60:
                            colors.append('background-color: #d4edda')
                        elif row[col] >= 40:
                            colors.append('background-color: #fff3cd')
                        else:
                            colors.append('background-color: #f8d7da')
                    else:
                        colors.append('')
                return colors
            
            st.dataframe(
                df.style.apply(highlight_best_strategy, axis=1),
                use_container_width=True
            )
            
            # Best and worst performers
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏆 Best Performing Strategy")
                best_strategy = strategy_metrics[0]
                st.write(f"**{best_strategy['Strategy']}**")
                st.write(f"Total P&L: {best_strategy['Total P&L']:.2f}")
                st.write(f"Win Rate: {best_strategy['Win Rate %']:.1f}%")
                st.write(f"Profit Factor: {best_strategy['Profit Factor']:.2f}")
            
            with col2:
                st.subheader("⚠️ Needs Improvement")
                worst_strategy = strategy_metrics[-1]
                st.write(f"**{worst_strategy['Strategy']}**")
                st.write(f"Total P&L: {worst_strategy['Total P&L']:.2f}")
                st.write(f"Win Rate: {worst_strategy['Win Rate %']:.1f}%")
                st.write(f"Profit Factor: {worst_strategy['Profit Factor']:.2f}")
        
        return
    
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