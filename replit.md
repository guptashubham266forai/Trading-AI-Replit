# Real-Time Trading Scanner (Stocks & Crypto)

## Overview

This is a comprehensive real-time trading analysis application built with Streamlit that supports both NSE stocks and cryptocurrencies. The unified platform provides live market data, technical analysis, trading signals, predictive analysis, and screening capabilities for both intraday and swing trading strategies. Users can switch between Indian stock market (NSE) and cryptocurrency markets, and choose between intraday (minutes to hours) or swing trading (days to weeks) approaches.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application
- **UI Components**: Interactive dashboards with real-time data visualization
- **Charting**: Plotly for candlestick charts and technical indicators
- **Layout**: Wide layout with expandable sidebar for controls
- **State Management**: Streamlit session state for persistent data across user interactions

### Backend Architecture
- **Data Processing**: Pandas for data manipulation and analysis
- **Real-time Data**: yfinance API for NSE stock data fetching
- **Technical Analysis**: Custom technical indicators module
- **Trading Logic**: Strategy-based signal generation system
- **Threading**: Asyncio and threading for concurrent data operations

### Data Architecture
- **Data Source**: Yahoo Finance API for NSE stocks (.NS suffix)
- **Caching**: In-memory caching with 5-minute expiry for performance
- **Data Structure**: Time-series data using Pandas DataFrames
- **Storage**: Session-based storage, no persistent database

## Key Components

### 1. Data Fetcher (`data_fetcher.py`)
- **Purpose**: Real-time market data acquisition
- **Features**: 
  - NSE market hours validation
  - Symbol formatting for yfinance compatibility
  - Intelligent caching mechanism
  - Pre-configured list of 50 liquid NSE stocks

### 2. Technical Indicators (`technical_indicators.py`)
- **Purpose**: Calculate technical analysis indicators
- **Indicators**: Moving averages, RSI, MACD, Bollinger Bands, Stochastic
- **Implementation**: NumPy and Pandas-based calculations
- **Design**: Modular indicator functions for reusability

### 3. Trading Strategies (`strategies.py`)
- **Purpose**: Generate buy/sell signals based on technical analysis
- **Strategies**: Moving Average Crossover, RSI-based, MACD signals
- **Risk Management**: Configurable risk percentages and risk-reward ratios
- **Signal Confidence**: Scoring system for trade recommendations

### 4. Stock Screener (`stock_screener.py`)
- **Purpose**: Filter stocks based on intraday trading criteria
- **Filters**: Volume analysis, price range, volatility screening
- **Focus**: Liquid stocks suitable for day trading
- **Criteria**: Minimum volume thresholds and volatility ranges

### 5. Utilities (`utils.py`)
- **Purpose**: Helper functions for formatting and calculations
- **Features**: Currency formatting, risk-reward calculations, position sizing
- **Localization**: Indian currency formatting (₹, Crore, Lakh)

### 6. Main Application (`app.py`)
- **Purpose**: Streamlit UI orchestration and user interface for NSE stocks
- **Features**: Interactive charts, real-time updates, signal display
- **Architecture**: Session state management for persistent data

### 7. Cryptocurrency Data Fetcher (`crypto_data_fetcher.py`)
- **Purpose**: Real-time cryptocurrency market data acquisition
- **Features**: 
  - 24/7 crypto market data (Bitcoin, Ethereum, Binance Coin, etc.)
  - Symbol formatting for major crypto pairs (BTC-USD, ETH-USD)
  - Faster refresh rates optimized for crypto market volatility
  - Pre-configured list of 25 liquid cryptocurrencies

### 8. Cryptocurrency Screener (`crypto_screener.py`)
- **Purpose**: Filter cryptocurrencies based on trading criteria optimized for crypto markets
- **Features**: Whale movement detection, unusual activity patterns, breakout candidates
- **Crypto-specific**: Higher volatility thresholds, volume spike detection, 24/7 market analysis

### 9. Cryptocurrency Application (`crypto_app.py`)
- **Purpose**: Dedicated Streamlit UI for cryptocurrency trading analysis
- **Features**: Crypto market overview, prediction analysis, professional crypto trading signals
- **Architecture**: Parallel session state management with stock scanner

### 10. Predictive Analysis (`predictive_analysis.py`)
- **Purpose**: Advanced prediction system for both stocks and cryptocurrencies
- **Features**: 
  - Smart money flow detection (institutional/whale activity)
  - Pre-breakout setup identification
  - Momentum divergence analysis
  - Accumulation/distribution pattern recognition
- **Timeframe**: 30 minutes to 2 hours advance warning before major moves

### 11. Swing Trading Strategies (`swing_strategies.py`)
- **Purpose**: Professional swing trading strategies for multi-day to multi-week positions
- **Features**: 
  - Advanced trend following with pullback entries
  - Mean reversion using Bollinger Bands and multi-timeframe RSI
  - Breakout detection with volume confirmation and volatility compression
  - Support/resistance level trading with pivot points
  - Pattern recognition (double tops/bottoms, cup and handle)
  - Volume analysis with OBV and institutional flow detection
- **Risk Management**: Higher risk tolerance (3%) and reward ratios (3:1) optimized for swing trades

### 12. Main Unified Application (`main_app.py`)
- **Purpose**: Unified Streamlit interface combining all trading approaches
- **Features**: 
  - Market selection (NSE stocks vs cryptocurrencies)
  - Trading style selection (intraday vs swing trading)
  - Adaptive strategy configuration based on selected approach
  - Universal charting with style-specific indicators
  - Dynamic risk management parameters
- **Architecture**: Single application managing multiple trading approaches and market types

### 13. Database Management (`database.py`)
- **Purpose**: PostgreSQL database integration for signal tracking and performance analysis
- **Features**: 
  - Trading signal storage with full metadata
  - Portfolio performance tracking with daily metrics
  - P&L calculation and risk metrics
  - Signal execution and closing tracking
- **Models**: TradingSignal, PortfolioPerformance tables with comprehensive fields

### 14. Performance Analyzer (`performance_analyzer.py`)
- **Purpose**: Comprehensive trading performance analysis and reporting
- **Features**: 
  - Individual trade performance tracking with P&L calculation
  - Portfolio growth analysis with cumulative returns
  - Strategy comparison and ranking by profitability
  - Risk metrics including drawdown and Sharpe ratio
  - Performance visualization with interactive charts
  - CSV export functionality for detailed reporting
  - Direct database queries for reliable signal history
- **Metrics**: Win rate, profit factor, expectancy, max drawdown, strategy performance

### 15. Live P&L Tracker (`live_tracker.py`)
- **Purpose**: Real-time performance tracking for active trading signals
- **Features**: 
  - Live P&L calculation for active positions
  - Real-time market price updates
  - Signal status monitoring (profit/loss/break-even)
  - Distance to target and stop-loss tracking
  - Current win rate and unrealized P&L display
  - Color-coded performance indicators
- **Integration**: Works with both stock and crypto data fetchers for unified tracking

## Data Flow

1. **Data Acquisition**: DataFetcher retrieves real-time NSE stock prices via yfinance
2. **Screening**: StockScreener filters stocks based on volume and volatility criteria
3. **Technical Analysis**: TechnicalIndicators calculates various technical metrics
4. **Strategy Processing**: TradingStrategies generates signals based on indicator combinations
5. **Visualization**: Streamlit renders interactive charts and displays trading recommendations
6. **Caching**: Data is cached for 5 minutes to optimize API calls and performance

## External Dependencies

### Core Dependencies
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **plotly**: Interactive charting and visualization
- **yfinance**: Yahoo Finance API for stock data
- **scipy**: Scientific computing for technical indicators

### API Integrations
- **Yahoo Finance**: Primary data source for NSE stock prices and historical data
- **NSE Market Hours**: Built-in market timing validation for Indian stock exchange

## Deployment Strategy

### Development Environment
- **Platform**: Replit-compatible Python environment
- **Requirements**: All dependencies installable via pip
- **Configuration**: Streamlit configuration for wide layout and custom theming

### Production Considerations
- **Scalability**: Caching mechanism reduces API load
- **Performance**: Asynchronous data fetching capabilities
- **Reliability**: Error handling for API failures and data inconsistencies
- **Market Hours**: Intelligent handling of market open/close status

### Key Architectural Decisions

1. **Streamlit Choice**: Selected for rapid prototyping and easy deployment of financial dashboards
2. **yfinance Integration**: Chosen for reliable NSE data access without API keys
3. **Modular Design**: Separated concerns into distinct modules for maintainability
4. **In-Memory Caching**: Balances performance with data freshness for real-time applications
5. **Session State**: Enables persistent user experience across interactions
6. **Indian Market Focus**: Specialized for NSE trading with local formatting and market hours

The architecture prioritizes real-time performance, user experience, and trading-focused functionality while maintaining code modularity and extensibility for future enhancements.