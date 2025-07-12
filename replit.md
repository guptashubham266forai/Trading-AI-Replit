# NSE Real-Time Stock Scanner

## Overview

This is a comprehensive real-time stock analysis and trading application built with Streamlit for the National Stock Exchange (NSE) of India. The system provides live market data, technical analysis, trading signals, and stock screening capabilities for intraday trading strategies.

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
- **Purpose**: Streamlit UI orchestration and user interface
- **Features**: Interactive charts, real-time updates, signal display
- **Architecture**: Session state management for persistent data

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