# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Elliott Wave financial analysis application that detects wave patterns in stock market data and generates trade recommendations. The main application (`app_v5_automated.py`) is a Flask web server that provides both a web interface and REST API for analyzing financial instruments using Elliott Wave theory combined with Fibonacci analysis and technical indicators.

## Main Application & Development Commands

### Running the Application
```bash
# Run the main Flask application using the startup script
./start_app.sh

# Or run directly with Python
python app_v5_automated.py

# Application runs on http://0.0.0.0:5001 by default
# Web interface available at http://127.0.0.1:5001
```

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up -d

# Stop Docker containers
docker-compose down

# Application runs on port 5001 in Docker
```

### Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt

# Main dependencies: yfinance, pandas, numpy, plotly, scipy, Flask, curl_cffi
```

### Testing Utilities
```bash
# Test Yahoo Finance rate limit fix
python fix_yfinance_ratelimit.py AAPL --start 2023-01-01 --end 2023-12-31 --interval 1wk

# Test interval parameter fixes  
python test_interval_fix.py
```

## Architecture Overview

### Core Analysis Pipeline
1. **Data Acquisition** (`get_stock_data()` in app_v5_automated.py:318): Uses yfinance with curl_cffi to bypass rate limits
2. **Wave Point Detection** (`find_potential_wave_points()` in app_v5_automated.py:393): Identifies peaks/troughs using scipy.signal
3. **Elliott Wave Analysis** (`find_elliott_waves()` in app_v5_automated.py:439): Scores and validates Elliott Wave sequences
4. **Pattern Recognition** (pattern_recognition.py): Identifies specific wave patterns (impulse, corrective, diagonal)
5. **Fibonacci Analysis** (fibonacci_analysis.py): Calculates retracement and extension levels
6. **Trade Signal Generation** (trade_signals.py): Generates entry/exit points with risk management

### Key Modules

#### `app_v5_automated.py` (Main Application)
- Flask web server with both web interface and REST API
- Core Elliott Wave detection algorithms
- Chart generation using Plotly
- Backtesting functionality
- Stock list management from `stocks_presets/` directory

#### `improved_elliott.py` 
- Enhanced Elliott Wave scoring algorithms
- Stricter rule enforcement for wave validation
- Advanced alternation guideline checking
- RSI divergence confirmation

#### `trade_signals.py`
- Trade signal generation based on wave analysis
- Risk management with stop-loss and take-profit calculations
- Position sizing algorithms
- Multiple confidence levels (conservative, moderate, aggressive)

#### `pattern_recognition.py`
- Specific Elliott Wave pattern identification
- Impulse wave validation (1-2-3-4-5 sequences)
- Corrective pattern detection (zigzag, flat, triangle)
- Ending diagonal pattern recognition

#### `fibonacci_analysis.py`
- Fibonacci retracement and extension calculations
- Price target projections
- Support/resistance level identification

### Data Flow
1. User inputs ticker symbol, date range, and interval via web interface or API
2. `get_stock_data()` fetches historical price data using yfinance with rate limit handling
3. `run_analysis()` orchestrates the complete analysis pipeline
4. Results are visualized with `plot_chart()` using Plotly
5. Trade recommendations generated via `generate_trade_recommendation()`
6. Backtesting performed with `run_backtest_simulation()` for historical validation

### API Endpoints
- `GET /` - Main web interface
- `GET /api/stock_lists` - Returns available stock preset lists from `stocks_presets/`
- `GET /api/test` - API health check
- `GET /debug/routes` - Lists all registered Flask routes

### File Structure Notes
- `stocks_presets/` contains predefined stock lists (.txt files)
- `templates/` contains HTML templates for the web interface
- `static/js/` contains client-side JavaScript for UI interactions
- Multiple Python files handle different aspects (fix files are utilities for specific issues)

### Key Configuration
- Application runs on port 5001 (both local and Docker)
- Uses `curl_cffi` for Yahoo Finance rate limit bypass
- Default analysis parameters: peak_order=8, ATR period=14, RSI period=14
- Fibonacci levels: 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.414, 1.618, 2.618

### Important Implementation Details
- Rate limiting handled via `curl_cffi` browser impersonation
- Wave point detection uses `scipy.signal.find_peaks` with prominence-based filtering
- Elliott Wave scoring uses multiple criteria: rule compliance, Fibonacci targets, volume analysis
- Chart rendering uses Plotly with candlestick charts and annotation overlays
- Backtest functionality compares historical predictions with actual price movements