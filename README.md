# 📈 Elliott Wave Analyzer

This is a Python-based web application for **Elliott Wave analysis** of financial markets. It detects wave structures (impulses and corrections), projects future price zones using **Fibonacci retracements/extensions**, and provides **trade setup recommendations** with entry, stop-loss, and take-profit levels.

---

## Image

![App preview](app.png)

## 🎯 Purpose

This tool is designed to help **technical traders, analysts, and developers** visualize Elliott Wave patterns, validate potential wave sequences, and simulate or plan trades using:
- Elliott Wave Theory (impulse + ABC correction)
- ATR and RSI indicators
- Price projection zones based on Fibonacci levels
- Trade recommendations with position sizing and risk control
- Historical backtests for validation

It's useful for strategy development, market pattern research, and educational purposes.

---

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ESJavadex/elliot-waves-auto.git
   cd elliott-wave-analyzer
   ```

2. **(Optional)** Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Use

1. **Run the Flask application**:
   ```bash
   python app.py
   ```

2. Open your browser and go to:  
   [http://127.0.0.1:5000](http://127.0.0.1:5000)

3. Enter a stock ticker (e.g. `AAPL`, `^SPX`), date range, and interval (`1d`, `1wk`, etc.).

4. Click "Submit" to:
   - Visualize detected Elliott Waves
   - See projected future price zones
   - Optionally simulate trade setups or run backtests

5. You can also toggle:
   - **Trade signal overlays** (entry/SL/TP)
   - **Backtest mode** to compare historical forecasts with real outcomes
   - **Multi-stock analysis** for batch processing

---

## 🛡️ Yahoo Finance Rate Limit Solution

The application now uses `curl_cffi` to avoid Yahoo Finance rate limiting issues. This solution:

1. Creates a session that impersonates Chrome browser
2. Bypasses the "Too Many Requests" errors that occur with frequent API calls
3. Is implemented in the `get_stock_data` function

You can also use the standalone utility to test this approach:

```bash
python fix_yfinance_ratelimit.py AAPL --start 2023-01-01 --end 2023-12-31 --interval 1wk
```

This utility demonstrates how to use the solution with any yfinance application.

---

## 🐳 Docker Deployment

You can also run the application using Docker, which is especially useful for deployment on a Raspberry Pi:

1. **Build and start the Docker container**:
   ```bash
   docker-compose up -d
   ```

2. Access the application in your browser at:
   [http://your-host-ip:5001](http://your-host-ip:5001)

3. **Stop the Docker container**:
   ```bash
   docker-compose down
   ```

The Docker configuration:
- Uses Python 3.11 as the base image
- Exposes port 5001 for web access
- Mounts the templates directory as a volume
- Includes resource limits for Raspberry Pi compatibility

---

> 💡 This app is for **educational and strategy development** only. Do not use it for live trading.