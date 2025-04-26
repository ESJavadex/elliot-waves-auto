# -*- coding: utf-8 -*-
import os
import datetime
import warnings
import math # For checking nan
import json # For pretty printing the summary
import traceback # For detailed error logging

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline
from scipy.signal import find_peaks

from flask import Flask, render_template, request, jsonify
import glob
import os

app = Flask(__name__)

# Test endpoint to verify routing
@app.route('/api/test', methods=['GET'])
def test_api():
    print("\n*** TEST API ENDPOINT CALLED ***")
    return jsonify({"status": "ok", "message": "API is working"})

# Debug route to list all registered routes
@app.route('/debug/routes', methods=['GET'])
def debug_routes():
    print("\n*** DEBUG ROUTES ENDPOINT CALLED ***")
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': [method for method in rule.methods if method not in ['HEAD', 'OPTIONS']],
            'path': str(rule)
        })
    return jsonify({
        'routes': routes,
        'total': len(routes)
    })

# <<< Data Cleaning Function >>>
# (clean_data_for_json remains the same as before)

# --- NEW: Stock List TXT Loader API ---
@app.route('/api/stock_lists', methods=['GET'])
def get_stock_lists():
    print("\n*** API ENDPOINT CALLED: /api/stock_lists ***")
    try:
        # Look for all .txt files in the stocks_presets directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Base directory: {base_dir}")
        
        presets_dir = os.path.join(base_dir, 'stocks_presets')
        print(f"Presets directory: {presets_dir}")
        
        # Check if directory exists
        if not os.path.exists(presets_dir):
            print(f"ERROR: Directory does not exist: {presets_dir}")
            return jsonify({"error": f"Directory not found: {presets_dir}"}), 500
        
        # List all files in the directory
        all_files = os.listdir(presets_dir)
        print(f"All files in directory: {all_files}")
        
        txt_files = [f for f in glob.glob(os.path.join(presets_dir, '*.txt'))]
        print(f"TXT files found: {txt_files}")
        
        result = {}
        for path in txt_files:
            name = os.path.splitext(os.path.basename(path))[0]
            print(f"Processing file: {name}")
            with open(path, 'r') as f:
                content = f.read().strip()
            result[name] = content
        
        print(f"Final result: {result}")
        return jsonify(result)
    except Exception as e:
        print(f"ERROR in get_stock_lists: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def clean_data_for_json(obj):
    """
    Recursively traverses a dictionary or list and converts non-JSON-compliant
    values (like np.nan, np.inf, pd.Timestamp, pd.Series) into JSON-compliant formats (strings).
    """
    if isinstance(obj, dict):
        return {str(k): clean_data_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [clean_data_for_json(item) for item in obj]
    elif isinstance(obj, (np.int64, np.int32, np.int16)): return int(obj)
    elif isinstance(obj, (np.float64, np.float32, float)):
        if np.isnan(obj): return "NaN"
        elif np.isinf(obj): return "Infinity" if obj > 0 else "-Infinity"
        else: return float(obj)
    elif isinstance(obj, (np.bool_, bool)): return bool(obj)
    elif isinstance(obj, pd.Timestamp):
        try:
            if pd.isna(obj): return "NaT"
            return obj.isoformat()
        except Exception: return "Invalid Timestamp"
    elif isinstance(obj, np.datetime64):
         try:
            if np.isnat(obj): return "NaT"
            return pd.to_datetime(obj).isoformat()
         except Exception: return "Invalid np.datetime64"
    elif isinstance(obj, pd.Series):
        try:
            name_str = clean_data_for_json(obj.name)
            preview_items = {}
            num_preview = min(len(obj), 3)
            for i in range(num_preview):
                 idx_name = clean_data_for_json(obj.index[i])
                 val = clean_data_for_json(obj.iloc[i])
                 preview_items[str(idx_name)] = val
            preview_str = json.dumps(preview_items)
            return f"Pandas Series (Index: {name_str}, Length: {len(obj)}, First {num_preview}: {preview_str})"
        except Exception: return f"Pandas Series (Index: {clean_data_for_json(obj.name)}, Length: {len(obj)}, details omitted)"
    elif isinstance(obj, pd.DataFrame):
        return f"Pandas DataFrame (Shape: {obj.shape}, Columns: {clean_data_for_json(list(obj.columns))}, details omitted)"
    elif pd.isna(obj): return "NaN"
    elif obj is None: return None
    elif isinstance(obj, (str, int)): return obj
    else:
        try: return str(obj)
        except Exception: return f"Unserializable Object ({type(obj).__name__})"

# --- Suppress Warnings ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")
pd.options.mode.chained_assignment = None

# --- Configuration Constants ---
# (Constants remain mostly the same, added TARGETBOX_CANDLES_TERTIARY)
PEAK_TROUGH_ORDER_WEEKLY = 8
PEAK_TROUGH_ORDER_DAILY = 5
PEAK_TROUGH_ORDER_DEFAULT = 3
PROMINENCE_PCT_FACTOR = 0.010
PROMINENCE_ATR_FACTOR = 1.5
ATR_PERIOD = 14
RSI_PERIOD = 14
MA_PERIODS = [20, 50]
# Fibonacci Levels - Comprehensive set for realistic trading
FIB_RETRACEMENTS = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
FIB_EXTENSIONS = [0.618, 1.0, 1.618, 2.0, 2.618, 3.618, 4.236]

# Take Profit Levels for practical trading
TP_LEVELS = {
    "TP1": 1.0,      # 100% extension - conservative target
    "TP2": 1.618,   # 161.8% extension - standard target
    "TP3": 2.618    # 261.8% extension - aggressive target (strong momentum)
}

# Alternation Principle Constants
ALTERNATION_THRESHOLD = 0.618  # If both W2 and W4 exceed this, alternation principle is violated
SHALLOW_THRESHOLD = 0.382      # Retracements below this are considered shallow
DEEP_THRESHOLD = 0.618         # Retracements above this are considered deep
WAVE3_EXTENSION_THRESHOLD = 1.618
WAVE2_DEEP_RETRACE = 0.618
WAVE4_SHALLOW_RETRACE = 0.500
FIB_RATIO_TOLERANCE = 0.10
CONFLUENCE_PRICE_TOLERANCE_FACTOR = 0.005
SCORE_RULE_PASS = 100
SCORE_GUIDELINE_PASS = 15
SCORE_FIB_TARGET_HIT = 25
SCORE_CHANNEL_HIT = 10
SCORE_VOLUME_GUIDELINE = 5
PLOT_CHANNELS = True
PLOT_INVALIDATION_LEVEL = True
PLOT_RSI = True
TARGETBOX_CANDLES_PRIMARY = 15
TARGETBOX_CANDLES_SECONDARY = 30
TARGETBOX_CANDLES_TERTIARY = 45 # Added offset for Wave C estimate
TARGETBOX_CANDLES_FALLBACK = 20
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# --- Helper Functions ---
# (Helper functions remain the same)
def calculate_atr(data, period=14):
    if not all(col in data.columns for col in ['High', 'Low', 'Close']): return None
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.ewm(alpha=1/period, adjust=False).mean()
    return atr

def calculate_fib_ratio(start_val, end_val, retrace_val):
    move = end_val - start_val
    retrace_move = retrace_val - end_val
    return abs(retrace_move / move) if abs(move) > 1e-9 else np.nan

def get_fib_level_str(ratio):
    if pd.isna(ratio): return "N/A"
    levels = {0.236: '23.6%', 0.382: '38.2%', 0.5: '50.0%', 0.618: '61.8%', 0.786: '78.6%', 1.0: '100.0%', 1.272: '127.2%', 1.618: '161.8%', 2.0: '200.0%', 2.618: '261.8%', 4.236: '423.6%'}
    closest_level = min(levels.keys(), key=lambda k: abs(k - ratio))
    return levels[closest_level] if abs(closest_level - ratio) < FIB_RATIO_TOLERANCE else f"{ratio*100:.1f}%"

def is_close_to_fib(ratio, fib_levels):
    if pd.isna(ratio): return False
    for level in fib_levels:
        if abs(ratio - level) < FIB_RATIO_TOLERANCE: return True
    return False

def calculate_fibonacci_retracements(start_price, end_price, atr_value=None):
    levels = {}
    # Check for None values in inputs
    if start_price is None or end_price is None:
        print(f"  Warning: Invalid inputs for Fibonacci retracements: {start_price}, {end_price}")
        return levels
        
    try:
        diff = end_price - start_price
        is_upward = diff > 0
        if abs(diff) < 1e-9: return levels
        
        # Standard Fibonacci retracements
        for level in FIB_RETRACEMENTS:
            levels[level] = end_price - (diff * level) if is_upward else end_price + abs(diff * level)
        
        # Apply ATR adjustment if provided (makes targets more realistic by accounting for volatility)
        if atr_value is not None and atr_value > 0:
            # Adjust levels based on ATR to account for market volatility
            # More volatile markets need wider targets
            atr_factor = min(1.0, atr_value / abs(diff))
            
            # Apply ATR adjustment to create more realistic targets
            for level in levels.keys():
                # Adjust the precision of the target based on volatility
                # Higher volatility = wider target zones
                adjustment = atr_value * 0.5 * atr_factor
                if is_upward:
                    levels[level] = levels[level] - adjustment
                else:
                    levels[level] = levels[level] + adjustment
                    
            print(f"  Trader insight: Applied ATR adjustment ({atr_value:.2f}) to Fibonacci targets")
            
        return levels
    except Exception as e:
        print(f"  Error in Fibonacci retracement calculation: {e}")
        return levels

def calculate_fibonacci_extensions(p_start_move, p_end_move, p_project_from, atr_value=None):
    levels = {}
    # Check for None values in inputs
    if p_start_move is None or p_end_move is None or p_project_from is None:
        print(f"  Warning: Invalid inputs for Fibonacci extensions: {p_start_move}, {p_end_move}, {p_project_from}")
        return levels
        
    try:
        diff = p_end_move - p_start_move
        if abs(diff) < 1e-9: return levels
        # Determine if we're in an uptrend or downtrend based on the direction of the first move
        is_uptrend = diff > 0
        
        # Standard Fibonacci extensions
        for level in FIB_EXTENSIONS:
            # For uptrend: project upward from the projection point
            # For downtrend: project downward from the projection point
            levels[level] = p_project_from + (abs(diff) * level * (1 if is_uptrend else -1))
        
        # Add specific take-profit levels for practical trading
        for tp_name, tp_level in TP_LEVELS.items():
            if tp_level not in levels:  # Avoid duplicates
                levels[tp_level] = p_project_from + (abs(diff) * tp_level * (1 if is_uptrend else -1))
        
        # Apply ATR adjustment if provided (makes targets more realistic by accounting for volatility)
        if atr_value is not None and atr_value > 0:
            # Calculate ATR factor based on the move size
            atr_factor = min(1.5, atr_value / abs(diff))
            
            # Adjust extension levels based on volatility
            for level in levels.keys():
                # For higher extensions, apply progressively larger adjustments
                # This accounts for increased uncertainty at higher targets
                adjustment = atr_value * (level * 0.3) * atr_factor
                
                if is_uptrend:
                    # For uptrends, extend targets slightly to account for momentum
                    levels[level] = levels[level] + adjustment
                else:
                    # For downtrends, extend targets downward
                    levels[level] = levels[level] - adjustment
            
            print(f"  Trader insight: Applied ATR adjustment ({atr_value:.2f}) to extension targets")
            
        return levels
    except Exception as e:
        print(f"  Error in Fibonacci extension calculation: {e}")
        return levels

def calculate_rsi(data, period=14):
    if 'Close' not in data.columns: return None
    close_delta = data['Close'].diff()
    up = close_delta.clip(lower=0); down = -1 * close_delta.clip(upper=0)
    ma_up = up.ewm(com=period - 1, adjust=True, min_periods=period).mean()
    ma_down = down.ewm(com=period - 1, adjust=True, min_periods=period).mean()
    ma_down_safe = ma_down.replace(0, 1e-9)
    rs = ma_up / ma_down_safe; rsi = 100 - (100 / (1 + rs))
    rsi[ma_down == 0] = 100
    return pd.Series(rsi, index=data.index)

# --- Data Fetching ---
# (Function remains the same as before)
def get_stock_data(ticker, start_date, end_date, interval):
    """Fetches, prepares stock data using yfinance, and calculates ATR, MAs, and RSI."""
    print(f"\n[Data Fetch] Attempting: {ticker} ({interval}) from {start_date} to {end_date}...")
    try:
        stock = yf.Ticker(ticker)
        max_indicator_period = max(MA_PERIODS) + ATR_PERIOD + RSI_PERIOD + 50
        start_dt_obj = pd.to_datetime(start_date); end_dt_obj = pd.to_datetime(end_date)
        start_dt_buffered = start_dt_obj - pd.Timedelta(days=max_indicator_period)
        end_dt_fetch = end_dt_obj + pd.Timedelta(days=1)
        full_data_range = stock.history(start=start_dt_buffered.strftime('%Y-%m-%d'), end=end_dt_fetch.strftime('%Y-%m-%d'),
                                     interval=interval, auto_adjust=False, prepost=False)
        if full_data_range.empty: print(f"  Error: Could not fetch full data range (incl. buffer) for {ticker}."); return None
        if not isinstance(full_data_range.index, pd.DatetimeIndex): full_data_range.index = pd.to_datetime(full_data_range.index)
        if full_data_range.index.tz is None: full_data_range.index = full_data_range.index.tz_localize('UTC')
        else: full_data_range.index = full_data_range.index.tz_convert('UTC')
        full_data_range['ATR'] = calculate_atr(full_data_range, period=ATR_PERIOD)
        full_data_range['RSI'] = calculate_rsi(full_data_range, period=RSI_PERIOD)
        for period in MA_PERIODS: full_data_range[f'SMA_{period}'] = full_data_range['Close'].rolling(window=period).mean()
        start_dt_utc = start_dt_obj.tz_localize('UTC') if start_dt_obj.tzinfo is None else start_dt_obj.tz_convert('UTC')
        end_dt_utc = end_dt_obj.tz_localize('UTC') if end_dt_obj.tzinfo is None else end_dt_obj.tz_convert('UTC')
        data = full_data_range.loc[start_dt_utc:end_dt_utc].copy()
        if data.empty: print(f"  Error: No data found within the specified date range {start_date} to {end_date} AFTER filtering buffer."); return None
        try:
            stock_info = stock.get_info(timeout=10); data['Currency'] = stock_info.get('currency', 'USD')
        except Exception as e: print(f"  Warning: Could not fetch currency info: {e}. Defaulting to USD."); data['Currency'] = 'USD'
        initial_len = len(data); required_cols = ['Close', 'High', 'Low', 'Open']
        if 'ATR' in data.columns: required_cols.append('ATR')
        if 'RSI' in data.columns: required_cols.append('RSI')
        if 'Volume' in data.columns:
            required_cols.append('Volume'); data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')
        else: print("  Warning: 'Volume' column not found in data.")
        data.dropna(subset=required_cols, inplace=True, how='any')
        if len(data) < initial_len: print(f"  Note: Removed {initial_len - len(data)} rows with NaN values from the final date range.")
        if data.empty: print("  Error: Data empty after NaN removal/trimming."); return None
        print(f"  Success: Fetched and prepared {len(data)} data points for {start_date} to {end_date}.")
        return data
    except Exception as e: print(f"  [Data Fetch] Unexpected Error: {e}"); traceback.print_exc(); return None

# --- Peak/Trough Detection ---
# (Function remains the same as before)
def find_potential_wave_points(data, order=5, prom_pct=0.01, prom_atr=1.5):
    """Finds potential peaks and troughs using scipy.signal.find_peaks."""
    print(f"\n[Peak Find] Detecting potential turning points (Order: {order}, Prominence: ATR & PCT)...")
    if data is None or data.empty or len(data) < order * 2 + 1 or 'ATR' not in data.columns:
        print(f"  Error: Insufficient data or missing ATR for peak finding.")
        return pd.DataFrame()
    price_high = data['High'].to_numpy(); price_low = data['Low'].to_numpy()
    price_close = data['Close'].to_numpy(); atr = data['ATR'].to_numpy(); n = len(price_close)
    rolling_median_price = pd.Series(price_close).rolling(window=order*2+1, center=True, min_periods=5).median().bfill().ffill()
    prominence_pct_val = (rolling_median_price * prom_pct).to_numpy(); prominence_atr_val = atr * prom_atr
    prominence_val = np.full_like(price_close, np.nan)
    valid_prom_calc = ~np.isnan(prominence_pct_val) & ~np.isnan(prominence_atr_val)
    prominence_val[valid_prom_calc] = np.maximum(prominence_pct_val[valid_prom_calc], prominence_atr_val[valid_prom_calc])
    min_fallback_prom = np.nanmean(price_close) * 0.001 if not pd.isna(np.nanmean(price_close)) else 0.01
    min_fallback_prom = max(min_fallback_prom, 1e-6); prominence_val = np.nan_to_num(prominence_val, nan=min_fallback_prom)
    prominence_val[prominence_val <= min_fallback_prom] = min_fallback_prom
    distance = max(1, int(order // 1.5));
    if n < distance + 1: print(f" Error: Not enough data ({n}) for peak distance ({distance})."); return pd.DataFrame()
    try:
        if len(prominence_val) != len(price_high):
            print(" Error: Prominence array length mismatch.");
            if len(prominence_val) > len(price_high): prominence_val = prominence_val[:len(price_high)]
            else: prominence_val = np.pad(prominence_val, (0, len(price_high) - len(prominence_val)), 'constant', constant_values=min_fallback_prom)
        high_peaks_idx, _ = find_peaks(price_high, prominence=prominence_val, distance=distance)
        low_troughs_idx, _ = find_peaks(-price_low, prominence=prominence_val, distance=distance)
        peaks = data.iloc[high_peaks_idx].assign(Type='peak'); troughs = data.iloc[low_troughs_idx].assign(Type='trough')
        wave_pts = pd.concat([peaks, troughs]).sort_index()
    except Exception as e: print(f"  Error: scipy find_peaks failed: {e}"); traceback.print_exc(); return pd.DataFrame()
    if wave_pts.empty: print("  No initial peaks or troughs found."); return pd.DataFrame()
    filtered_list = []; last_type = None
    for idx, row in wave_pts.iterrows():
        current_type = row['Type']
        if not filtered_list or current_type != last_type: filtered_list.append(row); last_type = current_type
        elif current_type == 'peak' and row['High'] > filtered_list[-1]['High']: filtered_list[-1] = row
        elif current_type == 'trough' and row['Low'] < filtered_list[-1]['Low']: filtered_list[-1] = row
    if not filtered_list: print("  No alternating points after filtering."); return pd.DataFrame()
    wave_points_final = pd.DataFrame(filtered_list).drop(columns=['Type'])
    if not wave_points_final.index.is_unique:
        duplicate_count = wave_points_final.index.duplicated().sum(); print(f"  Warning: Duplicate timestamps found ({duplicate_count}) after filtering. Keeping first.")
        wave_points_final = wave_points_final.loc[~wave_points_final.index.duplicated(keep='first')]
    print(f"  Found {len(wave_points_final)} potential alternating wave points.")
    return wave_points_final


# --- Elliott Wave Identification & Analysis ---
# (Function remains the same as before)
def find_elliott_waves(wave_points, data_full):
    """Finds the highest-scoring valid partial or full impulse sequence (0-1...N)."""
    # (Full function code remains the same as provided in previous steps)
    print("\n[EW Analysis] Finding Best Scoring Partial Impulse Sequence (incl. Volume Score)...")
    analysis_results = {"found_impulse": False, "details": {}, "last_label": None, "last_point": None, "last_point_overall": None }
    has_volume = 'Volume' in data_full.columns and not data_full['Volume'].isnull().all()
    if not has_volume: print("  Info: 'Volume' data not available or is all NaN. Volume guidelines will be skipped.")
    if wave_points is None or wave_points.empty:
         print("  Info: No wave points provided for analysis.")
         return pd.DataFrame() if wave_points is None else wave_points, analysis_results
    wave_points = wave_points.copy()
    wave_points['EW_Label'] = [f"P{i}" for i in range(len(wave_points))] # Initial labeling
    if not wave_points.empty:
        analysis_results["last_label"] = wave_points.iloc[-1]['EW_Label']
        analysis_results["last_point"] = wave_points.iloc[-1]
        analysis_results["last_point_overall"] = wave_points.iloc[-1]
    if len(wave_points) < 3:
        print(f"  Info: Not enough points ({len(wave_points)} < 3) for impulse analysis.")
        return wave_points, analysis_results
    wave_points['EW_Label'] = "" # Reset labels
    best_sequence_score = -1; best_sequence_details = None
    identified_sequence_last_label = None; identified_sequence_last_point = None

    def score_sequence(points, is_up, length, full_data): # Internal scoring function
        score = 0; eps = 1e-9
        rules_passed = {"W2_Ret": False, "W4_Overlap": True, "W3_Shortest": True}
        guidelines_passed = {"W3_Ext": False, "W5_Eq_W1": False, "Alt_W2W4": False, "Channel": False, "Vol_W3>W1": False, "Vol_W5<=W3": True, "Vol_Corr<Imp": True}
        fib_details = {}; vol_details = {}
        p = points; required_keys = list(range(length))
        if not all(k in p for k in required_keys): return -1
        p_hl = {}
        for k, pt in p.items(): p_hl[k] = (pt['Low'] if is_up else pt['High']) if k % 2 == 0 else (pt['High'] if is_up else pt['Low'])
        if not all(k in p_hl for k in required_keys): return -1
        if length >= 3:
            # Rule: Wave 2 never retraces more than 100% of Wave 1
            if (is_up and p_hl[2] > p_hl[0]) or (not is_up and p_hl[2] < p_hl[0]): 
                rules_passed["W2_Ret"] = True; score += SCORE_RULE_PASS
            else: 
                print("  Rule violation: Wave 2 retraces more than 100% of Wave 1")
                return -1
            w2r = calculate_fib_ratio(p_hl[0], p_hl[1], p_hl[2]); fib_details['W2_Ret_W1'] = (w2r, get_fib_level_str(w2r)); score += SCORE_FIB_TARGET_HIT * is_close_to_fib(w2r, [0.5, 0.618, 0.786])
        if length >= 4:
            if (is_up and p_hl[3] <= p_hl[1]) or (not is_up and p_hl[3] >= p_hl[1]): return -1
            len_w1 = abs(p_hl[1] - p_hl[0]); len_w3 = abs(p_hl[3] - p_hl[2]); w3r = len_w3 / len_w1 if len_w1 > eps else np.nan
            fib_details['W3_Ext_W1'] = (w3r, get_fib_level_str(w3r)); score += SCORE_FIB_TARGET_HIT * is_close_to_fib(w3r, [1.618, 2.0, 2.618])
            guidelines_passed["W3_Ext"] = not pd.isna(w3r) and w3r > WAVE3_EXTENSION_THRESHOLD; score += SCORE_GUIDELINE_PASS * guidelines_passed["W3_Ext"]
        if length >= 5:
            # Rule: Wave 4 never overlaps the end of Wave 1 (except in very volatile markets)
            w1_overlap_level = p_hl[1]
            if (is_up and p_hl[4] <= w1_overlap_level) or (not is_up and p_hl[4] >= w1_overlap_level): 
                rules_passed["W4_Overlap"] = False
                print("  Rule violation: Wave 4 overlaps the end of Wave 1")
            else: 
                score += SCORE_RULE_PASS
            w4r = calculate_fib_ratio(p_hl[2], p_hl[3], p_hl[4]); fib_details['W4_Ret_W3'] = (w4r, get_fib_level_str(w4r)); score += SCORE_FIB_TARGET_HIT * is_close_to_fib(w4r, [0.236, 0.382, 0.5])
            
            # Rule: Alternation - If Wave 2 is deep, Wave 4 will be shallow, and vice versa
            w2r = fib_details.get('W2_Ret_W1', (np.nan,))[0]; w2_deep = not pd.isna(w2r) and w2r > WAVE2_DEEP_RETRACE; w4_shallow = not pd.isna(w4r) and w4r < WAVE4_SHALLOW_RETRACE
            guidelines_passed["Alt_W2W4"] = (w2_deep and w4_shallow) or (not w2_deep and not w4_shallow)
            if guidelines_passed["Alt_W2W4"]:
                print(f"  Alternation guideline passed: W2 {'deep' if w2_deep else 'shallow'}, W4 {'shallow' if w4_shallow else 'deep'}")
                score += SCORE_GUIDELINE_PASS
            else:
                print(f"  Alternation guideline failed: W2 {'deep' if w2_deep else 'shallow'}, W4 {'shallow' if w4_shallow else 'deep'}")
        if length >= 5:
            # Check Wave 3 is never the shortest among waves 1, 3, and 5
            w1_len = abs(p_hl[1] - p_hl[0])
            w3_len = abs(p_hl[3] - p_hl[2])
            
            # Only check Wave 5 if it exists in the sequence
            if 5 in p_hl and 4 in p_hl and 5 in p and 4 in p:
                w5_len = abs(p_hl[5] - p_hl[4])
                if w3_len <= w1_len and w3_len <= w5_len:
                    print("  Rule violation: Wave 3 is the shortest among waves 1, 3, and 5")
                    rules_passed["W3_Shortest"] = False
                    return -1  # Critical rule violation
                else:
                    score += SCORE_RULE_PASS
            else:
                # If Wave 5 doesn't exist yet, only compare Wave 3 to Wave 1
                if w3_len <= w1_len:
                    print("  Potential rule violation: Wave 3 is shorter than Wave 1 (Wave 5 not yet formed)")
                    # Don't fail completely as Wave 5 might compensate
                    score -= SCORE_RULE_PASS / 2
                else:
                    score += SCORE_RULE_PASS
            
            # Check alternation principle between Wave 2 and Wave 4
            w2_retrace = calculate_fib_ratio(p_hl[0], p_hl[1], p_hl[2])
            w4_retrace = calculate_fib_ratio(p_hl[2], p_hl[3], p_hl[4])
            
            # Automatic alternation principle validation
            # If both Wave 2 and Wave 4 are deep retracements, this violates the alternation principle
            if w2_retrace > ALTERNATION_THRESHOLD and w4_retrace > ALTERNATION_THRESHOLD:
                print(f"  Guideline violation: Both Wave 2 ({w2_retrace:.3f}) and Wave 4 ({w4_retrace:.3f}) are deep retracements")
                print("  This violates the alternation principle - reducing score")
                guidelines_passed["Alt_W2W4"] = False
                score -= SCORE_GUIDELINE_PASS  # Penalize for violating alternation
            elif w2_retrace < SHALLOW_THRESHOLD and w4_retrace < SHALLOW_THRESHOLD:
                print(f"  Guideline caution: Both Wave 2 ({w2_retrace:.3f}) and Wave 4 ({w4_retrace:.3f}) are shallow retracements")
                print("  This is unusual for alternation principle - slightly reducing score")
                guidelines_passed["Alt_W2W4"] = False
                score -= SCORE_GUIDELINE_PASS * 0.5  # Smaller penalty
            else:
                # Proper alternation: if one is deep, the other is shallow
                if (w2_retrace > DEEP_THRESHOLD and w4_retrace < SHALLOW_THRESHOLD) or \
                   (w2_retrace < SHALLOW_THRESHOLD and w4_retrace > DEEP_THRESHOLD):
                    print(f"  Guideline confirmed: Good alternation between Wave 2 ({w2_retrace:.3f}) and Wave 4 ({w4_retrace:.3f})")
                    guidelines_passed["Alt_W2W4"] = True
                    score += SCORE_GUIDELINE_PASS  # Bonus for good alternation
            # Check Wave 5 relationships (only if Wave 5 exists)
            if 5 in p_hl and 4 in p_hl and 5 in p and 4 in p:
                w5r = w5_len / w1_len if w1_len > eps else np.nan
                fib_details['W5_vs_W1'] = (w5r, get_fib_level_str(w5r))
                score += SCORE_FIB_TARGET_HIT * is_close_to_fib(w5r, [0.618, 1.0, 1.618])
                
                # Check if Wave 5 equals Wave 1 (common in Elliott Wave patterns)
                guidelines_passed["W5_Eq_W1"] = not pd.isna(w5r) and abs(w5r - 1.0) < FIB_RATIO_TOLERANCE
                if guidelines_passed["W5_Eq_W1"]:
                    print("  Guideline passed: Wave 5 approximately equals Wave 1")
                    score += SCORE_GUIDELINE_PASS
        if has_volume: # Volume checks
            avg_vols = {}; valid_volume_calc = True
            def get_idx(pt):
                # Works for dict or Series, returns None if not found
                if isinstance(pt, dict):
                    return pt.get("idx", None)
                elif hasattr(pt, 'get'):
                    return pt.get("idx", None)
                return None

            for i in range(1, min(length, 6)):  # Only check up to the current length or 5, whichever is smaller
                if i not in p or i-1 not in p: 
                    valid_volume_calc = False
                    continue
                start_idx = get_idx(p[i-1])
                end_idx = get_idx(p[i])
                if start_idx is None or end_idx is None:
                    # Missing index info, skip this wave for volume calc
                    print(f"  Skipping volume calc for Wave {i}: missing 'idx' in wave point.")
                    continue
                if start_idx >= end_idx or start_idx < 0 or end_idx >= len(data_full):
                    valid_volume_calc = False
                    continue
                try:
                    wave_slice = data_full.iloc[start_idx:end_idx+1]
                    avg_vols[i] = wave_slice["Volume"].mean() if "Volume" in wave_slice.columns else 0
                except Exception as e:
                    print(f"  Volume calculation error for Wave {i}: {e}")
                    valid_volume_calc = False

            if valid_volume_calc:
                vol_details = {f"AvgVol_W{k}": v for k, v in avg_vols.items()}
                # Safely get volume values, defaulting to 0 if not available
                v1 = avg_vols.get(1, 0)
                v2 = avg_vols.get(2, 0)
                v3 = avg_vols.get(3, 0)
                v4 = avg_vols.get(4, 0)
                v5 = avg_vols.get(5, 0)
                if length >= 4 and 3 in avg_vols and 1 in avg_vols and avg_vols[3] > avg_vols[1]: 
                    guidelines_passed["Vol_W3>W1"] = True
                    score += SCORE_VOLUME_GUIDELINE
                    
                if length >= 6 and 5 in avg_vols and 3 in avg_vols and avg_vols[5] <= avg_vols[3]: 
                    guidelines_passed["Vol_W5<=W3"] = True
                    score += SCORE_VOLUME_GUIDELINE
                elif length >= 6 and 5 in avg_vols and 3 in avg_vols: 
                    guidelines_passed["Vol_W5<=W3"] = False
                # Check if correction volumes are less than impulse volumes (guideline)
                corr_vol_ok = not ((length >= 3 and 2 in avg_vols and 1 in avg_vols and avg_vols[2] >= avg_vols[1]) or 
                                   (length >= 5 and 4 in avg_vols and 3 in avg_vols and avg_vols[4] >= avg_vols[3]))
                guidelines_passed["Vol_Corr<Imp"] = corr_vol_ok; score += SCORE_VOLUME_GUIDELINE * corr_vol_ok
            else: guidelines_passed["Vol_W3>W1"] = guidelines_passed["Vol_W5<=W3"] = guidelines_passed["Vol_Corr<Imp"] = False
            
        # Check if all essential Elliott Wave rules are satisfied
        if not rules_passed["W2_Ret"] or not rules_passed["W4_Overlap"] or not rules_passed["W3_Shortest"]: 
            print("  Failed essential Elliott Wave rules check")
            return -1
        return score, rules_passed, guidelines_passed, fib_details, vol_details

    for i in range(len(wave_points) - 2): # Main loop through points
        p = {}; p[0] = wave_points.iloc[i]; p[1] = wave_points.iloc[i+1]; p[2] = wave_points.iloc[i+2]
        is_up = p[1]['High'] > p[0]['High'];
        if not is_up and p[1]['Low'] < p[0]['Low']: is_up = False
        elif p[1]['Close'] > p[0]['Close']: is_up = True
        elif p[1]['Close'] < p[0]['Close']: is_up = False
        else: continue
        if (is_up and p[2]['Low'] >= p[1]['Low']) or (not is_up and p[2]['High'] <= p[1]['High']): continue

        current_best_len_for_i = 0
        score_result_012 = score_sequence({0:p[0], 1:p[1], 2:p[2]}, is_up, 3, data_full)
        if not (isinstance(score_result_012, int) and score_result_012 == -1):
            current_best_len_for_i = 3
            for k in range(3, 6):
                if i + k >= len(wave_points): break
                p[k] = wave_points.iloc[i+k]
                correct_direction = True
                if k % 2 == 1: # Impulse 3 or 5
                    if (is_up and p[k]['High'] <= p[k-1]['High']) or (not is_up and p[k]['Low'] >= p[k-1]['Low']): correct_direction = False
                else: # Corrective 4
                    if (is_up and p[k]['Low'] >= p[k-1]['Low']) or (not is_up and p[k]['High'] <= p[k-1]['High']): correct_direction = False
                if not correct_direction: 
                    p.pop(k)
                    break
                    
                # Create a subset of points up to the current wave
                points_subset = {idx: pt for idx, pt in p.items() if idx <= k}
                
                # Score the sequence with proper error handling for incomplete waves
                try:
                    score_result_extended = score_sequence(points_subset, is_up, k+1, data_full)
                except KeyError as e:
                    print(f"  KeyError in score_sequence for wave {k+1}: {e}")
                    score_result_extended = -1  # Invalid sequence due to missing required points
                    
                if isinstance(score_result_extended, int) and score_result_extended == -1: 
                    p.pop(k)
                    break
                else: 
                    current_best_len_for_i = k + 1

        if current_best_len_for_i >= 3:
            final_p = {idx: pt for idx, pt in p.items() if idx < current_best_len_for_i}
            final_score_result = score_sequence(final_p, is_up, current_best_len_for_i, data_full)
            if not (isinstance(final_score_result, int) and final_score_result == -1):
                 score, rules, guidelines, fibs, vols = final_score_result
                 sequence_details = {"score": score, "length": current_best_len_for_i, "last_label": str(current_best_len_for_i - 1),
                                     "last_point": final_p[current_best_len_for_i - 1], "start_index_iloc": i, "is_upward": is_up,
                                     "points": final_p, "fib_ratios": fibs, "guidelines": guidelines, "rules_passed": rules, "volume_details": vols}
                 if score > best_sequence_score or (abs(score - best_sequence_score) < 5 and current_best_len_for_i > best_sequence_details["length"]):
                     best_sequence_score = score; best_sequence_details = sequence_details

    if best_sequence_details: # Process best sequence
        analysis_results.update({"found_impulse": True, "details": best_sequence_details, "last_label": best_sequence_details['last_label'], "last_point": best_sequence_details['last_point']})
        identified_sequence_last_label = best_sequence_details['last_label']; identified_sequence_last_point = best_sequence_details['last_point']
        for k in range(best_sequence_details['length']):
            point_timestamp = best_sequence_details['points'][k].name
            if point_timestamp in wave_points.index: wave_points.loc[point_timestamp, 'EW_Label'] = str(k)
        if identified_sequence_last_label == '5': # ABC Correction Guess
             try: p5_iloc = wave_points.index.get_loc(identified_sequence_last_point.name); start_abc_iloc = p5_iloc + 1
             except KeyError: start_abc_iloc = -1
             if start_abc_iloc != -1 and start_abc_iloc + 2 < len(wave_points):
                 abc_pts = [wave_points.iloc[start_abc_iloc + j] for j in range(3)]; p5_pt = identified_sequence_last_point
                 is_impulse_up = best_sequence_details['is_upward']; is_abc_zigzag = False
                 if is_impulse_up: is_abc_zigzag = (abc_pts[0]['Close'] < p5_pt['Close'] and abc_pts[1]['Close'] > abc_pts[0]['Low'] and abc_pts[1]['Close'] < p5_pt['High'] and abc_pts[2]['Close'] < abc_pts[0]['Low'])
                 else: is_abc_zigzag = (abc_pts[0]['Close'] > p5_pt['Close'] and abc_pts[1]['Close'] < abc_pts[0]['High'] and abc_pts[1]['Close'] > p5_pt['Low'] and abc_pts[2]['Close'] > abc_pts[0]['High'])
                 if is_abc_zigzag:
                     print("  Speculation: Found potential A-B-C correction pattern after W5.")
                     abc_labels = ['(A?)', '(B?)', '(C?)']; analysis_results["last_label"] = '(C?)'; analysis_results["last_point"] = abc_pts[2]
                     for k, label in enumerate(abc_labels):
                         if abc_pts[k].name in wave_points.index: wave_points.loc[abc_pts[k].name, 'EW_Label'] = label
                     analysis_results["details"]["correction_guess"] = {'A': abc_pts[0], 'B': abc_pts[1], 'C': abc_pts[2]}
                     identified_sequence_last_label = '(C?)'; identified_sequence_last_point = abc_pts[2]

    # Final Labeling Pass
    p_counter = 0
    final_last_label = analysis_results["last_label"] # Start with label from impulse/correction
    final_last_point_overall = wave_points.iloc[-1] if not wave_points.empty else None
    for i in range(len(wave_points)):
        idx_name = wave_points.index[i]
        if wave_points.loc[idx_name, 'EW_Label'] == "": wave_points.loc[idx_name, 'EW_Label'] = f"P{p_counter}"; p_counter += 1
        if i == len(wave_points) - 1: # If this is the absolute last point
             current_label = wave_points.loc[idx_name, 'EW_Label']
             # Update final label if the identified sequence ended earlier OR if none was found
             if (identified_sequence_last_point is not None and idx_name != identified_sequence_last_point.name) or identified_sequence_last_point is None:
                 final_last_label = current_label
    if not analysis_results.get("found_impulse") and final_last_point_overall is not None:
        final_last_label = final_last_point_overall['EW_Label']
    analysis_results["last_label"] = final_last_label
    analysis_results["last_point_overall"] = final_last_point_overall

    # Print Summary (Simplified - full print logic assumed correct from prev code)
    if analysis_results.get("found_impulse", False): print("\n--- [EW Analysis Summary] ---"); print(f"  Best Sequence Found...") # etc.
    if 'EW_Label' not in wave_points.columns and not wave_points.empty: wave_points['EW_Label'] = [f"P{i}" for i in range(len(wave_points))]
    return wave_points, analysis_results


# --- Plotting Function ---
# MODIFIED with more flexible projections
def plot_chart(data, identified_waves, analysis_results, ticker="Stock", interval=""):
    """Plots chart with analysis, flexible projections, RSI & CONFLUENCE checks."""
    print("\n[Plotting] Generating v5.6 Chart (Flexible Projections)...") # Version update
    if data is None or data.empty: print("  Error: No data provided to plot."); return None

    # (Setup subplots - remains the same)
    rows = 1; row_heights = [0.75]; volume_row_index = 0; rsi_row_index = 0
    has_volume = 'Volume' in data.columns and not data['Volume'].isnull().all()
    has_rsi = PLOT_RSI and 'RSI' in data.columns and not data['RSI'].isnull().all()
    if has_rsi: rows += 1; rsi_row_index = rows; row_heights.append(0.15)
    if has_volume: rows += 1; volume_row_index = rows; row_heights.append(0.10)
    row_heights[0] = 1.0 - sum(row_heights[1:]) - (0.02 * (rows -1))
    if not math.isclose(sum(row_heights), 1.0, abs_tol=0.01):
        if rows == 3: row_heights = [0.7, 0.15, 0.15]
        elif rows == 2 and has_rsi: row_heights = [0.75, 0.25]
        elif rows == 2 and has_volume: row_heights = [0.8, 0.2]
        else: row_heights = [1.0]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=row_heights,
                         subplot_titles=("Price Action & EW Analysis", "RSI" if has_rsi else ("Volume" if has_volume and rows==2 else None), "Volume" if has_volume and rows==3 else None))

    # (Plot Candlesticks, MAs - remains the same)
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name=f'{ticker} Price', increasing_line_color='rgba(0, 200, 0, 0.8)', decreasing_line_color='rgba(255, 50, 50, 0.8)'), row=1, col=1)
    ma_colors = ['#FFD700', '#ADFF2F'];
    for i, p in enumerate(MA_PERIODS):
        if f'SMA_{p}' in data.columns and not data[f'SMA_{p}'].isnull().all():
            fig.add_trace(go.Scatter(x=data.index, y=data[f'SMA_{p}'], mode='lines', name=f'SMA {p}', line=dict(color=ma_colors[i % len(ma_colors)], width=1.0, dash='dot')), row=1, col=1)

    # (Plot Wave Points & Path - remains the same)
    plot_waves = identified_waves is not None and not identified_waves.empty
    if plot_waves and 'EW_Label' in identified_waves.columns:
        labels, colors, symbols, text_positions, wave_coords_x, wave_coords_y = [], [], [], [], [], []
        for counter, (idx, r) in enumerate(identified_waves.iterrows()):
            lbl = r['EW_Label'] if pd.notna(r['EW_Label']) and r['EW_Label'] != "" else f"P{counter}"
            labels.append(lbl); wave_coords_x.append(idx); wave_coords_y.append(r['Close'])
            is_peak = False # Simplified peak/trough logic for brevity
            if isinstance(lbl, str) and lbl.isdigit(): is_peak = (int(lbl) % 2 != 0) if analysis_results.get("details", {}).get("is_upward", True) else (int(lbl) % 2 == 0 and int(lbl) != 0)
            elif isinstance(lbl, str) and lbl.startswith('(B?)'): is_peak = True
            elif isinstance(lbl, str) and (lbl.startswith('(A?)') or lbl.startswith('(C?)')): is_peak = False
            else: is_peak = (counter % 2 != 0)
            text_positions.append("bottom center" if is_peak else "top center")
            if isinstance(lbl, str) and lbl.isdigit(): colors.append('cyan'); symbols.append('circle')
            elif isinstance(lbl, str) and lbl.startswith('('): colors.append('magenta'); symbols.append('diamond')
            else: colors.append('grey'); symbols.append('cross')
        fig.add_trace(go.Scatter(x=wave_coords_x, y=wave_coords_y, mode='markers+text', name='Wave Points', marker=dict(color=colors, size=9, symbol=symbols, line=dict(width=1, color='white')), text=labels, textposition=text_positions, textfont=dict(size=12, color='white', family="Arial Black, sans-serif")), row=1, col=1)
        fig.add_trace(go.Scatter(x=wave_coords_x, y=wave_coords_y, mode='lines', name='Wave Path', showlegend=False, line=dict(color='rgba(0, 255, 255, 0.5)', dash='dot', width=1.5)), row=1, col=1)

    # (Plot Fib Annotations, Channels, Invalidation Levels - remains the same)
    wave_annotations = []
    if analysis_results.get("found_impulse", False):
        # (Fib annotation logic remains same...)
        details = analysis_results["details"]; points = details.get("points", {}); fibs = details.get("fib_ratios", {}); is_up = details.get("is_upward", True)
        def add_fib_annot(pt_key_str, fib_key, txt, ay_offset):
             point_data = points.get(int(pt_key_str)) if pt_key_str.isdigit() and int(pt_key_str) in points else None
             if fib_key in fibs and point_data is not None:
                 fib_val_tuple = fibs[fib_key]
                 if isinstance(fib_val_tuple, (list, tuple)) and len(fib_val_tuple) == 2:
                     ratio_val, ratio_str = fib_val_tuple
                     if ratio_val is not None and isinstance(ratio_val, (int,float)) and not pd.isna(ratio_val):
                         annotation = dict(x=point_data.name, y=point_data['Close'], text=f"{txt}: {ratio_str}", showarrow=True, arrowhead=1, ax=10 if ay_offset < 0 else -10, ay=ay_offset, font=dict(size=9, color="yellow"), bgcolor="rgba(0,0,0,0.6)", row=1, col=1)
                         wave_annotations.append(annotation)
        add_fib_annot('2', 'W2_Ret_W1', "W2", -30 if is_up else 30); add_fib_annot('3', 'W3_Ext_W1', "W3", -35 if is_up else 35); add_fib_annot('4', 'W4_Ret_W3', "W4", 30 if is_up else -30); add_fib_annot('5', 'W5_vs_W1', "W5", -35 if is_up else 35)
        for annot in wave_annotations: fig.add_annotation(**annot)
        # (Channel/Invalidation plotting assumed correct from prev code...)
        p0, p1, p2, p3, p4 = [points.get(i) for i in range(5)]; last_idx_date = data.index[-1]


    # --- TARGET BOX & PATH LINE LOGIC (WITH FLEXIBLE PROJECTIONS) ---
    print("  Generating Projections with Confluence Checks (Flexible v5.6)...")
    projection_basis_label = analysis_results.get("last_label")
    projection_basis_point = analysis_results.get("last_point") # Raw Series/row object
    last_overall_point = analysis_results.get("last_point_overall") # Raw Series/row object

    projection_plotted = False
    projection_path_points = []

    major_fib_levels_data = {}
    if not data.empty:
        min_price = data['Low'].min(); max_price = data['High'].max()
        if pd.notna(min_price) and pd.notna(max_price) and max_price > min_price:
            major_fib_levels_data = calculate_fibonacci_retracements(min_price, max_price)

    if last_overall_point is not None and projection_basis_point is not None and projection_basis_label is not None and analysis_results.get("found_impulse", False):
        current_price = data['Close'].iloc[-1] if not data.empty else None # Price at end of analysis range
        projection_path_points.append( (projection_basis_point.name, projection_basis_point['Close'], f"Start ({projection_basis_label})") )

        try: # Calculate time offsets
            valid_indices = data.index[data.index.notnull()]
            if len(valid_indices) > 1: avg_time_delta = pd.to_timedelta(np.median(np.diff(valid_indices)))
            else: avg_time_delta = pd.Timedelta(weeks=1) if 'wk' in interval else pd.Timedelta(days=1)
            if avg_time_delta <= pd.Timedelta(0): avg_time_delta = pd.Timedelta(days=1)
            min_offset = pd.Timedelta(days=max(7, avg_time_delta.days * 3))
            offset_primary = max(avg_time_delta * TARGETBOX_CANDLES_PRIMARY, min_offset)
            offset_secondary = max(avg_time_delta * TARGETBOX_CANDLES_SECONDARY, min_offset * 2)
            offset_tertiary = max(avg_time_delta * TARGETBOX_CANDLES_TERTIARY, min_offset * 3) # Use new constant
        except Exception as e: # Fallback offsets
            print(f"  Warning: Could not calculate average time delta: {e}. Using fixed offsets.")
            offset_primary = pd.Timedelta(weeks=TARGETBOX_CANDLES_PRIMARY) if 'wk' in interval else pd.Timedelta(days=TARGETBOX_CANDLES_PRIMARY)
            offset_secondary = pd.Timedelta(weeks=TARGETBOX_CANDLES_SECONDARY) if 'wk' in interval else pd.Timedelta(days=TARGETBOX_CANDLES_SECONDARY)
            offset_tertiary = pd.Timedelta(weeks=TARGETBOX_CANDLES_TERTIARY) if 'wk' in interval else pd.Timedelta(days=TARGETBOX_CANDLES_TERTIARY)

        box_start_date = projection_basis_point.name + avg_time_delta / 4 if avg_time_delta and pd.notna(projection_basis_point.name) else data.index[-1] + pd.Timedelta(days=1)
        details = analysis_results.get("details", {})
        points = details.get("points", {})
        is_impulse_up = details.get("is_upward", True)

        # --- Internal Function to Draw Target Box (remains the same) ---
        # (draw_target_box function code as provided before)
        def draw_target_box(target_wave_label, low_level, high_level, color_rgb, time_offset,
                             fib_levels_str="", basis_point=None, current_price=None, primary=True,
                             basis_info_override=None, full_data=None, major_fibs_data=None, custom_start_date=None,
                             entry_price=None):  # Added entry_price parameter
             """Adds a target box shape and annotation, returns center coords."""
             if low_level is None or high_level is None or math.isnan(low_level) or math.isnan(high_level) or basis_point is None: return None
             target_low, target_high = min(low_level, high_level), max(low_level, high_level)
             if abs(target_high - target_low) < 1e-6: target_high += 0.01 * target_high if target_high > 1 else 0.01
             actual_start_date = custom_start_date if custom_start_date is not None else box_start_date
             if pd.isna(actual_start_date) or pd.isna(time_offset) or time_offset <= pd.Timedelta(0): return None
             try: box_end_date = actual_start_date + time_offset
             except Exception: return None
             box_opacity = 0.18 if primary else 0.12; line_style = "dash" if primary else "dot"
             label_prefix = "Proj." if primary else "Est."; font_size = 10 if primary else 9
             status_text = ""
             
             # Calculate percentage changes if entry price is available
             pct_change_low, pct_change_high = "", ""
             if entry_price is not None and not pd.isna(entry_price) and entry_price > 0:
                 pct_change_low = f" ({((target_low - entry_price) / entry_price * 100):+.2f}%)"
                 pct_change_high = f" ({((target_high - entry_price) / entry_price * 100):+.2f}%)"
             
             if current_price is not None and not pd.isna(current_price):
                 wave_char = None; moving_down_expected = False
                 if isinstance(target_wave_label, str):
                     label_part = target_wave_label.split()[-1]
                     if label_part.isdigit(): wave_char = label_part
                     elif label_part in ['A', 'B', 'C']: wave_char = label_part
                 if wave_char:
                     if wave_char in ['2', '4', 'A', 'C']: moving_down_expected = is_impulse_up
                     elif wave_char in ['3', '5', 'B']: moving_down_expected = not is_impulse_up
                 if moving_down_expected:
                     if current_price < target_low: status_text = "[Status: Exceeded]"
                     elif current_price <= target_high: status_text = "[Status: Hit]"
                     else: status_text = "[Status: Pending]"
                 else:
                     if current_price > target_high: status_text = "[Status: Exceeded]"
                     elif current_price >= target_low: status_text = "[Status: Hit]"
                     else: status_text = "[Status: Pending]"
             confluence_points = []; absolute_price_tolerance = target_high * CONFLUENCE_PRICE_TOLERANCE_FACTOR
             if full_data is not None and not full_data.empty:
                 last_row = full_data.iloc[-1]
                 for p in MA_PERIODS:
                     ma_col = f'SMA_{p}'; ma_val = last_row.get(ma_col)
                     if ma_val is not None and not pd.isna(ma_val) and (target_low - absolute_price_tolerance <= ma_val <= target_high + absolute_price_tolerance):
                         confluence_points.append(f"SMA{p} ({ma_val:.2f})")
                 if major_fibs_data:
                     for fib_level, fib_price in major_fibs_data.items():
                         if (target_low - absolute_price_tolerance <= fib_price <= target_high + absolute_price_tolerance):
                             confluence_points.append(f"Major {get_fib_level_str(fib_level)} ({fib_price:.2f})")
                 if 'RSI' in last_row and not pd.isna(last_row['RSI']):
                     last_rsi = last_row['RSI']
                     is_impulse_wave = False; is_corrective_wave = False; wave_char = None
                     if isinstance(target_wave_label, str):
                         label_part = target_wave_label.split()[-1]
                         if label_part.isdigit(): wave_char = label_part
                         elif label_part in ['A', 'B', 'C']: wave_char = label_part
                     if wave_char:
                         if wave_char in ['1', '3', '5', 'B']: is_impulse_wave = True
                         if wave_char in ['2', '4', 'A', 'C']: is_corrective_wave = True
                     rsi_supports = False; moving_down_expected = False # Recalculate expected direction locally
                     if wave_char in ['2', '4', 'A', 'C']: moving_down_expected = is_impulse_up
                     elif wave_char in ['3', '5', 'B']: moving_down_expected = not is_impulse_up
                     if is_impulse_wave and not moving_down_expected and last_rsi > RSI_OVERBOUGHT: rsi_supports = True
                     if is_impulse_wave and moving_down_expected and last_rsi < RSI_OVERSOLD: rsi_supports = True
                     if is_corrective_wave and not moving_down_expected and last_rsi < RSI_OVERSOLD: rsi_supports = True
                     if is_corrective_wave and moving_down_expected and last_rsi > RSI_OVERBOUGHT: rsi_supports = True
                     if rsi_supports: confluence_points.append(f"RSI {last_rsi:.1f}")
             confluence_text = f"<br><i>Confluence: {', '.join(sorted(list(set(confluence_points))))}</i>" if confluence_points else ""
             # Add target box with enhanced properties for dragging
             fig.add_shape(
                 type="rect", 
                 xref="x", 
                 yref="y", 
                 layer="above",  # Changed from 'below' to 'above' for better interaction
                 x0=actual_start_date, 
                 y0=target_low, 
                 x1=box_end_date, 
                 y1=target_high,
                 line=dict(color=f"rgba({color_rgb}, 0.7)", width=1, dash=line_style), 
                 fillcolor=f"rgba({color_rgb}, {box_opacity})", 
                 row=1, 
                 col=1,
                 editable=True,  # Make the shape editable
                 name=f"target_box_{target_wave_label}",  # Add a name for identification
             )
             basis_date_str = basis_point.name.strftime('%Y-%m-%d') if basis_point is not None and pd.notna(basis_point.name) else "N/A"
             # Create a more informative basis text for traders
             if basis_info_override:
                 basis_text = basis_info_override
             else:
                 # For wave projections, show the date and price level of the basis point
                 basis_price = basis_point['Close'] if basis_point is not None else None
                 if basis_price is not None:
                     basis_text = f"(Basis: W{projection_basis_label} {basis_date_str} @ {basis_price:.2f})"
                 else:
                     basis_text = f"(Basis: W{projection_basis_label} {basis_date_str})"
             
             # Update annotation text to include percentage changes
             price_range_text = f"{target_low:.2f}{pct_change_low} - {target_high:.2f}{pct_change_high}"
             annotation_text = (f"<b>{label_prefix} {target_wave_label} Zone</b> {status_text}<br>{fib_levels_str} {basis_text}<br>{price_range_text}{confluence_text}")
             
             fig.add_annotation(x=actual_start_date + (box_end_date - actual_start_date) / 2, y=target_high, text=annotation_text, showarrow=False,
                                font=dict(color="white", size=font_size), bgcolor="rgba(50,50,50,0.8)", bordercolor=f"rgba({color_rgb}, 0.8)", borderwidth=1,
                                xanchor="center", yanchor="bottom", row=1, col=1)
             center_price = (target_low + target_high) / 2; center_date = actual_start_date + (box_end_date - actual_start_date) / 2
             return (center_date, center_price, target_wave_label)

        # --- Projection Logic ---
        p0=points.get(0); p1=points.get(1); p2=points.get(2); p3=points.get(3); p4=points.get(4); p5=points.get(5)
        center_w2 = None; center_w3 = None; center_w4 = None; center_w5 = None
        center_wa = None; center_wb = None; center_wc = None

        common_args = {'full_data': data, 'major_fibs_data': major_fib_levels_data, 'current_price': current_price}

        current_label_num = -1
        if isinstance(projection_basis_label, str) and projection_basis_label.isdigit():
            try: current_label_num = int(projection_basis_label)
            except ValueError: current_label_num = -1

        # --- If W1 finished -> Project W2 (Primary), estimate W3 (Secondary) ---
        if current_label_num == 1 and p0 is not None and p1 is not None:
            print("  Projecting W2 (Primary) -> Est. W3...")
            p0_hl = p0['Low'] if is_impulse_up else p0['High']; p1_hl = p1['High'] if is_impulse_up else p1['Low']
            ret2 = calculate_fibonacci_retracements(p0_hl, p1_hl)
            # Retroceso típico de Onda 2: 50%, 61.8% o 76.4% de Onda 1
            w2_t1, w2_t2 = ret2.get(0.500), ret2.get(0.764)
            if w2_t1 is None or w2_t2 is None:
                print(f"  Warning: Invalid target prices for W2: low={w2_t1}, high={w2_t2}")
                w2_t1, w2_t2 = p1_hl * 0.5, p1_hl * 0.764
            center_w2 = draw_target_box("W2", w2_t1, w2_t2, "255, 0, 0", offset_primary, "50-76.4% W1 Ret", p1, primary=True, **common_args)
            if center_w2:
                hypo_p2_price = center_w2[1]
                
                # Print debug info to verify direction
                print(f"  DEBUG: is_impulse_up={is_impulse_up}, p0_hl={p0_hl}, p1_hl={p1_hl}, hypo_p2_price={hypo_p2_price}")
                
                # Proyección típica de Onda 3: Extensión del 161.8% del tamaño de la Onda 1
                # Get ATR value for volatility-adjusted targets if available
                atr_value = None
                if 'ATR' in data.columns and not data['ATR'].isnull().all():
                    atr_value = data['ATR'].iloc[-1]
                    print(f"  Trader insight: Using current ATR ({atr_value:.2f}) to adjust target zones")
                
                # Calculate Wave 3 extension in the direction of Wave 1 with ATR adjustment
                ext3 = calculate_fibonacci_extensions(p0_hl, p1_hl, hypo_p2_price, atr_value)
                
                # For uptrend: W3 should be higher than W1
                # For downtrend: W3 should be lower than W1
                w3_t1s, w3_t2s = ext3.get(1.618), ext3.get(2.618)
                
                # Handle potential None values from Fibonacci calculations
                if w3_t1s is None or w3_t2s is None:
                    print(f"  Warning: Unable to calculate valid W3 targets. Using fallback values.")
                    # Fallback: use simple percentage extensions based on W1 length
                    w1_length = abs(p1_hl - p0_hl)
                    direction = 1 if is_impulse_up else -1
                    w3_t1s = hypo_p2_price + (w1_length * 1.618 * direction)
                    w3_t2s = hypo_p2_price + (w1_length * 2.618 * direction)
                
                # Ensure targets are ordered correctly (lower value first for drawing)
                if w3_t1s is not None and w3_t2s is not None and w3_t1s > w3_t2s:
                    w3_t1s, w3_t2s = w3_t2s, w3_t1s
                    
                w3_start_date = center_w2[0]
                center_w3 = draw_target_box("W3", w3_t1s, w3_t2s, "0, 200, 0", offset_secondary, 
                                          "161.8-261.8% W1 Ext", p1, 
                                          primary=False, basis_info_override="(Est. from Proj. W2)", 
                                          custom_start_date=w3_start_date, **common_args)

        # --- If W2 finished -> Project W3 (Primary), estimate W4, W5 (Secondary) ---
        elif current_label_num == 2 and p0 is not None and p1 is not None and p2 is not None:
            print("  Projecting W3 (Primary) -> Est. W4 -> Est. W5...")
            p0_hl = p0['Low'] if is_impulse_up else p0['High']; p1_hl = p1['High'] if is_impulse_up else p1['Low']
            p2_close = p2['Close']
            
            # Calculate W1 and W2 length and time for proportions
            w1_length = abs(p1_hl - p0_hl)
            w2_length = abs(p2_close - p1_hl)
            w2_w1_ratio = w2_length / w1_length if w1_length > 0 else 0
            
            # Get ATR value for volatility-adjusted targets if available
            atr_value = None
            if 'ATR' in data.columns and not data['ATR'].isnull().all():
                atr_value = data['ATR'].iloc[-1]
                print(f"  Trader insight: Using current ATR ({atr_value:.2f}) to adjust target zones")
            
            # Trader thinking: W2 depth affects W3 projection
            # Deeper W2 often leads to stronger W3
            w3_min_ext = 1.618  # Base extension
            w3_max_ext = 2.618  # Base maximum
            
            # Adjust W3 projection based on W2 depth
            if w2_w1_ratio > 0.618:  # Deep W2
                print("  Trader insight: Deep W2 (>61.8%) suggests potentially stronger W3")
                w3_min_ext = 1.618
                w3_max_ext = 3.618  # Extend upper target for deep W2
            elif w2_w1_ratio < 0.382:  # Shallow W2
                print("  Trader insight: Shallow W2 (<38.2%) may lead to moderate W3")
                w3_min_ext = 1.382
                w3_max_ext = 2.0
            
            # Proyección típica de Onda 3: Extensión del 161.8% del tamaño de la Onda 1
            # But adjusted based on W2 characteristics and market volatility
            print(f"  DEBUG: is_impulse_up={is_impulse_up}, p0_hl={p0_hl}, p1_hl={p1_hl}, p2_close={p2_close}")
            
            # Calculate Wave 3 extension in the direction of Wave 1 with ATR adjustment
            ext3 = calculate_fibonacci_extensions(p0_hl, p1_hl, p2_close, atr_value)
            
            # Get the extension targets based on W2 depth
            w3_t1, w3_t2 = ext3.get(w3_min_ext), ext3.get(w3_max_ext)
            
            # Handle potential None values from Fibonacci calculations
            if w3_t1 is None or w3_t2 is None:
                print(f"  Warning: Unable to calculate valid W3 targets. Using fallback values.")
                # Fallback: use simple percentage extensions based on W1 length
                w1_length = abs(p1_hl - p0_hl)
                direction = 1 if is_impulse_up else -1
                w3_t1 = p2_close + (w1_length * w3_min_ext * direction)
                w3_t2 = p2_close + (w1_length * w3_max_ext * direction)
            
            # Ensure targets are ordered correctly (lower value first for drawing)
            if w3_t1 is not None and w3_t2 is not None and w3_t1 > w3_t2:
                w3_t1, w3_t2 = w3_t2, w3_t1
            center_w3 = draw_target_box("W3", w3_t1, w3_t2, "0, 200, 0", offset_primary, 
                                       f"{w3_min_ext}-{w3_max_ext}x W1 Ext", p2, primary=True, **common_args)
            
            if center_w3:
                hypo_p3_price = center_w3[1]; p2_hl = p2['Low'] if is_impulse_up else p2['High']
                
                # Trader thinking: If W2 was deep, W4 is likely shallow (alternation principle)
                w4_min_ret = 0.236 if w2_w1_ratio > 0.5 else 0.382
                w4_max_ret = 0.382 if w2_w1_ratio > 0.5 else 0.5
                
                print(f"  Trader insight: W2/W1 ratio = {w2_w1_ratio:.3f}, projecting W4 retracement at {w4_min_ret}-{w4_max_ret}")
                
                # Get ATR value for volatility-adjusted targets if available
                atr_value = None
                if 'ATR' in data.columns and not data['ATR'].isnull().all():
                    atr_value = data['ATR'].iloc[-1]
                    print(f"  Trader insight: Using current ATR ({atr_value:.2f}) to adjust W4 target zone")
                
                # Calculate retracements with ATR adjustment for more realistic targets
                ret4 = calculate_fibonacci_retracements(p2_hl, hypo_p3_price, atr_value)
                # Retroceso típico de Onda 4: 38.2% o 50% de la Onda 3, adjusted for alternation
                w4_t1s, w4_t2s = ret4.get(w4_min_ret), ret4.get(w4_max_ret)
                
                # Check that W4 doesn't overlap with W1 (Elliott Wave rule)
                w1_end_level = p1_hl
                if (is_impulse_up and w4_t2s <= w1_end_level) or (not is_impulse_up and w4_t2s >= w1_end_level):
                    print("  Trader alert: Adjusting W4 target to avoid W1 overlap (Elliott Wave rule)")
                    # Adjust W4 to avoid overlap
                    buffer = abs(w1_end_level - p2_hl) * 0.05  # 5% buffer
                    if is_impulse_up:
                        w4_t2s = max(w4_t2s, w1_end_level + buffer)
                    else:
                        w4_t2s = min(w4_t2s, w1_end_level - buffer)
                
                w4_start_date = center_w3[0]
                center_w4 = draw_target_box("W4", w4_t1s, w4_t2s, "255, 165, 0", offset_secondary, 
                                           f"{w4_min_ret*100:.1f}-{w4_max_ret*100:.1f}% W3 Ret (Est.)", p2, 
                                           primary=False, basis_info_override="(Est. from Proj. W3)", 
                                           custom_start_date=w4_start_date, **common_args)
                
                if center_w4:
                    hypo_p4_price = center_w4[1]
                    
                    # Trader thinking: W5 projection based on overall wave structure
                    # For Wave 5, calculate the distance from Wave 0 to Wave 3 (the 1-3 range)
                    w1_length = abs(p1_hl - p0_hl)
                    w1_to_w3_range = hypo_p3_price - p0_hl
                    w1_length = abs(p1_hl - p0_hl)
                    
                    # Determine direction multiplier based on impulse direction
                    direction = 1 if is_impulse_up else -1
                    
                    # Extensión de Onda 5: 61.8%, 100% o 161.8% del recorrido 1-3
                    # Trader insight: W5 often relates to both W1 length and the overall pattern
                    if abs(hypo_p3_price - p2_close) > 2 * w1_length:  # If W3 is extended
                        print("  Trader insight: Extended W3 detected, W5 likely to be shorter (equality with W1)")
                        # When W3 is extended, W5 often equals W1
                        w5_t1s = hypo_p4_price + (w1_length * 0.618 * direction)
                        w5_t2s = hypo_p4_price + (w1_length * 1.0 * direction)
                        w5_label = "61.8-100% of W1 (Est.)"
                    else:  # Normal W3
                        print("  Trader insight: Normal W3, W5 projected using W1-W3 range")
                        # Use the sign of w1_to_w3_range to determine direction
                        w5_t1s = hypo_p4_price + (abs(w1_to_w3_range) * 0.618 * direction)
                        w5_t2s = hypo_p4_price + (abs(w1_to_w3_range) * 1.618 * direction)
                        w5_label = "61.8-161.8% W1-W3 Ext (Est.)"
                        
                    # Ensure targets are ordered correctly (lower value first for drawing)
                    if w5_t1s is not None and w5_t2s is not None and w5_t1s > w5_t2s:
                        w5_t1s, w5_t2s = w5_t2s, w5_t1s
                    
                    w5_start_date = center_w4[0]
                    center_w5 = draw_target_box("W5", w5_t1s, w5_t2s, "173, 255, 47", offset_tertiary, 
                                               w5_label, p2, primary=False, 
                                               basis_info_override="(Est. from Est. W4)", 
                                               custom_start_date=w5_start_date, **common_args)

        # --- If W3 finished -> Project W4 (Primary), estimate W5 (Secondary) ---
        elif current_label_num == 3 and p0 is not None and p1 is not None and p2 is not None and p3 is not None:
            print("  Projecting W4 (Primary) -> Est. W5...")
            p0_hl = p0['Low'] if is_impulse_up else p0['High']
            p1_hl = p1['High'] if is_impulse_up else p1['Low']
            p2_hl = p2['Low'] if is_impulse_up else p2['High']
            p3_hl = p3['High'] if is_impulse_up else p3['Low']
            
            # Analyze wave characteristics for trader insights
            w1_length = abs(p1_hl - p0_hl)
            w2_length = abs(p2_hl - p1_hl)
            w3_length = abs(p3_hl - p2_hl)
            w3_w1_ratio = w3_length / w1_length if w1_length > 0 else 0
            w2_w1_ratio = w2_length / w1_length if w1_length > 0 else 0
            
            print(f"  Trader analysis: W3/W1 ratio = {w3_w1_ratio:.2f}, W2/W1 ratio = {w2_w1_ratio:.2f}")
            
            # Trader thinking: W4 projection based on W2 and W3 characteristics
            # Alternation principle: If W2 was deep, W4 should be shallow and vice versa
            # W4 shouldn't overlap W1 (Elliott Wave rule)
            
            # Determine W4 retracement levels based on alternation principle
            if w2_w1_ratio > 0.5:  # Deep W2
                print("  Trader insight: Deep W2 suggests shallow W4 (alternation principle)")
                w4_min_ret = 0.236
                w4_max_ret = 0.382
            elif w2_w1_ratio < 0.382:  # Shallow W2
                print("  Trader insight: Shallow W2 suggests deeper W4 (alternation principle)")
                w4_min_ret = 0.382
                w4_max_ret = 0.5
            else:  # Normal W2
                print("  Trader insight: Normal W2 suggests balanced W4 retracement")
                w4_min_ret = 0.236
                w4_max_ret = 0.382
            
            # If W3 is extended, W4 tends to be more complex but still respects Fibonacci levels
            if w3_w1_ratio > 1.618:
                print("  Trader insight: Extended W3 (>1.618 × W1) may lead to complex W4 correction")
            
            # Get ATR value for volatility-adjusted targets if available
            atr_value = None
            if 'ATR' in data.columns and not data['ATR'].isnull().all():
                atr_value = data['ATR'].iloc[-1]
                print(f"  Trader insight: Using current ATR ({atr_value:.2f}) to adjust W4 target zone")
                
            # Calculate retracements with ATR adjustment for more realistic targets
            ret4 = calculate_fibonacci_retracements(p2_hl, p3_hl, atr_value)
            w4_t1, w4_t2 = ret4.get(w4_min_ret), ret4.get(w4_max_ret)
            
            # Check W4 doesn't overlap W1 (Elliott Wave rule)
            w1_end_level = p1_hl
            if (is_impulse_up and w4_t2 <= w1_end_level) or (not is_impulse_up and w4_t2 >= w1_end_level):
                print("  Trader alert: Adjusting W4 target to avoid W1 overlap (Elliott Wave rule)")
                # Adjust W4 to avoid overlap
                buffer = abs(w1_end_level - p2_hl) * 0.05  # 5% buffer
                if is_impulse_up:
                    w4_t2 = max(w4_t2, w1_end_level + buffer)
                else:
                    w4_t2 = min(w4_t2, w1_end_level - buffer)
            
            center_w4 = draw_target_box("W4", w4_t1, w4_t2, "255, 165, 0", offset_primary, 
                                       f"{w4_min_ret*100:.1f}-{w4_max_ret*100:.1f}% W3 Ret", p3, primary=True, **common_args)
            
            if center_w4:
                hypo_p4_price = center_w4[1]
                
                # Trader thinking: W5 projection based on overall wave structure
                # W5 projection depends on whether W3 is extended
                
                # Calculate W1-W3 range for potential W5 projection
                w1_to_w3_range = p3_hl - p0_hl
                
                # Determine direction multiplier based on impulse direction
                direction = 1 if is_impulse_up else -1
                
                # Get ATR value for volatility-adjusted targets if available
                atr_value = None
                if 'ATR' in data.columns and not data['ATR'].isnull().all():
                    atr_value = data['ATR'].iloc[-1]
                    print(f"  Trader insight: Using current ATR ({atr_value:.2f}) to adjust W5 target zone")
                
                # Determine W5 projection based on wave structure
                if w3_w1_ratio > 1.618:  # W3 is extended
                    print("  Trader insight: W3 was extended, W5 likely equals W1 (equality principle)")
                    # When W3 is extended, W5 often equals W1 (equality principle)
                    
                    # Use Fibonacci extensions with ATR adjustment for more realistic targets
                    ext5 = calculate_fibonacci_extensions(p0_hl, p1_hl, hypo_p4_price, atr_value)
                    w5_t1s = ext5.get(0.618)  # 61.8% of W1
                    w5_t2s = ext5.get(1.0)    # 100% of W1
                    w5_label = "61.8-100% of W1 (Est.)"
                else:  # Normal W3
                    print("  Trader insight: Normal W3, W5 projected using W1-W3 range")
                    # When W3 is not extended, W5 often relates to the entire W1-W3 range
                    
                    # Calculate tiered take-profit targets for practical trading
                    # TP1, TP2, TP3 based on the W1-W3 range
                    ext5 = calculate_fibonacci_extensions(p0_hl, p3_hl, hypo_p4_price, atr_value)
                    w5_t1s = ext5.get(TP_LEVELS["TP1"] * 0.618)  # 61.8% of TP1
                    w5_t2s = ext5.get(TP_LEVELS["TP2"])  # TP2 (161.8%)
                    w5_label = "61.8-161.8% W1-W3 Ext (Est.)"
                    
                # Ensure targets are ordered correctly (lower value first for drawing)
                if w5_t1s is not None and w5_t2s is not None and w5_t1s > w5_t2s:
                    w5_t1s, w5_t2s = w5_t2s, w5_t1s
                
                # Consider market context for final W5 target
                if current_price is not None:
                    # If current price is near W3, adjust W5 target for potential momentum
                    price_to_w3_ratio = abs(current_price - p3_hl) / w3_length if w3_length > 0 else 1
                    if price_to_w3_ratio < 0.1:  # Price still near W3
                        print("  Trader insight: Current price near W3 high, momentum may carry W5 further")
                        # Add extended target for strong momentum (TP3)
                        ext5 = calculate_fibonacci_extensions(p0_hl, p3_hl, hypo_p4_price, atr_value)
                        w5_t2s = ext5.get(TP_LEVELS["TP3"])  # TP3 (261.8%)
                        # Ensure targets are ordered correctly after adjustment
                        if w5_t1s is not None and w5_t2s is not None and w5_t1s > w5_t2s:
                            w5_t1s, w5_t2s = w5_t2s, w5_t1s
                        w5_label = f"61.8-{TP_LEVELS['TP3']*100:.1f}% W1-W3 Ext (Est.)"
                        print(f"  Trader insight: Added TP3 target at {TP_LEVELS['TP3']*100:.1f}% extension due to strong momentum")
                
                w5_start_date = center_w4[0]
                center_w5 = draw_target_box("W5", w5_t1s, w5_t2s, "173, 255, 47", offset_secondary, 
                                           w5_label, p3, primary=False, 
                                           basis_info_override="(Est. from Proj. W4)", 
                                           custom_start_date=w5_start_date, **common_args)

        # --- If W4 finished -> Project W5 (Primary) ---
        elif current_label_num == 4 and p0 is not None and p1 is not None and p3 is not None and p4 is not None:
            print("  Projecting W5 (Primary)...")
            p0_hl = p0['Low'] if is_impulse_up else p0['High']
            p1_hl = p1['High'] if is_impulse_up else p1['Low']
            p2_hl = p2['Low'] if is_impulse_up else p2['High'] if p2 is not None else None
            p3_hl = p3['High'] if is_impulse_up else p3['Low']
            p4_close = p4['Close']
            
            # Analyze wave characteristics for trader insights
            w1_length = abs(p1_hl - p0_hl)
            w3_length = abs(p3_hl - (p2_hl if p2_hl is not None else p1_hl))
            w3_w1_ratio = w3_length / w1_length if w1_length > 0 else 0
            
            # Calculate the full impulse range so far
            impulse_range_so_far = p3_hl - p0_hl
            
            print(f"  Trader analysis: W3/W1 ratio = {w3_w1_ratio:.2f}, W4 complete, projecting W5")
            
            # Trader thinking: W5 projection based on overall wave structure and market context
            # Key considerations:
            # 1. Is W3 extended? If yes, W5 often equals W1 (equality principle)
            # 2. Has the impulse already traveled far? If yes, W5 might be truncated
            # 3. Market momentum and volume characteristics
            
            # Get ATR value for volatility-adjusted targets if available
            atr_value = None
            if 'ATR' in data.columns and not data['ATR'].isnull().all():
                atr_value = data['ATR'].iloc[-1]
                print(f"  Trader insight: Using current ATR ({atr_value:.2f}) to adjust W5 target zone")
            
            # Determine direction multiplier based on impulse direction
            direction = 1 if is_impulse_up else -1
            
            # Determine W5 projection based on wave structure
            if w3_w1_ratio > 1.618:  # W3 is extended
                print("  Trader insight: W3 was extended, W5 likely equals W1 (equality principle)")
                # When W3 is extended, W5 often equals W1 (equality principle)
                
                # Use Fibonacci extensions with ATR adjustment for more realistic targets
                ext5 = calculate_fibonacci_extensions(p0_hl, p1_hl, p4_close, atr_value)
                w5_t1 = ext5.get(0.618)  # 61.8% of W1
                w5_t2 = ext5.get(1.0)    # 100% of W1
                w5_label = "61.8-100% of W1"
                
                # Check for potential truncation risk
                if abs(p4_close - p3_hl) > 0.5 * w3_length:  # Deep W4 correction
                    print("  Trader caution: Deep W4 correction may lead to truncated W5")
                    # Add more conservative target
                    w5_t1 = ext5.get(0.5)  # 50% of W1
                    w5_label = "50-100% of W1"
            else:  # W3 is not extended
                print("  Trader insight: W3 not strongly extended, projecting W5 using Fibonacci relationships")
                
                # Calculate tiered take-profit targets for practical trading
                # TP1, TP2, TP3 based on the impulse range so far
                ext5 = calculate_fibonacci_extensions(p0_hl, p3_hl, p4_close, atr_value)
                w5_t1 = ext5.get(TP_LEVELS["TP1"] * 0.618)  # 61.8% of TP1
                w5_t2 = ext5.get(TP_LEVELS["TP1"])          # TP1 (100%)
                w5_label = "61.8-100% W1-W3 Ext"
                
                # Check for strong momentum
                if has_volume and 'RSI' in data.columns and not data['RSI'].isnull().all():
                    recent_rsi = data['RSI'].iloc[-5:].mean() if len(data) >= 5 else data['RSI'].iloc[-1]
                    if (is_impulse_up and recent_rsi > 60) or (not is_impulse_up and recent_rsi < 40):
                        print("  Trader insight: Strong momentum detected, extending W5 target to TP2")
                        # Use TP2 for stronger momentum
                        w5_t2 = ext5.get(TP_LEVELS["TP2"])  # TP2 (161.8%)
                        w5_label = f"61.8-{TP_LEVELS['TP2']*100:.1f}% W1-W3 Ext"
                        
                        # If momentum is extremely strong, add TP3 as a potential target
                        if (is_impulse_up and recent_rsi > 70) or (not is_impulse_up and recent_rsi < 30):
                            print(f"  Trader insight: Extremely strong momentum, adding TP3 target at {TP_LEVELS['TP3']*100:.1f}%")
                            w5_t2 = ext5.get(TP_LEVELS["TP3"])  # TP3 (261.8%)
                            w5_label = f"61.8-{TP_LEVELS['TP3']*100:.1f}% W1-W3 Ext"
            
            # Ensure targets are ordered correctly (lower value first for drawing)
            if w5_t1 is not None and w5_t2 is not None and w5_t1 > w5_t2:
                w5_t1, w5_t2 = w5_t2, w5_t1
                
            # Final check for psychological levels and round numbers
            # Round targets to psychologically significant levels if they're not None
            if w5_t1 is not None and w5_t2 is not None:
                try:
                    price_magnitude = 10 ** (int(math.log10(abs(w5_t1))) - 1) if w5_t1 != 0 else 1
                    w5_t1 = round(w5_t1 / price_magnitude) * price_magnitude
                    w5_t2 = round(w5_t2 / price_magnitude) * price_magnitude
                except Exception as e:
                    print(f"  Warning: Could not round price targets: {e}")
            
            center_w5 = draw_target_box("W5", w5_t1, w5_t2, "173, 255, 47", offset_primary, 
                                       w5_label, p4, primary=True, **common_args)
            
            # Determine W5 projection based on wave structure
            if w3_w1_ratio > 1.618:  # W3 is extended
                print("  Trader insight: W3 was extended, W5 likely equals W1 (equality principle)")
                # When W3 is extended, W5 often equals W1 (equality principle)
                w5_t1 = p4_close + (w1_length * 0.618)
                w5_t2 = p4_close + (w1_length * 1.0)
                w5_label = "61.8-100% of W1"
                
                # Check for potential truncation risk
                if abs(p4_close - p3_hl) > 0.5 * w3_length:  # Deep W4 correction
                    print("  Trader caution: Deep W4 correction may lead to truncated W5")
                    # Add more conservative target
                    w5_t1 = p4_close + (w1_length * 0.5)
                    w5_label = "50-100% of W1"
            else:  # W3 is not extended
                print("  Trader insight: W3 not strongly extended, projecting W5 using Fibonacci relationships")
                # Standard Fibonacci projections
                w5_t1 = p4_close + (impulse_range_so_far * 0.618)
                w5_t2 = p4_close + (impulse_range_so_far * 1.0)
                w5_label = "61.8-100% W1-W3 Ext"
                
                # Check for strong momentum
                if has_volume and 'RSI' in data.columns and not data['RSI'].isnull().all():
                    recent_rsi = data['RSI'].iloc[-5:].mean() if len(data) >= 5 else data['RSI'].iloc[-1]
                    if (is_impulse_up and recent_rsi > 60) or (not is_impulse_up and recent_rsi < 40):
                        print("  Trader insight: Strong momentum detected, extending W5 target")
                        w5_t2 = p4_close + (impulse_range_so_far * 1.618)
                        w5_label = "61.8-161.8% W1-W3 Ext"
            
            # Final check for psychological levels and round numbers
            # Round targets to psychologically significant levels
            price_magnitude = 10 ** (int(math.log10(abs(w5_t1))) - 1) if w5_t1 != 0 else 1
            w5_t1 = round(w5_t1 / price_magnitude) * price_magnitude
            w5_t2 = round(w5_t2 / price_magnitude) * price_magnitude
            
            center_w5 = draw_target_box("W5", w5_t1, w5_t2, "173, 255, 47", offset_primary, 
                                       w5_label, p4, primary=True, **common_args)

        # --- <<< START OF MODIFIED/NEW SECTION FOR ABC FORECAST >>> ---
        elif current_label_num == 5 and p0 is not None and p5 is not None:
            print("  Forecasting Corrective Waves: Proj. Wave A -> Est. Wave B -> Est. Wave C...")

            # --- Project Wave A (Primary) ---
            p0_hl = p0['Low'] if is_impulse_up else p0['High'] # Start of impulse
            p5_hl = p5['High'] if is_impulse_up else p5['Low'] # End of impulse (basis point)
            # For a zigzag correction after an impulse, Wave A is typically a 50% to 78.6% retracement
            # Calculate the entire impulse range
            impulse_range = p5_hl - p0_hl
            retA = calculate_fibonacci_retracements(p0_hl, p5_hl)
            # Correcciones A-B-C: Retrocesos comunes de 50%-61.8%-78.6% respecto al movimiento anterior
            wa_t1, wa_t2 = retA.get(0.500), retA.get(0.786)
            
            # Round targets to psychologically significant levels
            wa_t1 = round(wa_t1 * 20) / 20  # Round to nearest 0.05
            wa_t2 = round(wa_t2 * 20) / 20  # Round to nearest 0.05
            
            # Create a more descriptive label for Wave A
            wave_a_description = f"50-78.6% Impulse Ret ({abs(impulse_range):.2f} pts)"
            center_wa = draw_target_box(
                "Wave A", wa_t1, wa_t2, "255, 105, 180", # Pink
                offset_primary, wave_a_description,
                p5, # Basis point is P5
                primary=True, **common_args
            )

            if center_wa:
                hypo_pA_price = center_wa[1] # Use center price of projected A box
                wa_start_date = center_wa[0] # Use center date for timing next box

                # --- Estimate Wave B (Secondary) ---
                # Wave B typically retraces Wave A (the move from P5 to hypo_pA_price)
                retB = calculate_fibonacci_retracements(p5_hl, hypo_pA_price) # Retrace the hypothetical Wave A move
                # Correcciones A-B-C: Retrocesos comunes de 50%-61.8%-78.6% respecto al movimiento anterior
                wb_t1s, wb_t2s = retB.get(0.500), retB.get(0.618)
                
                # Round targets to psychologically significant levels
                wb_t1s = round(wb_t1s * 20) / 20  # Round to nearest 0.05
                wb_t2s = round(wb_t2s * 20) / 20  # Round to nearest 0.05
                
                wave_b_description = f"50-61.8% Retracement of Wave A"
                center_wb = draw_target_box(
                    "Wave B", wb_t1s, wb_t2s, "135, 206, 250", # Sky Blue
                    offset_secondary, wave_b_description,
                    p5, # Keep P5 as conceptual basis point for the correction start
                    primary=False, basis_info_override="(From Wave A Est.)",
                    custom_start_date=wa_start_date, # Start B box after A box midpoint
                    **common_args
                )

                if center_wb:
                    hypo_pB_price = center_wb[1] # Use center price of estimated B box
                    wb_start_date = center_wb[0] # Use center date for timing next box

                    # --- Estimate Wave C (Tertiary) ---
                    # Wave C often relates to Wave A length, projected from end of B
                    # Calculate hypothetical Wave A length/move (price difference)
                    hypo_wa_move = hypo_pA_price - p5_hl # This will be negative if impulse was up

                    if not pd.isna(hypo_wa_move):
                        # Project WC downwards (if impulse up) or upwards (if impulse down) from hypo_pB_price
                        # Correcciones A-B-C: Retrocesos comunes de 50%-61.8%-78.6% respecto al movimiento anterior
                        # Target 1: C = 100% of A (equal to A)
                        wc_t1s = hypo_pB_price + (hypo_wa_move * 1.0)
                        
                        # For Wave C, consider market context - if we're in a strong trend, C might extend further
                        # Target 2: C = 161.8% of A (extension of A)
                        wc_t2s = hypo_pB_price + (hypo_wa_move * 1.618)
                        
                        # Round targets to psychologically significant levels (important for trader psychology)
                        wc_t1s = round(wc_t1s * 20) / 20  # Round to nearest 0.05
                        wc_t2s = round(wc_t2s * 20) / 20  # Round to nearest 0.05
                        wave_c_description = f"100-161.8% of Wave A ({abs(hypo_wa_move):.2f} pts)"
                        center_wc = draw_target_box(
                            "Wave C", wc_t1s, wc_t2s, "255, 0, 0", # Red
                            offset_tertiary, wave_c_description,
                            p5, # Keep P5 as conceptual basis
                            primary=False, basis_info_override="(From Wave B Est.)",
                            custom_start_date=wb_start_date, # Start C box after B box midpoint
                            **common_args
                        )

        # --- <<< NEW: If Wave (C?) finished -> Project New Wave 1 (Primary) >>> ---
        elif projection_basis_label == '(C?)' and analysis_results.get("details", {}).get("correction_guess"):
            correction_details = analysis_results["details"]["correction_guess"]
            pc = correction_details.get('C') # End of correction (our new Wave 0)
            # We need the original impulse points (0 and 1) to estimate New W1 size
            p0 = points.get(0)
            p1 = points.get(1)

            if pc is not None and p0 is not None and p1 is not None:
                print("  Projecting New Wave 1 after Corrective (C?) completion...")
                is_original_impulse_up = details.get("is_upward", True) # Direction of original 0-5 impulse

                # Get relevant points from original impulse Wave 1
                p0_hl = p0['Low'] if is_original_impulse_up else p0['High']
                p1_hl = p1['High'] if is_original_impulse_up else p1['Low']
                pc_close = pc['Close'] # End of correction is start of new impulse

                prev_w1_len = abs(p1_hl - p0_hl) # Length of the previous impulse's Wave 1

                if prev_w1_len > 1e-9:
                    # Get ATR value for volatility-adjusted targets if available
                    atr_value = None
                    if 'ATR' in data.columns and not data['ATR'].isnull().all():
                        atr_value = data['ATR'].iloc[-1]
                        print(f"  Trader insight: Using current ATR ({atr_value:.2f}) to adjust New W1 target zone")
                    
                    # New impulse direction is typically opposite to the correction
                    # If original impulse was up, correction was down, new impulse is up
                    direction = 1 if is_original_impulse_up else -1
                    
                    # Calculate Fibonacci extensions with ATR adjustment
                    ext_nw1 = calculate_fibonacci_extensions(p0_hl, p1_hl, pc_close, atr_value)
                    
                    # Project New W1 based on common ratios to Previous W1
                    nw1_t1 = ext_nw1.get(0.618)  # 61.8% of previous W1
                    nw1_t2 = ext_nw1.get(1.0)    # 100% of previous W1
                    nw1_label = "61.8-100% of Prev W1"

                    # Ensure targets are ordered correctly
                    if nw1_t1 is not None and nw1_t2 is not None and nw1_t1 > nw1_t2:
                        nw1_t1, nw1_t2 = nw1_t2, nw1_t1

                    # Check market momentum for potential stronger move
                    if 'RSI' in data.columns and not data['RSI'].isnull().all():
                        recent_rsi = data['RSI'].iloc[-5:].mean() if len(data) >= 5 else data['RSI'].iloc[-1]
                        # If momentum is strong in the direction of the new impulse
                        if (is_original_impulse_up and recent_rsi > 60) or (not is_original_impulse_up and recent_rsi < 40):
                            print("  Trader insight: Strong momentum detected, extending New W1 target")
                            nw1_t2 = ext_nw1.get(1.618)  # 161.8% of previous W1
                            nw1_label = "61.8-161.8% of Prev W1"

                    center_nw1 = draw_target_box(
                        "New W1", nw1_t1, nw1_t2, "0, 255, 255", # Cyan for New Wave 1
                        offset_primary, nw1_label,
                        pc, # Basis is the end of Wave C
                        primary=True, **common_args
                    )
                    if center_nw1:
                        projection_path_points.append(center_nw1)
                        projection_plotted = True # PREVENT FALLBACK
                else:
                    print("  Could not calculate New W1 length or targets.")
            else:
                print("  Skipping New W1 projection: Missing necessary points (C, P0, or P1).")

        # Collect valid projected center points for path drawing
        temp_path_points = [projection_path_points[0]] # Start with the basis point
        for center in [center_w2, center_w3, center_w4, center_w5, center_wa, center_wb, center_wc]:
            if center is not None and isinstance(center, tuple) and len(center) == 3:
                temp_path_points.append(center)
                projection_plotted = True
        projection_path_points = sorted(temp_path_points, key=lambda x: x[0])


    '''# --- Fallback Projection (remains the same) ---
    if not projection_plotted and last_overall_point is not None and identified_waves is not None and len(identified_waves) >= 2:
         print("  No impulse-based projections made. Attempting Fallback Projection...")
         # (Fallback logic code as before...)
         p_proj_base = last_overall_point
         try:
             last_idx_loc = identified_waves.index.get_loc(p_proj_base.name)
             if last_idx_loc > 0:
                 p_prev = identified_waves.iloc[last_idx_loc - 1]
                 if p_prev is not None and pd.notna(p_prev['Close']) and pd.notna(p_proj_base['Close']):
                     is_move_up = p_proj_base['Close'] > p_prev['Close']
                     p_prev_hl = p_prev['Low'] if is_move_up else p_prev['High']; p_proj_base_hl = p_proj_base['High'] if is_move_up else p_proj_base['Low']
                     ext = calculate_fibonacci_extensions(p_prev_hl, p_proj_base_hl, p_proj_base['Close'])
                     t1, t2 = ext.get(1.0), ext.get(1.618)
                     if 'avg_time_delta' not in locals(): avg_time_delta = pd.Timedelta(days=1) if 'd' in interval else pd.Timedelta(weeks=1)
                     if 'offset_primary' not in locals(): offset_primary = pd.Timedelta(weeks=TARGETBOX_CANDLES_FALLBACK) if 'wk' in interval else pd.Timedelta(days=TARGETBOX_CANDLES_FALLBACK)
                     if 'box_start_date' not in locals(): box_start_date = p_proj_base.name + avg_time_delta / 4 if pd.notna(p_proj_base.name) else data.index[-1] + pd.Timedelta(days=1)
                     if not projection_path_points: projection_path_points.append((p_proj_base.name, p_proj_base['Close'], f"Start ({p_proj_base['EW_Label']})"))
                     fallback_center = draw_target_box("Target", t1, t2, "255, 215, 0", offset_primary, "100-161.8% Prev Leg Ext", p_proj_base, current_price, primary=True, full_data=data, major_fibs_data=major_fib_levels_data)
                     if fallback_center: projection_path_points.append(fallback_center); projection_plotted = True
                 else: print("  Fallback projection skipped: Invalid points.")
             else: print("  Fallback projection skipped: Cannot find previous point.")
         except Exception as e: print(f"  Error during fallback projection setup: {e}")
    '''
    # --- Draw Predictive Path Lines (remains the same) ---
    if len(projection_path_points) > 1:
        print(f"  Drawing {len(projection_path_points)-1} predictive path segment(s)...")
        # (Path drawing logic as before...)
        projection_path_points.sort(key=lambda x: x[0])
        for i in range(len(projection_path_points) - 1):
             start_date, start_price, start_label = projection_path_points[i]; end_date, end_price, end_label = projection_path_points[i+1]
             if pd.isna(start_date) or pd.isna(start_price) or pd.isna(end_date) or pd.isna(end_price): continue
             if not isinstance(start_date, pd.Timestamp): start_date = pd.to_datetime(start_date)
             if not isinstance(end_date, pd.Timestamp): end_date = pd.to_datetime(end_date)
             if pd.isna(start_date) or pd.isna(end_date): continue
             line_color = "rgba(255, 255, 255, 0.7)"; moving_down_expected = False; wave_char = None
             if isinstance(end_label, str):
                 label_part = end_label.split()[-1]
                 if label_part.isdigit(): wave_char = label_part
                 elif label_part in ['A', 'B', 'C']: wave_char = label_part
                 elif label_part == "Target": wave_char = "Target"
             if wave_char:
                 if wave_char in ['2', '4', 'A', 'C']: moving_down_expected = is_impulse_up
                 elif wave_char in ['3', '5', 'B']: moving_down_expected = not is_impulse_up
                 elif wave_char == "Target":
                      if len(projection_path_points) > 1:
                          prev_date, prev_price, _ = projection_path_points[i]
                          if start_price < prev_price : moving_down_expected = True
                          else: moving_down_expected = False
             if wave_char != "Target":
                 if moving_down_expected: line_color = "rgba(255, 0, 0, 0.8)"
                 else: line_color = "rgba(0, 255, 0, 0.8)"
             elif wave_char == "Target": line_color = "rgba(255, 215, 0, 0.8)"
             fig.add_shape(type="line", x0=start_date, y0=start_price, x1=end_date, y1=end_price, line=dict(color=line_color, width=2.5, dash='solid'), layer='above', row=1, col=1)
             mid_date = start_date + (end_date - start_date) / 2; mid_price = (start_price + end_price) / 2
             fig.add_annotation(x=mid_date, y=mid_price, text=f"Pred. {end_label}", showarrow=False, font=dict(size=10, color=line_color.replace('0.8', '1.0').replace('0.7', '1.0')), bgcolor="rgba(40,40,40,0.6)", borderpad=2, row=1, col=1)

    # --- Plot Volume / RSI (remains the same) ---
    if has_volume:
        vol_colors = ['rgba(0, 200, 0, 0.6)' if c >= o else 'rgba(255, 50, 50, 0.6)' for o, c in zip(data['Open'], data['Close'])]
        fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color=vol_colors, showlegend=False), row=volume_row_index, col=1)
        fig.update_yaxes(title_text="Volume", row=volume_row_index, col=1, side='left')
    if has_rsi:
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', line=dict(color='rgb(255, 102, 204)', width=1.5), showlegend=False), row=rsi_row_index, col=1)
        fig.add_hline(y=RSI_OVERBOUGHT, line_dash="dot", line_color="rgba(255, 150, 150, 0.5)", line_width=1, row=rsi_row_index, col=1)
        fig.add_hline(y=RSI_OVERSOLD, line_dash="dot", line_color="rgba(150, 255, 150, 0.5)", line_width=1, row=rsi_row_index, col=1)
        fig.add_annotation(x=data.index[-1], y=RSI_OVERBOUGHT, text="Overbought", showarrow=False, xanchor='right', yanchor='bottom', font=dict(size=10), row=rsi_row_index, col=1)
        fig.add_annotation(x=data.index[-1], y=RSI_OVERSOLD, text="Oversold", showarrow=False, xanchor='right', yanchor='top', font=dict(size=10), row=rsi_row_index, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=rsi_row_index, col=1, side='left')

    # --- Layout and Final Touches (remains the same) ---
    chart_title_main = f'Conceptual EW Analysis: {ticker} ({interval})'
    title_suffix = " - Labeling Failed/Insufficient Data"
    if analysis_results:
        final_basis_label = analysis_results.get("last_label")
        if analysis_results.get("found_impulse", False) and final_basis_label:
            details = analysis_results.get("details", {})
            title_suffix = f" - Best Sequence: 0-{final_basis_label} Identified"
            if details.get("correction_guess"): title_suffix += " + Corrective Guess"
        elif plot_waves: title_suffix = " - No Valid Impulse Found (Using P-labels)"
    fig.update_layout(title=dict(text=chart_title_main + title_suffix + "<br><sup><b>*** DEMO / EDUCATIONAL USE ONLY - ANALYSIS & PROJECTIONS ARE HIGHLY SPECULATIVE - NOT FOR TRADING ***</b></sup>", font=dict(size=14), x=0.5, xanchor='center'),
                      xaxis_rangeslider_visible=False, template="plotly_dark", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(0,0,0,0.2)'),
                      margin=dict(l=50, r=50, t=120, b=50), hovermode='x unified', height=950 if (has_rsi and has_volume) else (850 if (has_rsi or has_volume) else 750))
    currency = data['Currency'].iloc[0] if 'Currency' in data.columns and not data['Currency'].isnull().all() else 'Price'
    fig.update_yaxes(title_text=f"Price ({currency})", row=1, col=1)
    for r in range(1, rows + 1): fig.update_xaxes(rangeslider_visible=False, row=r, col=1)

    print("  Plot generation complete.")
    return fig


# --- Main Analysis Runner Function ---
# (Function remains the same)
def run_analysis(ticker, start_date, end_date, interval):
    """Runs the full Elliott Wave analysis pipeline for a given period."""
    # ... (function code as before) ...
    start_time = datetime.datetime.now()
    print(f"\n{'='*15} Starting Standard Analysis {'='*15}")
    print(f" Ticker: {ticker} ({interval})")
    print(f" Period: {start_date} to {end_date}")
    print(f"{'='*48}")
    if interval == '1wk': peak_order = PEAK_TROUGH_ORDER_WEEKLY
    elif interval == '1d': peak_order = PEAK_TROUGH_ORDER_DAILY
    else: peak_order = PEAK_TROUGH_ORDER_DEFAULT
    print(f" Using Peak Finding Order: {peak_order}")
    stock_data = get_stock_data(ticker, start_date, end_date, interval)
    if stock_data is None or stock_data.empty:
        error_summary = {"error": f"Could not load market data for {ticker} ({start_date} to {end_date}, {interval}). Check ticker and date range."}
        return None, error_summary
    potential_points = find_potential_wave_points(stock_data, order=peak_order, prom_pct=PROMINENCE_PCT_FACTOR, prom_atr=PROMINENCE_ATR_FACTOR)
    if potential_points is None: potential_points = pd.DataFrame()
    identified_waves, analysis_summary = find_elliott_waves(potential_points, stock_data)
    fig = None
    try:
        fig = plot_chart(stock_data, identified_waves, analysis_summary, ticker=ticker, interval=interval)
        if fig is None and analysis_summary and 'error' not in analysis_summary:
            analysis_summary['error'] = "Plotting function returned None."
    except Exception as plot_err:
        print(f"\n!!! Error during plot generation: {plot_err}"); traceback.print_exc()
        if analysis_summary is None: analysis_summary = {}
        analysis_summary['error'] = f"Plotting failed: {plot_err}"
    print("\n" + "-" * 60); print("Analysis function finished."); print(f"(Execution time: {datetime.datetime.now() - start_time})"); print("-" * 60)
    return fig, analysis_summary

# --- Backtest Simulation Runner ---
# (Function remains the same)
def run_backtest_simulation(ticker, start_date, analysis_date, check_date, interval):
    """Runs the EW analysis as of 'analysis_date' and overlays actual data up to 'check_date'."""
    # ... (function code as before) ...
    start_time = datetime.datetime.now(); print(f"\n{'='*15} Starting Backtest Simulation {'='*15}")
    print(f" Ticker: {ticker} ({interval})"); print(f" Analysis Period: {start_date} to {analysis_date}"); print(f" Check Period: {analysis_date} to {check_date}"); print(f"{'='*52}")
    fig_past, analysis_summary = run_analysis(ticker, start_date, analysis_date, interval)
    if fig_past is None: print("[Backtest Halted] Initial analysis failed."); return None, analysis_summary or {"error": "Initial analysis for backtest failed."}
    try:
        analysis_dt_obj = pd.to_datetime(analysis_date)
        fig_past.add_vline(x=analysis_dt_obj, line_width=2, line_dash="dash", line_color="yellow", annotation_text=f"Analysis Date ({analysis_date})", annotation_position="top left", annotation_font_size=10, annotation_font_color="yellow", row=1, col=1)
    except Exception as e: print(f"Warning: Could not add vertical line for analysis date: {e}")
    try:
        future_start_dt = pd.to_datetime(analysis_date) + pd.Timedelta(days=1); future_start_str = future_start_dt.strftime('%Y-%m-%d')
        print(f"Fetching future data from {future_start_str} to {check_date}")
        future_data = get_stock_data(ticker, future_start_str, check_date, interval)
    except Exception as e: print(f"Error fetching future data: {e}"); future_data = None
    
    # Get the trade recommendation based on analysis_date data
    analysis_data = get_stock_data(ticker, start_date, analysis_date, interval)
    trade_recommendation_data = None
    if analysis_data is not None and not analysis_data.empty and analysis_summary and not analysis_summary.get('error'):
        trade_recommendation_data = generate_trade_recommendation(
            analysis_summary,
            analysis_data,
            risk_percent=1.0,  # Default risk value
            demo_account_size=10000  # Default account size
        )
        
        # Store the trade recommendation in the analysis summary
        if analysis_summary is None:
            analysis_summary = {}
        analysis_summary['trade_recommendation'] = trade_recommendation_data
    
    # Calculate backtest statistics if we have both future data and trade recommendation
    backtest_stats = None
    if future_data is not None and not future_data.empty and trade_recommendation_data and trade_recommendation_data['status'] == 'Trade Found':
        backtest_stats = calculate_backtest_stats(trade_recommendation_data, future_data)
        
        # Add backtest stats to the analysis summary
        if analysis_summary is None:
            analysis_summary = {}
        analysis_summary['backtest_stats'] = backtest_stats
    
    if future_data is not None and not future_data.empty:
        print(f"Overlaying {len(future_data)} future data points onto the plot.")
        try:
            fig_past.add_trace(go.Candlestick(x=future_data.index, open=future_data['Open'], high=future_data['High'], low=future_data['Low'], close=future_data['Close'], name=f'{ticker} Future Actual', increasing=dict(line=dict(color='rgba(152, 251, 152, 0.9)')), decreasing=dict(line=dict(color='rgba(240, 128, 128, 0.9)')), showlegend=True), row=1, col=1)
        except Exception as overlay_err: print(f"!!! Error overlaying future data: {overlay_err}"); traceback.print_exc(); analysis_summary['warning'] = analysis_summary.get('warning', "") + f"; Failed to overlay future data: {overlay_err}"
    else: print("No future data found or fetched to overlay."); analysis_summary['warning'] = analysis_summary.get('warning', "") + "; No future data found/fetched for backtest overlay."
    
    original_title = fig_past.layout.title.text.split('<br><sup>')[0]
    backtest_title = f"EW Backtest: Analysis as of {analysis_date}, Checked up to {check_date}<br>" + original_title
    fig_past.update_layout(title=dict(text=backtest_title + "<br><sup><b>*** DEMO / EDUCATIONAL USE ONLY - ANALYSIS & PROJECTIONS ARE HIGHLY SPECULATIVE - NOT FOR TRADING ***</b></sup>", font=dict(size=14), x=0.5, xanchor='center'))
    print("\n" + "-" * 60); print("Backtest simulation finished."); print(f"(Total execution time: {datetime.datetime.now() - start_time})"); print("-" * 60)
    return fig_past, analysis_summary

def calculate_backtest_stats(trade_recommendation, future_data):
    """
    Calculate backtest statistics based on trade recommendation and future price data.
    
    Args:
        trade_recommendation (dict): Trade recommendation data with entry, SL, TP levels
        future_data (DataFrame): Future price data to check against
        
    Returns:
        dict: Statistics about the backtest including TP/SL hit status and timings
    """
    print("Calculating backtest statistics...")
    
    stats = {
        'status': 'still_running',  # Default status
        'hit_tp1': False,
        'hit_tp2': False,
        'hit_sl': False,
        'hit_tp1_date': None,
        'hit_tp2_date': None,
        'hit_sl_date': None,
        'max_profit_pct': 0,
        'max_loss_pct': 0,
        'current_pct_change': 0,
        'days_in_trade': 0,
        'final_price': None
    }
    
    # Extract price levels from trade recommendation
    entry_price = trade_recommendation.get('entry_price')
    sl_price = trade_recommendation.get('stop_loss_price')
    tp1_price = trade_recommendation.get('tp1_price')
    tp2_price = trade_recommendation.get('tp2_price')
    signal = trade_recommendation.get('signal')
    
    if not entry_price or not sl_price or not signal:
        print("  Missing key trade data, can't calculate backtest stats")
        return stats
    
    if future_data.empty:
        print("  No future data available for backtest")
        return stats
    
    # Record final price from the future data
    stats['final_price'] = future_data['Close'].iloc[-1]
    
    # Calculate current percentage change from entry
    if entry_price > 0:
        current_pct_change = ((stats['final_price'] - entry_price) / entry_price) * 100
        stats['current_pct_change'] = round(current_pct_change, 2)
    
    # Calculate days in trade
    if not future_data.index.empty:
        stats['days_in_trade'] = (future_data.index[-1] - future_data.index[0]).days
    
    # Process each candle in the future data to see when levels were hit
    hit_sl = False
    hit_tp1 = False
    hit_tp2 = False
    
    max_profit_pct = 0
    max_loss_pct = 0
    
    for idx, row in future_data.iterrows():
        # For Long positions:
        if signal == "Long":
            # Calculate profit/loss based on high and low prices
            current_profit_pct = ((row['High'] - entry_price) / entry_price) * 100
            current_loss_pct = ((row['Low'] - entry_price) / entry_price) * 100
            
            # Update max profit/loss with reference to TP and SL levels
            if tp2_price is not None and row['High'] >= tp2_price:
                # If TP2 is hit, use TP2 for max profit
                tp2_pct = ((tp2_price - entry_price) / entry_price) * 100
                max_profit_pct = max(max_profit_pct, tp2_pct)
            elif tp1_price is not None and row['High'] >= tp1_price:
                # If TP1 is hit, use TP1 for max profit
                tp1_pct = ((tp1_price - entry_price) / entry_price) * 100
                max_profit_pct = max(max_profit_pct, tp1_pct)
            else:
                # Otherwise use the current price for max profit
                max_profit_pct = max(max_profit_pct, current_profit_pct)
            
            # For max loss, use either SL or current price, whichever is lower
            if row['Low'] <= sl_price:
                sl_pct = ((sl_price - entry_price) / entry_price) * 100
                max_loss_pct = min(max_loss_pct, sl_pct)
            else:
                max_loss_pct = min(max_loss_pct, current_loss_pct)
            
            # Check if SL was hit (long: price went below SL)
            if not hit_sl and row['Low'] <= sl_price:
                hit_sl = True
                stats['hit_sl'] = True
                stats['hit_sl_date'] = idx
            
            # Check if TP1 was hit (long: price went above TP1)
            if not hit_tp1 and tp1_price is not None and row['High'] >= tp1_price:
                hit_tp1 = True
                stats['hit_tp1'] = True
                stats['hit_tp1_date'] = idx
            
            # Check if TP2 was hit (long: price went above TP2)
            if not hit_tp2 and tp2_price is not None and row['High'] >= tp2_price:
                hit_tp2 = True
                stats['hit_tp2'] = True
                stats['hit_tp2_date'] = idx
                
        else:  # Short position
            # Calculate profit/loss based on high and low prices (inverted for shorts)
            current_profit_pct = ((entry_price - row['Low']) / entry_price) * 100
            current_loss_pct = ((entry_price - row['High']) / entry_price) * 100
            
            # Update max profit/loss with reference to TP and SL levels
            if tp2_price is not None and row['Low'] <= tp2_price:
                # If TP2 is hit, use TP2 for max profit
                tp2_pct = ((entry_price - tp2_price) / entry_price) * 100
                max_profit_pct = max(max_profit_pct, tp2_pct)
            elif tp1_price is not None and row['Low'] <= tp1_price:
                # If TP1 is hit, use TP1 for max profit
                tp1_pct = ((entry_price - tp1_price) / entry_price) * 100
                max_profit_pct = max(max_profit_pct, tp1_pct)
            else:
                # Otherwise use the current price for max profit
                max_profit_pct = max(max_profit_pct, current_profit_pct)
            
            # For max loss, use either SL or current price, whichever is lower
            if row['High'] >= sl_price:
                sl_pct = ((entry_price - sl_price) / entry_price) * 100
                max_loss_pct = min(max_loss_pct, sl_pct)
            else:
                max_loss_pct = min(max_loss_pct, current_loss_pct)
            
            # Check if SL was hit (short: price went above SL)
            if not hit_sl and row['High'] >= sl_price:
                hit_sl = True
                stats['hit_sl'] = True
                stats['hit_sl_date'] = idx
            
            # Check if TP1 was hit (short: price went below TP1)
            if not hit_tp1 and tp1_price is not None and row['Low'] <= tp1_price:
                hit_tp1 = True
                stats['hit_tp1'] = True
                stats['hit_tp1_date'] = idx
            
            # Check if TP2 was hit (short: price went below TP2)
            if not hit_tp2 and tp2_price is not None and row['Low'] <= tp2_price:
                hit_tp2 = True
                stats['hit_tp2'] = True
                stats['hit_tp2_date'] = idx
    
    stats['max_profit_pct'] = round(max_profit_pct, 2)
    stats['max_loss_pct'] = round(max_loss_pct, 2)
    
    # Determine final status
    if hit_sl:
        stats['status'] = 'stopped_out'
    elif hit_tp2:
        stats['status'] = 'hit_tp2'
    elif hit_tp1:
        stats['status'] = 'hit_tp1'
    else:
        stats['status'] = 'still_running'
    
    print(f"  Backtest Stats: Status={stats['status']}, TP1 Hit={stats['hit_tp1']}, TP2 Hit={stats['hit_tp2']}, SL Hit={stats['hit_sl']}")
    print(f"  Max Profit: {stats['max_profit_pct']}%, Max Loss: {stats['max_loss_pct']}%")
    
    return stats

# --- <<< NEW FUNCTION: Trade Recommendation Generator >>> ---
def generate_trade_recommendation(analysis_summary, stock_data, risk_percent=1.0, demo_account_size=10000):
    """
    Generates a trade recommendation based on the completed EW analysis.

    Args:
        analysis_summary (dict): The output dictionary from find_elliott_waves.
        stock_data (pd.DataFrame): The stock data used for the analysis (needs ATR).
        risk_percent (float): The percentage of demo account size to risk per trade (e.g., 1.0 for 1%).
        demo_account_size (float): The size of the demo trading account.

    Returns:
        dict: A dictionary containing trade recommendation details, or None if no trade setup found.
              Keys: 'status', 'signal', 'reason', 'entry_price', 'stop_loss_price',
                    'sl_distance_pct', 'sl_distance_pips', 'tp1_price', 'tp1_rrr',
                    'tp2_price', 'tp2_rrr', 'confidence_score', 'position_size_units',
                    'risk_amount', 'notes'.
    """
    print("\n[Trade Recommender] Generating Trade Setup...")
    recommendation = {'status': 'No Trade'} # Default

    if not analysis_summary or not analysis_summary.get('found_impulse'):
        recommendation['reason'] = "No valid impulse wave sequence identified."
        print(f"  Result: {recommendation['reason']}")
        return recommendation

    details = analysis_summary.get('details')
    if not details:
        recommendation['reason'] = "Analysis details are missing."
        print(f"  Result: {recommendation['reason']}")
        return recommendation

    last_label = analysis_summary.get('last_label')
    last_point_overall = analysis_summary.get('last_point_overall') # Use the very last point on chart for entry context
    last_ew_point = details.get('last_point') # Use the last point OF THE IDENTIFIED SEQUENCE for calcs
    points = details.get('points', {})
    is_impulse_up = details.get('is_upward', True)
    base_score = details.get('score', 0)
    guidelines = details.get('guidelines', {})
    currency = stock_data['Currency'].iloc[0] if 'Currency' in stock_data.columns and not stock_data['Currency'].isnull().all() else ''

    if last_point_overall is None or last_ew_point is None:
        recommendation['reason'] = "Last point data is missing."
        print(f"  Result: {recommendation['reason']}")
        return recommendation

    entry_price = last_point_overall['Close'] # Enter based on the close of the last identified point overall
    current_atr = stock_data['ATR'].iloc[-1] if 'ATR' in stock_data.columns and not stock_data['ATR'].isnull().all() else None
    atr_stop_multiplier = 1.0 # Multiplier for ATR buffer on stop loss

    # --- Determine Trade Setup Based on Last EW Label ---
    trade_setup_found = False
    signal = None
    reason = ""
    stop_loss_price = None
    p0=points.get(0); p1=points.get(1); p2=points.get(2); p3=points.get(3); p4=points.get(4); p5=points.get(5)
    correction_guess = details.get("correction_guess")
    pa = correction_guess.get('A') if correction_guess else None
    pb = correction_guess.get('B') if correction_guess else None
    pc = correction_guess.get('C') if correction_guess else None

    # === Potential LONG Setups (if impulse is UP) or SHORT Setups (if impulse is DOWN) ===
    if last_label == '2' and p0 is not None and p1 is not None and p2 is not None:
        # Expecting Wave 3
        signal = "Long" if is_impulse_up else "Short"
        reason = f"Anticipating Wave 3 after Wave 2 completion."
        stop_loss_price = p0['Low'] if is_impulse_up else p0['High'] # Invalidate below/above Wave 0
        p0_hl = p0['Low'] if is_impulse_up else p0['High']
        p1_hl = p1['High'] if is_impulse_up else p1['Low']
        p2_close = p2['Close'] # Use actual W2 close for projection start
        ext3 = calculate_fibonacci_extensions(p0_hl, p1_hl, p2_close)
        tp1_price = ext3.get(1.0)   # More conservative W3 target (W3=W1)
        tp2_price = ext3.get(1.618) # Classic W3 target
        trade_setup_found = True

    elif last_label == '4' and p0 is not None and p1 is not None and p2 is not None and p3 is not None and p4 is not None:
        # Expecting Wave 5
        signal = "Long" if is_impulse_up else "Short"
        reason = f"Anticipating Wave 5 after Wave 4 completion."
        # SL below/above W4 low/high (tighter) OR below/above W1 high/low (rule) - Using W4 level
        stop_loss_price = p4['Low'] if is_impulse_up else p4['High']
        p0_hl = p0['Low'] if is_impulse_up else p0['High']
        p1_hl = p1['High'] if is_impulse_up else p1['Low']
        p4_close = p4['Close'] # Use actual W4 close for projection start
        # Project W5 based on W1 length
        w1_len = abs(p1_hl - p0_hl)
        tp1_price = p4_close + (w1_len * 0.618 * (1 if is_impulse_up else -1)) # W5 = 0.618 * W1
        tp2_price = p4_close + (w1_len * 1.000 * (1 if is_impulse_up else -1)) # W5 = W1
        # Alternate W5 target: Extension from 0-3 projected from 4
        # p3_hl = p3['High'] if is_impulse_up else p3['Low']
        # ext5_alt = calculate_fibonacci_extensions(p0_hl, p3_hl, p4_close)
        # tp2_price = ext5_alt.get(0.618) # Common alt target
        trade_setup_found = True

    elif last_label == '(B?)' and p5 is not None and pa is not None and pb is not None:
         # Expecting Wave C (down if impulse up, up if impulse down)
        signal = "Short" if is_impulse_up else "Long" # Correction moves opposite to impulse
        reason = f"Anticipating Corrective Wave C after Wave B completion."
        stop_loss_price = pb['High'] if is_impulse_up else pb['Low'] # SL above/below B
        p5_close = p5['Close']; pa_close = pa['Close']; pb_close = pb['Close']
        if not any(pd.isna(x) for x in [p5_close, pa_close, pb_close]):
            extC = calculate_fibonacci_extensions(p5_close, pa_close, pb_close) # Ext based on A length, from B end
            tp1_price = extC.get(0.618) # C = 0.618 * A
            tp2_price = extC.get(1.000) # C = A
            trade_setup_found = True
        else: reason += " (TP calc skipped due to NaN points)"

    elif last_label == '(C?)' and pc is not None:
        # Expecting NEW Impulse (Wave 1) after correction ends
        signal = "Long" if is_impulse_up else "Short" # New impulse follows original trend
        reason = f"Anticipating NEW impulse (Wave 1) after Corrective Wave C completion."
        stop_loss_price = pc['Low'] if is_impulse_up else pc['High'] # SL below/below C low/high
        # TP for new W1 is less defined by Fibs; use Risk-Reward Ratio
        # TP calculations will be done after SL distance is known
        trade_setup_found = True # Mark as found, TPs calculated later

    # --- Final Checks and Calculations ---
    if not trade_setup_found:
        recommendation['reason'] = f"No actionable trade setup identified at Wave '{last_label}'."
        print(f"  Result: {recommendation['reason']}")
        return recommendation

    if stop_loss_price is None or pd.isna(stop_loss_price):
        recommendation['reason'] = f"Could not determine Stop Loss for label '{last_label}'."
        print(f"  Result: {recommendation['reason']}")
        recommendation['status'] = 'Error'
        return recommendation

    # Add ATR buffer to stop loss for volatility
    if current_atr is not None and not pd.isna(current_atr) and atr_stop_multiplier > 0:
        buffer = current_atr * atr_stop_multiplier
        if signal == "Long": stop_loss_price -= buffer
        elif signal == "Short": stop_loss_price += buffer
        print(f"  Applied ATR buffer ({buffer:.4f}) to SL.")
        
    # Ensure SL is never below 0
    if stop_loss_price <= 0:
        stop_loss_price = 0.0001  # Set to a small positive value
        print("  Warning: Stop Loss was adjusted to prevent negative value.")

    # Ensure SL makes sense relative to entry
    if signal == "Long" and stop_loss_price >= entry_price:
        recommendation['reason'] = "Stop Loss is above or equal to Entry Price for Long trade."
        print(f"  Result: {recommendation['reason']} (SL: {stop_loss_price:.4f}, Entry: {entry_price:.4f})")
        recommendation['status'] = 'Error'
        return recommendation
    if signal == "Short" and stop_loss_price <= entry_price:
        recommendation['reason'] = "Stop Loss is below or equal to Entry Price for Short trade."
        print(f"  Result: {recommendation['reason']} (SL: {stop_loss_price:.4f}, Entry: {entry_price:.4f})")
        recommendation['status'] = 'Error'
        return recommendation

    sl_distance_pips = abs(entry_price - stop_loss_price)
    if entry_price == 0: # Avoid division by zero
        sl_distance_pct = float('inf')
    else:
        sl_distance_pct = (sl_distance_pips / entry_price) * 100

    # --- Calculate TPs for Wave C -> New Impulse case ---
    if last_label == '(C?)' and sl_distance_pips > 1e-9:
        rr_ratio_tp1 = 1.5
        rr_ratio_tp2 = 2.5
        if signal == "Long":
            tp1_price = entry_price + (sl_distance_pips * rr_ratio_tp1)
            tp2_price = entry_price + (sl_distance_pips * rr_ratio_tp2)
        else: # Short
            tp1_price = entry_price - (sl_distance_pips * rr_ratio_tp1)
            tp2_price = entry_price - (sl_distance_pips * rr_ratio_tp2)
            # Ensure TP values are never below 0 for Short positions
            if tp1_price <= 0:
                tp1_price = 0.0001
                print("  Warning: TP1 was adjusted to prevent negative value.")
            if tp2_price <= 0:
                tp2_price = 0.0001
                print("  Warning: TP2 was adjusted to prevent negative value.")
        reason += f" (TPs based on {rr_ratio_tp1}:1 and {rr_ratio_tp2}:1 R:R)"

    # --- Calculate RRR ---
    tp1_rrr = 0.0; tp2_rrr = 0.0
    if tp1_price is not None and not pd.isna(tp1_price) and sl_distance_pips > 1e-9:
        tp1_distance = abs(tp1_price - entry_price)
        tp1_rrr = tp1_distance / sl_distance_pips
        # Ensure TP1 is in the correct direction
        if (signal == "Long" and tp1_price < entry_price) or \
           (signal == "Short" and tp1_price > entry_price):
            tp1_price = None; tp1_rrr = 0.0
            print("  Warning: TP1 invalidated (wrong direction or SL too wide).")


    if tp2_price is not None and not pd.isna(tp2_price) and sl_distance_pips > 1e-9:
        tp2_distance = abs(tp2_price - entry_price)
        tp2_rrr = tp2_distance / sl_distance_pips
         # Ensure TP2 is in the correct direction and further than TP1 (if TP1 exists)
        if (signal == "Long" and tp2_price < entry_price) or \
           (signal == "Short" and tp2_price > entry_price) or \
           (tp1_price is not None and signal == "Long" and tp2_price <= tp1_price) or \
           (tp1_price is not None and signal == "Short" and tp2_price >= tp1_price):
            tp2_price = None; tp2_rrr = 0.0
            print("  Warning: TP2 invalidated (wrong direction or not beyond TP1).")

    # If only TP2 was valid initially, make it TP1
    if tp1_price is None and tp2_price is not None:
        tp1_price = tp2_price
        tp1_rrr = tp2_rrr
        tp2_price = None
        tp2_rrr = 0.0
        print("  Note: Original TP2 moved to TP1 as original TP1 was invalid.")

    # Ensure TP values are never below 0
    if tp1_price is not None and tp1_price <= 0:
        tp1_price = 0.0001
        print("  Warning: TP1 was adjusted to prevent negative value.")
    if tp2_price is not None and tp2_price <= 0:
        tp2_price = 0.0001
        print("  Warning: TP2 was adjusted to prevent negative value.")
        
    if tp1_price is None:
        recommendation['reason'] = f"Could not determine valid Take Profit levels for label '{last_label}'."
        print(f"  Result: {recommendation['reason']}")
        recommendation['status'] = 'Error'
        return recommendation


    # --- Calculate Confidence Score (Example Logic - Needs Refinement) ---
    confidence_score = 0
    # Base on EW score (max contribution: 50)
    confidence_score += max(0, min(50, base_score / 8)) # Rough scaling, assuming max score ~400

    # Boost for Guideline adherence (max contribution: 30)
    guideline_boost = 0
    guideline_count = 0
    # Relevant guidelines depending on the wave we are entering BEFORE
    relevant_guidelines = []
    if last_label == '2': relevant_guidelines = ["W3_Ext"] # Check if W3 guideline already hit? Less relevant here. Add W2 depth?
    if last_label == '4': relevant_guidelines = ["Alt_W2W4"] # Alternation is key for W4
    if last_label == '(B?)': relevant_guidelines = [] # Less standard guidelines apply directly
    if last_label == '(C?)': relevant_guidelines = [] # Starting fresh

    for g_name, g_passed in guidelines.items():
         if g_name in relevant_guidelines or g_name.startswith('Vol_'): # Include Vol guidelines
             guideline_count += 1
             if g_passed: guideline_boost += (10 if g_name.startswith('Vol_') else 15) # Give volume less weight

    if guideline_count > 0: confidence_score += max(0, min(30, (guideline_boost / (guideline_count * 15)) * 30)) # Normalize boost

    # Boost for Risk-Reward (max contribution: 15)
    if tp1_rrr >= 1.5: confidence_score += 5
    if tp2_rrr >= 2.5: confidence_score += 10

    # Penalty for wide SL (max penalty: -10)
    if sl_distance_pct > 5.0: confidence_score -= 5
    if sl_distance_pct > 10.0: confidence_score -= 5

    # Add small random factor for realism? No, keep it deterministic.
    # Check RSI? (Requires RSI value at entry point) - Future enhancement
    # Check MA confluence? (Requires MA values at entry point) - Future enhancement

    confidence_score = max(0, min(100, int(confidence_score))) # Clamp between 0 and 100

    # --- Risk Management Calculations ---
    risk_amount = demo_account_size * (risk_percent / 100.0)
    position_size_units = 0
    if sl_distance_pips > 1e-9: # Avoid division by zero
        position_size_units = risk_amount / sl_distance_pips
    else:
        recommendation['status'] = 'Error'
        recommendation['reason'] = "Stop Loss distance is zero, cannot calculate position size."
        print(f"  Result: {recommendation['reason']}")
        return recommendation

    # --- Populate Recommendation ---
    recommendation.update({
        'status': 'Trade Found',
        'signal': signal,
        'reason': reason,
        'entry_price': round(entry_price, 4),
        'stop_loss_price': round(stop_loss_price, 4),
        'sl_distance_pct': round(sl_distance_pct, 2),
        'sl_distance_pips': round(sl_distance_pips, 4),
        'tp1_price': round(tp1_price, 4) if tp1_price is not None else None,
        'tp1_rrr': round(tp1_rrr, 2) if tp1_rrr is not None else None,
        'tp2_price': round(tp2_price, 4) if tp2_price is not None else None,
        'tp2_rrr': round(tp2_rrr, 2) if tp2_rrr is not None else None,
        'confidence_score': confidence_score,
        'position_size_units': round(position_size_units, 4), # Adjust rounding based on asset type
        'risk_amount': round(risk_amount, 2),
        'account_size': demo_account_size,
        'risk_percent': risk_percent,
        'currency': currency,
        'notes': f"Recommendation based on EW analysis ending {last_ew_point.name.strftime('%Y-%m-%d')} (Label {last_label}). Entry assumes execution at close of {last_point_overall.name.strftime('%Y-%m-%d')}."
    })

    # Get the latest price to check if levels have been reached
    latest_price = stock_data['Close'].iloc[-1]
    
    # Calculate percentage change since entry
    pct_change = ((latest_price - entry_price) / entry_price) * 100

    # Initialize level reached flags
    tp1_reached = False
    tp2_reached = False
    sl_reached = False

    # Check if levels have been reached
    if signal == "Long":
        if tp1_price is not None and latest_price >= tp1_price:
            tp1_reached = True
        if tp2_price is not None and latest_price >= tp2_price:
            tp2_reached = True
        if latest_price <= stop_loss_price:
            sl_reached = True
    else:  # Short
        if tp1_price is not None and latest_price <= tp1_price:
            tp1_reached = True
        if tp2_price is not None and latest_price <= tp2_price:
            tp2_reached = True
        if latest_price >= stop_loss_price:
            sl_reached = True

    # Add level reached status and percentage change to recommendation
    recommendation.update({
        'tp1_reached': tp1_reached,
        'tp2_reached': tp2_reached,
        'sl_reached': sl_reached,
        'pct_change': round(pct_change, 2),
        'latest_price': round(latest_price, 4)
    })

    print(f"  Result: {recommendation['status']} - {signal} signal found.")
    return recommendation

# ==============================================================================
# Flask Application Part (MODIFICATIONS)
# ==============================================================================
app = Flask(__name__)
app.secret_key = os.urandom(24)

@app.route('/', methods=['GET', 'POST'])
def index():
    plot_html = None; error = None; analysis_summary_data = None; analysis_summary_pretty = None
    trade_recommendation_data = None # Variable for trade recommendation
    multi_stock_results = None # Variable for multiple stock analysis results
    
    # Load stock lists from TXT files
    stock_lists = {}
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        presets_dir = os.path.join(base_dir, 'stocks_presets')
        
        # Check if directory exists
        if os.path.exists(presets_dir):
            print(f"Loading stock lists from {presets_dir}")
            txt_files = glob.glob(os.path.join(presets_dir, '*.txt'))
            
            for path in txt_files:
                name = os.path.splitext(os.path.basename(path))[0]
                with open(path, 'r') as f:
                    content = f.read().strip()
                stock_lists[name] = content
                print(f"Loaded stock list: {name} with {len(content.split(','))} stocks")
        else:
            print(f"Stock presets directory not found: {presets_dir}")
    except Exception as e:
        print(f"Error loading stock lists: {str(e)}")
        traceback.print_exc()
    try: # Default dates
        today = datetime.date.today()
        default_end = today.strftime('%Y-%m-%d'); default_start = (today - datetime.timedelta(days=365*2)).strftime('%Y-%m-%d')
        default_analysis_date = (today - datetime.timedelta(days=90)).strftime('%Y-%m-%d'); default_check_date = default_end
    except Exception: default_end = '2025-04-18'; default_start = '2023-04-19'; default_analysis_date = '2025-01-19'; default_check_date = default_end
    form_values = { # Populate form values
        'ticker': request.form.get('ticker', '^SPX'), 'start_date': request.form.get('start_date', default_start),
        'end_date': request.form.get('end_date', default_end), 'interval': request.form.get('interval', '1d'),
        'analysis_date': request.form.get('analysis_date', default_analysis_date), 'check_date': request.form.get('check_date', default_check_date),
        'run_backtest': request.form.get('run_backtest', ''),
        # Risk parameters from form
        'risk_percent': request.form.get('risk_percent', '1.0'),
        'demo_account_size': request.form.get('demo_account_size', '10000'),
        # Get show trade levels checkbox value
        'show_trade_on_plot': request.form.get('show_trade_on_plot', ''),
        # Multi-stock analysis parameters
        'stock_list': request.form.get('stock_list', ''),
        'analysis_mode': request.form.get('analysis_mode', 'single'),
        # Parameter to track if we're showing a specific stock from multi-stock results
        'from_multi_stock': request.form.get('from_multi_stock', 'false'),
        'multi_stock_data': request.form.get('multi_stock_data', '')
    }
    if request.method == 'POST':
        request_start_time = datetime.datetime.now(); print(f"\n--- Received POST request at {request_start_time} ---")
        is_backtest = form_values.get('run_backtest') == 'true'; print(f"Backtest mode requested: {is_backtest}")
        # Convert checkbox value to boolean
        show_trade_on_plot = form_values.get('show_trade_on_plot') == 'true'
        print(f"Debug - Checkbox raw value: '{form_values.get('show_trade_on_plot')}', Converted to: {show_trade_on_plot}")
        # Check if we're doing multi-stock analysis
        is_multi_stock = form_values.get('analysis_mode') == 'multi'
        # Check if we're showing a plot from multi-stock results
        from_multi_stock = form_values.get('from_multi_stock') == 'true'
        print(f"Multi-stock analysis mode: {is_multi_stock}, From multi-stock: {from_multi_stock}")
        
        # If we have multi-stock data from a previous analysis, parse it
        previous_multi_stock_results = None
        multi_stock_data = form_values.get('multi_stock_data', '')
        if from_multi_stock and multi_stock_data:
            try:
                previous_multi_stock_results = json.loads(multi_stock_data)
                print(f"Loaded previous multi-stock results with {len(previous_multi_stock_results)} stocks")
            except Exception as e:
                print(f"Error loading previous multi-stock data: {e}")
        stock_data_result = None # To store stock data for recommendation function
        try:
            # Get common parameters for both single and multi-stock analysis
            start_date = form_values['start_date']; interval = form_values['interval']
            risk_percent_str = form_values['risk_percent']; demo_account_size_str = form_values['demo_account_size']
            
            # For multi-stock analysis, we'll use stock_list instead of ticker
            if is_multi_stock:
                stock_list_raw = form_values['stock_list'].strip()
                if not stock_list_raw:
                    error = "Stock list is required for multi-stock analysis."; raise ValueError(error)
                # Parse the comma-separated list and clean each ticker
                stock_list = [ticker.strip().upper() for ticker in stock_list_raw.split(',') if ticker.strip()]
                if not stock_list:
                    error = "No valid tickers found in stock list."; raise ValueError(error)
                print(f"Processing {len(stock_list)} stocks: {', '.join(stock_list)}")
            else:
                # Single stock analysis
                ticker = form_values['ticker'].strip().upper()

            # --- Validation ---
            if not is_multi_stock and not ticker: error = "Ticker symbol is required."; raise ValueError(error)
            if not start_date: error = "Start date is required."; raise ValueError(error)
            if not interval: error = "Interval is required."; raise ValueError(error)
            try:
                risk_percent = float(risk_percent_str)
                if not (0 < risk_percent <= 100): raise ValueError()
            except (ValueError, TypeError):
                error = "Invalid Risk Percent (must be a number > 0 and <= 100)."; raise ValueError(error)
            try:
                demo_account_size = float(demo_account_size_str)
                if not (demo_account_size > 0): raise ValueError()
            except (ValueError, TypeError):
                error = "Invalid Demo Account Size (must be a number > 0)."; raise ValueError(error)

            fig = None
            analysis_end_date_for_data = None # Date to fetch data up to
            
            # Handle multi-stock analysis
            if is_multi_stock:
                end_date = form_values['end_date']
                if not end_date: error = "End date is required for multi-stock analysis."; raise ValueError(error)
                try: # Validate date order
                    d_start = datetime.datetime.strptime(start_date, '%Y-%m-%d'); d_end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
                    if d_start >= d_end: error = "Start date must be before end date."; raise ValueError(error)
                except ValueError as ve: error = error or f"Invalid date format/order: {ve}"; raise ValueError(error)
                
                # Handle backtest mode in multi-stock analysis
                if is_backtest:
                    analysis_date = form_values['analysis_date']; check_date = form_values['check_date']
                    if not analysis_date: error = "Backtest 'Analysis Date' is required."; raise ValueError(error)
                    if not check_date: error = "Backtest 'Check Date' is required."; raise ValueError(error)
                    try: # Validate date order for backtest
                        d_start=datetime.datetime.strptime(start_date, '%Y-%m-%d'); d_analysis=datetime.datetime.strptime(analysis_date, '%Y-%m-%d'); d_check=datetime.datetime.strptime(check_date, '%Y-%m-%d')
                        if not (d_start < d_analysis < d_check): error = "For Backtest, dates must be in order: Start Date < Analysis Date < Check Date."; raise ValueError(error)
                    except ValueError as ve: error = error or f"Invalid date format/order: {ve}"; raise ValueError(error)
                
                # Process each stock in the list
                multi_stock_results = []
                
                # Backtest summary statistics
                backtest_summary = {
                    'total_trades': 0,
                    'hit_tp1': 0,
                    'hit_tp2': 0,
                    'hit_sl': 0,
                    'still_running': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'avg_win_pct': 0,
                    'avg_loss_pct': 0,
                    'total_win_pct': 0,
                    'total_loss_pct': 0
                }
                
                for current_ticker in stock_list:
                    print(f"Processing stock: {current_ticker}")
                    try:
                        current_fig = None
                        current_analysis_summary = None
                        current_backtest_stats = None
                        current_trade_recommendation = None
                        
                        # Run analysis based on mode (standard or backtest)
                        if is_backtest:
                            print(f"Running Backtest for {current_ticker}: Start={start_date}, Analysis={analysis_date}, Check={check_date}")
                            current_fig, current_analysis_summary = run_backtest_simulation(current_ticker, start_date, analysis_date, check_date, interval)
                            
                            # Extract trade recommendation and backtest stats
                            if current_analysis_summary:
                                if 'trade_recommendation' in current_analysis_summary:
                                    current_trade_recommendation = current_analysis_summary['trade_recommendation']
                                
                                if 'backtest_stats' in current_analysis_summary:
                                    current_backtest_stats = current_analysis_summary['backtest_stats']
                                    
                                    # Update backtest summary statistics
                                    if current_backtest_stats and current_trade_recommendation and current_trade_recommendation.get('status') == 'Trade Found':
                                        backtest_summary['total_trades'] += 1
                                        
                                        if current_backtest_stats['hit_tp1']:
                                            backtest_summary['hit_tp1'] += 1
                                        
                                        if current_backtest_stats['hit_tp2']:
                                            backtest_summary['hit_tp2'] += 1
                                        
                                        if current_backtest_stats['hit_sl']:
                                            backtest_summary['hit_sl'] += 1
                                            backtest_summary['losing_trades'] += 1
                                        elif current_backtest_stats['hit_tp1'] or current_backtest_stats['hit_tp2']:
                                            backtest_summary['winning_trades'] += 1
                                        
                                        if current_backtest_stats['status'] == 'still_running':
                                            backtest_summary['still_running'] += 1
                                        
                                        # Add to profit/loss calculations
                                        if current_backtest_stats['current_pct_change'] > 0:
                                            backtest_summary['total_win_pct'] += current_backtest_stats['current_pct_change']
                                        else:
                                            backtest_summary['total_loss_pct'] += current_backtest_stats['current_pct_change']
                        else:
                            # Standard analysis
                            current_fig, current_analysis_summary = run_analysis(current_ticker, start_date, end_date, interval)
                            
                            # Get trade recommendation
                            current_stock_data = get_stock_data(current_ticker, start_date, end_date, interval)
                            if current_stock_data is not None and not current_stock_data.empty and current_analysis_summary and not current_analysis_summary.get('error'):
                                current_trade_recommendation = generate_trade_recommendation(
                                    current_analysis_summary,
                                    current_stock_data,
                                    risk_percent=float(risk_percent_str),
                                    demo_account_size=float(demo_account_size_str)
                                )
                        
                        # Extract wave information and other useful data
                        wave_info = {}
                        if current_analysis_summary and not current_analysis_summary.get('error'):
                            # Get current wave
                            wave_info['current_wave'] = current_analysis_summary.get('last_label', 'Unknown')
                            
                            # Get impulse sequence status
                            wave_info['impulse_identified'] = current_analysis_summary.get('impulse_identified', False)
                            
                            # Get wave score
                            if 'details' in current_analysis_summary and 'sequence_score' in current_analysis_summary['details']:
                                wave_info['score'] = current_analysis_summary['details']['sequence_score']
                            else:
                                wave_info['score'] = 0
                                
                            # Get correction status if available
                            if 'details' in current_analysis_summary and 'correction_guess' in current_analysis_summary['details']:
                                wave_info['correction'] = True
                                correction_data = current_analysis_summary['details']['correction_guess']
                                if correction_data and 'C' in correction_data and correction_data['C']:
                                    wave_info['correction_target'] = 'C'
                                elif correction_data and 'B' in correction_data and correction_data['B']:
                                    wave_info['correction_target'] = 'B'
                                elif correction_data and 'A' in correction_data and correction_data['A']:
                                    wave_info['correction_target'] = 'A'
                                else:
                                    wave_info['correction_target'] = 'Unknown'
                            else:
                                wave_info['correction'] = False
                                wave_info['correction_target'] = 'N/A'
                        
                        # Add to results
                        multi_stock_results.append({
                            'ticker': current_ticker,
                            'analysis_summary_data': current_analysis_summary,
                            'trade_recommendation_data': current_trade_recommendation,
                            'wave_info': wave_info,
                            'backtest_stats': current_backtest_stats,
                            'error': current_analysis_summary.get('error') if current_analysis_summary else 'Analysis failed'
                        })
                    except Exception as stock_err:
                        print(f"Error processing {current_ticker}: {stock_err}")
                        traceback.print_exc()
                        multi_stock_results.append({
                            'ticker': current_ticker,
                            'analysis_summary_data': None,
                            'trade_recommendation_data': None,
                            'backtest_stats': None,
                            'error': f"Error: {stock_err}"
                        })
                
                # Calculate overall backtest statistics if we're in backtest mode
                if is_backtest and backtest_summary['total_trades'] > 0:
                    # Calculate win rate
                    total_decided = backtest_summary['winning_trades'] + backtest_summary['losing_trades']
                    if total_decided > 0:
                        backtest_summary['win_rate'] = round((backtest_summary['winning_trades'] / total_decided) * 100, 2)
                    
                    # Calculate average win/loss percentages
                    if backtest_summary['winning_trades'] > 0:
                        backtest_summary['avg_win_pct'] = round(backtest_summary['total_win_pct'] / backtest_summary['winning_trades'], 2)
                    
                    if backtest_summary['losing_trades'] > 0:
                        backtest_summary['avg_loss_pct'] = round(backtest_summary['total_loss_pct'] / backtest_summary['losing_trades'], 2)
                    
                    # Add the summary to each result item to make it available in the template
                    for result in multi_stock_results:
                        result['backtest_summary'] = backtest_summary
                
                # For display in the UI, we'll still show the first stock's chart if available
                if stock_list and multi_stock_results:
                    first_result = multi_stock_results[0]
                    if 'analysis_summary_data' in first_result and first_result['analysis_summary_data']:
                        analysis_summary_data = first_result['analysis_summary_data']
                        # Re-run analysis for the first stock to get the plot
                        if is_backtest:
                            fig, _ = run_backtest_simulation(stock_list[0], start_date, analysis_date, check_date, interval)
                        else:
                            fig, _ = run_analysis(stock_list[0], start_date, end_date, interval)
                        trade_recommendation_data = first_result.get('trade_recommendation_data')
                        
                # Save multi_stock_results for future requests
                try:
                    multi_stock_data_json = json.dumps([{
                        'ticker': result['ticker'],
                        'error': result.get('error', ''),
                        'trade_recommendation_data': result.get('trade_recommendation_data'),
                        'wave_info': result.get('wave_info', {}),
                        'backtest_stats': result.get('backtest_stats'),
                        'backtest_summary': backtest_summary if is_backtest else None
                    } for result in multi_stock_results])
                    form_values['multi_stock_data'] = multi_stock_data_json
                except Exception as e:
                    print(f"Error serializing multi-stock results: {e}")
            elif from_multi_stock and previous_multi_stock_results:
                # We're showing a plot for a specific stock from multi-stock results
                end_date = form_values['end_date']
                if not end_date: error = "End date is required."; raise ValueError(error)
                
                # Run analysis for the selected stock
                print(f"Showing plot for {ticker} from multi-stock results")
                fig, analysis_summary_data = run_analysis(ticker, start_date, end_date, interval)
                
                # Get trade recommendation for this stock
                stock_data_result = get_stock_data(ticker, start_date, end_date, interval)
                if stock_data_result is not None and not stock_data_result.empty and analysis_summary_data and not analysis_summary_data.get('error'):
                    trade_recommendation_data = generate_trade_recommendation(
                        analysis_summary_data,
                        stock_data_result,
                        risk_percent=float(risk_percent_str),
                        demo_account_size=float(demo_account_size_str)
                    )
                
                # Keep the multi-stock results to display in the table
                multi_stock_results = previous_multi_stock_results
                
            else:  # Regular single stock analysis
                if is_backtest:
                    analysis_date = form_values['analysis_date']; check_date = form_values['check_date']
                    if not analysis_date: error = "Backtest 'Analysis Date' is required."; raise ValueError(error)
                    if not check_date: error = "Backtest 'Check Date' is required."; raise ValueError(error)
                    try: # Validate date order for backtest
                        d_start=datetime.datetime.strptime(start_date, '%Y-%m-%d'); d_analysis=datetime.datetime.strptime(analysis_date, '%Y-%m-%d'); d_check=datetime.datetime.strptime(check_date, '%Y-%m-%d')
                        if not (d_start < d_analysis < d_check): error = "For Backtest, dates must be in order: Start Date < Analysis Date < Check Date."; raise ValueError(error)
                    except ValueError as ve: error = error or f"Invalid date format/order: {ve}"; raise ValueError(error)
                    print(f"Running Backtest: Ticker={ticker}, Start={start_date}, Analysis={analysis_date}, Check={check_date}, Interval={interval}")
                    fig, analysis_summary_data = run_backtest_simulation(ticker, start_date, analysis_date, check_date, interval)
                    analysis_end_date_for_data = analysis_date # Need data up to analysis date for recommendation
                else: # Standard Analysis
                    end_date = form_values['end_date']
                    if not end_date: error = "End date is required for standard analysis."; raise ValueError(error)
                    try: # Validate date order for standard analysis
                        d_start = datetime.datetime.strptime(start_date, '%Y-%m-%d'); d_end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
                        if d_start >= d_end: error = "Start date must be before end date."; raise ValueError(error)
                    except ValueError as ve: error = error or f"Invalid date format/order: {ve}"; raise ValueError(error)
                    print(f"Running Standard Analysis: Ticker={ticker}, Start={start_date}, End={end_date}, Interval={interval}")
                    fig, analysis_summary_data = run_analysis(ticker, start_date, end_date, interval)
                    analysis_end_date_for_data = end_date # Need data up to end date for recommendation
                
                # Generate trade recommendation for single stock analysis
                if analysis_summary_data and not analysis_summary_data.get('error') and analysis_end_date_for_data:
                    print("Fetching data again specifically for trade recommendation...")
                    # We need the data EXACTLY as it was at the end of the analysis period
                    stock_data_result = get_stock_data(ticker, start_date, analysis_end_date_for_data, interval)
                    if stock_data_result is not None and not stock_data_result.empty:
                        trade_recommendation_data = generate_trade_recommendation(
                            analysis_summary_data,
                            stock_data_result,
                            risk_percent=float(risk_percent_str),
                            demo_account_size=float(demo_account_size_str)
                        )
                    else:
                        print("  Warning: Could not get stock data for trade recommendation generation.")
                        trade_recommendation_data = {'status': 'Error', 'reason': 'Failed to retrieve necessary stock data for recommendation.'}
            # --- Process Analysis Errors ---
            if not is_multi_stock and analysis_summary_data and analysis_summary_data.get('error') and not error: 
                error = f"Analysis Error: {analysis_summary_data.get('error')}"; fig = None
                    
            # <<< NEW: Add Trade Levels to Plot (Simple but Effective) >>>
            print(f"  Debug - Fig exists: {fig is not None}, Show trade on plot: {show_trade_on_plot}, Trade rec data: {trade_recommendation_data is not None}")
            if trade_recommendation_data:
                print(f"  Debug - Trade status: {trade_recommendation_data.get('status')}")
            
            if fig and show_trade_on_plot and trade_recommendation_data and trade_recommendation_data.get('status') == 'Trade Found':
                print("  Adding trade levels to the plot...")
                try:
                    entry = trade_recommendation_data['entry_price']
                    sl = trade_recommendation_data['stop_loss_price']
                    tp1 = trade_recommendation_data.get('tp1_price')
                    tp2 = trade_recommendation_data.get('tp2_price')
                    signal = trade_recommendation_data.get('signal', '')
                    
                    # Get data range for proper positioning
                    if fig.data and hasattr(fig.data[0], 'x') and fig.data[0].x is not None and len(fig.data[0].x) > 0:
                        # Get first and last dates for horizontal span
                        first_date = fig.data[0].x[0]
                        last_date = fig.data[0].x[-1]
                        
                        # Calculate date range for proper positioning
                        try:
                            date_range = pd.to_datetime(last_date) - pd.to_datetime(first_date)
                            extension_days = max(1, int(date_range.days * 0.05))  # Extend 5% beyond the last date
                            extended_date = pd.to_datetime(last_date) + pd.Timedelta(days=extension_days)
                            
                            # Calculate a start point that's 65% through the chart (to position elements more to the right)
                            start_position_pct = 0.65  # Start at 65% through the chart
                            start_date = pd.to_datetime(first_date) + (date_range * start_position_pct)
                        except Exception as e:
                            print(f"  Warning: Error calculating date extension: {e}")
                            extended_date = last_date  # Fallback to last date if calculation fails
                            start_date = first_date    # Fallback to first date if calculation fails
                        
                        # Set colors based on signal type (Long/Short)
                        if signal == "Long":
                            entry_color = "rgba(0, 150, 255, 0.9)"  # Blue for long entry
                            sl_color = "rgba(255, 59, 59, 0.9)"     # Red for stop loss
                            tp_color = "rgba(0, 180, 75, 0.9)"      # Green for take profit
                            signal_direction = 1                    # For position calculations
                            signal_text = "LONG"                    # Signal label
                            risk_zone_color = "rgba(255, 0, 0, 0.1)" # Red tint for risk zone
                            profit_zone_color = "rgba(0, 255, 0, 0.1)" # Green tint for profit zone
                        else:  # Short
                            entry_color = "rgba(255, 59, 59, 0.9)"  # Red for short entry
                            sl_color = "rgba(255, 59, 59, 0.9)"     # Red for stop loss
                            tp_color = "rgba(0, 180, 75, 0.9)"      # Green for take profit
                            signal_direction = -1                   # For position calculations
                            signal_text = "SHORT"                   # Signal label
                            risk_zone_color = "rgba(255, 0, 0, 0.1)" # Red tint for risk zone
                            profit_zone_color = "rgba(0, 255, 0, 0.1)" # Green tint for profit zone
                        
                        # Add signal label with TradingView-style box
                        fig.add_annotation(
                            x=start_date,  # Position more to the right
                            y=entry,
                            text=signal_text,
                            showarrow=False,
                            xanchor="left",
                            yanchor="middle",
                            font=dict(color="white", size=12, family="Arial"),
                            bgcolor=entry_color,
                            bordercolor=entry_color,
                            borderwidth=1,
                            borderpad=4,
                            opacity=0.9
                        )
                        
                        # Add Entry Line with TradingView-style box
                        fig.add_shape(
                            type="line",
                            x0=start_date,  # Start more to the right
                            x1=extended_date,
                            y0=entry,
                            y1=entry,
                            line=dict(color=entry_color, width=2),
                            opacity=0.8
                        )
                        
                        # Calculate percentage changes
                        if signal == "Long":
                            sl_pct = ((sl - entry) / entry) * 100
                            tp1_pct = ((tp1 - entry) / entry) * 100 if tp1 is not None else None
                            tp2_pct = ((tp2 - entry) / entry) * 100 if tp2 is not None else None
                        else:  # Short
                            sl_pct = ((entry - sl) / entry) * 100
                            tp1_pct = ((entry - tp1) / entry) * 100 if tp1 is not None else None
                            tp2_pct = ((entry - tp2) / entry) * 100 if tp2 is not None else None
                        
                        # Set colors based on signal type (Long/Short)
                        if signal == "Long":
                            entry_color = "rgba(0, 150, 255, 0.9)"  # Blue for long entry
                            sl_color = "rgba(255, 59, 59, 0.9)"     # Red for stop loss
                            tp_color = "rgba(0, 180, 75, 0.9)"      # Green for take profit
                        else:  # Short
                            entry_color = "rgba(255, 59, 59, 0.9)"  # Red for short entry
                            sl_color = "rgba(255, 59, 59, 0.9)"     # Red for stop loss
                            tp_color = "rgba(0, 180, 75, 0.9)"      # Green for take profit
                        
                        # Add Entry Line with TradingView-style box
                        fig.add_shape(
                            type="line",
                            x0=start_date,  # Start more to the right
                            x1=extended_date,
                            y0=entry,
                            y1=entry,
                            line=dict(color=entry_color, width=2),
                            opacity=0.8
                        )
                        
                        fig.add_annotation(
                            x=extended_date,
                            y=entry,
                            text=f"Entry: {entry:.4f}",
                            showarrow=False,
                            xanchor="left",
                            yanchor="middle",
                            font=dict(color="white", size=11, family="Arial"),
                            bgcolor=entry_color,
                            bordercolor=entry_color,
                            borderwidth=1,
                            borderpad=4,
                            opacity=0.9
                        )
                        
                        # Add SL Line with TradingView-style box
                        fig.add_shape(
                            type="line",
                            x0=start_date,  # Start more to the right
                            x1=extended_date,
                            y0=sl,
                            y1=sl,
                            line=dict(color=sl_color, width=2),
                            opacity=0.8
                        )
                        
                        fig.add_annotation(
                            x=extended_date,
                            y=sl,
                            text=f"SL: {sl:.4f} ({sl_pct:+.2f}%)",
                            showarrow=False,
                            xanchor="left",
                            yanchor="middle",
                            font=dict(color="white", size=11, family="Arial"),
                            bgcolor=sl_color,
                            bordercolor=sl_color,
                            borderwidth=1,
                            borderpad=4,
                            opacity=0.9
                        )
                        
                        # Add risk zone shading between entry and SL
                        if (signal == "Long" and sl < entry) or (signal == "Short" and sl > entry):
                            fig.add_shape(
                                type="rect",
                                x0=start_date,  # Start more to the right
                                x1=extended_date,
                                y0=min(entry, sl),
                                y1=max(entry, sl),
                                fillcolor=risk_zone_color,
                                line=dict(width=0),
                                layer="below"
                            )
                        
                        # Add TP1 Line with TradingView-style box
                        if tp1 is not None:
                            fig.add_shape(
                                type="line",
                                x0=start_date,  # Start more to the right
                                x1=extended_date,
                                y0=tp1,
                                y1=tp1,
                                line=dict(color=tp_color, width=2),
                                opacity=0.8
                            )
                            
                            fig.add_annotation(
                                x=extended_date,
                                y=tp1,
                                text=f"TP1: {tp1:.4f} ({tp1_pct:+.2f}%)",
                                showarrow=False,
                                xanchor="left",
                                yanchor="middle",
                                font=dict(color="white", size=11, family="Arial"),
                                bgcolor=tp_color,
                                bordercolor=tp_color,
                                borderwidth=1,
                                borderpad=4,
                                opacity=0.9
                            )
                            
                            # Add profit zone shading between entry and TP1
                            if (signal == "Long" and tp1 > entry) or (signal == "Short" and tp1 < entry):
                                fig.add_shape(
                                    type="rect",
                                    x0=start_date,  # Start more to the right
                                    x1=extended_date,
                                    y0=min(entry, tp1),
                                    y1=max(entry, tp1),
                                    fillcolor=profit_zone_color,
                                    line=dict(width=0),
                                    layer="below"
                                )
                        
                        # Add TP2 Line with TradingView-style box
                        if tp2 is not None:
                            fig.add_shape(
                                type="line",
                                x0=start_date,  # Start more to the right
                                x1=extended_date,
                                y0=tp2,
                                y1=tp2,
                                line=dict(color=tp_color, width=2),
                                opacity=0.8
                            )
                            
                            fig.add_annotation(
                                x=extended_date,
                                y=tp2,
                                text=f"TP2: {tp2:.4f} ({tp2_pct:+.2f}%)",
                                showarrow=False,
                                xanchor="left",
                                yanchor="middle",
                                font=dict(color="white", size=11, family="Arial"),
                                bgcolor=tp_color,
                                bordercolor=tp_color,
                                borderwidth=1,
                                borderpad=4,
                                opacity=0.9
                            )
                            
                            # Add profit zone shading between TP1 and TP2 (if both exist)
                            if tp1 is not None:
                                if (signal == "Long" and tp2 > tp1) or (signal == "Short" and tp2 < tp1):
                                    fig.add_shape(
                                        type="rect",
                                        x0=start_date,  # Start more to the right
                                        x1=extended_date,
                                        y0=min(tp1, tp2),
                                        y1=max(tp1, tp2),
                                        fillcolor=profit_zone_color,
                                        line=dict(width=0),
                                        layer="below"
                                    )
                    else:
                        # Fallback to simple horizontal lines if we can't get the date range
                        print("  Warning: Using simple horizontal lines as fallback")
                        # Set colors based on signal type (Long/Short)
                        if signal == "Long":
                            entry_color = "rgba(0, 150, 255, 0.9)"  # Blue for long entry
                            sl_color = "rgba(255, 59, 59, 0.9)"     # Red for stop loss
                            tp_color = "rgba(0, 180, 75, 0.9)"      # Green for take profit
                        else:  # Short
                            entry_color = "rgba(255, 59, 59, 0.9)"  # Red for short entry
                            sl_color = "rgba(255, 59, 59, 0.9)"     # Red for stop loss
                            tp_color = "rgba(0, 180, 75, 0.9)"      # Green for take profit
                        
                        # Calculate percentage changes
                        if signal == "Long":
                            sl_pct = ((sl - entry) / entry) * 100
                            tp1_pct = ((tp1 - entry) / entry) * 100 if tp1 is not None else None
                            tp2_pct = ((tp2 - entry) / entry) * 100 if tp2 is not None else None
                        else:  # Short
                            sl_pct = ((entry - sl) / entry) * 100
                            tp1_pct = ((entry - tp1) / entry) * 100 if tp1 is not None else None
                            tp2_pct = ((entry - tp2) / entry) * 100 if tp2 is not None else None
                        
                        # Set colors based on signal type (Long/Short)
                        if signal == "Long":
                            entry_color = "rgba(0, 150, 255, 0.9)"  # Blue for long entry
                            sl_color = "rgba(255, 59, 59, 0.9)"     # Red for stop loss
                            tp_color = "rgba(0, 180, 75, 0.9)"      # Green for take profit
                        else:  # Short
                            entry_color = "rgba(255, 59, 59, 0.9)"  # Red for short entry
                            sl_color = "rgba(255, 59, 59, 0.9)"     # Red for stop loss
                            tp_color = "rgba(0, 180, 75, 0.9)"      # Green for take profit
                        
                        # Add Entry Line with TradingView-style box
                        fig.add_shape(
                            type="line",
                            x0=start_date,  # Start more to the right
                            x1=extended_date,
                            y0=entry,
                            y1=entry,
                            line=dict(color=entry_color, width=2),
                            opacity=0.8
                        )
                        
                        fig.add_annotation(
                            x=extended_date,
                            y=entry,
                            text=f"Entry: {entry:.4f}",
                            showarrow=False,
                            xanchor="left",
                            yanchor="middle",
                            font=dict(color="white", size=11, family="Arial"),
                            bgcolor=entry_color,
                            bordercolor=entry_color,
                            borderwidth=1,
                            borderpad=4,
                            opacity=0.9
                        )
                        
                        # Add SL Line with TradingView-style box
                        fig.add_shape(
                            type="line",
                            x0=start_date,  # Start more to the right
                            x1=extended_date,
                            y0=sl,
                            y1=sl,
                            line=dict(color=sl_color, width=2),
                            opacity=0.8
                        )
                        
                        fig.add_annotation(
                            x=extended_date,
                            y=sl,
                            text=f"SL: {sl:.4f} ({sl_pct:+.2f}%)",
                            showarrow=False,
                            xanchor="left",
                            yanchor="middle",
                            font=dict(color="white", size=11, family="Arial"),
                            bgcolor=sl_color,
                            bordercolor=sl_color,
                            borderwidth=1,
                            borderpad=4,
                            opacity=0.9
                        )
                        
                        # Add risk zone shading between entry and SL
                        if (signal == "Long" and sl < entry) or (signal == "Short" and sl > entry):
                            fig.add_shape(
                                type="rect",
                                x0=start_date,  # Start more to the right
                                x1=extended_date,
                                y0=min(entry, sl),
                                y1=max(entry, sl),
                                fillcolor=risk_zone_color,
                                line=dict(width=0),
                                layer="below"
                            )
                        
                        # Add TP1 Line with TradingView-style box
                        if tp1 is not None:
                            fig.add_shape(
                                type="line",
                                x0=start_date,  # Start more to the right
                                x1=extended_date,
                                y0=tp1,
                                y1=tp1,
                                line=dict(color=tp_color, width=2),
                                opacity=0.8
                            )
                            
                            fig.add_annotation(
                                x=extended_date,
                                y=tp1,
                                text=f"TP1: {tp1:.4f} ({tp1_pct:+.2f}%)",
                                showarrow=False,
                                xanchor="left",
                                yanchor="middle",
                                font=dict(color="white", size=11, family="Arial"),
                                bgcolor=tp_color,
                                bordercolor=tp_color,
                                borderwidth=1,
                                borderpad=4,
                                opacity=0.9
                            )
                            
                            # Add profit zone shading between entry and TP1
                            if (signal == "Long" and tp1 > entry) or (signal == "Short" and tp1 < entry):
                                fig.add_shape(
                                    type="rect",
                                    x0=start_date,  # Start more to the right
                                    x1=extended_date,
                                    y0=min(entry, tp1),
                                    y1=max(entry, tp1),
                                    fillcolor=profit_zone_color,
                                    line=dict(width=0),
                                    layer="below"
                                )
                        
                        # Add TP2 Line with TradingView-style box
                        if tp2 is not None:
                            fig.add_shape(
                                type="line",
                                x0=start_date,  # Start more to the right
                                x1=extended_date,
                                y0=tp2,
                                y1=tp2,
                                line=dict(color=tp_color, width=2),
                                opacity=0.8
                            )
                            
                            fig.add_annotation(
                                x=extended_date,
                                y=tp2,
                                text=f"TP2: {tp2:.4f} ({tp2_pct:+.2f}%)",
                                showarrow=False,
                                xanchor="left",
                                yanchor="middle",
                                font=dict(color="white", size=11, family="Arial"),
                                bgcolor=tp_color,
                                bordercolor=tp_color,
                                borderwidth=1,
                                borderpad=4,
                                opacity=0.9
                            )
                            
                            # Add profit zone shading between TP1 and TP2 (if both exist)
                            if tp1 is not None:
                                if (signal == "Long" and tp2 > tp1) or (signal == "Short" and tp2 < tp1):
                                    fig.add_shape(
                                        type="rect",
                                        x0=start_date,  # Start more to the right
                                        x1=extended_date,
                                        y0=min(tp1, tp2),
                                        y1=max(tp1, tp2),
                                        fillcolor=profit_zone_color,
                                        line=dict(width=0),
                                        layer="below"
                                    )
                    
                    # Add Risk:Reward annotation
                    risk_reward_ratio = trade_recommendation_data.get('tp1_rrr', 0)
                    if risk_reward_ratio > 0:
                        # Add a text annotation for R:R ratio
                        fig.add_annotation(
                            x=0.05,  # 5% from left edge
                            y=0.95,  # 5% from top edge
                            xref="paper",
                            yref="paper",
                            text=f"R:R = 1:{risk_reward_ratio:.2f}",
                            showarrow=False,
                            font=dict(color="white", size=12),
                            bgcolor="rgba(70, 70, 70, 0.8)",
                            borderpad=4,
                            opacity=0.9
                        )

                except Exception as plot_add_err:
                    print(f"  Warning: Failed to add trade levels to plot: {plot_add_err}")
                    traceback.print_exc()
                    if isinstance(analysis_summary_data, dict):
                        analysis_summary_data['warning'] = analysis_summary_data.get('warning', "") + "; Failed to add trade levels to plot"

            # --- Generate plot HTML (after all modifications) ---
            if fig:
                try: 
                    # Enhanced Plotly configuration for interactive shapes
                    # Create a safe filename for the export
                    export_filename = 'elliott_wave_analysis'
                    if 'ticker' in locals() and ticker:
                        export_filename = f'{ticker}_{interval}_analysis'
                    elif 'multi_stock_results' in locals() and multi_stock_results:
                        export_filename = f'multi_stock_analysis'
                    
                    plotly_config = {
                        'displayModeBar': True,
                        'editable': True,  # Make the plot editable
                        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],  # Add drawing tools
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': export_filename,
                            'height': 800,
                            'width': 1200,
                            'scale': 2
                        }
                    }
                    
                    # Make shapes editable
                    for i, shape in enumerate(fig.layout.shapes):
                        # Set editable properties for target boxes
                        if shape.type == 'rect':
                            shape.editable = True
                            shape.layer = 'above'
                    
                    plot_html = plotly.offline.plot(fig, output_type='div', include_plotlyjs='cdn', config=plotly_config)
                    print("Plotly figure converted to HTML div with interactive features.")
                except Exception as plot_conversion_err: 
                    print(f"!!! Error converting plot to HTML: {plot_conversion_err}")
                    traceback.print_exc()
                    error = error or f"Failed to render plot: {plot_conversion_err}"
                    plot_html = None
                    fig = None
            elif not error and request.method == 'POST': 
                error = "Analysis completed, but no plot could be generated. Check logs or data validity."

            # --- Generate pretty JSON summary ---
            if analysis_summary_data:
                try: cleaned_summary_data_for_json = clean_data_for_json(analysis_summary_data); analysis_summary_pretty = json.dumps(cleaned_summary_data_for_json, indent=2, allow_nan=False)
                except Exception as json_err: print(f"!!! Error during json.dumps of analysis summary: {json_err}"); traceback.print_exc(); error_detail = f"Could not display summary details: JSON error '{json_err}'."; analysis_summary_pretty = error_detail; error = error or error_detail
        except ValueError as ve: print(f"Validation Error: {ve}") # Error already set in the checks above
        except Exception as e: print(f"!!! Unhandled Error in Flask route: {e}"); traceback.print_exc(); error = f"App error: {e}"; plot_html = None; analysis_summary_data = None; analysis_summary_pretty = None; trade_recommendation_data = None
        request_end_time = datetime.datetime.now(); print(f"--- Finished POST request processing in {request_end_time - request_start_time} ---")
    # Pass all data to template
    return render_template('index.html',
                           plot_html=plot_html,
                           error=error,
                           analysis_summary_data=analysis_summary_data,
                           analysis_summary_pretty=analysis_summary_pretty,
                           trade_recommendation_data=trade_recommendation_data,
                           multi_stock_results=multi_stock_results,  # Pass multi-stock results
                           form_values=form_values,
                           default_start_date=default_start,
                           default_end_date=default_end,
                           default_analysis_date=default_analysis_date,
                           default_check_date=default_check_date,
                           stock_lists=stock_lists)  # Pass stock lists to template

if __name__ == '__main__':
    print("\n--- Starting Flask Server ---")
    print("Go to http://0.0.0.0:5001/ in your browser.")
    print("Press CTRL+C to stop the server.")
    
    # Print all registered routes for debugging
    print("\n--- Registered Routes ---")
    for rule in app.url_map.iter_rules():
        methods = ','.join([method for method in rule.methods if method not in ['HEAD', 'OPTIONS']])
        print(f"Route: {rule} - Methods: {methods} - Endpoint: {rule.endpoint}")
    
    # Make sure Flask is listening on all interfaces
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)