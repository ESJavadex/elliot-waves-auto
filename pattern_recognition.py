"""
Advanced Elliott Wave Pattern Recognition Module
-----------------------------------------------
This module provides enhanced pattern recognition for Elliott Wave analysis,
focusing on identifying high-probability trading setups with multiple confirmations.
"""

import numpy as np
import pandas as pd
import math
from scipy.signal import find_peaks
import talib as ta

# --- Constants ---
# Fibonacci levels
FIB_LEVELS = {
    "retracement": [0.236, 0.382, 0.5, 0.618, 0.786, 0.886],
    "extension": [1.0, 1.272, 1.414, 1.618, 2.0, 2.618, 3.618, 4.236],
    "projection": [0.618, 1.0, 1.618, 2.0, 2.618]
}

# Wave relationship tolerances
TOLERANCE = 0.05  # 5% tolerance for wave relationships
STRICT_TOLERANCE = 0.03  # 3% tolerance for critical rules

# Scoring weights
WEIGHT_RULE = 100       # Critical rules
WEIGHT_GUIDELINE = 30   # Important guidelines
WEIGHT_FIB_HIT = 20     # Fibonacci relationships
WEIGHT_DIVERGENCE = 25  # Momentum divergence
WEIGHT_VOLUME = 15      # Volume patterns
WEIGHT_PATTERN = 50     # Pattern completion

# --- Pattern Recognition Functions ---

def identify_impulse_wave(data, wave_points, is_up=True):
    """
    Identify a high-probability impulse wave pattern with strict rule enforcement.
    
    Args:
        data: DataFrame with price and indicator data
        wave_points: DataFrame of potential wave points
        is_up: Boolean indicating if this is an uptrend (True) or downtrend (False)
        
    Returns:
        Dictionary with pattern details and confidence score
    """
    if len(wave_points) < 5:
        return {"valid": False, "score": 0, "reason": "Insufficient points for impulse wave"}
    
    # Extract the first 5 points for impulse wave analysis
    points = wave_points.iloc[:5].copy()
    
    # Initialize result
    result = {
        "valid": False,
        "score": 0,
        "pattern_type": "impulse",
        "is_up": is_up,
        "rules_passed": {},
        "guidelines_passed": {},
        "fib_relationships": {},
        "wave_measurements": {},
        "confirmation_indicators": {}
    }
    
    # Extract price levels based on trend direction
    price_levels = {}
    for i, point in enumerate(points.itertuples()):
        if i % 2 == 0:  # Waves 0, 2, 4 (even indices)
            price_levels[i] = point.Low if is_up else point.High
        else:  # Waves 1, 3, 5 (odd indices)
            price_levels[i] = point.High if is_up else point.Low
    
    # --- Rule 1: Wave 2 never retraces more than 100% of Wave 1 ---
    w2_retrace = (price_levels[1] - price_levels[2]) / (price_levels[1] - price_levels[0])
    rule1_passed = 0 <= w2_retrace <= 1.0
    result["rules_passed"]["wave2_retrace"] = rule1_passed
    
    if not rule1_passed:
        return {**result, "reason": "Wave 2 retraces more than 100% of Wave 1"}
    
    # --- Rule 2: Wave 3 is never the shortest among Waves 1, 3, and 5 ---
    wave1_length = abs(price_levels[1] - price_levels[0])
    wave3_length = abs(price_levels[3] - price_levels[2])
    wave5_length = abs(price_levels[4] - price_levels[3])
    
    rule2_passed = not (wave3_length < wave1_length and wave3_length < wave5_length)
    result["rules_passed"]["wave3_not_shortest"] = rule2_passed
    
    if not rule2_passed:
        return {**result, "reason": "Wave 3 is the shortest impulse wave"}
    
    # --- Rule 3: Wave 4 never overlaps Wave 1 (except in diagonal) ---
    if is_up:
        rule3_passed = price_levels[4] > price_levels[1]
    else:
        rule3_passed = price_levels[4] < price_levels[1]
    
    result["rules_passed"]["wave4_no_overlap"] = rule3_passed
    
    if not rule3_passed:
        # Check if this could be an ending diagonal
        diagonal_candidate = check_ending_diagonal(price_levels, is_up)
        if diagonal_candidate["valid"]:
            result["pattern_type"] = "ending_diagonal"
            result["rules_passed"]["ending_diagonal"] = True
            result["score"] += WEIGHT_PATTERN
        else:
            return {**result, "reason": "Wave 4 overlaps Wave 1 (not a valid diagonal)"}
    
    # --- Guideline 1: Wave 3 is typically the longest wave ---
    guideline1_passed = wave3_length > wave1_length and wave3_length > wave5_length
    result["guidelines_passed"]["wave3_longest"] = guideline1_passed
    
    # --- Guideline 2: Wave 5 is often equal to Wave 1 or has a 0.618 or 1.618 relationship ---
    w5_w1_ratio = wave5_length / wave1_length
    fib_relationship = is_fibonacci_ratio(w5_w1_ratio, FIB_LEVELS["extension"], TOLERANCE)
    result["guidelines_passed"]["wave5_fib_relation"] = fib_relationship
    result["fib_relationships"]["wave5_to_wave1"] = w5_w1_ratio
    
    # --- Guideline 3: Alternation between Waves 2 and 4 ---
    w2_shallow = w2_retrace < 0.5
    w4_retrace = (price_levels[3] - price_levels[4]) / (price_levels[3] - price_levels[2])
    w4_shallow = w4_retrace < 0.5
    
    alternation = (w2_shallow and not w4_shallow) or (not w2_shallow and w4_shallow)
    result["guidelines_passed"]["wave2_4_alternation"] = alternation
    result["fib_relationships"]["wave2_retrace"] = w2_retrace
    result["fib_relationships"]["wave4_retrace"] = w4_retrace
    
    # --- Calculate wave measurements for reference ---
    result["wave_measurements"] = {
        "wave1_length": wave1_length,
        "wave2_retrace": w2_retrace,
        "wave3_length": wave3_length,
        "wave3_extension": wave3_length / wave1_length,
        "wave4_retrace": w4_retrace,
        "wave5_length": wave5_length,
        "wave5_to_wave1_ratio": w5_w1_ratio
    }
    
    # --- Calculate confidence score ---
    score = 0
    
    # Add points for rules passed
    for rule_passed in result["rules_passed"].values():
        if rule_passed:
            score += WEIGHT_RULE
    
    # Add points for guidelines followed
    for guideline_passed in result["guidelines_passed"].values():
        if guideline_passed:
            score += WEIGHT_GUIDELINE
    
    # Add points for Fibonacci relationships
    for fib_relation in result["fib_relationships"].values():
        if is_fibonacci_ratio(fib_relation, FIB_LEVELS["retracement"] + FIB_LEVELS["extension"], TOLERANCE):
            score += WEIGHT_FIB_HIT
    
    # Check for momentum divergence between Waves 3 and 5
    if "RSI" in data.columns:
        try:
            # Get indices for waves 3 and 5
            wave3_idx = points.index[3]
            wave5_idx = points.index[4]
            
            # Get RSI values
            rsi3 = data.loc[wave3_idx, "RSI"]
            rsi5 = data.loc[wave5_idx, "RSI"]
            
            # Check for bearish divergence in uptrend or bullish divergence in downtrend
            if (is_up and price_levels[4] > price_levels[3] and rsi5 < rsi3) or \
               (not is_up and price_levels[4] < price_levels[3] and rsi5 > rsi3):
                result["confirmation_indicators"]["rsi_divergence"] = True
                score += WEIGHT_DIVERGENCE
            else:
                result["confirmation_indicators"]["rsi_divergence"] = False
        except:
            result["confirmation_indicators"]["rsi_divergence"] = "Error calculating"
    
    # Check volume pattern
    if "Volume" in data.columns:
        try:
            # Get average volume for each wave
            wave_volumes = {}
            for i in range(1, 5):  # Waves 1-5
                start_idx = points.index[i-1]
                end_idx = points.index[i]
                wave_volumes[i] = data.loc[start_idx:end_idx, "Volume"].mean()
            
            # Ideal volume pattern: Wave 3 > Wave 1 > Wave 5
            if wave_volumes[3] > wave_volumes[1] and wave_volumes[1] > wave_volumes[5]:
                result["confirmation_indicators"]["ideal_volume"] = True
                score += WEIGHT_VOLUME
            # Secondary pattern: Wave 3 > Wave 1 and Wave 3 > Wave 5
            elif wave_volumes[3] > wave_volumes[1] and wave_volumes[3] > wave_volumes[5]:
                result["confirmation_indicators"]["good_volume"] = True
                score += WEIGHT_VOLUME / 2
            else:
                result["confirmation_indicators"]["volume_pattern"] = False
        except:
            result["confirmation_indicators"]["volume_pattern"] = "Error calculating"
    
    # Update final score and validity
    result["score"] = score
    result["valid"] = score >= 200  # Minimum threshold for valid pattern
    
    return result

def check_ending_diagonal(price_levels, is_up):
    """
    Check if the price levels form a valid ending diagonal pattern.
    
    Args:
        price_levels: Dictionary of price levels for each wave point
        is_up: Boolean indicating if this is an uptrend (True) or downtrend (False)
        
    Returns:
        Dictionary with pattern validity and details
    """
    result = {"valid": False, "score": 0}
    
    # Rule 1: All waves are zigzags (can't check this with just price levels)
    
    # Rule 2: Wave 4 overlaps Wave 1
    if is_up:
        rule2 = price_levels[4] <= price_levels[1]
    else:
        rule2 = price_levels[4] >= price_levels[1]
    
    if not rule2:
        return result  # Not a diagonal if Wave 4 doesn't overlap Wave 1
    
    # Rule 3: Wave 3 is shorter than Wave 1
    wave1_length = abs(price_levels[1] - price_levels[0])
    wave3_length = abs(price_levels[3] - price_levels[2])
    rule3 = wave3_length < wave1_length
    
    # Rule 4: Wave 5 is shorter than Wave 3
    wave5_length = abs(price_levels[4] - price_levels[3])
    rule4 = wave5_length < wave3_length
    
    # Rule 5: Wedge formation (converging trendlines)
    if is_up:
        upper_slope1 = (price_levels[3] - price_levels[1]) / 2
        upper_slope2 = (price_levels[4] - price_levels[3])
        lower_slope1 = (price_levels[2] - price_levels[0]) / 2
        lower_slope2 = (price_levels[4] - price_levels[2]) / 2
    else:
        upper_slope1 = (price_levels[0] - price_levels[2]) / 2
        upper_slope2 = (price_levels[2] - price_levels[4]) / 2
        lower_slope1 = (price_levels[1] - price_levels[3]) / 2
        lower_slope2 = (price_levels[3] - price_levels[4])
    
    converging = (upper_slope2 < upper_slope1) and (lower_slope2 < lower_slope1)
    
    # Combine all rules
    result["valid"] = rule2 and rule3 and rule4 and converging
    result["score"] = sum([rule2, rule3, rule4, converging]) * WEIGHT_GUIDELINE
    
    return result

def identify_corrective_pattern(data, wave_points, is_up=True):
    """
    Identify corrective wave patterns (zigzag, flat, triangle, combination).
    
    Args:
        data: DataFrame with price and indicator data
        wave_points: DataFrame of potential wave points
        is_up: Boolean indicating if the main trend is up (True) or down (False)
        
    Returns:
        Dictionary with pattern details and confidence score
    """
    if len(wave_points) < 3:
        return {"valid": False, "score": 0, "reason": "Insufficient points for corrective pattern"}
    
    # For corrective patterns, we're looking for moves against the main trend
    # So we invert the is_up parameter
    correction_is_up = not is_up
    
    # Initialize result
    result = {
        "valid": False,
        "score": 0,
        "pattern_type": "unknown",
        "is_up": correction_is_up,
        "rules_passed": {},
        "guidelines_passed": {},
        "fib_relationships": {},
        "wave_measurements": {},
        "confirmation_indicators": {}
    }
    
    # Try to identify different corrective patterns
    zigzag_result = identify_zigzag(data, wave_points, correction_is_up)
    flat_result = identify_flat(data, wave_points, correction_is_up)
    triangle_result = identify_triangle(data, wave_points, correction_is_up)
    
    # Select the pattern with the highest score
    patterns = [
        ("zigzag", zigzag_result),
        ("flat", flat_result),
        ("triangle", triangle_result)
    ]
    
    best_pattern = max(patterns, key=lambda x: x[1]["score"])
    
    if best_pattern[1]["valid"]:
        result = best_pattern[1]
        result["pattern_type"] = best_pattern[0]
    
    return result

def identify_zigzag(data, wave_points, is_up):
    """
    Identify a zigzag corrective pattern (A-B-C).
    
    Args:
        data: DataFrame with price and indicator data
        wave_points: DataFrame of potential wave points
        is_up: Boolean indicating if this correction is up (True) or down (False)
        
    Returns:
        Dictionary with pattern details and confidence score
    """
    if len(wave_points) < 3:
        return {"valid": False, "score": 0, "reason": "Insufficient points for zigzag"}
    
    # Extract the first 3 points for A-B-C analysis
    points = wave_points.iloc[:3].copy()
    
    # Initialize result
    result = {
        "valid": False,
        "score": 0,
        "pattern_type": "zigzag",
        "is_up": is_up,
        "rules_passed": {},
        "guidelines_passed": {},
        "fib_relationships": {},
        "wave_measurements": {},
        "confirmation_indicators": {}
    }
    
    # Extract price levels based on correction direction
    price_levels = {}
    for i, point in enumerate(points.itertuples()):
        if i == 0:  # Wave A start
            price_levels[i] = point.Low if is_up else point.High
        elif i == 1:  # Wave A end / Wave B start
            price_levels[i] = point.High if is_up else point.Low
        elif i == 2:  # Wave B end / Wave C start
            price_levels[i] = point.Low if is_up else point.High
    
    # --- Rule 1: Wave B retraces less than 100% of Wave A ---
    wave_a_length = abs(price_levels[1] - price_levels[0])
    wave_b_retrace = abs(price_levels[2] - price_levels[1]) / wave_a_length
    
    rule1_passed = wave_b_retrace < 1.0
    result["rules_passed"]["wave_b_retrace"] = rule1_passed
    
    if not rule1_passed:
        return {**result, "reason": "Wave B retraces more than 100% of Wave A"}
    
    # --- Rule 2: Wave B retraces between 38.2% and 78.6% of Wave A (typical) ---
    rule2_passed = 0.382 <= wave_b_retrace <= 0.786
    result["guidelines_passed"]["wave_b_typical_retrace"] = rule2_passed
    
    # --- Guideline: Wave C is often equal to Wave A or has a Fibonacci relationship ---
    if len(points) > 2:
        wave_c_end = points.iloc[2].High if is_up else points.iloc[2].Low
        wave_c_length = abs(wave_c_end - price_levels[2])
        wave_c_to_a_ratio = wave_c_length / wave_a_length
        
        fib_relationship = is_fibonacci_ratio(wave_c_to_a_ratio, FIB_LEVELS["extension"], TOLERANCE)
        result["guidelines_passed"]["wave_c_fib_relation"] = fib_relationship
        result["fib_relationships"]["wave_c_to_wave_a"] = wave_c_to_a_ratio
        
        # --- Calculate wave measurements for reference ---
        result["wave_measurements"] = {
            "wave_a_length": wave_a_length,
            "wave_b_retrace": wave_b_retrace,
            "wave_c_length": wave_c_length,
            "wave_c_to_wave_a_ratio": wave_c_to_a_ratio
        }
    
    # --- Calculate confidence score ---
    score = 0
    
    # Add points for rules passed
    for rule_passed in result["rules_passed"].values():
        if rule_passed:
            score += WEIGHT_RULE
    
    # Add points for guidelines followed
    for guideline_passed in result["guidelines_passed"].values():
        if guideline_passed:
            score += WEIGHT_GUIDELINE
    
    # Add points for Fibonacci relationships
    for fib_relation in result["fib_relationships"].values():
        if is_fibonacci_ratio(fib_relation, FIB_LEVELS["retracement"] + FIB_LEVELS["extension"], TOLERANCE):
            score += WEIGHT_FIB_HIT
    
    # Update final score and validity
    result["score"] = score
    result["valid"] = score >= 150  # Lower threshold for corrective patterns
    
    return result

def identify_flat(data, wave_points, is_up):
    """
    Identify a flat corrective pattern (A-B-C where B retraces ~100% of A).
    
    Args:
        data: DataFrame with price and indicator data
        wave_points: DataFrame of potential wave points
        is_up: Boolean indicating if this correction is up (True) or down (False)
        
    Returns:
        Dictionary with pattern details and confidence score
    """
    if len(wave_points) < 3:
        return {"valid": False, "score": 0, "reason": "Insufficient points for flat"}
    
    # Extract the first 3 points for A-B-C analysis
    points = wave_points.iloc[:3].copy()
    
    # Initialize result
    result = {
        "valid": False,
        "score": 0,
        "pattern_type": "flat",
        "is_up": is_up,
        "rules_passed": {},
        "guidelines_passed": {},
        "fib_relationships": {},
        "wave_measurements": {},
        "confirmation_indicators": {}
    }
    
    # Extract price levels based on correction direction
    price_levels = {}
    for i, point in enumerate(points.itertuples()):
        if i == 0:  # Wave A start
            price_levels[i] = point.Low if is_up else point.High
        elif i == 1:  # Wave A end / Wave B start
            price_levels[i] = point.High if is_up else point.Low
        elif i == 2:  # Wave B end / Wave C start
            price_levels[i] = point.Low if is_up else point.High
    
    # --- Rule 1: Wave B retraces close to 100% of Wave A ---
    wave_a_length = abs(price_levels[1] - price_levels[0])
    wave_b_retrace = abs(price_levels[2] - price_levels[1]) / wave_a_length
    
    rule1_passed = 0.9 <= wave_b_retrace <= 1.1  # Allow slight variation
    result["rules_passed"]["wave_b_retrace"] = rule1_passed
    
    if not rule1_passed:
        return {**result, "reason": "Wave B does not retrace close to 100% of Wave A"}
    
    # --- Calculate confidence score ---
    score = 0
    
    # Add points for rules passed
    for rule_passed in result["rules_passed"].values():
        if rule_passed:
            score += WEIGHT_RULE
    
    # Update final score and validity
    result["score"] = score
    result["valid"] = score >= 100  # Lower threshold for flat patterns
    
    return result

def identify_triangle(data, wave_points, is_up):
    """
    Identify a triangle corrective pattern (A-B-C-D-E).
    
    Args:
        data: DataFrame with price and indicator data
        wave_points: DataFrame of potential wave points
        is_up: Boolean indicating if this correction is up (True) or down (False)
        
    Returns:
        Dictionary with pattern details and confidence score
    """
    if len(wave_points) < 5:
        return {"valid": False, "score": 0, "reason": "Insufficient points for triangle"}
    
    # Triangles require at least 5 points (A-B-C-D-E)
    # This is a simplified implementation
    return {"valid": False, "score": 0, "reason": "Triangle detection not fully implemented"}

def is_fibonacci_ratio(ratio, fib_levels, tolerance=0.05):
    """
    Check if a ratio is close to any Fibonacci level within tolerance.
    
    Args:
        ratio: The ratio to check
        fib_levels: List of Fibonacci levels to check against
        tolerance: Acceptable deviation from the Fibonacci level
        
    Returns:
        Boolean indicating if the ratio is close to a Fibonacci level
    """
    if math.isnan(ratio):
        return False
    
    for level in fib_levels:
        if abs(ratio - level) <= tolerance:
            return True
    
    return False

# --- Helper Functions ---

def clean_data_for_analysis(data):
    """
    Prepare data for Elliott Wave analysis by calculating necessary indicators.
    
    Args:
        data: DataFrame with OHLCV data
        
    Returns:
        DataFrame with additional indicators
    """
    df = data.copy()
    
    # Add RSI
    if "Close" in df.columns and len(df) > 14:
        try:
            df["RSI"] = ta.RSI(df["Close"].values, timeperiod=14)
        except:
            # Fallback if talib is not available
            try:
                delta = df["Close"].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                df["RSI"] = 100 - (100 / (1 + rs))
            except:
                pass
    
    # Add MACD
    if "Close" in df.columns and len(df) > 26:
        try:
            macd, signal, hist = ta.MACD(df["Close"].values, fastperiod=12, slowperiod=26, signalperiod=9)
            df["MACD"] = macd
            df["MACD_Signal"] = signal
            df["MACD_Hist"] = hist
        except:
            pass
    
    # Add ATR for volatility measurement
    if all(col in df.columns for col in ["High", "Low", "Close"]) and len(df) > 14:
        try:
            df["ATR"] = ta.ATR(df["High"].values, df["Low"].values, df["Close"].values, timeperiod=14)
        except:
            pass
    
    # Add Bollinger Bands
    if "Close" in df.columns and len(df) > 20:
        try:
            df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = ta.BBANDS(
                df["Close"].values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
            )
        except:
            pass
    
    return df
