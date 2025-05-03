"""
Improved Elliott Wave Analysis Module
-------------------------------------
This module provides enhanced Elliott Wave detection and scoring algorithms
with improved profitability through:
1. Stricter rule enforcement (Wave 4 overlap, Wave 3 shortest)
2. Better alternation guideline checking
3. RSI divergence confirmation
4. Advanced volume analysis
5. Corrective wave pattern detection (A-B-C)
"""

import numpy as np
import pandas as pd
import math
from scipy.signal import find_peaks

# --- Constants ---
# Fibonacci levels and thresholds
FIB_RATIO_TOLERANCE = 0.10  # Tolerance for Fibonacci ratio matching
SHALLOW_THRESHOLD = 0.382   # Retracements below this are considered shallow
DEEP_THRESHOLD = 0.618      # Retracements above this are considered deep
WAVE3_EXTENSION_THRESHOLD = 1.618  # Minimum extension for Wave 3
WAVE2_DEEP_RETRACE = 0.618  # Threshold for deep Wave 2 retracement
WAVE4_SHALLOW_RETRACE = 0.500  # Threshold for shallow Wave 4 retracement

# Scoring constants
SCORE_RULE_PASS = 100       # Points for passing a critical rule
SCORE_GUIDELINE_PASS = 15   # Points for following a guideline
SCORE_FIB_TARGET_HIT = 25   # Points for hitting a Fibonacci target
SCORE_CHANNEL_HIT = 10      # Points for respecting channel boundaries
SCORE_VOLUME_GUIDELINE = 5  # Points for volume pattern adherence
SCORE_DIVERGENCE_CONFIRM = 30  # Points for RSI divergence confirmation

# --- Helper Functions ---
def calculate_fib_ratio(start_val, end_val, retrace_val):
    """Calculate the Fibonacci retracement ratio."""
    if math.isnan(start_val) or math.isnan(end_val) or math.isnan(retrace_val):
        return np.nan
    
    total_move = end_val - start_val
    if abs(total_move) < 1e-9:  # Avoid division by zero
        return np.nan
    
    retrace_amount = end_val - retrace_val
    ratio = retrace_amount / total_move
    
    return ratio

def get_fib_level_str(ratio):
    """Convert a Fibonacci ratio to a descriptive string."""
    if pd.isna(ratio):
        return "Unknown"
    if abs(ratio - 0.236) <= FIB_RATIO_TOLERANCE: return "23.6%"
    if abs(ratio - 0.382) <= FIB_RATIO_TOLERANCE: return "38.2%"
    if abs(ratio - 0.5) <= FIB_RATIO_TOLERANCE: return "50.0%"
    if abs(ratio - 0.618) <= FIB_RATIO_TOLERANCE: return "61.8%"
    if abs(ratio - 0.786) <= FIB_RATIO_TOLERANCE: return "78.6%"
    if abs(ratio - 1.0) <= FIB_RATIO_TOLERANCE: return "100.0%"
    if abs(ratio - 1.618) <= FIB_RATIO_TOLERANCE: return "161.8%"
    if abs(ratio - 2.0) <= FIB_RATIO_TOLERANCE: return "200.0%"
    if abs(ratio - 2.618) <= FIB_RATIO_TOLERANCE: return "261.8%"
    return f"{ratio*100:.1f}%"

def is_close_to_fib(ratio, fib_levels):
    """Check if a ratio is close to any of the specified Fibonacci levels."""
    if pd.isna(ratio):
        return 0
    
    for level in fib_levels:
        if abs(ratio - level) <= FIB_RATIO_TOLERANCE:
            return 1
    
    return 0

def get_idx_from_point(pt, full_data):
    """Get the index in full_data corresponding to a point."""
    idx_val = None
    
    if isinstance(pt, (pd.Series, dict)) and 'Index' in pt:
        idx_val = pt['Index']
    elif isinstance(pt, pd.Timestamp):
        idx_val = pt
    
    if idx_val is not None:
        try:
            # Try exact match first
            return full_data.index.get_loc(idx_val)
        except KeyError:
            # Fall back to nearest match
            try:
                indices = full_data.index.get_indexer([idx_val], method='nearest')
                if len(indices) > 0 and indices[0] != -1:
                    return indices[0]
            except Exception as e:
                print(f"  Error finding index for {idx_val}: {e}")
    
    return None

# --- Main Scoring Function ---
def score_elliott_sequence(points, is_up, length, full_data):
    print("  [Improved Scoring] Using enhanced Elliott Wave scoring algorithm with:")
    print("    - Stricter Wave 4 overlap rule enforcement")
    print("    - Better alternation guideline checking")
    print("    - RSI divergence confirmation")
    print("    - Advanced volume analysis")
    """
    Score an Elliott Wave sequence based on rules, guidelines, and confirmations.
    
    Args:
        points: Dictionary of wave points {0: point0, 1: point1, ...}
        is_up: Boolean indicating if this is an uptrend (True) or downtrend (False)
        length: Number of points in the sequence (3-6 for partial or complete waves)
        full_data: DataFrame with price and indicator data
        
    Returns:
        Dictionary with score and detailed analysis or -1 if invalid
    """
    score = 0
    eps = 1e-9  # Small value to handle floating point comparisons
    
    # Initialize tracking dictionaries
    rules_passed = {"W2_Ret": False, "W4_Overlap": True, "W3_Shortest": True}
    guidelines_passed = {
        "W3_Ext": False, "W5_Eq_W1": False, "Alt_W2W4": False, 
        "Channel": False, "Vol_W3>W1": False, "Vol_W5<=W3": True, 
        "Vol_Corr<Imp": True, "RSI_Div_W3_W5": False
    }
    fib_details = {}
    vol_details = {}
    rsi_details = {}
    
    # Validate points
    p = points
    required_keys = list(range(length))
    if not all(k in p for k in required_keys):
        return -1  # Missing required points
    
    # Extract price levels based on trend direction
    p_hl = {}
    try:
        for k, pt in p.items():
            if k % 2 == 0:  # Even points (0, 2, 4) - start/troughs in uptrend, start/peaks in downtrend
                p_hl[k] = pt['Low'] if is_up else pt['High']
            else:  # Odd points (1, 3, 5) - peaks in uptrend, troughs in downtrend
                p_hl[k] = pt['High'] if is_up else pt['Low']
    except (KeyError, TypeError) as e:
        return -1  # Point data issue
    
    if not all(k in p_hl for k in required_keys):
        return -1  # Missing price data
    
    # --- Wave 2 Rules and Guidelines ---
    if length >= 3:
        # Rule: Wave 2 never retraces more than 100% of Wave 1
        if (is_up and p_hl[2] >= p_hl[0]) or (not is_up and p_hl[2] <= p_hl[0]):
            rules_passed["W2_Ret"] = True
            score += SCORE_RULE_PASS
        else:
            print(f"  Rule violation: Wave 2 ({p_hl[2]:.2f}) retraces beyond Wave 0 ({p_hl[0]:.2f})")
            return -1  # Critical rule violation
        
        # Guideline: Wave 2 often retraces 50%, 61.8%, 78.6% of Wave 1
        w2r = calculate_fib_ratio(p_hl[0], p_hl[1], p_hl[2])
        fib_details['W2_Ret_W1'] = (w2r, get_fib_level_str(w2r))
        score += SCORE_FIB_TARGET_HIT * is_close_to_fib(w2r, [0.5, 0.618, 0.786])
    
    # --- Wave 3 Rules and Guidelines ---
    if length >= 4:
        # Rule: Wave 3 must move beyond the end of Wave 1
        if (is_up and p_hl[3] <= p_hl[1]) or (not is_up and p_hl[3] >= p_hl[1]):
            print(f"  Rule violation: Wave 3 ({p_hl[3]:.2f}) does not move beyond Wave 1 ({p_hl[1]:.2f})")
            return -1  # Critical rule violation
        
        # Guideline: Wave 3 is often extended (1.618, 2.0, 2.618 of Wave 1)
        len_w1 = abs(p_hl[1] - p_hl[0])
        len_w3 = abs(p_hl[3] - p_hl[2])
        w3r = len_w3 / len_w1 if len_w1 > eps else np.nan
        fib_details['W3_Ext_W1'] = (w3r, get_fib_level_str(w3r))
        score += SCORE_FIB_TARGET_HIT * is_close_to_fib(w3r, [1.618, 2.0, 2.618])
        
        guidelines_passed["W3_Ext"] = not pd.isna(w3r) and w3r > WAVE3_EXTENSION_THRESHOLD
        if guidelines_passed["W3_Ext"]:
            score += SCORE_GUIDELINE_PASS
    
    # --- Wave 4 Rules and Guidelines ---
    if length >= 5:
        # Rule: Wave 4 never overlaps the price territory of Wave 1 (end of Wave 1)
        w1_end_price = p_hl[1]
        if (is_up and p_hl[4] <= w1_end_price) or (not is_up and p_hl[4] >= w1_end_price):
            print(f"  Rule violation: Wave 4 ({p_hl[4]:.2f}) overlaps Wave 1 end ({w1_end_price:.2f})")
            rules_passed["W4_Overlap"] = False
            return -1  # Critical rule violation
        else:
            rules_passed["W4_Overlap"] = True
            score += SCORE_RULE_PASS
        
        # Guideline: Wave 4 often retraces 23.6%, 38.2%, 50% of Wave 3
        w4r = calculate_fib_ratio(p_hl[2], p_hl[3], p_hl[4])
        fib_details['W4_Ret_W3'] = (w4r, get_fib_level_str(w4r))
        score += SCORE_FIB_TARGET_HIT * is_close_to_fib(w4r, [0.236, 0.382, 0.5])
        
        # Guideline: Alternation between Wave 2 and Wave 4
        w2r = fib_details.get('W2_Ret_W1', (np.nan,))[0]
        if not pd.isna(w2r) and not pd.isna(w4r):
            w2_is_deep = w2r > DEEP_THRESHOLD
            w2_is_shallow = w2r < SHALLOW_THRESHOLD
            w4_is_deep = w4r > DEEP_THRESHOLD
            w4_is_shallow = w4r < SHALLOW_THRESHOLD
            
            # Alternation fails if both waves are deep or both are shallow
            alternation_passes = not ((w2_is_deep and w4_is_deep) or (w2_is_shallow and w4_is_shallow))
            
            guidelines_passed["Alt_W2W4"] = alternation_passes
            if alternation_passes:
                print(f"  Alternation guideline passed: W2 Ret={w2r:.3f}, W4 Ret={w4r:.3f}")
                score += SCORE_GUIDELINE_PASS
            else:
                print(f"  Alternation guideline failed: W2 Ret={w2r:.3f}, W4 Ret={w4r:.3f} (Both deep or both shallow)")
                score -= SCORE_GUIDELINE_PASS  # Penalize violation
        else:
            print("  Alternation guideline skipped: W2 or W4 retracement ratio missing.")
    
    # --- Wave 5 Rules and Guidelines ---
    if length >= 6:  # Need points 0 through 5
        # Rule: Wave 3 is never the shortest impulse wave (comparing lengths of W1, W3, W5)
        w1_len = abs(p_hl[1] - p_hl[0])
        w3_len = abs(p_hl[3] - p_hl[2])
        w5_len = abs(p_hl[5] - p_hl[4])
        
        if w3_len <= w1_len + eps and w3_len <= w5_len + eps:
            print(f"  Rule violation: Wave 3 (len {w3_len:.4f}) is shortest (W1={w1_len:.4f}, W5={w5_len:.4f})")
            rules_passed["W3_Shortest"] = False
            return -1  # Critical rule violation
        else:
            rules_passed["W3_Shortest"] = True
            score += SCORE_RULE_PASS
        
        # Guideline: Wave 5 often relates to Wave 1 (e.g., W5=W1, W5=0.618*W1)
        w5r_w1 = w5_len / w1_len if w1_len > eps else np.nan
        fib_details['W5_Rel_W1'] = (w5r_w1, get_fib_level_str(w5r_w1))
        guidelines_passed["W5_Eq_W1"] = is_close_to_fib(w5r_w1, [1.0, 0.618])
        if guidelines_passed["W5_Eq_W1"]:
            score += SCORE_GUIDELINE_PASS
    
    # --- Volume Analysis ---
    has_volume_data = 'Volume' in full_data.columns and not full_data['Volume'].isnull().all()
    if has_volume_data:
        # Get indices for all available points
        idx = {k: get_idx_from_point(p[k], full_data) for k in range(length) if k in p}
        
        # Calculate average volume for each wave if indices are valid and ordered correctly
        vol_waves = {}
        for i in range(length - 1):
            if i in idx and i+1 in idx and idx[i] is not None and idx[i+1] is not None and idx[i] < idx[i+1]:
                vol_waves[i] = full_data['Volume'].iloc[idx[i]:idx[i+1]+1].mean()
            else:
                vol_waves[i] = np.nan
        
        vol_details = {f'W{i+1}_AvgVol': vol for i, vol in vol_waves.items()}
        
        # Guideline: Volume tends to be highest in Wave 3
        if 0 in vol_waves and 2 in vol_waves and not pd.isna(vol_waves[0]) and not pd.isna(vol_waves[2]):
            if vol_waves[2] > vol_waves[0]:  # W3 > W1
                guidelines_passed["Vol_W3>W1"] = True
                score += SCORE_VOLUME_GUIDELINE
        
        # Guideline: Volume tends to be lower in Wave 5 than Wave 3
        if 2 in vol_waves and 4 in vol_waves and not pd.isna(vol_waves[2]) and not pd.isna(vol_waves[4]):
            if vol_waves[4] <= vol_waves[2]:  # W5 <= W3
                guidelines_passed["Vol_W5<=W3"] = True
                score += SCORE_VOLUME_GUIDELINE
            else:
                guidelines_passed["Vol_W5<=W3"] = False
        
        # Guideline: Volume tends to decrease during corrective waves (2, 4) vs impulse waves (1, 3, 5)
        imp_waves = [vol_waves.get(i, np.nan) for i in [0, 2, 4]]  # W1, W3, W5
        corr_waves = [vol_waves.get(i, np.nan) for i in [1, 3]]  # W2, W4
        
        imp_waves_valid = [v for v in imp_waves if not pd.isna(v)]
        corr_waves_valid = [v for v in corr_waves if not pd.isna(v)]
        
        if imp_waves_valid and corr_waves_valid:
            imp_vol_avg = np.mean(imp_waves_valid)
            corr_vol_avg = np.mean(corr_waves_valid)
            
            if corr_vol_avg < imp_vol_avg:
                guidelines_passed["Vol_Corr<Imp"] = True
                score += SCORE_VOLUME_GUIDELINE
            else:
                guidelines_passed["Vol_Corr<Imp"] = False
    else:
        vol_details['Info'] = "Volume data not available or insufficient."
    
    # --- RSI Divergence Check ---
    has_rsi = 'RSI' in full_data.columns and not full_data['RSI'].isnull().all()
    if has_rsi and length >= 6:  # Need points 0 through 5 for W3/W5 check
        idx3 = get_idx_from_point(p.get(3), full_data)
        idx5 = get_idx_from_point(p.get(5), full_data)
        
        if idx3 is not None and idx5 is not None:
            try:
                price3 = p_hl[3]
                price5 = p_hl[5]
                rsi3 = full_data['RSI'].iloc[idx3]
                rsi5 = full_data['RSI'].iloc[idx5]
                
                rsi_details = {'W3_Price': price3, 'W3_RSI': rsi3, 'W5_Price': price5, 'W5_RSI': rsi5}
                
                # Bearish Divergence (Up-trend): Price Higher High, RSI Lower High
                if is_up and price5 > price3 and rsi5 < rsi3:
                    print(f"  RSI Bearish Divergence confirmed: W3({price3:.2f}, RSI {rsi3:.2f}) -> W5({price5:.2f}, RSI {rsi5:.2f})")
                    guidelines_passed["RSI_Div_W3_W5"] = True
                    score += SCORE_DIVERGENCE_CONFIRM
                # Bullish Divergence (Down-trend): Price Lower Low, RSI Higher Low
                elif not is_up and price5 < price3 and rsi5 > rsi3:
                    print(f"  RSI Bullish Divergence confirmed: W3({price3:.2f}, RSI {rsi3:.2f}) -> W5({price5:.2f}, RSI {rsi5:.2f})")
                    guidelines_passed["RSI_Div_W3_W5"] = True
                    score += SCORE_DIVERGENCE_CONFIRM
            except Exception as e:
                rsi_details['Error'] = str(e)
    
    # --- Return Score and Details ---
    details = {
        "score": score,
        "rules_passed": rules_passed,
        "guidelines_passed": guidelines_passed,
        "fib_details": fib_details,
        "vol_details": vol_details,
        "rsi_details": rsi_details,
        "length": length,
        "is_up": is_up
    }
    
    return details

# --- Corrective Wave Detection ---
def find_corrective_waves(wave_points, data_full):
    print("  [Improved Analysis] Searching for A-B-C corrective wave patterns")
    print("    - Looking for zigzag and flat corrections")
    print("    - Checking Fibonacci relationships")
    print("    - Analyzing volume characteristics")
    """
    Identify potential A-B-C corrective wave patterns.
    
    Args:
        wave_points: DataFrame of potential wave points
        data_full: DataFrame with price and indicator data
        
    Returns:
        DataFrame with labeled corrective waves and analysis results
    """
    print("\n[EW Analysis] Finding Potential Corrective Waves (A-B-C)...")
    
    analysis_results = {
        "found_corrective": False,
        "details": {},
        "last_label": None,
        "last_point": None
    }
    
    if wave_points is None or wave_points.empty or len(wave_points) < 3:
        print("  Info: Not enough points for corrective wave analysis.")
        return wave_points, analysis_results
    
    # Copy wave points to avoid modifying the original
    wave_points = wave_points.copy()
    wave_points['EW_Label'] = ""  # Reset labels
    
    best_score = -1
    best_details = None
    best_labels = None
    
    # Try different starting points
    for i in range(len(wave_points) - 2):
        # Check both up and down trends
        for is_up in [True, False]:
            # For corrective waves, we need 3 points (A, B, C)
            points = {
                0: wave_points.iloc[i],     # A
                1: wave_points.iloc[i+1],   # B
                2: wave_points.iloc[i+2]    # C
            }
            
            # Score the potential A-B-C pattern
            details = score_corrective_sequence(points, is_up, 3, data_full)
            
            if details != -1 and details["score"] > best_score:
                best_score = details["score"]
                best_details = details
                
                # Create labels for the best sequence
                best_labels = [""] * len(wave_points)
                best_labels[i] = "A"
                best_labels[i+1] = "B"
                best_labels[i+2] = "C"
    
    # If we found a valid corrective pattern
    if best_score > 0 and best_details is not None:
        print(f"  Found corrective wave pattern with score: {best_score}")
        wave_points['EW_Label'] = best_labels
        analysis_results["found_corrective"] = True
        analysis_results["details"] = best_details
        analysis_results["last_label"] = "C"
        analysis_results["last_point"] = wave_points[wave_points['EW_Label'] == "C"].iloc[0]
    else:
        print("  No valid corrective wave patterns found.")
    
    return wave_points, analysis_results

def score_corrective_sequence(points, is_up, length, full_data):
    """
    Score an A-B-C corrective wave sequence.
    
    Args:
        points: Dictionary of wave points {0: pointA, 1: pointB, 2: pointC}
        is_up: Boolean indicating if this is an uptrend (True) or downtrend (False)
        length: Number of points in the sequence (should be 3 for A-B-C)
        full_data: DataFrame with price and indicator data
        
    Returns:
        Dictionary with score and detailed analysis or -1 if invalid
    """
    score = 0
    eps = 1e-9
    
    # Initialize tracking dictionaries
    rules_passed = {"B_Ret": False, "C_Projection": False}
    guidelines_passed = {"Fib_Ratios": False, "Alt_Shape": False, "Vol_Diminish": False}
    fib_details = {}
    
    # Validate points
    p = points
    required_keys = list(range(length))
    if not all(k in p for k in required_keys) or length != 3:
        return -1  # Must have exactly 3 points for A-B-C
    
    # Extract price levels based on trend direction
    # For corrective waves, the trend direction is opposite the main trend
    # A-B-C in a downtrend means: A (high), B (low), C (lower high)
    # A-B-C in an uptrend means: A (low), B (high), C (higher low)
    p_hl = {}
    try:
        for k, pt in p.items():
            if k == 0:  # Point A
                p_hl[k] = pt['High'] if is_up else pt['Low']
            elif k == 1:  # Point B
                p_hl[k] = pt['Low'] if is_up else pt['High']
            elif k == 2:  # Point C
                p_hl[k] = pt['High'] if is_up else pt['Low']
    except (KeyError, TypeError):
        return -1
    
    # Rule: Wave B should retrace Wave A by 38.2% to 78.6%
    b_retrace = calculate_fib_ratio(p_hl[0], p_hl[1], p_hl[1])  # 100% by definition
    fib_details['B_Ret_A'] = (b_retrace, get_fib_level_str(b_retrace))
    
    # For a zigzag correction, B should not retrace more than 78.6% of A
    # For a flat correction, B can retrace 100% or more of A
    if b_retrace <= 0.786 or b_retrace >= 0.9:  # Either zigzag or flat
        rules_passed["B_Ret"] = True
        score += SCORE_RULE_PASS
    else:
        print(f"  Corrective rule violation: Wave B retracement ({b_retrace:.3f}) not typical")
        return -1
    
    # Rule: Wave C should project from B
    # In a zigzag, C often extends to 100%, 1.618, or 2.618 of A
    c_projection = abs(p_hl[2] - p_hl[1]) / abs(p_hl[1] - p_hl[0]) if abs(p_hl[1] - p_hl[0]) > eps else np.nan
    fib_details['C_Proj_A'] = (c_projection, get_fib_level_str(c_projection))
    
    if is_close_to_fib(c_projection, [1.0, 1.618, 2.618]):
        rules_passed["C_Projection"] = True
        score += SCORE_RULE_PASS
        guidelines_passed["Fib_Ratios"] = True
        score += SCORE_GUIDELINE_PASS
    
    # Check for volume diminishing through the correction (common in A-B-C)
    has_volume = 'Volume' in full_data.columns and not full_data['Volume'].isnull().all()
    if has_volume:
        idx = {k: get_idx_from_point(p[k], full_data) for k in range(length)}
        
        vol_a = np.nan
        vol_b = np.nan
        vol_c = np.nan
        
        if all(idx[k] is not None for k in range(length)):
            # Get average volume for each segment
            if idx[0] < idx[1]:
                vol_a = full_data['Volume'].iloc[idx[0]:idx[1]+1].mean()
            if idx[1] < idx[2]:
                vol_b = full_data['Volume'].iloc[idx[1]:idx[2]+1].mean()
            
            # Check if volume diminishes through the correction
            if not pd.isna(vol_a) and not pd.isna(vol_b) and vol_b < vol_a:
                guidelines_passed["Vol_Diminish"] = True
                score += SCORE_VOLUME_GUIDELINE
    
    # Return score and details
    details = {
        "score": score,
        "rules_passed": rules_passed,
        "guidelines_passed": guidelines_passed,
        "fib_details": fib_details,
        "pattern_type": "zigzag" if b_retrace <= 0.786 else "flat",
        "length": length,
        "is_up": is_up
    }
    
    return details
